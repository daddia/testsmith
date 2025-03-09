"""
Git History Analysis module.

This module provides functionality to analyze git history to identify
recently modified code and determine which functions need priority testing.
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, cast

# Define placeholders for git module types
if TYPE_CHECKING:
    from git import Repo
    from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

# Try to import git, but don't fail if it's not available
try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

    # Create placeholder for Git classes and error types to avoid import errors
    class GitPlaceholder:
        """Placeholder for git module when GitPython is not installed."""

        class Repo:
            """Placeholder for git.Repo when GitPython is not installed."""

            def __init__(self, path: Optional[str] = None) -> None:
                raise ImportError("GitPython is not installed")

            @staticmethod
            def iter_commits(**kwargs: Any) -> List[Any]:
                return []

        class InvalidGitRepositoryError(Exception):
            """Placeholder for git.InvalidGitRepositoryError."""

            pass

        class GitCommandError(Exception):
            """Placeholder for git.GitCommandError."""

            pass

        class NoSuchPathError(Exception):
            """Placeholder for git.NoSuchPathError."""

            pass

    # Define placeholder git module for import error cases
    git = GitPlaceholder  # type: ignore

from qa_agent.models import CodeFile, Function

logger = logging.getLogger(__name__)


class GitHistoryAnalyzer:
    """Analyzes git history to identify recently modified code."""

    def __init__(self, repo_path: str):
        """
        Initialize the git history analyzer.

        Args:
            repo_path: Path to the git repository
        """
        self.repo_path = repo_path
        self.repo: Any = None
        # Check if GitPython is available before proceeding
        if not GIT_AVAILABLE:
            logger.warning("GitPython is not installed. Git history analysis will be disabled.")
            logger.warning("Install it with: pip install gitpython")
            return
        self._initialize_repo()

    def _initialize_repo(self) -> None:
        """Initialize the git repository object."""
        if not GIT_AVAILABLE:
            return

        try:
            # Use the imported git module which is already available
            # (we only call this method if GIT_AVAILABLE is True)
            self.repo = git.Repo(self.repo_path)
            logger.info(f"Git repository initialized at {self.repo_path}")
        except git.InvalidGitRepositoryError:
            logger.warning(f"Not a valid Git repository: {self.repo_path}")
        except git.NoSuchPathError:
            logger.warning(f"Repository path does not exist: {self.repo_path}")
        except Exception as e:
            logger.exception(f"Error initializing Git repository: {str(e)}")

    def get_modified_files(self, days: int = 7) -> List[str]:
        """
        Get files that were modified in the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of modified file paths
        """
        if not self.repo:
            return []

        modified_files = []

        try:
            # Calculate the date threshold
            since_date = datetime.now() - timedelta(days=days)

            # Get commits since the threshold date
            commits = list(self.repo.iter_commits(since=since_date.strftime("%Y-%m-%d")))
            logger.info(f"Found {len(commits)} commits in the last {days} days")

            # Extract modified files from each commit
            for commit in commits:
                for file_path in commit.stats.files:
                    # Only include files that still exist
                    full_path = os.path.join(self.repo_path, file_path)
                    if os.path.exists(full_path):
                        modified_files.append(file_path)

            # Remove duplicates
            modified_files = list(set(modified_files))

        except Exception as e:
            logger.exception(f"Error getting modified files: {str(e)}")

        return modified_files

    def get_file_last_modified_date(self, file_path: str) -> Optional[datetime]:
        """
        Get the last modification date for a file.

        Args:
            file_path: Path to the file

        Returns:
            Datetime object representing the last modification date or None if not found
        """
        if not self.repo:
            return None

        try:
            # Get relative path if absolute path is provided
            rel_path = os.path.relpath(file_path, self.repo_path)

            # Get the last commit that modified this file
            commits = list(self.repo.iter_commits(paths=rel_path, max_count=1))

            if commits:
                return datetime.fromtimestamp(commits[0].committed_date)

        except Exception as e:
            logger.exception(f"Error getting last modified date for {file_path}: {str(e)}")

        return None

    def analyze_function_history(self, function: Function) -> None:
        """
        Analyze the git history for a specific function and update its attributes.

        Args:
            function: The function to analyze
        """
        if not self.repo:
            return

        try:
            # Get the file path relative to the repo
            file_path = os.path.relpath(function.file_path, self.repo_path)

            # Get last modification date for the function's file
            last_modified = self.get_file_last_modified_date(function.file_path)

            if last_modified:
                function.last_modified = last_modified.isoformat()

        except Exception as e:
            logger.exception(f"Error analyzing function history for {function.name}: {str(e)}")

    def filter_recently_modified_functions(
        self, functions: List[Function], days: int = 7
    ) -> List[Function]:
        """
        Filter functions to only include those in recently modified files.

        Args:
            functions: List of functions to filter
            days: Number of days to look back

        Returns:
            Filtered list of functions
        """
        modified_files = self.get_modified_files(days)

        if not modified_files:
            logger.warning(f"No modified files found in the last {days} days")
            return functions

        # Convert to set for faster lookups
        modified_files_set = set()
        for file_path in modified_files:
            # Handle both absolute and relative paths
            abs_path = (
                os.path.join(self.repo_path, file_path)
                if not os.path.isabs(file_path)
                else file_path
            )
            rel_path = (
                os.path.relpath(file_path, self.repo_path)
                if os.path.isabs(file_path)
                else file_path
            )
            modified_files_set.add(os.path.normpath(abs_path))
            modified_files_set.add(os.path.normpath(rel_path))

        # Filter functions from modified files
        recent_functions = []
        for function in functions:
            # Handle both absolute and relative function paths
            abs_path = (
                os.path.join(self.repo_path, function.file_path)
                if not os.path.isabs(function.file_path)
                else function.file_path
            )
            rel_path = (
                os.path.relpath(function.file_path, self.repo_path)
                if os.path.isabs(function.file_path)
                else function.file_path
            )

            normalized_abs = os.path.normpath(abs_path)
            normalized_rel = os.path.normpath(rel_path)

            if normalized_abs in modified_files_set or normalized_rel in modified_files_set:
                # Analyze and update function history
                self.analyze_function_history(function)
                recent_functions.append(function)

        logger.info(
            f"Found {len(recent_functions)} functions in {len(modified_files)} modified files"
        )
        return recent_functions
