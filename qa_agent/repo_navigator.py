"""
Repository navigation module.

This module provides functionality to navigate through a repository
and find relevant code files, including integration with Sourcegraph
for enhanced code context gathering.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, CodeSearchResult, FileType
from qa_agent.sourcegraph_client import SourcegraphClient

logger = logging.getLogger(__name__)


class RepoNavigator:
    """Navigates through a repository to find relevant code files."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the repository navigator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.repo_path = config.repo_path
        self.ignore_patterns = config.ignore_patterns or []

        # Initialize Sourcegraph client if enabled
        self.sourcegraph_client = None
        if config.sourcegraph_enabled:
            logger.info("Sourcegraph integration enabled")
            try:
                self.sourcegraph_client = SourcegraphClient(config)
                logger.info(
                    f"Sourcegraph client initialized with endpoint: {config.sourcegraph_api_endpoint}"
                )
                if config.sourcegraph_api_token:
                    logger.info("Sourcegraph API token provided")
                else:
                    logger.warning(
                        "No Sourcegraph API token provided - some features may be limited"
                    )
            except Exception as e:
                logger.error(f"Error initializing Sourcegraph client: {str(e)}")
                self.sourcegraph_client = None

    def find_all_code_files(self) -> List[CodeFile]:
        """
        Find all code files in the repository.

        Returns:
            List of CodeFile objects
        """
        logger.info(f"Finding all code files in {self.repo_path}")

        code_files = []

        for root, dirs, files in os.walk(self.repo_path):
            # Skip directories that match ignore patterns
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d))]

            for file in files:
                file_path = os.path.join(root, file)

                # Skip files that match ignore patterns
                if self._should_ignore(file_path):
                    continue

                # Check if the file is a code file
                if self._is_code_file(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        code_file = CodeFile(path=file_path, content=content)
                        code_files.append(code_file)
                    except UnicodeDecodeError:
                        logger.warning(f"Could not read file as text: {file_path}")
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")

        logger.info(f"Found {len(code_files)} code files")

        return code_files

    def find_related_files(self, file_path: str) -> List[CodeFile]:
        """
        Find files related to a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of related CodeFile objects
        """
        logger.info(f"Finding files related to {file_path}")

        all_files = self.find_all_code_files()
        file_name = os.path.basename(file_path)
        base_name, _ = os.path.splitext(file_name)

        # Extract imports from the file
        imports = self._extract_imports(file_path)

        related_files = []
        file_imports_map = {}

        # Create a map of imports for each file
        for code_file in all_files:
            if code_file.path != file_path:  # Skip the target file
                file_imports = self._extract_imports(code_file.path)
                file_imports_map[code_file.path] = file_imports

        for code_file in all_files:
            # Skip the file itself
            if code_file.path == file_path:
                continue

            # Look for files with similar names
            similar_name = os.path.basename(code_file.path)
            if similar_name.startswith(base_name) or base_name in similar_name:
                related_files.append(code_file)
                continue

            # Look for files that might be imported by our target file
            for imp in imports:
                if imp in code_file.path or imp in os.path.basename(code_file.path):
                    related_files.append(code_file)
                    break

            # Look for files that import our target file
            file_imports = file_imports_map.get(code_file.path, set())
            base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            if base_name_no_ext in file_imports:
                related_files.append(code_file)

        logger.info(f"Found {len(related_files)} related files")

        return related_files

    def _should_ignore(self, path: str) -> bool:
        """
        Check if a path should be ignored based on ignore patterns.

        Args:
            path: Path to check

        Returns:
            True if the path should be ignored, False otherwise
        """
        # Convert to relative path from repo root for consistent matching
        rel_path = os.path.relpath(path, self.repo_path)
        # Also ensure we're handling the path basename for filename matching
        basename = os.path.basename(path)

        for pattern in self.ignore_patterns:
            # Handle exact matches for directory and file names
            if pattern == basename or f"/{pattern}" in rel_path:
                return True

            # Handle extension wildcard patterns like "*.pyc"
            if pattern.startswith("*."):
                # Extract the extension part after '*.'
                ext_pattern = pattern[1:]  # This gives '.pyc' from '*.pyc'
                if path.endswith(ext_pattern):
                    return True

            # Handle other glob pattern matching
            elif "*" in pattern or "?" in pattern or "[" in pattern:
                try:
                    # Escape special regex chars except glob wildcards
                    regex_pattern = pattern
                    for char in ".^$+{}\\|()":
                        regex_pattern = regex_pattern.replace(char, "\\" + char)

                    # Convert glob wildcards to regex
                    regex_pattern = regex_pattern.replace("*", ".*")
                    regex_pattern = regex_pattern.replace("?", ".")

                    # Match both the full path and just the filename
                    if re.search(regex_pattern + "$", basename) or re.search(
                        regex_pattern, rel_path
                    ):
                        return True
                except re.error:
                    # If there's an error in the regex, fall back to simple matching
                    logger.warning(f"Invalid pattern '{pattern}', using simple string matching")
                    if pattern in rel_path or pattern in basename:
                        return True

        return False

    def _is_code_file(self, file_path: str) -> bool:
        """
        Check if a file is a code file based on extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a code file, False otherwise
        """
        _, ext = os.path.splitext(file_path)

        # Add more extensions as needed
        code_extensions = [
            ".py",  # Python
            ".js",  # JavaScript
            ".ts",  # TypeScript
            ".jsx",  # React JavaScript
            ".tsx",  # React TypeScript
            ".java",  # Java
            ".rb",  # Ruby
            ".php",  # PHP
            ".go",  # Go
            ".c",  # C
            ".cpp",  # C++
            ".h",  # C/C++ header
            ".cs",  # C#
            ".swift",  # Swift
            ".kt",  # Kotlin
            ".rs",  # Rust
            ".sql",  # SQL scripts
        ]

        return ext.lower() in code_extensions

    def _extract_imports(self, file_path: str) -> Set[str]:
        """
        Extract imports from a file.

        Args:
            file_path: Path to the file

        Returns:
            Set of import names
        """
        imports = set()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            _, ext = os.path.splitext(file_path)

            if ext.lower() == ".py":
                # Extract Python imports (both 'import' and 'from' statements)
                # Handle 'import x' and 'import x.y.z' formats
                import_matches = re.findall(
                    r"^import\s+([\w\.]+)(?:,\s*([\w\.]+))*", content, re.MULTILINE
                )
                for match_tuple in import_matches:
                    for imp in match_tuple:
                        if imp:  # Skip empty matches
                            imports.add(imp.split(".")[0])  # Get the base module

                # Handle 'from x import y' and 'from x.y import z' formats
                from_matches = re.findall(r"^from\s+([\w\.]+)\s+import", content, re.MULTILINE)
                for imp in from_matches:
                    if imp:
                        imports.add(imp.split(".")[0])  # Get the base module

            # Add support for other languages as needed

        except Exception as e:
            logger.error(f"Error extracting imports from {file_path}: {str(e)}")

        return imports

    def find_function_examples(self, function_name: str, limit: int = 5) -> List[CodeFile]:
        """
        Find examples of how a function is used in the codebase or via Sourcegraph.

        Args:
            function_name: Name of the function
            limit: Maximum number of examples to return

        Returns:
            List of CodeFile objects containing examples
        """
        logger.info(f"Finding examples for function: {function_name}")

        # First look for examples in the local repository
        all_files = self.find_all_code_files()
        local_examples = []

        # Simple pattern to find function calls in code
        pattern = rf"\b{re.escape(function_name)}\s*\("

        for code_file in all_files:
            if re.search(pattern, code_file.content):
                local_examples.append(code_file)
                if len(local_examples) >= limit:
                    break

        # If not enough local examples and Sourcegraph is enabled, search there
        if len(local_examples) < limit and self.sourcegraph_client:
            try:
                sg_results = self.sourcegraph_client.find_examples(
                    function_name, limit - len(local_examples)
                )

                for result in sg_results:
                    code_file = CodeFile(
                        path=f"sourcegraph://{result.repository}/{result.file_path}",
                        content=result.content,
                    )
                    local_examples.append(code_file)
            except Exception as e:
                logger.error(f"Error finding examples via Sourcegraph: {str(e)}")

        logger.info(f"Found {len(local_examples)} examples for function: {function_name}")
        return local_examples

    def find_semantic_similar_code(self, code_snippet: str, limit: int = 3) -> List[CodeFile]:
        """
        Find semantically similar code to the given snippet.

        Args:
            code_snippet: The code snippet to find similar code for
            limit: Maximum number of examples to return

        Returns:
            List of CodeFile objects with semantically similar code
        """
        logger.info("Searching for semantically similar code")

        similar_files = []

        # Use Sourcegraph's semantic search if available
        if self.sourcegraph_client:
            try:
                # The client method may be semantic_search or find_related_code depending on implementation
                if hasattr(self.sourcegraph_client, "find_related_code"):
                    results = self.sourcegraph_client.find_related_code(code_snippet, limit)
                elif hasattr(self.sourcegraph_client, "semantic_search"):
                    results = self.sourcegraph_client.semantic_search(code_snippet, limit)
                else:
                    logger.warning("Sourcegraph client has no semantic search capability")
                    results = []

                for result in results:
                    code_file = CodeFile(
                        path=f"sourcegraph://{result.repository}/{result.file_path}",
                        content=result.content,
                    )
                    similar_files.append(code_file)
            except Exception as e:
                logger.error(f"Error finding semantically similar code via Sourcegraph: {str(e)}")

        # If no results or Sourcegraph is not available, fall back to primitive similarity
        if not similar_files:
            all_files = self.find_all_code_files()

            # Very simple similarity: Check if important tokens from the snippet appear in the code
            # This could be improved with a more sophisticated algorithm
            important_tokens = self._extract_important_tokens(code_snippet)

            scored_files = []
            for code_file in all_files:
                score = 0
                for token in important_tokens:
                    if token in code_file.content:
                        score += 1

                if score > 0:
                    scored_files.append((code_file, score))

            # Sort by score (descending) and take the top 'limit' files
            scored_files.sort(key=lambda x: x[1], reverse=True)
            similar_files = [file for file, _ in scored_files[:limit]]

        logger.info(f"Found {len(similar_files)} semantically similar code files")
        return similar_files

    def get_code_intelligence(self, file_path: str, line: int) -> Optional[Dict[str, Any]]:
        """
        Get code intelligence information for a specific location.

        Args:
            file_path: Path to the file
            line: Line number (0-based)

        Returns:
            Dictionary with code intelligence information or None if not available
        """
        if not self.sourcegraph_client:
            return None

        try:
            result = self.sourcegraph_client.get_code_intelligence(file_path, line)

            if result:
                return {
                    "definitions": result.definitions,
                    "references": result.references,
                    "hover_info": result.hover_info,
                    "type_info": result.type_info,
                }
        except Exception as e:
            logger.error(f"Error getting code intelligence: {str(e)}")

        return None

    def should_ignore(self, path: str) -> bool:
        """
        Public method to check if a path should be ignored.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path should be ignored, False otherwise
        """
        return self._should_ignore(path)
        
    def _extract_important_tokens(self, code_snippet: str) -> List[str]:
        """
        Extract important tokens from a code snippet for similarity matching.

        Args:
            code_snippet: The code snippet

        Returns:
            List of important tokens
        """
        # Preserve some dot notation expressions that might be important (e.g., math.pi)
        dot_expressions = re.findall(r"\b[\w]+\.\w+\b", code_snippet)

        # Now remove common symbols and split into tokens
        cleaned = re.sub(r"[^\w\s]", " ", code_snippet)
        tokens = cleaned.split()

        # Filter out common keywords and short tokens
        stopwords = {
            "def",
            "if",
            "else",
            "for",
            "while",
            "return",
            "import",
            "from",
            "class",
            "and",
            "or",
            "not",
            "in",
            "is",
        }
        important_tokens = [
            token for token in tokens if token.lower() not in stopwords and len(token) > 2
        ]

        # Add back the dot expressions as they might be important for context
        important_tokens.extend(dot_expressions)

        return important_tokens
