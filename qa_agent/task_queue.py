"""
Task Queue module for parallel test generation and validation.

This module provides functionality to process test generation and validation
tasks in parallel using Python's concurrent.futures.
"""

import concurrent.futures
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from queue import Empty, Queue

# For type checking - git module types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from git import Repo
    from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

# Import git conditionally to handle when GitPython is not installed
try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

    # Create placeholder for Git classes and error types to avoid LSP errors
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

    git = GitPlaceholder  # type: ignore

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, Function, GeneratedTest, TestResult
from qa_agent.utils.logging import get_logger, log_exception, log_function_call, log_function_result

# Initialize logger for this module
logger = get_logger(__name__)


class TestTask:
    """Represents a single test generation and validation task."""

    def __init__(
        self, function: Function, context_files: Optional[List[CodeFile]] = None, priority: int = 0
    ):
        """
        Initialize a test task.

        Args:
            function: The function to test
            context_files: Related context files for the function
            priority: Task priority (lower is higher priority)
        """
        self.function = function
        self.context_files = context_files or []
        self.priority = priority
        self.generated_test: Optional[GeneratedTest] = None
        self.test_result: Optional[TestResult] = None
        self.status = "pending"  # pending, running, completed, failed
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.attempts = 0

    def __lt__(self, other: Any) -> bool:
        """For priority queue comparison - lower priority value means higher priority."""
        if not isinstance(other, TestTask):
            return NotImplemented  # type: ignore
        return self.priority < other.priority


class TaskResult:
    """Represents the result of a task processing."""

    def __init__(
        self,
        task: TestTask,
        success: bool,
        generated_test: Optional[GeneratedTest] = None,
        test_result: Optional[TestResult] = None,
        error: Optional[str] = None,
    ):
        """
        Initialize a task result.

        Args:
            task: The original task
            success: Whether the processing was successful
            generated_test: The generated test (if any)
            test_result: The test validation result (if any)
            error: Error message (if any)
        """
        self.task = task
        self.success = success
        self.generated_test = generated_test
        self.test_result = test_result
        self.error = error


class TaskQueue:
    """A queue for test tasks with parallel processing capabilities."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the task queue.

        Args:
            config: Configuration object
        """
        self.config = config
        self.tasks: Queue[TestTask] = Queue()
        self.results: List[TaskResult] = []
        self.running = False
        self._processed_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._lock = threading.Lock()  # For thread-safe operations

        # Log the initialization
        log_function_call(
            logger,
            "__init__",
            ("TaskQueue",),
            {
                "parallel": config.parallel_execution,
                "max_workers": config.max_workers,
                "incremental": config.incremental_testing,
            },
        )

    def add_task(self, task: TestTask) -> None:
        """
        Add a task to the queue.

        Args:
            task: The task to add
        """
        self.tasks.put(task)
        logger.debug("Task added to queue", function=task.function.name, priority=task.priority)

    def _get_all_tasks_safely(self) -> List[TestTask]:
        """
        Safely get all tasks from the queue without blocking.

        Returns:
            List of tasks
        """
        all_tasks = []
        while True:
            try:
                # Non-blocking get to prevent deadlocks
                task = self.tasks.get(block=False)
                all_tasks.append(task)
            except Empty:
                # Queue is empty, exit loop
                break
        return all_tasks

    def add_tasks(self, tasks: List[TestTask]) -> None:
        """
        Add multiple tasks to the queue.

        Args:
            tasks: The tasks to add
        """
        for task in tasks:
            self.add_task(task)
        logger.info(f"Added {len(tasks)} tasks to queue")

    def _process_task(
        self,
        task: TestTask,
        generate_func: Callable[[Function, List[CodeFile]], GeneratedTest],
        validate_func: Callable[[GeneratedTest], TestResult],
    ) -> TaskResult:
        """
        Process a single task.

        Args:
            task: The task to process
            generate_func: Function to generate a test
            validate_func: Function to validate a test

        Returns:
            TaskResult object
        """
        # Use a local copy of the task to prevent race conditions
        # when multiple threads access the same task objects
        local_task = TestTask(
            function=task.function, context_files=task.context_files, priority=task.priority
        )
        local_task.started_at = datetime.now()
        local_task.status = "running"
        local_task.attempts = task.attempts + 1

        # Generate a unique, thread-safe task ID using thread ID and timestamp
        thread_id = threading.get_ident()
        timestamp = int(time.time() * 1000)
        task_id = f"{local_task.function.name}_{thread_id}_{timestamp}"

        try:
            logger.info(
                f"Generating test for function: {local_task.function.name}", task_id=task_id
            )

            # Generate test with proper error handling
            try:
                generated_test = generate_func(local_task.function, local_task.context_files)
                local_task.generated_test = generated_test
            except Exception as gen_error:
                logger.exception(
                    f"Error generating test for function: {local_task.function.name}",
                    task_id=task_id,
                )
                raise gen_error  # Re-raise to be caught by the outer try/except

            # Validate test with proper error handling
            logger.info(
                f"Validating test for function: {local_task.function.name}", task_id=task_id
            )
            try:
                test_result = validate_func(generated_test)
                local_task.test_result = test_result
            except Exception as val_error:
                logger.exception(
                    f"Error validating test for function: {local_task.function.name}",
                    task_id=task_id,
                )
                raise val_error  # Re-raise to be caught by the outer try/except

            local_task.completed_at = datetime.now()
            local_task.status = "completed"

            # Thread-safe updates to counters with timeout to prevent deadlocks
            lock_acquired = self._lock.acquire(timeout=5.0)  # 5 second timeout
            try:
                if lock_acquired:
                    self._processed_count += 1
                    if test_result.success:
                        self._success_count += 1
                    else:
                        self._failure_count += 1
                else:
                    logger.warning(f"Could not acquire lock for counter updates: {task_id}")
            finally:
                if lock_acquired:
                    self._lock.release()

            # Update the original task object in a thread-safe way with timeout
            lock_acquired = self._lock.acquire(timeout=5.0)
            try:
                if lock_acquired:
                    task.generated_test = local_task.generated_test
                    task.test_result = local_task.test_result
                    task.status = local_task.status
                    task.completed_at = local_task.completed_at
                    task.attempts = local_task.attempts
                else:
                    logger.warning(f"Could not acquire lock for task updates: {task_id}")
            finally:
                if lock_acquired:
                    self._lock.release()

            return TaskResult(
                task=task,
                success=test_result.success,
                generated_test=generated_test,
                test_result=test_result,
            )

        except Exception as e:
            logger.exception(
                f"Error processing task for function: {local_task.function.name}", task_id=task_id
            )
            local_task.completed_at = datetime.now()
            local_task.status = "failed"
            local_task.error = str(e)

            # Thread-safe updates to counters with timeout to prevent deadlocks
            lock_acquired = self._lock.acquire(timeout=5.0)  # 5 second timeout
            try:
                if lock_acquired:
                    self._processed_count += 1
                    self._failure_count += 1
                else:
                    logger.warning(
                        f"Could not acquire lock for counter updates in error handler: {task_id}"
                    )
            finally:
                if lock_acquired:
                    self._lock.release()

            # Update the original task object in a thread-safe way with timeout
            lock_acquired = self._lock.acquire(timeout=5.0)
            try:
                if lock_acquired:
                    task.status = local_task.status
                    task.completed_at = local_task.completed_at
                    task.error = local_task.error
                    task.attempts = local_task.attempts
                else:
                    logger.warning(
                        f"Could not acquire lock for task updates in error handler: {task_id}"
                    )
            finally:
                if lock_acquired:
                    self._lock.release()

            return TaskResult(task=task, success=False, error=str(e))

    def process_tasks(
        self,
        generate_func: Callable[[Function, List[CodeFile]], GeneratedTest],
        validate_func: Callable[[GeneratedTest], TestResult],
    ) -> List[TaskResult]:
        """
        Process all tasks in the queue in parallel.

        Args:
            generate_func: Function to generate a test
            validate_func: Function to validate a test

        Returns:
            List of TaskResult objects
        """
        if self.running:
            logger.warning("Task processing is already running")
            return self.results

        self.running = True
        self.results = []

        if self.config.parallel_execution:
            logger.info(f"Processing tasks in parallel with {self.config.max_workers} workers")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                futures = []

                # Collect all tasks first to avoid queue issues
                all_tasks = self._get_all_tasks_safely()

                # Submit all tasks to the executor at once
                for task in all_tasks:
                    future = executor.submit(self._process_task, task, generate_func, validate_func)
                    futures.append(future)

                # Process results as they complete with a timeout to prevent hanging
                for future in futures:
                    try:
                        # Add a timeout to prevent tests from hanging indefinitely
                        # 60 seconds should be more than enough for any test function
                        result = future.result(timeout=60)
                        self.results.append(result)

                        # Log result
                        if result.success:
                            logger.info(f"Task completed successfully: {result.task.function.name}")
                        else:
                            logger.warning(
                                f"Task failed: {result.task.function.name} - {result.error}"
                            )

                    except concurrent.futures.TimeoutError:
                        logger.error("Timeout occurred while waiting for task to complete")
                        # Add a failure result to avoid blocking the workflow
                        task = None
                        for t in self.tasks.queue:
                            if not t.status == "completed":
                                task = t
                                break

                        if task:
                            self.results.append(
                                TaskResult(
                                    task=task,
                                    success=False,
                                    error="Task timed out after 60 seconds",
                                )
                            )
                    except Exception as e:
                        logger.exception("Error processing future")
        else:
            logger.info("Processing tasks sequentially")

            # Process tasks sequentially - first collect all tasks
            all_tasks = self._get_all_tasks_safely()

            # Process each task with a timeout mechanism
            for task in all_tasks:
                try:
                    # Use threading with timeout even for sequential processing
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            self._process_task, task, generate_func, validate_func
                        )
                        result = future.result(timeout=60)  # 60 second timeout per task
                        self.results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout occurred while processing task: {task.function.name}")
                    self.results.append(
                        TaskResult(
                            task=task, success=False, error="Task timed out after 60 seconds"
                        )
                    )
                except Exception as e:
                    logger.exception(f"Error processing task: {task.function.name}")
                    self.results.append(
                        TaskResult(
                            task=task, success=False, error=f"Task processing error: {str(e)}"
                        )
                    )

        self.running = False
        logger.info(
            f"Task processing complete: "
            f"{len(self.results)} tasks processed, "
            f"{self._success_count} succeeded, "
            f"{self._failure_count} failed"
        )

        return self.results

    def get_changed_files(self, days: int) -> List[str]:
        """
        Get list of files that have changed in the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of changed file paths
        """
        changed_files: List[str] = []

        # Check if GitPython is available
        if not GIT_AVAILABLE:
            logger.warning("GitPython is not installed. Incremental testing requires GitPython.")
            logger.warning("Install it with: pip install gitpython")

            # Fallback: Use filesystem modification times as a rough approximation
            logger.info("Falling back to filesystem modification times for changed files")
            try:
                return self._get_recently_modified_files_by_mtime(days)
            except Exception as e:
                logger.exception(f"Error getting changed files by mtime: {str(e)}")
                return changed_files

        try:
            # Calculate the date N days ago
            since_date = datetime.now() - timedelta(days=days)

            # Initialize Git repository
            repo = git.Repo(self.config.repo_path)

            # Get commits since the given date
            commits = list(repo.iter_commits(since=since_date.strftime("%Y-%m-%d")))

            if not commits:
                logger.warning(f"No commits found in the last {days} days")
                logger.info("Falling back to filesystem modification times")
                return self._get_recently_modified_files_by_mtime(days)

            # Get all files changed in these commits
            for commit in commits:
                for file in commit.stats.files:
                    if os.path.exists(os.path.join(self.config.repo_path, file)):
                        changed_files.append(str(file))

            # Remove duplicates
            changed_files = list(set(changed_files))
            logger.info(f"Found {len(changed_files)} files changed in the last {days} days")

        except git.InvalidGitRepositoryError:
            logger.warning(f"Not a valid Git repository: {self.config.repo_path}")
            logger.info("Falling back to filesystem modification times")
            return self._get_recently_modified_files_by_mtime(days)
        except git.NoSuchPathError:
            logger.warning(f"Repository path does not exist: {self.config.repo_path}")
        except git.GitCommandError as e:
            logger.warning(f"Git command error: {str(e)}")
            logger.info("Falling back to filesystem modification times")
            return self._get_recently_modified_files_by_mtime(days)
        except Exception as e:
            logger.exception(f"Error getting changed files: {str(e)}")
            logger.info("Falling back to filesystem modification times")
            return self._get_recently_modified_files_by_mtime(days)

        return changed_files

    def _get_recently_modified_files_by_mtime(self, days: int) -> List[str]:
        """
        Fallback method to get recently modified files using filesystem timestamps.

        Args:
            days: Number of days to look back

        Returns:
            List of recently modified file paths
        """
        changed_files: List[str] = []
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_timestamp = cutoff_time.timestamp()

        for root, dirs, files in os.walk(self.config.repo_path):
            # Skip hidden directories and files
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                # Skip hidden files
                if file.startswith("."):
                    continue

                file_path = os.path.join(root, file)
                # Skip directories that are typically not code (like test outputs, cache)
                if any(
                    ignore in file_path
                    for ignore in [
                        "__pycache__",
                        ".git",
                        ".pytest_cache",
                        ".coverage",
                        "node_modules",
                        "venv",
                        ".venv",
                        ".tox",
                    ]
                ):
                    continue

                try:
                    mtime = os.path.getmtime(file_path)
                    if mtime >= cutoff_timestamp:
                        rel_path = os.path.relpath(file_path, self.config.repo_path)
                        changed_files.append(rel_path)
                except Exception as e:
                    logger.debug(f"Error checking mtime for {file_path}: {str(e)}")

        logger.info(f"Found {len(changed_files)} files modified in the last {days} days (by mtime)")
        return changed_files

    def filter_functions_by_changed_files(
        self, functions: List[Function], days: int
    ) -> List[Function]:
        """
        Filter functions to only include those in files that have changed recently.

        Args:
            functions: List of functions to filter
            days: Number of days to look back for changes

        Returns:
            Filtered list of functions
        """
        if not self.config.incremental_testing:
            return functions

        changed_files = self.get_changed_files(days)

        if not changed_files:
            logger.warning("No changed files found, using all functions")
            return functions

        # Filter functions to only include those in changed files
        filtered_functions = [
            f
            for f in functions
            if any(f.file_path.endswith(changed_file) for changed_file in changed_files)
        ]

        logger.info(
            f"Filtered functions based on changed files: "
            f"{len(filtered_functions)} of {len(functions)} functions"
        )

        return filtered_functions


def create_task_queue(
    config: QAAgentConfig, functions: List[Function], context_files_map: Dict[str, List[CodeFile]]
) -> TaskQueue:
    """
    Create and populate a task queue based on functions and config.

    Args:
        config: Configuration object
        functions: List of functions to create tasks for
        context_files_map: Map of function paths to context files

    Returns:
        Populated TaskQueue object
    """
    queue = TaskQueue(config)

    # Apply incremental testing filter if enabled
    if config.incremental_testing:
        functions = queue.filter_functions_by_changed_files(functions, config.changed_since_days)

    # Create tasks for each function
    tasks = []
    for idx, function in enumerate(functions):
        context_files = context_files_map.get(function.file_path, [])
        # Lower index = higher priority (lower priority value)
        task = TestTask(function, context_files, priority=idx)
        tasks.append(task)

    # Add tasks to the queue
    queue.add_tasks(tasks)

    return queue
