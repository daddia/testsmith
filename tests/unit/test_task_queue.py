"""
Unit tests for the task queue module.

These tests verify the functionality of the task queue system for parallel
test generation and validation.
"""

import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, cast
from unittest.mock import MagicMock, call, patch

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, Function, GeneratedTest, TestResult
from qa_agent.task_queue import TaskQueue, TaskResult, TestTask, create_task_queue


@pytest.fixture
def mock_config() -> QAAgentConfig:
    """Create a mock configuration for testing."""
    config = QAAgentConfig()
    config.repo_path = "./test_repo"
    config.parallel_execution = True
    config.max_workers = 2
    config.incremental_testing = False
    return config


@pytest.fixture
def mock_function() -> Function:
    """Create a mock function for testing."""
    return Function(
        name="test_function",
        code="def test_function():\n    return True",
        file_path="test_module.py",
        start_line=1,
        end_line=2,
    )


@pytest.fixture
def mock_context_files() -> List[CodeFile]:
    """Create mock context files for testing."""
    return [CodeFile(path="test_module.py", content="def test_function():\n    return True")]


@pytest.fixture
def mock_test_task(mock_function: Function, mock_context_files: List[CodeFile]) -> TestTask:
    """Create a mock test task for testing."""
    return TestTask(function=mock_function, context_files=mock_context_files)


@pytest.fixture
def mock_generated_test(mock_function: Function) -> MagicMock:
    """Create a mock generated test for testing."""
    return MagicMock(spec=GeneratedTest)


@pytest.fixture
def mock_test_result() -> MagicMock:
    """Create a mock test result for testing."""
    return MagicMock(spec=TestResult)


class TestTestTask:
    """Tests for the TestTask class."""

    def test_init(self, mock_function: Function, mock_context_files: List[CodeFile]) -> None:
        """Test initialization of TestTask."""
        task = TestTask(function=mock_function, context_files=mock_context_files)

        assert task.function == mock_function
        assert task.context_files == mock_context_files
        assert task.priority == 0
        assert task.generated_test is None
        assert task.test_result is None
        assert task.status == "pending"
        assert task.created_at is not None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.error is None
        assert task.attempts == 0

    def test_lt_comparison(self, mock_function: Function) -> None:
        """Test priority comparison of tasks."""
        task1 = TestTask(function=mock_function, priority=1)
        task2 = TestTask(function=mock_function, priority=2)

        assert task1 < task2
        assert not (task2 < task1)


class TestTaskResult:
    """Tests for the TaskResult class."""

    def test_init(
        self, mock_test_task: TestTask, mock_generated_test: MagicMock, mock_test_result: MagicMock
    ) -> None:
        """Test initialization of TaskResult."""
        result = TaskResult(
            task=mock_test_task,
            success=True,
            generated_test=mock_generated_test,
            test_result=mock_test_result,
        )

        assert result.task == mock_test_task
        assert result.success is True
        assert result.generated_test == mock_generated_test
        assert result.test_result == mock_test_result
        assert result.error is None

    def test_init_with_error(self, mock_test_task: TestTask) -> None:
        """Test initialization of TaskResult with error."""
        result = TaskResult(task=mock_test_task, success=False, error="Test error")

        assert result.task == mock_test_task
        assert result.success is False
        assert result.generated_test is None
        assert result.test_result is None
        assert result.error == "Test error"


class TestTaskQueue:
    """Tests for the TaskQueue class."""

    def test_init(self, mock_config: QAAgentConfig) -> None:
        """Test initialization of TaskQueue."""
        queue = TaskQueue(mock_config)

        assert queue.config == mock_config
        assert queue.tasks.empty()
        assert queue.results == []
        assert queue.running is False
        assert queue._processed_count == 0
        assert queue._success_count == 0
        assert queue._failure_count == 0

    def test_add_task(self, mock_config: QAAgentConfig, mock_test_task: TestTask) -> None:
        """Test adding a task to the queue."""
        queue = TaskQueue(mock_config)
        queue.add_task(mock_test_task)

        assert not queue.tasks.empty()
        assert queue.tasks.qsize() == 1

    def test_add_tasks(self, mock_config: QAAgentConfig, mock_test_task: TestTask) -> None:
        """Test adding multiple tasks to the queue."""
        queue = TaskQueue(mock_config)
        tasks: List[TestTask] = [mock_test_task, mock_test_task]
        queue.add_tasks(tasks)

        assert not queue.tasks.empty()
        assert queue.tasks.qsize() == 2

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_process_tasks_parallel(
        self, mock_executor: MagicMock, mock_config: QAAgentConfig, mock_test_task: TestTask
    ) -> None:
        """Test processing tasks in parallel."""
        # Setup
        mock_config.parallel_execution = True
        queue = TaskQueue(mock_config)
        queue.add_task(mock_test_task)

        # Mock the executor and future
        mock_future = MagicMock()
        mock_future.result.return_value = MagicMock(spec=TaskResult, success=True)

        # Mock the executor context manager
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return the futures we submit
        with patch("concurrent.futures.as_completed", return_value=[mock_future]):
            # Execute
            generate_func: Callable[[Function, List[CodeFile]], GeneratedTest] = MagicMock()
            validate_func: Callable[[GeneratedTest], TestResult] = MagicMock()
            results = queue.process_tasks(generate_func, validate_func)

            # Verify
            assert len(results) == 1
            assert mock_executor_instance.submit.call_count == 1
            assert mock_future.result.call_count == 1

    def test_process_tasks_sequential(
        self, mock_config: QAAgentConfig, mock_test_task: TestTask
    ) -> None:
        """Test processing tasks sequentially."""
        # Setup
        mock_config.parallel_execution = False
        queue = TaskQueue(mock_config)
        queue.add_task(mock_test_task)

        # Mock the process_task method
        mock_result = MagicMock(spec=TaskResult, success=True)
        with patch.object(queue, "_process_task", return_value=mock_result) as mock_process:
            # Execute
            generate_func: Callable[[Function, List[CodeFile]], GeneratedTest] = MagicMock()
            validate_func: Callable[[GeneratedTest], TestResult] = MagicMock()
            results = queue.process_tasks(generate_func, validate_func)

            # Verify
            assert len(results) == 1
            assert results[0] == mock_result
            mock_process.assert_called_once_with(mock_test_task, generate_func, validate_func)

    def test_process_task(
        self,
        mock_config: QAAgentConfig,
        mock_test_task: TestTask,
        mock_generated_test: MagicMock,
        mock_test_result: MagicMock,
    ) -> None:
        """Test processing a single task."""
        # Setup
        queue = TaskQueue(mock_config)

        generate_func: Callable[[Function, List[CodeFile]], GeneratedTest] = MagicMock(
            return_value=mock_generated_test
        )
        validate_func: Callable[[GeneratedTest], TestResult] = MagicMock(
            return_value=mock_test_result
        )

        # Set success on the test result
        mock_test_result.success = True

        # Execute
        result = queue._process_task(mock_test_task, generate_func, validate_func)

        # Verify
        assert result.task == mock_test_task
        assert result.success == True
        assert result.generated_test == mock_generated_test
        assert result.test_result == mock_test_result
        assert mock_test_task.status == "completed"
        assert mock_test_task.generated_test == mock_generated_test
        assert mock_test_task.test_result == mock_test_result
        assert queue._processed_count == 1
        assert queue._success_count == 1
        assert queue._failure_count == 0

    def test_process_task_failure(
        self, mock_config: QAAgentConfig, mock_test_task: TestTask
    ) -> None:
        """Test processing a task that fails."""
        # Setup
        queue = TaskQueue(mock_config)

        generate_func: Callable[[Function, List[CodeFile]], GeneratedTest] = MagicMock(
            side_effect=Exception("Test error")
        )
        validate_func: Callable[[GeneratedTest], TestResult] = MagicMock()

        # Execute
        result = queue._process_task(mock_test_task, generate_func, validate_func)

        # Verify
        assert result.task == mock_test_task
        assert result.success == False
        assert result.generated_test is None
        assert result.test_result is None
        assert result.error == "Test error"
        assert mock_test_task.status == "failed"
        assert mock_test_task.error == "Test error"
        assert queue._processed_count == 1
        assert queue._success_count == 0
        assert queue._failure_count == 1


class TestCreateTaskQueue:
    """Tests for the create_task_queue function."""

    def test_create_task_queue(
        self,
        mock_config: QAAgentConfig,
        mock_function: Function,
        mock_context_files: List[CodeFile],
    ) -> None:
        """Test creating a task queue."""
        # Setup
        functions: List[Function] = [mock_function]
        context_files_map: Dict[str, List[CodeFile]] = {mock_function.file_path: mock_context_files}

        # Execute
        with patch(
            "qa_agent.task_queue.TaskQueue.filter_functions_by_changed_files",
            return_value=functions,
        ):
            queue = create_task_queue(mock_config, functions, context_files_map)

            # Verify
            assert isinstance(queue, TaskQueue)
            assert queue.config == mock_config
            assert queue.tasks.qsize() == 1

    def test_create_task_queue_with_incremental(
        self,
        mock_config: QAAgentConfig,
        mock_function: Function,
        mock_context_files: List[CodeFile],
    ) -> None:
        """Test creating a task queue with incremental testing."""
        # Setup
        mock_config.incremental_testing = True
        functions: List[Function] = [mock_function]
        filtered_functions: List[Function] = [mock_function]  # Same in this case
        context_files_map: Dict[str, List[CodeFile]] = {mock_function.file_path: mock_context_files}

        # Execute
        with patch(
            "qa_agent.task_queue.TaskQueue.filter_functions_by_changed_files",
            return_value=filtered_functions,
        ) as mock_filter:
            queue = create_task_queue(mock_config, functions, context_files_map)

            # Verify
            assert isinstance(queue, TaskQueue)
            assert queue.config == mock_config
            assert queue.tasks.qsize() == 1
            mock_filter.assert_called_once_with(functions, mock_config.changed_since_days)
