"""
Unit tests for the parallel workflow module.

These tests verify the functionality of the parallel implementation of the QA workflow
using task queues and parallel processing.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, cast

# Remember that pytest-mock is the preferred way to mock in tests
# This import is kept for backward compatibility
# IMPORTANT: When writing new tests, always use pytest's mocker fixture
# instead of importing directly from unittest.mock
from unittest.mock import MagicMock, patch

# Note: pytest import might be handled by pytest fixtures
try:
    import pytest
except ImportError:
    # For type checking and LSP purposes
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            return lambda f: f

    pytest = _MockPytest()

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, Function, GeneratedTest, TestResult
from qa_agent.parallel_workflow import ParallelQAWorkflow
from qa_agent.task_queue import TaskQueue, TaskResult, TestTask


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = QAAgentConfig()
    config.repo_path = "./test_repo"
    config.output_directory = "./test_output"
    config.parallel_execution = True
    config.max_workers = 2
    config.incremental_testing = False
    return config


@pytest.fixture
def mock_function():
    """Create a mock function for testing."""
    return Function(
        name="test_function",
        code="def test_function():\n    return True",
        file_path="test_module.py",
        start_line=1,
        end_line=2,
    )


@pytest.fixture
def mock_task_queue():
    """Create a mock task queue for testing."""
    return MagicMock(spec=TaskQueue)


class TestParallelQAWorkflow:
    """Tests for the ParallelQAWorkflow class."""

    def test_initialization(self, mock_config):
        """Test initialization with valid configuration."""
        workflow = ParallelQAWorkflow(mock_config)
        assert workflow.config == mock_config
        assert workflow.task_queue is None
        assert workflow.start_time is None
        assert workflow.end_time is None
        assert isinstance(workflow.stats, dict)

    @patch("qa_agent.parallel_workflow.CodeAnalysisAgent")
    def test_identify_functions(self, mock_agent_class, mock_config, mock_function):
        """Test the _identify_functions method."""
        # Setup
        mock_agent = mock_agent_class.return_value
        mock_agent.identify_critical_functions.return_value = [mock_function]

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow._identify_functions()

        # Verify
        assert result == [mock_function]
        mock_agent.identify_critical_functions.assert_called_once()

    @patch("qa_agent.parallel_workflow.CodeAnalysisAgent")
    def test_identify_functions_empty(self, mock_agent_class, mock_config):
        """Test the _identify_functions method when no critical functions are found."""
        # Setup
        mock_agent = mock_agent_class.return_value
        mock_agent.identify_critical_functions.return_value = []
        mock_agent.get_uncovered_functions.return_value = []

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow._identify_functions()

        # Verify
        assert result == []
        mock_agent.identify_critical_functions.assert_called_once()
        mock_agent.get_uncovered_functions.assert_called_once()

    @patch("qa_agent.parallel_workflow.CodeAnalysisAgent")
    def test_get_function_contexts(self, mock_agent_class, mock_config, mock_function):
        """Test the _get_function_contexts method."""
        # Setup
        mock_agent = mock_agent_class.return_value
        mock_context_files = [MagicMock(spec=CodeFile)]
        mock_agent.get_function_context.return_value = mock_context_files

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow._get_function_contexts([mock_function])

        # Verify
        assert result == {mock_function.file_path: mock_context_files}
        mock_agent.get_function_context.assert_called_once_with(mock_function)

    @patch("qa_agent.parallel_workflow.TestGenerationAgent")
    def test_generate_and_validate_test(self, mock_agent_class, mock_config, mock_function):
        """Test the _generate_and_validate_test method."""
        # Setup
        mock_agent = mock_agent_class.return_value
        mock_test = MagicMock()
        mock_agent.generate_test.return_value = mock_test
        # Create a proper List[CodeFile] with actual CodeFile instances
        mock_code_file = CodeFile(path="test/path.py", content="test content")
        mock_context_files = [mock_code_file]

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow._generate_and_validate_test(mock_function, mock_context_files)

        # Verify
        assert result == mock_test
        mock_agent.generate_test.assert_called_once_with(mock_function, mock_context_files)

    @patch("qa_agent.parallel_workflow.TestValidationAgent")
    def test_validate_test(self, mock_agent_class, mock_config):
        """Test the _validate_test method."""
        # Setup
        mock_agent = mock_agent_class.return_value
        mock_test = MagicMock()
        mock_result = MagicMock()
        mock_agent.validate_test.return_value = mock_result

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow._validate_test(mock_test)

        # Verify
        assert result == mock_result
        mock_agent.validate_test.assert_called_once_with(mock_test)

    @patch("qa_agent.parallel_workflow.TestValidationAgent")
    def test_fix_test(self, mock_agent_class, mock_config):
        """Test the _fix_test method."""
        # Setup
        mock_agent = mock_agent_class.return_value
        mock_test = MagicMock()
        mock_test_result = MagicMock()
        mock_fixed_test = MagicMock()
        mock_agent.fix_test.return_value = mock_fixed_test

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow._fix_test(mock_test, mock_test_result)

        # Verify
        assert result == mock_fixed_test
        mock_agent.fix_test.assert_called_once_with(mock_test, mock_test_result)

    def test_process_results(self, mock_config, mock_function):
        """Test the _process_results method."""
        # Setup
        workflow = ParallelQAWorkflow(mock_config)
        workflow.start_time = datetime.now()
        workflow.end_time = datetime.now()

        mock_task = MagicMock(spec=TestTask)
        mock_task.function = mock_function

        # Create mock GeneratedTest objects
        mock_generated_test_1 = MagicMock()
        mock_generated_test_1.test_file_path = "test_file_1.py"

        mock_generated_test_2 = MagicMock()
        mock_generated_test_2.test_file_path = "test_file_2.py"

        # Create mock test results
        mock_test_result_1 = MagicMock(spec=TestResult)
        mock_test_result_1.success = True
        mock_test_result_1.error_message = None

        mock_test_result_2 = MagicMock(spec=TestResult)
        mock_test_result_2.success = False
        mock_test_result_2.error_message = "Test execution error"

        # Create actual TaskResult objects (not mocks) to ensure they have all required attributes
        mock_result_1 = TaskResult(
            task=mock_task,
            success=True,
            generated_test=mock_generated_test_1,
            test_result=mock_test_result_1,
            error=None,
        )

        mock_result_2 = TaskResult(
            task=mock_task,
            success=False,
            generated_test=mock_generated_test_2,
            test_result=mock_test_result_2,
            error="Test error",
        )

        results = [mock_result_1, mock_result_2]

        # Execute
        with patch("builtins.open", MagicMock()):
            with patch("json.dump") as mock_json_dump:
                final_state = workflow._process_results(results)

        # Verify
        assert final_state["functions"] == [mock_function, mock_function]
        assert final_state["success_count"] == 1
        assert final_state["failure_count"] == 1
        assert final_state["status"] == "Completed: 1 successful, 1 failed"
        assert "stats" in final_state

    @patch("qa_agent.parallel_workflow.create_task_queue")
    @patch("qa_agent.parallel_workflow.os.makedirs")
    @patch("qa_agent.parallel_workflow.ParallelQAWorkflow._identify_functions")
    @patch("qa_agent.parallel_workflow.ParallelQAWorkflow._get_function_contexts")
    def test_run(
        self,
        mock_get_contexts,
        mock_identify,
        mock_makedirs,
        mock_create_queue,
        mock_config,
        mock_function,
        mock_task_queue,
    ):
        """Test the run method."""
        # Setup
        mock_identify.return_value = [mock_function]
        mock_contexts = {mock_function.file_path: []}
        mock_get_contexts.return_value = mock_contexts
        mock_create_queue.return_value = mock_task_queue

        # Create mock test result
        mock_test_result = MagicMock(spec=TestResult)
        mock_test_result.success = True
        mock_test_result.error_message = None

        # Create mock GeneratedTest object
        mock_generated_test = MagicMock()
        mock_generated_test.test_file_path = "test_file.py"

        # Create a task
        mock_task = MagicMock(spec=TestTask)
        mock_task.function = mock_function

        # Create actual TaskResult object (not mock)
        mock_result = TaskResult(
            task=mock_task,
            success=True,
            generated_test=mock_generated_test,
            test_result=mock_test_result,
        )

        mock_task_queue.process_tasks.return_value = [mock_result]

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        with patch.object(workflow, "_process_results") as mock_process:
            mock_process.return_value = {"status": "Success"}
            result = workflow.run()

        # Verify
        assert result == {"status": "Success"}
        mock_makedirs.assert_called_once_with(mock_config.output_directory, exist_ok=True)
        mock_identify.assert_called_once()
        mock_get_contexts.assert_called_once_with([mock_function])
        mock_create_queue.assert_called_once_with(mock_config, [mock_function], mock_contexts)
        mock_task_queue.process_tasks.assert_called_once()

    @patch("qa_agent.parallel_workflow.ParallelQAWorkflow._identify_functions")
    def test_run_with_no_functions(self, mock_identify, mock_config):
        """Test the run method when no functions are identified."""
        # Setup
        mock_identify.return_value = []

        # Execute
        workflow = ParallelQAWorkflow(mock_config)
        result = workflow.run()

        # Verify
        assert result["status"] == "No functions to test found"
        assert result["functions"] == []
        assert result["success_count"] == 0
        assert result["failure_count"] == 0
