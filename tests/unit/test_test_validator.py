"""
Unit tests for the test validator module.

These tests verify the functionality of running and validating generated tests.
"""

import os
from unittest.mock import MagicMock, mock_open

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import Function, GeneratedTest, TestResult
from qa_agent.test_validator import TestValidator


class TestTestValidator:
    """Tests for the TestValidator class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return QAAgentConfig(
            model_provider="openai",
            api_key="test-api-key",
            model_name="gpt-4o",
            repo_path="/test/repo",
            test_framework="pytest",
        )

    @pytest.fixture
    def mock_function(self):
        """Mock function for testing."""
        return Function(
            name="example_function",
            code="def example_function(a, b):\n    return a + b",
            file_path="/test/repo/src/module.py",
            start_line=10,
            end_line=12,
        )

    @pytest.fixture
    def mock_generated_test(self, mock_function):
        """Mock generated test for testing."""
        return GeneratedTest(
            function=mock_function,
            test_code="def test_example_function():\n    assert example_function(1, 2) == 3",
            test_file_path="/test/repo/tests/test_module.py",
            imports=["pytest"],
            mocks=[],
            fixtures=[],
        )

    def test_initialization(self, mock_config):
        """Test initialization with default settings."""
        validator = TestValidator(mock_config)

        assert validator.config == mock_config
        assert validator.copilot_adapter is None

    def test_initialization_with_copilot(self, mocker):
        """Test initialization with GitHub Copilot provider."""
        # Create config with GitHub Copilot
        config = QAAgentConfig(
            model_provider="github-copilot",
            api_key="test-api-key",
            copilot_settings={
                "endpoint": "https://api.github.com/copilot",
                "model_version": "latest",
            },
        )

        # Mock CopilotAdapter
        mock_copilot_adapter = mocker.patch("qa_agent.test_validator.CopilotAdapter")

        # Initialize validator
        validator = TestValidator(config)

        # Verify initialization
        assert validator.config == config
        mock_copilot_adapter.assert_called_once_with(config)

    def test_validate_nonexistent_test(self, mocker, mock_config, mock_generated_test):
        """Test validating a test that doesn't exist."""
        # Mock os.path.exists to return False
        mock_exists = mocker.patch("os.path.exists", return_value=False)

        # Create validator and run test
        validator = TestValidator(mock_config)
        result = validator.validate_test(mock_generated_test)

        # Verify result
        assert isinstance(result, TestResult)
        assert not result.success
        assert result.test_file == mock_generated_test.test_file_path
        assert result.target_function == mock_generated_test.function.name
        assert "Test file does not exist" in result.error_message

    def test_validate_successful_test(self, mocker, mock_config, mock_generated_test):
        """Test validating a successful test."""
        # Mock dependencies
        mock_exists = mocker.patch("os.path.exists", return_value=True)
        mock_chdir = mocker.patch("os.chdir")
        mock_getcwd = mocker.patch("os.getcwd", return_value="/original/dir")
        mock_run = mocker.patch("subprocess.run")

        # Setup mock subprocess result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = (
            "===== test session starts =====\n"
            + "collected 1 item\n"
            + "test_module.py::test_example_function PASSED [ 100%]\n"
            + "===== 1 passed in 0.12s =====\n"
            + "===== 100% coverage ====="
        )
        mock_run.return_value = mock_process

        # Create validator and run test
        validator = TestValidator(mock_config)
        result = validator.validate_test(mock_generated_test)

        # Verify result
        assert isinstance(result, TestResult)
        assert result.success
        assert result.test_file == mock_generated_test.test_file_path
        assert result.target_function == mock_generated_test.function.name
        assert result.output == mock_process.stdout
        assert result.coverage == 100.0

        # Verify directory changes
        mock_chdir.assert_any_call(mock_config.repo_path)
        mock_chdir.assert_any_call("/original/dir")

        # Verify subprocess call
        mock_run.assert_called_once()
        pytest_cmd = mock_run.call_args[0][0]
        assert (
            pytest_cmd[0].endswith("pytest")
            or pytest_cmd[0].endswith("python")
            and "pytest" in " ".join(pytest_cmd)
        )
        assert mock_generated_test.test_file_path in pytest_cmd

    def test_validate_failing_test(self, mocker, mock_config, mock_generated_test):
        """Test validating a failing test."""
        # Mock dependencies
        mock_exists = mocker.patch("os.path.exists", return_value=True)
        mock_chdir = mocker.patch("os.chdir")
        mock_getcwd = mocker.patch("os.getcwd", return_value="/original/dir")
        mock_run = mocker.patch("subprocess.run")

        # Setup mock subprocess result
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = (
            "===== test session starts =====\n"
            + "collected 1 item\n"
            + "test_module.py::test_example_function FAILED [ 100%]\n"
            + "E    AssertionError: assert 3 == 4\n"
            + "===== 1 failed in 0.12s =====\n"
            + "===== 50% coverage ====="
        )
        mock_run.return_value = mock_process

        # Create validator and run test
        validator = TestValidator(mock_config)
        result = validator.validate_test(mock_generated_test)

        # Verify result
        assert isinstance(result, TestResult)
        assert not result.success
        assert result.test_file == mock_generated_test.test_file_path
        assert result.target_function == mock_generated_test.function.name
        assert result.output == mock_process.stdout
        assert result.coverage == 50.0
        assert "AssertionError" in result.error_message

    def test_validate_exception_handling(self, mocker, mock_config, mock_generated_test):
        """Test handling exceptions during test validation."""
        # Mock dependencies
        mock_exists = mocker.patch("os.path.exists", return_value=True)
        mock_chdir = mocker.patch("os.chdir")
        mock_getcwd = mocker.patch("os.getcwd", return_value="/original/dir")
        mock_run = mocker.patch("subprocess.run", side_effect=Exception("Command execution failed"))
        mock_error = mocker.patch("logging.Logger.error")

        # Create validator and run test
        validator = TestValidator(mock_config)
        result = validator.validate_test(mock_generated_test)

        # Verify result
        assert isinstance(result, TestResult)
        assert not result.success
        assert result.test_file == mock_generated_test.test_file_path
        assert result.target_function == mock_generated_test.function.name
        assert "Command execution failed" in result.error_message

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error validating test" in mock_error.call_args[0][0]

        # Verify original directory was restored
        mock_chdir.assert_any_call("/original/dir")

    def test_extract_coverage(self, mock_config):
        """Test extracting coverage from test output."""
        validator = TestValidator(mock_config)

        # Test with coverage line
        output_with_coverage = (
            "===== test session starts =====\n"
            + "collected 1 item\n"
            + "test_module.py::test_example_function PASSED [ 100%]\n"
            + "===== 1 passed in 0.12s =====\n"
            + "Coverage: 75.5%"
        )

        coverage = validator._extract_coverage(output_with_coverage)
        assert coverage == 75.5

        # Test without coverage line
        output_no_coverage = (
            "===== test session starts =====\n"
            + "collected 1 item\n"
            + "test_module.py::test_example_function PASSED [ 100%]\n"
            + "===== 1 passed in 0.12s ====="
        )

        coverage = validator._extract_coverage(output_no_coverage)
        assert coverage is None

    def test_fix_test_with_llm(self, mocker, mock_config, mock_generated_test):
        """Test fixing a test using LLM."""
        # Mock dependencies
        mock_exists = mocker.patch("os.path.exists", return_value=True)
        mock_runnable_sequence = mocker.patch("qa_agent.test_validator.RunnableSequence")
        mock_chat_openai = mocker.patch("qa_agent.test_validator.ChatOpenAI")
        mock_str_output_parser = mocker.patch("qa_agent.test_validator.StrOutputParser")

        # Create test result with failure
        test_result = TestResult(
            success=False,
            test_file=mock_generated_test.test_file_path,
            target_function=mock_generated_test.function.name,
            output="===== test session starts =====\n"
            + "collected 1 item\n"
            + "test_module.py::test_example_function FAILED [ 100%]\n"
            + "E    AssertionError: assert 3 == 4\n"
            + "===== 1 failed in 0.12s =====",
            error_message="AssertionError: assert 3 == 4",
        )

        # Setup mock RunnableSequence
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = """
```python
def test_example_function():
    # Fixed test
    assert example_function(2, 2) == 4
```
"""
        mock_runnable_sequence.from_components.return_value = mock_chain

        # Create validator and fix test
        validator = TestValidator(mock_config)
        fixed_test = validator.fix_test(mock_generated_test, test_result)

        # Verify results
        assert isinstance(fixed_test, GeneratedTest)
        assert fixed_test.function == mock_generated_test.function
        assert "assert example_function(2, 2) == 4" in fixed_test.test_code
        assert fixed_test.test_file_path == mock_generated_test.test_file_path
        assert fixed_test.imports == mock_generated_test.imports

        # Verify RunnableSequence was called with expected parameters
        mock_chain.invoke.assert_called_once()
        # Parameters are now passed as a dictionary to invoke
        params = mock_chain.invoke.call_args[0][0]
        assert params["function_name"] == mock_generated_test.function.name
        assert params["function_code"] == mock_generated_test.function.code
        assert params["test_code"] == mock_generated_test.test_code
        assert params["error_message"] == test_result.error_message

    def test_fix_test_with_copilot(self, mocker):
        """Test fixing a test using GitHub Copilot."""
        # Mock dependencies
        mock_exists = mocker.patch("os.path.exists", return_value=True)
        mock_copilot_adapter_class = mocker.patch("qa_agent.test_validator.CopilotAdapter")

        # Create GitHub Copilot config
        config = QAAgentConfig(
            model_provider="github-copilot",
            api_key="test-api-key",
            copilot_settings={
                "endpoint": "https://api.github.com/copilot",
                "model_version": "latest",
            },
        )

        # Create test function
        function = Function(
            name="example_function",
            code="def example_function(a, b):\n    return a + b",
            file_path="/test/repo/src/module.py",
            start_line=10,
            end_line=12,
        )

        # Create generated test with error
        generated_test = GeneratedTest(
            function=function,
            test_code="def test_example_function():\n    assert example_function(1, 2) == 4",  # Incorrect test
            test_file_path="/test/repo/tests/test_module.py",
            imports=["pytest"],
            mocks=[],
            fixtures=[],
        )

        # Create test result with failure
        test_result = TestResult(
            success=False,
            test_file=generated_test.test_file_path,
            target_function=generated_test.function.name,
            output="===== test session starts =====\n"
            + "collected 1 item\n"
            + "test_module.py::test_example_function FAILED [ 100%]\n"
            + "E    AssertionError: assert 4 == 3\n"
            + "===== 1 failed in 0.12s =====",
            error_message="AssertionError: assert 4 == 3",
        )

        # Setup mock Copilot adapter
        mock_adapter = MagicMock()
        mock_adapter.refine_test.return_value = (
            "def test_example_function():\n    assert example_function(1, 2) == 3"
        )
        mock_copilot_adapter_class.return_value = mock_adapter

        # Create validator and fix test
        validator = TestValidator(config)
        fixed_test = validator.fix_test(generated_test, test_result)

        # Verify results
        assert isinstance(fixed_test, GeneratedTest)
        assert fixed_test.function == generated_test.function
        assert "assert example_function(1, 2) == 3" in fixed_test.test_code
        assert fixed_test.test_file_path == generated_test.test_file_path

        # Verify Copilot adapter was called with expected parameters
        mock_adapter.refine_test.assert_called_once_with(
            generated_test.test_code,
            {
                "success": False,
                "test_file": generated_test.test_file_path,
                "target_function": generated_test.function.name,
                "output": test_result.output,
                "error_message": test_result.error_message,
            },
        )

    def test_extract_code_from_response(self, mock_config):
        """Test extracting code from LLM response."""
        validator = TestValidator(mock_config)

        # Response with code block
        response_with_code = """
Here's the fixed test:

```python
def test_example_function():
    # Fixed assertion
    result = example_function(1, 2)
    assert result == 3
```

The issue was that the expected value was incorrect.
"""

        code = validator._extract_code_from_response(response_with_code)
        assert "def test_example_function():" in code
        assert "result = example_function(1, 2)" in code
        assert "assert result == 3" in code

        # Response without code block
        response_without_code = """
The test needs to be fixed by correcting the expected value.
Instead of expecting 4, it should expect 3.
"""

        code = validator._extract_code_from_response(response_without_code)
        assert code == response_without_code
