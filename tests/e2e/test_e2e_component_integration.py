"""
End-to-end tests for component integration in the QA Agent.

These tests verify that the different components of the QA Agent
work together correctly in integration scenarios.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from qa_agent.agents import CodeAnalysisAgent, TestGenerationAgent, TestValidationAgent
from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function, GeneratedTest, TestResult


class TestComponentIntegration:
    """End-to-end tests for component integration in the QA Agent."""

    @pytest.mark.e2e
    def test_code_analysis_to_test_generation(
        self, mocker, sample_repo_path, e2e_config, mock_openai_responses
    ):
        """Test the flow from code analysis to test generation."""
        # Set up configuration
        e2e_config.repo_path = sample_repo_path

        # Create a sample function
        function = Function(
            name="add_numbers",
            code="def add_numbers(a, b):\n    return a + b",
            file_path=os.path.join(sample_repo_path, "sample_module", "utils.py"),
            start_line=10,
            end_line=12,
            docstring="Add two numbers and return the result.",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="int",
            dependencies=[],
            complexity=1,
        )

        # Create sample context files
        context_file = CodeFile(
            path=os.path.join(sample_repo_path, "sample_module", "utils.py"),
            content="def add_numbers(a, b):\n    return a + b\n\ndef multiply_numbers(a, b):\n    return a * b",
            type=FileType.PYTHON,
        )

        # Create a mock generated test
        generated_test = GeneratedTest(
            function=function,
            test_code="import pytest\nfrom sample_module.utils import add_numbers\n\ndef test_add_numbers():\n    assert add_numbers(1, 2) == 3\n",
            test_file_path=os.path.join(sample_repo_path, "tests", "test_add_numbers.py"),
            imports=["pytest", "sample_module.utils.add_numbers"],
            mocks=[],
            fixtures=[],
        )

        # Mock the agents
        mocker.patch(
            "qa_agent.agents.CodeAnalysisAgent.identify_critical_functions", return_value=[function]
        )
        mocker.patch(
            "qa_agent.agents.CodeAnalysisAgent.get_function_context", return_value=[context_file]
        )
        mocker.patch(
            "qa_agent.test_generator.TestGenerator.generate_test", return_value=generated_test
        )
        mocker.patch("qa_agent.test_generator.TestGenerator.save_test_to_file")

        # Initialize agents
        code_analysis_agent = CodeAnalysisAgent(e2e_config)
        test_generation_agent = TestGenerationAgent(e2e_config)

        # Identify critical functions
        functions = code_analysis_agent.identify_critical_functions()
        assert len(functions) == 1
        assert functions[0].name == "add_numbers"

        # Get context for the function
        context_files = code_analysis_agent.get_function_context(functions[0])
        assert len(context_files) == 1

        # Generate a test for the function
        result = test_generation_agent.generate_test(functions[0], context_files)

        assert result is not None
        assert result.function.name == "add_numbers"
        assert "test_add_numbers" in result.test_code
        assert "pytest" in result.imports

    @pytest.mark.e2e
    def test_test_generation_to_validation(
        self, mocker, sample_repo_path, e2e_config, mock_openai_responses
    ):
        """Test the flow from test generation to test validation."""
        # Set up configuration
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_tests")
        os.makedirs(e2e_config.output_directory, exist_ok=True)

        # Create a sample function
        function = Function(
            name="add_numbers",
            code="def add_numbers(a, b):\n    return a + b",
            file_path=os.path.join(sample_repo_path, "sample_module", "utils.py"),
            start_line=10,
            end_line=12,
            docstring="Add two numbers and return the result.",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="int",
            dependencies=[],
            complexity=1,
        )

        # Create a mock generated test
        generated_test = GeneratedTest(
            function=function,
            test_code="import pytest\nfrom sample_module.utils import add_numbers\n\ndef test_add_numbers():\n    assert add_numbers(1, 2) == 3\n",
            test_file_path=os.path.join(sample_repo_path, "tests", "test_add_numbers.py"),
            imports=["pytest", "sample_module.utils.add_numbers"],
            mocks=[],
            fixtures=[],
        )

        # Mock test generation and saving
        mocker.patch("qa_agent.test_generator.TestGenerator.save_test_to_file")
        mocker.patch(
            "qa_agent.test_generator.TestGenerator.generate_test", return_value=generated_test
        )

        # Mock file existence and pytest run
        mocker.patch("os.path.exists", return_value=True)
        mock_run = mocker.patch("subprocess.run")

        # Initialize agents
        test_generation_agent = TestGenerationAgent(e2e_config)
        test_validation_agent = TestValidationAgent(e2e_config)

        # Generate test
        result = test_generation_agent.generate_test(function)

        # Verify test was generated
        assert result is not None
        assert result.function.name == "add_numbers"

        # Create a successful test result
        successful_result = TestResult(
            success=True,
            test_file=os.path.join(sample_repo_path, "tests", "test_add_numbers.py"),
            target_function="add_numbers",
            output="===== test session starts =====\ncollected 1 item\n\ntest_file.py::test_add_numbers PASSED\n\n====== 1 passed in 0.01s ======",
            coverage=100.0,
        )

        # Create a failing test result
        failing_result = TestResult(
            success=False,
            test_file=os.path.join(sample_repo_path, "tests", "test_add_numbers.py"),
            target_function="add_numbers",
            output="===== test session starts =====\ncollected 1 item\n\ntest_file.py::test_add_numbers FAILED\n\n====== 1 failed in 0.01s ======",
            error_message="AssertionError: expected 3, got 4",
        )

        # Mock successful test run
        mock_process = mocker.MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "===== test session starts =====\ncollected 1 item\n\ntest_file.py::test_add_numbers PASSED\n\n====== 1 passed in 0.01s ======"
        mock_run.return_value = mock_process

        # Mock the validation result
        mocker.patch(
            "qa_agent.test_validator.TestValidator.validate_test",
            side_effect=[successful_result, failing_result],
        )

        # Validate the test
        test_result = test_validation_agent.validate_test(result)

        # Verify test validation result
        assert test_result is not None
        assert test_result.success is True
        assert test_result.target_function == "add_numbers"

        # Validate again (should fail this time due to our side_effect)
        test_result = test_validation_agent.validate_test(result)

        # Verify test validation result
        assert test_result is not None
        assert test_result.success is False
        assert test_result.target_function == "add_numbers"

        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)

    @pytest.mark.e2e
    def test_full_component_chain(
        self, mocker, sample_repo_path, e2e_config, mock_openai_responses
    ):
        """Test the full chain of components working together."""
        # Set up configuration
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_tests_chain")
        os.makedirs(e2e_config.output_directory, exist_ok=True)

        # Create a sample function
        function = Function(
            name="add_numbers",
            code="def add_numbers(a, b):\n    return a + b",
            file_path=os.path.join(sample_repo_path, "sample_module", "utils.py"),
            start_line=10,
            end_line=12,
            docstring="Add two numbers and return the result.",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="int",
            dependencies=[],
            complexity=1,
        )

        # Create sample context files
        context_file = CodeFile(
            path=os.path.join(sample_repo_path, "sample_module", "utils.py"),
            content="def add_numbers(a, b):\n    return a + b\n\ndef multiply_numbers(a, b):\n    return a * b",
            type=FileType.PYTHON,
        )

        # Create a mock generated test
        generated_test = GeneratedTest(
            function=function,
            test_code="import pytest\nfrom sample_module.utils import add_numbers\n\ndef test_add_numbers():\n    assert add_numbers(1, 2) == 3\n",
            test_file_path=os.path.join(sample_repo_path, "tests", "test_add_numbers.py"),
            imports=["pytest", "sample_module.utils.add_numbers"],
            mocks=[],
            fixtures=[],
        )

        # Create a successful test result
        successful_result = TestResult(
            success=True,
            test_file=os.path.join(sample_repo_path, "tests", "test_add_numbers.py"),
            target_function="add_numbers",
            output="===== test session starts =====\ncollected 1 item\n\ntest_file.py::test_add_numbers PASSED\n\n====== 1 passed in 0.01s ======",
            coverage=100.0,
        )

        # Initialize agents after mocking
        code_analysis_agent = CodeAnalysisAgent(e2e_config)
        test_generation_agent = TestGenerationAgent(e2e_config)
        test_validation_agent = TestValidationAgent(e2e_config)

        # Mock all components
        mocker.patch(
            "qa_agent.agents.CodeAnalysisAgent.identify_critical_functions", return_value=[function]
        )
        mocker.patch(
            "qa_agent.agents.CodeAnalysisAgent.get_function_context", return_value=[context_file]
        )
        mocker.patch(
            "qa_agent.test_generator.TestGenerator.generate_test", return_value=generated_test
        )
        mocker.patch("qa_agent.test_generator.TestGenerator.save_test_to_file")
        mocker.patch(
            "qa_agent.test_validator.TestValidator.validate_test", return_value=successful_result
        )
        mocker.patch("os.path.exists", return_value=True)

        # 1. Identify critical functions
        functions = code_analysis_agent.identify_critical_functions()
        assert len(functions) == 1

        # 2. Get context for the first function
        context_files = code_analysis_agent.get_function_context(functions[0])
        assert len(context_files) == 1

        # 3. Generate a test for the function
        result = test_generation_agent.generate_test(functions[0], context_files)
        assert result is not None
        assert result.function.name == "add_numbers"
        assert "test_add_numbers" in result.test_code
        assert "pytest" in result.imports

        # 4. Validate the test
        test_result = test_validation_agent.validate_test(result)
        assert test_result is not None
        assert test_result.success is True
        assert test_result.target_function == "add_numbers"

        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)


if __name__ == "__main__":
    pytest.main(["-v", "test_e2e_component_integration.py"])
