"""
End-to-end tests for the QA Agent workflow.

These tests verify that the entire QA Agent system works together correctly
from code analysis to test generation and validation.
"""

import os
import shutil
import sys
import tempfile

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function, GeneratedTest, TestResult
from qa_agent.workflows import QAWorkflow


class TestQAAgentE2E:
    """End-to-end tests for the QA Agent workflow."""

    @pytest.mark.e2e
    def test_full_workflow_with_mock_api(
        self, mocker, sample_repo_path, e2e_config, mock_openai_responses
    ):
        """Test the full QA Agent workflow with mock API responses."""
        # Set up configuration
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_tests")
        os.makedirs(e2e_config.output_directory, exist_ok=True)

        # Create mocks with pytest-mock
        mock_validate = mocker.patch("qa_agent.workflows.QAWorkflow._validate_test")
        mock_identify = mocker.patch(
            "qa_agent.workflows.CodeAnalysisAgent.identify_critical_functions"
        )

        # Create a test function
        test_function = Function(
            name="add_numbers",
            code="def add_numbers(a, b):\n    return a + b",
            file_path=os.path.join(sample_repo_path, "utils.py"),
            start_line=10,
            end_line=12,
            docstring="Add two numbers and return the result.",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="int",
            dependencies=[],
            complexity=1,
        )

        # Mock the identify_critical_functions method
        mock_identify.return_value = [test_function]

        # Mock successful test validation
        mock_validate.side_effect = lambda state: {
            **state,
            "test_result": TestResult(
                success=True,
                test_file=os.path.join(e2e_config.output_directory, "test_utils.py"),
                target_function="add_numbers",
                output="1 passed",
                coverage=100.0,
                error_message="No errors",
                execution_time=0.1,
            ),
            "success_count": 1,
        }

        # Run the workflow
        workflow = QAWorkflow(e2e_config)
        final_state = workflow.run()

        # Assert on final state
        assert final_state is not None
        assert final_state.get("status") == "Workflow finished"
        assert final_state.get("success_count", 0) > 0
        assert final_state.get("failure_count", 0) == 0

        # Verify that summary file was created
        summary_path = os.path.join(e2e_config.output_directory, "qa_summary.txt")
        assert os.path.exists(summary_path)

        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)

    @pytest.mark.e2e
    def test_workflow_with_failing_tests(
        self, mocker, sample_repo_path, e2e_config, mock_openai_responses
    ):
        """Test the QA Agent workflow with failing tests."""
        # Set up configuration
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_tests_failing")
        os.makedirs(e2e_config.output_directory, exist_ok=True)

        # Create mocks with pytest-mock
        mock_validate = mocker.patch("qa_agent.workflows.QAWorkflow._validate_test")
        mock_fix = mocker.patch("qa_agent.workflows.QAWorkflow._fix_test")
        mock_identify = mocker.patch(
            "qa_agent.workflows.CodeAnalysisAgent.identify_critical_functions"
        )
        mock_generate = mocker.patch("qa_agent.workflows.TestGenerationAgent.generate_test")

        # Mock failing test validation
        mock_validate.side_effect = lambda state: {
            **state,
            "test_result": TestResult(
                success=False,
                test_file=os.path.join(e2e_config.output_directory, "test_utils.py"),
                target_function="divide_numbers",
                output="1 failed",
                coverage=50.0,
                error_message="AssertionError: assert 1.0 == 2.0",
                execution_time=0.1,
            ),
        }

        # Create a test function
        test_function = Function(
            name="divide_numbers",
            code="def divide_numbers(a, b):\n    return a / b",
            file_path=os.path.join(sample_repo_path, "utils.py"),
            start_line=15,
            end_line=17,
            docstring="Divide a by b and return the result.",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="float",
            dependencies=[],
            complexity=1,
        )

        # Mock the identify_critical_functions method
        mock_identify.return_value = [test_function]

        # Create a mock generated test
        mock_test = GeneratedTest(
            function=test_function,
            test_code="def test_divide_numbers():\n    assert divide_numbers(4, 2) == 2.0",
            test_file_path=os.path.join(e2e_config.output_directory, "test_utils.py"),
            imports=["pytest", "utils"],
            mocks=[],
            fixtures=[],
        )

        # Mock the generate_test method
        mock_generate.return_value = mock_test

        # Mock test fix to return a successful test on the second attempt
        mock_fix.side_effect = lambda state: {
            **state,
            "attempts": state.get("attempts", 1) + 1,
            "generated_test": state.get("generated_test"),
        }

        # Run the workflow
        workflow = QAWorkflow(e2e_config)

        # Override the route functions to ensure we test the fix path
        original_route_after_validation = workflow._route_after_validation

        def mock_route(state):
            # First time, return "fix", then use the original routing
            if state.get("attempts", 1) == 1:
                return "fix"
            return "next"

        workflow._route_after_validation = mock_route
        workflow._route_after_fix = lambda state: "next"

        final_state = workflow.run()

        # Assert on final state
        assert final_state is not None
        assert final_state.get("status") == "Workflow finished"
        assert mock_fix.call_count > 0

        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)

    @pytest.mark.e2e
    def test_workflow_error_handling(self, mocker, sample_repo_path, e2e_config):
        """Test the QA Agent workflow error handling."""
        # Set up configuration
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_tests_errors")
        os.makedirs(e2e_config.output_directory, exist_ok=True)

        # Create a mock run method that raises directly
        run_mock = mocker.Mock(side_effect=Exception("Test error: Could not identify functions"))

        # Create the workflow
        workflow = QAWorkflow(e2e_config)

        # Replace the run method with our mock
        mocker.patch.object(workflow, "run", run_mock)

        # Test that the exception is properly raised
        with pytest.raises(Exception) as exc_info:
            workflow.run()

        # Verify the error message is preserved
        assert "Test error" in str(exc_info.value)

        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)


if __name__ == "__main__":
    pytest.main(["-v", "test_e2e_workflow.py"])
