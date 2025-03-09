"""
End-to-end tests for the QA Agent CLI.

These tests verify that the CLI interface works correctly and handles
different command-line arguments and error conditions appropriately.
"""

import argparse
import os
import shutil
import sys
import tempfile

import pytest

from qa_agent.cli import run_cli
from qa_agent.config import QAAgentConfig
from qa_agent.workflows import QAWorkflow


class TestQAAgentCLI:
    """End-to-end tests for the QA Agent CLI."""

    @pytest.mark.e2e
    def test_cli_with_valid_args(self, mocker, sample_repo_path, disable_api_calls):
        """Test CLI with valid arguments."""
        # Create arguments
        args = argparse.Namespace(
            repo_path=sample_repo_path,
            config=None,
            verbose=True,
            output=os.path.join(tempfile.gettempdir(), "qa_agent_cli_tests"),
            target_coverage=80.0,
            enable_sourcegraph=False,
            sourcegraph_endpoint=None,
            sourcegraph_token=None,
        )

        # Mock the workflow run using pytest-mock
        mock_run = mocker.patch("qa_agent.workflows.QAWorkflow.run")
        mock_run.return_value = {
            "status": "Workflow finished",
            "success_count": 2,
            "failure_count": 0,
            "functions": ["function1", "function2"],
        }

        # Mock os.environ to include API key
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})

        # Run CLI
        run_cli(args)

        # Assert workflow was created and run
        assert mock_run.call_count == 1

    @pytest.mark.e2e
    def test_cli_with_missing_repo_path(self, disable_api_calls):
        """Test CLI with missing repository path."""
        # Create arguments with missing repo_path
        args = argparse.Namespace(
            repo_path=None,
            config=None,
            verbose=True,
            output=os.path.join(tempfile.gettempdir(), "qa_agent_cli_tests"),
            target_coverage=80.0,
            enable_sourcegraph=False,
            sourcegraph_endpoint=None,
            sourcegraph_token=None,
        )

        # Run CLI and expect system exit
        with pytest.raises(SystemExit) as exc_info:
            run_cli(args)

        # Check exit code
        assert exc_info.value.code == 1

    @pytest.mark.e2e
    def test_cli_with_invalid_repo_path(self, disable_api_calls):
        """Test CLI with invalid repository path."""
        # Create arguments with invalid repo_path
        args = argparse.Namespace(
            repo_path="/nonexistent/path",
            config=None,
            verbose=True,
            output=os.path.join(tempfile.gettempdir(), "qa_agent_cli_tests"),
            target_coverage=80.0,
            enable_sourcegraph=False,
            sourcegraph_endpoint=None,
            sourcegraph_token=None,
        )

        # Run CLI and expect system exit
        with pytest.raises(SystemExit) as exc_info:
            run_cli(args)

        # Check exit code
        assert exc_info.value.code == 1

    @pytest.mark.e2e
    def test_cli_with_custom_config(self, mocker, sample_repo_path, disable_api_calls):
        """Test CLI with custom configuration file."""
        # Create a temporary config file
        config_file = os.path.join(tempfile.gettempdir(), "qa_agent_config.yaml")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        with open(config_file, "w") as f:
            f.write(
                """
model_provider: openai
model_name: gpt-4o
repo_path: /tmp/repo
test_framework: pytest
output_directory: /tmp/output
target_coverage: 90.0
verbose: true
            """
            )

        # Create arguments with config file
        args = argparse.Namespace(
            repo_path=sample_repo_path,  # This should override the config file
            config=config_file,
            verbose=True,
            output=None,  # Use config file value
            target_coverage=None,  # Use config file value
            enable_sourcegraph=False,
            sourcegraph_endpoint=None,
            sourcegraph_token=None,
        )

        # Create a test config that will be returned by our mocks
        config = QAAgentConfig()
        config.repo_path = sample_repo_path  # Set it directly to the fixture path
        config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_cli_tests")
        os.makedirs(config.output_directory, exist_ok=True)
        config.target_coverage = 90.0
        config.api_key = "test-api-key"

        # Mock everything using pytest-mock
        mock_load_config = mocker.patch("qa_agent.cli.load_config", return_value=config)
        mock_update_config = mocker.patch(
            "qa_agent.cli.update_config_from_args", return_value=config
        )
        mock_workflow_class = mocker.patch("qa_agent.cli.QAWorkflow")
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})

        # Mock workflow instance
        mock_workflow_instance = mock_workflow_class.return_value
        mock_workflow_instance.run.return_value = {
            "status": "Workflow finished",
            "success_count": 2,
            "failure_count": 0,
            "functions": ["function1", "function2"],
        }

        # Run CLI
        run_cli(args)

        # Assert config was loaded and updated
        mock_load_config.assert_called_once_with(config_file)
        mock_update_config.assert_called_once()

        # Assert workflow was created and run
        mock_workflow_class.assert_called_once()
        mock_workflow_instance.run.assert_called_once()

        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(config.output_directory):
            shutil.rmtree(config.output_directory)

    @pytest.mark.e2e
    def test_cli_with_sourcegraph_integration(self, mocker, sample_repo_path, disable_api_calls):
        """Test CLI with Sourcegraph integration enabled."""
        # Create arguments with Sourcegraph integration
        output_dir = os.path.join(tempfile.gettempdir(), "qa_agent_cli_sg_tests")
        os.makedirs(output_dir, exist_ok=True)

        args = argparse.Namespace(
            repo_path=sample_repo_path,
            config=None,
            verbose=True,
            output=output_dir,
            target_coverage=80.0,
            enable_sourcegraph=True,
            sourcegraph_endpoint="https://sourcegraph.example.com/.api",
            sourcegraph_token="test-sourcegraph-token",
        )

        # Mock the workflow run using pytest-mock
        mock_workflow_class = mocker.patch("qa_agent.cli.QAWorkflow")
        mocker.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-api-key", "SOURCEGRAPH_TOKEN": "test-sourcegraph-token"},
        )

        # Mock workflow instance
        mock_workflow_instance = mock_workflow_class.return_value
        mock_workflow_instance.run.return_value = {
            "status": "Workflow finished",
            "success_count": 2,
            "failure_count": 0,
            "functions": ["function1", "function2"],
        }

        # Run CLI
        run_cli(args)

        # Assert workflow was created and run
        mock_workflow_class.assert_called_once()
        mock_workflow_instance.run.assert_called_once()

        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main(["-v", "test_e2e_cli.py"])
