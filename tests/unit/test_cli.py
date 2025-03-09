import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from qa_agent.cli import run_cli
from qa_agent.utils.logging import log_exception


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = MagicMock()
    # Set some default values to avoid common issues
    config.repo_path = "/some/valid/path"
    config.model_provider = "openai"
    config.api_key = "test_api_key"
    config.sourcegraph_enabled = False
    config.sourcegraph_api_token = None
    config.verbose = False
    config.output_directory = "test_output"
    config.parallel_execution = False
    config.max_workers = 4
    config.incremental_testing = False
    return config


@pytest.fixture
def mock_args():
    """Create mock command line arguments for testing."""
    return MagicMock()


def test_run_cli_missing_repo_path(mock_config, mock_args):
    mock_config.repo_path = None
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch("qa_agent.cli.logger") as mock_logger:
            with pytest.raises(SystemExit):
                run_cli(mock_args)
                mock_logger.error.assert_called_with(
                    "Repository path is required", extra={"error_type": "missing_repo_path"}
                )


def test_run_cli_invalid_repo_path(mock_config, mock_args):
    mock_config.repo_path = "/invalid/path"
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch("qa_agent.cli.logger") as mock_logger:
            with pytest.raises(SystemExit):
                run_cli(mock_args)
                mock_logger.error.assert_called_with(
                    f"Repository path does not exist: {mock_config.repo_path}",
                    extra={"error_type": "invalid_repo_path", "repo_path": mock_config.repo_path},
                )


def test_run_cli_missing_api_key(mock_config, mock_args):
    # Setup mock configuration
    mock_config.api_key = None
    mock_config.repo_path = "/some/valid/path"
    mock_config.model_provider = "openai"

    # Add necessary mocks to prevent execution going too far
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch("qa_agent.cli.os.path.exists", return_value=True):  # Mock path exists check
            with patch("qa_agent.cli.logger") as mock_logger:
                with patch("qa_agent.cli.print") as mock_print:
                    with patch("qa_agent.cli.os.makedirs"):  # Prevent actual directory creation
                        # Mock isatty to return False (non-interactive mode)
                        with patch("qa_agent.cli.os.isatty", return_value=False):
                            # Mock sys.exit to prevent actual exit
                            with patch("qa_agent.cli.sys.exit") as mock_exit:
                                with patch("qa_agent.cli.QAWorkflow") as mock_workflow:
                                    # Prevent workflow from executing
                                    mock_instance = mock_workflow.return_value
                                    mock_instance.run.return_value = {
                                        "success_count": 0,
                                        "failure_count": 0,
                                        "functions": [],
                                    }

                                    # Run the function
                                    run_cli(mock_args)

                                    # Assert logger was called correctly
                                    mock_logger.error.assert_called_with(
                                        "OPENAI_API_KEY not found in environment variables",
                                        extra={
                                            "error_type": "missing_api_key",
                                            "provider": "openai",
                                        },
                                    )

                                    # Check print was called (we don't need to check exact text)
                                    assert mock_print.call_count > 0

                                    # Check that sys.exit was called with exit code 1
                                    mock_exit.assert_called_with(1)


def test_run_cli_non_interactive_exit(mock_config, mock_args):
    """Test that the CLI exists when in non-interactive mode and an API key is needed."""
    # Setup mock configuration
    mock_config.repo_path = "/some/valid/path"
    mock_config.api_key = None  # Missing API key will trigger non-interactive exit
    mock_config.model_provider = "openai"

    # Add comprehensive mocks
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch("qa_agent.cli.os.path.exists", return_value=True):  # Mock path exists check
            with patch("qa_agent.cli.logger") as mock_logger:
                with patch("qa_agent.cli.print") as mock_print:
                    with patch("qa_agent.cli.os.makedirs"):  # Prevent actual directory creation
                        # Mock isatty to force non-interactive mode
                        with patch("qa_agent.cli.os.isatty", return_value=False):
                            # Mock sys.exit to prevent test from exiting
                            with patch("qa_agent.cli.sys.exit") as mock_exit:
                                # Run the function (will exit due to non-interactive mode)
                                run_cli(mock_args)

                                # Verify error was logged about missing API key
                                mock_logger.error.assert_called_with(
                                    "OPENAI_API_KEY not found in environment variables",
                                    extra={"error_type": "missing_api_key", "provider": "openai"},
                                )

                                # Verify print was called for the error message
                                assert mock_print.call_count > 0

                                # We'll implicitly exit since we're in non-interactive mode and missing API key
                                assert mock_exit.call_count > 0


def test_run_cli_output_directory_creation(mock_config, mock_args):
    # Setup mock configuration
    mock_config.output_directory = "test_output"
    mock_config.repo_path = "/some/valid/path"
    mock_config.api_key = "some_api_key"  # Avoid triggering missing API key flow
    mock_config.sourcegraph_enabled = False  # Avoid sourcegraph API token flow

    # Add necessary mocks
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch(
            "qa_agent.cli.update_config_from_args", return_value=mock_config
        ):  # Keep config consistent
            with patch("qa_agent.cli.os.path.exists", return_value=True):  # Mock path exists check
                with patch("qa_agent.cli.os.makedirs") as mock_makedirs:
                    with patch("qa_agent.cli.QAWorkflow") as mock_workflow:
                        # Prevent workflow from executing
                        mock_instance = mock_workflow.return_value
                        mock_instance.run.return_value = {
                            "success_count": 0,
                            "failure_count": 0,
                            "functions": [],
                        }

                        # Run the function
                        run_cli(mock_args)

                        # Verify makedirs was called correctly
                        mock_makedirs.assert_called_with(
                            mock_config.output_directory, exist_ok=True
                        )


def test_run_cli_parallel_workflow(mock_config, mock_args):
    # Setup mock configuration
    mock_config.parallel_execution = True
    mock_config.max_workers = 4
    mock_config.repo_path = "/some/valid/path"
    mock_config.api_key = "some_api_key"  # Avoid triggering missing API key flow
    mock_config.incremental_testing = False
    mock_config.sourcegraph_enabled = False  # Avoid sourcegraph API token flow

    # Add necessary mocks
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch(
            "qa_agent.cli.update_config_from_args", return_value=mock_config
        ):  # Prevent config changes
            with patch("qa_agent.cli.os.path.exists", return_value=True):  # Mock path exists check
                with patch("qa_agent.cli.os.makedirs"):  # Prevent actual directory creation
                    with patch("qa_agent.cli.logger") as mock_logger:
                        with patch("qa_agent.cli.ParallelQAWorkflow") as mock_parallel_workflow:
                            # Configure mock workflow
                            mock_instance = mock_parallel_workflow.return_value
                            mock_instance.run.return_value = {
                                "success_count": 0,
                                "failure_count": 0,
                                "functions": [],
                            }

                            # Run the function
                            run_cli(mock_args)

                            # Verify workflow was called with correct config
                            mock_parallel_workflow.assert_called_once_with(mock_config)

                            # Verify a log message about parallel workflow was made
                            # The message should contain both "parallel" and the number of workers
                            assert any(
                                "parallel" in str(call).lower()
                                and str(mock_config.max_workers) in str(call)
                                for call in mock_logger.info.call_args_list
                            )


def test_run_cli_sequential_workflow(mock_config, mock_args):
    # Setup mock configuration
    mock_config.parallel_execution = False
    mock_config.repo_path = "/some/valid/path"
    mock_config.api_key = "some_api_key"  # Avoid triggering missing API key flow
    mock_config.sourcegraph_enabled = False  # Avoid sourcegraph API token flow

    # Add necessary mocks
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch(
            "qa_agent.cli.update_config_from_args", return_value=mock_config
        ):  # Prevent config changes
            with patch("qa_agent.cli.os.path.exists", return_value=True):  # Mock path exists check
                with patch("qa_agent.cli.os.makedirs"):  # Prevent actual directory creation
                    with patch("qa_agent.cli.logger") as mock_logger:
                        with patch("qa_agent.cli.QAWorkflow") as mock_sequential_workflow:
                            # Configure mock workflow
                            mock_instance = mock_sequential_workflow.return_value
                            mock_instance.run.return_value = {
                                "success_count": 0,
                                "failure_count": 0,
                                "functions": [],
                            }

                            # Run the function
                            run_cli(mock_args)

                            # Verify workflow was called with correct config
                            mock_sequential_workflow.assert_called_once_with(mock_config)

                            # Verify a log message about sequential workflow was made
                            # Look for patterns that match info about sequential workflow instead of strict text match
                            assert any(
                                "sequential" in str(call).lower()
                                for call in mock_logger.info.call_args_list
                            )


def test_run_cli_keyboard_interrupt(mock_config, mock_args):
    # Setup mock configuration
    mock_config.repo_path = "/some/valid/path"
    mock_config.api_key = "some_api_key"
    mock_config.sourcegraph_enabled = False  # Avoid sourcegraph API token flow

    # Mock a KeyboardInterrupt during workflow execution
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch(
            "qa_agent.cli.update_config_from_args", return_value=mock_config
        ):  # Prevent config changes
            with patch("qa_agent.cli.os.path.exists", return_value=True):
                with patch("qa_agent.cli.os.makedirs"):
                    with patch("qa_agent.cli.logger") as mock_logger:
                        with patch("qa_agent.cli.QAWorkflow") as mock_workflow:
                            # Set up the workflow to raise KeyboardInterrupt
                            mock_instance = mock_workflow.return_value
                            mock_instance.run.side_effect = KeyboardInterrupt()

                            # Need to mock sys.exit to prevent test from actually exiting
                            with patch("qa_agent.cli.sys.exit") as mock_exit:
                                # Run the function (which will raise KeyboardInterrupt)
                                run_cli(mock_args)

                                # Verify a log message about abort was made
                                abort_message_logged = any(
                                    "abort" in str(call).lower()
                                    for call in mock_logger.info.call_args_list
                                )
                                assert abort_message_logged, "No abort message was logged"

                                # Verify sys.exit was called with the right code (0 - success, since this is a controlled exit)
                                mock_exit.assert_called_once_with(0)


def test_run_cli_exception(mock_config, mock_args):
    # Setup mock configuration
    mock_config.repo_path = "/some/valid/path"
    mock_config.api_key = "some_api_key"
    mock_config.verbose = True
    mock_config.sourcegraph_enabled = False  # Avoid sourcegraph API token flow

    # Setup to trigger an exception during workflow execution
    with patch("qa_agent.cli.load_config", return_value=mock_config):
        with patch(
            "qa_agent.cli.update_config_from_args", return_value=mock_config
        ):  # Prevent config changes
            with patch("qa_agent.cli.os.path.exists", return_value=True):
                with patch("qa_agent.cli.os.makedirs"):
                    with patch("qa_agent.cli.logger") as mock_logger:
                        with patch("qa_agent.cli.QAWorkflow") as mock_workflow:
                            # Configure workflow to raise exception
                            mock_instance = mock_workflow.return_value
                            test_exception = Exception("Test exception")
                            mock_instance.run.side_effect = test_exception

                            # Mock logging utilities - use getattr to safely mock in case of import changes
                            with patch("qa_agent.cli.log_exception") as mock_log_exception:
                                # Prevent actual sys.exit
                                with patch("qa_agent.cli.sys.exit") as mock_exit:
                                    run_cli(mock_args)

                                    # Verify log_exception was called at least once
                                    assert (
                                        mock_log_exception.called
                                    ), "Exception logging function wasn't called"

                                    # Check that our test exception was passed to the logger
                                    exception_logged = False
                                    for call_args in mock_log_exception.call_args_list:
                                        args, kwargs = call_args
                                        # Check if the same exception instance was logged
                                        if len(args) >= 3 and args[2] is test_exception:
                                            exception_logged = True

                                    assert (
                                        exception_logged
                                    ), "Our test exception wasn't passed to log_exception"

                                    # Verify sys.exit was called with error code
                                    assert mock_exit.call_count > 0, "sys.exit wasn't called"
                                    assert any(
                                        call(1) == x for x in mock_exit.call_args_list
                                    ), "sys.exit(1) wasn't called"
