"""
Command-line interface for the QA Agent.
"""

import argparse
import logging
import os
import sys
from typing import Any

from qa_agent.config import QAAgentConfig, load_config, update_config_from_args
from qa_agent.parallel_workflow import ParallelQAWorkflow
from qa_agent.utils.logging import configure_logging, get_logger, log_exception
from qa_agent.workflows import QAWorkflow

# Initialize logger for this module
logger = get_logger(__name__)


def run_cli(args: Any) -> None:
    """
    Run the QA agent from the command line.

    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)

    # Update configuration from arguments
    config = update_config_from_args(config, args)

    # Check for specific file testing via environment variable
    specific_file = os.environ.get("QA_AGENT_TEST_FILE")
    if specific_file:
        logger.info(f"Testing specific file: {specific_file}")
        # Add the specific file path to the config for use in the workflow
        config.specific_file = specific_file

    # Validate configuration
    if not config.repo_path:
        logger.error("Repository path is required", extra={"error_type": "missing_repo_path"})
        sys.exit(1)

    if not os.path.exists(config.repo_path):
        logger.error(
            f"Repository path does not exist: {config.repo_path}",
            extra={"error_type": "invalid_repo_path", "repo_path": config.repo_path},
        )
        sys.exit(1)

    if not config.api_key:
        logger.error(
            f"{config.model_provider.upper()}_API_KEY not found in environment variables",
            extra={"error_type": "missing_api_key", "provider": config.model_provider},
        )
        print(f"\nError: {config.model_provider.upper()}_API_KEY not found.")
        print(f"Please set the {config.model_provider.upper()}_API_KEY environment variable.")
        print(
            f"You can do this by running: export {config.model_provider.upper()}_API_KEY=your_api_key_here"
        )
        print(f"Or you can provide it when prompted during execution.")

    # Check if Sourcegraph integration is enabled but no token is provided
    if config.sourcegraph_enabled and not config.sourcegraph_api_token:
        logger.warning(
            "Sourcegraph integration is enabled but no API token is provided",
            extra={"integration": "sourcegraph"},
        )
        print("\nWarning: Sourcegraph integration is enabled but no API token is provided.")
        print("Some Sourcegraph features may not work without an API token.")
        print(
            "You can provide a token using --sourcegraph-token or the SOURCEGRAPH_API_TOKEN environment variable."
        )

        # Prompt for API key if in interactive mode
        try:
            if os.isatty(0):  # Check if running in interactive terminal
                key = input(f"Enter your {config.model_provider.upper()} API Key: ").strip()
                if key:
                    os.environ[f"{config.model_provider.upper()}_API_KEY"] = key
                    config.api_key = key
                    print(f"API key set for this session.")
                else:
                    sys.exit(1)
            else:
                sys.exit(1)
        except (AttributeError, OSError):
            sys.exit(1)

    # Create output directory if it doesn't exist
    if config.output_directory:
        os.makedirs(config.output_directory, exist_ok=True)

    # Display configuration
    logger.info(f"Repository path: {config.repo_path}")
    logger.info(f"Test framework: {config.test_framework}")
    logger.info(f"Output directory: {config.output_directory}")
    logger.info(f"Target coverage: {config.target_coverage}%")
    if hasattr(config, "specific_file") and config.specific_file:
        logger.info(f"Testing specific file: {config.specific_file}")

    # Run the QA workflow
    try:
        # Use parallel workflow if parallel execution is enabled
        if config.parallel_execution:
            logger.info(f"Using parallel workflow with {config.max_workers} workers")
            if config.incremental_testing:
                logger.info(
                    f"Incremental testing enabled (changed files in last {config.changed_since_days} days)"
                )
            workflow = ParallelQAWorkflow(config)
        else:
            logger.info("Using sequential workflow")
            workflow = QAWorkflow(config)

        final_state = workflow.run()

        # Print final summary
        success_count = final_state.get("success_count", 0)
        failure_count = final_state.get("failure_count", 0)
        total_count = final_state.get("functions", [])

        if total_count:
            total_count = len(total_count)
        else:
            total_count = 0

        print("\n" + "=" * 50)
        print("QA Agent Summary")
        print("=" * 50)
        print(f"Total functions processed: {total_count}")
        print(f"Tests generated successfully: {success_count}")
        print(f"Tests failed: {failure_count}")
        print(f"Success rate: {success_count / total_count * 100 if total_count else 0:.2f}%")
        print(f"Output directory: {config.output_directory}")

        # Print parallel execution stats if available
        if config.parallel_execution and "stats" in final_state:
            stats = final_state.get("stats", {})
            execution_time = stats.get("execution_time", 0)
            avg_time = stats.get("average_time_per_function", 0)
            print("\nParallel Execution Stats:")
            print(f"Total execution time: {execution_time:.2f}s")
            print(f"Average time per function: {avg_time:.2f}s")

        print("=" * 50)

    except KeyboardInterrupt:
        logger.info("Operation aborted by user")
        sys.exit(0)
    except Exception as e:
        # Log the error with context using log_exception instead of logger.error/exception
        log_exception(logger, "run_cli", e, {"verbose": config.verbose})
        sys.exit(1)


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(description="QA Agent for test generation and validation")
    parser.add_argument("--repo-path", "-r", type=str, help="Path to the repository")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for generated tests",
        default="./generated_tests",
    )
    parser.add_argument(
        "--target-coverage", "-t", type=float, help="Target coverage percentage", default=80.0
    )
    parser.add_argument("--file", "-f", type=str, help="Specific file to test", default=None)

    # Parallel processing arguments
    parallel_group = parser.add_argument_group("Parallel Processing")
    parallel_group.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Enable parallel test generation and validation",
    )
    parallel_group.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=4,
        help="Maximum number of worker processes/threads for parallel execution",
    )
    parallel_group.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        help="Only test files that have changed recently",
    )
    parallel_group.add_argument(
        "--changed-since",
        type=int,
        default=7,
        help="Only test files changed in the last N days (used with --incremental)",
    )

    # Sourcegraph integration arguments
    sourcegraph_group = parser.add_argument_group("Sourcegraph Integration")
    sourcegraph_group.add_argument(
        "--enable-sourcegraph",
        action="store_true",
        help="Enable Sourcegraph integration for enhanced context gathering",
    )
    sourcegraph_group.add_argument(
        "--sourcegraph-endpoint",
        type=str,
        help="Sourcegraph API endpoint (default: https://sourcegraph.com/.api)",
    )
    sourcegraph_group.add_argument(
        "--sourcegraph-token",
        type=str,
        help="Sourcegraph API token (can also use SOURCEGRAPH_API_TOKEN env var)",
    )

    # IP Protection arguments
    ip_protection_group = parser.add_argument_group("IP Protection")
    ip_protection_group.add_argument(
        "--enable-ip-protection",
        action="store_true",
        help="Enable IP protection when sending code to LLM providers",
    )
    ip_protection_group.add_argument(
        "--ip-protection-rules",
        type=str,
        help="Path to IP protection rules JSON file",
    )

    args = parser.parse_args()

    # Configure logging using utils.logging
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level)

    try:
        # Testing a specific file
        if args.file:
            print(f"Testing specific file: {args.file}")
            # Pass the file path as a module-level variable
            os.environ["QA_AGENT_TEST_FILE"] = args.file

        # Check for GitPython if incremental testing is enabled
        if getattr(args, "incremental", False):
            git_available = False
            try:
                import importlib.util

                git_spec = importlib.util.find_spec("git")
                git_available = git_spec is not None

                if git_available:
                    print("GitPython is available for incremental testing")
                else:
                    print(
                        "Warning: GitPython is not installed but is required for incremental testing"
                    )
                    print("Run 'pip install gitpython' to enable this feature")
            except Exception as e:
                print(f"Error checking for GitPython: {str(e)}")
                print("If incremental testing fails, install GitPython: pip install gitpython")

        run_cli(args)
        return 0  # Success
    except KeyboardInterrupt:
        print("\nOperation aborted by user.")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.exception("Error in CLI main function", exc_info=e)
        return 1  # Error


if __name__ == "__main__":
    sys.exit(main())
