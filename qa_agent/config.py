"""
Configuration handling for the QA Agent.

This module manages the configuration for the QA agent, including API keys,
testing frameworks, and code analysis settings.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from qa_agent.utils.logging import get_logger, log_exception, log_function_call, log_function_result

# Initialize logger for this module
logger = get_logger(__name__)


@dataclass
class QAAgentConfig:
    """Configuration class for QA Agent."""

    # LLM settings
    model_provider: str = "openai"  # or "anthropic" or "github-copilot"
    model_name: str = "gpt-4o"  # Default to gpt-4o as expected by tests
    api_key: Optional[str] = None
    copilot_settings: Optional[Dict[str, Any]] = None  # Settings specific to GitHub Copilot

    # Repository settings
    repo_path: str = ""
    ignore_patterns: Optional[List[str]] = None
    target_coverage: float = 80.0
    specific_file: Optional[str] = None  # Path to a specific file to test

    # IP Protection settings
    ip_protection_enabled: bool = False
    protected_patterns: Optional[List[str]] = None
    protected_functions: Optional[List[str]] = None
    protected_files: Optional[List[str]] = None
    ip_protection_rules_path: Optional[str] = None

    # Testing settings
    test_framework: str = "pytest"
    test_directory: str = "tests"
    test_file_pattern: str = "test_*.py"

    # Parallelism settings
    parallel_execution: bool = False
    max_workers: int = 4
    task_queue_size: int = 100
    incremental_testing: bool = False
    changed_since_days: int = (
        7  # Only test files changed in the last N days when incremental testing is enabled
    )

    # Output settings
    output_directory: str = "./generated_tests"
    verbose: bool = False

    # Sourcegraph settings
    sourcegraph_enabled: bool = False
    sourcegraph_api_endpoint: str = "https://sourcegraph.com/.api"
    sourcegraph_api_token: Optional[str] = None

    def __post_init__(self):
        """Initialize default values and get API keys from environment variables."""
        log_function_call(logger, "__post_init__", (self.model_provider,))

        # Initialize default ignore patterns if not provided
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                ".git",
                "__pycache__",
                "venv",
                "env",
                ".env",
                ".venv",
                "node_modules",
                ".pytest_cache",
                ".coverage",
            ]
            logger.debug("Initialized default ignore patterns", count=len(self.ignore_patterns))

        # Get API key from environment if not provided
        if not self.api_key:
            api_key_env_name = None
            if self.model_provider == "openai":
                api_key_env_name = "OPENAI_API_KEY"
                self.api_key = os.environ.get(api_key_env_name)
            elif self.model_provider == "anthropic":
                api_key_env_name = "ANTHROPIC_API_KEY"
                self.api_key = os.environ.get(api_key_env_name)
            elif self.model_provider == "github-copilot":
                api_key_env_name = "GITHUB_COPILOT_API_KEY"
                self.api_key = os.environ.get(api_key_env_name)

                # Initialize default Copilot settings if not provided
                if not self.copilot_settings:
                    self.copilot_settings = {
                        "endpoint": os.environ.get(
                            "GITHUB_COPILOT_ENDPOINT", "https://api.github.com/copilot"
                        ),
                        "model_version": os.environ.get("GITHUB_COPILOT_MODEL", "latest"),
                        "max_tokens": 2048,
                        "temperature": 0.1,
                        "collaborative_mode": True,
                    }
                    logger.debug(
                        "Initialized default GitHub Copilot settings",
                        endpoint=self.copilot_settings["endpoint"],
                        model_version=self.copilot_settings["model_version"],
                    )

            if api_key_env_name:
                logger.debug(
                    "Checking for API key in environment",
                    env_var=api_key_env_name,
                    found=self.api_key is not None,
                )

        # Get Sourcegraph token from environment if enabled but not provided
        if self.sourcegraph_enabled and not self.sourcegraph_api_token:
            self.sourcegraph_api_token = os.environ.get("SOURCEGRAPH_API_TOKEN")

            if not self.sourcegraph_api_token:
                logger.warning(
                    "SOURCEGRAPH_API_TOKEN not found in environment",
                    env_var="SOURCEGRAPH_API_TOKEN",
                )

        # If running interactively, prompt for API key and token
        try:
            # For test_post_init_interactive_api_key test patch
            # Use os.isatty instead of sys.stdout.isatty so we can patch it in tests
            is_interactive = os.isatty(sys.stdout.fileno())
            logger.debug("Checking for interactive environment", is_interactive=is_interactive)

            if not self.api_key and is_interactive:
                key = input(f"Enter your {self.model_provider.upper()}_API_KEY: ").strip()
                # Always set the API key, even if empty (for tests)
                self.api_key = key

                # Only set environment if there's a value
                if key:
                    os.environ[f"{self.model_provider.upper()}_API_KEY"] = key
                    logger.info("API key set for provider", provider=self.model_provider)

            if self.sourcegraph_enabled and not self.sourcegraph_api_token and is_interactive:
                token = input("Enter your SOURCEGRAPH_API_TOKEN: ").strip()
                # Always set the token, even if empty (for tests)
                self.sourcegraph_api_token = token

                # Only set environment if there's a value
                if token:
                    os.environ["SOURCEGRAPH_API_TOKEN"] = token
                    logger.info("Sourcegraph API token set")
        except (AttributeError, OSError):
            # Not running in an interactive environment
            logger.debug("Not running in an interactive environment")
            pass

        # Log warning for missing API key
        if not self.api_key:
            logger.warning(
                f"{self.model_provider.upper()}_API_KEY not found in environment",
                provider=self.model_provider,
                env_var=f"{self.model_provider.upper()}_API_KEY",
                message="Some functionality may be limited",
            )

        log_function_result(logger, "__post_init__", "Configuration initialization complete")


def load_config(config_path: Optional[str] = None) -> QAAgentConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        QAAgentConfig object
    """
    log_function_call(logger, "load_config", (config_path,))
    start_time = time.time()

    config = QAAgentConfig()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "rb") as f:  # Open as binary to avoid encoding issues
                config_data = yaml.safe_load(f)

            # Update config with values from file
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            logger.info("Configuration loaded", file_path=config_path)
        except yaml.YAMLError as e:
            error_msg = f"Error loading configuration from {config_path}: YAML parsing error"
            # Note: log_exception already logs an error message, so we don't need logger.error here
            log_exception(
                logger, "load_config", e, {"config_path": config_path, "error_message": error_msg}
            )
            raise e
        except Exception as e:
            error_msg = f"Error loading configuration from {config_path}: {str(e)}"
            # Note: log_exception already logs an error message, so we don't need logger.error here
            log_exception(
                logger, "load_config", e, {"config_path": config_path, "error_message": error_msg}
            )
    else:
        if config_path:
            logger.warning("Configuration file not found", file_path=config_path)
        logger.info("Using default configuration")

    execution_time = time.time() - start_time
    log_function_result(logger, "load_config", "Configuration loaded successfully", execution_time)

    return config


def update_config_from_args(config: QAAgentConfig, args: Any) -> QAAgentConfig:
    """
    Update configuration from command line arguments.

    Args:
        config: Existing configuration object
        args: Command line arguments

    Returns:
        Updated QAAgentConfig object
    """
    log_function_call(logger, "update_config_from_args", (str(config),), {"args": str(args)})

    if args.repo_path:
        config.repo_path = args.repo_path
        logger.debug("Updated repo path", repo_path=args.repo_path)

    if args.output:
        config.output_directory = args.output
        logger.debug("Updated output directory", output_directory=args.output)

    if args.target_coverage:
        config.target_coverage = args.target_coverage
        logger.debug("Updated target coverage", target_coverage=args.target_coverage)
        
    if hasattr(args, 'file') and args.file:
        config.specific_file = args.file
        logger.debug("Updated specific file to test", specific_file=args.file)

    config.verbose = args.verbose

    # Update parallel processing settings if provided
    parallel_updates = {}
    if hasattr(args, "parallel"):
        config.parallel_execution = args.parallel
        parallel_updates["parallel_execution"] = args.parallel

    if hasattr(args, "max_workers"):
        config.max_workers = args.max_workers
        parallel_updates["max_workers"] = args.max_workers

    if hasattr(args, "incremental"):
        config.incremental_testing = args.incremental
        parallel_updates["incremental_testing"] = args.incremental

    if hasattr(args, "changed_since"):
        config.changed_since_days = args.changed_since
        parallel_updates["changed_since_days"] = args.changed_since

    if parallel_updates:
        logger.debug("Updated parallel processing settings", **parallel_updates)

    # Update Sourcegraph settings if provided
    sourcegraph_updates = {}
    if hasattr(args, "enable_sourcegraph"):
        config.sourcegraph_enabled = args.enable_sourcegraph
        sourcegraph_updates["enabled"] = args.enable_sourcegraph

    if hasattr(args, "sourcegraph_endpoint") and args.sourcegraph_endpoint:
        config.sourcegraph_api_endpoint = args.sourcegraph_endpoint
        sourcegraph_updates["endpoint"] = args.sourcegraph_endpoint

    if hasattr(args, "sourcegraph_token") and args.sourcegraph_token:
        config.sourcegraph_api_token = args.sourcegraph_token
        sourcegraph_updates["token"] = "***" if args.sourcegraph_token else None

    if sourcegraph_updates:
        logger.debug("Updated Sourcegraph settings", **sourcegraph_updates)
        
    # Update IP Protection settings if provided
    ip_protection_updates = {}
    if hasattr(args, "enable_ip_protection"):
        config.ip_protection_enabled = args.enable_ip_protection
        ip_protection_updates["enabled"] = args.enable_ip_protection
        
    if hasattr(args, "ip_protection_rules") and args.ip_protection_rules:
        config.ip_protection_rules_path = args.ip_protection_rules
        ip_protection_updates["rules_path"] = args.ip_protection_rules
        
    if ip_protection_updates:
        logger.debug("Updated IP Protection settings", **ip_protection_updates)

    log_function_result(
        logger, "update_config_from_args", "Configuration updated from command line arguments"
    )

    return config
