"""
Unit tests for the config module.

These tests verify the functionality of loading, validating, and updating
configuration for the QA agent.
"""

import logging
import os
import sys
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from qa_agent.config import QAAgentConfig, load_config, update_config_from_args


class TestQAAgentConfig:
    """Tests for the QAAgentConfig class."""

    def test_default_initialization(self, monkeypatch):
        """Test initialization with default values."""
        # Clear any environment variables that might affect the test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GITHUB_COPILOT_API_KEY", raising=False)
        monkeypatch.delenv("SOURCEGRAPH_API_TOKEN", raising=False)

        # Patch isatty to avoid interactive prompts
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

        config = QAAgentConfig()

        # Test default values
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-4o"
        assert config.api_key is None
        assert config.repo_path == ""
        assert config.target_coverage == 80.0
        assert config.test_framework == "pytest"
        assert config.ip_protection_enabled is False
        assert config.sourcegraph_enabled is False

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = QAAgentConfig(
            model_provider="anthropic",
            model_name="claude-3",
            api_key="test-api-key",
            repo_path="/test/repo",
            target_coverage=90.0,
            ip_protection_enabled=True,
        )

        assert config.model_provider == "anthropic"
        assert config.model_name == "claude-3"
        assert config.api_key == "test-api-key"
        assert config.repo_path == "/test/repo"
        assert config.target_coverage == 90.0
        assert config.ip_protection_enabled is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=True)
    def test_post_init_api_key_from_env_openai(self):
        """Test retrieving OpenAI API key from environment variables."""
        config = QAAgentConfig(model_provider="openai")
        assert config.api_key == "env-openai-key"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-anthropic-key"}, clear=True)
    def test_post_init_api_key_from_env_anthropic(self):
        """Test retrieving Anthropic API key from environment variables."""
        config = QAAgentConfig(model_provider="anthropic")
        assert config.api_key == "env-anthropic-key"

    @patch.dict(os.environ, {"GITHUB_COPILOT_API_KEY": "env-copilot-key"}, clear=True)
    def test_post_init_api_key_from_env_copilot(self):
        """Test retrieving GitHub Copilot API key from environment variables."""
        config = QAAgentConfig(model_provider="github-copilot")
        assert config.api_key == "env-copilot-key"

        # Test default Copilot settings
        assert config.copilot_settings is not None
        assert config.copilot_settings["endpoint"] == "https://api.github.com/copilot"
        assert config.copilot_settings["temperature"] == 0.1

    @patch.dict(os.environ, {"SOURCEGRAPH_API_TOKEN": "env-sourcegraph-token"}, clear=True)
    def test_post_init_sourcegraph_token_from_env(self):
        """Test retrieving Sourcegraph API token from environment variables."""
        config = QAAgentConfig(sourcegraph_enabled=True)
        assert config.sourcegraph_api_token == "env-sourcegraph-token"

    @patch("os.isatty", return_value=True)
    @patch("builtins.input", return_value="interactive-api-key")
    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True)
    def test_post_init_interactive_api_key(self, mock_input, mock_isatty):
        """Test interactive API key input."""
        config = QAAgentConfig()
        assert config.api_key == "interactive-api-key"
        assert mock_input.call_count == 1

    @patch("os.isatty", return_value=True)
    @patch("builtins.input", return_value="interactive-sg-token")
    def test_post_init_interactive_sourcegraph_token(self, mock_input, mock_isatty):
        """Test interactive Sourcegraph token input."""
        config = QAAgentConfig(sourcegraph_enabled=True)
        assert config.sourcegraph_api_token == "interactive-sg-token"
        assert mock_input.call_count == 1

    def test_post_init_default_ignore_patterns(self):
        """Test default ignore patterns are set correctly."""
        config = QAAgentConfig()
        assert config.ignore_patterns is not None
        assert ".git" in config.ignore_patterns
        assert "__pycache__" in config.ignore_patterns
        assert ".pytest_cache" in config.ignore_patterns

    @patch.dict("os.environ", {}, clear=True)
    def test_post_init_warns_on_missing_api_key(self, caplog):
        """Test warning is logged when API key is missing."""
        with caplog.at_level(logging.WARNING):
            config = QAAgentConfig()
            # Force __post_init__ to run with empty environment
            config.__post_init__()
            assert "API_KEY not found" in caplog.text

    @patch.dict("os.environ", {}, clear=True)
    def test_post_init_warns_on_missing_sourcegraph_token(self, caplog):
        """Test warning is logged when Sourcegraph token is missing."""
        with caplog.at_level(logging.WARNING):
            config = QAAgentConfig(sourcegraph_enabled=True)
            # Force __post_init__ to run with empty environment
            config.__post_init__()
            assert "SOURCEGRAPH_API_TOKEN not found" in caplog.text


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_default(self):
        """Test loading config with no file path."""
        config = load_config()
        assert isinstance(config, QAAgentConfig)

    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        with patch("logging.Logger.warning") as mock_warning:
            config = load_config("nonexistent_file.yaml")
            assert isinstance(config, QAAgentConfig)
            mock_warning.assert_called_once()
            assert "not found" in mock_warning.call_args[0][0]

    def test_load_config_from_file(self):
        """Test loading config from an existing YAML file."""
        config_data = {
            "model_provider": "anthropic",
            "model_name": "claude-3",
            "repo_path": "/test/repo",
            "target_coverage": 90.0,
        }

        # Create a temporary file with config data
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_file.write(yaml.dump(config_data).encode("utf-8"))
            temp_file.flush()

        try:
            # Load config from the temp file
            config = load_config(temp_file.name)

            # Verify settings were loaded correctly
            assert config.model_provider == "anthropic"
            assert config.model_name == "claude-3"
            assert config.repo_path == "/test/repo"
            assert config.target_coverage == 90.0
        finally:
            # Clean up
            os.unlink(temp_file.name)

    def test_load_config_with_invalid_yaml(self):
        """Test loading config with invalid YAML file."""
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"invalid: yaml: file:")):
                with patch("yaml.safe_load", side_effect=yaml.YAMLError("YAML parsing error")):
                    # log_exception calls logger.error, so we need to patch it
                    with patch("qa_agent.config.log_exception") as mock_log_exception:
                        with pytest.raises(yaml.YAMLError):
                            config = load_config("config.yaml")

                        # Verify log_exception was called
                        mock_log_exception.assert_called_once()

                        # The call should have these args: (logger, function_name, exception, context)
                        args, kwargs = mock_log_exception.call_args
                        assert len(args) >= 3  # logger, function_name, exception

                        # The 4th argument is the context dictionary
                        # It might be passed as a positional or keyword argument
                        if len(args) > 3:
                            context = args[3]
                        else:
                            context = kwargs.get("context", {})

                        # Check the content of context
                        assert isinstance(context, dict)
                        assert "config_path" in context
                        assert "error_message" in context
                        assert "Error loading configuration" in context["error_message"]


class TestUpdateConfigFromArgs:
    """Tests for the update_config_from_args function."""

    def test_update_config_from_args(self):
        """Test updating config from command line arguments."""
        config = QAAgentConfig()
        args = MagicMock()
        args.repo_path = "/new/repo/path"
        args.output = "/new/output/path"
        args.target_coverage = 95.0
        args.verbose = True
        args.enable_sourcegraph = True
        args.sourcegraph_endpoint = "https://sg.example.com"
        args.sourcegraph_token = "new-sg-token"

        updated_config = update_config_from_args(config, args)

        assert updated_config.repo_path == "/new/repo/path"
        assert updated_config.output_directory == "/new/output/path"
        assert updated_config.target_coverage == 95.0
        assert updated_config.verbose is True
        assert updated_config.sourcegraph_enabled is True
        assert updated_config.sourcegraph_api_endpoint == "https://sg.example.com"
        assert updated_config.sourcegraph_api_token == "new-sg-token"

    def test_update_config_partial_args(self):
        """Test updating config with only some arguments."""
        config = QAAgentConfig(
            repo_path="/old/repo/path",
            output_directory="/old/output/path",
            target_coverage=80.0,
            verbose=False,
        )

        # Only update repo_path and verbose
        args = MagicMock()
        args.repo_path = "/new/repo/path"
        args.output = None
        args.target_coverage = None
        args.verbose = True

        # No sourcegraph attributes
        del args.enable_sourcegraph
        del args.sourcegraph_endpoint
        del args.sourcegraph_token

        updated_config = update_config_from_args(config, args)

        assert updated_config.repo_path == "/new/repo/path"
        assert updated_config.output_directory == "/old/output/path"  # Unchanged
        assert updated_config.target_coverage == 80.0  # Unchanged
        assert updated_config.verbose is True
