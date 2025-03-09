"""
Unit tests for the test generator module.

These tests verify the functionality of generating unit tests for functions.
"""

import os
from unittest.mock import MagicMock, mock_open

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function, GeneratedTest
from qa_agent.test_generator import TestGenerator


class TestTestGenerator:
    """Tests for the TestGenerator class."""

    @pytest.fixture
    def mock_runnable_sequence(self):
        """Mock RunnableSequence for testing."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = """
```python
import pytest
from unittest.mock import patch, MagicMock

def test_example_function():
    # Test implementation
    result = example_function(1, 2)
    assert result == 3
```

IMPORTS:
- pytest
- unittest.mock

MOCKS:
- mock_database

FIXTURES:
- fixture_data
"""
        return mock_chain

    def test_initialization_openai(self, mocker):
        """Test initialization with OpenAI provider."""
        # Mock dependencies
        mock_runnable_sequence = mocker.patch("qa_agent.test_generator.RunnableSequence")
        mock_chat_openai = mocker.patch("qa_agent.test_generator.ChatOpenAI")

        # Setup test
        config = QAAgentConfig(model_provider="openai", model_name="gpt-4o", api_key="test-api-key")

        # Create generator
        generator = TestGenerator(config)

        # Verify results
        assert generator.config == config
        assert generator.ip_protector is not None
        mock_chat_openai.assert_called_once_with(
            temperature=0, model=config.model_name, api_key=config.api_key
        )
        assert generator.copilot_adapter is None

    def test_initialization_anthropic(self, mocker):
        """Test initialization with Anthropic provider."""
        # Mock dependencies
        mock_runnable_sequence = mocker.patch("qa_agent.test_generator.RunnableSequence")
        mock_chat_anthropic = mocker.patch("qa_agent.test_generator.ChatAnthropic")

        # Setup test
        config = QAAgentConfig(
            model_provider="anthropic", model_name="claude-3", api_key="test-api-key"
        )

        # Create generator
        generator = TestGenerator(config)

        # Verify results
        assert generator.config == config
        mock_chat_anthropic.assert_called_once_with(
            temperature=0, model=config.model_name, api_key=config.api_key
        )
        assert generator.copilot_adapter is None

    def test_initialization_github_copilot(self, mocker):
        """Test initialization with GitHub Copilot provider."""
        # Mock dependencies
        mock_copilot_adapter = mocker.patch("qa_agent.test_generator.CopilotAdapter")

        # Setup test
        config = QAAgentConfig(
            model_provider="github-copilot",
            api_key="test-api-key",
            copilot_settings={
                "endpoint": "https://api.github.com/copilot",
                "model_version": "latest",
            },
        )

        # Create generator
        generator = TestGenerator(config)

        # Verify results
        assert generator.config == config
        assert generator.llm is None
        mock_copilot_adapter.assert_called_once_with(config)

    def test_initialization_unsupported_provider(self):
        """Test initialization with unsupported provider."""
        config = QAAgentConfig(model_provider="unsupported-provider", api_key="test-api-key")

        with pytest.raises(ValueError) as excinfo:
            TestGenerator(config)

        assert "Unsupported model provider" in str(excinfo.value)

    def test_create_test_generation_prompt(self, mock_config):
        """Test creating a test generation prompt."""
        generator = TestGenerator(mock_config)

        function = Function(
            name="example_function",
            code="def example_function(a, b):\n    return a + b",
            file_path="/test/repo/src/module.py",
            start_line=10,
            end_line=12,
        )

        prompt = generator._create_test_generation_prompt(function, "Context content")

        assert "function_name" in prompt.input_variables
        assert "function_code" in prompt.input_variables
        assert "file_path" in prompt.input_variables
        assert "context" in prompt.input_variables
        assert "test_framework" in prompt.input_variables

    def test_parse_test_response(self, mock_config):
        """Test parsing the LLM response."""
        generator = TestGenerator(mock_config)

        response = """
```python
import pytest
from unittest.mock import patch

def test_example_function():
    result = example_function(1, 2)
    assert result == 3
```

IMPORTS:
- pytest
- unittest.mock

MOCKS:
- mock_database

FIXTURES:
- test_data
"""

        test_code, imports, mocks, fixtures = generator._parse_test_response(response)

        assert "def test_example_function():" in test_code
        assert "result = example_function(1, 2)" in test_code
        assert "assert result == 3" in test_code

        assert "pytest" in imports
        assert "unittest.mock" in imports

        assert "mock_database" in mocks
        assert "test_data" in fixtures

    def test_get_test_file_path(self, mock_config):
        """Test generating a test file path."""
        mock_config.output_directory = "./test_output"
        generator = TestGenerator(mock_config)

        function = Function(
            name="example_function",
            code="def example_function(a, b):\n    return a + b",
            file_path="/test/repo/src/module.py",
            start_line=10,
            end_line=12,
        )

        test_file_path = generator._get_test_file_path(function)

        # Test file should be in output directory with test_ prefix and similar structure
        assert test_file_path.startswith("./test_output")
        assert test_file_path.endswith("test_module.py")

    def test_generate_test_with_llm(self, mocker, mock_function, mock_code_file):
        """Test generating a test with LLM."""
        # Mock dependencies
        mock_runnable_sequence = mocker.patch("qa_agent.test_generator.RunnableSequence")
        mock_str_output_parser = mocker.patch("qa_agent.test_generator.StrOutputParser")
        mock_chat_openai = mocker.patch("qa_agent.test_generator.ChatOpenAI")
        mock_parse = mocker.patch("qa_agent.test_generator.TestGenerator._parse_test_response")

        # Setup test configuration
        config = QAAgentConfig(
            model_provider="openai",
            model_name="gpt-4o",
            api_key="test-api-key",
            output_directory="./test_output",
        )

        # Setup mock chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Test response"
        mock_runnable_sequence.from_components.return_value = mock_chain

        # Setup mock parse response
        mock_parse.return_value = ("def test_function():\n    assert True", ["pytest"], [], [])

        # Create generator
        generator = TestGenerator(config)

        # Mock IP protector
        generator.ip_protector.redact_function = MagicMock(return_value=mock_function)
        generator.ip_protector.sanitize_context_files = MagicMock(return_value=[mock_code_file])
        generator.ip_protector.protect = MagicMock(return_value=mock_function.code)

        # Generate test
        result = generator.generate_test(function=mock_function, context_files=[mock_code_file])

        # Verify results
        assert isinstance(result, GeneratedTest)
        assert result.function == mock_function
        assert result.test_code == "def test_function():\n    assert True"
        assert result.imports == ["pytest"]
        assert result.mocks == []
        assert result.fixtures == []

        # Verify IP protection was used
        generator.ip_protector.redact_function.assert_called_once_with(mock_function)

        # Verify LLM was called with expected parameters
        mock_chain.invoke.assert_called_once()
        # Parameters are now passed as a dictionary to invoke
        params = mock_chain.invoke.call_args[0][0]
        assert params["function_name"] == mock_function.name
        # The function code should be passed to the LLM
        assert params["function_code"] is not None
        assert params["file_path"] == mock_function.file_path

    def test_generate_test_with_copilot(self, mocker, mock_function, mock_code_file):
        """Test generating a test with GitHub Copilot."""
        # Mock dependencies
        mock_copilot_adapter_class = mocker.patch("qa_agent.test_generator.CopilotAdapter")

        # Setup test configuration
        config = QAAgentConfig(
            model_provider="github-copilot",
            api_key="test-api-key",
            output_directory="./test_output",
            copilot_settings={
                "endpoint": "https://api.github.com/copilot",
                "model_version": "latest",
                "collaborative_mode": True,
            },
        )

        # Setup mock copilot adapter
        mock_adapter = MagicMock()
        mock_adapter.generate_test.return_value = (
            "def test_function():\n    assert True",
            ["pytest"],
            [],
            [],
        )
        mock_copilot_adapter_class.return_value = mock_adapter

        # Create generator
        generator = TestGenerator(config)

        # Generate test
        result = generator.generate_test(function=mock_function, context_files=[mock_code_file])

        # Verify results
        assert isinstance(result, GeneratedTest)
        assert result.function == mock_function
        assert result.test_code == "def test_function():\n    assert True"
        assert result.imports == ["pytest"]

        # Verify Copilot adapter was called
        mock_adapter.generate_test.assert_called_once_with(mock_function, [mock_code_file])

    def test_generate_test_with_copilot_collaborative(self, mocker, mock_function, mock_code_file):
        """Test generating a test collaboratively with GitHub Copilot."""
        # Mock dependencies
        mock_copilot_adapter_class = mocker.patch("qa_agent.test_generator.CopilotAdapter")

        # Setup test configuration
        config = QAAgentConfig(
            model_provider="github-copilot",
            api_key="test-api-key",
            output_directory="./test_output",
            copilot_settings={
                "endpoint": "https://api.github.com/copilot",
                "model_version": "latest",
                "collaborative_mode": True,
            },
        )

        # Setup mock copilot adapter
        mock_adapter = MagicMock()
        mock_adapter.collaborative_test_generation.return_value = (
            "def test_function():\n    assert True",
            ["pytest"],
            [],
            [],
        )
        mock_copilot_adapter_class.return_value = mock_adapter

        # Create generator
        generator = TestGenerator(config)

        # Generate test with feedback
        result = generator.generate_test(
            function=mock_function, context_files=[mock_code_file], feedback="Improve the assertion"
        )

        # Verify results
        assert isinstance(result, GeneratedTest)
        assert result.function == mock_function
        assert result.test_code == "def test_function():\n    assert True"
        assert result.imports == ["pytest"]

        # Verify Copilot adapter was called in collaborative mode
        mock_adapter.collaborative_test_generation.assert_called_once_with(
            mock_function, [mock_code_file], "Improve the assertion"
        )

    def test_save_test_to_file(self, mocker, mock_config, mock_generated_test):
        """Test saving a generated test to a file."""
        # Mock dependencies
        mock_makedirs = mocker.patch("os.makedirs")

        # Mock open to handle both reading unit test rules and writing the test file
        mock_open_file = mocker.patch("builtins.open", new_callable=mock_open)
        # For the unit_test_rules.md file load
        rules_content = "# Unit Test Rules\n- Write good tests"
        mock_open_file.return_value.__enter__.return_value.read.return_value = rules_content

        # Create generator
        generator = TestGenerator(mock_config)

        # Reset the mock to clear the call to load unit test rules
        mock_open_file.reset_mock()

        # Save test
        generator.save_test_to_file(mock_generated_test)

        # Verify directory was created
        mock_makedirs.assert_called_once()

        # Verify file was opened and written for the test file specifically
        mock_open_file.assert_called_once_with(mock_generated_test.test_file_path, "w")

        # Our implementation now makes two write calls (imports and test code)
        assert mock_open_file().write.call_count == 2

        # Get all written content by combining the call arguments
        imports_content = mock_open_file().write.call_args_list[0][0][0]
        test_content = mock_open_file().write.call_args_list[1][0][0]

        # Check content of written file
        assert "import pytest" in imports_content
        assert "import unittest.mock" in imports_content
        assert mock_generated_test.test_code == test_content
