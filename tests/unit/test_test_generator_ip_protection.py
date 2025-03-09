"""
Tests for the TestGenerator's integration with IP protection.
"""

from typing import Any, Dict, List, Optional, Union, cast
from unittest.mock import MagicMock, patch

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.ip_protector import IPProtector
from qa_agent.models import CodeFile, FileType, Function, GeneratedTest
from qa_agent.test_generator import TestGenerator


@pytest.fixture
def mock_config() -> QAAgentConfig:
    """Create a mock configuration for testing."""
    return QAAgentConfig(
        model_provider="openai",
        api_key="test-api-key",
        model_name="gpt-4o",
        repo_path="/test/repo",
        ip_protection_enabled=True,
        protected_patterns=["api_key", "password"],
        protected_functions=["authenticate", "encrypt"],
        protected_files=["/test/repo/src/secrets.py"],
    )


@pytest.fixture
def mock_function() -> Function:
    """Create a mock function for testing."""
    return Function(
        name="process_data",
        code="def process_data(data, api_key):\n    return call_api(data, api_key)",
        file_path="/test/repo/src/module.py",
        start_line=10,
        end_line=12,
    )


@pytest.fixture
def mock_context_files() -> List[CodeFile]:
    """Create mock context files for testing."""
    return [
        CodeFile(
            path="/test/repo/src/module.py",
            content="def process_data(data, api_key):\n    return call_api(data, api_key)",
        ),
        CodeFile(
            path="/test/repo/src/secrets.py",
            content='API_KEY = "12345-secret"\nPASSWORD = "secure123"',
        ),
    ]


@pytest.fixture
def mock_llm_chain() -> MagicMock:
    """Create a mock RunnableSequence."""
    mock_chain = MagicMock()
    # Set up mock invoke method instead of run
    mock_chain.invoke.return_value = """
```python
def test_process_data():
    # Mock the call_api function
    with patch('module.call_api') as mock_call_api:
        mock_call_api.return_value = {'status': 'success'}
        
        # Call the function
        result = process_data('test data', 'test-api-key')
        
        # Verify the result
        assert result == {'status': 'success'}
        
        # Verify call_api was called with the right parameters
        mock_call_api.assert_called_once_with('test data', 'test-api-key')
```

imports:
- unittest.mock

mocks:
- call_api

fixtures:
- None
"""
    return mock_chain


def test_test_generator_initializes_ip_protector(mock_config: QAAgentConfig) -> None:
    """Test that TestGenerator initializes with an IPProtector."""
    # Enable IP protection
    mock_config.ip_protection_enabled = True

    with (
        patch("qa_agent.test_generator.ChatOpenAI"),
        patch("qa_agent.test_generator.RunnableSequence"),
        patch("qa_agent.test_generator.PromptTemplate"),
        patch("qa_agent.test_generator.IPProtector") as mock_ip_protector_class,
    ):

        # Mock the IPProtector instance
        mock_ip_protector = MagicMock()
        mock_ip_protector_class.return_value = mock_ip_protector

        # Initialize the test generator
        generator = TestGenerator(mock_config)

        # Verify IPProtector was initialized
        mock_ip_protector_class.assert_called_once_with(
            protected_patterns=mock_config.protected_patterns,
            protected_functions=mock_config.protected_functions,
            protected_files=mock_config.protected_files,
        )

        # If rules path is set, verify load_protection_rules was called
        if mock_config.ip_protection_rules_path:
            mock_ip_protector.load_protection_rules.assert_called_once_with(
                mock_config.ip_protection_rules_path
            )


def test_generate_test_uses_ip_protection(
    mock_llm_chain: MagicMock,
    mock_config: QAAgentConfig,
    mock_function: Function,
    mock_context_files: List[CodeFile],
) -> None:
    """Test that generate_test method uses the IP protector."""
    with (
        patch("qa_agent.test_generator.ChatOpenAI"),
        patch("qa_agent.test_generator.RunnableSequence", return_value=mock_llm_chain),
        patch("qa_agent.test_generator.PromptTemplate"),
        patch("qa_agent.test_generator.IPProtector") as mock_ip_protector_class,
    ):

        # Mock the IPProtector instance
        mock_ip_protector = MagicMock()
        mock_ip_protector_class.return_value = mock_ip_protector

        # Mock IP protection methods
        mock_ip_protector.redact_function.return_value = mock_function
        mock_ip_protector.sanitize_context_files.return_value = mock_context_files
        mock_ip_protector.protect.side_effect = lambda x: x + " [PROTECTED]"

        # Initialize the test generator
        generator = TestGenerator(mock_config)

        # Generate a test
        result = generator.generate_test(mock_function, mock_context_files)

        # Verify IP protection was used
        mock_ip_protector.redact_function.assert_called_once_with(mock_function)
        mock_ip_protector.sanitize_context_files.assert_called_once_with(mock_context_files)
        assert mock_ip_protector.protect.call_count >= 1  # At least one prompt was protected


def test_generate_test_protects_sensitive_data(
    mock_llm_chain: MagicMock, mock_config: QAAgentConfig, mock_function: Function
) -> None:
    """Test that sensitive data is protected when generating tests."""
    with (
        patch("qa_agent.test_generator.ChatOpenAI"),
        patch("qa_agent.test_generator.RunnableSequence", return_value=mock_llm_chain),
        patch("qa_agent.test_generator.PromptTemplate"),
        patch("qa_agent.test_generator.IPProtector") as mock_ip_protector_class,
    ):

        # Mock a function with sensitive data
        sensitive_function = Function(
            name="authenticate",
            code='def authenticate(username, password):\n    api_key = "12345-secret"\n    return check_auth(username, password, api_key)',
            file_path="/test/repo/src/auth.py",
            start_line=1,
            end_line=3,
        )

        # Mock the IPProtector to detect and redact sensitive data
        mock_ip_protector = MagicMock()
        mock_ip_protector_class.return_value = mock_ip_protector

        # Mock the redaction
        redacted_function = Function(
            name="authenticate",
            code='def authenticate(username, [REDACTED]):\n    api_key = "[REDACTED]"\n    return check_auth(username, [REDACTED], [REDACTED])',
            file_path="/test/repo/src/auth.py",
            start_line=1,
            end_line=3,
        )
        mock_ip_protector.redact_function.return_value = redacted_function
        mock_ip_protector.protect.side_effect = lambda x: x

        # Initialize the test generator
        generator = TestGenerator(mock_config)

        # Generate a test for the sensitive function
        result = generator.generate_test(sensitive_function)

        # Verify the function was redacted
        mock_ip_protector.redact_function.assert_called_once()

        # Verify LLM was called with the redacted function
        prompt_arg = mock_llm_chain.invoke.call_args[1].get("function_code", "")
        assert 'api_key = "12345-secret"' not in prompt_arg
        assert "[REDACTED]" in prompt_arg


def test_generate_test_with_protected_function(
    mock_llm_chain: MagicMock, mock_config: QAAgentConfig
) -> None:
    """Test generating a test for a protected function."""
    with (
        patch("qa_agent.test_generator.ChatOpenAI"),
        patch("qa_agent.test_generator.RunnableSequence", return_value=mock_llm_chain),
        patch("qa_agent.test_generator.PromptTemplate"),
        patch("qa_agent.test_generator.IPProtector") as mock_ip_protector_class,
    ):

        # Mock a protected function
        protected_function = Function(
            name="encrypt",  # This name is in protected_functions list
            code="def encrypt(data, key):\n    return encrypted_data",
            file_path="/test/repo/src/crypto.py",
            start_line=1,
            end_line=2,
        )

        # Mock the IPProtector to fully redact the protected function
        mock_ip_protector = MagicMock()
        mock_ip_protector_class.return_value = mock_ip_protector

        # Mock the complete redaction of function
        redacted_function = Function(
            name="encrypt",
            code="[FUNCTION REDACTED FOR IP PROTECTION]",
            file_path="/test/repo/src/crypto.py",
            start_line=1,
            end_line=2,
        )
        mock_ip_protector.redact_function.return_value = redacted_function
        mock_ip_protector.protect.side_effect = lambda x: x

        # Initialize the test generator
        generator = TestGenerator(mock_config)

        # Generate a test for the protected function
        result = generator.generate_test(protected_function)

        # Verify the function was fully redacted
        mock_ip_protector.redact_function.assert_called_once()

        # Verify LLM was called with the redacted function
        prompt_arg = mock_llm_chain.invoke.call_args[1].get("function_code", "")
        assert "[FUNCTION REDACTED FOR IP PROTECTION]" in prompt_arg


def test_generate_test_with_protected_context_files(
    mock_llm_chain: MagicMock, mock_config: QAAgentConfig, mock_function: Function
) -> None:
    """Test generating a test with protected context files."""
    with (
        patch("qa_agent.test_generator.ChatOpenAI"),
        patch("qa_agent.test_generator.RunnableSequence", return_value=mock_llm_chain),
        patch("qa_agent.test_generator.PromptTemplate"),
        patch("qa_agent.test_generator.IPProtector") as mock_ip_protector_class,
    ):

        # Create context files including a protected file
        context_files: List[CodeFile] = [
            CodeFile(path="/test/repo/src/module.py", content="def helper():\n    return True"),
            CodeFile(
                path="/test/repo/src/secrets.py",  # This path is in protected_files list
                content='API_KEY = "12345-secret"\nPASSWORD = "secure123"',
            ),
        ]

        # Mock the IPProtector
        mock_ip_protector = MagicMock()
        mock_ip_protector_class.return_value = mock_ip_protector

        # Mock sanitize_context_files to redact the protected file
        sanitized_files: List[CodeFile] = [
            CodeFile(path="/test/repo/src/module.py", content="def helper():\n    return True"),
            CodeFile(
                path="/test/repo/src/secrets.py", content="[ENTIRE FILE REDACTED FOR IP PROTECTION]"
            ),
        ]
        mock_ip_protector.sanitize_context_files.return_value = sanitized_files
        mock_ip_protector.redact_function.return_value = mock_function
        mock_ip_protector.protect.side_effect = lambda x: x

        # Initialize the test generator
        generator = TestGenerator(mock_config)

        # Generate a test with the context files
        result = generator.generate_test(mock_function, context_files)

        # Verify context files were sanitized
        mock_ip_protector.sanitize_context_files.assert_called_once_with(context_files)

        # Verify LLM was called with redacted context
        prompt_arg = mock_llm_chain.invoke.call_args[1].get("context", "")
        assert 'API_KEY = "12345-secret"' not in prompt_arg
        assert "[ENTIRE FILE REDACTED FOR IP PROTECTION]" in prompt_arg
