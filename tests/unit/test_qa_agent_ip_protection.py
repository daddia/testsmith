"""
Tests for verifying the QA Agent uses the IP protector effectively.
"""

from unittest.mock import MagicMock

# Note: pytest import might be handled by pytest fixtures
try:
    import pytest
except ImportError:
    # For type checking and LSP purposes
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            return lambda f: f

    pytest = _MockPytest()

from qa_agent.agents import TestGenerationAgent
from qa_agent.config import QAAgentConfig
from qa_agent.ip_protector import IPProtector
from qa_agent.models import CodeFile, FileType, Function


@pytest.fixture
def mock_config():
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
def mock_function():
    """Create a mock function for testing."""
    return Function(
        name="process_data",
        code="def process_data(data, api_key):\n    return call_api(data, api_key)",
        file_path="/test/repo/src/module.py",
        start_line=10,
        end_line=12,
        parameters=[{"name": "data", "type": "Any"}, {"name": "api_key", "type": "str"}],
    )


@pytest.fixture
def mock_context_files():
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
def mock_test_generator(mocker, mock_config):
    """Create a mock TestGenerationAgent."""
    # Patch the RunnableSequence and PromptTemplate to avoid actual LLM calls
    mocker.patch("qa_agent.test_generator.RunnableSequence")
    mocker.patch("qa_agent.test_generator.PromptTemplate")
    mocker.patch("qa_agent.test_generator.ChatOpenAI")

    # Initialize a real TestGenerationAgent with mock config
    agent = TestGenerationAgent(mock_config)

    # Add an IPProtector attribute to the agent for testing
    agent.ip_protector = IPProtector(
        protected_patterns=mock_config.protected_patterns,
        protected_functions=mock_config.protected_functions,
        protected_files=mock_config.protected_files,
    )

    return agent


@pytest.fixture
def mock_ip_protector():
    """Create a mock IPProtector."""
    return MagicMock(spec=IPProtector)


def test_test_generation_agent_uses_ip_protection(
    mocker, mock_test_generator, mock_config, mock_function, mock_context_files
):
    """Test that TestGenerationAgent uses IP protection when generating tests."""
    # Mock IP protector's methods
    mock_redact_function = mocker.patch.object(
        mock_test_generator.ip_protector, "redact_function", return_value=mock_function
    )
    mock_sanitize = mocker.patch.object(
        mock_test_generator.ip_protector, "sanitize_context_files", return_value=mock_context_files
    )
    mock_protect = mocker.patch.object(
        mock_test_generator.ip_protector,
        "protect",
        side_effect=lambda x: x + " [PROTECTED]" if isinstance(x, str) else x,
    )

    # Create a properly formed test
    generated_test = MagicMock(
        function=mock_function,
        test_code="def test_process_data():\n    assert True",
        imports=[],
        test_functions=["test_process_data"],
        test_classes=[],
        test_file_path="./generated_tests/test_module.py",
        mocks=[],
        fixtures=[],
    )

    # Mock test_generator generate_test to avoid actually calling LLM
    mock_generate_test = mocker.patch.object(
        mock_test_generator.test_generator, "generate_test", return_value=generated_test
    )

    # Mock save_test_to_file to prevent actual file system operations
    mock_save = mocker.patch.object(mock_test_generator.test_generator, "save_test_to_file")

    # Generate a test
    result = mock_test_generator.generate_test(mock_function, mock_context_files)

    # Verify that IP protection was used
    mock_redact_function.assert_called_once_with(mock_function)
    mock_sanitize.assert_called_once_with(mock_context_files)

    # Verify that the generate_test was called with the correct arguments
    mock_generate_test.assert_called_once()

    # Verify that save_test_to_file was called with the generated test
    mock_save.assert_called_once_with(generated_test)


def test_agent_initializes_ip_protector_with_config(
    mocker, mock_test_generator, mock_ip_protector, mock_config
):
    """Test that the QA agent initializes the IP protector with configuration settings."""
    # Create a fresh config with IP protection enabled
    config = QAAgentConfig(
        model_provider="openai",
        api_key="test-api-key",
        model_name="gpt-4o",
        repo_path="/test/repo",
        ip_protection_enabled=True,
        protected_patterns=["api_key", "password"],
        protected_functions=["authenticate", "encrypt"],
        protected_files=["/test/repo/src/secrets.py"],
    )

    # Mock the IPProtector class - need to mock where it's imported
    mock_ip_class = mocker.patch(
        "qa_agent.ip_protector.IPProtector", return_value=mock_ip_protector
    )

    # Initialize the agent with our IP protection enabled config
    agent = TestGenerationAgent(config)

    # Verify that IPProtector was initialized with config settings
    mock_ip_class.assert_called_once_with(
        protected_patterns=config.protected_patterns,
        protected_functions=config.protected_functions,
        protected_files=config.protected_files,
    )


def test_collaborative_test_generation_with_ip_protection(
    mocker, mock_test_generator, mock_config, mock_function, mock_context_files
):
    """Test that collaborative test generation also uses IP protection."""
    # Mock IP protector's methods
    mock_redact_function = mocker.patch.object(
        mock_test_generator.ip_protector, "redact_function", return_value=mock_function
    )
    mock_sanitize = mocker.patch.object(
        mock_test_generator.ip_protector, "sanitize_context_files", return_value=mock_context_files
    )
    mock_protect = mocker.patch.object(
        mock_test_generator.ip_protector,
        "protect",
        side_effect=lambda x: x + " [PROTECTED]" if isinstance(x, str) else x,
    )

    # Create a properly formed test
    generated_test = MagicMock(
        function=mock_function,
        test_code="def test_process_data():\n    assert True",
        imports=[],
        test_functions=["test_process_data"],
        test_classes=[],
        test_file_path="./generated_tests/test_module.py",
        mocks=[],
        fixtures=[],
    )

    # Mock test_generator generate_test_collaboratively to avoid actually calling LLM
    mock_generate_collab = mocker.patch.object(
        mock_test_generator.test_generator,
        "generate_test_collaboratively",
        return_value=generated_test,
    )

    # Mock save_test_to_file to prevent actual file system operations
    mock_save = mocker.patch.object(mock_test_generator.test_generator, "save_test_to_file")

    # Generate a test collaboratively
    feedback = "Please add more assertions"
    result = mock_test_generator.generate_test_collaboratively(
        mock_function, mock_context_files, feedback
    )

    # Verify that IP protection was used
    mock_redact_function.assert_called_once_with(mock_function)
    mock_sanitize.assert_called_once_with(mock_context_files)

    # Verify that the feedback was protected
    mock_protect.assert_any_call(feedback)

    # Verify that save_test_to_file was called with the generated test
    mock_save.assert_called_once_with(generated_test)


def test_context_enhancement_with_ip_protection(
    mocker, mock_test_generator, mock_config, mock_function
):
    """Test that the _enhance_context_with_sourcegraph method respects IP protection."""
    # Mock the function that creates a Sourcegraph client
    mock_sg_client_class = mocker.patch("qa_agent.repo_navigator.SourcegraphClient")

    # Create a mock Sourcegraph client
    mock_sg_client = MagicMock()
    mock_sg_client_class.return_value = mock_sg_client

    # Add the mock Sourcegraph client to the test generator
    mock_test_generator.sourcegraph_client = mock_sg_client

    # Mock the semantic search method to return some CodeSearchResults
    search_result = MagicMock()
    search_result.content = "sensitive_api_key = '12345'"
    search_result.file_path = "src/example.py"
    search_result.repository = "github.com/org/repo"
    mock_sg_client.semantic_search.return_value = [search_result]

    # Mock the IP protector's protect method
    mock_protect = mocker.patch.object(
        mock_test_generator.ip_protector,
        "protect",
        side_effect=lambda x: x.replace(
            "sensitive_api_key = '12345'", "sensitive_api_key = '[REDACTED]'"
        ),
    )

    # Call the method
    context_files = mock_test_generator._enhance_context_with_sourcegraph(mock_function)

    # Verify that returned content was protected
    for file in context_files:
        if "example.py" in file.path:
            assert "sensitive_api_key = '[REDACTED]'" in file.content
            assert "sensitive_api_key = '12345'" not in file.content


def test_ip_protection_with_rules_file(mocker, mock_test_generator, mock_config, mock_function):
    """Test that IP protection rules can be loaded from a file."""
    # Set up a rules file path
    mock_config.ip_protection_rules_path = "/test/repo/ip_protection_rules.json"

    # Create a mock IPProtector instance
    mock_ip_instance = MagicMock()

    # Mock the IPProtector class - need to mock where it's imported
    mock_ip_class = mocker.patch("qa_agent.ip_protector.IPProtector", return_value=mock_ip_instance)

    # Initialize the agent
    agent = TestGenerationAgent(mock_config)

    # Verify that load_protection_rules was called with the rules path
    mock_ip_instance.load_protection_rules.assert_called_once_with(
        mock_config.ip_protection_rules_path
    )
