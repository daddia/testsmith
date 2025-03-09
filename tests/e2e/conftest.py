"""
E2E test fixtures for QA Agent tests.
"""

import os
import shutil
import sys
import tempfile

import pytest

from qa_agent.config import QAAgentConfig


@pytest.fixture
def sample_repo_path():
    """Create a temporary directory with sample Python files for testing."""
    # Create a temporary directory
    temp_dir = os.path.join(tempfile.gettempdir(), "qa_agent_test_repo")
    os.makedirs(temp_dir, exist_ok=True)

    # Create a sample module directory
    sample_module_dir = os.path.join(temp_dir, "sample_module")
    os.makedirs(sample_module_dir, exist_ok=True)

    # Create __init__.py
    with open(os.path.join(sample_module_dir, "__init__.py"), "w") as f:
        f.write("# Sample module")

    # Create a sample utils.py file
    with open(os.path.join(sample_module_dir, "utils.py"), "w") as f:
        f.write(
            """
# Sample utility functions

def add_numbers(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b

def subtract_numbers(a, b):
    \"\"\"Subtract b from a and return the result.\"\"\"
    return a - b

def multiply_numbers(a, b):
    \"\"\"Multiply two numbers and return the result.\"\"\"
    return a * b

def divide_numbers(a, b):
    \"\"\"Divide a by b and return the result.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
        )

    # Create a sample app.py file
    with open(os.path.join(sample_module_dir, "app.py"), "w") as f:
        f.write(
            """
# Sample application

from .utils import add_numbers, subtract_numbers

def main():
    \"\"\"Main function for the sample application.\"\"\"
    print("Welcome to the calculator app!")
    print("2 + 3 =", add_numbers(2, 3))
    print("5 - 2 =", subtract_numbers(5, 2))
    
if __name__ == "__main__":
    main()
"""
        )

    # Return the path to the sample repository
    yield temp_dir

    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest.fixture
def e2e_config():
    """Create a configuration for E2E testing."""
    config = QAAgentConfig()
    config.repo_path = ""  # Will be set by the test
    config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_tests")
    config.verbose = True
    config.api_key = "test-api-key"
    config.model_provider = "openai"
    config.model_name = "gpt-4o"
    config.target_coverage = 80.0
    return config


@pytest.fixture
def mock_openai_responses(mocker):
    """Mock OpenAI API responses for testing."""
    # Mock the OpenAI API responses
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # Mock response with a test for add_numbers
    mock_message = mocker.MagicMock()
    mock_message.content = """
```python
import pytest
from sample_module.utils import add_numbers

def test_add_numbers():
    assert add_numbers(1, 2) == 3
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
```
"""
    mock_choice = mocker.MagicMock()
    mock_choice.message = mock_message
    mock_create.return_value.choices = [mock_choice]
    
    yield mock_create


@pytest.fixture
def disable_api_calls(mocker):
    """Disable all API calls during testing."""
    # Patch all external API calls using pytest-mock
    mocker.patch("openai.resources.chat.completions.Completions.create")
    mocker.patch("qa_agent.copilot_adapter.CopilotAdapter._call_copilot_api")
    mocker.patch("qa_agent.sourcegraph_client.SourcegraphClient.search_code")
    mocker.patch("qa_agent.sourcegraph_client.SourcegraphClient.semantic_search")
    mocker.patch("qa_agent.sourcegraph_client.SourcegraphClient.get_code_intelligence")
    mocker.patch("qa_agent.sourcegraph_client.SourcegraphClient.find_examples")
    # No need for yield with mocker as it's handled automatically by pytest-mock
