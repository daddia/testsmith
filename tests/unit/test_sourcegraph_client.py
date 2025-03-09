"""
Unit tests for the Sourcegraph client.

These tests verify the functionality of integrating with Sourcegraph's APIs
for enhanced code context gathering and code intelligence.
"""

import pytest
import requests

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeIntelligenceResult, CodeSearchResult
from qa_agent.sourcegraph_client import SourcegraphClient


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return QAAgentConfig(
        sourcegraph_enabled=True,
        sourcegraph_api_endpoint="https://sourcegraph.example.com/.api",
        sourcegraph_api_token="test-token",
    )


@pytest.fixture
def client(mock_config, mocker):
    """Create a Sourcegraph client for testing."""
    # Mock the requests module to prevent actual HTTP calls
    mocker.patch("requests.post")
    mocker.patch("requests.get")

    # Create client with mock config
    client = SourcegraphClient(mock_config)

    # No need to set attributes directly, they're properly initialized from config

    return client


def test_init(mock_config, mocker):
    """Test initialization of the Sourcegraph client."""
    # Mock requests module to prevent actual HTTP calls
    mocker.patch("requests.post")

    client = SourcegraphClient(mock_config)

    # These attributes are set in __init__
    assert client.api_endpoint == mock_config.sourcegraph_api_endpoint
    assert client.api_token == mock_config.sourcegraph_api_token
    assert not client.limited_mode  # This would be True if authenticated


def test_search_code(mocker, client):
    """Test searching code using the Sourcegraph client."""
    # Setup mock response for requests.get (not post)
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    # The current implementation uses NDJSON format, not JSON
    mock_response.text = """
    {"type":"match","data":{"path":"src/module.py","repository":"github.com/user/repo","content":"def test_function():\\n    return True","lineNumber":1,"lineMatches":[{"line":"def test_function():"}],"commit":"abc123"}}
    """

    # Mock the requests.get method (note: not post)
    mock_get = mocker.patch("requests.get", return_value=mock_response)

    # Call the search_code method
    results = client.search_code("test_function", limit=1)

    # Verify request
    mock_get.assert_called_once()

    # Verify the request params
    call_args = mock_get.call_args[1]
    assert "params" in call_args
    assert "q" in call_args["params"]
    assert "test_function" in call_args["params"]["q"]

    # Verify results
    assert len(results) == 1
    assert results[0].file_path == "src/module.py"
    assert results[0].repository == "github.com/user/repo"
    assert results[0].content == "def test_function():\n    return True"
    assert results[0].line_start == 1


def test_semantic_search(mocker, client):
    """Test semantic code search using the Sourcegraph client."""
    # First mock the regular search call that semantic search uses internally
    mock_search_response = mocker.MagicMock()
    mock_search_response.status_code = 200
    # The current implementation uses NDJSON format
    mock_search_response.text = """
    {"type":"match","data":{"path":"src/similar_module.py","repository":"github.com/user/repo","content":"def similar_function():\\n    return True","lineNumber":1,"lineMatches":[{"line":"def similar_function():"}],"commit":"abc123"}}
    """

    # Mock the requests.get method for search
    mock_get = mocker.patch("requests.get", return_value=mock_search_response)

    # Call the semantic_search method
    results = client.semantic_search("function that returns true", limit=1)

    # Verify request
    mock_get.assert_called_once()

    # Verify the request params
    call_args = mock_get.call_args[1]
    assert "params" in call_args
    assert "q" in call_args["params"]
    assert "content:function that returns true patternType:semantic" in call_args["params"]["q"]

    # Verify results
    assert len(results) == 1
    assert results[0].file_path == "src/similar_module.py"
    assert results[0].repository == "github.com/user/repo"
    assert results[0].content == "def similar_function():\n    return True"
    assert results[0].line_start == 1


def test_get_code_intelligence(mocker, client):
    """Test getting code intelligence information using the Sourcegraph client."""
    # Setup mock responses for the three API calls

    # 1. Hover info
    hover_response = mocker.MagicMock()
    hover_response.status_code = 200
    hover_response.json.return_value = {
        "markdown": {"text": "Function to test something", "kind": "python"}
    }

    # 2. Definitions
    definitions_response = mocker.MagicMock()
    definitions_response.status_code = 200
    definitions_response.json.return_value = {
        "locations": [
            {
                "uri": "src/module.py",
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 16},
                },
            }
        ]
    }

    # 3. References
    references_response = mocker.MagicMock()
    references_response.status_code = 200
    references_response.json.return_value = {
        "locations": [
            {
                "uri": "test/test_module.py",
                "range": {
                    "start": {"line": 5, "character": 4},
                    "end": {"line": 5, "character": 20},
                },
            }
        ]
    }

    # Mock the requests.get method with side_effect to return different responses
    mock_get = mocker.patch(
        "requests.get", side_effect=[hover_response, definitions_response, references_response]
    )

    # Call the get_code_intelligence method
    result = client.get_code_intelligence("repo/src/module.py", 1, 5)

    # Verify request was called 3 times (hover, definitions, references)
    assert mock_get.call_count == 3

    # Verify the params in the first call
    first_call_args = mock_get.call_args_list[0][1]
    assert "params" in first_call_args
    assert first_call_args["params"]["repository"] == "repo"
    assert first_call_args["params"]["path"] == "src/module.py"
    assert first_call_args["params"]["line"] == 1

    # Verify results
    assert isinstance(result, CodeIntelligenceResult)
    assert result.hover_info == "Function to test something"
    assert result.type_info == "python"
    assert len(result.definitions) == 1
    assert len(result.references) == 1
    assert result.definitions[0]["uri"] == "src/module.py"
    assert result.references[0]["uri"] == "test/test_module.py"


def test_find_examples(mocker, client):
    """Test finding usage examples using the Sourcegraph client."""
    # Setup mock response for search_code method (called by find_examples)
    mock_search_response = mocker.MagicMock()
    mock_search_response.status_code = 200
    # The current implementation uses NDJSON format for search results
    mock_search_response.text = """
    {"type":"match","data":{"path":"src/usage.py","repository":"github.com/user/repo","content":"result = test_function()","lineNumber":1,"lineMatches":[{"line":"result = test_function()"}],"commit":"abc123"}}
    """

    # Mock the requests.get method that search_code uses
    mock_get = mocker.patch("requests.get", return_value=mock_search_response)

    # Call the find_examples method
    results = client.find_examples("test_function", limit=1)

    # Verify request
    mock_get.assert_called_once()

    # Verify the request params
    call_args = mock_get.call_args[1]
    assert "params" in call_args
    assert "q" in call_args["params"]
    query = call_args["params"]["q"]
    assert "test_function" in query

    # Verify results
    assert len(results) == 1
    assert results[0].file_path == "src/usage.py"
    assert results[0].content == "result = test_function()"
    assert results[0].repository == "github.com/user/repo"


def test_error_handling(mocker, client):
    """Test error handling in the Sourcegraph client."""
    # Setup mock response
    mock_response = mocker.MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    # Use requests.exceptions.HTTPError for more realistic testing
    http_error = requests.exceptions.HTTPError("401 Client Error: Unauthorized")
    mock_response.raise_for_status.side_effect = http_error

    # Mock the requests.get method (for search_code)
    mock_get = mocker.patch("requests.get", return_value=mock_response)

    # Mock the module logger to capture logging
    mock_logger = mocker.MagicMock()
    mocker.patch("qa_agent.sourcegraph_client.logger", mock_logger)

    # Call the search_code method
    results = client.search_code("test_function")

    # Should return empty list on error
    assert results == []

    # Should log error
    mock_logger.error.assert_called_once()
    assert "Error searching Sourcegraph code" in mock_logger.error.call_args[0][0]
