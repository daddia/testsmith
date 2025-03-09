# Sourcegraph Integration Guide

This document explains how to use and configure Sourcegraph integration with QA Agent to enhance test generation quality.

## Overview

QA Agent can leverage Sourcegraph to find similar code examples across public repositories, providing better context for AI-generated tests. This integration helps the AI model understand how similar functions are typically tested in real-world projects.

## Benefits

- **Improved test quality**: Generated tests are informed by real-world testing patterns
- **Better edge case coverage**: Identifies common edge cases from similar code
- **Reduced hallucinations**: Grounds the AI model in actual usage examples
- **More idiomatic tests**: Tests follow community best practices

## Requirements

To use Sourcegraph integration, you'll need:

1. A Sourcegraph account (free tier available)
2. A Sourcegraph API token
3. Internet access for API calls

## Getting a Sourcegraph API Token

1. Sign up or log in at [Sourcegraph.com](https://sourcegraph.com)
2. Go to **User menu** → **Settings** → **Access tokens**
3. Create a new token with the `search:rest` scope
4. Copy your token for use with QA Agent

## Configuration

### Command Line

Enable Sourcegraph integration via the command line:

```bash
qa-agent --repo-path /path/to/your/project \
         --enable-sourcegraph \
         --sourcegraph-token your-token-here
```

### Configuration File

Alternatively, enable it in your configuration YAML:

```yaml
# qa_config.yaml
repo_path: "/path/to/your/project"
sourcegraph_enabled: true
sourcegraph_api_endpoint: "https://sourcegraph.com/.api"
sourcegraph_api_token: "your-token-here"  # Or use environment variable
```

### Environment Variable

For better security, set your token as an environment variable:

```bash
export SOURCEGRAPH_API_TOKEN="your-token-here"
```

Then run QA Agent without explicitly specifying the token:

```bash
qa-agent --repo-path /path/to/your/project --enable-sourcegraph
```

## How It Works

When Sourcegraph integration is enabled:

1. QA Agent extracts key information from the function to be tested
2. It sends a search query to Sourcegraph to find similar functions
3. It performs a semantic search for similar testing patterns
4. Relevant code examples are added to the context for the AI model
5. The AI uses this enhanced context to generate more accurate tests

## Available Search Types

QA Agent uses three types of Sourcegraph searches:

### 1. Exact Code Search

Finds exact matches for function signatures or patterns:

```python
def calculate_tax(amount: float, rate: float) -> float:
```

### 2. Semantic Code Search

Finds code with similar meaning regardless of syntax:

```
"function to calculate tax based on amount and tax rate"
```

### 3. Function Example Search

Finds usage examples of specific functions or libraries:

```python
pytest.fixture
unittest.mock.patch
```

## Adjusting Query Scope

By default, QA Agent limits searches to popular repositories in the function's language. You can modify this by creating a custom Sourcegraph client configuration.

### Example Custom Configuration

```python
# Custom Sourcegraph query scope
custom_sourcegraph_settings = {
    "max_results": 10,  # More results for better context
    "search_scope": "repo:^github\.com/pytest-dev/pytest$ OR repo:^github\.com/python/cpython$",
    "min_stars": 100,  # Only repositories with at least 100 stars
    "languages": ["python", "typescript"]  # Search across multiple languages
}

# Use in configuration
config = QAAgentConfig(
    sourcegraph_enabled=True,
    sourcegraph_api_token="your-token",
    sourcegraph_custom_settings=custom_sourcegraph_settings
)
```

## IP Protection Considerations

When using Sourcegraph integration:

1. Your function names and signatures may be sent to Sourcegraph
2. No complete function implementations are sent if IP protection is enabled
3. Consider using the `protected_functions` setting to exclude sensitive functions

## Troubleshooting

### Rate Limiting

Sourcegraph has API rate limits. If you encounter rate limiting:

- Reduce the number of functions processed in a single run
- Add delays between API calls in the configuration
- Consider upgrading your Sourcegraph account

### Search Quality

If search results aren't relevant:

- Try modifying the function names to be more descriptive
- Add more detailed docstrings to your functions
- Adjust the semantic search parameters in the configuration

### No Results Found

If no results are found:

- Check your internet connection
- Verify your API token is valid
- Try searching for more common function patterns
- Ensure the language is correctly detected