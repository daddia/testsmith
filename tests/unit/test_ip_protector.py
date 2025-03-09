"""
Unit tests for the IP Protector module.

These tests verify that the IP Protector correctly redacts sensitive information
from code, functions, and files before they are sent to external LLMs.
"""

import re
from unittest.mock import mock_open, patch

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

from qa_agent.ip_protector import IPProtector
from qa_agent.models import CodeFile, FileType, Function


class TestIPProtector:
    """Tests for the IPProtector class."""

    def test_initialization_with_defaults(self):
        """Test initializing IPProtector with default values."""
        protector = IPProtector()

        assert protector.protected_patterns == []
        assert protector.protected_functions == []
        assert protector.protected_files == []

    def test_initialization_with_values(self):
        """Test initializing IPProtector with specific values."""
        protector = IPProtector(
            protected_patterns=["secret", "password"],
            protected_functions=["authenticate", "encrypt"],
            protected_files=["/path/to/sensitive/file.py"],
        )

        assert "secret" in protector.protected_patterns
        assert "password" in protector.protected_patterns
        assert "authenticate" in protector.protected_functions
        assert "encrypt" in protector.protected_functions
        assert "/path/to/sensitive/file.py" in protector.protected_files

    def test_protect_method(self):
        """Test the protect method for redacting sensitive code."""
        protector = IPProtector(protected_patterns=["api_key", "password"])

        code = """
def authenticate(username, password):
    # Check credentials
    api_key = "12345-secret-api-key"
    return check_auth(username, password, api_key)
"""

        protected_code = protector.protect(code)

        # Sensitive information should be redacted
        assert 'api_key = "12345-secret-api-key"' not in protected_code
        assert 'api_key = "[REDACTED]"' in protected_code
        assert "password" not in protected_code
        assert "[REDACTED]" in protected_code

    def test_redact_content(self):
        """Test the _redact_content method which is the core functionality."""
        protector = IPProtector(protected_patterns=["api_key\\s*=\\s*[\"'].*?[\"']", "password"])

        # Test with pattern matches
        content = 'api_key = "12345-secret"\npassword = "secure123"'
        redacted = protector._redact_content(content)

        assert 'api_key = "[REDACTED]"' in redacted
        assert "[REDACTED]" in redacted  # password is now fully replaced
        assert "secure123" not in redacted  # password value should be gone

        # Test with no matches
        content = 'username = "john"\nrole = "admin"'
        redacted = protector._redact_content(content)

        assert redacted == content  # No changes

    def test_redact_code_file_non_protected(self):
        """Test redacting a code file that is not in the protected files list."""
        protector = IPProtector(
            protected_patterns=["password"], protected_files=["/path/to/protected/file.py"]
        )

        code_file = CodeFile(path="/path/to/regular/file.py", content='password = "secure123"')

        redacted_file = protector.redact_code_file(code_file)

        # Only patterns should be redacted, file is not fully redacted
        assert redacted_file.path == code_file.path
        assert "[REDACTED]" in redacted_file.content
        assert "secure123" not in redacted_file.content

    def test_redact_code_file_protected(self):
        """Test redacting a code file that is in the protected files list."""
        protector = IPProtector(protected_files=["/path/to/protected/file.py"])

        code_file = CodeFile(
            path="/path/to/protected/file.py",
            content='def secret_function():\n    return "sensitive"',
        )

        redacted_file = protector.redact_code_file(code_file)

        # Entire file content should be redacted
        assert redacted_file.path == code_file.path
        assert redacted_file.content == "[ENTIRE FILE REDACTED FOR IP PROTECTION]"

    def test_redact_function_non_protected(self):
        """Test redacting a function that is not in the protected functions list."""
        protector = IPProtector(
            protected_patterns=["password"], protected_functions=["secret_function"]
        )

        function = Function(
            name="regular_function",
            code="def regular_function(password):\n    return password",
            file_path="/path/to/file.py",
            start_line=1,
            end_line=2,
        )

        redacted_function = protector.redact_function(function)

        # Only patterns should be redacted, function is not fully redacted
        assert redacted_function.name == function.name
        assert "def regular_function([REDACTED])" in redacted_function.code

    def test_redact_function_protected(self):
        """Test redacting a function that is in the protected functions list."""
        protector = IPProtector(protected_functions=["secret_function"])

        function = Function(
            name="secret_function",
            code='def secret_function():\n    return "sensitive"',
            file_path="/path/to/file.py",
            start_line=1,
            end_line=2,
        )

        redacted_function = protector.redact_function(function)

        # Function code should be redacted entirely
        assert redacted_function.name == function.name
        assert redacted_function.code == "[FUNCTION REDACTED FOR IP PROTECTION]"

    def test_redact_prompt(self):
        """Test redacting a prompt that contains sensitive information."""
        protector = IPProtector(protected_patterns=["api_key", "password"])

        prompt = """
Please generate a test for the following function:

```python
def authenticate(username, password):
    api_key = "12345-secret"
    return check_auth(username, password, api_key)
```
"""

        redacted_prompt = protector.redact_prompt(prompt)

        # Sensitive patterns should be redacted
        assert 'api_key = "12345-secret"' not in redacted_prompt
        assert 'api_key = "[REDACTED]"' in redacted_prompt
        assert "12345-secret" not in redacted_prompt
        assert "[REDACTED]" in redacted_prompt

    @patch("os.path.exists", return_value=False)
    def test_load_protection_rules_file_not_found(self, mock_exists):
        """Test loading protection rules when the file doesn't exist."""
        protector = IPProtector()

        with patch("structlog.stdlib.BoundLogger.warning") as mock_warning:
            protector.load_protection_rules("/path/to/nonexistent/rules.json")

            # Should log warning
            mock_warning.assert_called_once()
            assert "IP protection rules file not found" in mock_warning.call_args[0][0]

            # Protected lists should remain empty
            assert protector.protected_patterns == []
            assert protector.protected_functions == []
            assert protector.protected_files == []

    @patch("os.path.exists", return_value=True)
    def test_load_protection_rules_success(self, mock_exists):
        """Test successfully loading protection rules from a file."""
        protector = IPProtector()

        # Mock the file content
        mock_content = """{
            "protected_patterns": ["api_key", "password"],
            "protected_functions": ["authenticate", "encrypt"],
            "protected_files": ["/path/to/sensitive/file.py"]
        }"""

        with patch("builtins.open", mock_open(read_data=mock_content)):
            protector.load_protection_rules("/path/to/rules.json")

            # Protected lists should be populated
            assert "api_key" in protector.protected_patterns
            assert "password" in protector.protected_patterns
            assert "authenticate" in protector.protected_functions
            assert "encrypt" in protector.protected_functions
            assert "/path/to/sensitive/file.py" in protector.protected_files

    def test_sanitize_context_files(self):
        """Test sanitizing a list of context files."""
        protector = IPProtector(
            protected_patterns=["api_key"], protected_files=["/path/to/protected/file.py"]
        )

        context_files = [
            CodeFile(path="/path/to/regular/file.py", content='api_key = "12345-secret"'),
            CodeFile(
                path="/path/to/protected/file.py",
                content='def secret_function():\n    return "sensitive"',
            ),
        ]

        sanitized_files = protector.sanitize_context_files(context_files)

        # First file should have patterns redacted
        assert 'api_key = "[REDACTED]"' in sanitized_files[0].content

        # Second file should be entirely redacted
        assert sanitized_files[1].content == "[ENTIRE FILE REDACTED FOR IP PROTECTION]"

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        # Pattern with invalid regex syntax
        protector = IPProtector(protected_patterns=["[invalid regex)"])

        # Call the method directly to test regex error handling
        result = protector._redact_content("Some content with no redaction")

        # The method should complete without raising an exception
        # and return the original content since the regex was invalid
        assert result == "Some content with no redaction"

        # Test with a mix of valid and invalid patterns
        protector = IPProtector(protected_patterns=["api_key", "[invalid regex)"])

        # This should still replace the valid pattern
        result = protector._redact_content("This contains an api_key that should be redacted")

        # Valid patterns should be redacted, but execution should continue
        assert "api_key" not in result
        assert "[REDACTED]" in result

    def test_redact_function_with_patterns(self):
        """Test redacting a function that contains sensitive patterns but isn't fully protected."""

        # Setup patterns for API keys and passwords
        protector = IPProtector(protected_patterns=["api_key", "password"])

        function = Function(
            name="authenticate_user",
            code="def authenticate_user(username, password):\n    api_key = 'secret-key-123'\n    return call_auth_service(username, password, api_key)",
            file_path="/path/to/auth.py",
            start_line=10,
            end_line=12,
            complexity=1,
            parameters=[{"name": "username", "type": "str"}, {"name": "password", "type": "str"}],
            return_type="bool",
            docstring="Authenticate a user with username and password using our API key",
            dependencies=["call_auth_service"],
        )

        redacted_function = protector.redact_function(function)

        # Function content should be redacted but not completely removed
        print(f"Redacted function code: {redacted_function.code}")
        assert "[REDACTED]" in redacted_function.code
        assert "def authenticate_user" in redacted_function.code
        # Check that password is redacted
        assert (
            "password" not in redacted_function.code
            or 'password = "[REDACTED]"' in redacted_function.code
        )

        # Docstring should be redacted
        if redacted_function.docstring:
            assert "API key" not in redacted_function.docstring
            assert "password" not in redacted_function.docstring

        # Other attributes should be preserved
        assert redacted_function.name == function.name
        assert redacted_function.file_path == function.file_path
        assert redacted_function.parameters == function.parameters
        assert redacted_function.return_type == function.return_type
