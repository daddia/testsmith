"""
Unit tests for the test file naming conventions in the test generator.

These tests verify that the test generator correctly creates language-specific
test file names following each language's conventions.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function
from qa_agent.test_generator import TestGenerator


class TestTestFileNaming:
    """Tests for the test file naming conventions."""

    def setup_method(self):
        """Set up common test resources."""
        self.config = QAAgentConfig(
            model_provider="openai",
            model_name="o3-mini",
            api_key="test-api-key",
            output_directory="./test_output",
        )

        # Mock the LLM to avoid API calls during tests
        with patch("qa_agent.test_generator.ChatOpenAI"):
            self.test_generator = TestGenerator(self.config)

    def create_mock_function(self, file_path, name="example_function"):
        """Create a mock function for testing file naming."""
        return Function(
            name=name,
            code="def example_function():\n    return True",
            file_path=file_path,
            start_line=1,
            end_line=2,
            docstring="Example function for testing",
            parameters=[],
            return_type="bool",
            dependencies=[],
            complexity=1,
        )

    def test_python_file_naming(self):
        """Test that Python test files follow the test_*.py convention."""
        function = self.create_mock_function("/path/to/example.py")

        test_file_path = self.test_generator._get_test_file_path(function)

        # Extract just the filename for easier assertion
        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "test_example.py"
        assert test_file_path.startswith(self.config.output_directory)

    def test_javascript_file_naming(self):
        """Test that JavaScript test files follow the *.test.js convention."""
        function = self.create_mock_function("/path/to/example.js")

        test_file_path = self.test_generator._get_test_file_path(function)

        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "example.test.js"

    def test_typescript_file_naming(self):
        """Test that TypeScript test files follow the *.test.ts convention."""
        function = self.create_mock_function("/path/to/example.ts")

        test_file_path = self.test_generator._get_test_file_path(function)

        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "example.test.ts"

    def test_go_file_naming(self):
        """Test that Go test files follow the *_test.go convention."""
        function = self.create_mock_function("/path/to/example.go")

        test_file_path = self.test_generator._get_test_file_path(function)

        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "example_test.go"

    def test_php_file_naming(self):
        """Test that PHP test files follow the *Test.php convention."""
        function = self.create_mock_function("/path/to/example.php")

        test_file_path = self.test_generator._get_test_file_path(function)

        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "ExampleTest.php"
        # Test capitalization
        assert test_file_name.startswith("E")  # First letter should be capitalized

    def test_sql_file_naming(self):
        """Test that SQL test files follow the *.test.sql convention."""
        function = self.create_mock_function("/path/to/example.sql")

        test_file_path = self.test_generator._get_test_file_path(function)

        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "example.test.sql"

    def test_unsupported_extension(self):
        """Test handling of an unsupported file extension."""
        function = self.create_mock_function("/path/to/example.unknown")

        test_file_path = self.test_generator._get_test_file_path(function)

        test_file_name = os.path.basename(test_file_path)
        assert test_file_name == "test_example.unknown"

    def test_error_handling(self):
        """Test error handling for problematic file paths."""
        # Create a function with no extension in the file path
        function = self.create_mock_function("/path/to/no_extension")

        with patch("logging.getLogger"):
            test_file_path = self.test_generator._get_test_file_path(function)

        # Should not raise an exception and return a safe default
        assert os.path.basename(test_file_path) in ["test_no_extension", "test_generated.py"]

    @patch("os.path.basename")
    def test_exception_fallback(self, mock_basename):
        """Test fallback behavior when an exception occurs."""
        function = self.create_mock_function("/path/to/example.py")

        # Mock basename to raise an exception
        mock_basename.side_effect = Exception("Test exception")

        test_file_path = self.test_generator._get_test_file_path(function)

        # Should use the fallback name
        assert os.path.basename(test_file_path) == "test_generated.py"
