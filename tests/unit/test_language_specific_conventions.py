"""
Unit tests for language-specific test file naming conventions.

These tests verify that test files are generated with the correct naming
conventions for each supported programming language.
"""

import os

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function
from qa_agent.test_generator import TestGenerator


class TestLanguageSpecificConventions:
    """
    Tests for verifying language-specific test file naming conventions.
    """

    def setup_method(self):
        """Set up test environment."""
        self.config = QAAgentConfig(
            model_provider="openai", model_name="o3-mini", output_directory="./test_output"
        )
        self.test_generator = TestGenerator(self.config)

        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_directory, exist_ok=True)

    def test_python_file_naming(self):
        """Test Python test file naming convention (test_*.py)."""
        # Create a mock Python function
        function = Function(
            name="example_function",
            file_path="src/module.py",
            code="def example_function():\n    return True",
            start_line=1,
            end_line=2,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the test file path follows Python convention
        assert os.path.basename(test_file_path) == "test_module.py"

    def test_javascript_file_naming(self):
        """Test JavaScript test file naming convention (*.test.js)."""
        # Create a mock JavaScript function
        function = Function(
            name="exampleFunction",
            file_path="src/module.js",
            code="function exampleFunction() {\n  return true;\n}",
            start_line=1,
            end_line=3,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the test file path follows JavaScript convention
        assert os.path.basename(test_file_path) == "module.test.js"

    def test_typescript_file_naming(self):
        """Test TypeScript test file naming convention (*.test.ts)."""
        # Create a mock TypeScript function
        function = Function(
            name="exampleFunction",
            file_path="src/module.ts",
            code="function exampleFunction(): boolean {\n  return true;\n}",
            start_line=1,
            end_line=3,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the test file path follows TypeScript convention
        assert os.path.basename(test_file_path) == "module.test.ts"

    def test_php_file_naming(self):
        """Test PHP test file naming convention (*Test.php)."""
        # Create a mock PHP function
        function = Function(
            name="exampleFunction",
            file_path="src/Module.php",
            code="function exampleFunction() {\n  return true;\n}",
            start_line=1,
            end_line=3,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the test file path follows PHP convention
        assert os.path.basename(test_file_path) == "ModuleTest.php"

    def test_go_file_naming(self):
        """Test Go test file naming convention (*_test.go)."""
        # Create a mock Go function
        function = Function(
            name="ExampleFunction",
            file_path="src/module.go",
            code="func ExampleFunction() bool {\n  return true\n}",
            start_line=1,
            end_line=3,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the test file path follows Go convention
        assert os.path.basename(test_file_path) == "module_test.go"

    def test_sql_file_naming(self):
        """Test SQL test file naming convention (*.test.sql)."""
        # Create a mock SQL function
        function = Function(
            name="example_function",
            file_path="src/module.sql",
            code="CREATE FUNCTION example_function() RETURNS BOOLEAN AS $$\nBEGIN\n  RETURN TRUE;\nEND;\n$$ LANGUAGE plpgsql;",
            start_line=1,
            end_line=5,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the test file path follows SQL convention
        assert os.path.basename(test_file_path) == "module.test.sql"

    def test_unknown_extension_fallback(self):
        """Test fallback for unknown file extensions."""
        # Create a mock function with unsupported extension
        function = Function(
            name="example_function",
            file_path="src/module.xyz",
            code="example_function() { return true; }",
            start_line=1,
            end_line=1,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the fallback naming convention is used
        assert os.path.basename(test_file_path) == "test_module.xyz"

    def test_case_preservation(self):
        """Test that the original case of the filename is preserved."""
        # Create a mock function with mixed case in filename
        function = Function(
            name="example_function",
            file_path="src/MixedCase.py",
            code="def example_function():\n    return True",
            start_line=1,
            end_line=2,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify the case is preserved in the filename
        assert os.path.basename(test_file_path) == "test_MixedCase.py"

    def test_naming_with_dots(self):
        """Test naming convention when filename contains dots."""
        # Create a mock function with dots in filename
        function = Function(
            name="example_function",
            file_path="src/module.v1.2.js",
            code="function example_function() { return true; }",
            start_line=1,
            end_line=1,
        )

        # Generate test file path
        test_file_path = self.test_generator._get_test_file_path(function)

        # Verify dots are handled correctly
        assert os.path.basename(test_file_path) == "module.v1.2.test.js"
