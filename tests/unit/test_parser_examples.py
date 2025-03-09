"""
Unit tests for the parser module using the example directory structure.

These tests verify the functionality of parsing and analyzing source code files
from the organized examples directory.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from qa_agent.models import CodeFile, FileType, Function
from qa_agent.parser import (
    CodeParserFactory,
    JavaScriptCodeParser,
    PHPCodeParser,
    PythonCodeParser,
    SQLCodeParser,
)


class TestPythonCodeParserWithExamples:
    """Tests for the PythonCodeParser class using example files."""

    @pytest.fixture
    def parser(self):
        """Create a PythonCodeParser instance for testing."""
        return PythonCodeParser()

    @pytest.fixture
    def simple_python_example(self):
        """Create a CodeFile from the simple.py example."""
        example_path = os.path.join("examples", "python", "example_src", "simple.py")
        with open(example_path, "r") as f:
            content = f.read()

        return CodeFile(path=example_path, content=content)

    @pytest.fixture
    def complex_python_example(self):
        """Create a CodeFile from the complex.py example."""
        example_path = os.path.join("examples", "python", "example_src", "complex.py")
        with open(example_path, "r") as f:
            content = f.read()

        return CodeFile(path=example_path, content=content)

    def test_extract_functions_simple(self, parser, simple_python_example):
        """Test extracting functions from the simple Python example."""
        functions = parser.extract_functions(simple_python_example)

        # Should extract 4 functions
        assert len(functions) == 4

        # Check function names
        function_names = [f.name for f in functions]
        assert "add" in function_names
        assert "subtract" in function_names
        assert "multiply" in function_names
        assert "divide" in function_names

        # Check docstrings
        add_func = next(f for f in functions if f.name == "add")
        assert "Add two numbers" in add_func.docstring

        divide_func = next(f for f in functions if f.name == "divide")
        assert "Divide a by b" in divide_func.docstring

    def test_extract_functions_complex(self, parser, complex_python_example):
        """Test extracting functions from the complex Python example."""
        functions = parser.extract_functions(complex_python_example)

        # Check that expected functions are extracted
        function_names = [f.name for f in functions]
        assert "calculate_average" in function_names
        assert "validate_email" in function_names
        assert "process_user_data" in function_names
        assert "analyze_text_sentiment" in function_names
        assert "cache_decorator" in function_names
        assert "expensive_computation" in function_names

        # Check a specific function
        process_func = next(f for f in functions if f.name == "process_user_data")
        assert "Process user data with validation" in process_func.docstring
        assert len(process_func.parameters) > 0

        # Check type annotations
        avg_func = next(f for f in functions if f.name == "calculate_average")
        assert avg_func.parameters[0]["type"] == "List[float]"
        assert avg_func.return_type == "float"


class TestJavaScriptCodeParserWithExamples:
    """Tests for the JavaScriptCodeParser class using example files."""

    @pytest.fixture
    def parser(self):
        """Create a JavaScriptCodeParser instance for testing."""
        return JavaScriptCodeParser()

    @pytest.fixture
    def simple_js_example(self):
        """Create a CodeFile from the simple.js example."""
        example_path = os.path.join("examples", "js", "example_src", "simple.js")
        with open(example_path, "r") as f:
            content = f.read()

        return CodeFile(path=example_path, content=content)

    def test_extract_functions(self, parser, simple_js_example):
        """Test extracting functions from the simple JavaScript example."""
        with patch.object(parser, "_extract_function_regex") as mock_extract:
            # Set up the mock to return some functions
            mock_extract.return_value = [
                Function(
                    name="add",
                    code="function add(a, b) {\n  return a + b;\n}",
                    file_path=simple_js_example.path,
                    start_line=10,
                    end_line=12,
                ),
                Function(
                    name="subtract",
                    code="function subtract(a, b) {\n  return a - b;\n}",
                    file_path=simple_js_example.path,
                    start_line=20,
                    end_line=22,
                ),
            ]

            functions = parser.extract_functions(simple_js_example)

            # Check that the mock was called with the right arguments
            mock_extract.assert_called_once_with(simple_js_example)

            # Check that expected functions are extracted
            assert len(functions) == 2
            function_names = [f.name for f in functions]
            assert "add" in function_names
            assert "subtract" in function_names


class TestPHPCodeParserWithExamples:
    """Tests for the PHPCodeParser class using example files."""

    @pytest.fixture
    def parser(self):
        """Create a PHPCodeParser instance for testing."""
        return PHPCodeParser()

    @pytest.fixture
    def simple_php_example(self):
        """Create a CodeFile from the simple.php example."""
        example_path = os.path.join("examples", "php", "example_src", "simple.php")
        with open(example_path, "r") as f:
            content = f.read()

        return CodeFile(path=example_path, content=content)

    def test_extract_functions(self, parser, simple_php_example):
        """Test extracting functions from the simple PHP example."""
        with patch.object(parser, "_extract_function_regex") as mock_extract:
            # Set up the mock to return some functions
            mock_extract.return_value = [
                Function(
                    name="add",
                    code="function add($a, $b) {\n    return $a + $b;\n}",
                    file_path=simple_php_example.path,
                    start_line=10,
                    end_line=12,
                ),
                Function(
                    name="subtract",
                    code="function subtract($a, $b) {\n    return $a - $b;\n}",
                    file_path=simple_php_example.path,
                    start_line=20,
                    end_line=22,
                ),
            ]

            functions = parser.extract_functions(simple_php_example)

            # Check that the mock was called with the right arguments
            mock_extract.assert_called_once_with(simple_php_example)

            # Check that expected functions are extracted
            assert len(functions) == 2
            function_names = [f.name for f in functions]
            assert "add" in function_names
            assert "subtract" in function_names


class TestSQLCodeParserWithExamples:
    """Tests for the SQLCodeParser class using example files."""

    @pytest.fixture
    def parser(self):
        """Create a SQLCodeParser instance for testing."""
        return SQLCodeParser()

    @pytest.fixture
    def simple_sql_example(self):
        """Create a CodeFile from the simple.sql example."""
        example_path = os.path.join("examples", "sql", "example_src", "simple.sql")
        with open(example_path, "r") as f:
            content = f.read()

        return CodeFile(path=example_path, content=content)

    def test_extract_functions(self, parser, simple_sql_example):
        """Test extracting functions from the simple SQL example."""
        with patch.object(parser, "_extract_function_regex") as mock_extract:
            # Set up the mock to return some functions
            mock_extract.return_value = [
                Function(
                    name="is_username_available",
                    code="CREATE OR REPLACE FUNCTION is_username_available(p_username VARCHAR)\nRETURNS BOOLEAN AS $$\nDECLARE\n    user_count INTEGER;\nBEGIN\n    SELECT COUNT(*) \n    INTO user_count \n    FROM users \n    WHERE username = p_username;\n    \n    RETURN user_count = 0;\nEND;\n$$ LANGUAGE plpgsql;",
                    file_path=simple_sql_example.path,
                    start_line=20,
                    end_line=32,
                ),
                Function(
                    name="register_user",
                    code="CREATE OR REPLACE FUNCTION register_user(\n    p_username VARCHAR,\n    p_email VARCHAR,\n    p_password_hash VARCHAR\n)\nRETURNS INTEGER AS $$\nDECLARE\n    new_user_id INTEGER;\nBEGIN\n    IF NOT is_username_available(p_username) THEN\n        RAISE EXCEPTION 'Username already taken';\n    END IF;\n    \n    IF NOT is_email_available(p_email) THEN\n        RAISE EXCEPTION 'Email already registered';\n    END IF;\n    \n    INSERT INTO users (username, email, password_hash)\n    VALUES (p_username, p_email, p_password_hash)\n    RETURNING id INTO new_user_id;\n    \n    RETURN new_user_id;\nEND;\n$$ LANGUAGE plpgsql;",
                    file_path=simple_sql_example.path,
                    start_line=40,
                    end_line=65,
                ),
            ]

            functions = parser.extract_functions(simple_sql_example)

            # Check that the mock was called with the right arguments
            mock_extract.assert_called_once_with(simple_sql_example)

            # Check that expected functions are extracted
            assert len(functions) == 2
            function_names = [f.name for f in functions]
            assert "is_username_available" in function_names
            assert "register_user" in function_names
