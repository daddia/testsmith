"""
Basic end-to-end tests for the QA Agent.

These tests verify the most basic functionality to ensure the testing
framework is set up correctly.
"""

import os
import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function


class TestBasicE2E:
    """Basic end-to-end tests to verify setup."""

    @pytest.mark.e2e
    def test_config_loading(self, e2e_config):
        """Test that the configuration loads correctly."""
        # Verify basic configuration properties
        assert isinstance(e2e_config, QAAgentConfig)
        assert e2e_config.model_provider == "openai"
        assert e2e_config.model_name == "gpt-4o"
        assert e2e_config.api_key == "test-api-key"
        assert e2e_config.verbose is True

    @pytest.mark.e2e
    def test_sample_repo_structure(self, sample_repo_path):
        """Test that the sample repository structure is created correctly."""
        # Check that the directory exists
        assert os.path.isdir(sample_repo_path)
        
        # Check that the sample module directory exists
        sample_module_dir = os.path.join(sample_repo_path, "sample_module")
        assert os.path.isdir(sample_module_dir)
        
        # Check that the sample files exist
        assert os.path.isfile(os.path.join(sample_module_dir, "__init__.py"))
        assert os.path.isfile(os.path.join(sample_module_dir, "utils.py"))
        assert os.path.isfile(os.path.join(sample_module_dir, "app.py"))
        
        # Verify file contents with a simple check
        with open(os.path.join(sample_module_dir, "utils.py"), "r") as f:
            content = f.read()
            assert "def add_numbers(a, b):" in content
            assert "def subtract_numbers(a, b):" in content
            assert "def multiply_numbers(a, b):" in content
            assert "def divide_numbers(a, b):" in content

    @pytest.mark.e2e
    def test_function_model(self):
        """Test that the Function model works correctly."""
        # Create a Function instance
        function = Function(
            name="test_function",
            code="def test_function():\n    return 'test'",
            file_path="/path/to/file.py",
            start_line=1,
            end_line=2,
            docstring="Test function.",
            parameters=[],
            return_type="str",
            dependencies=[],
            complexity=1,
        )
        
        # Verify the function properties
        assert function.name == "test_function"
        assert function.code == "def test_function():\n    return 'test'"
        assert function.file_path == "/path/to/file.py"
        assert function.start_line == 1
        assert function.end_line == 2
        assert function.docstring == "Test function."
        assert function.parameters == []
        assert function.return_type == "str"
        assert function.dependencies == []
        assert function.complexity == 1

    @pytest.mark.e2e
    def test_code_file_model(self):
        """Test that the CodeFile model works correctly."""
        # Create a CodeFile instance
        code_file = CodeFile(
            path="/path/to/file.py",
            content="def test_function():\n    return 'test'",
            type=FileType.PYTHON,
        )
        
        # Verify the code file properties
        assert code_file.path == "/path/to/file.py"
        assert code_file.content == "def test_function():\n    return 'test'"
        assert code_file.type == FileType.PYTHON
        assert code_file.filename == "file.py"
        assert code_file.directory == "/path/to"