"""
Common test fixtures and configuration for QA Agent tests.

This file provides:
1. Common fixtures for testing
2. Custom pytest configuration hooks
3. Custom pytest collection behavior
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function, GeneratedTest, TestResult

# Add the root directory to the path to import qa_agent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_ignore_collect(collection_path, config):
    """
    Custom hook to explicitly ignore specific files or directories during collection.
    Returns True if the path should be ignored, False otherwise.

    Args:
        collection_path (pathlib.Path): The path being considered for collection
        config: The pytest config object
    """
    # Ignore all files in the qa_agent directory unless they are prefixed with test_
    path_str = str(collection_path)
    if "qa_agent" in path_str and not collection_path.name.startswith("test_"):
        return True
    return False


def pytest_configure(config):
    """
    Custom hook to configure pytest before test collection.
    """
    # Register custom markers
    config.addinivalue_line("markers", "e2e: Mark test as an end-to-end test")
    config.addinivalue_line("markers", "unit: Mark test as a unit test")
    config.addinivalue_line("markers", "integration: Mark test as an integration test")
    config.addinivalue_line("markers", "smoke: Mark test as a smoke test")
    config.addinivalue_line("markers", "performance: Mark test as a performance test")
    config.addinivalue_line("markers", "security: Mark test as a security test")


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = QAAgentConfig()
    config.repo_path = "/test/repo"
    config.test_directory = "tests"
    config.model_provider = "openai"
    config.model_name = "gpt-4o"
    config.api_key = "test-api-key"
    config.ip_protection_enabled = False
    config.verbose = True
    return config


@pytest.fixture
def mock_function():
    """Create a mock function for testing."""
    return Function(
        name="test_function",
        code="def test_function(a, b):\n    return a + b",
        file_path="/test/repo/src/module.py",
        start_line=10,
        end_line=12,
        docstring="Test function that adds two values.",
        parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
        return_type="int",
        dependencies=[],
        complexity=1,
    )


@pytest.fixture
def mock_code_file():
    """Create a mock code file for testing."""
    return CodeFile(
        path="/test/repo/src/module.py",
        content="def test_function(a, b):\n    return a + b",
        type=FileType.PYTHON,
    )


@pytest.fixture
def mock_generated_test(mock_function):
    """Create a mock generated test for testing."""
    return GeneratedTest(
        function=mock_function,
        test_code="def test_test_function():\n    assert test_function(1, 2) == 3",
        test_file_path="/test/repo/tests/test_module.py",
        imports=["pytest", "pytest_mock"],
        mocks=[],
        fixtures=[],
    )


@pytest.fixture
def mock_test_result():
    """Create a mock test result for testing."""
    return TestResult(
        success=True,
        test_file="/test/repo/tests/test_module.py",
        target_function="test_function",
        output="1 passed",
        coverage=80.0,
        error_message="No errors",
        execution_time=0.1,
    )
