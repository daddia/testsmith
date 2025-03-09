"""
Unit tests for the models module.

These tests verify the functionality of the data models used throughout the QA agent.
"""

import os
from datetime import datetime

import pytest

from qa_agent.models import (
    CodeFile,
    CodeIntelligenceResult,
    CodeSearchResult,
    CoverageReport,
    FileType,
    Function,
    GeneratedTest,
    TestResult,
)


class TestFileType:
    """Tests for the FileType enum."""

    def test_file_type_values(self):
        """Test the available file type values."""
        assert FileType.PYTHON.value == "python"
        assert FileType.JAVASCRIPT.value == "javascript"
        assert FileType.TYPESCRIPT.value == "typescript"
        assert FileType.PHP.value == "php"
        assert FileType.SQL.value == "sql"
        assert FileType.UNKNOWN.value == "unknown"


class TestCodeFile:
    """Tests for the CodeFile dataclass."""

    def test_initialization(self):
        """Test initialization with required values."""
        code_file = CodeFile(
            path="/test/repo/src/module.py", content="def test_function():\n    return True"
        )

        assert code_file.path == "/test/repo/src/module.py"
        assert code_file.content == "def test_function():\n    return True"
        assert code_file.type == FileType.PYTHON  # Auto-detected from extension

    def test_manual_file_type(self):
        """Test setting file type manually."""
        code_file = CodeFile(
            path="/test/repo/src/module.js",
            content="function testFunction() { return true; }",
            type=FileType.JAVASCRIPT,
        )

        assert code_file.type == FileType.JAVASCRIPT

    def test_post_init_file_type_detection(self):
        """Test file type detection in post_init for different extensions."""
        # Python
        assert CodeFile(path="test.py", content="").type == FileType.PYTHON

        # JavaScript
        assert CodeFile(path="test.js", content="").type == FileType.JAVASCRIPT

        # TypeScript
        assert CodeFile(path="test.ts", content="").type == FileType.TYPESCRIPT

        # PHP
        assert CodeFile(path="test.php", content="").type == FileType.PHP

        # SQL
        assert CodeFile(path="test.sql", content="").type == FileType.SQL

        # Unknown extension
        assert CodeFile(path="test.xyz", content="").type == FileType.UNKNOWN

    def test_filename_property(self):
        """Test the filename property."""
        code_file = CodeFile(path="/test/repo/src/module.py", content="")

        assert code_file.filename == "module.py"

    def test_directory_property(self):
        """Test the directory property."""
        code_file = CodeFile(path="/test/repo/src/module.py", content="")

        assert code_file.directory == "/test/repo/src"


class TestFunction:
    """Tests for the Function dataclass."""

    def test_initialization(self):
        """Test initialization with required values."""
        function = Function(
            name="test_function",
            code="def test_function(a, b):\n    return a + b",
            file_path="/test/repo/src/module.py",
            start_line=10,
            end_line=12,
        )

        assert function.name == "test_function"
        assert function.code == "def test_function(a, b):\n    return a + b"
        assert function.file_path == "/test/repo/src/module.py"
        assert function.start_line == 10
        assert function.end_line == 12
        assert function.docstring == ""
        assert function.parameters == []
        assert function.return_type == ""
        assert function.dependencies == []
        assert function.complexity == 0

    def test_initialization_with_all_fields(self):
        """Test initialization with all fields."""
        function = Function(
            name="test_function",
            code='def test_function(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b',
            file_path="/test/repo/src/module.py",
            start_line=10,
            end_line=13,
            docstring="Add two numbers.",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="int",
            dependencies=["math"],
            complexity=1,
        )

        assert function.name == "test_function"
        assert function.docstring == "Add two numbers."
        assert len(function.parameters) == 2
        assert function.parameters[0]["name"] == "a"
        assert function.parameters[0]["type"] == "int"
        assert function.return_type == "int"
        assert function.dependencies == ["math"]
        assert function.complexity == 1


class TestTestResult:
    """Tests for the TestResult dataclass."""

    def test_initialization(self):
        """Test initialization with required values."""
        test_result = TestResult(
            success=True,
            test_file="/test/repo/tests/test_module.py",
            target_function="test_function",
            output="1 passed in 0.1s",
        )

        assert test_result.success is True
        assert test_result.test_file == "/test/repo/tests/test_module.py"
        assert test_result.target_function == "test_function"
        assert test_result.output == "1 passed in 0.1s"
        assert test_result.coverage == 0.0
        assert test_result.error_message == ""
        assert test_result.execution_time == 0.0

    def test_initialization_with_all_fields(self):
        """Test initialization with all fields."""
        test_result = TestResult(
            success=False,
            test_file="/test/repo/tests/test_module.py",
            target_function="test_function",
            output="1 failed in 0.2s",
            coverage=75.5,
            error_message="AssertionError: expected 3, got 4",
            execution_time=0.203,
        )

        assert test_result.success is False
        assert test_result.coverage == 75.5
        assert test_result.error_message == "AssertionError: expected 3, got 4"
        assert test_result.execution_time == 0.203


class TestGeneratedTest:
    """Tests for the GeneratedTest dataclass."""

    def test_initialization(self, mock_function):
        """Test initialization with required values."""
        generated_test = GeneratedTest(
            function=mock_function,
            test_code="def test_test_function():\n    assert test_function(1, 2) == 3",
            test_file_path="/test/repo/tests/test_module.py",
            imports=["pytest"],
        )

        assert generated_test.function == mock_function
        assert (
            generated_test.test_code
            == "def test_test_function():\n    assert test_function(1, 2) == 3"
        )
        assert generated_test.test_file_path == "/test/repo/tests/test_module.py"
        assert generated_test.imports == ["pytest"]
        assert generated_test.mocks == []
        assert generated_test.fixtures == []
        assert generated_test.validated is False
        assert generated_test.validation_result is None

    def test_initialization_with_all_fields(self, mock_function, mock_test_result):
        """Test initialization with all fields."""
        generated_test = GeneratedTest(
            function=mock_function,
            test_code="def test_test_function():\n    assert test_function(1, 2) == 3",
            test_file_path="/test/repo/tests/test_module.py",
            imports=["pytest", "unittest.mock"],
            mocks=["mock_database"],
            fixtures=["fixture_data"],
            validated=True,
            validation_result=mock_test_result,
        )

        assert generated_test.mocks == ["mock_database"]
        assert generated_test.fixtures == ["fixture_data"]
        assert generated_test.validated is True
        assert generated_test.validation_result == mock_test_result


class TestCodeSearchResult:
    """Tests for the CodeSearchResult dataclass."""

    def test_initialization(self):
        """Test initialization with required values."""
        search_result = CodeSearchResult(
            file_path="src/module.py",
            repository="github.com/user/repo",
            content="def test_function(a, b):\n    return a + b",
            line_start=10,
            line_end=12,
        )

        assert search_result.file_path == "src/module.py"
        assert search_result.repository == "github.com/user/repo"
        assert search_result.content == "def test_function(a, b):\n    return a + b"
        assert search_result.line_start == 10
        assert search_result.line_end == 12
        assert search_result.commit is None
        assert search_result.url is None
        assert search_result.snippets == []
        assert search_result.match_score is None

    def test_initialization_with_all_fields(self):
        """Test initialization with all fields."""
        search_result = CodeSearchResult(
            file_path="src/module.py",
            repository="github.com/user/repo",
            content="def test_function(a, b):\n    return a + b",
            line_start=10,
            line_end=12,
            commit="abc123",
            url="https://github.com/user/repo/blob/abc123/src/module.py#L10-L12",
            snippets=["def test_function(a, b):"],
            match_score=0.95,
        )

        assert search_result.commit == "abc123"
        assert search_result.url == "https://github.com/user/repo/blob/abc123/src/module.py#L10-L12"
        assert search_result.snippets == ["def test_function(a, b):"]
        assert search_result.match_score == 0.95


class TestCodeIntelligenceResult:
    """Tests for the CodeIntelligenceResult dataclass."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        result = CodeIntelligenceResult()

        assert result.definitions == []
        assert result.references == []
        assert result.hover_info is None
        assert result.type_info is None

    def test_initialization_with_all_fields(self):
        """Test initialization with all fields."""
        definitions = [{"name": "test_function", "file": "module.py", "line": 10}]
        references = [{"file": "test.py", "line": 15}]

        result = CodeIntelligenceResult(
            definitions=definitions,
            references=references,
            hover_info="Function to test adding two numbers",
            type_info="(int, int) -> int",
        )

        assert result.definitions == definitions
        assert result.references == references
        assert result.hover_info == "Function to test adding two numbers"
        assert result.type_info == "(int, int) -> int"


class TestCoverageReport:
    """Tests for the CoverageReport dataclass."""

    def test_initialization(self, mock_function):
        """Test initialization."""
        now = datetime.now().isoformat()
        uncovered_function = mock_function
        covered_function = Function(
            name="covered_function",
            code="def covered_function():\n    return True",
            file_path="/test/repo/src/module.py",
            start_line=15,
            end_line=17,
        )

        report = CoverageReport(
            total_coverage=75.5,
            file_coverage={"/test/repo/src/module.py": 80.0, "/test/repo/src/other.py": 70.0},
            uncovered_functions=[uncovered_function],
            covered_functions=[covered_function],
            timestamp=now,
        )

        assert report.total_coverage == 75.5
        assert report.file_coverage["/test/repo/src/module.py"] == 80.0
        assert report.file_coverage["/test/repo/src/other.py"] == 70.0
        assert report.uncovered_functions == [uncovered_function]
        assert report.covered_functions == [covered_function]
        assert report.timestamp == now
