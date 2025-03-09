"""
Data models for the QA Agent.

This module defines the data structures used throughout the QA agent.
"""

import enum
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class FileType(enum.Enum):
    """Enum for file types."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    PHP = "php"
    SQL = "sql"
    UNKNOWN = "unknown"


@dataclass
class CodeFile:
    """Represents a source code file."""

    path: str
    content: str
    type: FileType = FileType.UNKNOWN

    def __post_init__(self) -> None:
        """Determine file type based on extension."""
        _, ext = os.path.splitext(self.path)
        if ext.lower() in [".py"]:
            self.type = FileType.PYTHON
        elif ext.lower() in [".js"]:
            self.type = FileType.JAVASCRIPT
        elif ext.lower() in [".ts"]:
            self.type = FileType.TYPESCRIPT
        elif ext.lower() in [".php"]:
            self.type = FileType.PHP
        elif ext.lower() in [".sql"]:
            self.type = FileType.SQL

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return os.path.basename(self.path)

    @property
    def directory(self) -> str:
        """Get the directory containing the file."""
        return os.path.dirname(self.path)


@dataclass
class Function:
    """Represents a function in source code."""

    name: str
    code: str
    file_path: str
    start_line: int
    end_line: int
    docstring: str = ""
    parameters: List[Dict[str, Optional[str]]] = field(default_factory=list)
    return_type: str = ""
    dependencies: List[str] = field(default_factory=list)
    complexity: int = 0  # Cyclomatic complexity
    cognitive_complexity: int = 0  # Cognitive complexity
    last_modified: str = ""  # Date of last modification
    call_frequency: int = 0  # How often this function is called


@dataclass
class TestResult:
    """Represents the result of a test execution."""

    success: bool
    test_file: str  # Use test_file instead of test_file_path for consistency
    target_function: str  # Use target_function instead of function for consistency
    output: str
    coverage: float = 0.0
    error_message: str = ""
    execution_time: float = 0.0
    console_analysis: Dict[str, Any] = field(default_factory=dict)  # Analysis of console output
    errors: List[str] = field(default_factory=list)  # List of error messages
    passes: int = 0  # Number of passed tests
    failures: int = 0  # Number of failed tests


@dataclass
class GeneratedTest:
    """Represents a generated test."""

    function: Function
    test_code: str
    test_file_path: str
    imports: List[str]
    test_functions: List[str] = field(default_factory=list)
    test_classes: List[str] = field(default_factory=list)
    mocks: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False
    validation_result: Optional[TestResult] = None


@dataclass
class CodeSearchResult:
    """Represents a result from Sourcegraph code search."""

    file_path: str
    repository: str
    content: str
    line_start: int
    line_end: int
    commit: Optional[str] = None
    url: Optional[str] = None
    snippets: List[str] = field(default_factory=list)
    match_score: Optional[float] = None


@dataclass
class CodeIntelligenceResult:
    """Represents code intelligence information from Sourcegraph."""

    definitions: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    hover_info: Optional[str] = None
    type_info: Optional[str] = None


@dataclass
class CoverageReport:
    """Represents a test coverage report."""

    total_coverage: float
    file_coverage: Dict[str, float]
    uncovered_functions: List[Function]
    covered_functions: List[Function]
    timestamp: str
    coverage_percentage: float = 0.0
