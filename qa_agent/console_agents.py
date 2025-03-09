"""
Console-based agents module.

This module defines agents that interact with console output in VS Code
to provide enhanced debugging and test analysis.
"""

import json
import logging

# Importing Tool classes from agents module with type comments for mypy
from typing import Any, Dict, List, Optional, Protocol, Tuple

from typing_extensions import TypedDict

from qa_agent.config import QAAgentConfig
from qa_agent.console_reader import analyze_console_output
from qa_agent.models import CodeFile, Function, GeneratedTest, TestResult


# Define Tool and BaseTool interfaces for type checking
class BaseTool(Protocol):
    """Protocol defining the interface for a tool."""

    def __init__(self, config: QAAgentConfig) -> None: ...


class ToolDict(TypedDict):
    """Type definition for a tool dictionary."""

    name: str
    func: object
    description: str


# Simple Tool class for type checking
class Tool:
    """Simple Tool class for type checking."""

    def __init__(self, name: str, func: object, description: str):
        self.name = name
        self.func = func
        self.description = description


logger = logging.getLogger(__name__)


class ConsoleAnalysisAgent:
    """Agent for analyzing console output to enhance debugging and testing."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the console analysis agent.

        Args:
            config: Configuration object
        """
        self.config = config

    def analyze_console_output(self, console_data: str) -> Dict[str, Any]:
        """
        Analyze console output for errors and test results.

        Args:
            console_data: Console output data

        Returns:
            Dictionary with analysis results
        """
        return analyze_console_output(console_data, "QA Agent")

    def identify_test_failures(self, console_data: str) -> List[str]:
        """
        Identify failing tests from console output.

        Args:
            console_data: Console output data

        Returns:
            List of failing test identifiers
        """
        analysis = self.analyze_console_output(console_data)
        failing_tests = analysis.get("test_results", {}).get("failing_tests", [])
        if failing_tests is None:
            return []
        # Ensure we're returning a list of strings
        return [str(test) for test in failing_tests if test is not None]

    def extract_error_messages(self, console_data: str) -> List[str]:
        """
        Extract error messages from console output.

        Args:
            console_data: Console output data

        Returns:
            List of error messages
        """
        analysis = self.analyze_console_output(console_data)
        errors = analysis.get("errors", [])
        if errors is None:
            return []
        # Ensure we're returning a list of strings
        return [str(error) for error in errors if error is not None]

    def get_test_coverage(self, console_data: str) -> Optional[int]:
        """
        Get test coverage percentage from console output.

        Args:
            console_data: Console output data

        Returns:
            Test coverage percentage or None if not found
        """
        analysis = self.analyze_console_output(console_data)
        coverage = analysis.get("test_results", {}).get("coverage")
        if coverage is not None and isinstance(coverage, float):
            return int(coverage)
        return None

    def suggest_test_fixes(self, console_data: str, test: GeneratedTest) -> Dict[str, Any]:
        """
        Suggest fixes for failing tests based on console output.

        Args:
            console_data: Console output data
            test: The failing test

        Returns:
            Dictionary with suggested fixes
        """
        analysis = self.analyze_console_output(console_data)
        failing_tests = analysis.get("test_results", {}).get("failing_tests", [])
        errors = analysis.get("errors", [])

        # Check if the test is in the failing tests list
        test_failed = any(test.function.name in fail_test for fail_test in failing_tests)

        if not test_failed:
            return {
                "test_failed": False,
                "message": f"Test for {test.function.name} did not fail according to console output.",
            }

        # Identify relevant error messages
        relevant_errors = []
        for error in errors:
            if test.function.name in error or test.test_file_path in error:
                relevant_errors.append(error)

        return {
            "test_failed": True,
            "function_name": test.function.name,
            "test_file": test.test_file_path,
            "errors": relevant_errors,
            "suggestions": [self._suggest_fix_for_error(error, test) for error in relevant_errors],
        }

    def _suggest_fix_for_error(self, error: str, test: GeneratedTest) -> str:
        """
        Suggest a fix for a specific error.

        Args:
            error: The error message
            test: The failing test

        Returns:
            Suggested fix
        """
        if "AssertionError" in error:
            return "Check your assert statements and expected values."
        elif "AttributeError" in error:
            return "Check for incorrect attribute access or missing attributes."
        elif "ImportError" in error or "ModuleNotFoundError" in error:
            return "Check your import statements."
        elif "TypeError" in error:
            return "Check argument types in function calls."
        elif "ValueError" in error:
            return "Check the values you're passing to functions."
        elif "NameError" in error:
            return "Check for undefined variables."
        else:
            return "Review the error message and test code carefully."


class ConsoleTool(BaseTool):
    """Tool for analyzing console output."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the console tool.

        Args:
            config: Configuration object
        """
        self.config = config
        self.console_agent = ConsoleAnalysisAgent(config)

    def analyze_console(self, console_data: str) -> Dict[str, Any]:
        """
        Analyze console output.

        Args:
            console_data: Console output data

        Returns:
            Dictionary with analysis results
        """
        return self.console_agent.analyze_console_output(console_data)

    def identify_test_failures(self, console_data: str) -> List[str]:
        """
        Identify failing tests from console output.

        Args:
            console_data: Console output data

        Returns:
            List of failing test identifiers
        """
        return self.console_agent.identify_test_failures(console_data)

    def extract_error_messages(self, console_data: str) -> List[str]:
        """
        Extract error messages from console output.

        Args:
            console_data: Console output data

        Returns:
            List of error messages
        """
        return self.console_agent.extract_error_messages(console_data)

    def get_test_coverage(self, console_data: str) -> Optional[int]:
        """
        Get test coverage percentage from console output.

        Args:
            console_data: Console output data

        Returns:
            Test coverage percentage or None if not found
        """
        return self.console_agent.get_test_coverage(console_data)

    def suggest_test_fixes(self, console_data: str, test: GeneratedTest) -> Dict[str, Any]:
        """
        Suggest fixes for failing tests based on console output.

        Args:
            console_data: Console output data
            test: The failing test

        Returns:
            Dictionary with suggested fixes
        """
        return self.console_agent.suggest_test_fixes(console_data, test)


class ConsoleTools:
    """Collection of console-related tools for QA agents."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize console tools.

        Args:
            config: Configuration object
        """
        self.config = config
        self.console_agent = ConsoleAnalysisAgent(config)

    def get_tools(self) -> List[Tool]:
        """
        Get console-related tools for QA agents.

        Returns:
            List of tools
        """
        return [
            Tool(
                name="analyze_console",
                func=lambda console_data: self.console_agent.analyze_console_output(console_data),
                description="Analyze console output for errors and test results",
            ),
            Tool(
                name="identify_test_failures",
                func=lambda console_data: self.console_agent.identify_test_failures(console_data),
                description="Identify failing tests from console output",
            ),
            Tool(
                name="extract_error_messages",
                func=lambda console_data: self.console_agent.extract_error_messages(console_data),
                description="Extract error messages from console output",
            ),
            Tool(
                name="get_test_coverage",
                func=lambda console_data: self.console_agent.get_test_coverage(console_data),
                description="Get test coverage percentage from console output",
            ),
            Tool(
                name="suggest_test_fixes",
                func=lambda console_data, test: self.console_agent.suggest_test_fixes(
                    console_data, test
                ),
                description="Suggest fixes for failing tests based on console output",
            ),
        ]
