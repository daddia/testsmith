"""
Test coverage analysis module.

This module provides functionality to analyze test coverage of a codebase
and identify functions with poor or no test coverage. Includes advanced
analysis with cognitive complexity, call graphs, and git history integration.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from qa_agent.call_graph_analyzer import CallGraphAnalyzer
from qa_agent.error_recovery import CircuitBreaker, ErrorHandler
from qa_agent.git_history_analyzer import GitHistoryAnalyzer
from qa_agent.models import CodeFile, CoverageReport, Function
from qa_agent.parser import CodeParserFactory
from qa_agent.utils.logging import get_logger, log_analyzed, log_exception

# Initialize logger for this module
logger = get_logger(__name__)


class CoverageAnalyzer:
    """Analyzes test coverage of a codebase."""

    def __init__(self, repo_path: str, test_framework: str = "pytest"):
        """
        Initialize the coverage analyzer.

        Args:
            repo_path: Path to the repository
            test_framework: Test framework to use (pytest, unittest, etc.)
        """
        self.repo_path = repo_path
        self.test_framework = test_framework

        # Set up error handling components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=60, half_open_max_calls=2
        )

        self.error_handler = ErrorHandler(
            max_retries=3, backoff_factor=1.5, circuit_breaker=self.circuit_breaker
        )

    def run_coverage_analysis(self) -> CoverageReport:
        """
        Run coverage analysis on the repository.

        Returns:
            CoverageReport object
        """
        logger.info(f"Running coverage analysis on {self.repo_path}")

        # Store original directory
        original_dir = os.getcwd()

        # Define a function to run with error handling
        def _run_analysis():
            try:
                # Change to the repository directory
                os.chdir(self.repo_path)

                # Run tests with coverage
                if self.test_framework == "pytest":
                    cmd = [
                        "python",
                        "-m",
                        "pytest",
                        "--cov=.",
                        "--cov-report=xml",
                        "--cov-report=json",
                    ]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        log_exception(
                            logger,
                            "run_coverage_analysis",
                            e,
                            {"framework": self.test_framework, "command": " ".join(cmd)},
                        )
                        # Create empty coverage files if they don't exist
                        if not os.path.exists("coverage.xml"):
                            with open("coverage.xml", "w") as f:
                                f.write(
                                    '<?xml version="1.0" ?><coverage version="6.4.4"></coverage>'
                                )
                        if not os.path.exists("coverage.json"):
                            with open("coverage.json", "w") as f:
                                f.write('{"meta": {"version": "6.4.4"}, "files": {}}')
                else:
                    raise ValueError(f"Unsupported test framework: {self.test_framework}")

                # Parse coverage report
                coverage_report = self._parse_coverage_report()
                return coverage_report

            except Exception as e:
                log_exception(logger, "run_coverage_analysis", e, {"repo_path": self.repo_path})
                # Re-raise for the error handler to handle
                raise
            finally:
                # Ensure we change back to the original directory even if there's an error
                os.chdir(original_dir)

        try:
            # Use error handler to handle retries
            result = self.error_handler.execute_with_retry(
                _run_analysis,
                operation_name="coverage_analysis",
                error_context={"repo_path": self.repo_path, "framework": self.test_framework},
                diagnostic_level="detailed",
            )
            return result
        except Exception:
            # If all retries fail, return an empty coverage report
            logger.error("All retries failed when running coverage analysis")
            return CoverageReport(
                total_coverage=0.0,
                file_coverage={},
                uncovered_functions=[],
                covered_functions=[],
                timestamp=datetime.now().isoformat(),
            )

    def _parse_coverage_report(self) -> CoverageReport:
        """
        Parse the coverage report generated by the coverage tool.

        Returns:
            CoverageReport object
        """
        try:
            # Read the coverage JSON report
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)

            return self.analyze_coverage_report(coverage_data)

        except Exception as e:
            log_exception(logger, "_parse_coverage_report", e)
            # Return an empty coverage report
            return CoverageReport(
                total_coverage=0.0,
                file_coverage={},
                uncovered_functions=[],
                covered_functions=[],
                timestamp=datetime.now().isoformat(),
            )

    def analyze_coverage_report(self, coverage_data: Dict) -> CoverageReport:
        """
        Analyze a coverage report data to extract information about covered and uncovered functions.

        Args:
            coverage_data: Coverage data dictionary

        Returns:
            CoverageReport object
        """
        try:
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            file_coverage = {}

            # Extract coverage for each file
            for file_path, file_data in coverage_data.get("files", {}).items():
                file_coverage[file_path] = file_data.get("summary", {}).get("percent_covered", 0)

            # Get uncovered and covered functions
            uncovered_functions, covered_functions = self._get_functions_coverage(coverage_data)

            # Calculate the overall function coverage percentage
            total_functions = len(uncovered_functions) + len(covered_functions)
            coverage_percentage = (
                100.0
                if total_functions == 0
                else (len(covered_functions) / total_functions) * 100.0
            )

            return CoverageReport(
                total_coverage=total_coverage,
                file_coverage=file_coverage,
                uncovered_functions=uncovered_functions,
                covered_functions=covered_functions,
                coverage_percentage=coverage_percentage,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            log_exception(logger, "analyze_coverage_report", e)
            # Return an empty coverage report
            return CoverageReport(
                total_coverage=0.0,
                file_coverage={},
                uncovered_functions=[],
                covered_functions=[],
                coverage_percentage=100.0,
                timestamp=datetime.now().isoformat(),
            )

    def _get_functions_coverage(self, coverage_data: Dict) -> Tuple[List[Function], List[Function]]:
        """
        Get functions with no or poor coverage.

        Args:
            coverage_data: Coverage data

        Returns:
            Tuple of (uncovered_functions, covered_functions)
        """
        uncovered_functions = []
        covered_functions = []

        # If coverage data is empty, find all functions in the codebase
        if not coverage_data.get("files", {}):
            logger.info("No coverage data found, analyzing codebase directly")
            # Analyze all Python files in the codebase
            for root, dirs, files in os.walk("."):
                # Skip directories that are likely to contain third-party code
                if any(
                    skip_dir in root
                    for skip_dir in [".git", "__pycache__", "venv", "env", ".pytest_cache"]
                ):
                    continue

                for file_name in files:
                    if file_name.endswith(".py") and not file_name.startswith("test_"):
                        file_path = os.path.join(root, file_name)
                        try:
                            with open(file_path, "r") as f:
                                content = f.read()

                            code_file = CodeFile(path=file_path, content=content)
                            parser = CodeParserFactory.get_parser(code_file.type)
                            if not parser:
                                continue

                            functions = parser.extract_functions(code_file)
                            # Consider all functions as uncovered
                            uncovered_functions.extend(functions)
                        except Exception as e:
                            log_exception(
                                logger,
                                "_get_functions_coverage",
                                e,
                                {"file_path": file_path, "phase": "no_coverage_data"},
                            )

            return uncovered_functions, covered_functions

        # Process each file in the coverage report
        for file_path, file_data in coverage_data.get("files", {}).items():
            # Skip test files
            if "test_" in file_path or file_path.startswith("test_"):
                continue

            try:
                # Read the file content
                with open(file_path, "r") as f:
                    content = f.read()

                code_file = CodeFile(path=file_path, content=content)

                # Get the appropriate parser for this file
                parser = CodeParserFactory.get_parser(code_file.type)
                if not parser:
                    continue

                # Extract functions from the file
                functions = parser.extract_functions(code_file)

                # Get uncovered line numbers
                uncovered_lines = set()
                for line_info in file_data.get("missing_lines", []):
                    if isinstance(line_info, int):
                        uncovered_lines.add(line_info)

                # Check each function for coverage
                for function in functions:
                    # Function is considered uncovered if any of its lines are uncovered
                    function_lines = set(range(function.start_line, function.end_line + 1))
                    uncovered_function_lines = function_lines.intersection(uncovered_lines)

                    # Calculate function coverage
                    function_coverage = 1.0 - (len(uncovered_function_lines) / len(function_lines))

                    if function_coverage < 1.0:
                        uncovered_functions.append(function)
                    else:
                        covered_functions.append(function)

            except Exception as e:
                log_exception(
                    logger,
                    "_get_functions_coverage",
                    e,
                    {"file_path": file_path, "phase": "with_coverage_data"},
                )

        # Sort uncovered functions by complexity (higher first)
        uncovered_functions.sort(key=lambda f: f.complexity or 0, reverse=True)

        return uncovered_functions, covered_functions

    def identify_critical_functions(
        self,
        uncovered_functions: List[Function],
        days_threshold: int = 7,
        complexity_threshold: int = 5,
        call_freq_threshold: int = 3,
    ) -> List[Function]:
        """
        Identify critical functions based on multiple factors:
        1. Cyclomatic complexity and cognitive complexity
        2. Call frequency (from call graph analysis)
        3. Recent modifications (from git history)

        Args:
            uncovered_functions: List of uncovered functions
            days_threshold: Number of days to consider a function as recently modified
            complexity_threshold: Complexity threshold for critical functions
            call_freq_threshold: Call frequency threshold for critical functions

        Returns:
            List of critical functions sorted by priority
        """
        if not uncovered_functions:
            return []

        # Create analyzers
        call_graph_analyzer = CallGraphAnalyzer(self.repo_path)
        git_analyzer = GitHistoryAnalyzer(self.repo_path)

        # Analyze call graph and git history
        logger.info("Building call graph for all functions")
        call_frequencies = call_graph_analyzer.build_call_graph(uncovered_functions)

        logger.info(f"Analyzing git history (last {days_threshold} days)")
        recently_modified = git_analyzer.filter_recently_modified_functions(
            uncovered_functions, days=days_threshold
        )

        # Track which functions are included and for what reason
        complexity_critical = set()
        call_freq_critical = set()
        recently_modified_critical = set()
        keyword_critical = set()

        # Find critical functions by different metrics
        critical_functions = []
        for function in uncovered_functions:
            func_id = f"{os.path.relpath(function.file_path, self.repo_path)}:{function.name}"

            # Check cyclomatic complexity
            if function.complexity and function.complexity >= complexity_threshold:
                critical_functions.append(function)
                complexity_critical.add(func_id)
                continue

            # Check cognitive complexity (with a slightly lower threshold)
            if (
                function.cognitive_complexity
                and function.cognitive_complexity >= complexity_threshold - 1
            ):
                critical_functions.append(function)
                complexity_critical.add(func_id)
                continue

            # Check call frequency
            if function.call_frequency and function.call_frequency >= call_freq_threshold:
                critical_functions.append(function)
                call_freq_critical.add(func_id)
                continue

            # Check recency of changes
            if function.last_modified:
                critical_functions.append(function)
                recently_modified_critical.add(func_id)
                continue

            # Check for critical keywords
            critical_keywords = [
                "process",
                "calculate",
                "validate",
                "authenticate",
                "verify",
                "analyze",
                "transform",
                "execute",
            ]
            if any(keyword in function.name.lower() for keyword in critical_keywords):
                critical_functions.append(function)
                keyword_critical.add(func_id)
                continue

        # Log statistics about critical functions using log_analyzed
        analysis_details = {
            "total_functions": len(critical_functions),
            "high_complexity": len(complexity_critical),
            "high_call_freq": len(call_freq_critical),
            "recently_modified": len(recently_modified_critical),
            "critical_keywords": len(keyword_critical),
        }
        log_analyzed(logger, "critical_functions", analysis_details)

        # Also log in a human-readable format
        logger.info(f"Found {len(critical_functions)} critical functions:")
        logger.info(f" - {len(complexity_critical)} with high complexity")
        logger.info(f" - {len(call_freq_critical)} with high call frequency")
        logger.info(f" - {len(recently_modified_critical)} recently modified")
        logger.info(f" - {len(keyword_critical)} with critical keywords")

        # Custom sort for critical functions with priority:
        # 1. Recently modified AND (high complexity OR high call frequency)
        # 2. High complexity AND high call frequency
        # 3. Recently modified
        # 4. High complexity
        # 5. High call frequency
        # 6. Critical keywords
        def critical_function_priority(func):
            func_id = f"{os.path.relpath(func.file_path, self.repo_path)}:{func.name}"

            # Priority weights
            recently_modified_weight = 3
            complexity_weight = 2
            call_freq_weight = 2
            keyword_weight = 1

            # Base score is the complexity
            score = func.complexity or 0

            # Add weights for different factors
            if func_id in recently_modified_critical:
                score += recently_modified_weight
            if func_id in complexity_critical:
                score += complexity_weight
            if func_id in call_freq_critical:
                score += call_freq_weight
            if func_id in keyword_critical:
                score += keyword_weight

            # Add cognitive complexity as a bonus
            if func.cognitive_complexity:
                score += func.cognitive_complexity / 2

            # Add call frequency as a bonus
            if func.call_frequency:
                score += func.call_frequency / 2

            return score

        # Sort by priority score (highest first)
        critical_functions.sort(key=critical_function_priority, reverse=True)

        return critical_functions
