"""
Call Graph Analysis module.

This module provides functionality to analyze function call relationships
and build a call graph to identify critical functions based on their
call frequency and importance in the code flow.
"""

import ast
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from qa_agent.models import CodeFile, Function
from qa_agent.parser import CodeParserFactory
from qa_agent.utils.logging import get_logger, log_exception, log_function_call, log_function_result

# Use the global logging utility
logger = get_logger(__name__)


class CallGraphAnalyzer:
    """Analyzes function call relationships to build a call graph."""

    def __init__(self, repo_path: str):
        """
        Initialize the call graph analyzer.

        Args:
            repo_path: Path to the repository
        """
        self.repo_path = repo_path
        self.call_graph = defaultdict(set)  # Maps function name to set of called functions
        self.reverse_call_graph = defaultdict(set)  # Maps function to set of callers
        self.function_freq = defaultdict(int)  # Maps function name to call frequency
        self.functions_by_path = {}  # Maps file path to list of functions in that file

    def build_call_graph(self, functions: List[Function]) -> Dict[str, int]:
        """
        Build a call graph from a list of functions and calculate call frequencies.

        Args:
            functions: List of functions to analyze

        Returns:
            Dictionary mapping function identifiers to call frequencies
        """
        # Log function call
        log_function_call(logger, "build_call_graph", args=(len(functions),))

        try:
            # Build a map of functions by path for quick lookup
            for func in functions:
                full_id = f"{os.path.relpath(func.file_path, self.repo_path)}:{func.name}"
                self.functions_by_path[full_id] = func

            # First pass: extract calls from function bodies
            for func in functions:
                func_id = f"{os.path.relpath(func.file_path, self.repo_path)}:{func.name}"
                self._extract_function_calls(func, func_id)

            # Second pass: calculate call frequencies by traversing the call graph
            for func_id in self.functions_by_path:
                self._calculate_call_frequency(func_id)

            # Update the call_frequency attribute in each function
            for func in functions:
                func_id = f"{os.path.relpath(func.file_path, self.repo_path)}:{func.name}"
                if func_id in self.function_freq:
                    func.call_frequency = self.function_freq[func_id]

            # Log function result
            log_function_result(
                logger,
                "build_call_graph",
                f"Built call graph with {len(self.function_freq)} functions",
            )
            return self.function_freq
        except Exception as e:
            # Log any exceptions
            log_exception(logger, "build_call_graph", e)
            raise

    def _extract_function_calls(self, function: Function, func_id: str) -> None:
        """
        Extract function calls from a function body.

        Args:
            function: The function to analyze
            func_id: Identifier for the function (path:name)
        """
        if function.file_path.endswith(".py"):
            self._extract_python_function_calls(function, func_id)
        elif function.file_path.endswith(".js") or function.file_path.endswith(".ts"):
            self._extract_js_function_calls(function, func_id)
        elif function.file_path.endswith(".php"):
            self._extract_php_function_calls(function, func_id)

    def _extract_python_function_calls(self, function: Function, func_id: str) -> None:
        """
        Extract function calls from a Python function body using AST.

        Args:
            function: The Python function to analyze
            func_id: Identifier for the function (path:name)
        """
        try:
            # Parse the function code
            tree = ast.parse(function.code)

            # Find all function calls in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Direct function call: func()
                        called_func = node.func.id

                        # Find the called function in our function list
                        for potential_path, potential_func in self.functions_by_path.items():
                            if potential_func.name == called_func:
                                # Add to call graph and reverse call graph
                                self.call_graph[func_id].add(potential_path)
                                self.reverse_call_graph[potential_path].add(func_id)

                    elif isinstance(node.func, ast.Attribute):
                        # Method call: obj.method()
                        if isinstance(node.func.value, ast.Name):
                            # Simple method call: module.func()
                            module = node.func.value.id
                            method = node.func.attr

                            # Look for methods in the same module
                            for potential_path, potential_func in self.functions_by_path.items():
                                if potential_func.name == method and module in potential_path:
                                    self.call_graph[func_id].add(potential_path)
                                    self.reverse_call_graph[potential_path].add(func_id)

        except (SyntaxError, IndentationError) as e:
            # Handle parsing errors specifically
            log_exception(
                logger,
                "_extract_python_function_calls",
                e,
                {"function_name": function.name, "error_type": type(e).__name__},
            )
        except Exception as e:
            # Catch other unexpected errors
            log_exception(
                logger,
                "_extract_python_function_calls",
                e,
                {"function_name": function.name, "error_type": type(e).__name__},
            )

    def _extract_js_function_calls(self, function: Function, func_id: str) -> None:
        """
        Extract function calls from a JavaScript/TypeScript function using regex.

        Args:
            function: The JavaScript function to analyze
            func_id: Identifier for the function (path:name)
        """
        # Simple regex pattern to find function calls
        # This is a simplified approach; a proper parser would be better
        pattern = r"(\w+)\s*\("
        for match in re.finditer(pattern, function.code):
            called_func = match.group(1)

            # Skip some common built-in functions and keywords
            if called_func in ("if", "for", "while", "switch", "return", "console", "log"):
                continue

            # Find the called function in our function list
            for potential_path, potential_func in self.functions_by_path.items():
                if potential_func.name == called_func and potential_path.endswith((".js", ".ts")):
                    self.call_graph[func_id].add(potential_path)
                    self.reverse_call_graph[potential_path].add(func_id)

    def _extract_php_function_calls(self, function: Function, func_id: str) -> None:
        """
        Extract function calls from a PHP function using regex.

        Args:
            function: The PHP function to analyze
            func_id: Identifier for the function (path:name)
        """
        # Simple regex pattern to find function calls
        pattern = r"(\w+)\s*\("
        for match in re.finditer(pattern, function.code):
            called_func = match.group(1)

            # Skip some common built-in functions and keywords
            if called_func in (
                "if",
                "for",
                "foreach",
                "while",
                "switch",
                "return",
                "echo",
                "print",
            ):
                continue

            # Find the called function in our function list
            for potential_path, potential_func in self.functions_by_path.items():
                if potential_func.name == called_func and potential_path.endswith(".php"):
                    self.call_graph[func_id].add(potential_path)
                    self.reverse_call_graph[potential_path].add(func_id)

    def _calculate_call_frequency(self, func_id: str, visited: Optional[Set[str]] = None) -> int:
        """
        Calculate the call frequency for a function by traversing the reverse call graph.

        Args:
            func_id: The function identifier to calculate frequency for
            visited: Set of already visited functions to prevent infinite recursion

        Returns:
            Call frequency (number of times the function is called)
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion with cycles in the call graph
        if func_id in visited:
            return 0

        visited.add(func_id)

        # If already calculated, return the cached value
        if func_id in self.function_freq and self.function_freq[func_id] > 0:
            return self.function_freq[func_id]

        # Calculate frequency as the sum of direct calls plus calls to callers
        # Direct calls (each caller counts as one direct call)
        direct_calls = len(self.reverse_call_graph[func_id])

        # Sum up the frequencies of all callers (weighted by how critical they are)
        caller_freq = 0
        for caller in self.reverse_call_graph[func_id]:
            # Prevent infinite recursion
            if caller != func_id and caller not in visited:
                caller_freq += self._calculate_call_frequency(caller, visited.copy())

        # Calculate final frequency (direct calls have more weight)
        frequency = direct_calls * 2 + caller_freq
        self.function_freq[func_id] = frequency

        return frequency

    def get_critical_functions(
        self, functions: List[Function], threshold: int = 5
    ) -> List[Function]:
        """
        Identify critical functions based on call frequency.

        Args:
            functions: List of functions to filter
            threshold: Minimum call frequency to consider a function critical

        Returns:
            List of critical functions
        """
        # Log function call
        log_function_call(
            logger,
            "get_critical_functions",
            args=(len(functions),),
            kwargs={"threshold": threshold},
        )

        try:
            # Build call graph if not already built
            if not self.function_freq:
                self.build_call_graph(functions)

            critical_functions = []

            # Filter functions based on call frequency
            for func in functions:
                func_id = f"{os.path.relpath(func.file_path, self.repo_path)}:{func.name}"
                if func_id in self.function_freq and self.function_freq[func_id] >= threshold:
                    critical_functions.append(func)

            # Sort by call frequency (higher first)
            critical_functions.sort(
                key=lambda f: self.function_freq.get(
                    f"{os.path.relpath(f.file_path, self.repo_path)}:{f.name}", 0
                ),
                reverse=True,
            )

            # Log function result
            log_function_result(
                logger,
                "get_critical_functions",
                f"Found {len(critical_functions)} critical functions",
            )
            return critical_functions

        except Exception as e:
            # Log any exceptions
            log_exception(logger, "get_critical_functions", e)
            raise
