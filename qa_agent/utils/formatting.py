"""
Formatting utilities for QA Agent.

This module contains functions for formatting output and displaying information
in a user-friendly way.
"""

import logging
from typing import Any, Dict, List, Optional

from qa_agent.models import Function, GeneratedTest, TestResult

logger = logging.getLogger(__name__)


def format_function_info(function: Function) -> str:
    """
    Format function information for display.

    Args:
        function: Function object

    Returns:
        Formatted function information
    """
    info = [
        f"Function: {function.name}",
        f"File: {function.file_path}",
        f"Lines: {function.start_line}-{function.end_line}",
    ]

    if function.complexity:
        info.append(f"Complexity: {function.complexity}")

    if function.parameters:
        params = ", ".join(
            [f"{p.get('name')}:{p.get('type') or 'Any'}" for p in function.parameters]
        )
        info.append(f"Parameters: {params}")

    if function.return_type:
        info.append(f"Return Type: {function.return_type}")

    if function.docstring:
        # Truncate long docstrings
        docstring = function.docstring
        if len(docstring) > 100:
            docstring = docstring[:97] + "..."
        info.append(f"Docstring: {docstring}")

    return "\n".join(info)


def format_test_result(test_result: TestResult) -> str:
    """
    Format test result for display.

    Args:
        test_result: Test result object

    Returns:
        Formatted test result
    """
    info = [
        f"Test File: {test_result.test_file}",
        f"Target Function: {test_result.target_function}",
        f"Success: {test_result.success}",
    ]

    if test_result.coverage is not None:
        info.append(f"Coverage: {test_result.coverage:.2f}%")

    if test_result.execution_time is not None:
        info.append(f"Execution Time: {test_result.execution_time:.2f}s")

    if test_result.error_message:
        # Truncate long error messages
        error = test_result.error_message
        if len(error) > 200:
            error = error[:197] + "..."
        info.append(f"Error: {error}")

    # Add truncated output
    output = test_result.output
    if output and len(output) > 200:
        output = output[:197] + "..."
    info.append(f"Output: {output}")

    return "\n".join(info)