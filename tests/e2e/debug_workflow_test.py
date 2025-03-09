#!/usr/bin/env python
"""
Debug script for the workflow test.
"""

import os
import sys
import tempfile
import traceback
from typing import Any, Dict

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function, GeneratedTest, TestResult
from qa_agent.workflows import QAWorkflow

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def main():
    # Set up configuration
    config = QAAgentConfig()
    config.model_provider = "openai"
    config.repo_path = tempfile.mkdtemp()
    config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_error_recovery_tests")
    os.makedirs(config.output_directory, exist_ok=True)

    # Create a function for testing
    function = Function(
        name="divide_numbers",
        code="def divide_numbers(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
        file_path=os.path.join(config.repo_path, "sample_module", "utils.py"),
        start_line=25,
        end_line=29,
        docstring="Divide a by b and return the result.",
        parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
        return_type="float",
        dependencies=[],
        complexity=2,
        cognitive_complexity=2,
        last_modified="",
        call_frequency=0,
    )

    # Create a mock test
    initial_test = GeneratedTest(
        function=function,
        test_code="import pytest\nfrom sample_module.utils import divide_numbers\n\ndef test_divide_numbers():\n    # This test will fail\n    assert divide_numbers(5, 2) == 3.0\n",
        test_file_path=os.path.join(config.output_directory, "test_divide_numbers.py"),
        imports=["pytest", "sample_module.utils.divide_numbers"],
        mocks=[],
        fixtures=[],
        test_functions=["test_divide_numbers"],
        test_classes=[],
    )

    # Run the workflow
    workflow = QAWorkflow(config)

    # Override the required methods to return minimal working state
    workflow._identify_functions = lambda: {
        "functions": [function],
        "current_function_index": 0,
        "current_function": function,
        "status": "Identified functions",
    }

    workflow._get_function_context = lambda state: {
        "functions": [function],
        "current_function_index": 0,
        "current_function": function,
        "context_files": [],
        "status": "Got context",
    }

    workflow._generate_test = lambda state: {
        "functions": [function],
        "current_function_index": 0,
        "current_function": function,
        "context_files": [],
        "generated_test": initial_test,
        "status": "Generated test",
    }

    # Simple implementations of routing functions
    workflow._route_after_validation = lambda state: "next"
    workflow._route_after_fix = lambda state: "validate"
    workflow._route_after_next = lambda state: "finish"

    print("\n===== Testing QAWorkflow.run() with mock implementation =====")

    try:
        # Step by step debugging of the components
        workflow_graph = workflow.workflow
        print(f"Created workflow graph: {type(workflow_graph)}")

        compiled_workflow = workflow_graph.compile()
        print(f"Compiled workflow: {type(compiled_workflow)}")

        # Directly invoke the workflow with a basic state
        state: Dict[str, Any] = {"messages": []}
        print(f"Initial state: {state}")

        final_state = workflow.run()
        print(f"Final state: {final_state}")

        print("SUCCESS: Workflow completed successfully")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
