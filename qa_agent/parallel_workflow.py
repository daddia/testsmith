"""
Parallel Workflow module.

This module provides a parallel implementation of the QA workflow using
the task queue system.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from qa_agent.agents import (
    CodeAnalysisAgent,
    ConsoleAnalysisAgent,
    TestGenerationAgent,
    TestValidationAgent,
)
from qa_agent.config import QAAgentConfig
from qa_agent.error_recovery import CircuitBreaker, CircuitBreakerOpenError, ErrorHandler
from qa_agent.models import CodeFile, Function, GeneratedTest, TestResult
from qa_agent.task_queue import TaskQueue, TaskResult, TestTask, create_task_queue
from qa_agent.utils.logging import get_logger, log_exception, log_function_call, log_function_result

# Initialize logger for this module
logger = get_logger(__name__)


class ParallelQAWorkflow:
    """Parallel QA workflow for test generation and validation."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the parallel QA workflow.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize agents
        self.code_analysis_agent: CodeAnalysisAgent = CodeAnalysisAgent(config)
        self.test_generation_agent: TestGenerationAgent = TestGenerationAgent(config)
        self.test_validation_agent: TestValidationAgent = TestValidationAgent(config)
        self.console_analysis_agent: ConsoleAnalysisAgent = ConsoleAnalysisAgent(config)

        # Task queue will be initialized when we have functions
        self.task_queue = None

        # Initialize error handling components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,  # Open circuit after 5 consecutive failures
            recovery_timeout=60,  # Wait 60 seconds before trying again (half-open)
            half_open_max_calls=2,  # Allow 2 calls in half-open state
        )

        self.error_handler = ErrorHandler(
            max_retries=3, backoff_factor=1.5, circuit_breaker=self.circuit_breaker
        )

        # Register protected operations
        self.circuit_breaker.register_operation("generate_test")
        self.circuit_breaker.register_operation("validate_test")
        self.circuit_breaker.register_operation("fix_test")

        # Statistics
        self.start_time = None
        self.end_time = None
        self.stats: Dict[str, Any] = {
            "total_functions": 0,
            "success_count": 0,
            "failure_count": 0,
            "execution_time": 0.0,
            "average_time_per_function": 0.0,
            "circuit_breaker_trips": 0,
        }

        log_function_call(
            logger,
            "__init__",
            ("ParallelQAWorkflow",),
            {
                "parallel": config.parallel_execution,
                "max_workers": config.max_workers,
                "incremental": config.incremental_testing,
                "error_handling": "enhanced",
                "circuit_breaker": "enabled",
            },
        )

    def _identify_functions(self) -> List[Function]:
        """
        Identify critical functions with poor test coverage.

        Returns:
            List of critical functions
        """
        logger.info("Identifying critical functions...")

        # Identify critical functions
        functions = self.code_analysis_agent.identify_critical_functions()

        if not functions:
            logger.warning("No critical functions found")
            functions = self.code_analysis_agent.get_uncovered_functions()

            if not functions:
                logger.warning("No uncovered functions found")
                return []

        logger.info(f"Found {len(functions)} functions to test")
        return functions

    def _get_function_contexts(self, functions: List[Function]) -> Dict[str, List[CodeFile]]:
        """
        Get context for multiple functions.

        Args:
            functions: List of functions

        Returns:
            Dictionary mapping function file paths to context files
        """
        logger.info(f"Getting context for {len(functions)} functions...")

        context_map = {}
        for idx, function in enumerate(functions):
            logger.info(f"Getting context for function {idx+1}/{len(functions)}: {function.name}")
            context_files = self.code_analysis_agent.get_function_context(function)
            context_map[function.file_path] = context_files

        return context_map

    def _generate_and_validate_test(
        self, function: Function, context_files: List[CodeFile]
    ) -> GeneratedTest:
        """
        Generate a test for a function with enhanced error handling.

        Args:
            function: The function to test
            context_files: Related context files

        Returns:
            GeneratedTest object
        """
        # Use the error handler with circuit breaker
        try:
            generated_test = self.error_handler.execute_with_retry(
                self.test_generation_agent.generate_test,
                function,
                context_files,
                error_message=f"Failed to generate test for function: {function.name}",
                operation_name="generate_test",
                context={"function_name": function.name, "file_path": function.file_path},
            )
            return generated_test
        except CircuitBreakerOpenError as e:
            # If circuit breaker is open, log and create a minimal test
            logger.warning(
                f"Circuit breaker prevented test generation for {function.name}: {str(e)}"
            )
            self.stats["circuit_breaker_trips"] = self.stats.get("circuit_breaker_trips", 0) + 1

            # Create a minimal placeholder test to allow the workflow to continue
            # Generate a standardized test file path since we can't access internal method
            function_dir = os.path.dirname(function.file_path)
            function_name = os.path.basename(function.file_path)
            test_dir = os.path.join(function_dir, "tests") if function_dir else "tests"
            # Default to Python-style naming convention for safety
            test_file_path = os.path.join(test_dir, f"test_{function_name}")

            minimal_test = GeneratedTest(
                function=function,
                test_code=f"# Test generation blocked by circuit breaker\n# {str(e)}\n",
                test_file_path=test_file_path,
                imports=[],
                metadata={"circuit_breaker_error": str(e)},
            )
            return minimal_test

    def _validate_test(self, test: GeneratedTest) -> TestResult:
        """
        Validate a test with enhanced error handling.

        Args:
            test: The test to validate

        Returns:
            TestResult object
        """
        # If test was created as a placeholder due to circuit breaker, return a failed result
        if test.metadata and "circuit_breaker_error" in test.metadata:
            return TestResult(
                success=False,
                test_file=test.test_file_path,
                target_function=test.function.name,
                error_message=f"Test validation skipped: {test.metadata['circuit_breaker_error']}",
                output="Circuit breaker prevented test validation",
                execution_time=0.0,
                coverage=0.0,
            )

        # Use the error handler with circuit breaker
        try:
            return self.error_handler.execute_with_retry(
                self.test_validation_agent.validate_test,
                test,
                error_message=f"Failed to validate test for function: {test.function.name}",
                operation_name="validate_test",
                context={"function_name": test.function.name, "file_path": test.function.file_path},
            )
        except CircuitBreakerOpenError as e:
            # If circuit breaker is open, log and return a failed result
            logger.warning(
                f"Circuit breaker prevented test validation for {test.function.name}: {str(e)}"
            )
            self.stats["circuit_breaker_trips"] = self.stats.get("circuit_breaker_trips", 0) + 1

            return TestResult(
                success=False,
                test_file=test.test_file_path,
                target_function=test.function.name,
                error_message=f"Test validation blocked by circuit breaker: {str(e)}",
                output="Circuit breaker prevented test validation",
                execution_time=0.0,
                coverage=0.0,
            )

    def _fix_test(self, test: "GeneratedTest", test_result: "TestResult") -> "GeneratedTest":
        """
        Fix a failing test with enhanced error handling.

        Args:
            test: The failing test
            test_result: The test result

        Returns:
            Fixed GeneratedTest object
        """
        # Use the error handler with circuit breaker
        try:
            return self.error_handler.execute_with_retry(
                self.test_validation_agent.fix_test,
                test,
                test_result,
                error_message=f"Failed to fix test for function: {test.function.name}",
                operation_name="fix_test",
                context={"function_name": test.function.name, "file_path": test.function.file_path},
            )
        except CircuitBreakerOpenError as e:
            # If circuit breaker is open, log and return the original test
            logger.warning(
                f"Circuit breaker prevented test fixing for {test.function.name}: {str(e)}"
            )
            self.stats["circuit_breaker_trips"] = self.stats.get("circuit_breaker_trips", 0) + 1

            # Add a comment to the test code about the circuit breaker
            test_with_comment = GeneratedTest(
                function=test.function,
                test_code=test.test_code
                + f"\n\n# Test fixing blocked by circuit breaker: {str(e)}\n",
                test_file_path=test.test_file_path,
                imports=test.imports,
                metadata={"circuit_breaker_error": str(e)},
            )
            return test_with_comment

    def _process_results(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Process task results to generate a final report.

        Args:
            results: List of task results

        Returns:
            Final state dictionary
        """
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count

        functions = [r.task.function for r in results]

        # Calculate statistics
        self.stats["total_functions"] = len(results)
        self.stats["success_count"] = success_count
        self.stats["failure_count"] = failure_count

        if self.end_time and self.start_time:
            execution_time = (self.end_time - self.start_time).total_seconds()
            # Use a properly typed dictionary that accepts any values
            stats_dict: Dict[str, Any] = dict(self.stats)
            stats_dict["execution_time"] = execution_time
            if len(results) > 0:
                avg_time = execution_time / len(results)
                stats_dict["average_time_per_function"] = avg_time
            self.stats = stats_dict

        # Save results to JSON file
        results_file = os.path.join(self.config.output_directory, "test_results.json")
        try:
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "stats": self.stats,
                        "results": [
                            {
                                "function_name": r.task.function.name,
                                "file_path": r.task.function.file_path,
                                "success": r.success,
                                "test_file": (
                                    r.generated_test.test_file_path if r.generated_test else None
                                ),
                                "error": r.error
                                or (r.test_result.error_message if r.test_result else None),
                            }
                            for r in results
                        ],
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.exception(f"Error saving results to {results_file}: {str(e)}")

        # Return final state in the format expected by the CLI
        return {
            "functions": functions,
            "success_count": success_count,
            "failure_count": failure_count,
            "status": f"Completed: {success_count} successful, {failure_count} failed",
            "stats": self.stats,
        }

    def _fix_failing_tests(self, results: List[TaskResult]) -> List[TaskResult]:
        """
        Attempt to fix failing tests with enhanced error handling.

        Args:
            results: List of task results

        Returns:
            Updated list of task results
        """
        # Filter for failing tests with generated tests
        # Skip tests that were blocked by circuit breaker
        failing_tests = [
            r
            for r in results
            if not r.success
            and r.generated_test
            and r.test_result
            and not (
                r.generated_test.metadata and "circuit_breaker_error" in r.generated_test.metadata
            )
        ]

        if not failing_tests:
            return results

        logger.info(f"Attempting to fix {len(failing_tests)} failing tests...")

        # Try to fix each failing test with circuit breaker protection
        circuit_breaker_trips = 0
        for result in failing_tests:
            try:
                logger.info(f"Fixing test for function: {result.task.function.name}")

                # Wrap the fix and validate operations in circuit breaker protection
                try:
                    # Use error handler to fix the test only if both test and result are present
                    if result.generated_test is not None and result.test_result is not None:
                        fixed_test = self._fix_test(result.generated_test, result.test_result)
                    else:
                        logger.warning(
                            f"Skipping test fixing for {result.task.function.name}: missing test or result"
                        )
                        continue

                    # Use error handler to validate the fixed test
                    fixed_result = self._validate_test(fixed_test)

                    # Update the task result
                    result.generated_test = fixed_test
                    result.test_result = fixed_result
                    result.success = fixed_result.success

                    if fixed_result.success:
                        logger.info(
                            f"Successfully fixed test for function: {result.task.function.name}"
                        )
                    else:
                        logger.warning(
                            f"Failed to fix test for function: {result.task.function.name}"
                        )

                except CircuitBreakerOpenError as e:
                    circuit_breaker_trips += 1
                    result.error = f"Circuit breaker prevented test fixing: {str(e)}"
                    logger.warning(
                        f"Circuit breaker prevented fixing test for {result.task.function.name}: {str(e)}"
                    )

            except Exception as e:
                logger.exception(f"Error fixing test for function: {result.task.function.name}")
                result.error = f"Error fixing test: {str(e)}"

        # Count how many tests were fixed
        fixed_count = sum(1 for r in failing_tests if r.success)
        logger.info(f"Fixed {fixed_count} of {len(failing_tests)} failing tests")

        if circuit_breaker_trips > 0:
            logger.warning(f"Circuit breaker prevented fixing {circuit_breaker_trips} tests")
            self.stats["circuit_breaker_trips"] = (
                self.stats.get("circuit_breaker_trips", 0) + circuit_breaker_trips
            )

        return results

    def run(self) -> Dict[str, Any]:
        """
        Run the parallel QA workflow.

        Returns:
            Final state dictionary
        """
        self.start_time = datetime.now()
        logger.info("Starting parallel QA workflow")

        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_directory, exist_ok=True)

            # Step 1: Identify functions to test
            functions = self._identify_functions()
            if not functions:
                logger.warning("No functions to test found")
                self.end_time = datetime.now()
                return {
                    "functions": [],
                    "success_count": 0,
                    "failure_count": 0,
                    "status": "No functions to test found",
                }

            # Step 2: Get context for all functions
            context_map = self._get_function_contexts(functions)

            # Step 3: Create and populate task queue
            self.task_queue = create_task_queue(self.config, functions, context_map)

            # Step 4: Process all tasks
            results = self.task_queue.process_tasks(
                self._generate_and_validate_test, self._validate_test
            )

            # Step 5: Try to fix failing tests if configured
            if self.config.parallel_execution:
                results = self._fix_failing_tests(results)

            # Step 6: Process results
            self.end_time = datetime.now()
            final_state = self._process_results(results)

            logger.info("Parallel QA workflow finished successfully")

            execution_time = (self.end_time - self.start_time).total_seconds()
            logger.info(f"Total execution time: {execution_time:.2f}s")

            return final_state

        except Exception as e:
            self.end_time = datetime.now()
            log_exception(logger, "run", e)
            logger.error(f"Error running parallel QA workflow: {str(e)}")
            return {
                "functions": [],
                "success_count": 0,
                "failure_count": 0,
                "status": f"Error: {str(e)}",
            }
