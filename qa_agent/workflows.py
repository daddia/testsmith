"""
Workflows module.

This module defines the workflows for the QA agent using LangGraph.
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from qa_agent.agents import (
    CodeAnalysisAgent,
    ConsoleAnalysisAgent,
    TestGenerationAgent,
    TestValidationAgent,
)
from qa_agent.config import QAAgentConfig
from qa_agent.console_reader import analyze_console_output
from qa_agent.error_recovery import Checkpoint, ErrorHandler, get_diagnostic_info, truncate_context
from qa_agent.models import CodeFile, Function, GeneratedTest, TestResult
from qa_agent.utils.formatting import format_function_info
from qa_agent.utils.logging import get_logger, log_exception, log_function_call, log_function_result

# Use structured logger
logger = get_logger(__name__)


class WorkflowState(TypedDict, total=False):
    """State for QA workflow."""

    functions: List[Function]
    current_function_index: int
    current_function: Optional[Function]
    context_files: List[CodeFile]
    generated_test: Optional[GeneratedTest]
    test_result: Optional[TestResult]
    attempts: int
    success_count: int
    failure_count: int
    messages: List[Union[HumanMessage, AIMessage]]
    status: str
    progress_stats: List[Dict[str, Any]]

    # Recovery and error handling fields
    recovery_attempts: int
    last_checkpoint: str
    checkpoints: Dict[str, str]
    error: Dict[str, Any]

    # Statistics and diagnostics
    execution_time: float
    diagnosis: Dict[str, Any]


class QAWorkflow:
    """QA workflow for test generation and validation."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the QA workflow.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize agents
        self.code_analysis_agent = CodeAnalysisAgent(config)
        self.test_generation_agent = TestGenerationAgent(config)
        self.test_validation_agent = TestValidationAgent(config)
        self.console_analysis_agent = ConsoleAnalysisAgent(config)

        # Initialize LLM
        if config.model_provider == "openai":
            from pydantic import SecretStr

            self.llm = ChatOpenAI(
                temperature=0,
                model=config.model_name,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                api_key=SecretStr(config.api_key) if config.api_key else None,
            )
        else:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

        # Initialize workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build the workflow graph.

        Returns:
            StateGraph object
        """
        # Initialize the graph
        # Initialize the graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("identify_functions", self._identify_functions)
        workflow.add_node("get_function_context", self._get_function_context)
        workflow.add_node("generate_test", self._generate_test)
        workflow.add_node("validate_test", self._validate_test)
        workflow.add_node("fix_test", self._fix_test)
        workflow.add_node("next_function", self._next_function)
        workflow.add_node("finish", self._finish)

        # Add edges
        workflow.add_edge("identify_functions", "get_function_context")
        workflow.add_edge("get_function_context", "generate_test")
        workflow.add_edge("generate_test", "validate_test")

        # Conditional edges from validate_test
        workflow.add_conditional_edges(
            "validate_test",
            self._route_after_validation,
            {"fix": "fix_test", "next": "next_function", "finish": "finish"},
        )

        # Conditional edges from fix_test
        workflow.add_conditional_edges(
            "fix_test",
            self._route_after_fix,
            {"validate": "validate_test", "next": "next_function"},
        )

        # Conditional edges from next_function
        workflow.add_conditional_edges(
            "next_function",
            self._route_after_next,
            {"context": "get_function_context", "finish": "finish"},
        )

        # Set the entry point
        workflow.set_entry_point("identify_functions")

        return workflow

    def _identify_functions(self, state: WorkflowState) -> WorkflowState:
        """
        Identify critical functions with poor test coverage.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        logger.info("Identifying critical functions...")
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize state if necessary
        if "functions" not in state:
            state["functions"] = []
        if "current_function_index" not in state:
            state["current_function_index"] = -1
        if "success_count" not in state:
            state["success_count"] = 0
        if "failure_count" not in state:
            state["failure_count"] = 0
        if "messages" not in state:
            state["messages"] = []
        if "checkpoints" not in state:
            state["checkpoints"] = {}

        # Check if we need to identify functions or if we're resuming from a failure
        # If we already have functions in the state and are resuming, use them
        if state.get("functions") and state.get("recovery_attempts", 0) > 0:
            logger.info(f"Resuming with {len(state['functions'])} previously identified functions")
            functions = state["functions"]
        else:
            # Start fresh - identify critical functions
            try:
                # Identify critical functions
                functions = self.code_analysis_agent.identify_critical_functions()

                if not functions:
                    logger.warning("No critical functions found")
                    functions = self.code_analysis_agent.get_uncovered_functions()

                    if not functions:
                        logger.warning("No uncovered functions found")
                        state["status"] = "No functions to test found"
                        return state

                logger.info(f"Found {len(functions)} functions to test")
            except Exception as e:
                # Handle errors during function identification
                error_msg = f"Error identifying functions: {str(e)}"
                logger.error(error_msg)
                log_exception(logger, "_identify_functions", e)

                # Create minimal state with error information
                state["status"] = f"Error: {error_msg}"
                state["error"] = {
                    "message": str(e),
                    "type": type(e).__name__,
                    "phase": "identify_functions",
                    "timestamp": datetime.now().isoformat(),
                }

                # Save error checkpoint
                error_checkpoint = checkpoint_manager.save(state, "error_identify")
                if error_checkpoint:
                    state["checkpoints"]["error_identify"] = error_checkpoint

                return state

        # Update state
        state["functions"] = functions
        if "current_function_index" not in state or state["current_function_index"] < 0:
            state["current_function_index"] = 0
            state["current_function"] = functions[0]

        state["status"] = f"Identified {len(functions)} functions to test"

        # Add message to state
        if not any(
            msg.content.startswith("Identified")
            for msg in state.get("messages", [])
            if isinstance(msg, HumanMessage)
        ):
            state["messages"].append(
                HumanMessage(
                    content=f"Identified {len(functions)} functions to test. Starting with {functions[0].name}."
                )
            )

        # Save checkpoint after function identification
        identify_checkpoint = checkpoint_manager.save(state, "after_identify")
        if identify_checkpoint:
            state["checkpoints"]["after_identify"] = identify_checkpoint
            state["last_checkpoint"] = identify_checkpoint

        return state

    def _get_function_context(self, state: WorkflowState) -> WorkflowState:
        """
        Get context for the current function.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize checkpoints dictionary if it doesn't exist
        if "checkpoints" not in state:
            state["checkpoints"] = {}

        current_function = state.get("current_function")

        if not current_function:
            logger.error("No current function in state")
            state["status"] = "Error: No current function"

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(state, "error_context_no_function")
            if error_checkpoint:
                state["checkpoints"]["error_context"] = error_checkpoint

            return state

        logger.info(f"Getting context for function: {current_function.name}")

        # Check if we're resuming and already have context
        if state.get("context_files") and state.get("recovery_attempts", 0) > 0:
            logger.info(
                f"Resuming with {len(state['context_files'])} previously gathered context files"
            )
            return state

        try:
            # Get context files
            context_files = self.code_analysis_agent.get_function_context(current_function)

            # Update state
            state["context_files"] = context_files
            state["status"] = f"Got context for {current_function.name}"

            # Add message to state if not already present
            if not any(
                msg.content.startswith(f"Got context for {current_function.name}")
                for msg in state.get("messages", [])
                if isinstance(msg, AIMessage)
            ):
                state["messages"].append(
                    AIMessage(
                        content=f"Got context for {current_function.name} with {len(context_files)} related files."
                    )
                )

            # Save checkpoint after getting context
            context_checkpoint = checkpoint_manager.save(state, f"context_{current_function.name}")
            if context_checkpoint:
                state["checkpoints"][f"context_{current_function.name}"] = context_checkpoint
                state["last_checkpoint"] = context_checkpoint

            return state

        except Exception as e:
            # Handle errors during context gathering
            error_msg = f"Error gathering context for {current_function.name}: {str(e)}"
            logger.error(error_msg)
            log_exception(logger, "_get_function_context", e)

            # Create error information
            state["status"] = f"Error: {error_msg}"
            state["error"] = {
                "message": str(e),
                "type": type(e).__name__,
                "phase": "get_function_context",
                "function_name": current_function.name,
                "timestamp": datetime.now().isoformat(),
            }

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(
                state, f"error_context_{current_function.name}"
            )
            if error_checkpoint:
                state["checkpoints"][f"error_context_{current_function.name}"] = error_checkpoint

            # Keep the workflow going by returning the state without context files
            # The next steps will need to handle the missing context files gracefully
            state["context_files"] = []
            return state

    def _generate_test(self, state: WorkflowState) -> WorkflowState:
        """
        Generate a test for the current function.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize checkpoints dictionary if it doesn't exist
        if "checkpoints" not in state:
            state["checkpoints"] = {}

        current_function = state.get("current_function")
        context_files = state.get("context_files", [])

        if not current_function:
            logger.error("No current function in state")
            state["status"] = "Error: No current function"

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(state, "error_generate_no_function")
            if error_checkpoint:
                state["checkpoints"]["error_generate"] = error_checkpoint

            return state

        # Check if we already have a generated test (for resuming after failure)
        if state.get("generated_test") and state.get("recovery_attempts", 0) > 0:
            logger.info(f"Resuming with previously generated test for {current_function.name}")
            return state

        logger.info(f"Generating test for function: {current_function.name}")

        try:
            # Check if context files are reasonable in size
            total_context_size = sum(
                len(getattr(cf, "content", "")) for cf in context_files if hasattr(cf, "content")
            )
            max_context_size = 50000  # Set a reasonable limit

            if total_context_size > max_context_size:
                logger.warning(
                    f"Large context size ({total_context_size} chars), truncating to avoid token limit issues"
                )
                # Sort by size and keep only essential context
                context_files = sorted(
                    context_files,
                    key=lambda cf: len(getattr(cf, "content", "")) if hasattr(cf, "content") else 0,
                )
                # Keep at most 5 files or truncate to reasonable size
                if len(context_files) > 5:
                    context_files = context_files[:5]
                    logger.info(f"Reduced context to {len(context_files)} files")

            # Generate test with potentially truncated context
            generated_test = self.test_generation_agent.generate_test(
                current_function, context_files
            )

            # Update state
            state["generated_test"] = generated_test
            state["attempts"] = 1
            state["status"] = f"Generated test for {current_function.name}"

            # Add message to state if not already present
            if not any(
                msg.content.startswith(f"Generated test for {current_function.name}")
                for msg in state.get("messages", [])
                if isinstance(msg, AIMessage)
            ):
                state["messages"].append(
                    AIMessage(
                        content=f"Generated test for {current_function.name} and saved to {generated_test.test_file_path}."
                    )
                )

            # Save checkpoint after generating test
            generate_checkpoint = checkpoint_manager.save(
                state, f"generate_{current_function.name}"
            )
            if generate_checkpoint:
                state["checkpoints"][f"generate_{current_function.name}"] = generate_checkpoint
                state["last_checkpoint"] = generate_checkpoint

            return state

        except Exception as e:
            # Handle errors during test generation
            error_msg = f"Error generating test for {current_function.name}: {str(e)}"
            logger.error(error_msg)
            log_exception(logger, "_generate_test", e)

            # Try to diagnose the error
            error_type = type(e).__name__
            context_size_error = "context_length_exceeded" in str(
                e
            ) or "maximum context length" in str(e)

            # Create error information
            state["status"] = f"Error: {error_msg}"
            state["error"] = {
                "message": str(e),
                "type": error_type,
                "phase": "generate_test",
                "function_name": current_function.name,
                "context_size_error": context_size_error,
                "timestamp": datetime.now().isoformat(),
            }

            # Handle context length errors specially
            if context_size_error:
                logger.warning("Context length exceeded, attempting with reduced context")
                try:
                    # Use minimal context approach - just the function itself
                    truncated_context = []
                    if len(context_files) > 0:
                        # Keep only the first/primary context file (likely containing the function itself)
                        truncated_context = [context_files[0]]

                    # Try again with minimal context
                    generated_test = self.test_generation_agent.generate_test(
                        current_function, truncated_context
                    )

                    # If successful, update state and continue
                    state["generated_test"] = generated_test
                    state["attempts"] = 1
                    state["status"] = (
                        f"Generated test for {current_function.name} (with reduced context)"
                    )
                    state["messages"].append(
                        AIMessage(
                            content=f"Generated test for {current_function.name} with reduced context due to token limits. "
                            f"Saved to {generated_test.test_file_path}."
                        )
                    )

                    # Save recovery checkpoint
                    recovery_checkpoint = checkpoint_manager.save(
                        state, f"recovery_generate_{current_function.name}"
                    )
                    if recovery_checkpoint:
                        state["checkpoints"][
                            f"recovery_generate_{current_function.name}"
                        ] = recovery_checkpoint
                        state["last_checkpoint"] = recovery_checkpoint

                    return state

                except Exception as recovery_error:
                    # Recovery attempt failed
                    logger.error(f"Recovery attempt failed: {str(recovery_error)}")
                    state["error"]["recovery_error"] = str(recovery_error)

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(
                state, f"error_generate_{current_function.name}"
            )
            if error_checkpoint:
                state["checkpoints"][f"error_generate_{current_function.name}"] = error_checkpoint

            # Even if generation failed, return the state for the workflow to handle
            return state

    def _validate_test(self, state: WorkflowState) -> WorkflowState:
        """
        Validate the generated test.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize checkpoints dictionary if it doesn't exist
        if "checkpoints" not in state:
            state["checkpoints"] = {}

        generated_test = state.get("generated_test")

        if not generated_test:
            logger.error("No generated test in state")
            state["status"] = "Error: No generated test"

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(state, "error_validate_no_test")
            if error_checkpoint:
                state["checkpoints"]["error_validate"] = error_checkpoint

            return state

        # Check if we already have a test result (for resuming after failure)
        if state.get("test_result") and state.get("recovery_attempts", 0) > 0:
            logger.info(
                f"Resuming with previously validated test for {generated_test.function.name}"
            )
            return state

        logger.info(f"Validating test for function: {generated_test.function.name}")

        try:
            # Validate test
            test_result = self.test_validation_agent.validate_test(generated_test)

            # Store console data separately for improved error analysis
            console_data = test_result.output if test_result else None

            # Analyze console output regardless of test success (for coverage info)
            if console_data:
                try:
                    # Use console analysis agent to get more insights
                    console_analysis = self.console_analysis_agent.analyze_console_output(
                        console_data
                    )
                    error_messages = self.console_analysis_agent.extract_error_messages(
                        console_data
                    )
                    test_coverage = self.console_analysis_agent.get_test_coverage(console_data)
                    test_failures = self.console_analysis_agent.identify_test_failures(console_data)

                    # Log insights from console analysis
                    if error_messages:
                        logger.info(f"Console analysis found {len(error_messages)} error messages")
                        for error in error_messages[:3]:  # Log up to 3 error messages
                            logger.info(f"Error: {error}")

                    if test_failures:
                        logger.info(
                            f"Console analysis identified {len(test_failures)} failing tests"
                        )
                        for test in test_failures[:3]:  # Log up to 3 failing tests
                            logger.info(f"Failing test: {test}")

                    if test_coverage is not None:
                        logger.info(f"Test coverage from console: {test_coverage}%")
                        # Update test result with coverage from console if not already set
                        if test_result.coverage is None:
                            test_result.coverage = test_coverage

                    # Enhanced error message from console analysis for failing tests
                    if (
                        not test_result.success
                        and error_messages
                        and (not test_result.error_message or len(test_result.error_message) < 20)
                    ):
                        test_result.error_message = error_messages[0]

                    # Add detailed analysis to the test result
                    if not hasattr(test_result, "console_analysis"):
                        test_result.console_analysis = {}

                    test_result.console_analysis = {
                        "error_messages": error_messages,
                        "test_failures": test_failures,
                        "test_coverage": test_coverage,
                    }

                except Exception as e:
                    logger.warning(f"Error analyzing console output: {str(e)}")
                    log_exception(
                        logger,
                        "_validate_test",
                        e,
                        {"phase": "console_analysis", "function": generated_test.function.name},
                    )

            # Update state
            state["test_result"] = test_result
            state["status"] = f"Validated test for {generated_test.function.name}"

            # Construct message with detailed information
            if test_result.success:
                message_content = f"Test for {generated_test.function.name} passed!"
                if test_result.coverage is not None:
                    message_content += f" Coverage: {test_result.coverage}%"
                state["messages"].append(AIMessage(content=message_content))
            else:
                message_content = f"Test for {generated_test.function.name} failed."

                # Add detailed error information from console analysis if available
                error_messages_found = False
                if test_result and console_data and hasattr(test_result, "console_analysis"):
                    try:
                        error_messages = test_result.console_analysis.get("error_messages", [])
                        if error_messages:
                            error_messages_found = True
                            message_content += "\n\nErrors detected:"
                            for idx, error in enumerate(error_messages[:3]):
                                message_content += f"\n{idx+1}. {error}"
                    except Exception:
                        # Ignore exceptions in error message extraction for messaging
                        pass

                if not error_messages_found:
                    message_content += f"\nError: {test_result.error_message or 'Unknown error'}"

                state["messages"].append(AIMessage(content=message_content))

            # Save validation checkpoint
            validate_checkpoint = checkpoint_manager.save(
                state, f"validate_{generated_test.function.name}"
            )
            if validate_checkpoint:
                state["checkpoints"][
                    f"validate_{generated_test.function.name}"
                ] = validate_checkpoint
                state["last_checkpoint"] = validate_checkpoint

            return state

        except Exception as e:
            # Handle errors during test validation
            error_msg = f"Error validating test for {generated_test.function.name}: {str(e)}"
            logger.error(error_msg)
            log_exception(logger, "_validate_test", e)

            # Create error information
            state["status"] = f"Error: {error_msg}"
            state["error"] = {
                "message": str(e),
                "type": type(e).__name__,
                "phase": "validate_test",
                "function_name": generated_test.function.name,
                "timestamp": datetime.now().isoformat(),
            }

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(
                state, f"error_validate_{generated_test.function.name}"
            )
            if error_checkpoint:
                state["checkpoints"][
                    f"error_validate_{generated_test.function.name}"
                ] = error_checkpoint

            # Create a minimal test result to allow workflow to continue
            state["test_result"] = TestResult(
                success=False,
                target_function=generated_test.function.name,
                test_file=generated_test.test_file_path,
                output=f"Error during validation: {str(e)}\n{traceback.format_exc()}",
                error_message=f"Validation error: {str(e)}",
                coverage=0.0,
            )

            return state

    def _fix_test(self, state: WorkflowState) -> WorkflowState:
        """
        Fix a failing test.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize checkpoints dictionary if it doesn't exist
        if "checkpoints" not in state:
            state["checkpoints"] = {}

        generated_test = state.get("generated_test")
        test_result = state.get("test_result")
        attempts = state.get("attempts", 1)

        if not generated_test or not test_result:
            logger.error("Missing generated test or test result in state")
            state["status"] = "Error: Missing data for test fix"

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(state, "error_fix_missing_data")
            if error_checkpoint:
                state["checkpoints"]["error_fix"] = error_checkpoint

            return state

        # Check if we've already tried too many times
        max_fix_attempts = 3  # Maximum number of fix attempts per test
        if attempts > max_fix_attempts:
            logger.warning(f"Too many fix attempts ({attempts}), moving to next function")
            state["status"] = f"Too many fix attempts for {generated_test.function.name}"
            state["messages"].append(
                AIMessage(
                    content=f"After {attempts} attempts, the test for {generated_test.function.name} still fails. "
                    f"Moving to the next function."
                )
            )

            # Save max_attempts checkpoint
            max_attempts_checkpoint = checkpoint_manager.save(
                state, f"max_attempts_{generated_test.function.name}"
            )
            if max_attempts_checkpoint:
                state["checkpoints"][
                    f"max_attempts_{generated_test.function.name}"
                ] = max_attempts_checkpoint

            return state

        logger.info(
            f"Fixing test for function: {generated_test.function.name} (attempt {attempts}/{max_fix_attempts})"
        )

        try:
            # Get console output for enhanced error analysis
            console_data = test_result.output
            fix_suggestions = {"test_failed": False, "suggestions": []}

            if console_data:
                # Analyze console output for additional insights
                try:
                    console_analysis = self.console_analysis_agent.analyze_console_output(
                        console_data
                    )
                    error_messages = self.console_analysis_agent.extract_error_messages(
                        console_data
                    )
                    test_failures = self.console_analysis_agent.identify_test_failures(console_data)

                    # Log insights from console analysis
                    if error_messages:
                        logger.info(f"Console analysis found {len(error_messages)} error messages")
                        for error in error_messages[:3]:  # Log up to 3 error messages
                            logger.info(f"Error: {error}")

                    if test_failures:
                        logger.info(
                            f"Console analysis identified {len(test_failures)} failing tests"
                        )
                        for test in test_failures[:3]:  # Log up to 3 failing tests
                            logger.info(f"Failing test: {test}")

                    # Get fix suggestions from console analysis
                    fix_suggestions = self.console_analysis_agent.suggest_test_fixes(
                        console_data, generated_test
                    )
                    if fix_suggestions.get("test_failed", False):
                        logger.info("Console analysis provided fix suggestions")
                        for suggestion in fix_suggestions.get("suggestions", [])[
                            :3
                        ]:  # Log up to 3 suggestions
                            logger.info(f"Fix suggestion: {suggestion}")
                except Exception as e:
                    logger.warning(f"Error analyzing console output: {str(e)}")
                    log_exception(
                        logger,
                        "_fix_test",
                        e,
                        {"phase": "console_analysis", "function": generated_test.function.name},
                    )

            # Fix test using the standard approach enhanced with console insights
            fixed_test = self.test_validation_agent.fix_test(generated_test, test_result)

            # Update state
            state["generated_test"] = fixed_test
            state["attempts"] = attempts + 1
            state["status"] = (
                f"Fixed test for {generated_test.function.name} (attempt {attempts}/{max_fix_attempts})"
            )

            # Add message to state with enhanced information from console analysis
            message_content = f"Fixed test for {generated_test.function.name} and saved to {fixed_test.test_file_path}."
            fix_suggestions_found = False

            # Safely access fix suggestions if they exist
            if console_data and fix_suggestions:
                try:
                    # Use the suggestions if available
                    suggestions = fix_suggestions.get("suggestions", [])
                    if suggestions:
                        fix_suggestions_found = True
                        message_content += (
                            "\n\nBased on console analysis, the following improvements were made:"
                        )
                        for idx, suggestion in enumerate(suggestions[:3]):
                            message_content += f"\n{idx+1}. {suggestion}"
                except Exception as e:
                    # Ignore any errors in building the message
                    logger.warning(f"Error processing fix suggestions for message: {str(e)}")
                    pass

            state["messages"].append(AIMessage(content=message_content))

            # Save fix checkpoint
            fix_checkpoint = checkpoint_manager.save(
                state, f"fix_{generated_test.function.name}_{attempts}"
            )
            if fix_checkpoint:
                state["checkpoints"][
                    f"fix_{generated_test.function.name}_{attempts}"
                ] = fix_checkpoint
                state["last_checkpoint"] = fix_checkpoint

            return state

        except Exception as e:
            # Handle errors during test fixing
            error_msg = f"Error fixing test for {generated_test.function.name}: {str(e)}"
            logger.error(error_msg)
            log_exception(logger, "_fix_test", e)

            # Create error information
            state["status"] = f"Error: {error_msg}"
            state["error"] = {
                "message": str(e),
                "type": type(e).__name__,
                "phase": "fix_test",
                "function_name": generated_test.function.name,
                "attempts": attempts,
                "timestamp": datetime.now().isoformat(),
            }

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(
                state, f"error_fix_{generated_test.function.name}"
            )
            if error_checkpoint:
                state["checkpoints"][f"error_fix_{generated_test.function.name}"] = error_checkpoint

            # Increment attempts and continue
            state["attempts"] = attempts + 1

            # If we've hit max attempts, indicate it's time to move on
            if state["attempts"] > max_fix_attempts:
                state["status"] = (
                    f"Error fixing test for {generated_test.function.name}, max attempts reached"
                )

            return state

    def _next_function(self, state: WorkflowState) -> WorkflowState:
        """
        Move to the next function.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize checkpoints dictionary if it doesn't exist
        if "checkpoints" not in state:
            state["checkpoints"] = {}

        functions = state.get("functions", [])
        current_index = state.get("current_function_index", 0)
        success_count = state.get("success_count", 0)
        failure_count = state.get("failure_count", 0)
        test_result = state.get("test_result")

        try:
            # Update success/failure count
            if test_result:
                if test_result.success:
                    success_count += 1
                    logger.info(f"Test succeeded for function at index {current_index}")
                else:
                    failure_count += 1
                    logger.info(f"Test failed for function at index {current_index}")

            # Move to the next function
            next_index = current_index + 1

            # Record progress statistics
            progress_stats = {
                "processed_count": next_index,
                "total_count": len(functions),
                "success_count": success_count,
                "failure_count": failure_count,
                "progress_percentage": int((next_index / len(functions)) * 100) if functions else 0,
                "timestamp": datetime.now().isoformat(),
            }

            if "progress_stats" not in state:
                state["progress_stats"] = []

            # Keep track of progress over time
            state["progress_stats"].append(progress_stats)

            # Check if we've processed all functions
            if next_index >= len(functions):
                logger.info("All functions processed")

                # Update state
                state["current_function_index"] = next_index
                state["current_function"] = None
                state["success_count"] = success_count
                state["failure_count"] = failure_count
                state["status"] = "All functions processed"

                # Add message to state
                state["messages"].append(
                    AIMessage(
                        content=f"All {len(functions)} functions processed. "
                        f"Tests generated successfully: {success_count}. "
                        f"Tests failed: {failure_count}."
                    )
                )

                # Save milestone checkpoint
                all_done_checkpoint = checkpoint_manager.save(state, "all_functions_completed")
                if all_done_checkpoint:
                    state["checkpoints"]["all_functions_completed"] = all_done_checkpoint
                    state["last_checkpoint"] = all_done_checkpoint
            else:
                # Move to the next function
                next_function = functions[next_index]

                logger.info(
                    f"Moving to next function: {next_function.name} ({next_index+1}/{len(functions)})"
                )

                # Update state
                state["current_function_index"] = next_index
                state["current_function"] = next_function
                state["success_count"] = success_count
                state["failure_count"] = failure_count
                state["generated_test"] = None
                state["test_result"] = None
                state["attempts"] = 1  # Reset attempts for the new function
                state["status"] = f"Moving to next function: {next_function.name}"

                # Add message to state
                state["messages"].append(
                    AIMessage(
                        content=f"Moving to next function: {next_function.name} ({next_index+1}/{len(functions)})."
                    )
                )

                # Save checkpoint for the transition to next function
                next_function_checkpoint = checkpoint_manager.save(
                    state, f"next_function_{next_index}"
                )
                if next_function_checkpoint:
                    state["checkpoints"][f"next_function_{next_index}"] = next_function_checkpoint
                    state["last_checkpoint"] = next_function_checkpoint

            return state

        except Exception as e:
            # Handle errors during function transition
            error_msg = f"Error moving to next function: {str(e)}"
            logger.error(error_msg)
            log_exception(logger, "_next_function", e)

            # Create error information
            state["status"] = f"Error: {error_msg}"
            state["error"] = {
                "message": str(e),
                "type": type(e).__name__,
                "phase": "next_function",
                "current_index": current_index,
                "next_index": current_index + 1 if current_index < len(functions) else None,
                "timestamp": datetime.now().isoformat(),
            }

            # Save error checkpoint
            error_checkpoint = checkpoint_manager.save(
                state, f"error_next_function_{current_index}"
            )
            if error_checkpoint:
                state["checkpoints"][f"error_next_function_{current_index}"] = error_checkpoint

            # Try to recover by still updating the state as best we can
            state["success_count"] = success_count
            state["failure_count"] = failure_count

            # If possible, still try to move to the next function
            try:
                next_index = current_index + 1
                if next_index < len(functions):
                    state["current_function_index"] = next_index
                    state["current_function"] = functions[next_index]
                    state["generated_test"] = None
                    state["test_result"] = None
                    state["attempts"] = 1
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {str(recovery_error)}")
                # If recovery fails, at least ensure state is consistent
                state["current_function_index"] = current_index

            return state

    def _finish(self, state: WorkflowState) -> WorkflowState:
        """
        Finish the workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        functions = state.get("functions", [])
        success_count = state.get("success_count", 0)
        failure_count = state.get("failure_count", 0)

        logger.info("Workflow finished")

        # Generate summary
        summary = (
            f"QA Workflow completed.\n"
            f"Total functions processed: {len(functions)}\n"
            f"Tests generated successfully: {success_count}\n"
            f"Tests failed: {failure_count}\n"
        )

        # Save summary to file
        summary_path = os.path.join(self.config.output_directory, "qa_summary.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        with open(summary_path, "w") as f:
            f.write(summary)

        logger.info(f"Summary saved to {summary_path}")

        # Update state
        state["status"] = "Workflow finished"

        # Add message to state
        state["messages"].append(
            AIMessage(content=f"Workflow finished. Summary saved to {summary_path}.")
        )

        return state

    def _route_after_validation(self, state: WorkflowState) -> Literal["fix", "next", "finish"]:
        """
        Decide what to do after test validation.

        Args:
            state: Current workflow state

        Returns:
            Next node to execute
        """
        test_result = state.get("test_result")
        attempts = state.get("attempts", 1)
        current_function_index = state.get("current_function_index", 0)
        functions = state.get("functions", [])

        # Enhanced error handling - check for invalid state conditions
        if not test_result:
            logger.error("No test result in state")
            return "next"

        # Safety check to prevent infinite loops
        if not functions:
            logger.warning("No functions in state, finishing workflow")
            return "finish"

        # Check if we're at the last function
        is_last_function = current_function_index >= len(functions) - 1

        # If the test passed or max attempts reached, move to next function or finish
        if test_result.success:
            logger.info("Test passed, moving to next function")
            if is_last_function:
                logger.info("Last function processed successfully, finishing")
                return "finish"
            return "next"

        # If we've tried to fix the test too many times, move to next function or finish
        if attempts >= 3:
            logger.warning(f"Test failed after {attempts} attempts, moving on")
            if is_last_function:
                logger.info("Last function processed, finishing")
                return "finish"
            return "next"

        # Otherwise, try to fix the test
        logger.info("Test failed, attempting to fix")
        return "fix"

    def _route_after_fix(self, state: WorkflowState) -> Literal["validate", "next"]:
        """
        Decide what to do after test fix.

        Args:
            state: Current workflow state

        Returns:
            Next node to execute
        """
        attempts = state.get("attempts", 1)
        current_function_index = state.get("current_function_index", 0)
        functions = state.get("functions", [])

        # Safety check to prevent infinite loops
        if attempts >= 10:  # Absolute maximum attempts as a failsafe
            logger.error(
                f"Maximum fix attempts reached ({attempts}), forcing move to next function"
            )
            return "next"

        # Standard check for reasonable number of attempts
        if attempts >= 3:
            logger.warning(f"Too many fix attempts ({attempts}), moving to next function")
            return "next"

        # Otherwise, validate the fixed test
        logger.info(f"Validating fixed test (attempt {attempts})")
        return "validate"

    def _route_after_next(self, state: WorkflowState) -> Literal["context", "finish"]:
        """
        Decide what to do after moving to the next function.

        Args:
            state: Current workflow state

        Returns:
            Next node to execute
        """
        current_function = state.get("current_function")
        current_function_index = state.get("current_function_index", 0)
        functions = state.get("functions", [])

        # If we have no current function or we've reached the end, we're done
        if not current_function or current_function_index >= len(functions):
            logger.info("No more functions to process, finishing")
            return "finish"

        # Safety check to prevent infinite loops - this is critical
        # If we have a current function but somehow got stuck in a loop
        function_max_index = len(functions) - 1
        if current_function_index > function_max_index:
            logger.warning(
                f"Function index {current_function_index} exceeds max index {function_max_index}, finishing"
            )
            return "finish"

        # Otherwise, get context for the next function
        logger.info(
            f"Getting context for next function: {current_function.name} (index {current_function_index})"
        )
        return "context"

    def run(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the QA workflow.

        Args:
            resume_from_checkpoint: Path to a checkpoint file to resume from

        Returns:
            Final workflow state
        """
        # Create output directories if they don't exist
        os.makedirs(self.config.output_directory, exist_ok=True)
        checkpoint_dir = os.path.join(self.config.output_directory, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize checkpoint manager
        checkpoint_manager = Checkpoint(checkpoint_dir, "qa_workflow")

        # Initialize error handler for retries
        error_handler = ErrorHandler(max_retries=3)

        # Track workflow start time
        workflow_start_time = datetime.now()
        logger.info("Starting QA workflow", timestamp=workflow_start_time.isoformat())

        # Compile the workflow
        # Note: recursion_limit parameter is no longer supported in the current langgraph version
        compiled_workflow = self.workflow.compile()

        # Initialize state from checkpoint if resuming
        if resume_from_checkpoint:
            try:
                # Try to load the checkpoint file directly if specified
                logger.info(f"Attempting to resume from checkpoint: {resume_from_checkpoint}")
                loaded_state = checkpoint_manager.load(resume_from_checkpoint)

                if loaded_state:
                    logger.info("Successfully resumed from checkpoint")
                    # Use the loaded state as our starting point
                    state = loaded_state

                    # Check where we left off based on the status
                    if state.get("status", "").startswith("Generated test"):
                        # If we had generated a test but not validated it, we should restart from validation
                        if "generated_test" in state and state["generated_test"]:
                            logger.info("Resuming from test validation")
                            state["status"] = "Resuming from test validation"

                    # Truncate context if needed (to avoid token limit errors)
                    state = truncate_context(state)
                else:
                    logger.warning("Could not load checkpoint, starting fresh")
                    state = {"messages": []}
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                state = {"messages": []}
        else:
            # Start fresh
            state = {"messages": []}

        # Run the workflow with error handling and checkpointing
        try:
            # Add recovery information to state
            state["recovery_attempts"] = state.get("recovery_attempts", 0)
            state["last_checkpoint"] = state.get("last_checkpoint", None)

            # Check for previously interrupted workflow
            if state.get("recovery_attempts", 0) > 0:
                logger.info(f"Workflow previously interrupted {state['recovery_attempts']} times")

            # Add diagnostic info to help troubleshoot issues
            diagnostic_info = get_diagnostic_info()
            logger.info(
                "System diagnostic information",
                python_version=diagnostic_info.get("python", {}).get("version", "unknown"),
                platform=diagnostic_info.get("platform", "unknown"),
                memory=diagnostic_info.get("memory", {}),
            )

            # Save initial checkpoint
            if not resume_from_checkpoint:
                initial_checkpoint = checkpoint_manager.save(state, "initial")
                if initial_checkpoint:
                    state["last_checkpoint"] = initial_checkpoint

            # Use error handler to execute with retries
            def _run_workflow(state):
                """Execute workflow with the given state."""
                return compiled_workflow.invoke(state)

            # Execute workflow with retries for recoverable errors
            final_state = error_handler.execute_with_retry(
                _run_workflow,
                state,
                error_message="Failed to execute workflow",
                # Specify which exceptions to retry
                recoverable_exceptions=[
                    ValueError,  # Common in data processing
                    KeyError,  # Missing key in state dict
                    ConnectionError,  # Network issues with LLM
                    TimeoutError,  # Timeouts with LLM or other services
                    # Add more exception types as needed
                ],
            )

            # Calculate statistics
            workflow_end_time = datetime.now()
            execution_time = (workflow_end_time - workflow_start_time).total_seconds()

            logger.info(
                "QA workflow finished successfully",
                execution_time=f"{execution_time:.2f}s",
                success_count=final_state.get("success_count", 0),
                failure_count=final_state.get("failure_count", 0),
            )

            # Save final error logs if any
            if error_handler.error_logs:
                error_log_file = error_handler.save_error_logs(checkpoint_dir, "qa_workflow_errors")
                if error_log_file:
                    logger.info(f"Error logs saved to {error_log_file}")

                # Analyze error patterns
                error_analysis = error_handler.analyze_error_trend()
                if error_analysis["error_count"] > 0:
                    logger.info(
                        "Error analysis",
                        error_count=error_analysis["error_count"],
                        common_errors=error_analysis["common_errors"],
                    )

            # Save final successful state
            final_checkpoint = checkpoint_manager.save(final_state, "final")
            if final_checkpoint:
                logger.info(f"Final state saved to checkpoint: {final_checkpoint}")

            # Clean up old checkpoints to save space
            checkpoint_manager.clean()

            return final_state

        except Exception as e:
            # Handle unrecoverable errors
            error_detail = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Unrecoverable error running QA workflow: {error_detail}")
            logger.debug(f"Stack trace:\n{stack_trace}")

            # Check if we have a token limit error with OpenAI
            context_length_error = (
                "context_length_exceeded" in error_detail
                or "maximum context length" in error_detail
            )
            if context_length_error:
                logger.error(
                    "Token limit exceeded. Attempting to reduce context size and continue."
                )

                # If we still have a valid state, try to truncate and save it
                if "state" in locals() and isinstance(state, dict):
                    # Increment recovery attempts counter
                    state["recovery_attempts"] = state.get("recovery_attempts", 0) + 1

                    # Aggressive truncation for context length errors
                    truncated_state = truncate_context(
                        state, max_tokens=8000
                    )  # Reduce to half of standard limit

                    # Save truncated state as recovery checkpoint
                    recovery_checkpoint = checkpoint_manager.save(truncated_state, "recovery")
                    if recovery_checkpoint:
                        logger.info(
                            f"Truncated state saved to recovery checkpoint: {recovery_checkpoint}"
                        )

                    # Include recovery instructions in final state
                    truncated_state["status"] = (
                        f"Error: {error_detail[:100]}... Resume with recovery checkpoint."
                    )
                    truncated_state["recovery_checkpoint"] = recovery_checkpoint
                    return truncated_state

            # For other errors, create a basic error state
            workflow_end_time = datetime.now()
            execution_time = (workflow_end_time - workflow_start_time).total_seconds()

            error_state = {
                "functions": state.get("functions", []),
                "current_function_index": state.get("current_function_index", 0),
                "current_function": state.get("current_function", None),
                "context_files": [],  # Don't include large context in error state
                "generated_test": state.get("generated_test", None),
                "test_result": state.get("test_result", None),
                "attempts": state.get("attempts", 0),
                "success_count": state.get("success_count", 0),
                "failure_count": state.get("failure_count", 0),
                "messages": (
                    state.get("messages", [])[-5:] if state.get("messages") else []
                ),  # Keep only last 5 messages
                "status": f"Error: {error_detail[:200]}",
                "error": {
                    "message": error_detail,
                    "type": type(e).__name__,
                    "time": workflow_end_time.isoformat(),
                    "execution_time": execution_time,
                },
            }

            # Save error state to checkpoint
            error_checkpoint = checkpoint_manager.save(error_state, "error")
            if error_checkpoint:
                logger.info(f"Error state saved to checkpoint: {error_checkpoint}")
                error_state["recovery_checkpoint"] = error_checkpoint

            return error_state
