#!/usr/bin/env python3
"""
Structured Logging Utility

This module provides a unified structured logging interface for the QA Agent,
combining Python's standard logging with structlog for enhanced log processing
and formatting.
"""

import datetime
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, TextIO, Union, cast

import structlog
from structlog.processors import StackInfoRenderer, TimeStamper
from structlog.stdlib import BoundLogger, LoggerFactory

# Configure default log level from environment variable or default to INFO
DEFAULT_LOG_LEVEL = os.environ.get("QA_AGENT_LOG_LEVEL", "INFO").upper()

# Configure log format based on environment (development vs production)
ENVIRONMENT = os.environ.get("QA_AGENT_ENV", "development").lower()

# Define log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(
    log_level: str = DEFAULT_LOG_LEVEL,
    env: str = ENVIRONMENT,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the logging system for the entire QA Agent.

    Args:
        log_level: The log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        env: The environment (development or production)
        log_file: Optional path to a log file to write logs to
    """
    # Convert string log level to logging constant
    level = LOG_LEVELS.get(log_level, logging.INFO)

    # Configure handlers
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(cast(logging.Handler, file_handler))

    # Configure stdlib logging
    logging.basicConfig(
        level=level,
        format="%(message)s",  # We'll let structlog handle formatting
        handlers=handlers,
    )

    # Configure processor pipeline based on environment
    # Type annotation needed for mypy to recognize the correct processor types
    processors: List[
        Callable[
            [Any, str, MutableMapping[str, Any]],
            Union[Mapping[str, Any], str, bytes, bytearray, tuple],
        ]
    ] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    ]

    # Add pretty formatting for development environment
    if env == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON formatting for production
        processors.extend(
            [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(sort_keys=True),
            ]
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> BoundLogger:
    """
    Get a structured logger for the given name.

    Args:
        name: The name of the logger, typically __name__

    Returns:
        A structured logger instance
    """
    # This ensures the underlying stdlib logger has the right name
    logger = logging.getLogger(name)
    # Return a wrapped structlog logger
    return cast(BoundLogger, structlog.wrap_logger(logger))


# Add convenience functions for common patterns


def add_context_to_logger(logger: BoundLogger, **kwargs: Any) -> BoundLogger:
    """
    Add permanent context to a logger.

    Args:
        logger: The logger to add context to
        **kwargs: Key-value pairs to add to the logger context

    Returns:
        A new logger with the added context
    """
    return logger.bind(**kwargs)


def log_function_call(
    logger: BoundLogger,
    function_name: str,
    args: tuple = (),
    kwargs: Dict[str, Any] = {},
) -> None:
    """
    Log a function call with its arguments.

    Args:
        logger: The logger to use
        function_name: Name of the function being called
        args: Positional arguments to the function (optional)
        kwargs: Keyword arguments to the function (optional)
    """
    args_str = repr(args) if args else "[]"
    kwargs_str = repr(kwargs) if kwargs else "{}"
    logger.info(f"{function_name} called", args=args_str, kwargs=kwargs_str)


def log_function_result(
    logger: BoundLogger,
    function_name: str,
    result: Any,
    execution_time: Optional[float] = None,
) -> None:
    """
    Log a function result and optionally its execution time.

    Args:
        logger: The logger to use
        function_name: Name of the function that returned the result
        result: The function result (will be converted to string)
        execution_time: Optional execution time in seconds
    """
    if execution_time is not None:
        logger.info(
            f"{function_name} returned",
            result=str(result),
            execution_time=f"{execution_time:.3f}s",
        )
    else:
        logger.info(f"{function_name} returned", result=str(result))


def log_exception(
    logger: BoundLogger,
    function_name: str,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an exception with additional context.

    Args:
        logger: The logger to use
        function_name: Name of the function where the exception occurred
        exception: The exception instance
        context: Optional additional context as key-value pairs
    """
    log_context = {"function": function_name, "exception_type": type(exception).__name__}
    if context:
        log_context.update(context)

    logger.exception(f"Exception in {function_name}", **log_context)


# Action logging functions for standardized message formats
def log_opened(logger: BoundLogger, file_path: str) -> None:
    """
    Log when a file is opened.

    Args:
        logger: The logger to use
        file_path: Path to the file that was opened
    """
    logger.info("Opened", file_path=file_path)


def log_parsed(logger: BoundLogger, file_path: str) -> None:
    """
    Log when a file is parsed.

    Args:
        logger: The logger to use
        file_path: Path to the file that was parsed
    """
    logger.info("Parsed", file_path=file_path)


def log_redacted(logger: BoundLogger, file_path: str, function_name: Optional[str] = None) -> None:
    """
    Log when content is redacted for IP protection.

    Args:
        logger: The logger to use
        file_path: Path to the file that was redacted
        function_name: Optional name of the function that was redacted
    """
    if function_name:
        logger.info("Redacted", file_path=file_path, function_name=function_name)
    else:
        logger.info("Redacted", file_path=file_path)


def log_generated(logger: BoundLogger, file_path: str) -> None:
    """
    Log when a test file is generated.

    Args:
        logger: The logger to use
        file_path: Path to the generated file
    """
    logger.info("Generated", file_path=file_path)


def log_validated(
    logger: BoundLogger,
    file_path: str,
    success: bool = True,
    coverage: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log when a test file is validated.

    Args:
        logger: The logger to use
        file_path: Path to the validated file
        success: Whether validation was successful
        coverage: Optional test coverage percentage
        error: Optional error message if validation failed
    """
    context = {"file_path": file_path}
    if coverage is not None:
        context["coverage"] = f"{coverage:.2f}%"
    if error:
        context["error"] = error

    if success:
        logger.info("Validated", **context)
    else:
        logger.warning("Validation failed", **context)


def log_analyzed(
    logger: BoundLogger, target: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log when analysis is performed.

    Args:
        logger: The logger to use
        target: What was analyzed (e.g., "Console output", "Coverage report")
        details: Optional details about the analysis results
    """
    context = {"target": target}
    if details:
        context.update(details)

    logger.info("Analyzed", **context)


def log_edited(logger: BoundLogger, file_path: str, reason: Optional[str] = None) -> None:
    """
    Log when a file is edited.

    Args:
        logger: The logger to use
        file_path: Path to the edited file
        reason: Optional reason for editing
    """
    context = {"file_path": file_path}
    if reason:
        context["reason"] = reason

    logger.info("Edited", **context)


def log_executed(
    logger: BoundLogger, command: str, success: bool = True, exit_code: Optional[int] = None
) -> None:
    """
    Log when a command is executed.

    Args:
        logger: The logger to use
        command: The command that was executed
        success: Whether execution was successful
        exit_code: Optional exit code of the command
    """
    context = {"command": command}
    if exit_code is not None:
        context["exit_code"] = str(exit_code)

    if success:
        logger.info("Executed", **context)
    else:
        logger.warning("Execution failed", **context)


# Configure default logging
configure_logging()
