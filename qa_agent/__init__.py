"""
QA Agent Package

A Python-based QA agent for automatically identifying, generating, and validating unit tests to improve code quality and test coverage.
"""

__version__ = "0.1.0"

# Export error handling components for easy access
from qa_agent.error_recovery import (
    Checkpoint,
    CircuitBreaker,
    CircuitBreakerOpenError,
    ErrorHandler,
    QAAgentError,
    get_diagnostic_info,
    recover_from_error,
    truncate_context,
)
