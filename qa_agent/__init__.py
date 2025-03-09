"""
QA Agent Package

A Python-based QA agent for automatically identifying, generating, and validating unit tests to improve code quality and test coverage.
"""

__version__ = "0.1.0"

# Export error handling components for easy access
from qa_agent.error_recovery import (
    ErrorHandler,
    CircuitBreaker,
    Checkpoint,
    QAAgentError,
    CircuitBreakerOpenError,
    recover_from_error,
    get_diagnostic_info,
    truncate_context,
)
