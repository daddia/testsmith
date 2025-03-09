#!/usr/bin/env python3
"""
QA Agent - Main entry point

This script runs the QA agent that automatically identifies, generates,
and validates unit tests to improve code quality.

This file is maintained for backward compatibility and direct script execution.
For normal usage, the CLI entry point provided by the package should be used.
"""

import sys
from qa_agent.cli import main as cli_main

if __name__ == "__main__":
    # Call the main function from qa_agent.cli
    sys.exit(cli_main())
