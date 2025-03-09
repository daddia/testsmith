"""
File utilities for QA Agent.

This module contains functions for working with files and code.
"""

import logging
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def copy_test_to_repo(test_file_path: str, repo_path: str, test_dir: str = "tests") -> str:
    """
    Copy a generated test file to the repository.

    Args:
        test_file_path: Path to the test file
        repo_path: Path to the repository
        test_dir: Name of the test directory in the repository

    Returns:
        Path to the copied test file
    """
    # Get the filename
    filename = os.path.basename(test_file_path)

    # Construct the destination path
    dest_path = os.path.join(repo_path, test_dir, filename)

    # Create the test directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Copy the file
    shutil.copy2(test_file_path, dest_path)

    logger.info(f"Copied test file to {dest_path}")

    return dest_path


def clean_code_for_llm(code: str) -> str:
    """
    Clean code for LLM processing.

    Args:
        code: Code to clean

    Returns:
        Cleaned code
    """
    # Remove any long inline comments or docstrings
    code = re.sub(r'""".*?"""', '"""..."""', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "'''...'''", code, flags=re.DOTALL)

    # Remove long lists or dicts
    code = re.sub(r"\[([^]]{100,})\]", "[...]", code, flags=re.DOTALL)
    code = re.sub(r"\{([^}]{100,})\}", "{...}", code, flags=re.DOTALL)

    return code


def create_temporary_test_file(code: str) -> str:
    """
    Create a temporary test file.

    Args:
        code: Test code

    Returns:
        Path to the temporary file
    """
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "temp_test.py")

    with open(temp_file, "w") as f:
        f.write(code)

    return temp_file
