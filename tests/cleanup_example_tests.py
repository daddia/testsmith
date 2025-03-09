#!/usr/bin/env python3
"""
Cleanup and test generation script for example files.

This script cleans up the example test directories by removing all generated test files,
then generates new tests for the example source files using the updated test generation
code with proper language-specific naming conventions.
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root to the path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Define the base examples directory
EXAMPLES_DIR = os.path.join(project_root, "examples")


def cleanup_generated_tests():
    """Clean up all generated test files in the examples directory."""
    languages = ["go", "js", "php", "python", "sql"]

    for language in languages:
        lang_dir = os.path.join(EXAMPLES_DIR, language)
        if not os.path.exists(lang_dir):
            logger.warning(f"Language directory not found: {lang_dir}")
            continue

        generated_tests_dir = os.path.join(lang_dir, "generated_tests")
        if not os.path.exists(generated_tests_dir):
            logger.warning(f"Generated tests directory not found: {generated_tests_dir}")
            continue

        logger.info(f"Cleaning up generated tests for {language}...")

        # Remove all files in the directory (keeping the directory)
        for filename in os.listdir(generated_tests_dir):
            file_path = os.path.join(generated_tests_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Removed: {file_path}")


def generate_tests_for_example_files():
    """
    Generate tests for all example source files using our TestGenerator
    with the updated filename conventions.
    """
    try:
        # Import our test generator and related modules
        from qa_agent.config import QAAgentConfig
        from qa_agent.models import CodeFile, FileType, Function
        from qa_agent.test_generator import TestGenerator

        logger.info("Initializing TestGenerator with example configuration...")

        # Create a configuration for test generation
        config = QAAgentConfig(
            model_provider="openai",
            model_name="o3-mini",  # Using the more efficient model
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            repo_path=project_root,
            output_directory=project_root,
        )

        # Initialize the test generator
        test_generator = TestGenerator(config)

        # Define the languages and their example files
        languages = {
            "python": {"ext": ".py", "dir": "python"},
            "javascript": {"ext": ".js", "dir": "js"},
            "go": {"ext": ".go", "dir": "go"},
            "php": {"ext": ".php", "dir": "php"},
            "sql": {"ext": ".sql", "dir": "sql"},
        }

        # Process each language's example files
        for lang, info in languages.items():
            example_src_dir = os.path.join(EXAMPLES_DIR, info["dir"], "example_src")
            generated_tests_dir = os.path.join(EXAMPLES_DIR, info["dir"], "generated_tests")

            if not os.path.exists(example_src_dir):
                logger.warning(f"Example source directory not found: {example_src_dir}")
                continue

            # Make sure the generated tests directory exists
            os.makedirs(generated_tests_dir, exist_ok=True)

            # Process all example files
            for filename in os.listdir(example_src_dir):
                if filename.endswith(info["ext"]):
                    file_path = os.path.join(example_src_dir, filename)
                    process_example_file(test_generator, file_path, generated_tests_dir, lang)

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
    except Exception as e:
        logger.error(f"Error generating tests: {e}")


def process_example_file(test_generator, file_path, output_dir, language):
    """
    Process a single example file to generate a test using the TestGenerator.
    """
    from qa_agent.models import CodeFile, FileType, Function

    try:
        logger.info(f"Processing example file: {file_path}")

        # Read the file content
        with open(file_path, "r") as f:
            file_content = f.read()

        # Create a simple function model from the file
        # (In a real application, we would parse the file to extract functions)
        filename = os.path.basename(file_path)
        module_name = os.path.splitext(filename)[0]

        # Create a simple function with the whole file as the "function" for demonstration
        function = Function(
            name=module_name,
            code=file_content,
            file_path=file_path,
            start_line=1,
            end_line=len(file_content.split("\n")),
            docstring=f"Example {language} function",
            parameters=[],
            return_type="",
            dependencies=[],
            complexity=1,
        )

        # Create a code file object
        code_file = CodeFile(path=file_path, content=file_content)

        # Generate test using our test generator
        generated_test = test_generator.generate_test(function, [code_file])

        # Update the test file path to write to the correct output directory
        original_path = generated_test.test_file_path
        filename = os.path.basename(original_path)
        generated_test.test_file_path = os.path.join(output_dir, filename)

        # Save the test to file
        test_generator.save_test_to_file(generated_test)
        logger.info(f"Generated test: {generated_test.test_file_path}")

    except Exception as e:
        logger.error(f"Error processing example file {file_path}: {e}")


def main():
    """Main entry point for the script."""
    logger.info("Starting cleanup of example test files...")
    cleanup_generated_tests()
    logger.info("Cleanup completed!")

    logger.info("Starting test generation for example files...")
    generate_tests_for_example_files()
    logger.info("Test generation completed!")


if __name__ == "__main__":
    main()
