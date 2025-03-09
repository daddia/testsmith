#!/usr/bin/env python3
"""
Demonstration of language-specific test file naming conventions.

This script demonstrates how the QA Agent's test generator creates test files
with naming conventions appropriate for each programming language.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def demonstrate_naming_conventions():
    """Demonstrate the test file naming conventions for each language."""
    try:
        # Import modules needed for demonstration
        from qa_agent.models import Function
        from qa_agent.test_generator import TestGenerator
        from qa_agent.config import QAAgentConfig
        
        # Create a demo output directory
        output_dir = os.path.join(project_root, 'examples', 'naming_demo')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize test generator with our configuration
        config = QAAgentConfig(
            model_provider="openai",
            model_name="o3-mini",
            api_key="demo-only",  # Not actually used in this demo
            output_directory=output_dir
        )
        
        # Create a test generator instance
        test_generator = TestGenerator(config)
        
        # Define example functions for each language
        examples = [
            {"path": "/path/to/example.py", "name": "python_function", "language": "Python"},
            {"path": "/path/to/example.js", "name": "javascript_function", "language": "JavaScript"},
            {"path": "/path/to/example.ts", "name": "typescript_function", "language": "TypeScript"},
            {"path": "/path/to/example.go", "name": "go_function", "language": "Go"},
            {"path": "/path/to/example.php", "name": "php_function", "language": "PHP"},
            {"path": "/path/to/example.sql", "name": "sql_function", "language": "SQL"},
            {"path": "/path/to/example.rb", "name": "ruby_function", "language": "Ruby (unsupported)"}
        ]
        
        logger.info("Demonstrating language-specific test file naming conventions:")
        print("\nLanguage-Specific Test File Naming Conventions")
        print("=" * 50)
        print(f"{'Language':<15} {'Source File':<20} {'Test File':<25}")
        print("-" * 50)
        
        # Generate test file paths for each example
        for example in examples:
            # Create a mock function object
            function = Function(
                name=example["name"],
                code=f"def {example['name']}():\n    pass",
                file_path=example["path"],
                start_line=1,
                end_line=2,
                docstring=f"Example {example['language']} function",
                parameters=[],
                return_type="",
                dependencies=[],
                complexity=1
            )
            
            # Get the test file path using our naming conventions
            test_file_path = test_generator._get_test_file_path(function)
            test_file_name = os.path.basename(test_file_path)
            
            # Print the results
            source_file = os.path.basename(example["path"])
            print(f"{example['language']:<15} {source_file:<20} {test_file_name:<25}")
            
        print("=" * 50)
        print("Note: The QA Agent automatically selects the appropriate naming")
        print("convention based on the file extension of the source file.\n")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")

if __name__ == "__main__":
    demonstrate_naming_conventions()