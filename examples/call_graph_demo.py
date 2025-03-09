"""
Example script demonstrating the call graph analysis.

This script shows how to use the new call graph analysis to identify critical
functions based on call frequency.
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qa_agent.call_graph_analyzer import CallGraphAnalyzer
from qa_agent.models import CodeFile, Function, FileType
from qa_agent.parser import PythonCodeParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Example functions with call relationships for testing
EXAMPLE_FUNCTIONS = {
    "utils.py": """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
    
def calculate_area(length, width):
    return multiply(length, width)
    
def calculate_perimeter(length, width):
    return add(add(length, width), add(length, width))
""",

    "math_operations.py": """
from utils import add, multiply

def square(x):
    return multiply(x, x)
    
def cube(x):
    return multiply(square(x), x)
    
def sum_squares(numbers):
    result = 0
    for number in numbers:
        result = add(result, square(number))
    return result
""",

    "geometry.py": """
from utils import calculate_area, calculate_perimeter

def rectangle_properties(length, width):
    area = calculate_area(length, width)
    perimeter = calculate_perimeter(length, width)
    return {
        "area": area,
        "perimeter": perimeter
    }
    
def square_properties(side):
    return rectangle_properties(side, side)
"""
}


def create_mock_functions():
    """Create mock functions for testing the call graph analyzer."""
    functions = []
    parser = PythonCodeParser()
    
    for file_name, content in EXAMPLE_FUNCTIONS.items():
        code_file = CodeFile(
            path=file_name,
            content=content,
            type=FileType.PYTHON
        )
        
        funcs = parser.extract_functions(code_file)
        functions.extend(funcs)
        
    return functions


def main():
    """Run the call graph analysis demo."""
    logger.info("Running call graph analysis demo")
    
    # Create sample repository path
    repo_path = os.path.abspath(os.path.dirname(__file__))
    
    # Create a call graph analyzer
    call_graph_analyzer = CallGraphAnalyzer(repo_path)
    
    # Create mock functions
    functions = create_mock_functions()
    
    logger.info(f"Created {len(functions)} mock functions for analysis")
    
    # Build call graph
    logger.info("Building call graph...")
    call_frequencies = call_graph_analyzer.build_call_graph(functions)
    
    # Display function call frequencies
    logger.info("\nFunction Call Frequencies:")
    logger.info("-" * 60)
    
    # Sort functions by call frequency
    sorted_frequencies = sorted(
        call_frequencies.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for func_id, frequency in sorted_frequencies:
        logger.info(f"{func_id}: {frequency}")
    
    # Identify critical functions
    logger.info("\nIdentifying critical functions...")
    critical_functions = call_graph_analyzer.get_critical_functions(
        functions, threshold=2
    )
    
    logger.info(f"Found {len(critical_functions)} critical functions")
    
    # Display critical functions
    if critical_functions:
        logger.info("\nCritical Functions:")
        logger.info("-" * 60)
        
        for i, func in enumerate(critical_functions, 1):
            # Get function's call frequency
            func_id = f"{os.path.relpath(func.file_path, repo_path)}:{func.name}"
            frequency = call_frequencies.get(func_id, 0)
            
            logger.info(f"{i}. {func.name} (calls: {frequency})")
            logger.info(f"   File: {func.file_path}")
            logger.info(f"   Lines: {func.start_line}-{func.end_line}")
            logger.info(f"   Complexity: {func.complexity}")
            logger.info("")
    

if __name__ == "__main__":
    main()