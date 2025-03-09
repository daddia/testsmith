"""
Example script demonstrating the cognitive complexity analysis.

This script shows how to use the new cognitive complexity metrics to analyze
Python functions without running the full coverage analysis.
"""

import os
import sys
import ast
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qa_agent.parser import PythonCodeParser
from qa_agent.models import CodeFile, Function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Example function with complex control flow for testing
EXAMPLE_FUNCTION = """
def analyze_code_complexity(code, max_complexity=10):
    '''
    Analyze code complexity and provide recommendations.
    
    Args:
        code: The code to analyze
        max_complexity: Maximum allowable complexity
        
    Returns:
        Dictionary with analysis results
    '''
    results = {'complexity': 0, 'warnings': []}
    
    if not code or not isinstance(code, str):
        results['warnings'].append("Invalid code input")
        return results
        
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # Track complexity metrics
        complexity = 0
        nested_level = 0
        
        # Analyze each node in the AST
        for node in ast.walk(tree):
            # Check for control flow structures
            if isinstance(node, (ast.If, ast.For, ast.While)):
                complexity += 1
                
                # Check for nested structures
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.If, ast.For, ast.While)):
                        complexity += 2  # Nested structures add more complexity
                        
                        if nested_level > 2:
                            results['warnings'].append(f"Deeply nested control flow at line {node.lineno}")
            
            # Check for complex boolean operations
            elif isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):
                    complexity += len(node.values) - 1
                    
                    if len(node.values) > 3:
                        results['warnings'].append(f"Complex boolean expression at line {node.lineno}")
            
            # Check for exception handling
            elif isinstance(node, ast.Try):
                complexity += 1 + len(node.handlers)
                
        # Record the final complexity
        results['complexity'] = complexity
        
        # Add recommendations
        if complexity > max_complexity:
            results['warnings'].append(f"Overall complexity {complexity} exceeds maximum {max_complexity}")
            
            # Suggest refactoring strategies
            if complexity > max_complexity * 2:
                results['warnings'].append("Consider splitting into multiple functions")
            else:
                results['warnings'].append("Consider simplifying control flow")
                
        return results
        
    except SyntaxError as e:
        results['warnings'].append(f"Syntax error: {str(e)}")
        return results
    except Exception as e:
        results['warnings'].append(f"Analysis error: {str(e)}")
        return results
"""


def main():
    """Run the cognitive complexity analysis demo."""
    logger.info("Running cognitive complexity analysis demo")
    
    # Create a parser
    parser = PythonCodeParser()
    
    # Create a mock code file
    code_file = CodeFile(
        path="example_function.py",
        content=EXAMPLE_FUNCTION
    )
    
    # Extract function(s) from the code
    functions = parser.extract_functions(code_file)
    
    if not functions:
        logger.error("No functions found in the example code")
        return
        
    # Analyze the first function found
    function = functions[0]
    
    # Display the analysis results
    logger.info(f"Function: {function.name}")
    logger.info(f"Lines: {function.start_line}-{function.end_line}")
    logger.info(f"Cyclomatic complexity: {function.complexity}")
    logger.info(f"Cognitive complexity: {function.cognitive_complexity}")
    
    # Give an interpretation of the complexity scores
    logger.info("\nComplexity Score Interpretation:")
    
    # Cyclomatic complexity interpretation
    if function.complexity <= 5:
        cc_rating = "low"
        cc_desc = "simple with few paths"
    elif function.complexity <= 10:
        cc_rating = "moderate"
        cc_desc = "reasonably complex with multiple paths"
    else:
        cc_rating = "high"
        cc_desc = "highly complex with many execution paths"
    
    logger.info(f"Cyclomatic complexity: {cc_rating} - This function is {cc_desc}")
    
    # Cognitive complexity interpretation
    if function.cognitive_complexity <= 5:
        cogn_rating = "low"
        cogn_desc = "easy to understand"
    elif function.cognitive_complexity <= 15:
        cogn_rating = "moderate"
        cogn_desc = "moderately difficult to understand"
    else:
        cogn_rating = "high"
        cogn_desc = "difficult to understand and maintain"
    
    logger.info(f"Cognitive complexity: {cogn_rating} - This function is {cogn_desc}")
    
    # Provide recommendations based on complexity
    logger.info("\nRecommendations:")
    
    if function.complexity > 10 or function.cognitive_complexity > 15:
        logger.info("- Consider refactoring this function into smaller, more focused functions")
        logger.info("- Reduce nesting levels by extracting nested logic into separate functions")
        logger.info("- Simplify complex boolean expressions")
    elif function.complexity > 5 or function.cognitive_complexity > 8:
        logger.info("- Function could benefit from some simplification")
        logger.info("- Consider breaking down the most complex parts")
    else:
        logger.info("- Function complexity is reasonable, no significant changes needed")
    

if __name__ == "__main__":
    main()