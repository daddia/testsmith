"""
Enhanced Test Coverage Analysis Demo

This script demonstrates the enhanced test coverage analysis that integrates:
1. Function complexity metrics (cyclomatic and cognitive complexity)
2. Call graph analysis for identifying frequently called functions
3. Git history integration to focus on recently modified code

The combined approach provides a more intelligent way to prioritize
which functions should be tested first.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import random

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qa_agent.parser import PythonCodeParser
from qa_agent.coverage_analyzer import CoverageAnalyzer
from qa_agent.call_graph_analyzer import CallGraphAnalyzer
from qa_agent.git_history_analyzer import GitHistoryAnalyzer
from qa_agent.models import Function, CodeFile, FileType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_sample_functions(num_functions=20):
    """Create sample functions with varying complexity for demonstration."""
    functions = []
    
    # Create a variety of files to simulate a real codebase
    files = ["utils.py", "core.py", "api.py", "models.py", "views.py"]
    
    for i in range(num_functions):
        # Randomize attributes to create diverse function profiles
        file_path = random.choice(files)
        name = f"function_{i}"
        complexity = random.randint(1, 25)  # Cyclomatic complexity
        cognitive_complexity = random.randint(5, 100)  # Cognitive complexity
        
        # Create some correlation between the complexities (not perfect, but realistic)
        if complexity > 15:
            cognitive_complexity = max(cognitive_complexity, 50)
        
        # Randomize last_modified dates (some recent, some older)
        days_ago = random.randint(1, 30)
        last_modified = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        # Randomly assign call frequency (how often the function is called)
        call_frequency = random.randint(0, 50)
        
        # Create dummy code that somewhat reflects complexity
        code_lines = [f"def {name}(param1, param2):"]
        for j in range(min(complexity, 10)):  # Add some if statements
            code_lines.append(f"    if param{j % 2 + 1} > {j}:")
            code_lines.append(f"        result = param{j % 2 + 1} * {j + 1}")
        code_lines.append("    return result")
        
        function = Function(
            name=name,
            code="\n".join(code_lines),
            file_path=file_path,
            start_line=1,
            end_line=len(code_lines),
            complexity=complexity,
            cognitive_complexity=cognitive_complexity,
            last_modified=last_modified,
            call_frequency=call_frequency
        )
        
        functions.append(function)
    
    return functions


def prioritize_functions(functions: List[Function], days_threshold: int = 7) -> List[Function]:
    """
    Prioritize functions for testing based on multiple factors:
    1. Complexity (both cyclomatic and cognitive)
    2. Call frequency (from call graph analysis)
    3. Recent modifications (from git history)
    
    Args:
        functions: List of functions to prioritize
        days_threshold: Number of days to consider a change "recent"
        
    Returns:
        Prioritized list of functions
    """
    logger.info("Prioritizing functions for testing...")
    
    # Calculate a priority score for each function
    now = datetime.now()
    scored_functions = []
    
    for function in functions:
        # Base score starts at 0
        score = 0
        
        # Factor 1: Code Complexity
        # Weight cyclomatic complexity (0-25 range in our sample)
        if function.complexity:
            if function.complexity > 15:
                score += 30  # High complexity
            elif function.complexity > 8:
                score += 20  # Medium complexity
            else:
                score += 10  # Low complexity
        
        # Weight cognitive complexity (0-100 range in our sample)
        if function.cognitive_complexity:
            if function.cognitive_complexity > 50:
                score += 30  # High cognitive complexity
            elif function.cognitive_complexity > 20:
                score += 20  # Medium cognitive complexity
            else:
                score += 10  # Low cognitive complexity
        
        # Factor 2: Call Frequency
        # Functions called more often are more critical
        if function.call_frequency:
            if function.call_frequency > 20:
                score += 25  # Frequently called
            elif function.call_frequency > 10:
                score += 15  # Moderately called
            elif function.call_frequency > 0:
                score += 5   # Rarely called
        
        # Factor 3: Recent Modifications
        # Recently modified code is higher priority
        if function.last_modified:
            last_modified = datetime.fromisoformat(function.last_modified)
            days_since_modified = (now - last_modified).days
            
            if days_since_modified <= days_threshold:
                score += 40  # Modified very recently
            elif days_since_modified <= days_threshold * 2:
                score += 20  # Modified somewhat recently
            elif days_since_modified <= days_threshold * 3:
                score += 5   # Modified a while ago
                
        # Append function with its score for sorting
        scored_functions.append((function, score))
    
    # Sort by score in descending order
    scored_functions.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the functions in priority order
    prioritized_functions = [func for func, score in scored_functions]
    
    return prioritized_functions


def generate_test_recommendations(functions: List[Function], top_n: int = 5) -> List[Dict]:
    """
    Generate test recommendations for prioritized functions.
    
    Args:
        functions: Prioritized list of functions
        top_n: Number of top functions to recommend tests for
        
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    for i, function in enumerate(functions[:top_n]):
        # Determine recommendation reason based on attributes
        reasons = []
        
        if function.complexity and function.complexity > 15:
            reasons.append(f"high cyclomatic complexity ({function.complexity})")
        
        if function.cognitive_complexity and function.cognitive_complexity > 50:
            reasons.append(f"high cognitive complexity ({function.cognitive_complexity})")
        
        if function.call_frequency and function.call_frequency > 20:
            reasons.append(f"frequently called ({function.call_frequency} times)")
        
        if function.last_modified:
            last_modified = datetime.fromisoformat(function.last_modified)
            days_ago = (datetime.now() - last_modified).days
            if days_ago <= 7:
                reasons.append(f"recently modified ({days_ago} days ago)")
        
        # Create the recommendation
        recommendation = {
            "priority": i + 1,
            "function": function.name,
            "file": function.file_path,
            "reasons": reasons,
            "test_file": f"test_{function.file_path.replace('.py', '')}_{function.name}.py"
        }
        
        recommendations.append(recommendation)
    
    return recommendations


def main():
    """Run the enhanced test coverage analysis demo."""
    logger.info("Running enhanced test coverage analysis demo")
    
    # Create sample repository path
    repo_path = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"Analyzing repository at: {repo_path}")
    
    # Create sample functions
    functions = create_sample_functions(20)
    logger.info(f"Created {len(functions)} sample functions")
    
    # Prioritize functions
    prioritized_functions = prioritize_functions(functions)
    
    # Generate test recommendations
    recommendations = generate_test_recommendations(prioritized_functions, top_n=10)
    
    # Display results
    logger.info("\nTest Priorities (Top 10 functions to test):")
    logger.info("-" * 80)
    
    for rec in recommendations:
        logger.info(f"Priority {rec['priority']}: {rec['function']} in {rec['file']}")
        
        # Show reasons for prioritization
        reasons_text = ", ".join(rec["reasons"])
        logger.info(f"  Reasons: {reasons_text}")
        
        # Show recommendation
        logger.info(f"  Recommended test file: {rec['test_file']}")
        logger.info("")
    
    # Provide a summary of the prioritization factors used
    logger.info("\nPrioritization factors used:")
    logger.info("1. Function complexity (cyclomatic and cognitive complexity)")
    logger.info("2. Call frequency (how often the function is called by others)")
    logger.info("3. Recent modifications (when the function was last changed)")
    logger.info("")
    logger.info("The enhanced analysis prioritizes functions that:")
    logger.info("- Have high complexity (risk of bugs and difficulty to maintain)")
    logger.info("- Are called frequently (high impact if they fail)")
    logger.info("- Were recently modified (higher likelihood of new bugs)")


if __name__ == "__main__":
    main()