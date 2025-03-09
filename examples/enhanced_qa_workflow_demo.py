"""
Enhanced QA Workflow Demo

This script demonstrates how the enhanced test coverage analysis
integrates with the existing QA Agent workflow.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qa_agent.models import Function, CodeFile, FileType, CoverageReport
from qa_agent.parser import PythonCodeParser
from qa_agent.call_graph_analyzer import CallGraphAnalyzer
from qa_agent.git_history_analyzer import GitHistoryAnalyzer
from qa_agent.coverage_analyzer import CoverageAnalyzer
from qa_agent.config import QAAgentConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class EnhancedQAWorkflow:
    """Enhanced QA workflow with improved test prioritization."""
    
    def __init__(self, config: QAAgentConfig):
        """
        Initialize the enhanced QA workflow.
        
        Args:
            config: QA Agent configuration
        """
        self.config = config
        self.repo_path = config.repo_path
        
        # Initialize analyzers
        self.coverage_analyzer = CoverageAnalyzer(self.repo_path, config.test_framework)
        self.call_graph_analyzer = CallGraphAnalyzer(self.repo_path)
        self.git_history_analyzer = GitHistoryAnalyzer(self.repo_path)
        
        # Parsing and results
        self.functions = []
        self.critical_functions = []
        
    def run(self) -> List[Function]:
        """
        Run the enhanced QA workflow.
        
        Returns:
            List of prioritized functions for testing
        """
        logger.info(f"Running enhanced QA workflow on {self.repo_path}")
        
        # Step 1: Gather all functions from the codebase
        self.functions = self._gather_functions()
        logger.info(f"Found {len(self.functions)} functions in the codebase")
        
        # Step 2: Build call graph for call frequency analysis
        self._analyze_call_frequencies()
        
        # Step 3: Analyze git history for recent changes
        self._analyze_git_history()
        
        # Step 4: Run test coverage analysis
        self._analyze_test_coverage()
        
        # Step 5: Prioritize functions for testing
        self.critical_functions = self._prioritize_functions()
        logger.info(f"Identified {len(self.critical_functions)} critical functions for testing")
        
        return self.critical_functions
    
    def _gather_functions(self) -> List[Function]:
        """
        Gather all functions from the codebase.
        
        Returns:
            List of functions
        """
        logger.info("Gathering functions from codebase...")
        
        functions = []
        code_files = self._get_code_files()
        
        for code_file in code_files:
            parser = PythonCodeParser()  # Simplified for demo, would use a factory
            
            try:
                file_functions = parser.extract_functions(code_file)
                functions.extend(file_functions)
                logger.debug(f"Extracted {len(file_functions)} functions from {code_file.path}")
            except Exception as e:
                logger.error(f"Error extracting functions from {code_file.path}: {str(e)}")
        
        return functions
    
    def _get_code_files(self) -> List[CodeFile]:
        """
        Get all code files from the repository.
        
        Returns:
            List of code files
        """
        # This is a simplified version for the demo
        # In the real implementation, this would walk the repo directory
        
        # Create a sample code file
        sample_content = """
def sample_function(a, b):
    result = 0
    if a > b:
        result = a - b
    else:
        result = a + b
    return result

def complex_function(items):
    result = []
    for item in items:
        if item > 0:
            for subitem in item:
                if subitem % 2 == 0:
                    result.append(subitem)
    return result
"""
        
        sample_file = CodeFile(
            path=os.path.join(self.repo_path, "sample.py"),
            content=sample_content,
            type=FileType.PYTHON
        )
        
        return [sample_file]
    
    def _analyze_call_frequencies(self) -> None:
        """Analyze call frequencies using the call graph analyzer."""
        logger.info("Analyzing function call frequencies...")
        
        call_frequencies = self.call_graph_analyzer.build_call_graph(self.functions)
        
        # Update function objects with call frequency data
        for function in self.functions:
            func_id = f"{function.file_path}:{function.name}"
            function.call_frequency = call_frequencies.get(func_id, 0)
    
    def _analyze_git_history(self) -> None:
        """Analyze git history for recent changes."""
        logger.info(f"Analyzing git history (last {self.config.changed_since_days} days)...")
        
        # Get modified files
        if not self.config.incremental_testing:
            logger.info("Skipping git history analysis (incremental testing disabled)")
            return
            
        modified_files = self.git_history_analyzer.get_modified_files(
            days=self.config.changed_since_days
        )
        
        # Update function objects with last modified information
        for function in self.functions:
            if function.file_path in modified_files:
                last_modified = self.git_history_analyzer.get_file_last_modified_date(
                    function.file_path
                )
                if last_modified:
                    function.last_modified = last_modified.isoformat()
    
    def _analyze_test_coverage(self) -> None:
        """Analyze test coverage and update function objects."""
        logger.info("Analyzing test coverage...")
        
        # Run coverage analysis
        coverage_report = self.coverage_analyzer.run_coverage_analysis()
        
        # Update function complexity information from the report
        if coverage_report:
            uncovered_functions = {
                f"{func.file_path}:{func.name}": func 
                for func in coverage_report.uncovered_functions
            }
            
            for function in self.functions:
                func_id = f"{function.file_path}:{function.name}"
                if func_id in uncovered_functions:
                    # Copy complexity data from coverage report
                    uncovered_func = uncovered_functions[func_id]
                    function.complexity = uncovered_func.complexity
                    function.cognitive_complexity = uncovered_func.cognitive_complexity
    
    def _prioritize_functions(self) -> List[Function]:
        """
        Prioritize functions for testing based on multiple criteria.
        
        Returns:
            List of prioritized functions
        """
        logger.info("Prioritizing functions for testing...")
        
        # Calculate priority score for each function
        scored_functions = []
        
        for function in self.functions:
            score = 0
            
            # Factor 1: Complexity
            if function.complexity and function.complexity > 10:
                score += min(function.complexity * 2, 50)
                
            if function.cognitive_complexity and function.cognitive_complexity > 15:
                score += min(function.cognitive_complexity, 50)
            
            # Factor 2: Call frequency
            if function.call_frequency:
                score += min(function.call_frequency * 3, 50)
            
            # Factor 3: Recent modification
            if function.last_modified:
                score += 40  # Recently modified files are important
            
            scored_functions.append((function, score))
        
        # Sort by score in descending order
        scored_functions.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N or those with scores above threshold
        threshold = 30  # Minimum score to be considered critical
        critical_functions = [
            func for func, score in scored_functions 
            if score >= threshold
        ]
        
        return critical_functions[:min(len(critical_functions), 10)]  # Top 10 at most


def create_sample_config():
    """Create a sample QA Agent configuration for the demo."""
    config = QAAgentConfig()
    config.repo_path = os.path.abspath(os.path.dirname(__file__))
    config.test_framework = "pytest"
    config.incremental_testing = True
    config.changed_since_days = 7
    return config


def display_test_recommendations(functions: List[Function]) -> None:
    """
    Display test recommendations for prioritized functions.
    
    Args:
        functions: List of prioritized functions
    """
    logger.info("\nTest Recommendations:")
    logger.info("-" * 80)
    
    for i, function in enumerate(functions, 1):
        logger.info(f"{i}. {function.name} ({function.file_path})")
        
        details = []
        if function.complexity:
            details.append(f"Cyclomatic complexity: {function.complexity}")
        if function.cognitive_complexity:
            details.append(f"Cognitive complexity: {function.cognitive_complexity}")
        if function.call_frequency:
            details.append(f"Call frequency: {function.call_frequency}")
        if function.last_modified:
            details.append(f"Recently modified: {function.last_modified}")
            
        logger.info(f"   Details: {', '.join(details)}")
        logger.info("")


def main():
    """Run the enhanced QA workflow demo."""
    logger.info("Running enhanced QA workflow demo")
    
    # Create configuration
    config = create_sample_config()
    
    # Create and run the workflow
    workflow = EnhancedQAWorkflow(config)
    critical_functions = workflow.run()
    
    # Display recommendations
    display_test_recommendations(critical_functions)
    
    logger.info("\nEnhanced QA Workflow Improvements:")
    logger.info("1. More intelligent test prioritization based on multiple factors")
    logger.info("2. Better use of limited testing resources by focusing on critical code")
    logger.info("3. Integration of cognitive complexity for more accurate risk assessment")
    logger.info("4. Call graph analysis to identify frequently used functions")
    logger.info("5. Git history integration to focus on recently modified code")


if __name__ == "__main__":
    main()