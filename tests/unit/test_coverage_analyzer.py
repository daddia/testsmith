import pytest
from unittest.mock import patch, MagicMock
from qa_agent.coverage_analyzer import CoverageAnalyzer, Function, CodeFile

# Mock logger to avoid actual logging during tests
mock_logger = MagicMock()

# Mock CodeParserFactory and its methods
mock_parser = MagicMock()
mock_parser.extract_functions.return_value = [
    Function(name="func1", code="def func1(): pass", file_path="file1.py", start_line=1, end_line=10, complexity=5),
    Function(name="func2", code="def func2(): pass", file_path="file2.py", start_line=11, end_line=20, complexity=3)
]

mock_code_parser_factory = MagicMock()
mock_code_parser_factory.get_parser.return_value = mock_parser

# Mock os.walk to simulate file system
# Making it a single entry to generate exactly 2 functions (one for file1.py)
mock_os_walk = [
    (".", [], ["file1.py"])
]

# Mock open to simulate file reading
mock_open = MagicMock()
mock_open.return_value.__enter__.return_value.read.return_value = "def func1(): pass\ndef func2(): pass"

@pytest.fixture
def coverage_analyzer():
    # Create an instance of CoverageAnalyzer with mocked repo path
    analyzer = CoverageAnalyzer(repo_path="/mock/repo/path")
    return analyzer

def test_get_functions_coverage_no_coverage_data(coverage_analyzer):
    # Test case where coverage data is empty
    with patch('os.walk', return_value=mock_os_walk), \
         patch('builtins.open', mock_open), \
         patch('qa_agent.coverage_analyzer.CodeParserFactory', mock_code_parser_factory):
        
        uncovered, covered = coverage_analyzer._get_functions_coverage({'files': {}})
        
        assert len(uncovered) == 2
        assert len(covered) == 0
        assert uncovered[0].name == "func1"
        assert uncovered[1].name == "func2"

def test_get_functions_coverage_with_coverage_data(coverage_analyzer):
    # Test case where coverage data is provided
    # Set up mock to have different behavior for file1.py and file2.py
    def mock_open_func(file_path, *args, **kwargs):
        mock = MagicMock()
        if file_path == 'file1.py':
            mock.__enter__.return_value.read.return_value = "def func1(): pass"
        else:
            mock.__enter__.return_value.read.return_value = "def func2(): pass"
        return mock
    
    # Create mock for extract_functions that returns different functions for each file
    def mock_extract_functions(code_file):
        if code_file.path == 'file1.py':
            return [Function(name="func1", code="def func1(): pass", file_path="file1.py", start_line=1, end_line=10, complexity=5)]
        else:
            return [Function(name="func2", code="def func2(): pass", file_path="file2.py", start_line=1, end_line=10, complexity=3)]
    
    # Apply mock_extract_functions to our mock_parser
    mock_parser.extract_functions.side_effect = mock_extract_functions
    
    coverage_data = {
        'files': {
            'file1.py': {'missing_lines': [1, 2, 3]},
            'file2.py': {'missing_lines': []}
        }
    }
    
    with patch('builtins.open', mock_open_func), \
         patch('qa_agent.coverage_analyzer.CodeParserFactory', mock_code_parser_factory):
        
        uncovered, covered = coverage_analyzer._get_functions_coverage(coverage_data)
        
        assert len(uncovered) == 1
        assert len(covered) == 1
        assert uncovered[0].name == "func1"
        assert covered[0].name == "func2"
        
    # Reset the mock for other tests
    mock_parser.extract_functions.side_effect = None

def test_get_functions_coverage_with_exception_handling(coverage_analyzer):
    # Test case to ensure exceptions are handled
    mock_error_open = MagicMock()
    mock_error_open.side_effect = Exception("File read error")
    mock_log_exception = MagicMock()
    
    with patch('builtins.open', mock_error_open), \
         patch('qa_agent.coverage_analyzer.CodeParserFactory', mock_code_parser_factory), \
         patch('qa_agent.coverage_analyzer.log_exception', mock_log_exception):
        
        uncovered, covered = coverage_analyzer._get_functions_coverage({'files': {'file1.py': {}}})
        
        assert len(uncovered) == 0
        assert len(covered) == 0
        mock_log_exception.assert_called()

def test_get_functions_coverage_with_no_parser(coverage_analyzer):
    # Test case where no parser is available
    mock_code_parser_factory.get_parser.return_value = None
    
    with patch('builtins.open', mock_open), \
         patch('qa_agent.coverage_analyzer.CodeParserFactory', mock_code_parser_factory):
        
        uncovered, covered = coverage_analyzer._get_functions_coverage({'files': {'file1.py': {}}})
        
        assert len(uncovered) == 0
        assert len(covered) == 0