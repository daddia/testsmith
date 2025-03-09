# Corrected test code here

import pytest
from qa_agent.console_reader import ReposWorklowConsoleReader, IDNAError, IDNABidiError, InvalidCodepoint, InvalidCodepointContext
from typing import List, Dict, Any

@pytest.fixture
def sample_lines():
    """Fixture to provide sample console output lines."""
    return [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.8.5, pytest-6.2.1, py-1.10.0, pluggy-0.13.1",
        "rootdir: /path/to/project",
        "collected 3 items",
        "",
        "test_sample.py::test_passed PASSED",
        "test_sample.py::test_failed FAILED",
        "test_sample.py::test_skipped SKIPPED",
        "",
        "=========================== short test summary info ===========================",
        "FAILED test_sample.py::test_failed - AssertionError: assert 1 == 2",
        "============================== 1 failed, 1 passed, 1 skipped in 0.12s ===============================",
        "---------- coverage: platform linux, python 3.8.5-final-0 -----------",
        "Name                      Stmts   Miss  Cover",
        "---------------------------------------------",
        "test_sample.py               10      2    80%",
        "TOTAL                        10      2    80%",
    ]

def test_extract_pytest_results_basic(sample_lines):
    """Test basic extraction of pytest results."""
    reader = ReposWorklowConsoleReader()
    results = reader.extract_pytest_results(sample_lines)
    
    assert results['summary']['total'] == 3
    assert results['summary']['passed'] == 1
    assert results['summary']['failed'] == 1
    assert results['summary']['skipped'] == 1
    assert results['summary']['errors'] == 0
    assert results['coverage'] == 80
    assert len(results['tests']) == 3
    assert results['failing_tests'] == ['test_sample.py::test_failed']

def test_extract_pytest_results_edge_case():
    """Test edge case where only one test is passed."""
    lines = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.8.5, pytest-6.2.1, py-1.10.0, pluggy-0.13.1",
        "rootdir: /path/to/project",
        "collected 1 item",
        "",
        "test_sample.py::test_edge_case PASSED",
        "",
        "============================== 1 passed in 0.01s ===============================",
    ]
    reader = ReposWorklowConsoleReader()
    results = reader.extract_pytest_results(lines)
    
    assert results['summary']['total'] == 1
    assert results['summary']['passed'] == 1
    assert results['summary']['failed'] == 0
    assert results['summary']['skipped'] == 0
    assert results['summary']['errors'] == 0
    assert results['coverage'] is None
    assert len(results['tests']) == 1
    assert results['tests'][0]['status'] == 'PASSED'

@pytest.mark.parametrize("error_line, expected_exception", [
    ("IDNAError test line", IDNAError),
    ("IDNABidiError test line", IDNABidiError),
    ("InvalidCodepoint test line", InvalidCodepoint),
    ("InvalidCodepointContext test line", InvalidCodepointContext),
])
def test_extract_pytest_results_exceptions(error_line, expected_exception):
    """Test that specific lines trigger the expected exceptions."""
    lines = [
        "============================= test session starts ==============================",
        error_line,
    ]
    reader = ReposWorklowConsoleReader()
    with pytest.raises(expected_exception):
        reader.extract_pytest_results(lines)

def test_extract_pytest_results_no_tests():
    """Test behavior when no tests are present."""
    lines = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.8.5, pytest-6.2.1, py-1.10.0, pluggy-0.13.1",
        "rootdir: /path/to/project",
        "collected 0 items",
        "",
        "============================== no tests ran in 0.01s ===============================",
    ]
    reader = ReposWorklowConsoleReader()
    results = reader.extract_pytest_results(lines)
    
    assert results['summary']['total'] == 0
    assert results['summary']['passed'] == 0
    assert results['summary']['failed'] == 0
    assert results['summary']['skipped'] == 0
    assert results['summary']['errors'] == 0
    assert results['coverage'] is None
    assert len(results['tests']) == 0
    assert len(results['failing_tests']) == 0