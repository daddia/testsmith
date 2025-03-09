
"""
Unit tests for the untested methods in the error_recovery.py module.

This test file specifically focuses on testing the methods in the ErrorHandler class
that have not been covered in the existing test suite, including:
- save_error_logs
- _get_diagnostic_summary
- analyze_error_trend
- _calculate_error_trend
"""

import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta

import pytest
from pytest_mock import MockerFixture

from qa_agent.error_recovery import ErrorHandler, get_diagnostic_info


class TestErrorHandlerMethods:
    """Tests for the untested methods in the ErrorHandler class."""

    def setup_method(self):
        """Set up test environment with an ErrorHandler instance."""
        self.error_handler = ErrorHandler(max_retries=2, backoff_factor=0.1)
        # Prepare sample error logs for testing
        now = datetime.now()
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        two_hours_ago = (now - timedelta(hours=2)).isoformat()
        
        self.error_handler.error_logs = [
            {
                "timestamp": two_hours_ago,
                "function": "test_func1",
                "exception_type": "ValueError",
                "exception_message": "Invalid value",
                "recoverable": True,
            },
            {
                "timestamp": one_hour_ago,
                "function": "test_func2",
                "exception_type": "ConnectionError",
                "exception_message": "Connection failed",
                "recoverable": False,
            },
            {
                "timestamp": now.isoformat(),
                "function": "test_func1",
                "exception_type": "ValueError",
                "exception_message": "Another invalid value",
                "recoverable": True,
            }
        ]

    def test_save_error_logs(self):
        """Test saving error logs to a file."""
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        try:
            # Save logs to the temporary directory
            log_file = self.error_handler.save_error_logs(temp_dir, "test_logs")
            
            # Verify file was created
            assert os.path.exists(log_file)
            
            # Read back the file and verify content
            with open(log_file, "r") as f:
                log_data = json.load(f)
            
            # Verify structure and content
            assert "timestamp" in log_data
            assert "error_count" in log_data
            assert log_data["error_count"] == 3
            assert "logs" in log_data
            assert len(log_data["logs"]) == 3
            
            # Verify error categorization was added
            for log in log_data["logs"]:
                assert "category" in log
                if log["exception_type"] == "ValueError":
                    assert log["category"] == "validation"
                elif log["exception_type"] == "ConnectionError":
                    assert log["category"] == "connection"
            
            # Verify summary and trend data
            assert "summary" in log_data
            assert "trend" in log_data
            assert "system_diagnostics" in log_data
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    def test_get_diagnostic_summary(self):
        """Test generation of diagnostic summary from error logs."""
        summary = self.error_handler._get_diagnostic_summary(max_entries=2)
        
        # Verify summary structure
        assert "total_errors" in summary
        assert summary["total_errors"] == 3
        
        # Verify error distribution
        assert "error_distribution" in summary
        assert len(summary["error_distribution"]) > 0
        
        # Verify most common error is ValueError (appears twice)
        value_errors = [e for e in summary["error_distribution"] if e["type"] == "ValueError"]
        assert len(value_errors) == 1
        assert value_errors[0]["count"] == 2
        
        # Verify problematic functions
        assert "problematic_functions" in summary
        func1_errors = [f for f in summary["problematic_functions"] if f["function"] == "test_func1"]
        assert len(func1_errors) == 1
        assert func1_errors[0]["errors"] == 2
        
        # Verify recent errors are included and limited to max_entries
        assert "recent_errors" in summary
        assert len(summary["recent_errors"]) <= 2
        
        # Verify trend is included
        assert "trend" in summary

    def test_analyze_error_trend(self):
        """Test analysis of error trends from logs."""
        analysis = self.error_handler.analyze_error_trend()
        
        # Verify analysis structure
        assert "error_count" in analysis
        assert analysis["error_count"] == 3
        
        # Verify common errors
        assert "common_errors" in analysis
        assert len(analysis["common_errors"]) > 0
        
        # Verify problematic functions
        assert "problematic_functions" in analysis
        assert len(analysis["problematic_functions"]) > 0
        
        # Verify error trend
        assert "error_trend" in analysis

    def test_calculate_error_trend(self):
        """Test calculation of error trends over time."""
        trend_data = self.error_handler._calculate_error_trend()
        
        # Verify trend data structure
        assert "trend" in trend_data
        assert trend_data["trend"] in ["increasing", "decreasing", "stable", "not_enough_data"]
        
        # Test with insufficient data
        self.error_handler.error_logs = [{"timestamp": datetime.now().isoformat()}]
        insufficient_data = self.error_handler._calculate_error_trend()
        assert insufficient_data["trend"] == "not_enough_data"
        
        # Test with no timestamps (should still work)
        self.error_handler.error_logs = [{"function": "test"}] * 3
        no_timestamps = self.error_handler._calculate_error_trend()
        assert "trend" in no_timestamps

    def test_empty_error_logs(self):
        """Test behavior with empty error logs."""
        self.error_handler.error_logs = []
        
        # Test save_error_logs with empty logs
        temp_dir = tempfile.mkdtemp()
        try:
            log_file = self.error_handler.save_error_logs(temp_dir, "empty_logs")
            assert log_file == ""  # Should return empty string with no logs
        finally:
            shutil.rmtree(temp_dir)
        
        # Test diagnostic summary with empty logs
        summary = self.error_handler._get_diagnostic_summary()
        assert "status" in summary
        assert summary["status"] == "no_errors"
        
        # Test analyze_error_trend with empty logs
        analysis = self.error_handler.analyze_error_trend()
        assert analysis["error_count"] == 0
        assert analysis["common_errors"] == []
