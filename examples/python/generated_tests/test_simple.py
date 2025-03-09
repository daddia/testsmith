"""
Unit tests for simple.py module.

These tests verify the functionality of basic arithmetic operations.
"""

import pytest
from examples.python.example_src.simple import add_numbers, subtract_numbers, multiply_numbers, divide_numbers


def test_add_numbers():
    """Test the add_numbers function with various inputs."""
    # Test positive numbers
    assert add_numbers(2, 3) == 5
    
    # Test negative numbers
    assert add_numbers(-2, -3) == -5
    
    # Test mixed numbers
    assert add_numbers(2, -3) == -1
    
    # Test floating point numbers
    assert add_numbers(2.5, 3.5) == 6.0


def test_subtract_numbers():
    """Test the subtract_numbers function with various inputs."""
    # Test positive numbers
    assert subtract_numbers(5, 3) == 2
    
    # Test negative numbers
    assert subtract_numbers(-5, -3) == -2
    
    # Test mixed numbers
    assert subtract_numbers(5, -3) == 8
    
    # Test floating point numbers
    assert subtract_numbers(5.5, 3.5) == 2.0


def test_multiply_numbers():
    """Test the multiply_numbers function with various inputs."""
    # Test positive numbers
    assert multiply_numbers(2, 3) == 6
    
    # Test negative numbers
    assert multiply_numbers(-2, -3) == 6
    
    # Test mixed numbers
    assert multiply_numbers(2, -3) == -6
    
    # Test floating point numbers
    assert multiply_numbers(2.5, 3.5) == 8.75


def test_divide_numbers():
    """Test the divide_numbers function with various inputs."""
    # Test positive numbers
    assert divide_numbers(6, 3) == 2
    
    # Test negative numbers
    assert divide_numbers(-6, -3) == 2
    
    # Test mixed numbers
    assert divide_numbers(6, -3) == -2
    
    # Test floating point numbers
    assert divide_numbers(6.6, 3.3) == 2.0


def test_divide_by_zero():
    """Test the divide_numbers function raises ZeroDivisionError when dividing by zero."""
    with pytest.raises(ZeroDivisionError):
        divide_numbers(6, 0)