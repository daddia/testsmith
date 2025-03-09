"""
Simple example Python functions for QA Agent test generation.

These functions demonstrate basic arithmetic operations that are simple to test.
"""

def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b

def subtract_numbers(a, b):
    """
    Subtract b from a.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Difference of a and b
    """
    return a - b

def multiply_numbers(a, b):
    """
    Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of a and b
    """
    return a * b

def divide_numbers(a, b):
    """
    Divide a by b.
    
    Args:
        a: First number (dividend)
        b: Second number (divisor)
        
    Returns:
        Quotient of a divided by b
        
    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b