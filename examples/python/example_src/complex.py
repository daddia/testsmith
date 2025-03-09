"""
Complex example Python functions for QA Agent test generation.

These functions demonstrate more complex operations that require more
sophisticated test approaches, including error handling, input validation,
and dependency management.
"""

from typing import List, Dict, Optional, Union, Iterable
import re
import math


def validate_email(email: str) -> bool:
    """
    Validate if a string is a properly formatted email address.
    
    Args:
        email: The email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    if not isinstance(email, str):
        return False
    
    # Basic email validation pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return bool(re.match(pattern, email))


def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary with mean, median, and standard deviation
        
    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate statistics of an empty list")
    
    # Calculate mean
    mean = sum(numbers) / len(numbers)
    
    # Calculate median
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median = sorted_numbers[n//2]
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = math.sqrt(variance)
    
    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev
    }


def filter_and_transform(items: List[Dict[str, any]],
                         filter_key: str,
                         filter_value: any,
                         transform_func: callable) -> List[Dict[str, any]]:
    """
    Filter items by a key-value pair and transform them using a function.
    
    Args:
        items: List of dictionaries to filter and transform
        filter_key: The key to filter on
        filter_value: The value to match
        transform_func: Function to transform matching items
        
    Returns:
        List of transformed items that match the filter
    """
    result = []
    
    for item in items:
        # Skip items that don't have the filter key
        if filter_key not in item:
            continue
            
        # Check if the item matches the filter
        if item[filter_key] == filter_value:
            # Apply the transformation and add to result
            transformed_item = transform_func(item)
            result.append(transformed_item)
    
    return result


class CacheManager:
    """A simple cache manager with LRU (Least Recently Used) eviction."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of items to store in cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str) -> Optional[any]:
        """
        Get an item from the cache.
        
        Args:
            key: The key to look up
            
        Returns:
            The cached value or None if not found
        """
        if key in self.cache:
            # Update access order (move to end = most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: any) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: The key to store
            value: The value to cache
        """
        # If key already exists, update access order
        if key in self.cache:
            self.access_order.remove(key)
        
        # If cache is full, remove least recently used item
        elif len(self.cache) >= self.max_size:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        # Add new item
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.access_order = []
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        return len(self.cache)