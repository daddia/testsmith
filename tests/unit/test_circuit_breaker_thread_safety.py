"""
Unit tests for validating thread safety of the CircuitBreaker class.

These tests verify that the CircuitBreaker class correctly handles concurrent
access in a multi-threaded environment.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from qa_agent.error_recovery import CircuitBreaker, CircuitBreakerState


class TestCircuitBreakerThreadSafety(unittest.TestCase):
    """Tests for thread safety of the CircuitBreaker class."""

    def setUp(self):
        """Set up test environment."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for faster tests (using int as required)
            half_open_max_calls=2,
        )

    def test_concurrent_register_operations(self):
        """Test concurrent registering of operations."""
        num_threads = 20
        operations_per_thread = 5

        def register_operations(thread_idx):
            for i in range(operations_per_thread):
                op_name = f"op_{thread_idx}_{i}"
                self.circuit_breaker.register_operation(op_name)

        # Run registration concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_operations, i) for i in range(num_threads)]
            # Wait for all to complete
            for future in futures:
                future.result()

        # Verify all operations were registered
        total_expected = num_threads * operations_per_thread
        self.assertEqual(
            len(self.circuit_breaker.protected_operations),
            total_expected,
            f"Expected {total_expected} registered operations, but got {len(self.circuit_breaker.protected_operations)}",
        )

    def test_concurrent_failures_threshold(self):
        """Test that failure threshold works correctly under concurrent load."""
        operation_name = "test_operation"
        self.circuit_breaker.register_operation(operation_name)

        num_threads = 10

        def record_failure():
            self.circuit_breaker.record_failure(operation_name)

        # Run failures concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_failure) for _ in range(num_threads)]
            for future in futures:
                future.result()

        # After recording more failures than the threshold, the circuit should be open
        self.assertEqual(
            self.circuit_breaker.state,
            CircuitBreakerState.OPEN,
            f"Expected circuit state to be OPEN, but got {self.circuit_breaker.state}",
        )

        # Verify the circuit breaker is actually preventing execution
        self.assertFalse(
            self.circuit_breaker.can_execute(operation_name),
            "Expected can_execute to return False for an open circuit",
        )

    def test_concurrent_success_recovery(self):
        """Test circuit recovery with concurrent successes."""
        operation_name = "recovery_test"
        self.circuit_breaker.register_operation(operation_name)

        # Force circuit into OPEN state
        for _ in range(self.circuit_breaker.failure_threshold):
            self.circuit_breaker.record_failure(operation_name)

        self.assertEqual(
            self.circuit_breaker.state,
            CircuitBreakerState.OPEN,
            "Circuit should be OPEN after reaching failure threshold",
        )

        # Wait for recovery timeout to elapse
        time.sleep(self.circuit_breaker.recovery_timeout + 0.1)

        # Now the circuit should transition to HALF_OPEN when checked
        self.assertTrue(
            self.circuit_breaker.can_execute(operation_name),
            "Circuit should allow execution after recovery timeout",
        )
        self.assertEqual(
            self.circuit_breaker.state,
            CircuitBreakerState.HALF_OPEN,
            "Circuit should be in HALF_OPEN state after recovery timeout",
        )

        # Record concurrent successes
        num_threads = self.circuit_breaker.half_open_max_calls

        def record_success():
            time.sleep(0.05)  # Small delay to ensure some overlap
            self.circuit_breaker.record_success(operation_name)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_success) for _ in range(num_threads)]
            for future in futures:
                future.result()

        # Circuit should be CLOSED after enough successes
        self.assertEqual(
            self.circuit_breaker.state,
            CircuitBreakerState.CLOSED,
            "Circuit should be CLOSED after recording enough successes in HALF_OPEN state",
        )

    def test_mixed_concurrent_operations(self):
        """Test concurrent mix of successes and failures."""
        operation_name = "mixed_test"
        self.circuit_breaker.register_operation(operation_name)

        num_threads = 20
        operations = []

        # Create a mix of success/failure operations to execute concurrently
        for i in range(num_threads):
            if i % 3 == 0:  # 1/3 failures, 2/3 successes
                operations.append(lambda: self.circuit_breaker.record_failure(operation_name))
            else:
                operations.append(lambda: self.circuit_breaker.record_success(operation_name))

        # Run mixed operations concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(op) for op in operations]
            for future in futures:
                future.result()

        # Verify the circuit breaker stats are updated correctly
        total_operations = (
            self.circuit_breaker.stats["total_successes"]
            + self.circuit_breaker.stats["total_failures"]
        )
        self.assertEqual(
            total_operations,
            num_threads,
            f"Expected {num_threads} total operations recorded in stats, but got {total_operations}",
        )

    def test_stats_under_load(self):
        """Test that statistics tracking is thread-safe."""
        operation_name = "stats_test"
        self.circuit_breaker.register_operation(operation_name)

        num_success_threads = 15
        num_failure_threads = 10

        def record_success():
            self.circuit_breaker.record_success(operation_name)

        def record_failure():
            self.circuit_breaker.record_failure(operation_name)

        # Run both types of operations concurrently
        with ThreadPoolExecutor(max_workers=num_success_threads + num_failure_threads) as executor:
            success_futures = [executor.submit(record_success) for _ in range(num_success_threads)]
            failure_futures = [executor.submit(record_failure) for _ in range(num_failure_threads)]

            # Wait for all to complete
            for future in success_futures + failure_futures:
                future.result()

        # Verify stats match expected counts
        self.assertEqual(
            self.circuit_breaker.stats["total_successes"],
            num_success_threads,
            f"Expected {num_success_threads} total successes, but got {self.circuit_breaker.stats['total_successes']}",
        )
        self.assertEqual(
            self.circuit_breaker.stats["total_failures"],
            num_failure_threads,
            f"Expected {num_failure_threads} total failures, but got {self.circuit_breaker.stats['total_failures']}",
        )

    def test_lock_timeout_safety(self):
        """Test that lock timeouts prevent deadlocks."""
        operation_name = "deadlock_test"
        self.circuit_breaker.register_operation(operation_name)

        # Create a custom circuit breaker to test lock timeout
        cb_with_timeout = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        cb_with_timeout.register_operation(operation_name)

        # Force acquire the lock but don't release it
        lock_acquired = cb_with_timeout._lock.acquire()
        self.assertTrue(lock_acquired, "Should be able to acquire the lock")

        # Now try operations that require the lock - they should handle the timeout
        try:
            # These should not block indefinitely due to timeout
            result1 = cb_with_timeout.can_execute(operation_name)
            result2 = cb_with_timeout.get_state(operation_name)
            cb_with_timeout.record_success(operation_name)
            cb_with_timeout.record_failure(operation_name)

            # Verify default behaviors on lock timeout
            self.assertTrue(
                result1, "can_execute should return True as safe default on lock timeout"
            )
            self.assertEqual(
                result2,
                CircuitBreakerState.CLOSED,
                "get_state should return CLOSED as safe default on lock timeout",
            )
        finally:
            # Release the lock to avoid affecting other tests
            cb_with_timeout._lock.release()


if __name__ == "__main__":
    unittest.main()
