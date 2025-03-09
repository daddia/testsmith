<?php
/**
 * Simple example PHP functions for QA Agent test generation.
 * 
 * These functions demonstrate basic arithmetic operations that are simple to test.
 */

/**
 * Add two numbers together.
 * 
 * @param float $a First number
 * @param float $b Second number
 * @return float Sum of a and b
 */
function addNumbers($a, $b) {
    return $a + $b;
}

/**
 * Subtract b from a.
 * 
 * @param float $a First number
 * @param float $b Second number
 * @return float Difference of a and b
 */
function subtractNumbers($a, $b) {
    return $a - $b;
}

/**
 * Multiply two numbers.
 * 
 * @param float $a First number
 * @param float $b Second number
 * @return float Product of a and b
 */
function multiplyNumbers($a, $b) {
    return $a * $b;
}

/**
 * Divide a by b.
 * 
 * @param float $a First number (dividend)
 * @param float $b Second number (divisor)
 * @return float Quotient of a divided by b
 * @throws \InvalidArgumentException If b is zero
 */
function divideNumbers($a, $b) {
    if ($b == 0) {
        throw new \InvalidArgumentException("Cannot divide by zero");
    }
    return $a / $b;
}