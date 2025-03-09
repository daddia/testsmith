/**
 * Simple example JavaScript functions for QA Agent test generation.
 * 
 * These functions demonstrate basic arithmetic operations that are simple to test.
 */

/**
 * Add two numbers together.
 * 
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Sum of a and b
 */
function addNumbers(a, b) {
  return a + b;
}

/**
 * Subtract b from a.
 * 
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Difference of a and b
 */
function subtractNumbers(a, b) {
  return a - b;
}

/**
 * Multiply two numbers.
 * 
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Product of a and b
 */
function multiplyNumbers(a, b) {
  return a * b;
}

/**
 * Divide a by b.
 * 
 * @param {number} a - First number (dividend)
 * @param {number} b - Second number (divisor)
 * @returns {number} Quotient of a divided by b
 * @throws {Error} If b is zero
 */
function divideNumbers(a, b) {
  if (b === 0) {
    throw new Error("Cannot divide by zero");
  }
  return a / b;
}

// Export functions for use in other modules
module.exports = {
  addNumbers,
  subtractNumbers,
  multiplyNumbers,
  divideNumbers
};