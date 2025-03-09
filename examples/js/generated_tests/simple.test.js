/**
 * Unit tests for simple.js module.
 * 
 * These tests verify the functionality of basic arithmetic operations.
 */

// Import the functions to test
const { addNumbers, subtractNumbers, multiplyNumbers, divideNumbers } = require('../example_src/simple');

describe('Calculator Functions', () => {
  // Test suite for addNumbers function
  describe('addNumbers', () => {
    test('adds positive numbers correctly', () => {
      expect(addNumbers(2, 3)).toBe(5);
    });

    test('adds negative numbers correctly', () => {
      expect(addNumbers(-2, -3)).toBe(-5);
    });

    test('adds mixed sign numbers correctly', () => {
      expect(addNumbers(2, -3)).toBe(-1);
    });

    test('adds decimal numbers correctly', () => {
      expect(addNumbers(2.5, 3.5)).toBe(6.0);
    });
  });

  // Test suite for subtractNumbers function
  describe('subtractNumbers', () => {
    test('subtracts positive numbers correctly', () => {
      expect(subtractNumbers(5, 3)).toBe(2);
    });

    test('subtracts negative numbers correctly', () => {
      expect(subtractNumbers(-5, -3)).toBe(-2);
    });

    test('subtracts mixed sign numbers correctly', () => {
      expect(subtractNumbers(5, -3)).toBe(8);
    });

    test('subtracts decimal numbers correctly', () => {
      expect(subtractNumbers(5.5, 3.5)).toBe(2.0);
    });
  });

  // Test suite for multiplyNumbers function
  describe('multiplyNumbers', () => {
    test('multiplies positive numbers correctly', () => {
      expect(multiplyNumbers(2, 3)).toBe(6);
    });

    test('multiplies negative numbers correctly', () => {
      expect(multiplyNumbers(-2, -3)).toBe(6);
    });

    test('multiplies mixed sign numbers correctly', () => {
      expect(multiplyNumbers(2, -3)).toBe(-6);
    });

    test('multiplies decimal numbers correctly', () => {
      expect(multiplyNumbers(2.5, 3.5)).toBe(8.75);
    });
  });

  // Test suite for divideNumbers function
  describe('divideNumbers', () => {
    test('divides positive numbers correctly', () => {
      expect(divideNumbers(6, 3)).toBe(2);
    });

    test('divides negative numbers correctly', () => {
      expect(divideNumbers(-6, -3)).toBe(2);
    });

    test('divides mixed sign numbers correctly', () => {
      expect(divideNumbers(6, -3)).toBe(-2);
    });

    test('divides decimal numbers correctly', () => {
      expect(divideNumbers(6.6, 3.3)).toBeCloseTo(2.0);
    });

    test('throws error when dividing by zero', () => {
      expect(() => {
        divideNumbers(6, 0);
      }).toThrow('Cannot divide by zero');
    });
  });
});