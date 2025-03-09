// Package calculator provides simple arithmetic functions for QA Agent test generation.
package calculator

import (
	"errors"
	"fmt"
)

// AddNumbers adds two numbers together.
// Returns the sum of a and b.
func AddNumbers(a, b float64) float64 {
	return a + b
}

// SubtractNumbers subtracts b from a.
// Returns the difference of a and b.
func SubtractNumbers(a, b float64) float64 {
	return a - b
}

// MultiplyNumbers multiplies two numbers.
// Returns the product of a and b.
func MultiplyNumbers(a, b float64) float64 {
	return a * b
}

// DivideNumbers divides a by b.
// Returns the quotient of a divided by b.
// Returns an error if b is zero.
func DivideNumbers(a, b float64) (float64, error) {
	if b == 0 {
		return 0, errors.New("cannot divide by zero")
	}
	return a / b, nil
}

// Calculator represents a simple calculator with memory.
type Calculator struct {
	Memory float64
}

// New creates a new Calculator with memory set to 0.
func New() *Calculator {
	return &Calculator{Memory: 0}
}

// Add adds a number to the memory and returns the new value.
func (c *Calculator) Add(n float64) float64 {
	c.Memory += n
	return c.Memory
}

// Subtract subtracts a number from the memory and returns the new value.
func (c *Calculator) Subtract(n float64) float64 {
	c.Memory -= n
	return c.Memory
}

// Clear resets the memory to 0.
func (c *Calculator) Clear() {
	c.Memory = 0
}

// String returns a string representation of the calculator's memory.
func (c *Calculator) String() string {
	return fmt.Sprintf("Calculator memory: %.2f", c.Memory)
}