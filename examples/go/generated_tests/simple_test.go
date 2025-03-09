// Package calculator_test provides unit tests for the calculator package
package calculator_test

import (
	"errors"
	"testing"

	"examples/go/example_src/calculator" // Import the calculator package to test
)

// TestAddNumbers tests the AddNumbers function with various inputs
func TestAddNumbers(t *testing.T) {
	// Test positive numbers
	result := calculator.AddNumbers(2, 3)
	if result != 5 {
		t.Errorf("AddNumbers(2, 3) = %f; expected 5", result)
	}

	// Test negative numbers
	result = calculator.AddNumbers(-2, -3)
	if result != -5 {
		t.Errorf("AddNumbers(-2, -3) = %f; expected -5", result)
	}

	// Test mixed numbers
	result = calculator.AddNumbers(2, -3)
	if result != -1 {
		t.Errorf("AddNumbers(2, -3) = %f; expected -1", result)
	}

	// Test floating point numbers
	result = calculator.AddNumbers(2.5, 3.5)
	if result != 6.0 {
		t.Errorf("AddNumbers(2.5, 3.5) = %f; expected 6.0", result)
	}
}

// TestSubtractNumbers tests the SubtractNumbers function with various inputs
func TestSubtractNumbers(t *testing.T) {
	// Test positive numbers
	result := calculator.SubtractNumbers(5, 3)
	if result != 2 {
		t.Errorf("SubtractNumbers(5, 3) = %f; expected 2", result)
	}

	// Test negative numbers
	result = calculator.SubtractNumbers(-5, -3)
	if result != -2 {
		t.Errorf("SubtractNumbers(-5, -3) = %f; expected -2", result)
	}

	// Test mixed numbers
	result = calculator.SubtractNumbers(5, -3)
	if result != 8 {
		t.Errorf("SubtractNumbers(5, -3) = %f; expected 8", result)
	}

	// Test floating point numbers
	result = calculator.SubtractNumbers(5.5, 3.5)
	if result != 2.0 {
		t.Errorf("SubtractNumbers(5.5, 3.5) = %f; expected 2.0", result)
	}
}

// TestMultiplyNumbers tests the MultiplyNumbers function with various inputs
func TestMultiplyNumbers(t *testing.T) {
	// Test positive numbers
	result := calculator.MultiplyNumbers(2, 3)
	if result != 6 {
		t.Errorf("MultiplyNumbers(2, 3) = %f; expected 6", result)
	}

	// Test negative numbers
	result = calculator.MultiplyNumbers(-2, -3)
	if result != 6 {
		t.Errorf("MultiplyNumbers(-2, -3) = %f; expected 6", result)
	}

	// Test mixed numbers
	result = calculator.MultiplyNumbers(2, -3)
	if result != -6 {
		t.Errorf("MultiplyNumbers(2, -3) = %f; expected -6", result)
	}

	// Test floating point numbers
	result = calculator.MultiplyNumbers(2.5, 3.5)
	if result != 8.75 {
		t.Errorf("MultiplyNumbers(2.5, 3.5) = %f; expected 8.75", result)
	}
}

// TestDivideNumbers tests the DivideNumbers function with various inputs
func TestDivideNumbers(t *testing.T) {
	// Test positive numbers
	result, err := calculator.DivideNumbers(6, 3)
	if err != nil {
		t.Errorf("DivideNumbers(6, 3) returned unexpected error: %v", err)
	}
	if result != 2 {
		t.Errorf("DivideNumbers(6, 3) = %f; expected 2", result)
	}

	// Test negative numbers
	result, err = calculator.DivideNumbers(-6, -3)
	if err != nil {
		t.Errorf("DivideNumbers(-6, -3) returned unexpected error: %v", err)
	}
	if result != 2 {
		t.Errorf("DivideNumbers(-6, -3) = %f; expected 2", result)
	}

	// Test mixed numbers
	result, err = calculator.DivideNumbers(6, -3)
	if err != nil {
		t.Errorf("DivideNumbers(6, -3) returned unexpected error: %v", err)
	}
	if result != -2 {
		t.Errorf("DivideNumbers(6, -3) = %f; expected -2", result)
	}

	// Test floating point numbers
	result, err = calculator.DivideNumbers(6.6, 3.3)
	if err != nil {
		t.Errorf("DivideNumbers(6.6, 3.3) returned unexpected error: %v", err)
	}
	if result != 2.0 {
		t.Errorf("DivideNumbers(6.6, 3.3) = %f; expected 2.0", result)
	}
}

// TestDivideByZero tests that DivideNumbers returns an error when dividing by zero
func TestDivideByZero(t *testing.T) {
	_, err := calculator.DivideNumbers(6, 0)
	if err == nil {
		t.Error("DivideNumbers(6, 0) did not return an error, expected divide by zero error")
	}
	
	expectedErr := errors.New("cannot divide by zero")
	if err.Error() != expectedErr.Error() {
		t.Errorf("DivideNumbers(6, 0) returned error %v, expected %v", err, expectedErr)
	}
}

// TestCalculator tests the Calculator struct and its methods
func TestCalculator(t *testing.T) {
	// Create a new calculator
	calc := calculator.New()
	
	// Test initial memory is 0
	if calc.Memory != 0 {
		t.Errorf("New calculator memory = %f; expected 0", calc.Memory)
	}
	
	// Test Add method
	result := calc.Add(5)
	if result != 5 || calc.Memory != 5 {
		t.Errorf("After calc.Add(5), memory = %f, result = %f; expected both to be 5", calc.Memory, result)
	}
	
	// Test Subtract method
	result = calc.Subtract(3)
	if result != 2 || calc.Memory != 2 {
		t.Errorf("After calc.Subtract(3), memory = %f, result = %f; expected both to be 2", calc.Memory, result)
	}
	
	// Test Clear method
	calc.Clear()
	if calc.Memory != 0 {
		t.Errorf("After calc.Clear(), memory = %f; expected 0", calc.Memory)
	}
	
	// Test String method
	expected := "Calculator memory: 0.00"
	if calc.String() != expected {
		t.Errorf("calc.String() = %s; expected %s", calc.String(), expected)
	}
}