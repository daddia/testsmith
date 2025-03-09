<?php
/**
 * Unit tests for Simple.php module.
 * 
 * These tests verify the functionality of basic arithmetic operations.
 */

use PHPUnit\Framework\TestCase;

/**
 * Test case for the simple arithmetic functions.
 */
class SimpleTest extends TestCase
{
    /**
     * Test the addNumbers function with various inputs.
     */
    public function testAddNumbers()
    {
        // Test positive numbers
        $this->assertEquals(5, addNumbers(2, 3));
        
        // Test negative numbers
        $this->assertEquals(-5, addNumbers(-2, -3));
        
        // Test mixed numbers
        $this->assertEquals(-1, addNumbers(2, -3));
        
        // Test floating point numbers
        $this->assertEquals(6.0, addNumbers(2.5, 3.5));
    }
    
    /**
     * Test the subtractNumbers function with various inputs.
     */
    public function testSubtractNumbers()
    {
        // Test positive numbers
        $this->assertEquals(2, subtractNumbers(5, 3));
        
        // Test negative numbers
        $this->assertEquals(-2, subtractNumbers(-5, -3));
        
        // Test mixed numbers
        $this->assertEquals(8, subtractNumbers(5, -3));
        
        // Test floating point numbers
        $this->assertEquals(2.0, subtractNumbers(5.5, 3.5));
    }
    
    /**
     * Test the multiplyNumbers function with various inputs.
     */
    public function testMultiplyNumbers()
    {
        // Test positive numbers
        $this->assertEquals(6, multiplyNumbers(2, 3));
        
        // Test negative numbers
        $this->assertEquals(6, multiplyNumbers(-2, -3));
        
        // Test mixed numbers
        $this->assertEquals(-6, multiplyNumbers(2, -3));
        
        // Test floating point numbers
        $this->assertEquals(8.75, multiplyNumbers(2.5, 3.5));
    }
    
    /**
     * Test the divideNumbers function with various inputs.
     */
    public function testDivideNumbers()
    {
        // Test positive numbers
        $this->assertEquals(2, divideNumbers(6, 3));
        
        // Test negative numbers
        $this->assertEquals(2, divideNumbers(-6, -3));
        
        // Test mixed numbers
        $this->assertEquals(-2, divideNumbers(6, -3));
        
        // Test floating point numbers
        $this->assertEquals(2.0, divideNumbers(6.6, 3.3));
    }
    
    /**
     * Test the divideNumbers function throws exception when dividing by zero.
     */
    public function testDivideByZero()
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Cannot divide by zero');
        divideNumbers(6, 0);
    }
}