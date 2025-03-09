/*
 * Simple example SQL functions for QA Agent test generation.
 * 
 * These functions demonstrate basic SQL operations that can be tested.
 * Assumes PostgreSQL syntax.
 */

-- Function to add two numbers together
CREATE OR REPLACE FUNCTION add_numbers(a NUMERIC, b NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    RETURN a + b;
END;
$$ LANGUAGE plpgsql;

-- Function to subtract b from a
CREATE OR REPLACE FUNCTION subtract_numbers(a NUMERIC, b NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    RETURN a - b;
END;
$$ LANGUAGE plpgsql;

-- Function to multiply two numbers
CREATE OR REPLACE FUNCTION multiply_numbers(a NUMERIC, b NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    RETURN a * b;
END;
$$ LANGUAGE plpgsql;

-- Function to divide a by b with error handling
CREATE OR REPLACE FUNCTION divide_numbers(a NUMERIC, b NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    IF b = 0 THEN
        RAISE EXCEPTION 'Cannot divide by zero';
    END IF;
    
    RETURN a / b;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate average of a set of numbers in a table
CREATE OR REPLACE FUNCTION calculate_average(table_name TEXT, column_name TEXT)
RETURNS NUMERIC AS $$
DECLARE
    avg_value NUMERIC;
BEGIN
    EXECUTE format('SELECT AVG(%I) FROM %I', column_name, table_name)
    INTO avg_value;
    
    RETURN COALESCE(avg_value, 0);
END;
$$ LANGUAGE plpgsql;