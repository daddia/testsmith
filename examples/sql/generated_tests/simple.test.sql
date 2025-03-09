-- Simple SQL test file for testing the functions in simple.sql
-- Uses pgTAP for PostgreSQL testing

BEGIN;

-- Load pgTAP extensions
-- NOTE: This requires pgTAP to be installed in the database
\i pgtap.sql

-- Plan the tests - specify how many tests will be run
SELECT plan(10);

-- Test add_numbers function
SELECT is(
    add_numbers(2, 3),
    5,
    'add_numbers(2, 3) should return 5'
);

SELECT is(
    add_numbers(-2, -3),
    -5,
    'add_numbers(-2, -3) should return -5'
);

-- Test subtract_numbers function
SELECT is(
    subtract_numbers(5, 3),
    2,
    'subtract_numbers(5, 3) should return 2'
);

SELECT is(
    subtract_numbers(-5, -3),
    -2,
    'subtract_numbers(-5, -3) should return -2'
);

-- Test multiply_numbers function
SELECT is(
    multiply_numbers(2, 3),
    6,
    'multiply_numbers(2, 3) should return 6'
);

SELECT is(
    multiply_numbers(-2, -3),
    6,
    'multiply_numbers(-2, -3) should return 6'
);

-- Test divide_numbers function
SELECT is(
    divide_numbers(6, 3),
    2,
    'divide_numbers(6, 3) should return 2'
);

SELECT is(
    divide_numbers(-6, -3),
    2,
    'divide_numbers(-6, -3) should return 2'
);

-- Test divide_numbers exception handling
SELECT throws_ok(
    'SELECT divide_numbers(6, 0)',
    'Cannot divide by zero',
    'divide_numbers(6, 0) should throw "Cannot divide by zero" exception'
);

-- Test calculate_average function
-- First create a test table
CREATE TEMPORARY TABLE test_numbers (
    id SERIAL PRIMARY KEY,
    value NUMERIC
);

-- Insert test data
INSERT INTO test_numbers (value) VALUES (10), (20), (30), (40);

-- Test the calculate_average function
SELECT is(
    calculate_average('test_numbers', 'value'),
    25,
    'calculate_average should return the correct average (25) of values'
);

-- Finish the tests
SELECT * FROM finish();

ROLLBACK;