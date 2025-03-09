#!/bin/bash
# Display the structure of the examples directory and demonstrate test naming conventions

echo "============================================================"
echo "QA Agent Example Directory Structure and Test File Naming"
echo "============================================================"
echo ""

# Display the directory structure
echo "Directory Structure:"
echo "-------------------"
find . -type d -not -path "*/\.*" | sort | sed -e 's/[^-][^\/]*\//  |/g' -e 's/|\([^ ]\)/|-\1/'
echo ""

# Count source and test files by language
echo "Source Files and Test Files by Language:"
echo "--------------------------------------"
printf "%-15s %-20s %-20s\n" "Language" "Source Files" "Test Files"
echo "--------------------------------------"

# Python
PYTHON_SRC=$(find ./python/example_src -name "*.py" 2>/dev/null | wc -l)
PYTHON_TEST=$(find ./python/generated_tests -name "test_*.py" 2>/dev/null | wc -l)
printf "%-15s %-20s %-20s\n" "Python" "$PYTHON_SRC" "$PYTHON_TEST"

# JavaScript
JS_SRC=$(find ./js/example_src -name "*.js" 2>/dev/null | wc -l)
JS_TEST=$(find ./js/generated_tests -name "*.test.js" 2>/dev/null | wc -l)
printf "%-15s %-20s %-20s\n" "JavaScript" "$JS_SRC" "$JS_TEST"

# PHP
PHP_SRC=$(find ./php/example_src -name "*.php" 2>/dev/null | wc -l)
PHP_TEST=$(find ./php/generated_tests -name "*Test.php" 2>/dev/null | wc -l)
printf "%-15s %-20s %-20s\n" "PHP" "$PHP_SRC" "$PHP_TEST"

# Go
GO_SRC=$(find ./go/example_src -name "*.go" 2>/dev/null | wc -l)
GO_TEST=$(find ./go/generated_tests -name "*_test.go" 2>/dev/null | wc -l)
printf "%-15s %-20s %-20s\n" "Go" "$GO_SRC" "$GO_TEST"

# SQL
SQL_SRC=$(find ./sql/example_src -name "*.sql" 2>/dev/null | wc -l)
SQL_TEST=$(find ./sql/generated_tests -name "*.test.sql" 2>/dev/null | wc -l)
printf "%-15s %-20s %-20s\n" "SQL" "$SQL_SRC" "$SQL_TEST"

echo ""

# Display naming convention examples
echo "File Naming Convention Examples:"
echo "------------------------------"
printf "%-15s %-25s %-25s\n" "Language" "Source File" "Test File"
echo "------------------------------"

# Find one example of each type if available
PYTHON_SRC_EXAMPLE=$(find ./examples/python/example_src -name "*.py" 2>/dev/null | head -n 1)
PYTHON_TEST_EXAMPLE=$(find ./examples/python/generated_tests -name "test_*.py" 2>/dev/null | head -n 1)
if [ -n "$PYTHON_SRC_EXAMPLE" ] && [ -n "$PYTHON_TEST_EXAMPLE" ]; then
    printf "%-15s %-25s %-25s\n" "Python" "$(basename $PYTHON_SRC_EXAMPLE)" "$(basename $PYTHON_TEST_EXAMPLE)"
fi

JS_SRC_EXAMPLE=$(find ./examples/js/example_src -name "*.js" 2>/dev/null | head -n 1)
JS_TEST_EXAMPLE=$(find ./examples/js/generated_tests -name "*.test.js" 2>/dev/null | head -n 1)
if [ -n "$JS_SRC_EXAMPLE" ] && [ -n "$JS_TEST_EXAMPLE" ]; then
    printf "%-15s %-25s %-25s\n" "JavaScript" "$(basename $JS_SRC_EXAMPLE)" "$(basename $JS_TEST_EXAMPLE)"
fi

PHP_SRC_EXAMPLE=$(find ./examples/php/example_src -name "*.php" 2>/dev/null | head -n 1)
PHP_TEST_EXAMPLE=$(find ./examples/php/generated_tests -name "*Test.php" 2>/dev/null | head -n 1)
if [ -n "$PHP_SRC_EXAMPLE" ] && [ -n "$PHP_TEST_EXAMPLE" ]; then
    printf "%-15s %-25s %-25s\n" "PHP" "$(basename $PHP_SRC_EXAMPLE)" "$(basename $PHP_TEST_EXAMPLE)"
fi

GO_SRC_EXAMPLE=$(find ./examples/go/example_src -name "*.go" 2>/dev/null | head -n 1)
GO_TEST_EXAMPLE=$(find ./examples/go/generated_tests -name "*_test.go" 2>/dev/null | head -n 1)
if [ -n "$GO_SRC_EXAMPLE" ] && [ -n "$GO_TEST_EXAMPLE" ]; then
    printf "%-15s %-25s %-25s\n" "Go" "$(basename $GO_SRC_EXAMPLE)" "$(basename $GO_TEST_EXAMPLE)"
fi

SQL_SRC_EXAMPLE=$(find ./examples/sql/example_src -name "*.sql" 2>/dev/null | head -n 1)
SQL_TEST_EXAMPLE=$(find ./examples/sql/generated_tests -name "*.test.sql" 2>/dev/null | head -n 1)
if [ -n "$SQL_SRC_EXAMPLE" ] && [ -n "$SQL_TEST_EXAMPLE" ]; then
    printf "%-15s %-25s %-25s\n" "SQL" "$(basename $SQL_SRC_EXAMPLE)" "$(basename $SQL_TEST_EXAMPLE)"
fi

echo ""
echo "All test files follow the language-specific naming conventions:"
echo "  - Python: test_*.py"
echo "  - JavaScript: *.test.js"
echo "  - PHP: *Test.php"
echo "  - Go: *_test.go"
echo "  - SQL: *.test.sql"
echo ""
echo "============================================================"