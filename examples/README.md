# QA Agent Examples Directory

This directory contains language-specific examples to demonstrate the test generation capabilities of the QA Agent across different programming languages. The examples showcase how the QA Agent adapts to each language's ecosystem, conventions, and testing frameworks.

## Directory Structure

```
examples/
├── display_structure.sh            # Script to display the example directory structure
├── demonstrate_test_naming.py      # Script to demonstrate test naming conventions
├── call_graph_demo.py              # Demo for call graph analysis
├── complexity_analysis_demo.py     # Demo for cognitive complexity analysis
├── python/                         # Python examples
│   ├── example_src/                # Source files
│   └── generated_tests/            # Generated test files
├── js/                             # JavaScript examples
│   ├── example_src/                # Source files
│   └── generated_tests/            # Generated test files
├── php/                            # PHP examples
│   ├── example_src/                # Source files
│   └── generated_tests/            # Generated test files
├── go/                             # Go examples
│   ├── example_src/                # Source files
│   └── generated_tests/            # Generated test files
└── sql/                            # SQL examples
    ├── example_src/                # Source files
    └── generated_tests/            # Generated test files
```

## Language-Specific Test File Naming Conventions

The QA Agent follows language-specific conventions for test file naming to ensure generated tests integrate seamlessly with standard tooling and workflows:

### Python

- **Convention**: `test_*.py`
- **Reasoning**: This naming pattern is the standard for pytest, the most popular Python testing framework. Files prefixed with `test_` are automatically discovered by pytest's test collection mechanism.
- **Example**: `example.py` → `test_example.py`

### JavaScript/TypeScript

- **Convention**: `*.test.js` / `*.test.ts`
- **Reasoning**: This is the standard pattern used by Jest, React Testing Library, and most modern JavaScript testing frameworks. The `.test.` suffix is easier to match with the source file and follows modern JS conventions.
- **Example**: `utils.js` → `utils.test.js`

### PHP

- **Convention**: `*Test.php`
- **Reasoning**: PHPUnit, the standard testing framework for PHP, uses this CamelCase naming convention. The `Test` suffix is standard in the PHP ecosystem.
- **Example**: `User.php` → `UserTest.php`

### Go

- **Convention**: `*_test.go`
- **Reasoning**: Go's standard library testing package requires test files to end with `_test.go`. This convention is enforced by the Go testing ecosystem and build tools.
- **Example**: `parser.go` → `parser_test.go`

### SQL

- **Convention**: `*.test.sql`
- **Reasoning**: SQL testing is less standardized, but this convention clearly identifies test files and aligns with JavaScript's `.test.` pattern for consistency. Works well with pgTap and other SQL testing frameworks.
- **Example**: `functions.sql` → `functions.test.sql`

## Demo Scripts

- **display_structure.sh**: Run this script to display the directory structure and count source/test files for each language.
- **demonstrate_test_naming.py**: Run this script to see examples of the naming conventions in action.

## Usage

To generate tests for these example files, you can use the cleanup_example_tests.py script in the tests directory:

```bash
python tests/cleanup_example_tests.py
```

This will clean all existing test files and regenerate them using the QA Agent's test generator with proper language-specific naming conventions.

## Test Generation Process

For each language:

1. The QA Agent detects the language based on file extension
2. It parses the source code to extract functions/methods
3. It applies language-specific testing practices and conventions
4. It generates tests using the appropriate naming convention
5. The tests are saved to the language's generated_tests directory

## Error Handling

The QA Agent includes robust error handling to manage issues that may arise during test generation for different languages:

- Fallback naming conventions are used if a specific language isn't fully supported
- Detailed error logs are generated during the test creation process
- Validation ensures the generated tests match the language-specific syntax and framework requirements