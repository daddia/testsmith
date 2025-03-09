# Test File Naming Conventions

This document explains the language-specific test file naming conventions used in the QA Agent.

## Language-Specific Test File Naming Conventions

The QA Agent automatically applies the following naming conventions when generating test files:

| Language | Source File | Test File | Naming Convention |
|----------|-------------|-----------|-------------------|
| Python | example.py | test_example.py | test_*.py |
| JavaScript | example.js | example.test.js | *.test.js |
| TypeScript | example.ts | example.test.ts | *.test.ts |
| Go | example.go | example_test.go | *_test.go |
| PHP | example.php | ExampleTest.php | *Test.php |
| SQL | example.sql | example.test.sql | *.test.sql |

## Rationale for Each Convention

### Python (`test_*.py`)
- Standard naming convention used by pytest
- Files prefixed with `test_` are automatically discovered by pytest's test collection mechanism
- Widely adopted in the Python ecosystem

### JavaScript/TypeScript (`*.test.js` / `*.test.ts`)
- Standard pattern used by Jest, React Testing Library, and other modern JS testing frameworks
- The `.test.` suffix provides clear association with the source file
- Modern JavaScript testing tools automatically discover tests with this pattern

### Go (`*_test.go`)
- Required by Go's standard library testing package
- The Go toolchain expects tests to be in files with a `_test` suffix
- This is a strict convention in the Go ecosystem, not just a preference

### PHP (`*Test.php`)
- Standard convention for PHPUnit
- Uses CamelCase with a `Test` suffix per PHP community standards
- Automatically discovered by PHPUnit test runners

### SQL (`*.test.sql`)
- Similar to JS pattern for consistency
- Works well with pgTAP and other SQL testing frameworks
- Clearly identifies test files in SQL codebases

## Implementation in the QA Agent

The QA Agent implements these conventions in the `_get_test_file_path` method of the `TestGenerator` class:

```python
def _get_test_file_path(self, function: Function) -> str:
    # Extract module name and file extension
    file_basename = os.path.basename(function.file_path)
    module_name, file_extension = os.path.splitext(file_basename)
    
    # Mapping of file extensions to language-specific naming patterns
    naming_conventions = {
        # Python: test_*.py
        '.py': lambda name: f"test_{name}.py",
        
        # JavaScript/TypeScript: *.test.js
        '.js': lambda name: f"{name}.test.js",
        '.jsx': lambda name: f"{name}.test.jsx",
        '.ts': lambda name: f"{name}.test.ts",
        '.tsx': lambda name: f"{name}.test.tsx",
        
        # Go: *_test.go
        '.go': lambda name: f"{name}_test.go",
        
        # PHP: *Test.php (PHPUnit convention)
        '.php': lambda name: f"{name[0].upper() + name[1:]}Test.php",
        
        # SQL: *.test.sql
        '.sql': lambda name: f"{name}.test.sql"
    }
    
    # Determine language from file extension and generate test file name
    if file_extension.lower() in naming_conventions:
        test_file_name = naming_conventions[file_extension.lower()](module_name)
    else:
        # Default to language-agnostic naming for unsupported extensions
        test_file_name = f"test_{module_name}{file_extension}"
    
    return os.path.join(self.config.output_directory, test_file_name)
```

## Fallback Behavior

For unsupported languages, the QA Agent defaults to the Python-style naming convention (`test_*.*`), which is widely recognized across different programming ecosystems.