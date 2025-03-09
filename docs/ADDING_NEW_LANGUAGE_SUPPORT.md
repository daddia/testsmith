# Adding Support for a New Programming Language

This document provides instructions for extending the QA Agent to support additional programming languages beyond the currently supported ones (Python, JavaScript/TypeScript, PHP, Go, SQL).

## Overview

Adding support for a new programming language involves several components:

1. Creating a language-specific parser in `qa_agent/parser.py`
2. Adding the language's file extension to the test naming conventions in `qa_agent/test_generator.py`
3. Creating language-specific test templates in `qa_agent/prompts/`
4. Adding example files to the `examples/` directory

## Step 1: Add a Language-Specific Parser

In `qa_agent/parser.py`, create a new parser class that extends the `BaseCodeParser` class:

```python
class NewLanguageCodeParser(BaseCodeParser):
    """Parser for NewLanguage code files."""

    def extract_functions(self, code_file: CodeFile) -> List[Function]:
        """
        Extract functions from a NewLanguage code file.
        
        Args:
            code_file: The code file to parse
            
        Returns:
            List of Function objects
        """
        functions = []
        
        # Implement language-specific parsing logic here
        # This might use regex patterns, AST parsing, or other techniques
        # specific to the language
        
        return functions
        
    def find_function_end_line(self, code_lines: List[str], start_line: int) -> int:
        """
        Find the end line of a function.
        
        Args:
            code_lines: Lines of code
            start_line: Start line of the function
            
        Returns:
            End line of the function
        """
        # Implement logic to determine where a function ends in this language
        pass
        
    def calculate_complexity(self, function_body: str) -> int:
        """
        Calculate cyclomatic complexity of a function.
        
        Args:
            function_body: Body of the function
            
        Returns:
            Cyclomatic complexity score
        """
        # Implement complexity calculation for the language
        pass
```

Then update the `CodeParserFactory` to recognize the new language:

```python
@staticmethod
def get_parser(file_path: str) -> BaseCodeParser:
    """
    Get a parser for a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Code parser instance
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.py':
        return PythonCodeParser()
    elif ext in ('.js', '.jsx', '.ts', '.tsx'):
        return JavaScriptCodeParser()
    elif ext == '.php':
        return PHPCodeParser()
    elif ext == '.go':
        return GoCodeParser()
    elif ext == '.sql':
        return SQLCodeParser()
    elif ext == '.your_extension':  # Add your new extension here
        return NewLanguageCodeParser()
    else:
        return BaseCodeParser()  # Fallback to a basic parser
```

## Step 2: Add Test Naming Convention

In `qa_agent/test_generator.py`, update the `_get_test_file_path` method to include the naming convention for your new language:

```python
naming_conventions = {
    # Existing conventions...
    
    # Add your new language here
    '.your_extension': lambda name: f"your_naming_convention_{name}.your_extension",
}
```

Choose a naming convention that follows the language's community standards for test files.

## Step 3: Create Language-Specific Test Templates

If your language requires a specific test format or framework, create a test template for it in `qa_agent/prompts/`:

```
# Test Template for NewLanguage

This template is used to generate tests for NewLanguage.

## Basic Template Structure

```new_language
// Import the testing framework
import TestFramework from 'your-test-framework';

// Test class or suite
TestSuite('{{function_name}}Tests', function() {
    // Test cases
    Test('should behave correctly with valid input', function() {
        // Arrange
        const input = ...;
        
        // Act
        const result = {{function_name}}(input);
        
        // Assert
        Assert.equals(expected, result);
    });
});
```

Then update the prompt template in the `_create_test_generation_prompt` method to include language-specific instructions.

## Step 4: Add Example Files

1. Create a directory structure for examples:

```
examples/new_language/
├── example_src/        # Source files
│   ├── simple.ext      # Simple example
│   └── complex.ext     # Complex example
└── generated_tests/    # Generated test files
```

2. Add sample source files that showcase key language features.

3. Run the test generator on these files to create example test files.

4. Update the `examples/README.md` to include your new language.

## Step 5: Add Unit Tests

Create tests to verify that the new language parser works correctly:

1. Add test cases to `tests/unit/test_parser.py` for the new parser.
2. Add test cases to `tests/unit/test_test_file_naming.py` for the new naming convention.
3. Add example-based tests to `tests/unit/test_parser_examples.py` using your sample files.

## Step 6: Update Documentation

1. Update `examples/README.md` to include the new language.
2. Update `examples/test_naming_conventions.md` with the new language.
3. Add any language-specific notes to other documentation files.

## Example: Adding Ruby Support

Here's a brief example of adding support for Ruby:

1. Create a `RubyCodeParser` class in `parser.py`:

```python
class RubyCodeParser(BaseCodeParser):
    """Parser for Ruby code files."""

    def extract_functions(self, code_file: CodeFile) -> List[Function]:
        functions = []
        # Parse Ruby methods using regex
        method_pattern = re.compile(r'def\s+(\w+).*?end', re.DOTALL)
        for match in method_pattern.finditer(code_file.content):
            # Extract method details and add to functions list
            # ...
        return functions
```

2. Add Ruby test naming convention:

```python
naming_conventions = {
    # ...
    '.rb': lambda name: f"{name}_test.rb",  # Ruby convention
}
```

3. Create examples:

```
examples/ruby/
├── example_src/
│   ├── simple.rb
│   └── complex.rb
└── generated_tests/
    ├── simple_test.rb
    └── complex_test.rb
```

4. Update the tests and documentation accordingly.

## Conclusion

By following these steps, you can extend the QA Agent to support any programming language with its own specific parsing logic and test generation conventions.