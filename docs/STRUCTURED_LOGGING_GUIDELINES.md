# Structured Logging Guidelines

This document outlines best practices for using the structured logging system in QA Agent.

## Overview

QA Agent uses `structlog` to provide contextual, structured logs that are easier to parse, filter, and analyse. This approach allows for better debugging, monitoring, and tracking of the agent's operations.

## Key Benefits

- **Context-rich logs**: Each log entry contains relevant metadata about the operation
- **Consistent format**: All logs follow a standard format for easier parsing
- **Hierarchical context**: Context is inherited across log calls within the same scope
- **Machine-readable**: JSON output format is easily consumed by log aggregators

## Basic Usage

Always import and use the logging utilities from `utils.logging`:

```python
from utils.logging import get_logger, add_context_to_logger

# Create a logger with the module name
logger = get_logger(__name__)

# Add permanent context to the logger
logger = add_context_to_logger(logger, component="TestGenerator", function_name="generate_test")

# Log with additional context
logger.info("Starting test generation", function=func_name, complexity=complexity)
```

## Logging Levels

Use the appropriate log level based on the importance of the message:

| Level | When to Use |
| ----- | ----------- |
| `debug` | Detailed information, useful only for diagnosing issues |
| `info` | General information about normal operation |
| `warning` | Indication of a potential issue that doesn't prevent operation |
| `error` | A significant issue that disrupts operation but doesn't crash the system |
| `critical` | A severe issue that may lead to application failure |

## Helper Functions

QA Agent provides specialised logging helper functions for common operations:

```python
from utils.logging import (
    log_function_call,
    log_function_result,
    log_exception,
    log_opened,
    log_parsed,
    log_redacted,
    log_generated,
    log_validated,
    log_analyzed,
    log_edited,
    log_executed
)

# Log a function call with arguments
log_function_call(logger, "generate_test", args=(function,), kwargs={"context_files": context_files})

# Log the result of a function
log_function_result(logger, "generate_test", result, execution_time=duration)

# Log an exception with context
try:
    # Some operation
except Exception as e:
    log_exception(logger, "generate_test", e, context={"function": function.name})

# Log when a file is opened
log_opened(logger, file_path)

# Log when content is redacted for IP protection
log_redacted(logger, file_path, function_name="calculate_pricing")

# Log when a test is generated
log_generated(logger, output_path)

# Log when a test is validated
log_validated(logger, test_path, success=True, coverage=85.5)
```

## Context Guidelines

Include relevant context in your logs to make them more useful:

- **Always include**: Component name, function/method name, operation type
- **When available**: File paths, function names, test names, execution times
- **For errors**: Exception details, traceback information
- **For operations with files**: File paths, line numbers
- **For API calls**: Request parameters (excluding sensitive data)

## Example Logging Patterns

### Function Entry and Exit

```python
def process_file(file_path):
    logger = get_logger(__name__)
    log_function_call(logger, "process_file", args=(file_path,))
    
    start_time = time.time()
    try:
        # Function logic here
        result = do_processing(file_path)
        
        duration = time.time() - start_time
        log_function_result(logger, "process_file", result, execution_time=duration)
        return result
    except Exception as e:
        log_exception(logger, "process_file", e, context={"file_path": file_path})
        raise
```

### Tracking Test Generation

```python
def generate_test(function, context_files=None):
    logger = get_logger(__name__)
    logger = add_context_to_logger(logger, function_name=function.name)
    
    logger.info("Starting test generation", 
                file_path=function.file_path, 
                context_count=len(context_files or []))
    
    # Generate test logic
    
    logger.info("Test generation complete", 
                output_file=output_path,
                test_lines=len(test_code.split('\n')))
    
    log_generated(logger, output_path)
```

## Best Practices

1. **Be selective**: Log meaningful events, not everything
2. **Add context**: Include relevant metadata with each log
3. **Use consistent names**: Standardise field names across the codebase
4. **Don't log sensitive data**: Avoid API keys, credentials, or personal information
5. **Structure hierarchically**: Use nested objects for complex data
6. **Use helper functions**: Prefer helper functions for common logging patterns
7. **Log exceptions**: Always log exceptions with appropriate context

## Log Processing

Structured logs can be easily processed and analysed:

```python
# Filter logs by component
jq '.component == "TestGenerator"' logs.json

# Find all errors
jq 'select(.level == "error")' logs.json

# Find logs for a specific function
jq 'select(.function_name == "calculate_complexity")' logs.json
```