# QA Agent End-to-End Tests

This directory contains end-to-end tests for the QA Agent system. These tests verify that the entire system works together correctly and capture real-world usage scenarios.

## Test Structure

The end-to-end tests are organized into several categories:

1. **Workflow Tests**: Test the complete QA Agent workflow from code analysis to test generation and validation
2. **CLI Tests**: Test the command-line interface for the QA Agent
3. **Component Integration Tests**: Test the integration between different components of the system
4. **Real Project Tests**: Test the QA Agent on real project files from the QA Agent codebase itself

## Running the Tests

To run all end-to-end tests:

```bash
pytest tests/e2e -v
```

To run a specific test file:

```bash
pytest tests/e2e/test_e2e_workflow.py -v
```

To run tests with the "e2e" marker:

```bash
pytest -m e2e -v
```

## Test Dependencies

The end-to-end tests use fixtures defined in `conftest.py` to set up test environments, including:

- Sample repository with Python code for testing
- Mock configuration for controlled testing
- Mock API responses to avoid external API calls
- Test utilities for verification

## Adding New Tests

When adding new end-to-end tests:

1. Use the `@pytest.mark.e2e` decorator to mark them as end-to-end tests
2. Follow the pattern of existing tests for consistency
3. Use mocks to avoid external dependencies when appropriate
4. Include proper cleanup to avoid test artifacts

## Test Data

The tests use a combination of:

- Dynamically generated test code repositories
- The actual QA Agent codebase for real-world testing
- Mock API responses for controlled testing

## Troubleshooting

If end-to-end tests are failing, check:

1. API key environment variables if testing with real APIs
2. File permissions for temporary directories
3. Network connectivity for tests that use Sourcegraph integration