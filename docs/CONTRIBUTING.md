# Contributing to QA Agent

Thank you for your interest in contributing to QA Agent! This document provides guidelines and instructions for contributing to this project.

## Development Workflow

1. **Fork and Clone**: Fork the repository and clone it to your local machine.

2. **Set Up Development Environment**:
   ```bash
   pip install -e .
   pip install flake8 black isort mypy pytest pytest-cov
   ```

3. **Create a Branch**: Create a branch for your feature or bugfix.

4. **Make Changes**: Make your changes to the codebase.

5. **Format and Lint Your Code**:
   ```bash
   black qa_agent utils tests
   isort qa_agent utils tests
   flake8 qa_agent utils tests
   mypy qa_agent utils tests
   ```

6. **Run Tests**:
   ```bash
   pytest --cov=qa_agent --cov=utils tests/
   ```

7. **Submit a Pull Request**: Push your changes to your fork and submit a pull request.

## CI/CD Pipeline

Our CI/CD pipeline automatically performs the following checks on all pull requests:

1. **Code Quality Checks**:
   - Black formatting
   - Isort import sorting
   - Flake8 linting
   - Mypy type checking

2. **Tests**:
   - Runs all unit and integration tests
   - Generates code coverage reports
   - Uploads coverage to Codecov

3. **Versioning**:
   - Automatically bumps version numbers on merged PRs
   - Creates GitHub releases with semantic versioning

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages to automate versioning and changelog generation:

- `feat(component): add new feature` - for new features (minor version bump)
- `fix(component): resolve issue` - for bug fixes (patch version bump)
- `docs(component): update documentation` - for documentation changes
- `refactor(component): improve code structure` - for code refactoring
- `test(component): add new tests` - for adding or updating tests
- `chore(component): update dependencies` - for maintenance tasks

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all your interactions with the project.

## Questions?

If you have any questions, please open an issue or reach out to the maintainers.