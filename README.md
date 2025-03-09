# QA Agent

QA Agent is an advanced Python-based tool that leverages AI to automatically analyze, generate, and validate unit tests, enhancing code quality and test coverage with intelligent insights.

## Features

- 🤖 **AI-driven test generation**: Automatically creates unit tests for your functions
- 📊 **Code coverage analysis**: Identifies functions with poor or no test coverage
- 🔍 **Intelligent test validation**: Validates tests and suggests fixes for failing tests
- 🔄 **Continuous improvement**: Uses feedback from test failures to improve test generation
- 🌍 **Sourcegraph integration**: Enhances test generation with global code intelligence (NEW!)

## Installation

```bash
# Clone the repository
git clone https://github.com/daddia/qa-agent-dev.git
cd qa-agent-dev

# Install dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
python main.py --repo-path ./your_project --output ./generated_tests
```

### With Sourcegraph Integration

```bash
# Enable Sourcegraph integration
python main.py --repo-path ./your_project --enable-sourcegraph --sourcegraph-token YOUR_TOKEN
```

### All Options

```bash
python main.py --help
```

Output:
```
usage: main.py [-h] [--repo-path REPO_PATH] [--config CONFIG] [--verbose]
              [--output OUTPUT] [--target-coverage TARGET_COVERAGE]
              [--enable-sourcegraph] [--sourcegraph-endpoint SOURCEGRAPH_ENDPOINT]
              [--sourcegraph-token SOURCEGRAPH_TOKEN]

QA Agent for test generation and validation

options:
  -h, --help            show this help message and exit
  --repo-path REPO_PATH, -r REPO_PATH
                        Path to the repository
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --verbose, -v         Enable verbose logging
  --output OUTPUT, -o OUTPUT
                        Output directory for generated tests
  --target-coverage TARGET_COVERAGE, -t TARGET_COVERAGE
                        Target coverage percentage

Sourcegraph Integration:
  --enable-sourcegraph  Enable Sourcegraph integration for enhanced context gathering
  --sourcegraph-endpoint SOURCEGRAPH_ENDPOINT
                        Sourcegraph API endpoint (default: https://sourcegraph.com/.api)
  --sourcegraph-token SOURCEGRAPH_TOKEN
                        Sourcegraph API token (can also use SOURCEGRAPH_API_TOKEN env var)
```

## Documentation

- [Sourcegraph Integration](docs/SOURCEGRAPH_INTEGRATION.md)
- [TODO List](docs/TODO.md)

## How It Works

1. **Code Analysis**: QA Agent analyzes your codebase to identify functions with poor or no test coverage
2. **Context Gathering**: It gathers context for each function, including related files and code examples
3. **Test Generation**: Using AI models, it generates unit tests tailored to each function
4. **Test Validation**: It runs the generated tests to validate them
5. **Test Refinement**: For failing tests, it analyzes the failures and suggests fixes

With Sourcegraph integration, the context gathering step is enhanced with global code intelligence, resulting in better test generation.

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (if using Claude models)
- `SOURCEGRAPH_API_TOKEN`: Your Sourcegraph API token

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: All PRs and commits are automatically tested
- **Code Quality Checks**: Enforces code formatting and style with Black, isort, and flake8
- **Type Checking**: Validates type annotations with mypy
- **Security Scanning**: Uses CodeQL to detect security vulnerabilities
- **Coverage Reporting**: Tracks test coverage with pytest-cov and Codecov
- **Automated Versioning**: Implements semantic versioning with automatic version bumping

Status badges:

[![CI/CD Pipeline](https://github.com/daddia/qa-agent-dev/actions/workflows/main.yml/badge.svg)](https://github.com/daddia/qa-agent-dev/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/daddia/qa-agent-dev/branch/main/graph/badge.svg)](https://codecov.io/gh/daddia/qa-agent-dev)
[![CodeQL](https://github.com/daddia/qa-agent-dev/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/daddia/qa-agent-dev/actions/workflows/codeql-analysis.yml)

## Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit contributions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.