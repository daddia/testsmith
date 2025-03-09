"""
Test validation module.

This module provides functionality to run and validate generated tests.
"""

import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import structlog

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableSequence
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback for when imports aren't available
    class ChatOpenAI:
        def __init__(self, **kwargs):
            pass

    class RunnableSequence:
        @classmethod
        def from_components(cls, **kwargs):
            return cls()

        def invoke(self, **kwargs):
            return ""

    class PromptTemplate:
        def __init__(self, template, input_variables):
            self.template = template
            self.input_variables = input_variables

    class StrOutputParser:
        def parse(self, text):
            return text

    class ChatAnthropic:
        def __init__(self, **kwargs):
            pass


from qa_agent.config import QAAgentConfig
from qa_agent.copilot_adapter import CopilotAdapter
from qa_agent.models import GeneratedTest, TestResult

# Get structured logger
logger = structlog.get_logger(__name__)


class TestValidator:
    """Validates generated tests by running them and checking for errors."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the test validator.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize Copilot adapter if needed
        self.copilot_adapter = None
        if config.model_provider == "github-copilot":
            self.copilot_adapter = CopilotAdapter(config)

    def validate_test(self, test: GeneratedTest) -> TestResult:
        """
        Validate a generated test by running it.

        Args:
            test: The generated test to validate

        Returns:
            TestResult object
        """
        logger.info(f"Validating test: {test.test_file_path}")

        # Initialize IP protector if enabled
        ip_protector = None
        if hasattr(self.config, "ip_protection_enabled") and self.config.ip_protection_enabled:
            from qa_agent.ip_protector import IPProtector

            ip_protector = IPProtector(
                protected_patterns=self.config.protected_patterns,
                protected_functions=self.config.protected_functions,
                protected_files=self.config.protected_files,
            )
            # Load rules from file if specified
            if (
                hasattr(self.config, "ip_protection_rules_path")
                and self.config.ip_protection_rules_path
            ):
                ip_protector.load_protection_rules(self.config.ip_protection_rules_path)

        # Ensure the test file exists
        if not os.path.exists(test.test_file_path):
            logger.error(f"Test file does not exist: {test.test_file_path}")
            return TestResult(
                success=False,
                test_file=test.test_file_path,
                target_function=test.function.name,
                output="",
                error_message="Test file does not exist",
            )

        # Remember the original directory
        original_dir = os.getcwd()

        try:
            # Change to the repository directory
            if self.config.repo_path:
                os.chdir(self.config.repo_path)

            # Get the path relative to the current directory
            rel_path = os.path.relpath(test.test_file_path)

            # Set up the command for running the test
            if self.config.test_framework == "pytest":
                # Add coverage options if available
                coverage_options = []
                if any(pkg in sys.modules for pkg in ["pytest_cov", "pytest-cov"]):
                    coverage_options = ["--cov=."]

                    # Add specific coverage source if specified
                    if hasattr(self.config, "coverage_source") and self.config.coverage_source:
                        coverage_options = [f"--cov={self.config.coverage_source}"]

                cmd = ["python", "-m", "pytest", rel_path, "-v", "--no-header"] + coverage_options
                logger.debug(f"Running pytest with command: {' '.join(cmd)}")
            else:
                raise ValueError(f"Unsupported test framework: {self.config.test_framework}")

            start_time = time.time()

            # Run the test
            result = subprocess.run(cmd, capture_output=True, text=True)

            end_time = time.time()
            execution_time = end_time - start_time

            # Change back to the original directory
            os.chdir(original_dir)

            # Parse the test output
            success = result.returncode == 0
            output = result.stdout
            error_message = result.stderr if result.stderr else None

            # Apply IP protection to output if enabled
            if ip_protector is not None:
                if output:
                    output = ip_protector.protect(output)
                if error_message:
                    error_message = ip_protector.protect(error_message)

            # Try to extract coverage information
            coverage = self._extract_coverage(output)

            # Log result with appropriate messaging
            if success:
                if coverage is not None:
                    logger.info(
                        f"Test validation successful with {coverage:.2f}% coverage",
                        test_file=test.test_file_path,
                        target_function=test.function.name,
                    )
                else:
                    logger.info(
                        "Test validation successful",
                        test_file=test.test_file_path,
                        target_function=test.function.name,
                    )
            else:
                if coverage is not None:
                    logger.warning(
                        f"Validation failed with {coverage:.2f}% coverage",
                        test_file=test.test_file_path,
                        target_function=test.function.name,
                    )
                else:
                    logger.warning(
                        "Test validation failed",
                        test_file=test.test_file_path,
                        target_function=test.function.name,
                    )

            return TestResult(
                success=success,
                test_file=test.test_file_path,
                target_function=test.function.name,
                output=output,
                error_message=error_message,
                coverage=coverage,
                execution_time=execution_time,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running test: {e.stderr}")
            return TestResult(
                success=False,
                test_file=test.test_file_path,
                target_function=test.function.name,
                output=e.stdout,
                error_message=e.stderr,
            )
        except Exception as e:
            logger.error(f"Error validating test: {str(e)}")
            return TestResult(
                success=False,
                test_file=test.test_file_path,
                target_function=test.function.name,
                output="",
                error_message=str(e),
            )
        finally:
            # Ensure we change back to the original directory even if there's an error
            if "original_dir" in locals():
                os.chdir(original_dir)

    def _extract_coverage(self, output: str) -> Optional[float]:
        """
        Extract coverage percentage from pytest output.

        Args:
            output: Test output

        Returns:
            Coverage percentage or None if not found
        """
        import re

        # Look for coverage information in the output using multiple patterns
        # Pattern 1: Standard pytest-cov format
        # Example: "TOTAL                             12      2    83%"
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if coverage_match:
            return float(coverage_match.group(1))

        # Pattern 2: Simple coverage percentage
        # Example: "Coverage: 75.5%"
        coverage_match = re.search(r"Coverage:\s*(\d+\.\d+)%", output)
        if coverage_match:
            return float(coverage_match.group(1))

        # Pattern 3: Percentage at the end of line
        # Example: "===== 100% coverage ====="
        coverage_match = re.search(r"=*\s+(\d+(?:\.\d+)?)%\s+coverage\s+=*", output)
        if coverage_match:
            return float(coverage_match.group(1))

        # Pattern 4: Just a percentage with coverage keyword
        # Example: "75% coverage"
        coverage_match = re.search(r"(\d+(?:\.\d+)?)%\s+coverage", output)
        if coverage_match:
            return float(coverage_match.group(1))

        return None

    def fix_test(self, test: GeneratedTest, test_result: TestResult) -> GeneratedTest:
        """
        Attempt to fix a failing test using LLM or GitHub Copilot.

        Args:
            test: The failing test
            test_result: The result of the test validation

        Returns:
            Fixed GeneratedTest object
        """
        logger.info(f"Attempting to fix failing test: {test.test_file_path}")

        # Get IP protector instance for protecting sensitive code
        ip_protector = None
        if self.config.ip_protection_enabled:
            from qa_agent.ip_protector import IPProtector

            ip_protector = IPProtector(
                protected_patterns=self.config.protected_patterns,
                protected_functions=self.config.protected_functions,
                protected_files=self.config.protected_files,
            )
            # Load rules from file if specified
            if (
                hasattr(self.config, "ip_protection_rules_path")
                and self.config.ip_protection_rules_path
            ):
                ip_protector.load_protection_rules(self.config.ip_protection_rules_path)
        else:
            from qa_agent.ip_protector import IPProtector

            ip_protector = IPProtector()  # Default empty instance

        # Use appropriate model provider or adapter
        if self.config.model_provider == "github-copilot":
            # Use GitHub Copilot for test fixing
            fixed_test_code = self._fix_with_copilot(test, test_result, ip_protector)
        else:
            # Use LLM for test fixing
            fixed_test_code = self._fix_with_llm(test, test_result, ip_protector)

        # Create a new GeneratedTest with the fixed code
        fixed_test = GeneratedTest(
            function=test.function,
            test_code=fixed_test_code,
            test_file_path=test.test_file_path,
            imports=test.imports,
            mocks=test.mocks,
            fixtures=test.fixtures,
        )

        return fixed_test

    def _fix_with_llm(self, test: GeneratedTest, test_result: TestResult, ip_protector=None) -> str:
        """
        Fix a test using LLM.

        Args:
            test: The failing test
            test_result: The test result
            ip_protector: Optional IP protector instance

        Returns:
            Fixed test code
        """
        # Initialize appropriate LLM based on provider
        if self.config.model_provider == "openai":
            llm = ChatOpenAI(
                temperature=0,
                model=self.config.model_name,
                api_key=self.config.api_key,
            )
        elif self.config.model_provider == "anthropic":
            llm = ChatAnthropic(
                temperature=0,
                model=self.config.model_name,
                api_key=self.config.api_key,
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config.model_provider}")

        # Protect function code if IP protector is available
        function_code = test.function.code
        if ip_protector is not None:
            function_code = ip_protector.protect(function_code)
            logger.debug("IP protection applied to function code")

        # Create prompt for fixing the test
        template = """
You are a professional software quality assurance engineer specializing in automated testing.
Your task is to fix a failing unit test.

# Function Information
- Name: {function_name}
- File: {function_file_path}

# Original Function:
```python
{function_code}
```

# Failed Test:
```python
{test_code}
```

# Error Information:
```
{error_message}
```

# Test Output:
```
{test_output}
```

Please analyze the failing test and fix the issues. Provide the corrected test code.

# Response Format:
```python
# Corrected test code here
```
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "function_name",
                "function_file_path",
                "function_code",
                "test_code",
                "error_message",
                "test_output",
            ],
        )

        # Prepare parameters with IP protection if applicable
        params = {
            "function_name": test.function.name,
            "function_file_path": test.function.file_path,
            "function_code": function_code,
            "test_code": test.test_code,
            "error_message": test_result.error_message or "No error message available",
            "test_output": test_result.output or "No test output available",
        }

        # If IP protector is available, protect the entire prompt
        if ip_protector is not None:
            # Protect each parameter value
            for key, value in params.items():
                if isinstance(value, str):
                    params[key] = ip_protector.protect(value)

        # Create a RunnableSequence instead of LLMChain
        chain = RunnableSequence.from_components(prompt, llm, StrOutputParser())

        # Use invoke() instead of run()
        response = chain.invoke(params)

        # Extract the fixed test code from the response
        return self._extract_code_from_response(response)

    def _fix_with_copilot(
        self, test: GeneratedTest, test_result: TestResult, ip_protector=None
    ) -> str:
        """
        Fix a test using GitHub Copilot.

        Args:
            test: The failing test
            test_result: The test result
            ip_protector: Optional IP protector instance

        Returns:
            Fixed test code
        """
        # Check if we already have a Copilot adapter instance
        if self.copilot_adapter is None:
            # Initialize Copilot adapter if needed
            self.copilot_adapter = CopilotAdapter(self.config)

        # Protect function code if IP protector is available
        function_code = test.function.code
        if ip_protector is not None:
            function_code = ip_protector.protect(function_code)
            logger.debug("IP protection applied to function code for Copilot")

        # Convert TestResult to a dict for the Copilot adapter
        test_result_dict = {
            "function_name": test.function.name,
            "function_file_path": test.function.file_path,
            "function_code": function_code,
            "test_file_path": test.test_file_path,
            "success": test_result.success,
            "coverage": test_result.coverage,
            "error_message": test_result.error_message or "No error message available",
            "output": test_result.output or "No test output available",
        }

        # If IP protector is available, protect the string values in the dict
        if ip_protector is not None:
            for key, value in test_result_dict.items():
                if isinstance(value, str):
                    test_result_dict[key] = ip_protector.protect(value)

        # Use Copilot to refine the test
        return self.copilot_adapter.refine_test(test.test_code, test_result_dict)

    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from LLM response.

        Args:
            response: LLM response string

        Returns:
            Extracted code
        """
        import re

        # Look for Python code blocks
        code_match = re.search(r"```python\s+(.*?)\s+```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # If no code blocks found, try to extract any code-like content
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith("```python"):
                in_code = True
                continue
            elif line.strip() == "```" and in_code:
                in_code = False
                continue
            elif in_code:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        # If still no code found, return the entire response
        return response
