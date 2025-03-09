"""
GitHub Copilot integration module.

This module provides functionality to integrate with GitHub Copilot's API
for test generation and collaborative programming.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from qa_agent.config import QAAgentConfig
from qa_agent.error_recovery import CircuitBreaker, ErrorHandler
from qa_agent.models import CodeFile, Function
from qa_agent.utils.logging import log_exception

logger = logging.getLogger(__name__)


class CopilotAdapter:
    """Adapter for interfacing with GitHub Copilot API."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the GitHub Copilot adapter.

        Args:
            config: Configuration object with Copilot settings
        """
        if config.model_provider != "github-copilot":
            raise ValueError("Model provider must be 'github-copilot' to use CopilotAdapter")

        self.api_key = config.api_key
        self.settings = config.copilot_settings or {}
        self.endpoint = self.settings.get("endpoint", "https://api.github.com/copilot")
        self.model_version = self.settings.get("model_version", "latest")
        self.max_tokens = self.settings.get("max_tokens", 2048)
        self.temperature = self.settings.get("temperature", 0.1)
        self.collaborative_mode = self.settings.get("collaborative_mode", True)

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Set up error handling components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=60, half_open_max_calls=2
        )

        self.error_handler = ErrorHandler(
            max_retries=3, backoff_factor=1.5, circuit_breaker=self.circuit_breaker
        )

        logger.info(f"Initialized GitHub Copilot adapter with model version: {self.model_version}")

    def generate_test(
        self, function: Function, context_files: Optional[List[CodeFile]] = None
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Generate a test using GitHub Copilot.

        Args:
            function: The function to generate a test for
            context_files: Additional files for context

        Returns:
            Tuple of (test_code, imports, mocks, fixtures)
        """
        logger.info(f"Generating test for function {function.name} using GitHub Copilot")

        # Prepare context from additional files
        context = ""
        if context_files:
            for file in context_files:
                context += f"\n# File: {file.path}\n{file.content}\n"

        # Create the prompt for Copilot
        prompt = self._create_test_generation_prompt(function, context)

        # Make the API call to Copilot
        response = self._call_copilot_api(prompt)

        # Parse the response
        test_code, imports, mocks, fixtures = self._parse_copilot_response(response)

        return test_code, imports, mocks, fixtures

    def _create_test_generation_prompt(self, function: Function, context: str) -> str:
        """
        Create a prompt for test generation.

        Args:
            function: The function to generate a test for
            context: Additional context

        Returns:
            Prompt string for Copilot
        """
        prompt = f"""
Generate a comprehensive pytest unit test for the following Python function:

# Function Information
- Name: {function.name}
- File: {function.file_path}

# Function Code:
```python
{function.code}
```

# Additional Context:
{context}

# Requirements:
1. Use pytest for testing
2. Include comprehensive assertions to verify the function's behavior
3. Add appropriate mocks or fixtures if necessary
4. Include test cases for edge cases and error handling
5. Provide clear comments explaining your test strategy

Please provide the complete test code.
"""
        return prompt

    def _call_copilot_api(self, prompt: str) -> str:
        """
        Call the GitHub Copilot API with the provided prompt.

        Args:
            prompt: The prompt to send to Copilot

        Returns:
            The response from Copilot
        """
        url = f"{self.endpoint}/completions"

        payload = {
            "model": self.model_version,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1,
            "n": 1,
            "stream": False,
        }

        def api_call():
            """Make the actual API call with proper error handling."""
            try:
                api_response = requests.post(url, headers=self.headers, json=payload)
                api_response.raise_for_status()

                data = api_response.json()
                return data.get("choices", [{}])[0].get("text", "")
            except requests.RequestException as e:
                # Safely extract status code from response if it exists
                status_code = None
                if hasattr(e, "response") and e.response:
                    if hasattr(e.response, "status_code"):
                        status_code = e.response.status_code

                error_context = {
                    "url": url,
                    "model": self.model_version,
                    "status_code": status_code,
                    "error_type": type(e).__name__,
                }
                log_exception(logger, f"Error calling GitHub Copilot API", e, error_context)
                raise  # Re-raise for ErrorHandler to handle

        try:
            # Use ErrorHandler to handle retries with exponential backoff
            result = self.error_handler.execute_with_retry(
                api_call,
                operation_name="copilot_api_call",
                error_context={"prompt_length": len(prompt)},
                diagnostic_level="detailed",
            )
            return result
        except Exception as e:
            logger.error(f"All retries failed when calling GitHub Copilot API: {str(e)}")
            # After all retries failed, return a fallback message
            return "# Error occurred while generating test with GitHub Copilot after multiple retry attempts"

    def _parse_copilot_response(self, response: str) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Parse the Copilot response to extract test code, imports, mocks, and fixtures.

        Args:
            response: Copilot response

        Returns:
            Tuple of (test_code, imports, mocks, fixtures)
        """
        import re

        # Extract code from the response
        code_match = re.search(r"```python\s+(.*?)\s+```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response.strip()

        # Extract imports
        imports = []
        import_pattern = re.compile(r"^(?:from|import)\s+.*$", re.MULTILINE)
        for match in import_pattern.finditer(code):
            imports.append(match.group(0))

        # Extract mocks (lines containing mock or patch)
        mocks = []
        mock_pattern = re.compile(r"^.*(?:mock|patch).*$", re.MULTILINE | re.IGNORECASE)
        for match in mock_pattern.finditer(code):
            line = match.group(0)
            if not line.strip().startswith("#"):  # Ignore comments
                mocks.append(line)

        # Extract fixtures (pytest fixtures)
        fixtures = []
        fixture_pattern = re.compile(
            r"^@pytest\.fixture.*?def\s+(\w+).*?:", re.MULTILINE | re.DOTALL
        )
        for match in fixture_pattern.finditer(code):
            fixtures.append(match.group(0))

        return code, imports, mocks, fixtures

    def collaborative_test_generation(
        self,
        function: Function,
        context_files: Optional[List[CodeFile]] = None,
        feedback: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Generate a test collaboratively with human feedback.

        Args:
            function: The function to generate a test for
            context_files: Additional files for context
            feedback: Optional feedback from the user or QA agent

        Returns:
            Tuple of (test_code, imports, mocks, fixtures)
        """
        if not self.collaborative_mode:
            logger.warning("Collaborative mode is disabled. Using standard test generation.")
            return self.generate_test(function, context_files)

        logger.info(f"Generating test for function {function.name} in collaborative mode")

        # Prepare context from additional files
        context = ""
        if context_files:
            for file in context_files:
                context += f"\n# File: {file.path}\n{file.content}\n"

        # Create the prompt for Copilot, including feedback if provided
        prompt = self._create_collaborative_prompt(function, context, feedback)

        # Make the API call to Copilot
        response = self._call_copilot_api(prompt)

        # Parse the response
        test_code, imports, mocks, fixtures = self._parse_copilot_response(response)

        return test_code, imports, mocks, fixtures

    def _create_collaborative_prompt(
        self, function: Function, context: str, feedback: Optional[str] = None
    ) -> str:
        """
        Create a prompt for collaborative test generation.

        Args:
            function: The function to generate a test for
            context: Additional context
            feedback: Optional feedback from the user or QA agent

        Returns:
            Prompt string for Copilot
        """
        feedback_section = ""
        if feedback:
            feedback_section = f"""
# Feedback from QA Agent:
{feedback}

Please adjust the test based on this feedback.
"""

        prompt = f"""
Generate a comprehensive pytest unit test for the following Python function in a collaborative session:

# Function Information
- Name: {function.name}
- File: {function.file_path}

# Function Code:
```python
{function.code}
```

# Additional Context:
{context}
{feedback_section}

# Requirements:
1. Use pytest for testing
2. Include comprehensive assertions to verify the function's behavior
3. Add appropriate mocks or fixtures if necessary
4. Include test cases for edge cases and error handling
5. Provide clear comments explaining your test strategy

Please provide the complete test code.
"""
        return prompt

    def refine_test(self, original_test: str, test_result: Dict[str, Any]) -> str:
        """
        Refine a test based on test results.

        Args:
            original_test: The original test code
            test_result: The result of the test execution

        Returns:
            Refined test code
        """
        logger.info("Refining test with GitHub Copilot based on test results")

        success = test_result.get("success", False)
        error_message = test_result.get("error_message", "")

        if success:
            logger.info("Test already successful, no refinement needed")
            return original_test

        # Create a prompt for Copilot to refine the test
        prompt = f"""
I have a failing test that needs to be fixed. Here are the details:

# Original Test Code:
```python
{original_test}
```

# Error Message:
```
{error_message}
```

Please fix the test to make it pass. Provide the complete corrected test code.
"""

        # Make the API call to Copilot
        response = self._call_copilot_api(prompt)

        # Extract code from the response
        import re

        code_match = re.search(r"```python\s+(.*?)\s+```", response, re.DOTALL)
        if code_match:
            refined_code = code_match.group(1).strip()
        else:
            refined_code = response.strip()

        return refined_code
