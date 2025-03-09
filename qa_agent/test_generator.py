"""
Test generation module.

This module provides functionality to generate unit tests for functions.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from qa_agent.ip_protector import IPProtector  # Added IPProtector import
from qa_agent.models import CodeFile, Function, GeneratedTest
from qa_agent.parser import CodeParserFactory
from qa_agent.utils.file import clean_code_for_llm

logger = logging.getLogger(__name__)


class TestGenerator:
    """Generates unit tests for functions."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the test generator.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize IP protector with configuration settings if enabled
        if config.ip_protection_enabled:
            # Pass the config object directly to IPProtector
            self.ip_protector = IPProtector(config)
            # Load rules from file if specified
            if config.ip_protection_rules_path:
                self.ip_protector.load_protection_rules(config.ip_protection_rules_path)
        else:
            self.ip_protector = IPProtector()

        # Load unit test rules during initialization
        self.unit_test_rules = self._load_unit_test_rules()

        # Initialize LLM or adapter based on model provider
        if config.model_provider == "openai":
            # OpenAI models
            # Some models like o3-mini don't support temperature
            unsupported_temp_models = ["o3-mini", "tts-1", "tts-1-hd", "whisper-1"]
            if config.model_name in unsupported_temp_models:
                from pydantic import SecretStr

                self.llm = ChatOpenAI(
                    model=config.model_name,
                    api_key=SecretStr(config.api_key) if config.api_key else None,
                )
                logger.info(
                    f"Initialized OpenAI LLM with model: {config.model_name} (without temperature parameter)"
                )
            else:
                from pydantic import SecretStr

                self.llm = ChatOpenAI(
                    temperature=0,
                    model=config.model_name,
                    api_key=SecretStr(config.api_key) if config.api_key else None,
                )
                logger.info(
                    f"Initialized OpenAI LLM with model: {config.model_name} (temperature=0)"
                )
            self.copilot_adapter = None
        elif config.model_provider == "anthropic":
            # Anthropic models
            from pydantic import SecretStr

            if config.model_name == "claude-3-haiku":
                # Claude 3 Haiku might not support temperature
                self.llm = ChatAnthropic(
                    model_name=config.model_name,  # Changed model to model_name for Anthropic
                    api_key=SecretStr(config.api_key) if config.api_key else None,
                )
                logger.info(
                    f"Initialized Anthropic LLM with model: {config.model_name} (without temperature parameter)"
                )
            else:
                self.llm = ChatAnthropic(
                    temperature=0,
                    model_name=config.model_name,  # Changed model to model_name for Anthropic
                    api_key=SecretStr(config.api_key) if config.api_key else None,
                )
                logger.info(
                    f"Initialized Anthropic LLM with model: {config.model_name} (temperature=0)"
                )
            self.copilot_adapter = None
        elif config.model_provider == "github-copilot":
            self.llm = None
            self.copilot_adapter = CopilotAdapter(config)
        else:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

    def _load_unit_test_rules(self) -> str:
        """
        Load unit test rules from file.

        Returns:
            String containing the unit test rules
        """
        unit_test_rules = ""
        rules_path = os.path.join(os.path.dirname(__file__), "prompts", "unit_test_rules.md")
        try:
            with open(rules_path, "r") as f:
                unit_test_rules = f.read()
            logger.info(f"Successfully loaded unit test rules from {rules_path}")

            # Check if rules are too long and summarize them if needed
            max_rules_length = 2000  # Character limit for rules
            if len(unit_test_rules) > max_rules_length:
                # If rules are too long, use a condensed version
                condensed_rules = """# Unit Test Rules (Condensed)
- Write independent, isolated, and repeatable tests
- Follow Arrange-Act-Assert pattern
- Test edge cases and error conditions
- Use descriptive names and clear assertions
- Mock external dependencies
- Test one functionality per test
- Ensure tests are deterministic
- Handle exceptions appropriately
- Keep tests simple and readable"""
                logger.warning(
                    f"Unit test rules were condensed from {len(unit_test_rules)} to {len(condensed_rules)} characters"
                )
                unit_test_rules = condensed_rules

        except Exception as e:
            logger.error(f"Error loading unit test rules: {e}")
            # Provide fallback rules
            unit_test_rules = """# Unit Test Rules
- Write independent, isolated, and repeatable tests
- Follow Arrange-Act-Assert pattern
- Test edge cases and error conditions
- Use descriptive names and clear assertions"""

        return unit_test_rules

    def generate_test(
        self,
        function: Function,
        context_files: Optional[List[CodeFile]] = None,
        feedback: Optional[str] = None,
    ) -> GeneratedTest:
        """
        Generate a unit test for a function.

        Args:
            function: The function to generate a test for
            context_files: Additional files for context
            feedback: Optional feedback for collaborative test generation

        Returns:
            GeneratedTest object
        """
        logger.info(f"Generating test for function: {function.name} in {function.file_path}")

        # Apply IP protection to function and context
        redacted_function = self.ip_protector.redact_function(function)
        sanitized_context_files = None
        if context_files:
            sanitized_context_files = self.ip_protector.sanitize_context_files(context_files)

        # Prepare context from additional files if provided
        context = ""
        max_context_length = 4000  # Character limit for context to prevent token overflow
        total_context_length = 0

        if sanitized_context_files:
            for file in sanitized_context_files:
                file_content_chunk = f"\n# File: {file.path}\n{file.content}\n"
                # Check if adding this file would exceed our max context length
                if total_context_length + len(file_content_chunk) > max_context_length:
                    # Add a truncation notice
                    context += "\n# Additional files were truncated to prevent context overflow\n"
                    logger.warning(f"Context for {function.name} was truncated due to length")
                    break

                context += file_content_chunk
                total_context_length += len(file_content_chunk)

        # Use protected function code - make sure it's just the code without extra formatting
        # This ensures consistency for tests comparing exact string values
        protected_function_code = redacted_function.code.strip()

        # Limit function code size if it's very large
        max_function_length = 8000  # Character limit for function code
        if len(protected_function_code) > max_function_length:
            # Truncate the function code while preserving important parts
            header_size = int(max_function_length * 0.3)  # Keep 30% of the beginning
            footer_size = int(max_function_length * 0.3)  # Keep 30% of the end
            middle_message = "\n# ... [TRUNCATED - Function too large] ...\n"

            protected_function_code = (
                protected_function_code[:header_size]
                + middle_message
                + protected_function_code[-footer_size:]
            )
            logger.warning(f"Function code for {function.name} was truncated due to length")

        # Generate test based on the model provider
        if self.config.model_provider == "github-copilot":
            # Use Copilot adapter for test generation
            if (
                feedback
                and self.copilot_adapter is not None
                and getattr(self.config, "copilot_settings", {}).get("collaborative_mode", True)
            ):
                # Collaborative mode with feedback
                test_code, imports, mocks, fixtures = (
                    self.copilot_adapter.collaborative_test_generation(
                        function, context_files, feedback
                    )
                )
            elif self.copilot_adapter is not None:
                # Standard Copilot test generation
                test_code, imports, mocks, fixtures = self.copilot_adapter.generate_test(
                    function, context_files
                )
            else:
                raise ValueError(
                    "Copilot adapter not initialized but model_provider is 'github-copilot'"
                )
        else:
            # Use LangChain LLM for test generation
            # Prepare test generation prompt
            test_prompt = self._create_test_generation_prompt(function, context)

            # Generate test using LLM with special handling for models that don't support temperature
            unsupported_temp_models = ["o3-mini", "tts-1", "tts-1-hd", "whisper-1"]
            if (
                self.config.model_provider == "openai"
                and self.config.model_name in unsupported_temp_models
            ):
                # Direct call to ChatOpenAI for models without temperature support
                logger.info(
                    f"Using direct call to ChatOpenAI for {self.config.model_name} model (without temperature)"
                )

                # Format the prompt manually instead of using LLMChain
                formatted_prompt = test_prompt.format(
                    function_name=function.name,
                    function_code=protected_function_code,
                    file_path=function.file_path,
                    context=context or "",
                    test_framework=self.config.test_framework,
                    unit_test_rules=self.unit_test_rules,
                )

                # Apply IP protection to the formatted prompt
                protected_prompt = self.ip_protector.protect(formatted_prompt)

                # Call the LLM directly
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a professional software quality assurance engineer specializing in automated testing."
                    ),
                    HumanMessage(content=protected_prompt),
                ]

                # Get response without using temperature parameter
                llm_response = self.llm.invoke(messages)
                response = llm_response.content
            else:
                # For other models, use the RunnableSequence approach
                # Create parameters dict and protect each value
                params = {
                    "function_name": function.name,
                    "function_code": protected_function_code,
                    "file_path": function.file_path,
                    "context": self.ip_protector.protect(context) if context else "",
                    "test_framework": self.config.test_framework,
                    "unit_test_rules": self.unit_test_rules,
                }

                # Use the prompt to format the input, then directly call the LLM
                formatted_prompt = test_prompt.format(**params)

                # Call the LLM directly
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a professional software quality assurance engineer specializing in automated testing."
                    ),
                    HumanMessage(content=formatted_prompt),
                ]

                # Get response from the LLM
                llm_response = self.llm.invoke(messages)
                response = llm_response.content

            # Parse the response to extract the test code
            test_code, imports, mocks, fixtures = self._parse_test_response(response)

        # Create test file path
        test_file_path = self._get_test_file_path(function)

        # Extract test functions and test classes
        import re

        # Find test functions (starting with 'test_' or ending with '_test')
        test_functions = []
        test_func_pattern = re.compile(r"^def\s+(test_\w+|\w+_test)\s*\(", re.MULTILINE)
        test_functions = [match.group(1) for match in test_func_pattern.finditer(test_code)]

        # Find test classes (usually classes that inherit from TestCase or start with 'Test')
        test_classes = []
        test_class_pattern = re.compile(r"^class\s+(Test\w+|\w+Test)\s*\(", re.MULTILINE)
        test_classes = [match.group(1) for match in test_class_pattern.finditer(test_code)]

        return GeneratedTest(
            function=function,
            test_code=test_code,
            test_file_path=test_file_path,
            imports=imports,
            test_functions=test_functions,
            test_classes=test_classes,
            mocks=mocks,
            fixtures=fixtures,
        )

    def _create_test_generation_prompt(self, function: Function, context: str) -> PromptTemplate:
        """
        Create a prompt for test generation.

        Args:
            function: The function to generate a test for
            context: Additional context

        Returns:
            PromptTemplate object
        """
        # Use the pre-loaded unit test rules

        template = """
You are a professional software quality assurance engineer specializing in automated testing.
Your task is to generate a comprehensive unit test for the following function using {test_framework}.

# Function Information
- Name: {function_name}
- File: {file_path}

# Function Code:
```python
{function_code}
```

# Additional Context (if any):
{context}

# Unit Test Rules:
{unit_test_rules}

# Requirements:
1. Use {test_framework} for testing
2. Include comprehensive assertions to verify the function's behavior
3. Add appropriate mocks or fixtures if necessary
4. Include test cases for edge cases and error handling
5. Provide clear comments explaining your test strategy

# Response Format:
```python
# Test code here
```
"""
        return PromptTemplate(
            template=template,
            input_variables=[
                "function_name",
                "file_path",
                "function_code",
                "context",
                "test_framework",
                "unit_test_rules",
            ],
        )

    def _generate_test_with_llm(
        self,
        function: Function,
        context_files: Optional[List[CodeFile]] = None,
        feedback: Optional[str] = None,
    ) -> tuple:
        """
        Internal method to generate test code using an LLM.

        Args:
            function: The function to generate a test for
            context_files: Additional files for context
            feedback: Optional feedback for collaborative test generation

        Returns:
            tuple: (test_code, imports, mocks, fixtures)
        """
        # Prepare context from additional files if provided
        context = ""
        max_context_length = 4000  # Character limit for context to prevent token overflow
        total_context_length = 0

        if context_files:
            for file in context_files:
                file_content_chunk = f"\n# File: {file.path}\n{file.content}\n"
                # Check if adding this file would exceed our max context length
                if total_context_length + len(file_content_chunk) > max_context_length:
                    # Add a truncation notice
                    context += "\n# Additional files were truncated to prevent context overflow\n"
                    logger.warning(f"Context for {function.name} was truncated due to length")
                    break

                context += file_content_chunk
                total_context_length += len(file_content_chunk)

        # Use function code
        function_code = function.code

        # Limit function code size if it's very large
        max_function_length = 8000  # Character limit for function code
        if len(function_code) > max_function_length:
            # Truncate the function code while preserving important parts
            header_size = int(max_function_length * 0.3)  # Keep 30% of the beginning
            footer_size = int(max_function_length * 0.3)  # Keep 30% of the end
            middle_message = "\n# ... [TRUNCATED - Function too large] ...\n"

            function_code = (
                function_code[:header_size] + middle_message + function_code[-footer_size:]
            )
            logger.warning(f"Function code for {function.name} was truncated due to length")

        # Create test generation prompt
        test_prompt = self._create_test_generation_prompt(function, context)

        # Handle feedback if provided
        if feedback:
            # Update the template to include feedback
            template = test_prompt.template + "\n# Previous Feedback:\n{feedback}\n"
            test_prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "function_name",
                    "file_path",
                    "function_code",
                    "context",
                    "test_framework",
                    "unit_test_rules",
                    "feedback",
                ],
            )

        try:
            # Generate test using LLM with special handling for models that don't support temperature
            unsupported_temp_models = ["o3-mini", "tts-1", "tts-1-hd", "whisper-1"]
            if (
                self.config.model_provider == "openai"
                and self.config.model_name in unsupported_temp_models
            ):
                # Direct call to ChatOpenAI for models without temperature support
                logger.info(
                    f"Using direct call to ChatOpenAI for {self.config.model_name} model (without temperature)"
                )

                # Format the prompt manually instead of using LLMChain
                params = {
                    "function_name": function.name,
                    "function_code": function_code,
                    "file_path": function.file_path,
                    "context": context or "",
                    "test_framework": self.config.test_framework,
                    "unit_test_rules": self.unit_test_rules,
                }

                if feedback:
                    params["feedback"] = feedback

                formatted_prompt = test_prompt.format(**params)

                # Call the LLM directly
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a professional software quality assurance engineer specializing in automated testing."
                    ),
                    HumanMessage(content=formatted_prompt),
                ]

                # Get response without using temperature parameter
                llm_response = self.llm.invoke(messages)
                response = llm_response.content
            else:
                # For other models, use the RunnableSequence approach
                # Create parameters dict
                params = {
                    "function_name": function.name,
                    "function_code": function_code,
                    "file_path": function.file_path,
                    "context": context or "",
                    "test_framework": self.config.test_framework,
                    "unit_test_rules": self.unit_test_rules,
                }

                if feedback:
                    params["feedback"] = feedback

                # Use the prompt to format the input, then directly call the LLM
                formatted_prompt = test_prompt.format(**params)

                # Call the LLM directly
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a professional software quality assurance engineer specializing in automated testing."
                    ),
                    HumanMessage(content=formatted_prompt),
                ]

                # Get response from the LLM
                llm_response = self.llm.invoke(messages)
                response = llm_response.content

            # Parse the response to extract the test code
            return self._parse_test_response(response)

        except Exception as e:
            logger.error(f"Error generating test with LLM for {function.name}: {str(e)}")
            raise

    def generate_test_collaboratively(
        self,
        function: Function,
        context_files: Optional[List[CodeFile]] = None,
        feedback: Optional[str] = None,
    ) -> GeneratedTest:
        """
        Generate a unit test collaboratively, using feedback from users or QA Agent.

        This is particularly useful when tests need to be refined or when working with
        GitHub Copilot integration.

        Args:
            function: The function to generate a test for
            context_files: Additional files for context
            feedback: Feedback to incorporate into the test generation

        Returns:
            GeneratedTest object
        """
        logger.info(
            f"Generating test collaboratively for function: {function.name} in {function.file_path}"
        )

        # Apply IP protection to function and context
        redacted_function = self.ip_protector.redact_function(function)
        sanitized_context_files = None
        if context_files:
            sanitized_context_files = self.ip_protector.sanitize_context_files(context_files)

        # Generate the test using the collaborative method
        test_code, imports, test_functions, test_classes = self._generate_test_collaboratively(
            function, sanitized_context_files, feedback
        )

        # Construct the test file path
        test_file_path = self._get_test_file_path(function)

        # Create and return the GeneratedTest object
        return GeneratedTest(
            function=function,
            test_code=test_code,
            imports=imports,
            test_functions=test_functions,
            test_classes=test_classes,
            test_file_path=test_file_path,
            mocks=[],  # Add empty mocks list
            fixtures=[],  # Add empty fixtures list
        )

    def _generate_test_collaboratively(
        self,
        function: Function,
        context_files: Optional[List[CodeFile]] = None,
        feedback: Optional[str] = None,
    ) -> tuple:
        """
        Internal method to generate test code collaboratively.

        Args:
            function: The function to generate a test for
            context_files: Additional files for context
            feedback: Feedback from previous test generation or validation

        Returns:
            tuple: (test_code, imports, test_functions, test_classes)
        """
        if self.config.model_provider == "github-copilot" and self.copilot_adapter is not None:
            return self.copilot_adapter.collaborative_test_generation(
                function, context_files, feedback
            )
        else:
            # For non-Copilot providers, we'll use the LLM with feedback
            test_code, imports, mocks, fixtures = self._generate_test_with_llm(
                function, context_files, feedback
            )

            # Extract test functions and test classes
            test_functions = []
            test_classes = []

            # Simple regex to extract test functions and classes
            import re

            # Find test functions (starting with 'test_' or ending with '_test')
            test_func_pattern = re.compile(r"^def\s+(test_\w+|\w+_test)\s*\(", re.MULTILINE)
            test_functions = [match.group(1) for match in test_func_pattern.finditer(test_code)]

            # Find test classes (usually classes that inherit from TestCase or start with 'Test')
            test_class_pattern = re.compile(r"^class\s+(Test\w+|\w+Test)\s*\(", re.MULTILINE)
            test_classes = [match.group(1) for match in test_class_pattern.finditer(test_code)]

            return test_code, imports, test_functions, test_classes

    def _parse_test_response(
        self, response: Union[str, Any]
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Parse the LLM response to extract test code, imports, mocks, fixtures, test functions and test classes.

        Args:
            response: LLM response (string or object with content attribute)

        Returns:
            Tuple of (test_code, imports, mocks, fixtures)
        """
        import re

        # Handle case where response might be an object with content attribute (LangChain 0.1.3 models)
        if not isinstance(response, str) and hasattr(response, "content"):
            response = response.content

        # Ensure we're working with a string
        if not isinstance(response, str):
            response = str(response)

        # Extract code from the response
        code_match = re.search(r"```python\s+(.*?)\s+```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response.strip()

        # Extract imports from code and get module names
        imports = []
        # Match "import module" or "import module as alias"
        import_pattern = re.compile(r"^import\s+([^\s,]+)(?:\s+as\s+\w+)?", re.MULTILINE)
        for match in import_pattern.finditer(code):
            module_name = match.group(1)  # Keep the full module path
            imports.append(module_name)

        # Match "from module import name" style imports
        from_import_pattern = re.compile(r"^from\s+([^\s,]+)\s+import", re.MULTILINE)
        for match in from_import_pattern.finditer(code):
            module_name = match.group(1)  # Keep the full module path
            imports.append(module_name)

        # Also check for structured IMPORTS section
        imports_section_pattern = re.compile(r"IMPORTS:\s*\n((?:[-*]\s*.*\s*\n)+)", re.MULTILINE)
        imports_match = imports_section_pattern.search(response)
        if imports_match:
            imports_section = imports_match.group(1)
            import_list_pattern = re.compile(r"[-*]\s*([\w\.]+)", re.MULTILINE)
            for match in import_list_pattern.finditer(imports_section):
                module_name = match.group(1)  # Keep the full module path
                if module_name and module_name not in imports:
                    imports.append(module_name)

        # Extract mocks (lines containing mock or patch)
        mocks = []
        mock_pattern = re.compile(r"^.*(?:mock|patch).*$", re.MULTILINE | re.IGNORECASE)
        for match in mock_pattern.finditer(code):
            line = match.group(0)
            if not line.strip().startswith("#"):  # Ignore comments
                mocks.append(line)

        # Also check for structured MOCKS section
        mocks_section_pattern = re.compile(r"MOCKS:\s*\n((?:[-*]\s*.*\s*\n)+)", re.MULTILINE)
        mocks_match = mocks_section_pattern.search(response)
        if mocks_match:
            mocks_section = mocks_match.group(1)
            mock_list_pattern = re.compile(r"[-*]\s*([\w\.]+)", re.MULTILINE)
            for match in mock_list_pattern.finditer(mocks_section):
                mock_name = match.group(1)
                if mock_name and mock_name not in mocks:
                    mocks.append(mock_name)

        # Extract fixtures (pytest fixtures)
        fixtures = []
        fixture_pattern = re.compile(
            r"^@pytest\.fixture.*?def\s+(\w+).*?:", re.MULTILINE | re.DOTALL
        )
        for match in fixture_pattern.finditer(code):
            fixtures.append(match.group(0))

        # Also check for structured FIXTURES section
        fixtures_section_pattern = re.compile(r"FIXTURES:\s*\n((?:[-*]\s*.*\s*\n)+)", re.MULTILINE)
        fixtures_match = fixtures_section_pattern.search(response)
        if fixtures_match:
            fixtures_section = fixtures_match.group(1)
            fixture_list_pattern = re.compile(r"[-*]\s*([\w\.]+)", re.MULTILINE)
            for match in fixture_list_pattern.finditer(fixtures_section):
                fixture_name = match.group(1)
                if fixture_name and fixture_name not in fixtures:
                    fixtures.append(fixture_name)

        # For backwards compatibility, only return the original tuple
        # The _generate_test_collaboratively method will extract the test functions and classes
        return code, imports, mocks, fixtures

    def _get_test_file_path(self, function: Function) -> str:
        """
        Generate a path for the test file following language-specific conventions.

        Args:
            function: The function to generate a test for

        Returns:
            Path to the test file
        """
        import logging
        import os

        logger = logging.getLogger(__name__)

        try:
            # Extract module name and file extension from file path
            file_basename = os.path.basename(function.file_path)
            module_name, file_extension = os.path.splitext(file_basename)

            # Mapping of file extensions to language-specific naming patterns
            naming_conventions = {
                # Python: test_*.py
                ".py": lambda name: f"test_{name}.py",
                # JavaScript/TypeScript: *.test.js
                ".js": lambda name: f"{name}.test.js",
                ".jsx": lambda name: f"{name}.test.jsx",
                ".ts": lambda name: f"{name}.test.ts",
                ".tsx": lambda name: f"{name}.test.tsx",
                # Go: *_test.go
                ".go": lambda name: f"{name}_test.go",
                # PHP: *Test.php (PHPUnit convention)
                ".php": lambda name: f"{name[0].upper() + name[1:]}Test.php",
                # SQL: *.test.sql
                ".sql": lambda name: f"{name}.test.sql",
            }

            # Determine language from file extension and generate test file name
            if file_extension.lower() in naming_conventions:
                test_file_name = naming_conventions[file_extension.lower()](module_name)
                logger.info(f"Using {file_extension} naming convention: {test_file_name}")
            else:
                # Default to language-agnostic naming for unsupported extensions
                test_file_name = f"test_{module_name}{file_extension}"
                logger.warning(
                    f"No specific naming convention for {file_extension} files, "
                    f"using default: {test_file_name}"
                )

            # Create test directory if it doesn't exist
            os.makedirs(self.config.output_directory, exist_ok=True)

            return os.path.join(self.config.output_directory, test_file_name)

        except Exception as e:
            # Fallback to a safe default in case of errors
            logger.error(f"Error determining test file name: {str(e)}")
            safe_name = "test_generated.py"
            logger.info(f"Using fallback test file name: {safe_name}")

            # Ensure output directory exists
            os.makedirs(self.config.output_directory, exist_ok=True)

            return os.path.join(self.config.output_directory, safe_name)

    def save_test_to_file(self, test: GeneratedTest) -> None:
        """
        Save a generated test to a file.

        Args:
            test: The generated test to save
        """
        logger.info(f"Saving test to file: {test.test_file_path}")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(test.test_file_path), exist_ok=True)

        # Generate imports section
        imports_code = ""
        if test.imports:
            for imp in test.imports:
                imports_code += f"import {imp}\n"

        # Write the test to file with imports
        with open(test.test_file_path, "w") as f:
            # First write imports
            if imports_code:
                f.write(imports_code + "\n")
            # Then write the test code
            f.write(test.test_code)
