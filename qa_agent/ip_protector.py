"""
IP Protection module.

This module provides functionality to protect intellectual property by redacting
sensitive code and data before sending it to external services like LLMs.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Union

from qa_agent.models import CodeFile, Function
from qa_agent.utils.logging import get_logger, log_function_call, log_redacted

# Initialize logger for this module
logger = get_logger(__name__)


class IPProtector:
    """
    Protects intellectual property by redacting sensitive code and data.
    """

    def __init__(
        self,
        config_or_patterns=None,
        protected_functions: Optional[List[str]] = None,
        protected_files: Optional[List[str]] = None,
    ):
        """
        Initialize the IP protector.

        Args:
            config_or_patterns: Either a QAAgentConfig object or a list of regex patterns to redact
            protected_functions: List of function names that should be protected
            protected_files: List of file paths that should be fully protected
        """
        # Check if first argument is a QAAgentConfig
        from qa_agent.config import QAAgentConfig
        if isinstance(config_or_patterns, QAAgentConfig):
            config = config_or_patterns
            self.protected_patterns = config.protected_patterns or []
            self.protected_functions = config.protected_functions or []
            self.protected_files = config.protected_files or []
        else:
            self.protected_patterns = config_or_patterns or []
            self.protected_functions = protected_functions or []
            self.protected_files = protected_files or []

        log_function_call(
            logger,
            "__init__",
            ("IPProtector",),
            {
                "patterns_count": len(self.protected_patterns),
                "functions_count": len(self.protected_functions),
                "files_count": len(self.protected_files),
            },
        )

    def protect(self, content: str) -> str:
        """
        Redact sensitive content.

        Args:
            content: The content to redact

        Returns:
            Redacted content
        """
        return self._redact_content(content)

    def _redact_content(self, content: Optional[str]) -> str:
        """
        Apply redaction patterns to content.

        Args:
            content: The content to redact

        Returns:
            Redacted content
        """
        if content is None:
            return ""

        redacted = content

        # Apply each pattern for redaction
        for pattern in self.protected_patterns:
            try:
                # Handle specific patterns for API keys and passwords
                if pattern == "api_key":
                    # Handle assignment pattern (api_key = "value")
                    regex = r'(api_key\s*=\s*)[\'"][^\'"]*[\'"]'
                    redacted = re.sub(regex, r'\1"[REDACTED]"', redacted)

                    # Handle api_key as parameter (function(api_key))
                    redacted = re.sub(r"([,(]\s*)api_key([,)])", r"\1[REDACTED]\2", redacted)

                    # Handle "API key" in natural language text (case-insensitive)
                    redacted = re.sub(r"\bAPI\s+key\b", "[REDACTED]", redacted, flags=re.IGNORECASE)

                    # Handle remaining instances as bare word only if not part of another word
                    if "api_key" in redacted:
                        # This regex ensures we only match 'api_key' as a whole word
                        redacted = re.sub(r"\bapi_key\b(?!\s*=)", "[REDACTED]", redacted)
                elif pattern == "password":
                    # Keep password variable names but redact the values
                    redacted = re.sub(
                        r'password\s*=\s*[\'"][^\'"]*[\'"]', r'password = "[REDACTED]"', redacted
                    )

                    # Also redact password parameters in function signatures
                    redacted = re.sub(r"([,(]\s*)password([,)])", r"\1[REDACTED]\2", redacted)

                    # Redact "password" in natural language (case-insensitive)
                    redacted = re.sub(r"\bpassword\b", "[REDACTED]", redacted, flags=re.IGNORECASE)
                # Handle API key regex patterns
                elif "api_key" in pattern and any(c in pattern for c in ".*?+()[]{}^$\\|"):
                    redacted = re.sub(
                        r'api_key\s*=\s*[\'"][^\'"]*[\'"]', r'api_key = "[REDACTED]"', redacted
                    )
                # Handle secret regex patterns
                elif "secret" in pattern and any(c in pattern for c in ".*?+()[]{}^$\\|"):
                    redacted = re.sub(
                        r'secret\s*=\s*[\'"][^\'"]*[\'"]', r'secret = "[REDACTED]"', redacted
                    )
                # Handle token regex patterns
                elif "token" in pattern and any(c in pattern for c in ".*?+()[]{}^$\\|"):
                    redacted = re.sub(
                        r'token\s*=\s*[\'"][^\'"]*[\'"]', r'token = "[REDACTED]"', redacted
                    )
                # Handle password regex patterns
                elif "password" in pattern and any(c in pattern for c in ".*?+()[]{}^$\\|"):
                    redacted = re.sub(
                        r'password\s*=\s*[\'"][^\'"]*[\'"]', r'password = "[REDACTED]"', redacted
                    )
                # If it's a simple string pattern, replace it carefully
                elif all(c not in pattern for c in ".*?+()[]{}^$\\|"):
                    # Make sure we only replace whole words, case-insensitive
                    redacted = re.sub(
                        r"\b" + re.escape(pattern) + r"\b",
                        "[REDACTED]",
                        redacted,
                        flags=re.IGNORECASE,
                    )
                else:
                    # For regex patterns, use regex substitution directly
                    try:
                        # Try to use the pattern as-is since it's already a regex pattern
                        redacted = re.sub(pattern, r'"[REDACTED]"', redacted)
                    except re.error:
                        # For variable assignments, try to maintain the structure
                        try:
                            if "=" in pattern:
                                var_pattern = pattern.split("=")[0].strip() + r"\s*=\s*"
                                redacted = re.sub(
                                    var_pattern + r'[\'"][^\'"]*[\'"]',
                                    var_pattern + r'"[REDACTED]"',
                                    redacted,
                                )
                            else:
                                # Fallback to simple string replacement for invalid regex
                                if pattern in redacted:
                                    redacted = redacted.replace(pattern, "[REDACTED]")
                        except re.error:
                            # Final fallback to simple string replacement
                            if pattern in redacted:
                                redacted = redacted.replace(pattern, "[REDACTED]")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
                # Even with invalid regex, try simple string replacement if applicable
                if pattern in redacted:
                    redacted = redacted.replace(pattern, "[REDACTED]")

        return redacted

    def redact_code_file(self, code_file: CodeFile) -> CodeFile:
        """
        Redact sensitive data from a CodeFile.

        Args:
            code_file: The code file to redact

        Returns:
            Redacted CodeFile
        """
        # If the file is in the protected files list, fully redact it
        file_should_be_redacted = False
        
        for protected_pattern in self.protected_files:
            # Handle glob patterns like **/secrets.py
            if protected_pattern.startswith("**/"):
                file_basename = os.path.basename(code_file.path)
                pattern_basename = protected_pattern.split("**/")[1]
                if file_basename == pattern_basename:
                    file_should_be_redacted = True
                    break
            # Handle exact file path matches
            elif code_file.path.endswith(protected_pattern):
                file_should_be_redacted = True
                break
                
        if file_should_be_redacted:
            log_redacted(logger, code_file.path)
            return CodeFile(
                path=code_file.path,
                content="[REDACTED PROTECTED FILE]",
                type=code_file.type,
            )

        # Otherwise apply pattern redaction
        redacted_content = self._redact_content(code_file.content)

        if redacted_content != code_file.content:
            log_redacted(logger, code_file.path)

        return CodeFile(path=code_file.path, content=redacted_content, type=code_file.type)

    def redact_function(self, function: Function) -> Function:
        """
        Redact sensitive data from a Function.

        Args:
            function: The function to redact

        Returns:
            Redacted Function
        """
        # If the function name matches a protected pattern, fully redact it
        function_should_be_redacted = False
        
        for pattern in self.protected_functions:
            # Handle wildcard patterns like auth_* or *_credentials
            if pattern.endswith('*') and function.name.startswith(pattern[:-1]):
                function_should_be_redacted = True
                break
            elif pattern.startswith('*') and function.name.endswith(pattern[1:]):
                function_should_be_redacted = True
                break
            # Handle exact matches
            elif function.name == pattern:
                function_should_be_redacted = True
                break
                
        if function_should_be_redacted:
            log_redacted(logger, function.file_path, function.name)
            return Function(
                name=function.name,
                code="[REDACTED PROTECTED FUNCTION]",
                file_path=function.file_path,
                start_line=function.start_line,
                end_line=function.end_line,
                complexity=function.complexity,
                parameters=function.parameters,
                return_type=function.return_type,
                docstring="[REDACTED]",
                dependencies=function.dependencies,
            )

        # Otherwise apply pattern redaction
        redacted_code = self._redact_content(function.code)
        redacted_docstring = self._redact_content(function.docstring or "")

        if redacted_code != function.code:
            log_redacted(logger, function.file_path, function.name)

        return Function(
            name=function.name,
            code=redacted_code,
            file_path=function.file_path,
            start_line=function.start_line,
            end_line=function.end_line,
            complexity=function.complexity,
            parameters=function.parameters,
            return_type=function.return_type,
            docstring=redacted_docstring if function.docstring else "",
            dependencies=function.dependencies,
        )

    def redact_prompt(self, prompt: str) -> str:
        """
        Redact sensitive data from a prompt.

        Args:
            prompt: The prompt to redact

        Returns:
            Redacted prompt
        """
        return self._redact_content(prompt)

    def sanitize_context_files(self, context_files: List[CodeFile]) -> List[CodeFile]:
        """
        Sanitize a list of context files.

        Args:
            context_files: List of code files to sanitize

        Returns:
            List of sanitized code files
        """
        return [self.redact_code_file(file) for file in context_files]

    def load_protection_rules(self, rules_path: str) -> None:
        """
        Load protection rules from a JSON file.

        Args:
            rules_path: Path to the JSON rules file
        """
        if not os.path.exists(rules_path):
            logger.warning(f"IP protection rules file not found at {rules_path}")
            return

        try:
            with open(rules_path, "r") as f:
                rules = json.load(f)

            # Update protected lists from rules
            self.protected_patterns = rules.get("protected_patterns", self.protected_patterns)
            self.protected_functions = rules.get("protected_functions", self.protected_functions)
            self.protected_files = rules.get("protected_files", self.protected_files)

            logger.info(
                f"Loaded IP protection rules from {rules_path}: "
                f"{len(self.protected_patterns)} patterns, "
                f"{len(self.protected_functions)} functions, "
                f"{len(self.protected_files)} files"
            )

        except Exception as e:
            logger.error(f"Error loading IP protection rules: {str(e)}")
