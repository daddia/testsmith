"""
End-to-end tests for IP protection features.

These tests verify that the IP protection mechanisms correctly redact
sensitive code and information before it's sent to external services.
"""

import os
import json
import shutil
import tempfile

import pytest

from qa_agent.config import QAAgentConfig
from qa_agent.models import CodeFile, FileType, Function
from qa_agent.ip_protector import IPProtector
from qa_agent.agents import TestGenerationAgent


class TestIPProtectionE2E:
    """End-to-end tests for IP protection features."""

    @pytest.mark.e2e
    def test_ip_protection_basic_redaction(self, mocker, sample_repo_path, e2e_config, disable_api_calls):
        """Test basic redaction of sensitive code and data."""
        # Set up configuration with IP protection enabled
        e2e_config.repo_path = sample_repo_path
        e2e_config.ip_protection_enabled = True
        e2e_config.protected_patterns = [
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
        ]
        
        # Create an IPProtector
        ip_protector = IPProtector(e2e_config)
        
        # Create test code with sensitive information
        sensitive_code = """
def authenticate_user(username, password):
    \"\"\"Authenticate a user with the API.\"\"\"
    api_key = "sk_test_123456789abcdef"
    secret = "super_secret_value"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    
    # Make API request
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"username": username, "password": password}
    response = requests.post("https://api.example.com/auth", json=payload, headers=headers)
    
    return response.json()
"""
        
        # Test redaction
        redacted_code = ip_protector._redact_content(sensitive_code)
        
        # Verify redactions
        assert "api_key = \"sk_test_123456789abcdef\"" not in redacted_code
        assert "api_key = \"[REDACTED]\"" in redacted_code
        assert "secret = \"super_secret_value\"" not in redacted_code
        assert "secret = \"[REDACTED]\"" in redacted_code
        assert "token = \"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\"" not in redacted_code
        assert "token = \"[REDACTED]\"" in redacted_code
        
        # Original code remains unchanged
        assert "api_key = \"sk_test_123456789abcdef\"" in sensitive_code

    @pytest.mark.e2e
    def test_ip_protection_with_code_files(self, mocker, sample_repo_path, e2e_config, disable_api_calls):
        """Test IP protection with code files."""
        # Set up configuration with IP protection enabled
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_ip_protection_tests")
        os.makedirs(e2e_config.output_directory, exist_ok=True)
        e2e_config.ip_protection_enabled = True
        e2e_config.protected_files = ["**/secrets.py", "**/credentials.py"]
        
        # Create an IPProtector
        ip_protector = IPProtector(e2e_config)
        
        # Create test code files
        regular_file = CodeFile(
            path=os.path.join(sample_repo_path, "sample_module/utils.py"),
            content="def add_numbers(a, b):\n    return a + b",
            type=FileType.PYTHON,
        )
        
        sensitive_file = CodeFile(
            path=os.path.join(sample_repo_path, "sample_module/secrets.py"),
            content="API_KEY = \"sk_test_123456789abcdef\"\nPASSWORD = \"secure_password123\"",
            type=FileType.PYTHON,
        )
        
        # Test redaction of code files
        redacted_regular = ip_protector.redact_code_file(regular_file)
        redacted_sensitive = ip_protector.redact_code_file(sensitive_file)
        
        # Verify redactions
        assert redacted_regular.content == regular_file.content  # Regular file shouldn't be redacted
        assert redacted_sensitive.content != sensitive_file.content  # Sensitive file should be redacted
        assert "[REDACTED PROTECTED FILE]" in redacted_sensitive.content

    @pytest.mark.e2e
    def test_ip_protection_with_functions(self, mocker, sample_repo_path, e2e_config, disable_api_calls):
        """Test IP protection with functions."""
        # Set up configuration with IP protection enabled
        e2e_config.repo_path = sample_repo_path
        e2e_config.ip_protection_enabled = True
        e2e_config.protected_functions = ["auth_*", "*_with_credentials"]
        
        # Create an IPProtector
        ip_protector = IPProtector(e2e_config)
        
        # Create test functions
        regular_function = Function(
            name="calculate_total",
            code="def calculate_total(items):\n    return sum(item.price for item in items)",
            file_path=os.path.join(sample_repo_path, "sample_module/utils.py"),
            start_line=10,
            end_line=11,
            docstring=None,
            parameters=[{"name": "items", "type": "list"}],
            return_type="float",
            dependencies=[],
            complexity=1,
        )
        
        sensitive_function1 = Function(
            name="auth_user",
            code="def auth_user(username, password):\n    return api.authenticate(username, password, API_KEY)",
            file_path=os.path.join(sample_repo_path, "sample_module/auth.py"),
            start_line=15,
            end_line=16,
            docstring=None,
            parameters=[{"name": "username", "type": "str"}, {"name": "password", "type": "str"}],
            return_type="bool",
            dependencies=["api"],
            complexity=1,
        )
        
        sensitive_function2 = Function(
            name="request_with_credentials",
            code="def request_with_credentials(url):\n    return requests.get(url, headers={'Authorization': f'Bearer {TOKEN}'})",
            file_path=os.path.join(sample_repo_path, "sample_module/api.py"),
            start_line=20,
            end_line=21,
            docstring=None,
            parameters=[{"name": "url", "type": "str"}],
            return_type="Response",
            dependencies=["requests"],
            complexity=1,
        )
        
        # Test redaction of functions
        redacted_regular = ip_protector.redact_function(regular_function)
        redacted_sensitive1 = ip_protector.redact_function(sensitive_function1)
        redacted_sensitive2 = ip_protector.redact_function(sensitive_function2)
        
        # Verify redactions
        assert redacted_regular.code == regular_function.code  # Regular function shouldn't be redacted
        assert redacted_sensitive1.code != sensitive_function1.code  # Sensitive function should be redacted
        assert "[REDACTED PROTECTED FUNCTION]" in redacted_sensitive1.code
        assert redacted_sensitive2.code != sensitive_function2.code  # Sensitive function should be redacted
        assert "[REDACTED PROTECTED FUNCTION]" in redacted_sensitive2.code

    @pytest.mark.e2e
    def test_ip_protection_with_test_generation(self, mocker, sample_repo_path, e2e_config, disable_api_calls):
        """Test IP protection during test generation process."""
        # Set up configuration with IP protection enabled
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_ip_test_gen")
        os.makedirs(e2e_config.output_directory, exist_ok=True)
        e2e_config.ip_protection_enabled = True
        e2e_config.protected_patterns = [r"api_key\s*=\s*['\"][^'\"]+['\"]"]
        
        # Create a function with sensitive data
        function = Function(
            name="get_data_from_api",
            code='def get_data_from_api(endpoint):\n    api_key = "sk_test_123456789abcdef"\n    headers = {"Authorization": f"Bearer {api_key}"}\n    return requests.get(f"https://api.example.com/{endpoint}", headers=headers).json()',
            file_path=os.path.join(sample_repo_path, "sample_module/api_client.py"),
            start_line=10,
            end_line=13,
            docstring=None,
            parameters=[{"name": "endpoint", "type": "str"}],
            return_type="dict",
            dependencies=["requests"],
            complexity=1,
        )
        
        # Create context files
        context_file = CodeFile(
            path=os.path.join(sample_repo_path, "sample_module/api_client.py"),
            content='import requests\n\ndef get_data_from_api(endpoint):\n    api_key = "sk_test_123456789abcdef"\n    headers = {"Authorization": f"Bearer {api_key}"}\n    return requests.get(f"https://api.example.com/{endpoint}", headers=headers).json()',
            type=FileType.PYTHON,
        )
        
        # Create a spy on IPProtector redact_function
        real_ip_protector_redact_function = IPProtector.redact_function
        
        def spy_redact_function(self, function):
            # Call the real method
            result = real_ip_protector_redact_function(self, function)
            # Store the result for verification
            spy_redact_function.last_result = result
            return result
        
        # Set the spy attribute
        spy_redact_function.last_result = None
        
        # Replace the real function with our spy
        mocker.patch.object(IPProtector, "redact_function", spy_redact_function)
        
        # Initialize test generation agent
        test_generator = TestGenerationAgent(e2e_config)
        
        # Mock the test generator's generate_test method to avoid executing real code
        # and prevent test file writing operations that might cause issues
        mocker.patch.object(
            test_generator.test_generator,
            "generate_test", 
            return_value=mocker.MagicMock(test_file_path="/tmp/mocked_test.py")
        )
        
        # Mock the save_test_to_file method to prevent file system operations
        mocker.patch.object(
            test_generator.test_generator, 
            "save_test_to_file",
            return_value=None
        )
        
        # Generate test 
        try:
            test_generator.generate_test(function, [context_file])
        except Exception as e:
            # We don't care about possible exceptions after our checks
            # as we're only verifying the IP protection works
            pass
        
        # Verify that the redact_function method was called and stored a result
        assert spy_redact_function.last_result is not None
        
        # Get the result and check that the sensitive data was redacted
        redacted_function = spy_redact_function.last_result
        assert "api_key = \"sk_test_123456789abcdef\"" not in redacted_function.code
        assert "[REDACTED]" in redacted_function.code
        
        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)

    @pytest.mark.e2e
    def test_load_protection_rules_from_file(self, mocker, sample_repo_path, e2e_config, disable_api_calls):
        """Test loading IP protection rules from a file."""
        # Set up configuration with IP protection enabled
        e2e_config.repo_path = sample_repo_path
        e2e_config.output_directory = os.path.join(tempfile.gettempdir(), "qa_agent_ip_rules_tests")
        os.makedirs(e2e_config.output_directory, exist_ok=True)
        e2e_config.ip_protection_enabled = True
        
        # Create a rules file
        rules_file = os.path.join(e2e_config.output_directory, "ip_protection_rules.json")
        rules = {
            "protected_patterns": [
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"password\s*=\s*['\"][^'\"]+['\"]",
            ],
            "protected_functions": ["auth_*", "*_with_secret"],
            "protected_files": ["**/secrets.py", "**/credentials.py"],
        }
        
        with open(rules_file, "w") as f:
            json.dump(rules, f)
        
        # Set the rules file path in config
        e2e_config.ip_protection_rules_path = rules_file
        
        # Create an IPProtector
        ip_protector = IPProtector(e2e_config)
        
        # Since we're using a real file, we don't need to mock os.path.exists
        # Explicitly call load_protection_rules
        ip_protector.load_protection_rules(rules_file)
        
        # Verify rules were loaded
        assert len(ip_protector.protected_patterns) == 2
        assert len(ip_protector.protected_functions) == 2
        assert len(ip_protector.protected_files) == 2
        
        # Test redaction with loaded rules
        sensitive_code = 'api_key = "sk_test_123456789abcdef"\npassword = "secure_password123"'
        redacted_code = ip_protector._redact_content(sensitive_code)
        
        # Verify redactions
        assert "api_key = \"sk_test_123456789abcdef\"" not in redacted_code
        assert "api_key = \"[REDACTED]\"" in redacted_code
        assert "password = \"secure_password123\"" not in redacted_code
        assert "password = \"[REDACTED]\"" in redacted_code
        
        # Clean up
        if os.path.exists(e2e_config.output_directory):
            shutil.rmtree(e2e_config.output_directory)


if __name__ == "__main__":
    pytest.main(["-v", "test_e2e_ip_protection.py"])