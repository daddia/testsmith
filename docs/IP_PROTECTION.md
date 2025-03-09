# IP Protection Features

QA Agent includes robust IP (Intellectual Property) protection features to safeguard your sensitive code when using external AI providers like OpenAI or Anthropic.

## Overview

When generating tests, portions of your code are sent to AI providers for analysis. The IP protection system ensures sensitive or proprietary code is redacted before transmission, preventing exposure of trade secrets or confidential algorithms.

## How IP Protection Works

1. **Pattern Recognition**: Identifies sensitive code patterns, functions, or files
2. **Redaction**: Replaces sensitive content with placeholders
3. **Safe Transmission**: Sends only the redacted content to external services
4. **Context Preservation**: Maintains enough context for effective test generation

## Enabling IP Protection

### Command Line

```bash
# Basic IP protection
qa-agent --repo-path /path/to/your/project --ip-protection

# With custom rules file
qa-agent --repo-path /path/to/your/project --ip-protection --ip-rules-path ./ip_rules.json
```

### Configuration File

```yaml
# qa_config.yaml
repo_path: "/path/to/your/project"
ip_protection_enabled: true
ip_protection_rules_path: "./ip_rules.json"
```

## Creating a Rules File

The IP protection rules file is a JSON file that specifies what content should be protected:

```json
{
  "protected_patterns": [
    "SECRET_[A-Z0-9_]+",
    "API_KEY_[A-Z0-9]+",
    "encryption\\.(encrypt|decrypt)",
    "hash_password"
  ],
  "protected_functions": [
    "calculate_pricing_algorithm",
    "predict_user_behavior",
    "generate_security_token"
  ],
  "protected_files": [
    "billing/pricing_algorithm.py",
    "security/encryption.py",
    "ml/proprietary_model.py"
  ]
}
```

Save this file as `ip_protection_rules.json` in your project directory.

## Protection Categories

### Protected Patterns

Regular expressions that identify sensitive code patterns, such as:
- Security credentials
- Encryption methods
- Proprietary algorithms
- Trade secrets

Example:
```json
"protected_patterns": [
  "API_KEY_[A-Z0-9]+",
  "hash_password\\(.*\\)"
]
```

### Protected Functions

Specific function names that should be completely redacted:

Example:
```json
"protected_functions": [
  "calculate_pricing",
  "predict_user_behavior"
]
```

### Protected Files

Entire files that contain sensitive code and should be completely redacted:

Example:
```json
"protected_files": [
  "billing/pricing.py",
  "ml/proprietary_model.py"
]
```

## Redaction Example

### Original Code

```python
def calculate_price(base_price, user_tier):
    """Calculate final price based on user tier."""
    if user_tier == "premium":
        return base_price * 0.85  # 15% discount for premium users
    elif user_tier == "enterprise":
        # Apply our proprietary enterprise pricing algorithm
        return base_price * ENTERPRISE_FACTOR - calculate_enterprise_discount(base_price)
    else:
        return base_price
```

### Redacted Code (if "calculate_enterprise_discount" is protected)

```python
def calculate_price(base_price, user_tier):
    """Calculate final price based on user tier."""
    if user_tier == "premium":
        return base_price * 0.85  # 15% discount for premium users
    elif user_tier == "enterprise":
        # Apply our proprietary enterprise pricing algorithm
        return base_price * ENTERPRISE_FACTOR - [REDACTED FUNCTION CALL]
    else:
        return base_price
```

## Best Practices

1. **Start with broad protection** and refine based on results
2. **Regularly review redaction patterns** as your codebase evolves
3. **Balance protection with context** - over-redaction can lead to poor test quality
4. **Test your rules file** on a small subset of code first
5. **Monitor generated tests** for any leaked sensitive information

## How to Verify Protection

QA Agent logs all redactions when verbose mode is enabled:

```bash
qa-agent --repo-path /path/to/your/project --ip-protection --verbose
```

Look for lines like:
```
2025-03-07 01:25:32 [info] Redacted function calculate_enterprise_discount from billing/pricing.py
```

## Limitations

The IP protection system has some limitations:

1. It only protects code sent to external AI providers (not local operations)
2. It relies on pattern matching, which may miss novel patterns
3. Function and variable names remain visible unless specifically protected
4. Over-aggressive protection may reduce test quality

## Troubleshooting

### Poor Test Quality

If generated tests are of poor quality after enabling IP protection:

1. Check if too much context is being redacted
2. Try reducing the scope of protection
3. Add more docstrings to explain function behavior without revealing implementation

### Missing Protection

If sensitive content is still visible in AI requests:

1. Enable verbose logging to see what's being sent
2. Add more specific patterns or function names
3. Consider protecting entire files instead of individual patterns