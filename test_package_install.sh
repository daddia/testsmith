#!/usr/bin/env bash
# Script to test package installation in an isolated environment

set -e  # Exit on error

echo "=== Testing QA Agent package installation ==="

# Create a temporary directory for testing
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Clean up on exit
cleanup() {
    echo "Cleaning up..."
    rm -rf "$TEMP_DIR"
    echo "Done."
}
trap cleanup EXIT

# Build the package
echo "Building package..."
python -m build

# Find the built wheel file
WHEEL_FILE=$(ls -t dist/*.whl | head -1)
echo "Using wheel file: $WHEEL_FILE"

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv "$TEMP_DIR/venv"
source "$TEMP_DIR/venv/bin/activate"

# Install the wheel
echo "Installing wheel..."
pip install "$WHEEL_FILE"

# Verify installation
echo "Verifying installation..."
# Test importing the package
python -c "from qa_agent import __init__; print('Import successful')"

# Test the CLI entry point
which qa-agent
qa-agent --help

echo "=== Package installation test completed successfully ==="