#!/usr/bin/env python3
"""
Script to test the QA Agent package installation in an isolated environment.
This is a platform-independent alternative to the shell script.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run_command(cmd, cwd=None, env=None):
    """Run a command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        sys.exit(1)
    return result.stdout


def main():
    """Main function to test package installation."""
    print("=== Testing QA Agent package installation ===")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")

        # Build the package
        print("Building package...")
        run_command([sys.executable, "-m", "build"])

        # Find the built wheel file
        dist_dir = Path("dist")
        wheel_files = list(dist_dir.glob("*.whl"))
        wheel_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not wheel_files:
            print("No wheel files found. Build may have failed.")
            sys.exit(1)

        wheel_file = wheel_files[0]
        print(f"Using wheel file: {wheel_file}")

        # Create a virtual environment
        venv_dir = Path(temp_dir) / "venv"
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip=True)

        # Get path to python in the virtual environment
        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"

        # Install the wheel
        print("Installing wheel...")
        run_command([str(venv_python), "-m", "pip", "install", str(wheel_file)])

        # Verify import works
        print("Verifying import...")
        import_check = run_command(
            [str(venv_python), "-c", "from qa_agent import __init__; print('Import successful')"]
        )
        print(import_check)

        # Verify CLI entry point works
        print("Verifying CLI entry point...")
        if sys.platform == "win32":
            cli_path = venv_dir / "Scripts" / "qa-agent.exe"
        else:
            cli_path = venv_dir / "bin" / "qa-agent"

        if not cli_path.exists():
            print(f"CLI entry point not found at {cli_path}")
            sys.exit(1)

        help_output = run_command([str(cli_path), "--help"])
        print("CLI entry point help output:")
        print(help_output)

    print("=== Package installation test completed successfully ===")


if __name__ == "__main__":
    main()
