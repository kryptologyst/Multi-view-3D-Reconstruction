#!/usr/bin/env python3
"""Setup script for multi-view 3D reconstruction project."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Multi-view 3D Reconstruction project...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Warning: Some dependencies may not have installed correctly")
    
    # Install pre-commit hooks
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        print("Warning: Pre-commit installation failed")
    else:
        run_command("pre-commit install", "Installing pre-commit hooks")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "results",
        "assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run basic tests
    if not run_command("python -m pytest tests/ -v", "Running basic tests"):
        print("Warning: Some tests failed")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py")
    print("2. Run evaluation: python scripts/evaluate.py --checkpoint checkpoints/best.pt")
    print("3. Launch demo: streamlit run demo/app.py")
    print("4. Open notebook: jupyter notebook notebooks/demo.ipynb")


if __name__ == "__main__":
    main()
