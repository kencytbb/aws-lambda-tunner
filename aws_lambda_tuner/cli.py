"""
CLI entry point for AWS Lambda Tuner.
This module imports and exposes the main CLI functionality.
"""

from .cli_module import cli, main

__all__ = ["cli", "main"]

# Allow running as module
if __name__ == "__main__":
    main()
