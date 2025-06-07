# Contributing to AWS Lambda Tuner

First off, thank you for considering contributing to AWS Lambda Tuner! It's people like you that make this tool better for everyone.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include logs and error messages if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python style guide (PEP 8)
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Development Process

1. Fork the repo and create your branch from `main`
2. Set up your development environment:
   ```bash
   git clone https://github.com/your-username/aws-lambda-tunner.git
   cd aws-lambda-tunner
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. Make your changes:
   * Write/update tests as needed
   * Follow the existing code style
   * Update documentation as needed

4. Run tests and linting:
   ```bash
   # Run tests
   pytest
   
   # Run linting
   black aws_lambda_tuner tests
   flake8 aws_lambda_tuner tests
   mypy aws_lambda_tuner
   ```

5. Commit your changes:
   * Use clear and meaningful commit messages
   * Reference issues and pull requests liberally

6. Push to your fork and submit a pull request

## Python Style Guide

* Follow PEP 8
* Use Black for code formatting
* Use type hints where appropriate
* Maximum line length is 100 characters
* Use descriptive variable names
* Add docstrings to all public functions and classes

## Testing

* Write unit tests for all new functionality
* Ensure all tests pass before submitting PR
* Aim for high test coverage (>80%)
* Use pytest for testing
* Mock external dependencies (AWS services)

## Documentation

* Update README.md if needed
* Add docstrings to new functions/classes
* Update CHANGELOG.md for notable changes
* Include examples for new features

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create a pull request
4. After merge, create a release on GitHub
5. Package will be automatically published to PyPI

## Questions?

Feel free to open an issue for any questions about contributing!
