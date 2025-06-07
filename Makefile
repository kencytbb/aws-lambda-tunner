.PHONY: help clean test coverage lint format install install-dev docs serve-docs build upload

help:
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make test         Run tests"
	@echo "  make coverage     Run tests with coverage"
	@echo "  make lint         Run linting"
	@echo "  make format       Format code"
	@echo "  make docs         Build documentation"
	@echo "  make serve-docs   Serve documentation locally"
	@echo "  make build        Build distribution packages"
	@echo "  make clean        Clean build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e .[dev,docs]
	pre-commit install

test:
	python -m pytest

coverage:
	python -m pytest --cov=aws_lambda_tuner --cov-report=html --cov-report=term

lint:
	flake8 aws_lambda_tuner tests
	mypy aws_lambda_tuner
	isort --check-only aws_lambda_tuner tests
	black --check aws_lambda_tuner tests

format:
	isort aws_lambda_tuner tests
	black aws_lambda_tuner tests

docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server

build:
	python -m build

upload:
	python -m twine upload dist/*

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete