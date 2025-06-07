# AWS Lambda Tuner Testing Infrastructure

This document describes the comprehensive testing infrastructure for the AWS Lambda Tuner project, including testing strategies, how to run different test suites, and documentation of mocking strategies and test data.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Test Configuration](#test-configuration)
- [Mocking Strategy](#mocking-strategy)
- [Test Data Generation](#test-data-generation)
- [Test Helpers and Utilities](#test-helpers-and-utilities)
- [Coverage and Quality](#coverage-and-quality)
- [Performance Testing](#performance-testing)
- [Contributing to Tests](#contributing-to-tests)
- [Troubleshooting](#troubleshooting)

## Overview

The AWS Lambda Tuner testing infrastructure is designed to provide comprehensive coverage of all components and workflows while maintaining fast execution times and reliable results. The testing strategy follows a pyramid approach with unit tests at the base, integration tests in the middle, and end-to-end tests at the top, complemented by performance and stress tests.

### Testing Goals

- **Comprehensive Coverage**: Achieve >90% code coverage across all modules
- **Fast Feedback**: Unit tests complete in seconds, integration tests in minutes
- **Reliable Results**: Tests are deterministic and reproducible
- **Real-world Simulation**: Test scenarios reflect actual usage patterns
- **Performance Validation**: Ensure optimization algorithms scale efficiently

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Global pytest configuration and fixtures
├── README.md                      # This documentation
├── utils/                         # Testing utilities and helpers
│   ├── __init__.py
│   ├── fixtures.py               # Common test fixtures
│   ├── mock_aws.py              # AWS service mocking
│   ├── test_data_generators.py  # Realistic test data generation
│   └── test_helpers.py          # Assertion helpers and validators
├── unit/                         # Unit tests for individual components
│   ├── __init__.py
│   ├── test_models.py           # Data model tests
│   ├── test_analyzers.py        # Analyzer component tests
│   ├── test_config.py           # Configuration module tests
│   └── ...
├── integration/                  # Integration tests for component interactions
│   ├── __init__.py
│   ├── test_workload_optimization.py
│   └── ...
├── e2e/                         # End-to-end workflow tests
│   ├── __init__.py
│   ├── test_complete_optimization_scenarios.py
│   └── ...
└── performance/                 # Performance and load tests
    ├── __init__.py
    ├── test_optimization_algorithms.py
    └── ...
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests focus on individual components in isolation, using mocks for external dependencies.

**Markers**: `@pytest.mark.unit`

**Coverage**:
- Data models validation and serialization
- Individual analyzer components
- Configuration parsing and validation
- Utility functions
- Error handling and edge cases

**Example**:
```python
@pytest.mark.unit
def test_memory_test_result_validation(validator):
    result = MemoryTestResult(
        memory_size=512,
        iterations=5,
        avg_duration=120.0,
        # ... other parameters
    )
    assert validator.validate_memory_test_result(result, expected_memory_size=512)
```

### Integration Tests (`tests/integration/`)

Integration tests verify interactions between components and test complete workflows with mocked AWS services.

**Markers**: `@pytest.mark.integration`, `@pytest.mark.aws`

**Coverage**:
- Orchestrator with analyzer integration
- Workload-specific optimization workflows
- Strategy-specific optimization paths
- Configuration loading and processing
- Report generation workflows

**Example**:
```python
@pytest.mark.integration
@pytest.mark.workload_cpu
def test_cpu_intensive_workload_optimization(orchestrator, mock_aws_services):
    config = TunerConfig(
        function_arn='arn:aws:lambda:us-east-1:123456789012:function:test',
        memory_sizes=[512, 1024, 2048, 3008],
        strategy='speed',
        workload_type='cpu_intensive'
    )
    result = orchestrator.run_optimization(config)
    assert result.recommendation.optimal_memory_size >= 1024
```

### End-to-End Tests (`tests/e2e/`)

End-to-end tests validate complete user scenarios from configuration to final reports.

**Markers**: `@pytest.mark.e2e`, `@pytest.mark.slow`

**Coverage**:
- Complete optimization scenarios
- Multi-function optimization
- Configuration file workflows
- Report and visualization generation
- Error recovery scenarios

**Example**:
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_complete_cost_optimization_scenario(orchestrator, temp_dir):
    # Load config, run optimization, generate reports, save results
    config = TunerConfig.from_file('config.json')
    result = orchestrator.run_optimization(config)
    # Verify complete workflow including file outputs
```

### Performance Tests (`tests/performance/`)

Performance tests validate algorithm efficiency and scalability.

**Markers**: `@pytest.mark.performance`, `@pytest.mark.slow`, `@pytest.mark.benchmark`

**Coverage**:
- Algorithm performance under load
- Memory usage during optimization
- Scalability with different input sizes
- Concurrent execution performance
- Stress testing and memory leak detection

## Running Tests

### Prerequisites

Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m e2e                    # End-to-end tests only
pytest -m performance            # Performance tests only

# Run tests by directory
pytest tests/unit/               # All unit tests
pytest tests/integration/        # All integration tests
pytest tests/e2e/               # All e2e tests
pytest tests/performance/        # All performance tests
```

### Advanced Test Execution

```bash
# Run with coverage
pytest --cov=aws_lambda_tuner --cov-report=html

# Run in parallel
pytest -n auto                   # Auto-detect CPU cores
pytest -n 4                     # Use 4 processes

# Run specific workload tests
pytest -m workload_cpu           # CPU-intensive workload tests
pytest -m workload_io            # I/O-bound workload tests
pytest -m workload_memory        # Memory-intensive workload tests

# Run specific strategy tests
pytest -m strategy_cost          # Cost optimization tests
pytest -m strategy_speed         # Speed optimization tests
pytest -m strategy_balanced      # Balanced optimization tests

# Run with specific timeout
pytest --timeout=300             # 5-minute timeout per test

# Generate HTML report
pytest --html=test_report.html --self-contained-html
```

### Test Selection

```bash
# Run fast tests only (exclude slow tests)
pytest -m "not slow"

# Run AWS-related tests
pytest -m aws

# Run parametrized tests
pytest -m parametrize

# Combine markers
pytest -m "unit and not slow"
pytest -m "integration and workload_cpu"
```

## Test Configuration

### pytest Configuration

The test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers --strict-config"
markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for component interactions",
    "e2e: End-to-end tests for complete workflows",
    "performance: Performance and load tests",
    "slow: Tests that take more than 30 seconds",
    "aws: Tests that interact with AWS services (mocked)",
    # ... workload and strategy markers
]
timeout = 300
```

### Environment Variables

```bash
# Test environment configuration
export AWS_ACCESS_KEY_ID=testing
export AWS_SECRET_ACCESS_KEY=testing
export AWS_DEFAULT_REGION=us-east-1

# Performance test configuration
export PYTEST_TIMEOUT=600         # Extended timeout for performance tests
export PYTEST_MAXFAIL=5          # Stop after 5 failures
```

## Mocking Strategy

### AWS Service Mocking

The testing infrastructure uses comprehensive AWS service mocking to simulate real AWS Lambda behavior without actual AWS calls.

#### Mock Components

1. **MockLambdaClient** (`tests/utils/mock_aws.py`):
   - Simulates Lambda function invocations
   - Configurable performance characteristics
   - Realistic duration calculations based on memory
   - Cold start simulation
   - Error rate simulation

2. **MockCloudWatchClient**:
   - Simulates CloudWatch metrics
   - Mock metric data generation
   - Historical data simulation

3. **MockCloudWatchLogsClient**:
   - Simulates CloudWatch Logs
   - Mock log event generation
   - REPORT log parsing simulation

#### Mock Configuration

```python
# Configure mock behavior for specific test scenarios
mock_aws_services.configure_function_behavior(
    'test-function',
    base_duration=1000,      # Base execution time
    cold_start_rate=0.3,     # 30% cold start rate
    error_rate=0.02,         # 2% error rate
    memory_sensitivity=0.8   # High memory sensitivity
)
```

#### Workload-Specific Mocking

Different workload types are simulated with realistic performance characteristics:

- **CPU-intensive**: High base duration, high memory sensitivity
- **I/O-bound**: Lower base duration, low memory sensitivity, higher cold start impact
- **Memory-intensive**: Very high memory sensitivity, moderate base duration
- **Balanced**: Medium values across all metrics

## Test Data Generation

### TestDataGenerator Class

The `TestDataGenerator` class provides realistic test data for various scenarios:

```python
generator = TestDataGenerator(seed=42)  # Reproducible results

# Generate memory test results
result = generator.generate_memory_test_result(
    memory_size=512,
    iterations=10,
    workload_type='cpu_intensive'
)

# Generate complete tuning results
tuning_results = generator.generate_tuning_results(
    memory_sizes=[256, 512, 1024],
    workload_type='balanced'
)
```

### Data Characteristics

- **Realistic Performance Curves**: Memory vs. duration relationships reflect real Lambda behavior
- **Cost Calculations**: Based on actual AWS Lambda pricing
- **Workload Patterns**: Different workload types show appropriate sensitivity patterns
- **Variance and Noise**: Realistic variation in execution times
- **Error Simulation**: Configurable error rates and patterns

### Fixture-Based Data

Common test data is available through fixtures:

```python
def test_example(sample_memory_results, sample_performance_analysis):
    # Use pre-generated test data
    assert len(sample_memory_results) == 3
    assert sample_performance_analysis.efficiency_scores is not None
```

## Test Helpers and Utilities

### Validators (`TestValidators`)

Comprehensive validation functions for all data models:

```python
validator = TestValidators()

# Validate model structure and constraints
validator.validate_memory_test_result(result, expected_memory_size=512)
validator.validate_recommendation(rec, strategy='cost')
validator.validate_performance_analysis(analysis)
validator.validate_tuning_result(complete_result)
```

### Assertions (`TestAssertions`)

Enhanced assertions for domain-specific validations:

```python
assertions = TestAssertions()

# Validate performance trends
assertions.assert_memory_performance_trend(results, 'decreasing')
assertions.assert_cost_trend(results, 'increasing')
assertions.assert_efficiency_optimal(scores, optimal_memory)
assertions.assert_recommendation_consistency(recommendation, analysis)
```

### Test Helpers (`TestHelpers`)

Utility functions for test setup and common operations:

```python
helpers = TestHelpers()

# Create mock responses
response = helpers.create_mock_response(
    status_code=200,
    duration=150.0,
    cold_start=True
)

# Timing validations
helpers.assert_timing_reasonable(start_time, end_time)

# Approximate equality assertions
helpers.assert_approximately_equal(actual, expected, tolerance=0.05)
```

### Performance Helpers (`PerformanceTestHelpers`)

Specialized helpers for performance testing:

```python
perf_helpers = PerformanceTestHelpers()

# Measure execution time
result, duration = perf_helpers.measure_execution_time(func, *args)

# Assert performance bounds
perf_helpers.assert_performance_within_bounds(
    duration, 5.0, "Operation description"
)
```

## Coverage and Quality

### Coverage Requirements

- **Overall Coverage**: >90% line coverage
- **Branch Coverage**: >85% branch coverage
- **Critical Paths**: 100% coverage for optimization algorithms

### Coverage Reporting

```bash
# Generate coverage report
pytest --cov=aws_lambda_tuner --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html

# Generate XML report for CI
pytest --cov=aws_lambda_tuner --cov-report=xml
```

### Quality Checks

The test suite includes quality validations:

- **Model Consistency**: Verify data model constraints and relationships
- **Performance Bounds**: Ensure algorithms meet performance requirements
- **Memory Usage**: Monitor memory consumption during tests
- **Error Handling**: Validate graceful error handling and recovery

## Performance Testing

### Performance Test Categories

1. **Algorithm Performance**: Validate optimization algorithm efficiency
2. **Scalability Tests**: Test performance with varying input sizes
3. **Stress Tests**: Test system limits and error conditions
4. **Memory Tests**: Monitor memory usage and detect leaks
5. **Concurrency Tests**: Validate parallel execution performance

### Performance Benchmarks

Key performance requirements:

- **Single Function Optimization**: <30 seconds for typical configurations
- **Analysis Algorithm**: <2 seconds for large datasets
- **Memory Usage**: <500MB increase during optimization
- **Scalability**: Sub-linear scaling with input size

### Running Performance Tests

```bash
# Run all performance tests
pytest -m performance

# Run benchmarks only
pytest -m benchmark

# Run stress tests (slow)
pytest -m "performance and slow"

# Run with performance profiling
pytest -m performance --profile
```

## Contributing to Tests

### Adding New Tests

1. **Choose the Right Category**: Place tests in the appropriate directory (unit/integration/e2e/performance)
2. **Use Appropriate Markers**: Mark tests with relevant pytest markers
3. **Follow Naming Conventions**: Use descriptive test names starting with `test_`
4. **Use Fixtures**: Leverage existing fixtures and create new ones as needed
5. **Document Complex Tests**: Add docstrings for complex test scenarios

### Test Writing Guidelines

```python
@pytest.mark.unit
@pytest.mark.parametrize("input_value,expected", [
    (256, True),
    (0, False),
])
def test_memory_validation(input_value, expected, validator):
    """Test memory size validation with various inputs."""
    if expected:
        # Test valid case
        result = create_memory_test_result(memory_size=input_value)
        assert validator.validate_memory_test_result(result)
    else:
        # Test invalid case
        with pytest.raises(ConfigurationError):
            create_memory_test_result(memory_size=input_value)
```

### Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Deterministic Results**: Use fixed seeds for random data generation
3. **Clear Assertions**: Use descriptive assertion messages
4. **Appropriate Scope**: Keep unit tests focused, integration tests comprehensive
5. **Performance Awareness**: Be mindful of test execution time
6. **Mock Appropriately**: Mock external dependencies but not internal logic

## Troubleshooting

### Common Issues

#### Test Timeouts
```bash
# Increase timeout for slow tests
pytest --timeout=600 -m slow

# Run without timeout
pytest --timeout=0
```

#### Memory Issues
```bash
# Run tests sequentially to reduce memory usage
pytest -n 1

# Monitor memory usage
pytest --memray tests/performance/
```

#### AWS Mock Issues
```bash
# Verify AWS credentials are set for testing
export AWS_ACCESS_KEY_ID=testing
export AWS_SECRET_ACCESS_KEY=testing

# Reset mock state between tests
pytest --reset-mocks
```

#### Coverage Issues
```bash
# Debug coverage collection
pytest --cov=aws_lambda_tuner --cov-report=term-missing

# Check for uncovered lines
pytest --cov=aws_lambda_tuner --cov-fail-under=90
```

### Debugging Tests

```python
# Add debugging output
import pytest
pytest.set_trace()  # Breakpoint

# Use verbose output
pytest -v -s

# Show local variables in failures
pytest --tb=long

# Run specific test with debugging
pytest tests/unit/test_models.py::test_specific_function -v -s
```

### Test Data Issues

```python
# Reset test data generator
generator = TestDataGenerator(seed=42)  # Fixed seed

# Verify test data consistency
assert generator.generate_memory_test_result(256, 10, 'balanced').memory_size == 256

# Debug test data generation
data = generator.generate_tuning_results()
print(json.dumps(data, indent=2, default=str))
```

### CI/CD Integration

```yaml
# Example GitHub Actions configuration
- name: Run Tests
  run: |
    pytest -m "unit and not slow" --cov=aws_lambda_tuner
    pytest -m "integration and not slow"
    pytest -m "e2e and not slow" --maxfail=1

- name: Performance Tests
  run: pytest -m "performance and not slow" --benchmark-json=benchmark.json
```

## Support

For questions about the testing infrastructure:

1. Check this documentation first
2. Review existing test examples
3. Check the test utilities in `tests/utils/`
4. Refer to pytest documentation for advanced features
5. Create an issue for testing infrastructure improvements

---

This comprehensive testing infrastructure ensures the AWS Lambda Tuner is reliable, performant, and maintainable while providing fast feedback to developers and comprehensive validation of all functionality.