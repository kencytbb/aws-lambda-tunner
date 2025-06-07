# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Lambda Performance Tuner v2.0.0 is a comprehensive Python tool for optimizing AWS Lambda functions across different workload types with AI-powered recommendations, real-time monitoring, and professional reporting.

## Core Architecture

### Key Components
- **TunerOrchestrator** (`orchestrator_module.py`): Main coordination engine for optimization workflows
- **TunerConfig** (`config_module.py`): Configuration management with validation and template support
- **PerformanceAnalyzer** (`analyzers/analyzer.py`): Core performance analysis engine
- **CLI Module** (`cli_module.py`): Click-based command-line interface

### Modular Structure
- **`analyzers/`**: Workload-specific analyzers (on-demand, continuous, scheduled)
- **`strategies/`**: Optimization strategies (cost, speed, balanced, comprehensive) 
- **`intelligence/`**: AI components (recommendation engine, pattern recognizer, cost predictor)
- **`monitoring/`**: Performance monitoring and alerting systems
- **`reporting/`**: Report generation and multi-format export
- **`providers/`**: AWS service integration layer
- **`templates/`**: Configuration templates for different workload types

## Development Commands

### Make Commands
```bash
make install        # Install package in development mode
make install-dev    # Install with dev dependencies + pre-commit hooks
make test          # Run pytest test suite
make coverage      # Run tests with coverage report (90% minimum)
make lint          # Run flake8, mypy, isort, black checks
make format        # Format code with isort and black
make docs          # Build documentation
make build         # Build distribution packages
make clean         # Clean build artifacts
```

### Testing Commands
```bash
# Test categories (use pytest markers)
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only  
pytest -m e2e            # End-to-end tests only
pytest -m performance    # Performance tests only

# Workload-specific tests
pytest -m workload_cpu   # CPU-intensive workload tests
pytest -m strategy_cost  # Cost optimization strategy tests

# Coverage
pytest --cov=aws_lambda_tuner --cov-report=html
```

### CLI Entry Point
```bash
aws-lambda-tuner --version  # Verify installation
aws-lambda-tuner tune --config tuner.config.json
```

## Code Patterns

### Configuration Management
- Uses `@dataclass` with post-init validation
- Template-based configuration with workload types: `on_demand`, `continuous`, `scheduled` 
- Strategies: `cost`, `speed`, `balanced`, `comprehensive`
- Configuration files typically named `tuner.config.json`

### Async Orchestration
- Primary orchestration uses `async/await` for concurrent Lambda invocations
- `TunerOrchestrator.run_with_reporting()` is the main entry point
- Uses asyncio-throttle for rate limiting AWS API calls

### Data Models
Key dataclasses in `models.py`:
- **`MemoryTestResult`**: Individual memory configuration results
- **`PerformanceAnalysis`**: Complete analysis with metrics and recommendations
- **`TuningResult`**: Complete tuning session results with metadata

### Testing Architecture
- **Fixtures**: Comprehensive fixtures in `conftest.py` and `utils/fixtures.py`
- **Mocking**: Uses `moto` for AWS service mocking, custom mock providers in `utils/mock_aws.py`
- **Parametrized Tests**: Extensive use for different workload types, memory sizes, strategies
- **Test Data**: Generators in `utils/test_data_generators.py`

## Quality Standards

### Code Style
- **Line Length**: 100 characters (black configuration)
- **Type Hints**: Required for all functions and methods (mypy enforced)
- **Import Sorting**: isort with black profile
- **Coverage**: 90% minimum requirement

### File Naming Conventions
- Test files: `test_*.py`
- Module files: `*_module.py` for enhanced/refactored versions
- Analysis results: `tuning-results.json`, `*-results.json`
- Configuration: `tuner.config.json`, `*.config.json`

## Key Dependencies

### Runtime Dependencies
- **AWS**: boto3, aiohttp for AWS API integration
- **CLI**: click for command-line interface
- **Data**: pandas, numpy for analysis
- **Visualization**: matplotlib, seaborn, plotly for charts
- **AI/ML**: scikit-learn for pattern recognition

### Development Dependencies  
- **Testing**: pytest with extensive plugin support (asyncio, mock, benchmark, etc.)
- **Quality**: black, flake8, mypy, isort, pre-commit
- **AWS Mocking**: moto for comprehensive AWS service simulation

## Workload Types and Optimization

### Workload Types
- **`on_demand`**: API gateways, web applications (cold start sensitive)
- **`continuous`**: Stream processing, batch jobs (sustained performance)
- **`scheduled`**: Cron jobs, event-driven processing (periodic optimization)

### Memory Testing
- Default memory sizes: `[256, 512, 1024, 1536, 2048]` MB
- Testing typically uses 10-20 iterations per memory configuration
- Results stored in `tuning-results/` directory by default

## Development Workflow Requirements

### CRITICAL: Test-Driven Development
**ANY code change MUST include corresponding tests at the appropriate level:**

1. **Unit Tests** (`tests/unit/`): For individual functions/classes
2. **Integration Tests** (`tests/integration/`): For component interactions  
3. **E2E Tests** (`tests/e2e/`): For complete workflows
4. **Performance Tests** (`tests/performance/`): For optimization algorithms

### CRITICAL: Test Execution Validation
**Before any commit, MUST run and verify:**
```bash
make lint          # MUST pass - no exceptions
make test          # MUST pass all tests
make coverage      # MUST maintain 90% coverage minimum
```

### CRITICAL: Documentation Updates
**When adding/modifying features, MUST update:**
- **README.md**: Update CLI commands, examples, and user guide sections
- **Function docstrings**: Update type hints and parameter descriptions
- **Configuration templates**: Update JSON templates in `templates/` if config changes
- **Example files**: Update `examples/` if new usage patterns are introduced

### Test Categories Required
- **New functions**: Unit tests + integration tests
- **CLI commands**: CLI workflow tests in `tests/integration/test_cli_workflows.py`
- **Configuration changes**: Config validation tests in `tests/unit/test_config.py`
- **AWS integrations**: Mock AWS tests using moto in appropriate test files
- **Workload strategies**: Strategy-specific tests with proper markers

### Test Markers Usage
Use appropriate pytest markers for new tests:
```bash
@pytest.mark.unit                    # Individual component tests
@pytest.mark.integration             # Component interaction tests
@pytest.mark.e2e                     # End-to-end workflow tests
@pytest.mark.workload_<type>         # Workload-specific tests
@pytest.mark.strategy_<strategy>     # Strategy-specific tests
@pytest.mark.aws                     # AWS service interaction tests
```

## Report Generation

### Output Formats
- **Interactive**: HTML dashboards with Plotly.js visualizations
- **Export**: PDF, Excel, JSON, CSV formats via `reporting/export_formats.py`
- **Templates**: Workload-specific report templates in `reporting/workload_report_templates.py`

### Sample Reports
Example reports available in `examples/reports/`:
- `sample_web_api_report.html` - Web API optimization analysis
- `sample_cost_projection.html` - Financial impact analysis  
- `sample_dashboard.html` - Real-time monitoring dashboard