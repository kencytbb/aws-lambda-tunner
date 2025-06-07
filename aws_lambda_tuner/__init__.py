"""
AWS Lambda Performance Tuner

A comprehensive tool for optimizing AWS Lambda functions for cost and performance.
"""

__version__ = "2.0.0"
__author__ = "AWS Lambda Tuner Contributors"
__email__ = "support@lambdatuner.com"

# Import main components
from .config_module import TunerConfig, ConfigManager
from .orchestrator_module import TunerOrchestrator
from .report_service import ReportGenerator

# from .visualization_module import VisualizationEngine  # Commented out for architecture compatibility
from .analyzers.analyzer import PerformanceAnalyzer
from .reporting.service import ReportingService
from .models import (
    MemoryTestResult,
    Recommendation,
    PerformanceAnalysis,
    ColdStartAnalysis,
    ConcurrencyAnalysis,
    WorkloadAnalysis,
    TuningResult,
)
from .exceptions import (
    TunerException,
    ConfigurationError,
    LambdaExecutionError,
    InvalidPayloadError,
    AWSPermissionError,
    ValidationError,
    ReportGenerationError,
    VisualizationError,
)

# Convenience imports
from .orchestrator import run_tuning_session, test_single_configuration
from .reports import (
    generate_summary_report,
    generate_detailed_report,
    export_to_json,
    export_to_csv,
    export_to_html,
)

# from .visualization import (  # Commented out for architecture compatibility
#     create_performance_chart,
#     create_cost_chart,
#     create_distribution_chart,
#     create_optimization_chart
# )

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core classes
    "TunerConfig",
    "ConfigManager",
    "TunerOrchestrator",
    "ReportGenerator",
    "PerformanceAnalyzer",
    "ReportingService",
    # 'VisualizationEngine',  # Commented out for architecture compatibility
    # Data models
    "MemoryTestResult",
    "Recommendation",
    "PerformanceAnalysis",
    "ColdStartAnalysis",
    "ConcurrencyAnalysis",
    "WorkloadAnalysis",
    "TuningResult",
    # Exceptions
    "TunerException",
    "ConfigurationError",
    "LambdaExecutionError",
    "InvalidPayloadError",
    "AWSPermissionError",
    "ValidationError",
    "ReportGenerationError",
    "VisualizationError",
    # Convenience functions
    "run_tuning_session",
    "test_single_configuration",
    "generate_summary_report",
    "generate_detailed_report",
    "export_to_json",
    "export_to_csv",
    "export_to_html",
    # 'create_performance_chart',  # Commented out for architecture compatibility
    # 'create_cost_chart',
    # 'create_distribution_chart',
    # 'create_optimization_chart'
]
