"""
Custom exceptions for the AWS Lambda Tuner package.
"""


class TunerException(Exception):
    """Base exception for all tuner-related errors."""
    pass


class ConfigurationError(TunerException):
    """Raised when there's an error in the configuration."""
    pass


class LambdaExecutionError(TunerException):
    """Raised when Lambda execution fails."""
    pass


class InvalidPayloadError(TunerException):
    """Raised when the Lambda payload is invalid."""
    pass


class AWSPermissionError(TunerException):
    """Raised when AWS permissions are insufficient."""
    pass


class ValidationError(TunerException):
    """Raised when validation fails."""
    pass


class ReportGenerationError(TunerException):
    """Raised when report generation fails."""
    pass


class VisualizationError(TunerException):
    """Raised when visualization generation fails."""
    pass


class TemplateNotFoundError(TunerException):
    """Raised when a configuration template is not found."""
    pass


class ConcurrencyLimitError(TunerException):
    """Raised when concurrency limits are exceeded."""
    pass


class TimeoutError(TunerException):
    """Raised when operations timeout."""
    pass
