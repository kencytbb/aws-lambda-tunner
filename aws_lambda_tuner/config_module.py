"""
Configuration management for AWS Lambda Tuner.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

from .exceptions import ConfigurationError, TemplateNotFoundError
from .utils import validate_arn, get_memory_sizes, load_json_file

logger = logging.getLogger(__name__)


@dataclass
class TunerConfig:
    """Configuration for Lambda tuning operations."""

    function_arn: str

    # Workload pattern configuration
    workload_type: Literal["on_demand", "scheduled", "continuous"] = "on_demand"
    expected_concurrency: int = 10
    traffic_pattern: Literal["burst", "steady", "variable"] = "burst"
    cold_start_sensitivity: Literal["low", "medium", "high"] = "medium"

    payload: Union[str, dict] = "{}"
    memory_sizes: Optional[List[int]] = None
    iterations: int = 10
    strategy: str = "balanced"
    concurrent_executions: int = 5
    timeout: int = 300
    dry_run: bool = False
    output_dir: str = "./tuning-results"
    warmup_runs: int = 2
    region: Optional[str] = None
    profile: Optional[str] = None

    # Additional fields for comprehensive compatibility
    parallel_invocations: bool = True
    warmup_invocations: int = 2
    discard_outliers: float = 0.2
    cost_per_gb_second: float = 0.0000166667
    cost_per_request: float = 0.0000002
    skip_baseline: bool = False
    baseline_samples: int = 20
    validation_samples: int = 20
    auto_optimize: bool = False
    report_formats: List[str] = field(default_factory=lambda: ["json", "html"])
    include_charts: bool = True
    include_raw_data: bool = False

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Validate ARN
        if not validate_arn(self.function_arn):
            raise ConfigurationError(f"Invalid Lambda function ARN: {self.function_arn}")

        # Set default memory sizes based on strategy if not provided
        if self.memory_sizes is None:
            self.memory_sizes = get_memory_sizes(self.strategy)

        # Validate memory sizes
        for size in self.memory_sizes:
            if not (128 <= size <= 10240):
                raise ConfigurationError(
                    f"Invalid memory size: {size}. Must be between 128 and 10240 MB"
                )
            if size % 64 != 0 and size != 128:
                raise ConfigurationError(
                    f"Invalid memory size: {size}. Must be a multiple of 64 MB"
                )

        # Validate iterations
        if self.iterations < 1:
            raise ConfigurationError("Iterations must be at least 1")

        # Validate workload type
        if self.workload_type not in ["on_demand", "scheduled", "continuous"]:
            raise ConfigurationError(
                "workload_type must be one of: on_demand, scheduled, continuous"
            )

        # Validate traffic pattern
        if self.traffic_pattern not in ["burst", "steady", "variable"]:
            raise ConfigurationError("traffic_pattern must be one of: burst, steady, variable")

        # Validate cold start sensitivity
        if self.cold_start_sensitivity not in ["low", "medium", "high"]:
            raise ConfigurationError("cold_start_sensitivity must be one of: low, medium, high")

        # Validate expected concurrency
        if self.expected_concurrency < 1:
            raise ConfigurationError("expected_concurrency must be at least 1")

        # Workload-specific validation
        self._validate_workload_configuration()

        # Validate strategy
        valid_strategies = ["speed", "cost", "balanced", "comprehensive"]
        if self.strategy not in valid_strategies:
            raise ConfigurationError(
                f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}"
            )

        # Validate concurrent executions
        if self.concurrent_executions < 1:
            raise ConfigurationError("Concurrent executions must be at least 1")

        # Extract region from ARN if not provided
        if not self.region:
            try:
                arn_parts = self.function_arn.split(":")
                self.region = arn_parts[3]
            except IndexError:
                raise ConfigurationError("Could not extract region from ARN")

    def _validate_workload_configuration(self):
        """Validate workload-specific configuration."""
        import warnings

        # On-demand workloads should prioritize cold start reduction
        if self.workload_type == "on_demand":
            if (
                self.cold_start_sensitivity == "low"
                and self.memory_sizes
                and min(self.memory_sizes) < 512
            ):
                warnings.warn(
                    "On-demand workloads with low cold start sensitivity may benefit from higher memory allocations"
                )

        # Continuous workloads should optimize for sustained performance
        elif self.workload_type == "continuous":
            if self.traffic_pattern == "burst" and self.expected_concurrency > 100:
                warnings.warn(
                    "Continuous workloads with burst traffic at high concurrency may experience throttling"
                )

        # Scheduled workloads should balance cost and readiness
        elif self.workload_type == "scheduled":
            if self.cold_start_sensitivity == "high" and self.strategy == "cost":
                warnings.warn(
                    "Scheduled workloads with high cold start sensitivity may conflict with cost optimization strategy"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TunerConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, filepath: str) -> "TunerConfig":
        """Load configuration from file."""
        data = load_json_file(filepath)
        return cls.from_dict(data)

    def save(self, filepath: str):
        """Save configuration to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigManager:
    """Manages configuration loading, validation, and templates."""

    def __init__(self):
        """Initialize configuration manager."""
        self.templates_dir = Path(__file__).parent / "templates"
        self._templates_cache = {}

    def get_template_path(self, template_name: str) -> Path:
        """Get path to configuration template."""
        template_file = f"{template_name}_optimized.json"
        if template_name == "balanced":
            template_file = "comprehensive.json"
        elif template_name in ["on_demand", "continuous", "scheduled"]:
            template_file = f"{template_name}_optimized.json"

        template_path = self.templates_dir / template_file
        if not template_path.exists():
            raise TemplateNotFoundError(f"Template not found: {template_name}")

        return template_path

    def load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a configuration template."""
        if template_name in self._templates_cache:
            return self._templates_cache[template_name].copy()

        template_path = self.get_template_path(template_name)
        template_data = load_json_file(str(template_path))

        self._templates_cache[template_name] = template_data
        return template_data.copy()

    def create_from_template(self, template_name: str, **overrides) -> TunerConfig:
        """Create configuration from template with overrides."""
        template_data = self.load_template(template_name)

        # Apply overrides
        for key, value in overrides.items():
            if value is not None:
                template_data[key] = value

        # Set default function ARN if not provided
        if "function_arn" not in template_data or not template_data.get("function_arn"):
            template_data["function_arn"] = (
                "arn:aws:lambda:us-east-1:123456789012:function:my-function"
            )

        return TunerConfig.from_dict(template_data)

    def merge_configs(
        self, base_config: TunerConfig, override_config: Dict[str, Any]
    ) -> TunerConfig:
        """Merge configuration with overrides."""
        base_dict = base_config.to_dict()

        # Merge with overrides
        for key, value in override_config.items():
            if value is not None:
                base_dict[key] = value

        return TunerConfig.from_dict(base_dict)

    def validate_config(self, config: TunerConfig) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        # Check if using small memory sizes with speed strategy
        if config.strategy == "speed" and min(config.memory_sizes) < 512:
            warnings.append(
                "Using small memory sizes with 'speed' strategy may not yield optimal results"
            )

        # Check if using large memory sizes with cost strategy
        if config.strategy == "cost" and max(config.memory_sizes) > 2048:
            warnings.append("Using large memory sizes with 'cost' strategy may increase costs")

        # Workload-specific validations
        if config.workload_type == "on_demand":
            if config.cold_start_sensitivity == "high" and min(config.memory_sizes) < 512:
                warnings.append(
                    "On-demand workloads with high cold start sensitivity should use memory >= 512MB"
                )
            if config.expected_concurrency > 50:
                warnings.append(
                    "On-demand workloads with high expected concurrency may benefit from provisioned concurrency"
                )

        elif config.workload_type == "continuous":
            if config.traffic_pattern == "burst" and config.expected_concurrency > 100:
                warnings.append(
                    "Continuous workloads with burst patterns at high concurrency may experience throttling"
                )
            if config.strategy == "cost" and config.cold_start_sensitivity == "high":
                warnings.append(
                    "Continuous workloads prioritizing cost may conflict with cold start requirements"
                )

        elif config.workload_type == "scheduled":
            if config.cold_start_sensitivity == "high" and config.strategy == "cost":
                warnings.append(
                    "Scheduled workloads with high cold start sensitivity may conflict with cost optimization"
                )
            if config.traffic_pattern == "burst" and config.iterations < 10:
                warnings.append(
                    "Scheduled workloads with burst patterns should use more iterations for reliable results"
                )

        # Traffic pattern validations
        if config.traffic_pattern == "burst" and config.concurrent_executions < 5:
            warnings.append(
                "Burst traffic patterns should use higher concurrent executions for realistic testing"
            )

        # Check iteration count
        if config.iterations < 5:
            warnings.append("Low iteration count may produce unreliable results")
        elif config.iterations > 100:
            warnings.append("High iteration count will significantly increase tuning time and cost")

        # Check concurrent executions
        if config.concurrent_executions > 20:
            warnings.append("High concurrent executions may hit Lambda concurrency limits")

        # Check timeout
        if config.timeout < 60:
            warnings.append("Low timeout may cause premature termination of long-running functions")

        return warnings

    def create_default_templates(self):
        """Create default configuration templates if they don't exist."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        templates = {
            "speed_optimized.json": {
                "memory_sizes": [1024, 1536, 2048, 3008],
                "iterations": 20,
                "strategy": "speed",
                "concurrent_executions": 10,
                "timeout": 300,
                "warmup_runs": 3,
            },
            "cost_optimized.json": {
                "memory_sizes": [128, 256, 512, 768, 1024],
                "iterations": 15,
                "strategy": "cost",
                "concurrent_executions": 5,
                "timeout": 300,
                "warmup_runs": 2,
            },
            "comprehensive.json": {
                "memory_sizes": [128, 256, 512, 768, 1024, 1536, 2048, 2560, 3008],
                "iterations": 10,
                "strategy": "balanced",
                "concurrent_executions": 5,
                "timeout": 300,
                "warmup_runs": 2,
            },
            "development.json": {
                "memory_sizes": [512, 1024],
                "iterations": 3,
                "strategy": "balanced",
                "concurrent_executions": 2,
                "timeout": 60,
                "warmup_runs": 1,
                "dry_run": True,
            },
            "on_demand_optimized.json": {
                "workload_type": "on_demand",
                "expected_concurrency": 5,
                "traffic_pattern": "burst",
                "cold_start_sensitivity": "high",
                "memory_sizes": [512, 1024, 1536, 2048, 3008],
                "iterations": 12,
                "strategy": "speed",
                "concurrent_executions": 8,
                "timeout": 300,
                "warmup_runs": 3,
            },
            "continuous_optimized.json": {
                "workload_type": "continuous",
                "expected_concurrency": 50,
                "traffic_pattern": "steady",
                "cold_start_sensitivity": "medium",
                "memory_sizes": [256, 512, 768, 1024, 1536, 2048],
                "iterations": 15,
                "strategy": "balanced",
                "concurrent_executions": 10,
                "timeout": 300,
                "warmup_runs": 2,
                "auto_optimize": True,
            },
            "scheduled_optimized.json": {
                "workload_type": "scheduled",
                "expected_concurrency": 20,
                "traffic_pattern": "variable",
                "cold_start_sensitivity": "medium",
                "memory_sizes": [128, 256, 512, 1024, 1536],
                "iterations": 10,
                "strategy": "balanced",
                "concurrent_executions": 6,
                "timeout": 300,
                "warmup_runs": 2,
            },
        }

        for filename, template_data in templates.items():
            template_path = self.templates_dir / filename
            if not template_path.exists():
                # Add default values
                template_data.update(
                    {"function_arn": "", "payload": "{}", "output_dir": "./tuning-results"}
                )

                with open(template_path, "w") as f:
                    json.dump(template_data, f, indent=2)

                logger.info(f"Created template: {filename}")


# Create global config manager instance
config_manager = ConfigManager()
