"""Configuration management for AWS Lambda tuner."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal


@dataclass
class TunerConfig:
    """Configuration for Lambda performance tuning."""

    # Required settings
    function_arn: str = ""

    # Workload pattern configuration
    workload_type: Literal["on_demand", "scheduled", "continuous"] = "on_demand"
    expected_concurrency: int = 10
    traffic_pattern: Literal["burst", "steady", "variable"] = "burst"
    cold_start_sensitivity: Literal["low", "medium", "high"] = "medium"

    # Memory configuration
    memory_sizes: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 1536, 2048, 3008])

    # Test configuration
    iterations: int = 10
    payload: Dict[str, Any] = field(default_factory=dict)
    strategy: str = "balanced"  # cost, speed, balanced
    parallel_invocations: bool = True

    # AWS configuration
    region: Optional[str] = None

    # Advanced options
    warmup_invocations: int = 2
    timeout: int = 300
    discard_outliers: float = 0.2
    cost_per_gb_second: float = 0.0000166667  # AWS Lambda pricing
    cost_per_request: float = 0.0000002

    # Baseline and validation
    skip_baseline: bool = False
    baseline_samples: int = 20
    validation_samples: int = 20

    # Optimization
    auto_optimize: bool = False

    # Auto-retuning triggers
    auto_retuning_enabled: bool = False
    auto_retuning_triggers: Dict[str, Any] = field(
        default_factory=lambda: {
            "performance_degradation": {
                "enabled": True,
                "duration_degradation_threshold": 0.2,  # 20% degradation
                "error_rate_threshold": 0.05,  # 5% error rate
                "auto_approve": False,
            },
            "cost_threshold": {
                "enabled": True,
                "monthly_cost_threshold": None,  # Set by user
                "cost_increase_threshold": 0.3,  # 30% cost increase
                "auto_approve": True,
            },
            "traffic_pattern_change": {
                "enabled": True,
                "pattern_change_threshold": 0.7,  # Pattern change score
                "auto_approve": True,
            },
            "scheduled": {
                "enabled": False,
                "schedule": "0 2 * * 0",  # Weekly at 2 AM Sunday (cron format)
                "auto_approve": True,
            },
        }
    )

    # Monitoring configuration
    monitoring_enabled: bool = False
    monitoring_interval_minutes: int = 5
    baseline_window_hours: int = 24
    performance_window_minutes: int = 15

    # Alert configuration
    alerts_enabled: bool = False
    alert_channels: List[str] = field(default_factory=lambda: ["email"])
    alert_cooldown_minutes: int = 30

    # Notification settings
    email_notifications: Dict[str, str] = field(default_factory=dict)
    sns_topic_arn: Optional[str] = None
    slack_webhook_url: Optional[str] = None

    # Reporting
    output_dir: str = "./reports"
    report_formats: List[str] = field(default_factory=lambda: ["json", "html"])
    include_charts: bool = True
    include_raw_data: bool = False

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.function_arn:
            raise ValueError("function_arn is required")

        if not self.region:
            # Try to detect region from ARN or environment
            if ":" in self.function_arn:
                parts = self.function_arn.split(":")
                if len(parts) >= 4:
                    self.region = parts[3]

            if not self.region:
                self.region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Validate workload type
        if self.workload_type not in ["on_demand", "scheduled", "continuous"]:
            raise ValueError("workload_type must be one of: on_demand, scheduled, continuous")

        # Validate traffic pattern
        if self.traffic_pattern not in ["burst", "steady", "variable"]:
            raise ValueError("traffic_pattern must be one of: burst, steady, variable")

        # Validate cold start sensitivity
        if self.cold_start_sensitivity not in ["low", "medium", "high"]:
            raise ValueError("cold_start_sensitivity must be one of: low, medium, high")

        # Validate expected concurrency
        if self.expected_concurrency < 1:
            raise ValueError("expected_concurrency must be at least 1")

        # Workload-specific validation
        self._validate_workload_configuration()

        if self.strategy not in ["cost", "speed", "balanced"]:
            raise ValueError("strategy must be one of: cost, speed, balanced")

        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")

        if not self.memory_sizes:
            raise ValueError("memory_sizes cannot be empty")

        # Validate memory sizes
        for size in self.memory_sizes:
            if size < 128 or size > 10240:
                raise ValueError(f"Invalid memory size: {size}MB (must be 128-10240)")

        # Validate auto-retuning configuration
        if self.auto_retuning_enabled:
            self._validate_auto_retuning_configuration()

        # Validate monitoring configuration
        if self.monitoring_enabled:
            self._validate_monitoring_configuration()

        # Validate alert configuration
        if self.alerts_enabled:
            self._validate_alert_configuration()

    def _validate_workload_configuration(self):
        """Validate workload-specific configuration."""
        # On-demand workloads should prioritize cold start reduction
        if self.workload_type == "on_demand":
            if self.cold_start_sensitivity == "low" and min(self.memory_sizes or [128]) < 512:
                import warnings

                warnings.warn(
                    "On-demand workloads with low cold start sensitivity may benefit from higher memory allocations"
                )

        # Continuous workloads should optimize for sustained performance
        elif self.workload_type == "continuous":
            if self.traffic_pattern == "burst" and self.expected_concurrency > 100:
                import warnings

                warnings.warn(
                    "Continuous workloads with burst traffic at high concurrency may experience throttling"
                )

        # Scheduled workloads should balance cost and readiness
        elif self.workload_type == "scheduled":
            if self.cold_start_sensitivity == "high" and self.strategy == "cost":
                import warnings

                warnings.warn(
                    "Scheduled workloads with high cold start sensitivity may conflict with cost optimization strategy"
                )

    def _validate_auto_retuning_configuration(self):
        """Validate auto-retuning configuration."""
        if not isinstance(self.auto_retuning_triggers, dict):
            raise ValueError("auto_retuning_triggers must be a dictionary")

        # Validate trigger configurations
        for trigger_name, trigger_config in self.auto_retuning_triggers.items():
            if not isinstance(trigger_config, dict):
                raise ValueError(f"Trigger config for '{trigger_name}' must be a dictionary")

            if trigger_name == "performance_degradation":
                threshold = trigger_config.get("duration_degradation_threshold", 0.2)
                if not 0 < threshold < 1:
                    raise ValueError("duration_degradation_threshold must be between 0 and 1")

                error_threshold = trigger_config.get("error_rate_threshold", 0.05)
                if not 0 < error_threshold < 1:
                    raise ValueError("error_rate_threshold must be between 0 and 1")

            elif trigger_name == "cost_threshold":
                cost_threshold = trigger_config.get("monthly_cost_threshold")
                if cost_threshold is not None and cost_threshold <= 0:
                    raise ValueError("monthly_cost_threshold must be positive")

                cost_increase = trigger_config.get("cost_increase_threshold", 0.3)
                if not 0 < cost_increase < 2:
                    raise ValueError("cost_increase_threshold must be between 0 and 2")

            elif trigger_name == "traffic_pattern_change":
                pattern_threshold = trigger_config.get("pattern_change_threshold", 0.7)
                if not 0 < pattern_threshold < 1:
                    raise ValueError("pattern_change_threshold must be between 0 and 1")

            elif trigger_name == "scheduled":
                schedule = trigger_config.get("schedule")
                if trigger_config.get("enabled", False) and not schedule:
                    raise ValueError("Schedule must be provided when scheduled trigger is enabled")

    def _validate_monitoring_configuration(self):
        """Validate monitoring configuration."""
        if self.monitoring_interval_minutes < 1:
            raise ValueError("monitoring_interval_minutes must be at least 1")

        if self.baseline_window_hours < 1:
            raise ValueError("baseline_window_hours must be at least 1")

        if self.performance_window_minutes < 1:
            raise ValueError("performance_window_minutes must be at least 1")

    def _validate_alert_configuration(self):
        """Validate alert configuration."""
        valid_channels = ["email", "sns", "slack", "webhook", "cloudwatch"]

        for channel in self.alert_channels:
            if channel not in valid_channels:
                raise ValueError(
                    f"Invalid alert channel: {channel}. Must be one of: {valid_channels}"
                )

        if self.alert_cooldown_minutes < 1:
            raise ValueError("alert_cooldown_minutes must be at least 1")

        # Validate notification configurations
        if "email" in self.alert_channels and not self.email_notifications:
            import warnings

            warnings.warn("Email alerts enabled but no email configuration provided")

        if "sns" in self.alert_channels and not self.sns_topic_arn:
            import warnings

            warnings.warn("SNS alerts enabled but no SNS topic ARN provided")

        if "slack" in self.alert_channels and not self.slack_webhook_url:
            import warnings

            warnings.warn("Slack alerts enabled but no webhook URL provided")

    @classmethod
    def from_file(cls, config_path: str) -> "TunerConfig":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "function_arn": self.function_arn,
            "workload_type": self.workload_type,
            "expected_concurrency": self.expected_concurrency,
            "traffic_pattern": self.traffic_pattern,
            "cold_start_sensitivity": self.cold_start_sensitivity,
            "memory_sizes": self.memory_sizes,
            "iterations": self.iterations,
            "payload": self.payload,
            "strategy": self.strategy,
            "parallel_invocations": self.parallel_invocations,
            "region": self.region,
            "warmup_invocations": self.warmup_invocations,
            "timeout": self.timeout,
            "discard_outliers": self.discard_outliers,
            "cost_per_gb_second": self.cost_per_gb_second,
            "cost_per_request": self.cost_per_request,
            "skip_baseline": self.skip_baseline,
            "baseline_samples": self.baseline_samples,
            "validation_samples": self.validation_samples,
            "auto_optimize": self.auto_optimize,
            "auto_retuning_enabled": self.auto_retuning_enabled,
            "auto_retuning_triggers": self.auto_retuning_triggers,
            "monitoring_enabled": self.monitoring_enabled,
            "monitoring_interval_minutes": self.monitoring_interval_minutes,
            "baseline_window_hours": self.baseline_window_hours,
            "performance_window_minutes": self.performance_window_minutes,
            "alerts_enabled": self.alerts_enabled,
            "alert_channels": self.alert_channels,
            "alert_cooldown_minutes": self.alert_cooldown_minutes,
            "email_notifications": self.email_notifications,
            "sns_topic_arn": self.sns_topic_arn,
            "slack_webhook_url": self.slack_webhook_url,
            "output_dir": self.output_dir,
            "report_formats": self.report_formats,
            "include_charts": self.include_charts,
            "include_raw_data": self.include_raw_data,
        }

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_default_config(function_arn: str, output_path: str = "tuner.config.json") -> TunerConfig:
    """Create a default configuration file."""
    config = TunerConfig(
        function_arn=function_arn,
        memory_sizes=[128, 256, 512, 1024, 1536, 2048, 3008],
        iterations=10,
        payload={},
        strategy="balanced",
        parallel_invocations=True,
        auto_optimize=False,
        report_formats=["json", "html"],
    )

    config.save_to_file(output_path)
    return config
