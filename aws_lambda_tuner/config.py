"""Configuration management for AWS Lambda tuner."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class TunerConfig:
    """Configuration for Lambda performance tuning."""
    
    # Required settings
    function_arn: str = ""
    
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
                self.region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        
        if self.strategy not in ['cost', 'speed', 'balanced']:
            raise ValueError("strategy must be one of: cost, speed, balanced")
        
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        
        if not self.memory_sizes:
            raise ValueError("memory_sizes cannot be empty")
        
        # Validate memory sizes
        for size in self.memory_sizes:
            if size < 128 or size > 10240:
                raise ValueError(f"Invalid memory size: {size}MB (must be 128-10240)")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TunerConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'function_arn': self.function_arn,
            'memory_sizes': self.memory_sizes,
            'iterations': self.iterations,
            'payload': self.payload,
            'strategy': self.strategy,
            'parallel_invocations': self.parallel_invocations,
            'region': self.region,
            'warmup_invocations': self.warmup_invocations,
            'timeout': self.timeout,
            'auto_optimize': self.auto_optimize,
            'output_dir': self.output_dir,
            'report_formats': self.report_formats,
            'include_charts': self.include_charts
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
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
        report_formats=["json", "html"]
    )
    
    config.save_to_file(output_path)
    return config