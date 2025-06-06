"""Data models for AWS Lambda tuner."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class MemoryTestResult:
    """Results from testing a specific memory configuration."""
    memory_size: int
    iterations: int
    avg_duration: float
    p95_duration: float
    p99_duration: float
    avg_cost: float
    total_cost: float
    cold_starts: int
    errors: int
    raw_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Recommendation:
    """Optimization recommendation."""
    strategy: str
    current_memory_size: int
    optimal_memory_size: int
    should_optimize: bool
    cost_change_percent: float = 0.0
    duration_change_percent: float = 0.0
    reasoning: str = ""
    confidence_score: float = 0.0
    estimated_monthly_savings: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PerformanceAnalysis:
    """Complete performance analysis results."""
    memory_results: Dict[int, MemoryTestResult]
    efficiency_scores: Dict[int, float]
    cost_optimal: Dict[str, Any]
    speed_optimal: Dict[str, Any]
    balanced_optimal: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[Dict[str, Any]]
    baseline_comparison: Optional[Dict[str, Any]] = None


@dataclass
class TuningResult:
    """Complete tuning session results."""
    function_arn: str
    timestamp: datetime
    strategy: str
    memory_results: Dict[int, MemoryTestResult]
    baseline_results: Optional[List[Dict[str, Any]]]
    analysis: PerformanceAnalysis
    recommendation: Recommendation
    duration: float