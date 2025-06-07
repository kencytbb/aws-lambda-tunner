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
class ColdStartAnalysis:
    """Analysis results for cold start patterns."""

    cold_start_ratio: float
    avg_cold_start_duration: float
    avg_warm_start_duration: float
    cold_start_impact_score: float
    memory_vs_cold_start_correlation: float
    optimal_memory_for_cold_starts: int
    cold_start_patterns: Dict[str, Any]


@dataclass
class ConcurrencyPattern:
    """Concurrency pattern analysis results."""

    pattern_type: str  # 'burst', 'steady', 'gradual_ramp', 'spike'
    frequency: float  # How often this pattern occurs
    intensity: float  # Magnitude of the pattern
    duration_ms: float  # How long the pattern typically lasts
    impact_on_performance: float  # Impact score on overall performance
    recommendations: List[str]  # Specific recommendations for this pattern


@dataclass
class ConcurrencyAnalysis:
    """Analysis results for concurrency patterns."""

    avg_concurrent_executions: float
    peak_concurrent_executions: int
    concurrency_utilization: float
    scaling_efficiency: float
    throttling_events: int
    recommended_concurrency_limit: Optional[int]
    concurrency_patterns: Dict[str, Any]
    identified_patterns: List[ConcurrencyPattern] = field(default_factory=list)


@dataclass
class WorkloadAnalysis:
    """Analysis results for workload-specific optimization."""

    workload_type: str
    resource_utilization: Dict[str, float]
    optimization_opportunities: List[Dict[str, Any]]
    workload_specific_recommendations: List[Dict[str, Any]]
    cost_vs_performance_curve: Dict[str, Any]


@dataclass
class TimeBasedTrend:
    """Time-based performance trend analysis."""

    time_period: str
    metric_trends: Dict[str, List[float]]
    seasonal_patterns: Dict[str, Any]
    performance_degradation: bool
    trend_confidence: float
    forecast: Dict[str, Any]


@dataclass
class AdvancedPerformanceAnalysis:
    """Extended performance analysis with advanced metrics."""

    memory_results: Dict[int, MemoryTestResult]
    efficiency_scores: Dict[int, float]
    cost_optimal: Dict[str, Any]
    speed_optimal: Dict[str, Any]
    balanced_optimal: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[Dict[str, Any]]
    baseline_comparison: Optional[Dict[str, Any]] = None
    cold_start_analysis: Optional[ColdStartAnalysis] = None
    concurrency_analysis: Optional[ConcurrencyAnalysis] = None
    workload_analysis: Optional[WorkloadAnalysis] = None
    time_based_trends: Optional[List[TimeBasedTrend]] = None


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
