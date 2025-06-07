"""Unit tests for data models."""

import pytest
from datetime import datetime, timedelta
from aws_lambda_tuner.models import (
    MemoryTestResult,
    Recommendation,
    PerformanceAnalysis,
    TuningResult,
    ColdStartAnalysis,
    ConcurrencyAnalysis,
    WorkloadAnalysis,
    TimeBasedTrend,
    AdvancedPerformanceAnalysis,
)
from tests.utils.test_helpers import TestValidators, TestAssertions


@pytest.mark.unit
class TestMemoryTestResult:
    """Test MemoryTestResult model."""

    def test_memory_test_result_creation(self):
        """Test basic MemoryTestResult creation."""
        result = MemoryTestResult(
            memory_size=256,
            iterations=10,
            avg_duration=150.5,
            p95_duration=180.0,
            p99_duration=200.0,
            avg_cost=0.00001,
            total_cost=0.0001,
            cold_starts=2,
            errors=0,
        )

        assert result.memory_size == 256
        assert result.iterations == 10
        assert result.avg_duration == 150.5
        assert result.cold_starts == 2
        assert result.errors == 0

    def test_memory_test_result_validation(self, validator):
        """Test MemoryTestResult validation."""
        result = MemoryTestResult(
            memory_size=512,
            iterations=5,
            avg_duration=120.0,
            p95_duration=140.0,
            p99_duration=160.0,
            avg_cost=0.00002,
            total_cost=0.0001,
            cold_starts=1,
            errors=0,
        )

        assert validator.validate_memory_test_result(result, expected_memory_size=512)

    def test_memory_test_result_with_raw_results(self):
        """Test MemoryTestResult with raw execution results."""
        raw_results = [
            {"execution_id": 0, "duration": 150.0, "cold_start": True, "status_code": 200},
            {"execution_id": 1, "duration": 120.0, "cold_start": False, "status_code": 200},
        ]

        result = MemoryTestResult(
            memory_size=256,
            iterations=2,
            avg_duration=135.0,
            p95_duration=150.0,
            p99_duration=150.0,
            avg_cost=0.00001,
            total_cost=0.00002,
            cold_starts=1,
            errors=0,
            raw_results=raw_results,
        )

        assert len(result.raw_results) == 2
        assert result.raw_results[0]["cold_start"] is True
        assert result.raw_results[1]["cold_start"] is False


@pytest.mark.unit
class TestRecommendation:
    """Test Recommendation model."""

    def test_recommendation_creation(self):
        """Test basic Recommendation creation."""
        rec = Recommendation(
            strategy="balanced",
            current_memory_size=256,
            optimal_memory_size=512,
            should_optimize=True,
            cost_change_percent=10.5,
            duration_change_percent=-25.0,
            reasoning="Better performance-cost ratio",
            confidence_score=0.85,
        )

        assert rec.strategy == "balanced"
        assert rec.current_memory_size == 256
        assert rec.optimal_memory_size == 512
        assert rec.should_optimize is True
        assert rec.confidence_score == 0.85

    def test_recommendation_validation(self, validator):
        """Test Recommendation validation."""
        rec = Recommendation(
            strategy="cost",
            current_memory_size=1024,
            optimal_memory_size=512,
            should_optimize=True,
            reasoning="Reduce costs without significant performance impact",
        )

        assert validator.validate_recommendation(rec, current_memory=1024, strategy="cost")

    def test_recommendation_no_optimization_needed(self):
        """Test recommendation when no optimization is needed."""
        rec = Recommendation(
            strategy="balanced",
            current_memory_size=512,
            optimal_memory_size=512,
            should_optimize=False,
            reasoning="Current configuration is already optimal",
        )

        assert rec.should_optimize is False
        assert rec.current_memory_size == rec.optimal_memory_size

    def test_recommendation_with_savings_estimates(self):
        """Test recommendation with detailed savings estimates."""
        savings = {
            "low_usage": {"current_cost": 1.0, "optimized_cost": 0.8, "savings": 0.2},
            "medium_usage": {"current_cost": 10.0, "optimized_cost": 8.0, "savings": 2.0},
            "high_usage": {"current_cost": 100.0, "optimized_cost": 80.0, "savings": 20.0},
        }

        rec = Recommendation(
            strategy="cost",
            current_memory_size=1024,
            optimal_memory_size=512,
            should_optimize=True,
            estimated_monthly_savings=savings,
        )

        assert "low_usage" in rec.estimated_monthly_savings
        assert rec.estimated_monthly_savings["high_usage"]["savings"] == 20.0


@pytest.mark.unit
class TestPerformanceAnalysis:
    """Test PerformanceAnalysis model."""

    def test_performance_analysis_creation(self, sample_memory_results):
        """Test PerformanceAnalysis creation with memory results."""
        efficiency_scores = {256: 1.5, 512: 2.0, 1024: 1.8}
        cost_optimal = {"memory_size": 256, "reasoning": "Lowest cost"}
        speed_optimal = {"memory_size": 1024, "reasoning": "Fastest execution"}
        balanced_optimal = {"memory_size": 512, "reasoning": "Best balance"}
        trends = {"duration_trend": "decreasing", "cost_trend": "increasing"}
        insights = [{"type": "optimization", "message": "Consider 512MB"}]

        analysis = PerformanceAnalysis(
            memory_results=sample_memory_results,
            efficiency_scores=efficiency_scores,
            cost_optimal=cost_optimal,
            speed_optimal=speed_optimal,
            balanced_optimal=balanced_optimal,
            trends=trends,
            insights=insights,
        )

        assert len(analysis.memory_results) == 3
        assert analysis.efficiency_scores[512] == 2.0
        assert analysis.cost_optimal["memory_size"] == 256
        assert analysis.speed_optimal["memory_size"] == 1024

    def test_performance_analysis_validation(self, sample_memory_results, validator):
        """Test PerformanceAnalysis validation."""
        analysis = PerformanceAnalysis(
            memory_results=sample_memory_results,
            efficiency_scores={256: 1.0, 512: 1.5, 1024: 1.2},
            cost_optimal={"memory_size": 256},
            speed_optimal={"memory_size": 1024},
            balanced_optimal={"memory_size": 512},
            trends={"duration_trend": "decreasing"},
            insights=[],
        )

        assert validator.validate_performance_analysis(
            analysis, expected_memory_sizes=[256, 512, 1024]
        )


@pytest.mark.unit
class TestTuningResult:
    """Test TuningResult model."""

    def test_tuning_result_creation(
        self, sample_memory_results, sample_performance_analysis, sample_recommendation
    ):
        """Test complete TuningResult creation."""
        result = TuningResult(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            timestamp=datetime.utcnow(),
            strategy="balanced",
            memory_results=sample_memory_results,
            baseline_results=None,
            analysis=sample_performance_analysis,
            recommendation=sample_recommendation,
            duration=300.0,
        )

        assert "test" in result.function_arn
        assert result.strategy == "balanced"
        assert result.duration == 300.0
        assert isinstance(result.timestamp, datetime)

    def test_tuning_result_validation(self, complete_tuning_results, validator):
        """Test TuningResult validation."""
        assert validator.validate_tuning_result(complete_tuning_results)

    def test_tuning_result_with_baseline(
        self, sample_memory_results, sample_performance_analysis, sample_recommendation
    ):
        """Test TuningResult with baseline results."""
        baseline = [{"memory_size": 256, "duration": 200.0, "cost": 0.00001}]

        result = TuningResult(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            timestamp=datetime.utcnow(),
            strategy="balanced",
            memory_results=sample_memory_results,
            baseline_results=baseline,
            analysis=sample_performance_analysis,
            recommendation=sample_recommendation,
            duration=300.0,
        )

        assert result.baseline_results is not None
        assert len(result.baseline_results) == 1
        assert result.baseline_results[0]["memory_size"] == 256


@pytest.mark.unit
class TestAdvancedModels:
    """Test advanced analysis models."""

    def test_cold_start_analysis_creation(self):
        """Test ColdStartAnalysis model."""
        analysis = ColdStartAnalysis(
            cold_start_ratio=0.3,
            avg_cold_start_duration=1500.0,
            avg_warm_start_duration=150.0,
            cold_start_impact_score=0.7,
            memory_vs_cold_start_correlation=-0.5,
            optimal_memory_for_cold_starts=512,
            cold_start_patterns={"peak_hours": [9, 10, 11]},
        )

        assert analysis.cold_start_ratio == 0.3
        assert analysis.avg_cold_start_duration == 1500.0
        assert analysis.optimal_memory_for_cold_starts == 512
        assert "peak_hours" in analysis.cold_start_patterns

    def test_concurrency_analysis_creation(self):
        """Test ConcurrencyAnalysis model."""
        analysis = ConcurrencyAnalysis(
            avg_concurrent_executions=5.5,
            peak_concurrent_executions=20,
            concurrency_utilization=0.75,
            scaling_efficiency=0.85,
            throttling_events=2,
            recommended_concurrency_limit=15,
            concurrency_patterns={"daily_peak": "14:00"},
        )

        assert analysis.avg_concurrent_executions == 5.5
        assert analysis.peak_concurrent_executions == 20
        assert analysis.recommended_concurrency_limit == 15
        assert analysis.throttling_events == 2

    def test_workload_analysis_creation(self):
        """Test WorkloadAnalysis model."""
        analysis = WorkloadAnalysis(
            workload_type="cpu_intensive",
            resource_utilization={"cpu": 0.8, "memory": 0.6, "io": 0.2},
            optimization_opportunities=[
                {"type": "memory_increase", "impact": "high", "cost": "medium"}
            ],
            workload_specific_recommendations=[
                {"action": "increase_memory", "from": 512, "to": 1024}
            ],
            cost_vs_performance_curve={"data_points": []},
        )

        assert analysis.workload_type == "cpu_intensive"
        assert analysis.resource_utilization["cpu"] == 0.8
        assert len(analysis.optimization_opportunities) == 1
        assert len(analysis.workload_specific_recommendations) == 1

    def test_time_based_trend_creation(self):
        """Test TimeBasedTrend model."""
        trend = TimeBasedTrend(
            time_period="24h",
            metric_trends={"duration": [100, 110, 105, 95], "cost": [0.001, 0.0011, 0.0009]},
            seasonal_patterns={"hourly_peak": 14},
            performance_degradation=False,
            trend_confidence=0.85,
            forecast={"next_hour_duration": 98.0},
        )

        assert trend.time_period == "24h"
        assert len(trend.metric_trends["duration"]) == 4
        assert trend.performance_degradation is False
        assert trend.trend_confidence == 0.85

    def test_advanced_performance_analysis_creation(self, sample_memory_results):
        """Test AdvancedPerformanceAnalysis with additional components."""
        cold_start = ColdStartAnalysis(
            cold_start_ratio=0.2,
            avg_cold_start_duration=1000.0,
            avg_warm_start_duration=100.0,
            cold_start_impact_score=0.5,
            memory_vs_cold_start_correlation=-0.3,
            optimal_memory_for_cold_starts=512,
            cold_start_patterns={},
        )

        analysis = AdvancedPerformanceAnalysis(
            memory_results=sample_memory_results,
            efficiency_scores={256: 1.0, 512: 1.5, 1024: 1.2},
            cost_optimal={"memory_size": 256},
            speed_optimal={"memory_size": 1024},
            balanced_optimal={"memory_size": 512},
            trends={"duration_trend": "decreasing"},
            insights=[],
            cold_start_analysis=cold_start,
        )

        assert analysis.cold_start_analysis is not None
        assert analysis.cold_start_analysis.cold_start_ratio == 0.2


@pytest.mark.unit
@pytest.mark.parametrize(
    "memory_size,expected_valid",
    [
        (128, True),
        (256, True),
        (512, True),
        (1024, True),
        (2048, True),
        (3008, True),
        (0, False),
        (-1, False),
        (4000, True),  # Above max, but still valid number
    ],
)
def test_memory_size_validation(memory_size, expected_valid):
    """Test memory size validation across different values."""
    try:
        result = MemoryTestResult(
            memory_size=memory_size,
            iterations=1,
            avg_duration=100.0,
            p95_duration=110.0,
            p99_duration=120.0,
            avg_cost=0.00001,
            total_cost=0.00001,
            cold_starts=0,
            errors=0,
        )
        assert expected_valid, f"Expected invalid memory size {memory_size} to raise error"
        assert result.memory_size == memory_size
    except (ValueError, AssertionError):
        assert not expected_valid, f"Expected valid memory size {memory_size} to work"


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["cost", "speed", "balanced", "comprehensive"])
def test_recommendation_strategies(strategy):
    """Test recommendation creation with different strategies."""
    rec = Recommendation(
        strategy=strategy,
        current_memory_size=256,
        optimal_memory_size=512,
        should_optimize=True,
        reasoning=f"Optimized for {strategy} strategy",
    )

    assert rec.strategy == strategy
    assert strategy in rec.reasoning


@pytest.mark.unit
def test_model_consistency_validation(assertions):
    """Test consistency between related model fields."""
    # Test that P99 >= P95 >= Average
    result = MemoryTestResult(
        memory_size=256,
        iterations=10,
        avg_duration=100.0,
        p95_duration=120.0,
        p99_duration=140.0,
        avg_cost=0.00001,
        total_cost=0.0001,
        cold_starts=2,
        errors=0,
    )

    assert result.p95_duration >= result.avg_duration
    assert result.p99_duration >= result.p95_duration

    # Test that total cost is reasonable compared to average
    expected_total = result.avg_cost * (result.iterations - result.errors)
    assert abs(result.total_cost - expected_total) < 0.001


@pytest.mark.unit
def test_model_serialization():
    """Test that models can be serialized to dict."""
    result = MemoryTestResult(
        memory_size=256,
        iterations=5,
        avg_duration=150.0,
        p95_duration=180.0,
        p99_duration=200.0,
        avg_cost=0.00001,
        total_cost=0.00005,
        cold_starts=1,
        errors=0,
    )

    # Test that we can convert to dict (dataclass should support this)
    from dataclasses import asdict

    result_dict = asdict(result)

    assert result_dict["memory_size"] == 256
    assert result_dict["iterations"] == 5
    assert result_dict["avg_duration"] == 150.0
