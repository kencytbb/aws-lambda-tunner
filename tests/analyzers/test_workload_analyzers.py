"""Tests for workload-specific analyzers."""

import pytest
from unittest.mock import Mock, patch
import statistics

from aws_lambda_tuner.analyzers.on_demand_analyzer import OnDemandAnalyzer
from aws_lambda_tuner.analyzers.continuous_analyzer import ContinuousAnalyzer
from aws_lambda_tuner.analyzers.scheduled_analyzer import ScheduledAnalyzer
from aws_lambda_tuner.models import MemoryTestResult, Recommendation


@pytest.fixture
def mock_config():
    """Mock configuration for analyzers."""
    return {"memory_sizes": [128, 256, 512, 1024], "iterations": 10, "timeout": 30}


@pytest.fixture
def on_demand_memory_results():
    """Sample memory results for on-demand workload (high cold start ratio)."""
    return {
        128: MemoryTestResult(
            memory_size=128,
            iterations=10,
            avg_duration=2000.0,
            p95_duration=2500.0,
            p99_duration=3000.0,
            avg_cost=0.0000002,
            total_cost=0.000002,
            cold_starts=6,  # High cold start ratio
            errors=1,
            raw_results=[
                {
                    "duration": 2500,
                    "cost": 0.0000002,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2800,
                    "cost": 0.0000002,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1600,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2600,
                    "cost": 0.0000002,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1700,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2700,
                    "cost": 0.0000002,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2500,
                    "cost": 0.0000002,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1600,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
            ],
        ),
        256: MemoryTestResult(
            memory_size=256,
            iterations=10,
            avg_duration=1800.0,
            p95_duration=2200.0,
            p99_duration=2500.0,
            avg_cost=0.0000003,
            total_cost=0.000003,
            cold_starts=4,  # Medium cold start ratio
            errors=0,
            raw_results=[
                {
                    "duration": 2200,
                    "cost": 0.0000003,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1400,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2300,
                    "cost": 0.0000003,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2100,
                    "cost": 0.0000003,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1600,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 2000,
                    "cost": 0.0000003,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1400,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1600,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
            ],
        ),
        512: MemoryTestResult(
            memory_size=512,
            iterations=10,
            avg_duration=1500.0,
            p95_duration=1800.0,
            p99_duration=2000.0,
            avg_cost=0.0000004,
            total_cost=0.000004,
            cold_starts=2,  # Low cold start ratio
            errors=0,
            raw_results=[
                {
                    "duration": 1800,
                    "cost": 0.0000004,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1300,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1900,
                    "cost": 0.0000004,
                    "is_cold_start": True,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1400,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1600,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1400,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1300,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
                {
                    "duration": 1500,
                    "cost": 0.0000004,
                    "is_cold_start": False,
                    "concurrent_executions": 1,
                },
            ],
        ),
    }


@pytest.fixture
def continuous_memory_results():
    """Sample memory results for continuous workload (high concurrency, sustained load)."""
    return {
        128: MemoryTestResult(
            memory_size=128,
            iterations=50,
            avg_duration=1200.0,
            p95_duration=1400.0,
            p99_duration=1600.0,
            avg_cost=0.0000002,
            total_cost=0.00001,
            cold_starts=2,  # Low cold starts for continuous
            errors=2,
            raw_results=[
                {
                    "duration": d,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": c,
                    "cpu_utilization": 0.8,
                    "memory_utilization": 0.9,
                }
                for d, c in zip(
                    [1200, 1150, 1250, 1180, 1220] * 10,  # Consistent durations
                    [10, 12, 15, 18, 20] * 10,  # High concurrency
                )
            ],
        ),
        256: MemoryTestResult(
            memory_size=256,
            iterations=50,
            avg_duration=1000.0,
            p95_duration=1150.0,
            p99_duration=1200.0,
            avg_cost=0.0000003,
            total_cost=0.000015,
            cold_starts=1,
            errors=1,
            raw_results=[
                {
                    "duration": d,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": c,
                    "cpu_utilization": 0.7,
                    "memory_utilization": 0.8,
                }
                for d, c in zip([1000, 980, 1020, 990, 1010] * 10, [12, 15, 18, 20, 22] * 10)
            ],
        ),
        512: MemoryTestResult(
            memory_size=512,
            iterations=50,
            avg_duration=800.0,
            p95_duration=900.0,
            p99_duration=950.0,
            avg_cost=0.0000005,
            total_cost=0.000025,
            cold_starts=0,
            errors=0,
            raw_results=[
                {
                    "duration": d,
                    "cost": 0.0000005,
                    "is_cold_start": False,
                    "concurrent_executions": c,
                    "cpu_utilization": 0.6,
                    "memory_utilization": 0.7,
                }
                for d, c in zip([800, 790, 810, 795, 805] * 10, [15, 18, 20, 22, 25] * 10)
            ],
        ),
    }


@pytest.fixture
def scheduled_memory_results():
    """Sample memory results for scheduled workload (predictable patterns)."""
    return {
        128: MemoryTestResult(
            memory_size=128,
            iterations=20,
            avg_duration=1500.0,
            p95_duration=1600.0,
            p99_duration=1700.0,
            avg_cost=0.0000002,
            total_cost=0.000004,
            cold_starts=1,  # Low cold starts for scheduled
            errors=0,
            raw_results=[
                {
                    "duration": 1500 + (i % 3) * 10,
                    "cost": 0.0000002,
                    "is_cold_start": False,
                    "concurrent_executions": 2 + (i % 2),
                    "timestamp": 1000 + i * 100,
                }
                for i in range(20)  # Very predictable pattern
            ],
        ),
        256: MemoryTestResult(
            memory_size=256,
            iterations=20,
            avg_duration=1200.0,
            p95_duration=1250.0,
            p99_duration=1300.0,
            avg_cost=0.0000003,
            total_cost=0.000006,
            cold_starts=1,
            errors=0,
            raw_results=[
                {
                    "duration": 1200 + (i % 2) * 5,
                    "cost": 0.0000003,
                    "is_cold_start": False,
                    "concurrent_executions": 3 + (i % 2),
                    "timestamp": 1000 + i * 100,
                }
                for i in range(20)
            ],
        ),
        512: MemoryTestResult(
            memory_size=512,
            iterations=20,
            avg_duration=1000.0,
            p95_duration=1050.0,
            p99_duration=1100.0,
            avg_cost=0.0000005,
            total_cost=0.00001,
            cold_starts=0,
            errors=0,
            raw_results=[
                {
                    "duration": 1000 + (i % 2) * 2,
                    "cost": 0.0000005,
                    "is_cold_start": False,
                    "concurrent_executions": 4 + (i % 2),
                    "timestamp": 1000 + i * 100,
                }
                for i in range(20)
            ],
        ),
    }


class TestOnDemandAnalyzer:
    """Test cases for OnDemandAnalyzer."""

    def test_initialization(self, mock_config):
        """Test OnDemandAnalyzer initialization."""
        analyzer = OnDemandAnalyzer(mock_config)

        assert analyzer.workload_type == "on_demand"
        assert analyzer.cold_start_threshold == 0.1
        assert analyzer.latency_sensitivity == "high"
        assert analyzer.cost_tolerance == "medium"

    def test_analyze_for_minimal_cold_starts(self, mock_config, on_demand_memory_results):
        """Test analysis for minimal cold starts."""
        analyzer = OnDemandAnalyzer(mock_config)

        analysis = analyzer.analyze_for_minimal_cold_starts(on_demand_memory_results)

        assert isinstance(analysis, dict)
        assert "optimal_configuration" in analysis
        assert "provisioned_concurrency_recommendation" in analysis
        assert "latency_impact_analysis" in analysis
        assert "cold_start_metrics" in analysis

        # Optimal configuration should have lower cold starts
        optimal_memory = analysis["optimal_configuration"]["memory_size"]
        optimal_result = on_demand_memory_results[optimal_memory]
        optimal_cold_start_ratio = optimal_result.cold_starts / optimal_result.iterations

        # Should be better than the worst case
        worst_memory = max(
            on_demand_memory_results.keys(),
            key=lambda k: on_demand_memory_results[k].cold_starts
            / on_demand_memory_results[k].iterations,
        )
        worst_result = on_demand_memory_results[worst_memory]
        worst_cold_start_ratio = worst_result.cold_starts / worst_result.iterations

        assert optimal_cold_start_ratio <= worst_cold_start_ratio

    def test_provisioned_concurrency_recommendation(self, mock_config, on_demand_memory_results):
        """Test provisioned concurrency analysis."""
        analyzer = OnDemandAnalyzer(mock_config)

        cold_start_analysis = analyzer.analyze_cold_starts(on_demand_memory_results)
        recommendation = analyzer._analyze_provisioned_concurrency_benefits(
            on_demand_memory_results, cold_start_analysis
        )

        assert isinstance(recommendation, dict)
        assert "recommended" in recommendation
        assert "reason" in recommendation
        assert "cold_start_penalty_ms" in recommendation
        assert "potential_latency_reduction" in recommendation
        assert "provisioned_concurrency_considerations" in recommendation

        # Should recommend provisioned concurrency for high cold start ratio
        if cold_start_analysis.cold_start_ratio > analyzer.cold_start_threshold:
            assert recommendation["recommended"] is True

    def test_get_on_demand_recommendation(self, mock_config, on_demand_memory_results):
        """Test getting on-demand specific recommendation."""
        analyzer = OnDemandAnalyzer(mock_config)

        recommendation = analyzer.get_on_demand_recommendation(on_demand_memory_results)

        assert isinstance(recommendation, Recommendation)
        assert recommendation.strategy == "on_demand_optimized"
        assert recommendation.current_memory_size in on_demand_memory_results
        assert recommendation.optimal_memory_size in on_demand_memory_results
        assert isinstance(recommendation.should_optimize, bool)
        assert isinstance(recommendation.reasoning, str)
        assert 0 <= recommendation.confidence_score <= 1

    def test_cold_start_optimal_configuration(self, mock_config, on_demand_memory_results):
        """Test finding cold start optimal configuration."""
        analyzer = OnDemandAnalyzer(mock_config)
        cold_start_analysis = analyzer.analyze_cold_starts(on_demand_memory_results)

        optimal_config = analyzer._find_cold_start_optimal_configuration(
            on_demand_memory_results, cold_start_analysis
        )

        assert isinstance(optimal_config, dict)
        assert "memory_size" in optimal_config
        assert "cold_start_ratio" in optimal_config
        assert "avg_duration" in optimal_config
        assert "avg_cost" in optimal_config
        assert "score" in optimal_config
        assert "reasoning" in optimal_config

        # Should select a configuration with reasonable cold start ratio
        assert 0 <= optimal_config["cold_start_ratio"] <= 1

    def test_user_experience_impact_rating(self, mock_config):
        """Test user experience impact rating."""
        analyzer = OnDemandAnalyzer(mock_config)

        # Test different impact levels
        minimal_impact = analyzer._rate_user_experience_impact(0.02, 100)  # 2% ratio, 100ms penalty
        moderate_impact = analyzer._rate_user_experience_impact(
            0.1, 200
        )  # 10% ratio, 200ms penalty
        significant_impact = analyzer._rate_user_experience_impact(
            0.2, 500
        )  # 20% ratio, 500ms penalty
        severe_impact = analyzer._rate_user_experience_impact(
            0.5, 1000
        )  # 50% ratio, 1000ms penalty

        assert minimal_impact in ["minimal", "moderate", "significant", "severe"]
        assert severe_impact in ["significant", "severe"]  # Should be high impact

    def test_should_optimize_for_on_demand(self, mock_config):
        """Test optimization decision logic for on-demand workloads."""
        analyzer = OnDemandAnalyzer(mock_config)

        # Significant cold start improvement with acceptable costs
        should_optimize = analyzer._should_optimize_for_on_demand(
            current_memory=128,
            optimal_memory=512,
            cost_change=30.0,  # 30% cost increase (acceptable)
            duration_change=10.0,  # 10% performance improvement
            cold_start_improvement=25.0,  # 25% cold start improvement
        )

        assert should_optimize is True

        # Minimal cold start improvement
        should_not_optimize = analyzer._should_optimize_for_on_demand(
            current_memory=128,
            optimal_memory=256,
            cost_change=20.0,
            duration_change=5.0,
            cold_start_improvement=5.0,  # Only 5% improvement
        )

        assert should_not_optimize is False


class TestContinuousAnalyzer:
    """Test cases for ContinuousAnalyzer."""

    def test_initialization(self, mock_config):
        """Test ContinuousAnalyzer initialization."""
        analyzer = ContinuousAnalyzer(mock_config)

        assert analyzer.workload_type == "continuous"
        assert analyzer.cost_sensitivity == "high"
        assert analyzer.throughput_priority == "high"
        assert analyzer.latency_tolerance == "medium"

    def test_analyze_for_sustained_throughput(self, mock_config, continuous_memory_results):
        """Test analysis for sustained throughput."""
        analyzer = ContinuousAnalyzer(mock_config)

        analysis = analyzer.analyze_for_sustained_throughput(continuous_memory_results)

        assert isinstance(analysis, dict)
        assert "throughput_optimal_configuration" in analysis
        assert "cost_efficiency_analysis" in analysis
        assert "sustained_performance_metrics" in analysis
        assert "resource_utilization_efficiency" in analysis
        assert "scaling_recommendations" in analysis

    def test_throughput_optimal_configuration(self, mock_config, continuous_memory_results):
        """Test finding throughput optimal configuration."""
        analyzer = ContinuousAnalyzer(mock_config)
        concurrency_analysis = analyzer.analyze_concurrency_patterns(continuous_memory_results)

        optimal_config = analyzer._find_throughput_optimal_configuration(
            continuous_memory_results, concurrency_analysis
        )

        assert isinstance(optimal_config, dict)
        assert "memory_size" in optimal_config
        assert "throughput_efficiency_score" in optimal_config
        assert "estimated_throughput" in optimal_config
        assert "cost_per_request" in optimal_config
        assert "error_rate" in optimal_config
        assert "reasoning" in optimal_config

        # Should have positive throughput efficiency
        assert optimal_config["throughput_efficiency_score"] > 0
        assert optimal_config["estimated_throughput"] > 0

    def test_cost_efficiency_analysis(self, mock_config, continuous_memory_results):
        """Test cost efficiency analysis for continuous workloads."""
        analyzer = ContinuousAnalyzer(mock_config)

        cost_analysis = analyzer._analyze_cost_efficiency_for_continuous(continuous_memory_results)

        assert isinstance(cost_analysis, dict)
        assert "most_cost_efficient_memory" in cost_analysis
        assert "cost_efficiency_metrics" in cost_analysis
        assert "monthly_cost_projections" in cost_analysis
        assert "cost_optimization_recommendations" in cost_analysis

        # Cost efficient memory should be in the test data
        assert cost_analysis["most_cost_efficient_memory"] in continuous_memory_results

    def test_sustained_performance_metrics(self, mock_config, continuous_memory_results):
        """Test sustained performance metrics calculation."""
        analyzer = ContinuousAnalyzer(mock_config)

        sustained_metrics = analyzer._calculate_sustained_performance_metrics(
            continuous_memory_results
        )

        assert isinstance(sustained_metrics, dict)

        for memory_size, metrics in sustained_metrics.items():
            assert memory_size in continuous_memory_results
            assert "performance_consistency" in metrics
            assert "sustained_throughput_capacity" in metrics
            assert "performance_degradation_risk" in metrics

            assert 0 <= metrics["performance_consistency"] <= 1
            assert metrics["sustained_throughput_capacity"] >= 0
            assert metrics["performance_degradation_risk"] in ["low", "medium", "high"]

    def test_get_continuous_recommendation(self, mock_config, continuous_memory_results):
        """Test getting continuous workload recommendation."""
        analyzer = ContinuousAnalyzer(mock_config)

        recommendation = analyzer.get_continuous_recommendation(continuous_memory_results)

        assert isinstance(recommendation, Recommendation)
        assert recommendation.strategy == "continuous_optimized"
        assert recommendation.current_memory_size in continuous_memory_results
        assert recommendation.optimal_memory_size in continuous_memory_results
        assert isinstance(recommendation.should_optimize, bool)
        assert isinstance(recommendation.reasoning, str)
        assert 0 <= recommendation.confidence_score <= 1

    def test_should_optimize_for_continuous(self, mock_config):
        """Test optimization decision logic for continuous workloads."""
        analyzer = ContinuousAnalyzer(mock_config)

        # Significant cost savings with acceptable performance
        should_optimize = analyzer._should_optimize_for_continuous(
            current_memory=512,
            optimal_memory=256,
            cost_change=-15.0,  # 15% cost reduction
            duration_change=-10.0,  # 10% performance degradation (acceptable)
            throughput_improvement=5.0,
        )

        assert should_optimize is True

        # Significant throughput improvement
        should_optimize_throughput = analyzer._should_optimize_for_continuous(
            current_memory=128,
            optimal_memory=512,
            cost_change=20.0,
            duration_change=10.0,
            throughput_improvement=20.0,  # 20% throughput improvement
        )

        assert should_optimize_throughput is True

    def test_resource_utilization_efficiency(self, mock_config, continuous_memory_results):
        """Test resource utilization efficiency analysis."""
        analyzer = ContinuousAnalyzer(mock_config)

        utilization_analysis = analyzer._analyze_resource_utilization_efficiency(
            continuous_memory_results
        )

        assert isinstance(utilization_analysis, dict)

        for memory_size, analysis in utilization_analysis.items():
            assert memory_size in continuous_memory_results
            assert "memory_utilization" in analysis
            assert "cpu_utilization" in analysis
            assert "resource_efficiency_score" in analysis
            assert "over_provisioning_risk" in analysis

            assert 0 <= analysis["resource_efficiency_score"] <= 1
            assert analysis["over_provisioning_risk"] in ["low", "medium", "high"]


class TestScheduledAnalyzer:
    """Test cases for ScheduledAnalyzer."""

    def test_initialization(self, mock_config):
        """Test ScheduledAnalyzer initialization."""
        analyzer = ScheduledAnalyzer(mock_config)

        assert analyzer.workload_type == "scheduled"
        assert analyzer.cost_weight == 0.6
        assert analyzer.performance_weight == 0.4
        assert analyzer.predictability_factor == 0.8
        assert analyzer.optimization_frequency == "weekly"

    def test_analyze_for_cost_performance_balance(self, mock_config, scheduled_memory_results):
        """Test analysis for cost-performance balance."""
        analyzer = ScheduledAnalyzer(mock_config)

        analysis = analyzer.analyze_for_cost_performance_balance(scheduled_memory_results)

        assert isinstance(analysis, dict)
        assert "sweet_spot_configuration" in analysis
        assert "balanced_efficiency_scores" in analysis
        assert "predictability_analysis" in analysis
        assert "optimization_stability" in analysis
        assert "resource_allocation_recommendations" in analysis
        assert "cost_performance_trade_offs" in analysis

    def test_balanced_efficiency_scores(self, mock_config, scheduled_memory_results):
        """Test balanced efficiency score calculation."""
        analyzer = ScheduledAnalyzer(mock_config)

        scores = analyzer._calculate_balanced_efficiency_scores(scheduled_memory_results)

        assert isinstance(scores, dict)
        assert len(scores) == len(scheduled_memory_results)

        for memory_size, score in scores.items():
            assert memory_size in scheduled_memory_results
            assert 0 <= score <= 1  # Efficiency scores should be normalized

    def test_cost_performance_sweet_spot(self, mock_config, scheduled_memory_results):
        """Test finding cost-performance sweet spot."""
        analyzer = ScheduledAnalyzer(mock_config)

        balanced_efficiency = analyzer._calculate_balanced_efficiency_scores(
            scheduled_memory_results
        )
        sweet_spot = analyzer._find_cost_performance_sweet_spot(
            scheduled_memory_results, balanced_efficiency
        )

        assert isinstance(sweet_spot, dict)
        assert "memory_size" in sweet_spot
        assert "balanced_efficiency_score" in sweet_spot
        assert "avg_cost" in sweet_spot
        assert "avg_duration" in sweet_spot
        assert "cost_trade_off_vs_cheapest" in sweet_spot
        assert "speed_trade_off_vs_fastest" in sweet_spot
        assert "reasoning" in sweet_spot
        assert "is_cost_optimal" in sweet_spot
        assert "is_speed_optimal" in sweet_spot

        # Sweet spot memory should be in the results
        assert sweet_spot["memory_size"] in scheduled_memory_results

    def test_predictable_patterns_analysis(self, mock_config, scheduled_memory_results):
        """Test predictable pattern analysis."""
        analyzer = ScheduledAnalyzer(mock_config)

        predictability_analysis = analyzer._analyze_predictable_patterns(scheduled_memory_results)

        assert isinstance(predictability_analysis, dict)
        assert "predictability_by_memory" in predictability_analysis
        assert "most_predictable_config" in predictability_analysis
        assert "predictability_recommendations" in predictability_analysis

        # Check predictability metrics
        for memory_size, metrics in predictability_analysis["predictability_by_memory"].items():
            assert memory_size in scheduled_memory_results
            assert "duration_predictability" in metrics
            assert "cost_predictability" in metrics
            assert "overall_predictability" in metrics
            assert "consistency_rating" in metrics

            assert 0 <= metrics["overall_predictability"] <= 1

    def test_optimization_stability(self, mock_config, scheduled_memory_results):
        """Test optimization stability calculation."""
        analyzer = ScheduledAnalyzer(mock_config)

        stability = analyzer._calculate_optimization_stability(scheduled_memory_results)

        assert isinstance(stability, dict)
        assert "stability_margin" in stability
        assert "configuration_sensitivity" in stability
        assert "top_configurations" in stability
        assert "optimization_confidence" in stability
        assert "recommendation" in stability

        assert 0 <= stability["optimization_confidence"] <= 1
        assert isinstance(stability["top_configurations"], list)

    def test_get_scheduled_recommendation(self, mock_config, scheduled_memory_results):
        """Test getting scheduled workload recommendation."""
        analyzer = ScheduledAnalyzer(mock_config)

        recommendation = analyzer.get_scheduled_recommendation(scheduled_memory_results)

        assert isinstance(recommendation, Recommendation)
        assert recommendation.strategy == "scheduled_balanced"
        assert recommendation.current_memory_size in scheduled_memory_results
        assert recommendation.optimal_memory_size in scheduled_memory_results
        assert isinstance(recommendation.should_optimize, bool)
        assert isinstance(recommendation.reasoning, str)
        assert 0 <= recommendation.confidence_score <= 1

    def test_consistency_rating(self, mock_config):
        """Test consistency rating logic."""
        analyzer = ScheduledAnalyzer(mock_config)

        very_high = analyzer._rate_consistency(0.95)
        high = analyzer._rate_consistency(0.85)
        medium = analyzer._rate_consistency(0.75)
        low = analyzer._rate_consistency(0.60)
        very_low = analyzer._rate_consistency(0.30)

        assert very_high == "very_high"
        assert high == "high"
        assert medium == "medium"
        assert low == "low"
        assert very_low == "very_low"

    def test_pareto_efficient_configurations(self, mock_config, scheduled_memory_results):
        """Test finding Pareto efficient configurations."""
        analyzer = ScheduledAnalyzer(mock_config)

        # Create trade-off data
        trade_offs = {}
        for memory_size, result in scheduled_memory_results.items():
            trade_offs[memory_size] = {"cost": result.avg_cost, "duration": result.avg_duration}

        pareto_efficient = analyzer._find_pareto_efficient_configurations(trade_offs)

        assert isinstance(pareto_efficient, list)
        assert len(pareto_efficient) > 0

        # All efficient configurations should be in the original data
        for memory_size in pareto_efficient:
            assert memory_size in scheduled_memory_results

    def test_should_optimize_for_scheduled(self, mock_config):
        """Test optimization decision logic for scheduled workloads."""
        analyzer = ScheduledAnalyzer(mock_config)

        # Significant efficiency improvement with acceptable impacts
        should_optimize = analyzer._should_optimize_for_scheduled(
            current_memory=128,
            optimal_memory=256,
            cost_change=15.0,  # 15% cost increase (acceptable)
            duration_change=10.0,  # 10% performance improvement
            efficiency_improvement=8.0,  # 8% efficiency improvement
        )

        assert should_optimize is True

        # Already optimal
        should_not_optimize = analyzer._should_optimize_for_scheduled(
            current_memory=256,
            optimal_memory=256,
            cost_change=0.0,
            duration_change=0.0,
            efficiency_improvement=0.0,
        )

        assert should_not_optimize is False

    def test_resource_allocation_recommendations(self, mock_config, scheduled_memory_results):
        """Test resource allocation recommendations."""
        analyzer = ScheduledAnalyzer(mock_config)

        balanced_efficiency = analyzer._calculate_balanced_efficiency_scores(
            scheduled_memory_results
        )
        sweet_spot = analyzer._find_cost_performance_sweet_spot(
            scheduled_memory_results, balanced_efficiency
        )

        recommendations = analyzer._generate_resource_allocation_recommendations(
            scheduled_memory_results, sweet_spot
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        for rec in recommendations:
            assert "type" in rec
            assert "priority" in rec
            assert "current_recommendation" in rec
            assert "rationale" in rec
            assert "implementation" in rec

            assert rec["priority"] in ["high", "medium", "low"]
            assert rec["type"] in [
                "memory_allocation",
                "concurrency_allocation",
                "cost_consideration",
                "monitoring",
            ]


class TestWorkloadAnalyzerIntegration:
    """Integration tests for workload analyzers."""

    def test_analyzer_factory_pattern(self, mock_config):
        """Test that all analyzers can be instantiated and have required methods."""
        analyzers = [
            OnDemandAnalyzer(mock_config),
            ContinuousAnalyzer(mock_config),
            ScheduledAnalyzer(mock_config),
        ]

        for analyzer in analyzers:
            # All should inherit from PerformanceAnalyzer
            assert hasattr(analyzer, "analyze")
            assert hasattr(analyzer, "get_recommendation")
            assert hasattr(analyzer, "workload_type")

            # Should have workload-specific recommendation methods
            if isinstance(analyzer, OnDemandAnalyzer):
                assert hasattr(analyzer, "get_on_demand_recommendation")
                assert hasattr(analyzer, "analyze_for_minimal_cold_starts")
            elif isinstance(analyzer, ContinuousAnalyzer):
                assert hasattr(analyzer, "get_continuous_recommendation")
                assert hasattr(analyzer, "analyze_for_sustained_throughput")
            elif isinstance(analyzer, ScheduledAnalyzer):
                assert hasattr(analyzer, "get_scheduled_recommendation")
                assert hasattr(analyzer, "analyze_for_cost_performance_balance")

    def test_workload_specific_optimizations(
        self,
        mock_config,
        on_demand_memory_results,
        continuous_memory_results,
        scheduled_memory_results,
    ):
        """Test that different analyzers optimize for different objectives."""
        on_demand_analyzer = OnDemandAnalyzer(mock_config)
        continuous_analyzer = ContinuousAnalyzer(mock_config)
        scheduled_analyzer = ScheduledAnalyzer(mock_config)

        # Get recommendations from each analyzer
        on_demand_rec = on_demand_analyzer.get_on_demand_recommendation(on_demand_memory_results)
        continuous_rec = continuous_analyzer.get_continuous_recommendation(
            continuous_memory_results
        )
        scheduled_rec = scheduled_analyzer.get_scheduled_recommendation(scheduled_memory_results)

        # Verify different strategies
        assert on_demand_rec.strategy == "on_demand_optimized"
        assert continuous_rec.strategy == "continuous_optimized"
        assert scheduled_rec.strategy == "scheduled_balanced"

        # Verify reasoning reflects different priorities
        assert (
            "cold start" in on_demand_rec.reasoning.lower()
            or "latency" in on_demand_rec.reasoning.lower()
        )
        assert (
            "throughput" in continuous_rec.reasoning.lower()
            or "cost" in continuous_rec.reasoning.lower()
        )
        assert (
            "balance" in scheduled_rec.reasoning.lower()
            or "efficiency" in scheduled_rec.reasoning.lower()
        )
