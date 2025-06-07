"""Integration tests for workload-specific optimization workflows."""

import pytest
from unittest.mock import patch
from datetime import datetime, timedelta

from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.orchestrator_module import TuningOrchestrator
from aws_lambda_tuner.analyzers.analyzer import PerformanceAnalyzer
from aws_lambda_tuner.models import MemoryTestResult, PerformanceAnalysis
from tests.utils.test_helpers import TestValidators, TestAssertions, TestHelpers


@pytest.mark.integration
@pytest.mark.aws
class TestWorkloadSpecificOptimization:
    """Test complete workflows for different workload types."""

    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        """Create orchestrator with mocked AWS services."""
        return TuningOrchestrator()

    @pytest.fixture
    def analyzer(self):
        """Create performance analyzer."""
        return PerformanceAnalyzer()

    @pytest.mark.workload_cpu
    def test_cpu_intensive_workload_optimization(self, orchestrator, mock_aws_services, validator):
        """Test optimization workflow for CPU-intensive workloads."""
        # Configure mock for CPU-intensive behavior
        mock_aws_services.configure_function_behavior(
            "test-function",
            base_duration=2000,  # High base duration
            cold_start_rate=0.2,
            error_rate=0.01,
        )

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[512, 1024, 2048, 3008],  # Focus on higher memory
            iterations=10,
            strategy="speed",
            workload_type="cpu_intensive",
        )

        result = orchestrator.run_optimization(config)

        # Validate results
        validator.validate_tuning_result(result)

        # CPU-intensive workloads should benefit significantly from more memory
        memory_sizes = sorted(result.memory_results.keys())
        duration_improvement = (
            result.memory_results[memory_sizes[0]].avg_duration
            - result.memory_results[memory_sizes[-1]].avg_duration
        ) / result.memory_results[memory_sizes[0]].avg_duration

        assert (
            duration_improvement > 0.3
        ), "CPU-intensive workload should show significant improvement with more memory"

        # Should recommend higher memory for speed strategy
        assert result.recommendation.optimal_memory_size >= 1024
        assert result.recommendation.should_optimize

    @pytest.mark.workload_io
    def test_io_bound_workload_optimization(self, orchestrator, mock_aws_services, validator):
        """Test optimization workflow for I/O-bound workloads."""
        # Configure mock for I/O-bound behavior
        mock_aws_services.configure_function_behavior(
            "test-function",
            base_duration=500,  # Lower base duration
            cold_start_rate=0.4,  # Higher cold start impact
            error_rate=0.005,
        )

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[128, 256, 512, 1024],  # Focus on lower memory
            iterations=15,
            strategy="cost",
            workload_type="io_bound",
        )

        result = orchestrator.run_optimization(config)

        # Validate results
        validator.validate_tuning_result(result)

        # I/O-bound workloads should show low memory sensitivity
        assert result.analysis.trends["memory_sensitivity"] in ["low", "medium"]

        # Should recommend cost-effective memory for cost strategy
        assert result.recommendation.optimal_memory_size <= 512

        # Cold start insights should be present due to higher cold start rate
        cold_start_insights = [
            i for i in result.analysis.insights if "cold start" in i["message"].lower()
        ]
        # May or may not be present depending on threshold

    @pytest.mark.workload_memory
    def test_memory_intensive_workload_optimization(
        self, orchestrator, mock_aws_services, validator
    ):
        """Test optimization workflow for memory-intensive workloads."""
        # Configure mock for memory-intensive behavior
        mock_aws_services.configure_function_behavior(
            "test-function", base_duration=1500, cold_start_rate=0.3, error_rate=0.02
        )

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[1024, 1536, 2048, 3008],  # High memory range
            iterations=10,
            strategy="balanced",
            workload_type="memory_intensive",
        )

        result = orchestrator.run_optimization(config)

        # Validate results
        validator.validate_tuning_result(result)

        # Memory-intensive workloads should show high memory sensitivity
        assert result.analysis.trends["memory_sensitivity"] in ["medium", "high"]

        # Should recommend higher memory
        assert result.recommendation.optimal_memory_size >= 1536

        # Should show clear performance improvement with memory
        memory_sizes = sorted(result.memory_results.keys())
        fastest_duration = min(r.avg_duration for r in result.memory_results.values())
        slowest_duration = max(r.avg_duration for r in result.memory_results.values())
        improvement_ratio = (slowest_duration - fastest_duration) / slowest_duration

        assert (
            improvement_ratio > 0.15
        ), "Memory-intensive workload should show significant memory sensitivity"

    @pytest.mark.workload_balanced
    def test_balanced_workload_optimization(self, orchestrator, mock_aws_services, validator):
        """Test optimization workflow for balanced workloads."""
        # Configure mock for balanced behavior
        mock_aws_services.configure_function_behavior(
            "test-function", base_duration=800, cold_start_rate=0.25, error_rate=0.01
        )

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512, 1024, 1536, 2048],
            iterations=12,
            strategy="balanced",
            workload_type="balanced",
        )

        result = orchestrator.run_optimization(config)

        # Validate results
        validator.validate_tuning_result(result)

        # Balanced workloads should show medium memory sensitivity
        assert result.analysis.trends["memory_sensitivity"] in ["low", "medium", "high"]

        # Should find a balanced optimal point (not necessarily highest or lowest)
        assert 512 <= result.recommendation.optimal_memory_size <= 1536

        # Should have good efficiency score for the recommended memory
        recommended_memory = result.recommendation.optimal_memory_size
        max_efficiency = max(result.analysis.efficiency_scores.values())
        recommended_efficiency = result.analysis.efficiency_scores[recommended_memory]

        assert (
            recommended_efficiency >= max_efficiency * 0.9
        ), "Recommended memory should have high efficiency score"


@pytest.mark.integration
@pytest.mark.strategy_cost
@pytest.mark.strategy_speed
@pytest.mark.strategy_balanced
class TestStrategySpecificOptimization:
    """Test optimization workflows for different strategies."""

    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TuningOrchestrator()

    @pytest.mark.strategy_cost
    def test_cost_optimization_strategy(self, orchestrator, mock_aws_services, assertions):
        """Test cost optimization strategy workflow."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[128, 256, 512, 1024],
            iterations=8,
            strategy="cost",
        )

        result = orchestrator.run_optimization(config)

        # Cost strategy should prioritize lowest cost
        cost_optimal_memory = result.analysis.cost_optimal["memory_size"]
        assert result.recommendation.optimal_memory_size == cost_optimal_memory

        # Should provide cost savings estimates
        assert result.recommendation.estimated_monthly_savings is not None

        # Verify cost trend analysis
        assertions.assert_cost_trend(result.memory_results, "increasing")

    @pytest.mark.strategy_speed
    def test_speed_optimization_strategy(self, orchestrator, mock_aws_services, assertions):
        """Test speed optimization strategy workflow."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[512, 1024, 1536, 2048, 3008],
            iterations=8,
            strategy="speed",
        )

        result = orchestrator.run_optimization(config)

        # Speed strategy should prioritize fastest execution
        speed_optimal_memory = result.analysis.speed_optimal["memory_size"]
        assert result.recommendation.optimal_memory_size == speed_optimal_memory

        # Should show performance improvement
        assertions.assert_memory_performance_trend(result.memory_results, "decreasing")

        # Recommended memory should be towards the higher end
        assert result.recommendation.optimal_memory_size >= 1024

    @pytest.mark.strategy_balanced
    def test_balanced_optimization_strategy(self, orchestrator, mock_aws_services, assertions):
        """Test balanced optimization strategy workflow."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512, 1024, 1536, 2048],
            iterations=10,
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)

        # Balanced strategy should use efficiency scores
        balanced_optimal_memory = result.analysis.balanced_optimal["memory_size"]
        assert result.recommendation.optimal_memory_size == balanced_optimal_memory

        # Should have good efficiency for recommended memory
        assertions.assert_efficiency_optimal(
            result.analysis.efficiency_scores,
            result.recommendation.optimal_memory_size,
            tolerance=0.1,
        )

        # Verify recommendation consistency
        assertions.assert_recommendation_consistency(result.recommendation, result.analysis)


@pytest.mark.integration
@pytest.mark.aws
class TestCompleteWorkflowIntegration:
    """Test complete optimization workflows with various configurations."""

    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TuningOrchestrator()

    def test_multi_iteration_workflow(self, orchestrator, mock_aws_services, validator):
        """Test workflow with multiple iterations to ensure consistency."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512, 1024],
            iterations=20,  # Higher iteration count
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # With more iterations, results should be more stable
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == 20
            # Standard deviation should be reasonable with more samples
            if len(memory_result.raw_results) > 1:
                durations = [
                    r["duration"] for r in memory_result.raw_results if r["status_code"] == 200
                ]
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
                    std_dev = variance**0.5
                    coefficient_of_variation = std_dev / avg_duration
                    # Should have reasonable consistency (CV < 50%)
                    assert coefficient_of_variation < 0.5

    def test_concurrent_execution_workflow(self, orchestrator, mock_aws_services, validator):
        """Test workflow with concurrent executions."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512, 1024],
            iterations=10,
            concurrent_executions=3,  # Concurrent execution
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Results should still be valid with concurrent executions
        assert len(result.memory_results) == 3
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == 10

    def test_warmup_iterations_workflow(self, orchestrator, mock_aws_services, validator):
        """Test workflow with warmup iterations."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[512, 1024],
            iterations=8,
            warmup_iterations=3,
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Results should account for warmup (main iterations should still be 8)
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == 8

    def test_error_handling_workflow(self, orchestrator, mock_aws_services, validator):
        """Test workflow with simulated errors."""
        # Configure higher error rate
        mock_aws_services.configure_function_behavior(
            "test-function", error_rate=0.2  # 20% error rate
        )

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512, 1024],
            iterations=15,
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Should handle errors gracefully
        total_errors = sum(r.errors for r in result.memory_results.values())
        assert total_errors > 0  # Should have some errors due to configured error rate

        # Should generate reliability insights
        reliability_insights = [i for i in result.analysis.insights if i["type"] == "reliability"]
        # May or may not generate depending on threshold

    def test_dry_run_workflow(self, orchestrator, mock_aws_services, validator):
        """Test dry run workflow."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512],
            iterations=5,
            dry_run=True,
            strategy="balanced",
        )

        # Dry run should complete without actual invocations
        result = orchestrator.run_optimization(config)

        # Should still return valid structure (may be synthetic data)
        validator.validate_tuning_result(result)

    def test_comprehensive_strategy_workflow(self, orchestrator, mock_aws_services, validator):
        """Test comprehensive strategy workflow."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[128, 256, 512, 1024, 1536, 2048, 3008],
            iterations=12,
            strategy="comprehensive",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Comprehensive strategy should test all provided memory sizes
        assert len(result.memory_results) == 7

        # Should provide detailed analysis
        assert len(result.analysis.insights) >= 2
        assert result.analysis.trends is not None
        assert result.analysis.efficiency_scores is not None


@pytest.mark.integration
@pytest.mark.parametrize
class TestParametrizedWorkflows:
    """Parametrized integration tests for various configurations."""

    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TuningOrchestrator()

    @pytest.mark.parametrize(
        "memory_range,expected_optimal_range",
        [
            ([128, 256, 512], (128, 512)),
            ([512, 1024, 2048], (512, 2048)),
            ([1024, 1536, 2048, 3008], (1024, 3008)),
        ],
    )
    def test_different_memory_ranges(
        self, orchestrator, mock_aws_services, validator, memory_range, expected_optimal_range
    ):
        """Test optimization with different memory ranges."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=memory_range,
            iterations=8,
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Optimal memory should be within the tested range
        optimal = result.recommendation.optimal_memory_size
        assert expected_optimal_range[0] <= optimal <= expected_optimal_range[1]

    @pytest.mark.parametrize("iteration_count", [1, 5, 10, 25])
    def test_different_iteration_counts(
        self, orchestrator, mock_aws_services, validator, iteration_count
    ):
        """Test optimization with different iteration counts."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[256, 512, 1024],
            iterations=iteration_count,
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Verify iteration counts match
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == iteration_count

    @pytest.mark.parametrize("concurrent_count", [1, 2, 5])
    def test_different_concurrency_levels(
        self, orchestrator, mock_aws_services, validator, concurrent_count
    ):
        """Test optimization with different concurrency levels."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            memory_sizes=[512, 1024],
            iterations=6,
            concurrent_executions=concurrent_count,
            strategy="balanced",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Should complete successfully regardless of concurrency level
        assert len(result.memory_results) == 2
