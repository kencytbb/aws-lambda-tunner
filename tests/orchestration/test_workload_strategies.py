"""Integration tests for workload strategies."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from aws_lambda_tuner.strategies.workload_strategy import (
    WorkloadStrategy,
    WorkloadType,
    WorkloadCharacteristics,
    TestingStrategy,
    OptimizationResult,
)
from aws_lambda_tuner.strategies.on_demand_strategy import OnDemandStrategy
from aws_lambda_tuner.strategies.continuous_strategy import ContinuousStrategy
from aws_lambda_tuner.strategies.scheduled_strategy import ScheduledStrategy


class TestWorkloadStrategies:
    """Test workload strategy implementations."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock()
        config.function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
        config.region = "us-east-1"
        config.payload = {"test": "data"}
        config.dry_run = True
        config.warmup_runs = 2
        config.concurrent_executions = 5
        config.iterations = 10
        config.cost_per_gb_second = 0.0000166667
        config.cost_per_request = 0.0000002
        return config

    @pytest.fixture
    def aws_provider(self):
        """Create mock AWS provider."""
        provider = Mock()
        provider.get_function_configuration = AsyncMock()
        provider.update_function_memory = AsyncMock()
        provider.invoke_function = AsyncMock()
        provider.set_provisioned_concurrency = AsyncMock()
        provider.delete_provisioned_concurrency = AsyncMock()
        return provider

    @pytest.fixture
    def on_demand_strategy(self, config, aws_provider):
        """Create OnDemandStrategy instance."""
        return OnDemandStrategy(config, aws_provider)

    @pytest.fixture
    def continuous_strategy(self, config, aws_provider):
        """Create ContinuousStrategy instance."""
        return ContinuousStrategy(config, aws_provider)

    @pytest.fixture
    def scheduled_strategy(self, config, aws_provider):
        """Create ScheduledStrategy instance."""
        return ScheduledStrategy(config, aws_provider)


class TestOnDemandStrategy(TestWorkloadStrategies):
    """Test OnDemandStrategy implementation."""

    @pytest.mark.asyncio
    async def test_analyze_workload(self, on_demand_strategy):
        """Test on-demand workload analysis."""
        mock_metrics = {
            "invocations": [
                {"value": 100, "timestamp": datetime.utcnow()},
                {"value": 120, "timestamp": datetime.utcnow()},
            ]
        }

        mock_concurrency = {"concurrent_executions": [{"value": 5}, {"value": 8}, {"value": 6}]}

        with patch.object(
            on_demand_strategy, "_get_function_metrics"
        ) as mock_get_metrics, patch.object(
            on_demand_strategy, "_get_concurrent_executions_metrics"
        ) as mock_get_concurrency:

            mock_get_metrics.return_value = mock_metrics
            mock_get_concurrency.return_value = mock_concurrency

            workload = await on_demand_strategy.analyze_workload()

            assert workload.workload_type == WorkloadType.ON_DEMAND
            assert workload.cold_start_sensitivity == "high"
            assert workload.response_time_requirement == "strict"
            assert workload.peak_concurrency == 8

    def test_create_testing_strategy_high_cold_start_sensitivity(self, on_demand_strategy):
        """Test creating testing strategy for high cold start sensitivity."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.ON_DEMAND,
            invocation_frequency="medium",
            traffic_pattern="burst",
            cold_start_sensitivity="high",
            cost_sensitivity="medium",
            response_time_requirement="strict",
            peak_concurrency=10,
        )

        strategy = on_demand_strategy.create_testing_strategy(workload)

        assert isinstance(strategy, TestingStrategy)
        assert 512 in strategy.memory_sizes  # Should include higher memory sizes
        assert strategy.warmup_runs == 5  # More warmup for cold start sensitive
        assert strategy.traffic_simulation is True
        assert strategy.provisioned_concurrency_levels is not None  # Should test PC

    def test_create_testing_strategy_cost_sensitive(self, on_demand_strategy):
        """Test creating testing strategy for cost sensitive workload."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.ON_DEMAND,
            invocation_frequency="low",
            traffic_pattern="steady",
            cold_start_sensitivity="medium",
            cost_sensitivity="high",
            response_time_requirement="moderate",
            peak_concurrency=3,
        )

        strategy = on_demand_strategy.create_testing_strategy(workload)

        # Should focus on lower memory sizes for cost optimization
        assert max(strategy.memory_sizes) <= 1536
        assert strategy.provisioned_concurrency_levels is None  # No PC for low frequency

    @pytest.mark.asyncio
    async def test_execute_optimization(self, on_demand_strategy, aws_provider):
        """Test executing on-demand optimization."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.ON_DEMAND,
            invocation_frequency="medium",
            traffic_pattern="burst",
            cold_start_sensitivity="high",
            cost_sensitivity="medium",
            response_time_requirement="strict",
        )

        strategy = TestingStrategy(
            memory_sizes=[512, 1024],
            iterations_per_memory=5,
            concurrent_executions=3,
            warmup_runs=2,
            traffic_simulation=True,
        )

        # Mock function configuration
        aws_provider.get_function_configuration.return_value = {"MemorySize": 512}

        # Mock test results
        with patch.object(
            on_demand_strategy, "_test_memory_configuration_with_monitoring"
        ) as mock_test:
            mock_test.side_effect = [
                # 512MB results
                {
                    "memory_mb": 512,
                    "executions": [
                        {"duration": 800, "cold_start": True},
                        {"duration": 500, "cold_start": False},
                    ],
                    "success_rate": 100,
                    "avg_duration": 650,
                    "avg_billed_duration": 700,
                },
                # 1024MB results
                {
                    "memory_mb": 1024,
                    "executions": [
                        {"duration": 600, "cold_start": True},
                        {"duration": 400, "cold_start": False},
                    ],
                    "success_rate": 100,
                    "avg_duration": 500,
                    "avg_billed_duration": 550,
                },
            ]

            result = await on_demand_strategy.execute_optimization(workload, strategy)

            assert isinstance(result, OptimizationResult)
            assert result.workload_type == WorkloadType.ON_DEMAND
            assert result.optimal_memory in [512, 1024]
            assert result.confidence_score > 0

    def test_calculate_configuration_score(self, on_demand_strategy):
        """Test configuration scoring for on-demand workloads."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.ON_DEMAND,
            invocation_frequency="high",
            traffic_pattern="burst",
            cold_start_sensitivity="high",
            cost_sensitivity="medium",
            response_time_requirement="strict",
        )

        results = {
            "memory_mb": 1024,
            "avg_duration": 500,
            "avg_billed_duration": 600,
            "success_rate": 95,
            "executions": [
                {"cold_start": True, "duration": 800},
                {"cold_start": False, "duration": 400},
                {"cold_start": False, "duration": 450},
            ],
        }

        score = on_demand_strategy._calculate_configuration_score(results, workload)

        assert isinstance(score, float)
        assert score >= 0  # Should be non-negative


class TestContinuousStrategy(TestWorkloadStrategies):
    """Test ContinuousStrategy implementation."""

    @pytest.mark.asyncio
    async def test_analyze_workload(self, continuous_strategy):
        """Test continuous workload analysis."""
        mock_metrics = {
            "invocations": [
                {"value": 500, "timestamp": datetime.utcnow()},
                {"value": 520, "timestamp": datetime.utcnow()},
                {"value": 480, "timestamp": datetime.utcnow()},
            ]
        }

        mock_concurrency = {"concurrent_executions": [{"value": 20}, {"value": 25}, {"value": 18}]}

        with patch.object(
            continuous_strategy, "_get_function_metrics"
        ) as mock_get_metrics, patch.object(
            continuous_strategy, "_get_concurrent_executions_metrics"
        ) as mock_get_concurrency:

            mock_get_metrics.return_value = mock_metrics
            mock_get_concurrency.return_value = mock_concurrency

            workload = await continuous_strategy.analyze_workload()

            assert workload.workload_type == WorkloadType.CONTINUOUS
            assert workload.cold_start_sensitivity == "medium"
            assert workload.response_time_requirement == "moderate"
            assert workload.peak_concurrency == 25

    def test_create_testing_strategy_high_frequency(self, continuous_strategy):
        """Test creating testing strategy for high frequency continuous workload."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.CONTINUOUS,
            invocation_frequency="high",
            traffic_pattern="steady",
            cold_start_sensitivity="medium",
            cost_sensitivity="low",
            response_time_requirement="moderate",
            peak_concurrency=50,
        )

        strategy = continuous_strategy.create_testing_strategy(workload)

        assert isinstance(strategy, TestingStrategy)
        assert strategy.iterations_per_memory >= 25  # More iterations for high frequency
        assert strategy.test_duration_minutes == 10  # Time-based testing for steady traffic
        assert strategy.time_based_testing is True
        assert (
            strategy.provisioned_concurrency_levels is not None
        )  # Should test PC for high frequency

    @pytest.mark.asyncio
    async def test_test_sustained_performance(self, continuous_strategy, aws_provider):
        """Test sustained performance testing."""
        # Mock multiple waves of execution results
        aws_provider.invoke_function.side_effect = [
            # Wave 1
            {"duration": 500},
            {"duration": 510},
            {"duration": 505},
            {"duration": 520},
            {"duration": 515},
            {"duration": 525},
            {"duration": 530},
            {"duration": 535},
            {"duration": 528},
            {"duration": 540},
            {"duration": 545},
            {"duration": 538},
            # Wave 2
            {"duration": 515},
            {"duration": 525},
            {"duration": 520},
            {"duration": 535},
            {"duration": 530},
            {"duration": 540},
            {"duration": 545},
            {"duration": 550},
            {"duration": 542},
            {"duration": 555},
            {"duration": 560},
            {"duration": 548},
            # Continue for remaining waves...
        ] * 10  # Ensure enough mock responses

        with patch("asyncio.sleep"):  # Speed up the test
            result = await continuous_strategy._test_sustained_performance(1024)

        assert "performance_waves" in result
        assert "performance_degradation" in result
        assert "stability_score" in result
        assert len(result["performance_waves"]) > 0

    @pytest.mark.asyncio
    async def test_test_throughput_scaling(self, continuous_strategy, aws_provider):
        """Test throughput scaling testing."""
        # Mock concurrent execution results
        aws_provider.invoke_function.return_value = {"duration": 500, "status_code": 200}

        result = await continuous_strategy._test_throughput_scaling(1024)

        assert "concurrency_levels" in result
        assert "optimal_concurrency" in result
        assert "scaling_efficiency" in result

    def test_calculate_continuous_configuration_score(self, continuous_strategy):
        """Test configuration scoring for continuous workloads."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.CONTINUOUS,
            invocation_frequency="high",
            traffic_pattern="steady",
            cold_start_sensitivity="medium",
            cost_sensitivity="high",
            response_time_requirement="moderate",
        )

        results = {
            "memory_mb": 1024,
            "avg_duration": 500,
            "avg_billed_duration": 550,
            "success_rate": 98,
            "executions": [{"duration": 500}, {"duration": 510}, {"duration": 490}],
            "sustained_performance": {"stability_score": 0.95, "performance_degradation": False},
            "throughput_scaling": {"scaling_efficiency": 0.85},
        }

        score = continuous_strategy._calculate_continuous_configuration_score(results, workload)

        assert isinstance(score, float)
        assert score >= 0


class TestScheduledStrategy(TestWorkloadStrategies):
    """Test ScheduledStrategy implementation."""

    @pytest.mark.asyncio
    async def test_analyze_workload(self, scheduled_strategy):
        """Test scheduled workload analysis."""
        mock_recent_metrics = {
            "invocations": [
                {"value": 50, "timestamp": datetime.utcnow()},
                {"value": 60, "timestamp": datetime.utcnow()},
            ]
        }

        mock_weekly_metrics = {
            "invocations": [
                {"value": 50, "timestamp": datetime.utcnow() - timedelta(days=i)} for i in range(7)
            ]
        }

        mock_concurrency = {"concurrent_executions": [{"value": 5}, {"value": 8}, {"value": 3}]}

        with patch.object(
            scheduled_strategy, "_get_function_metrics"
        ) as mock_get_metrics, patch.object(
            scheduled_strategy, "_get_concurrent_executions_metrics"
        ) as mock_get_concurrency:

            mock_get_metrics.side_effect = [mock_recent_metrics, mock_weekly_metrics]
            mock_get_concurrency.return_value = mock_concurrency

            workload = await scheduled_strategy.analyze_workload()

            assert workload.workload_type == WorkloadType.SCHEDULED
            assert workload.cold_start_sensitivity == "medium"
            assert workload.response_time_requirement == "moderate"
            assert workload.peak_concurrency == 8

    def test_create_testing_strategy_bursty_pattern(self, scheduled_strategy):
        """Test creating testing strategy for bursty scheduled workload."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.SCHEDULED,
            invocation_frequency="medium",
            traffic_pattern="bursty",
            cold_start_sensitivity="medium",
            cost_sensitivity="medium",
            response_time_requirement="moderate",
            peak_concurrency=15,
        )

        strategy = scheduled_strategy.create_testing_strategy(workload)

        assert isinstance(strategy, TestingStrategy)
        assert 2048 in strategy.memory_sizes  # Should include higher memory for bursts
        assert strategy.iterations_per_memory == 18  # More iterations for bursty patterns
        assert strategy.provisioned_concurrency_levels is not None  # Should test PC for bursty

    @pytest.mark.asyncio
    async def test_test_schedule_simulation(self, scheduled_strategy, aws_provider):
        """Test schedule simulation testing."""
        # Mock execution results for different phases
        aws_provider.invoke_function.side_effect = [
            # Immediate start executions
            {"duration": 800, "cold_start": True},
            {"duration": 600, "cold_start": False},
            {"duration": 580, "cold_start": False},
            {"duration": 590, "cold_start": False},
            {"duration": 570, "cold_start": False},
            # Post-idle executions
            {"duration": 850, "cold_start": True},
            {"duration": 620, "cold_start": False},
            {"duration": 610, "cold_start": False},
        ]

        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.SCHEDULED,
            invocation_frequency="medium",
            traffic_pattern="predictable",
            cold_start_sensitivity="medium",
            cost_sensitivity="medium",
            response_time_requirement="moderate",
        )

        with patch("asyncio.sleep"):  # Speed up the test
            result = await scheduled_strategy._test_schedule_simulation(1024, workload)

        assert "immediate_start_performance" in result
        assert "post_idle_performance" in result
        assert "schedule_readiness_score" in result
        assert 0 <= result["schedule_readiness_score"] <= 1

    @pytest.mark.asyncio
    async def test_test_resource_efficiency(self, scheduled_strategy, aws_provider):
        """Test resource efficiency testing."""
        # Mock execution results with memory usage data
        aws_provider.invoke_function.side_effect = [
            {"duration": 500, "billed_duration": 600, "memory_used": 700},
            {"duration": 520, "billed_duration": 600, "memory_used": 750},
            {"duration": 510, "billed_duration": 600, "memory_used": 720},
            {"duration": 530, "billed_duration": 700, "memory_used": 780},
            {"duration": 500, "billed_duration": 600, "memory_used": 690},
            {"duration": 515, "billed_duration": 600, "memory_used": 730},
            {"duration": 525, "billed_duration": 700, "memory_used": 760},
            {"duration": 505, "billed_duration": 600, "memory_used": 710},
        ]

        with patch("asyncio.sleep"):  # Speed up the test
            result = await scheduled_strategy._test_resource_efficiency(1024)

        assert "memory_utilization_ratio" in result
        assert "cost_per_execution" in result
        assert "resource_waste_score" in result
        assert "efficiency_rating" in result
        assert result["efficiency_rating"] in ["excellent", "good", "fair", "poor"]

    @pytest.mark.asyncio
    async def test_test_idle_recovery(self, scheduled_strategy, aws_provider):
        """Test idle recovery testing."""
        aws_provider.invoke_function.side_effect = [
            # Pre-idle execution
            {"duration": 500, "cold_start": False},
            # Post-idle execution
            {"duration": 800, "cold_start": True},
        ]

        with patch("asyncio.sleep"):  # Speed up the test
            result = await scheduled_strategy._test_idle_recovery(1024)

        assert "pre_idle_performance" in result
        assert "post_idle_performance" in result
        assert "recovery_time_seconds" in result
        assert "idle_impact_score" in result
        assert result["idle_impact_score"] >= 0

    def test_classify_scheduled_frequency(self, scheduled_strategy):
        """Test scheduled frequency classification."""
        recent_metrics = {"invocations": [{"value": 100}, {"value": 120}, {"value": 80}]}

        weekly_metrics = {
            "invocations": [
                {"value": 50},
                {"value": 60},
                {"value": 40},
                {"value": 55},
                {"value": 45},
                {"value": 65},
                {"value": 35},
            ]
        }

        frequency = scheduled_strategy._classify_scheduled_frequency(recent_metrics, weekly_metrics)

        assert frequency in ["high", "medium", "low"]

    def test_analyze_scheduled_pattern(self, scheduled_strategy):
        """Test scheduled pattern analysis."""
        weekly_metrics = {
            "invocations": [
                {"value": 100, "timestamp": datetime(2024, 1, 1, 9, 0)},  # 9 AM
                {"value": 120, "timestamp": datetime(2024, 1, 1, 10, 0)},  # 10 AM
                {"value": 110, "timestamp": datetime(2024, 1, 2, 9, 0)},  # Next day 9 AM
                {"value": 115, "timestamp": datetime(2024, 1, 2, 10, 0)},  # Next day 10 AM
            ]
        }

        pattern = scheduled_strategy._analyze_scheduled_pattern(weekly_metrics)

        assert pattern in ["predictable", "steady", "bursty", "unpredictable"]

    def test_calculate_scheduled_configuration_score(self, scheduled_strategy):
        """Test configuration scoring for scheduled workloads."""
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.SCHEDULED,
            invocation_frequency="medium",
            traffic_pattern="predictable",
            cold_start_sensitivity="medium",
            cost_sensitivity="high",
            response_time_requirement="moderate",
        )

        results = {
            "memory_mb": 1024,
            "avg_duration": 600,
            "avg_billed_duration": 700,
            "success_rate": 96,
            "executions": [{"duration": 600}, {"duration": 620}, {"duration": 580}],
            "schedule_simulation": {"schedule_readiness_score": 0.9},
            "resource_efficiency": {"memory_utilization_ratio": 0.75},
            "idle_recovery": {"idle_impact_score": 0.2},
        }

        score = scheduled_strategy._calculate_scheduled_configuration_score(results, workload)

        assert isinstance(score, float)
        assert score >= 0


class TestWorkloadStrategyBase(TestWorkloadStrategies):
    """Test base WorkloadStrategy functionality."""

    @pytest.mark.asyncio
    async def test_optimize_workflow(self, on_demand_strategy):
        """Test complete optimization workflow."""
        with patch.object(on_demand_strategy, "analyze_workload") as mock_analyze, patch.object(
            on_demand_strategy, "create_testing_strategy"
        ) as mock_create, patch.object(on_demand_strategy, "execute_optimization") as mock_execute:

            # Setup mocks
            mock_workload = WorkloadCharacteristics(
                workload_type=WorkloadType.ON_DEMAND,
                invocation_frequency="medium",
                traffic_pattern="burst",
                cold_start_sensitivity="high",
                cost_sensitivity="medium",
                response_time_requirement="strict",
            )

            mock_strategy = TestingStrategy(
                memory_sizes=[512, 1024],
                iterations_per_memory=10,
                concurrent_executions=5,
                warmup_runs=3,
            )

            mock_result = OptimizationResult(
                workload_type=WorkloadType.ON_DEMAND,
                optimal_memory=1024,
                optimal_provisioned_concurrency=None,
                cost_savings_percent=15.0,
                performance_improvement_percent=25.0,
                confidence_score=0.9,
                reasoning="Test reasoning",
                test_results={},
                recommendations=["Test recommendation"],
            )

            mock_analyze.return_value = mock_workload
            mock_create.return_value = mock_strategy
            mock_execute.return_value = mock_result

            result = await on_demand_strategy.optimize()

            assert isinstance(result, OptimizationResult)
            assert result.optimal_memory == 1024
            assert result.confidence_score == 0.9

            mock_analyze.assert_called_once()
            mock_create.assert_called_once_with(mock_workload)
            mock_execute.assert_called_once_with(mock_workload, mock_strategy)

    def test_analyze_traffic_pattern(self, on_demand_strategy):
        """Test traffic pattern analysis."""
        # Test steady pattern
        steady_metrics = {
            "invocations": [{"value": 100}, {"value": 105}, {"value": 95}, {"value": 102}]
        }
        pattern = on_demand_strategy._analyze_traffic_pattern(steady_metrics)
        assert pattern == "steady"

        # Test bursty pattern
        bursty_metrics = {
            "invocations": [{"value": 50}, {"value": 200}, {"value": 60}, {"value": 180}]
        }
        pattern = on_demand_strategy._analyze_traffic_pattern(bursty_metrics)
        assert pattern == "bursty"

    def test_classify_invocation_frequency(self, on_demand_strategy):
        """Test invocation frequency classification."""
        # Test high frequency
        high_freq_metrics = {"invocations": [{"value": 500}, {"value": 600}, {"value": 550}]}
        frequency = on_demand_strategy._classify_invocation_frequency(high_freq_metrics)
        assert frequency == "high"

        # Test low frequency
        low_freq_metrics = {"invocations": [{"value": 5}, {"value": 8}, {"value": 3}]}
        frequency = on_demand_strategy._classify_invocation_frequency(low_freq_metrics)
        assert frequency == "low"

    @pytest.mark.asyncio
    async def test_test_memory_configuration_with_monitoring(
        self, on_demand_strategy, aws_provider
    ):
        """Test memory configuration testing with monitoring."""
        # Mock function configuration and invocations
        aws_provider.get_function_configuration.return_value = {"MemorySize": 1024}
        aws_provider.invoke_function.side_effect = [
            # Warmup executions
            {"duration": 800, "cold_start": True},
            {"duration": 500, "cold_start": False},
            # Test executions
            {"duration": 520, "cold_start": False, "memory_used": 700},
            {"duration": 510, "cold_start": False, "memory_used": 720},
            {"duration": 530, "cold_start": False, "memory_used": 680},
            {"duration": 500, "cold_start": False, "memory_used": 710},
            {"duration": 515, "cold_start": False, "memory_used": 690},
        ]

        result = await on_demand_strategy._test_memory_configuration_with_monitoring(1024, 5)

        assert result["memory_mb"] == 1024
        assert len(result["executions"]) == 5  # Should exclude warmup executions
        assert "avg_duration" in result
        assert "p95_duration" in result
        assert "success_rate" in result

    def test_calculate_test_metrics(self, on_demand_strategy):
        """Test test metrics calculation."""
        executions = [
            {"duration": 500, "billed_duration": 600, "cold_start": True},
            {"duration": 450, "billed_duration": 500, "cold_start": False},
            {"duration": 480, "billed_duration": 500, "cold_start": False},
            {"error": "timeout", "cold_start": False},
            {"duration": 470, "billed_duration": 500, "cold_start": False},
        ]

        metrics = on_demand_strategy._calculate_test_metrics(executions)

        assert metrics["cold_starts"] == 1
        assert metrics["errors"] == 1
        assert metrics["success_rate"] == 80  # 4 successful out of 5
        assert metrics["avg_duration"] == 475  # Average of successful executions
        assert metrics["min_duration"] == 450
        assert metrics["max_duration"] == 500

    def test_simulate_memory_test(self, on_demand_strategy):
        """Test memory test simulation."""
        results = on_demand_strategy._simulate_memory_test(1024, 10)

        assert len(results["executions"]) == 10
        assert results["cold_starts"] == 1  # First execution should be cold start
        assert results["errors"] >= 0
        assert "avg_duration" in results
        assert "success_rate" in results
