"""Integration tests for enhanced orchestrator functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from aws_lambda_tuner.orchestrator_module import TunerOrchestrator
from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.exceptions import LambdaExecutionError


class TestEnhancedOrchestrator:
    """Test enhanced orchestrator features."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            workload_type="on_demand",
            expected_concurrency=5,
            traffic_pattern="burst",
            cold_start_sensitivity="high",
            memory_sizes=[512, 1024, 1536],
            iterations=5,
            concurrent_executions=3,
            dry_run=True,
        )

    @pytest.fixture
    def orchestrator(self, config):
        """Create orchestrator instance."""
        return TunerOrchestrator(config)

    @pytest.mark.asyncio
    async def test_workload_aware_testing_on_demand(self, orchestrator):
        """Test workload-aware testing for on-demand workloads."""
        with patch.object(orchestrator, "_test_on_demand_workload") as mock_test:
            mock_test.return_value = {
                "memory_mb": 1024,
                "executions": [],
                "cold_start_performance": {},
                "burst_performance": {},
                "total_executions": 5,
                "successful_executions": 5,
                "failed_executions": 0,
                "workload_type": "on_demand",
            }

            result = await orchestrator.workload_aware_testing(1024)

            assert result["workload_type"] == "on_demand"
            assert result["memory_mb"] == 1024
            mock_test.assert_called_once_with(1024)

    @pytest.mark.asyncio
    async def test_workload_aware_testing_continuous(self, orchestrator):
        """Test workload-aware testing for continuous workloads."""
        orchestrator.config.workload_type = "continuous"

        with patch.object(orchestrator, "_test_continuous_workload") as mock_test:
            mock_test.return_value = {
                "memory_mb": 1024,
                "executions": [],
                "sustained_performance": {},
                "concurrency_performance": {},
                "total_executions": 5,
                "successful_executions": 5,
                "failed_executions": 0,
                "workload_type": "continuous",
            }

            result = await orchestrator.workload_aware_testing(1024)

            assert result["workload_type"] == "continuous"
            assert result["memory_mb"] == 1024
            mock_test.assert_called_once_with(1024)

    @pytest.mark.asyncio
    async def test_workload_aware_testing_scheduled(self, orchestrator):
        """Test workload-aware testing for scheduled workloads."""
        orchestrator.config.workload_type = "scheduled"

        with patch.object(orchestrator, "_test_scheduled_workload") as mock_test:
            mock_test.return_value = {
                "memory_mb": 1024,
                "executions": [],
                "time_window_performance": {},
                "resource_efficiency": {},
                "total_executions": 5,
                "successful_executions": 5,
                "failed_executions": 0,
                "workload_type": "scheduled",
            }

            result = await orchestrator.workload_aware_testing(1024)

            assert result["workload_type"] == "scheduled"
            assert result["memory_mb"] == 1024
            mock_test.assert_called_once_with(1024)

    @pytest.mark.asyncio
    async def test_time_window_testing(self, orchestrator):
        """Test time window testing functionality."""
        with patch.object(orchestrator, "_run_concurrent_invocations") as mock_invoke:
            mock_invoke.side_effect = [
                # Immediate execution results
                [
                    {"execution_id": 0, "duration": 800, "cold_start": True},
                    {"execution_id": 1, "duration": 500, "cold_start": False},
                ],
                # Recovery execution results
                [{"execution_id": 2, "duration": 750, "cold_start": True}],
            ]

            result = await orchestrator.time_window_testing(1024)

            assert result["memory_mb"] == 1024
            assert "window_analysis" in result
            assert "idle_recovery" in result
            assert len(result["executions"]) == 3
            assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_stage_optimization(self, orchestrator):
        """Test multi-stage optimization workflow."""
        # Mock all the stages
        with patch.object(orchestrator, "_establish_baseline") as mock_baseline, patch.object(
            orchestrator, "_initial_memory_sweep"
        ) as mock_sweep, patch.object(
            orchestrator, "_focused_optimization"
        ) as mock_focused, patch.object(
            orchestrator, "_workload_validation"
        ) as mock_validation, patch.object(
            orchestrator, "_generate_recommendations"
        ) as mock_recommendations:

            # Setup mocks
            mock_baseline.return_value = {"baseline": "data"}
            mock_sweep.return_value = {
                "configurations": [
                    {"memory_mb": 512, "successful_executions": 5, "executions": []},
                    {"memory_mb": 1024, "successful_executions": 5, "executions": []},
                ]
            }
            mock_focused.return_value = {
                "top_configurations": [{"memory_mb": 1024, "optimization_score": 0.8}]
            }
            mock_validation.return_value = {"workload_compliance": {1024: 0.9}}
            mock_recommendations.return_value = {
                "optimal_memory_size": 1024,
                "confidence_score": 0.9,
            }

            result = await orchestrator.multi_stage_optimization()

            assert "stages" in result
            assert "recommendations" in result
            assert result["function_arn"] == orchestrator.config.function_arn

            # Verify all stages were called
            mock_baseline.assert_called_once()
            mock_sweep.assert_called_once()
            mock_focused.assert_called_once()
            mock_validation.assert_called_once()
            mock_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_demand_workload_testing(self, orchestrator):
        """Test on-demand workload specific testing."""
        with patch.object(orchestrator, "_test_cold_start_scenarios") as mock_cold, patch.object(
            orchestrator, "_test_burst_pattern"
        ) as mock_burst:

            mock_cold.return_value = {
                "executions": [
                    {"execution_id": 0, "duration": 1000, "cold_start": True},
                    {"execution_id": 1, "duration": 500, "cold_start": False},
                ],
                "cold_start_analysis": {"count": 1, "avg_duration": 1000},
                "warm_start_analysis": {"count": 1, "avg_duration": 500},
            }

            mock_burst.return_value = {
                "executions": [{"execution_id": 2, "duration": 600, "cold_start": False}],
                "burst_analysis": {"total_executions": 1, "avg_duration": 600, "success_rate": 1.0},
            }

            result = await orchestrator._test_on_demand_workload(1024)

            assert result["workload_type"] == "on_demand"
            assert result["memory_mb"] == 1024
            assert len(result["executions"]) == 3
            assert result["successful_executions"] == 3
            assert result["failed_executions"] == 0
            assert "cold_start_performance" in result
            assert "burst_performance" in result

    @pytest.mark.asyncio
    async def test_continuous_workload_testing(self, orchestrator):
        """Test continuous workload specific testing."""
        with patch.object(
            orchestrator, "_test_sustained_performance"
        ) as mock_sustained, patch.object(
            orchestrator, "_test_concurrency_patterns"
        ) as mock_concurrency:

            mock_sustained.return_value = {
                "executions": [
                    {"execution_id": 0, "duration": 500, "cold_start": False},
                    {"execution_id": 1, "duration": 520, "cold_start": False},
                ],
                "performance_stability": {"wave_performances": [500, 520], "stability_score": 0.95},
            }

            mock_concurrency.return_value = {
                "executions": [{"execution_id": 2, "duration": 550, "cold_start": False}],
                "concurrency_analysis": {"optimal_concurrency": 5},
            }

            result = await orchestrator._test_continuous_workload(1024)

            assert result["workload_type"] == "continuous"
            assert result["memory_mb"] == 1024
            assert len(result["executions"]) == 3
            assert result["successful_executions"] == 3
            assert "sustained_performance" in result
            assert "concurrency_performance" in result

    @pytest.mark.asyncio
    async def test_scheduled_workload_testing(self, orchestrator):
        """Test scheduled workload specific testing."""
        with patch.object(orchestrator, "time_window_testing") as mock_time_window, patch.object(
            orchestrator, "_test_resource_efficiency"
        ) as mock_efficiency:

            mock_time_window.return_value = {
                "executions": [{"execution_id": 0, "duration": 700, "cold_start": True}],
                "window_analysis": {
                    "immediate": {
                        "duration": 2.5,
                        "avg_execution_time": 700,
                        "cold_start_ratio": 1.0,
                    }
                },
            }

            mock_efficiency.return_value = {
                "executions": [{"execution_id": 1, "duration": 650, "cold_start": False}],
                "efficiency_metrics": {"memory_utilization": 0.75, "resource_waste_ratio": 0.25},
            }

            result = await orchestrator._test_scheduled_workload(1024)

            assert result["workload_type"] == "scheduled"
            assert result["memory_mb"] == 1024
            assert len(result["executions"]) == 2
            assert result["successful_executions"] == 2
            assert "time_window_performance" in result
            assert "resource_efficiency" in result

    @pytest.mark.asyncio
    async def test_cold_start_scenarios(self, orchestrator):
        """Test cold start scenario testing."""
        with patch.object(orchestrator, "_run_concurrent_invocations") as mock_invoke:
            # Setup mock responses for different execution phases
            mock_invoke.side_effect = [
                # Cold start execution
                [{"execution_id": 0, "duration": 1200, "cold_start": True}],
                # Brief pause execution
                [
                    {"execution_id": 1, "duration": 1100, "cold_start": True},
                    {"execution_id": 2, "duration": 550, "cold_start": False},
                ],
                # Warm executions
                [
                    {"execution_id": 3, "duration": 500, "cold_start": False},
                    {"execution_id": 4, "duration": 520, "cold_start": False},
                    {"execution_id": 5, "duration": 510, "cold_start": False},
                ],
            ]

            result = await orchestrator._test_cold_start_scenarios(1024)

            assert len(result["executions"]) == 6
            assert "cold_start_analysis" in result
            assert "warm_start_analysis" in result

            cold_analysis = result["cold_start_analysis"]
            warm_analysis = result["warm_start_analysis"]

            assert cold_analysis["count"] == 2
            assert warm_analysis["count"] == 4
            assert cold_analysis["avg_duration"] > warm_analysis["avg_duration"]

    @pytest.mark.asyncio
    async def test_sustained_performance_testing(self, orchestrator):
        """Test sustained performance testing."""
        with patch.object(orchestrator, "_run_concurrent_invocations") as mock_invoke:
            # Mock three waves of executions
            mock_invoke.side_effect = [
                # Wave 1
                [
                    {"execution_id": 0, "duration": 500},
                    {"execution_id": 1, "duration": 510},
                    {"execution_id": 2, "duration": 505},
                ],
                # Wave 2
                [
                    {"execution_id": 3, "duration": 520},
                    {"execution_id": 4, "duration": 515},
                    {"execution_id": 5, "duration": 525},
                ],
                # Wave 3
                [
                    {"execution_id": 6, "duration": 530},
                    {"execution_id": 7, "duration": 535},
                    {"execution_id": 8, "duration": 528},
                ],
            ]

            result = await orchestrator._test_sustained_performance(1024)

            assert len(result["executions"]) == 9
            assert "performance_stability" in result

            stability = result["performance_stability"]
            assert len(stability["wave_performances"]) == 3
            assert "stability_score" in stability
            assert 0 <= stability["stability_score"] <= 1

    @pytest.mark.asyncio
    async def test_concurrency_patterns_testing(self, orchestrator):
        """Test concurrency patterns testing."""
        with patch.object(orchestrator, "_run_concurrent_invocations") as mock_invoke:
            # Mock responses for different concurrency levels
            mock_invoke.side_effect = [
                # Concurrency 1
                [{"execution_id": 0, "duration": 500}],
                # Concurrency 2 (half of configured)
                [{"execution_id": 1, "duration": 510}, {"execution_id": 2, "duration": 520}],
                # Concurrency 3 (full configured)
                [
                    {"execution_id": 3, "duration": 520},
                    {"execution_id": 4, "duration": 530},
                    {"execution_id": 5, "duration": 525},
                ],
            ]

            result = await orchestrator._test_concurrency_patterns(1024)

            assert "concurrency_analysis" in result

            analysis = result["concurrency_analysis"]
            assert "concurrency_performance" in analysis
            assert "optimal_concurrency" in analysis

            # Check that we tested the expected concurrency levels
            perf_data = analysis["concurrency_performance"]
            assert 1 in perf_data
            assert 1 in perf_data  # Half of 3 is 1 (integer division)
            assert 3 in perf_data

    def test_find_optimal_concurrency(self, orchestrator):
        """Test finding optimal concurrency level."""
        concurrency_performance = {
            1: {"avg_duration": 500, "success_rate": 1.0, "total_executions": 2},
            3: {"avg_duration": 520, "success_rate": 1.0, "total_executions": 6},
            5: {"avg_duration": 600, "success_rate": 0.8, "total_executions": 10},
        }

        optimal = orchestrator._find_optimal_concurrency(concurrency_performance)

        # Should pick concurrency level 3 as it balances performance and success rate
        assert optimal == 3

    def test_calculate_execution_cost(self, orchestrator):
        """Test execution cost calculation."""
        executions = [
            {"billed_duration": 1000, "duration": 950},  # 1 second billed
            {"billed_duration": 500, "duration": 450},  # 0.5 seconds billed
            {"billed_duration": 1500, "duration": 1400},  # 1.5 seconds billed
        ]

        cost = orchestrator._calculate_execution_cost(1024, executions)

        # Should be average cost across all executions
        assert cost > 0
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_workload_validation(self, orchestrator):
        """Test workload validation stage."""
        focused_results = {
            "top_configurations": [
                {
                    "memory_mb": 1024,
                    "executions": [
                        {"duration": 500, "cold_start": False},
                        {"duration": 520, "cold_start": False},
                    ],
                    "successful_executions": 2,
                    "total_executions": 2,
                },
                {
                    "memory_mb": 1536,
                    "executions": [
                        {"duration": 400, "cold_start": False},
                        {"duration": 420, "cold_start": False},
                    ],
                    "successful_executions": 2,
                    "total_executions": 2,
                },
            ]
        }

        result = await orchestrator._workload_validation(focused_results)

        assert "workload_compliance" in result
        assert "performance_validation" in result
        assert "cost_validation" in result

        # Should have validation scores for both memory sizes
        assert 1024 in result["workload_compliance"]
        assert 1536 in result["workload_compliance"]

    def test_calculate_workload_compliance(self, orchestrator):
        """Test workload compliance calculation."""
        config_on_demand = {
            "executions": [
                {"cold_start": False, "duration": 500},
                {"cold_start": False, "duration": 520},
                {"cold_start": True, "duration": 800},  # Only 1 cold start out of 3
            ],
            "successful_executions": 3,
            "total_executions": 3,
            "burst_performance": {"avg_duration": 550},
        }

        score = orchestrator._calculate_workload_compliance(config_on_demand)

        # Should get bonus for low cold start ratio and burst performance
        assert 0 <= score <= 1
        assert score > 0.5  # Should be reasonably good score

    def test_calculate_performance_score(self, orchestrator):
        """Test performance score calculation."""
        config = {
            "executions": [{"duration": 500}, {"duration": 520}, {"duration": 510}],
            "successful_executions": 3,
            "total_executions": 3,
        }

        score = orchestrator._calculate_performance_score(config)

        assert 0 <= score <= 1
        assert score > 0  # Should have positive score for successful executions

    def test_calculate_cost_score(self, orchestrator):
        """Test cost score calculation."""
        config = {
            "memory_mb": 1024,
            "executions": [
                {"duration": 1000, "billed_duration": 1000},
                {"duration": 1100, "billed_duration": 1100},
            ],
        }

        score = orchestrator._calculate_cost_score(config)

        assert isinstance(score, float)
        assert score >= 0  # Cost score should be non-negative

    @pytest.mark.asyncio
    async def test_error_handling_in_multi_stage(self, orchestrator):
        """Test error handling in multi-stage optimization."""
        with patch.object(orchestrator, "_establish_baseline") as mock_baseline:
            mock_baseline.side_effect = LambdaExecutionError("Test error")

            with pytest.raises(LambdaExecutionError):
                await orchestrator.multi_stage_optimization()

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, orchestrator):
        """Test dry run mode functionality."""
        # Orchestrator is already configured for dry run
        assert orchestrator.config.dry_run is True

        # Test workload-aware testing in dry run mode
        result = await orchestrator.workload_aware_testing(1024)

        # Should return simulated results
        assert result["memory_mb"] == 1024
        assert result["workload_type"] == "on_demand"
        assert len(result["executions"]) > 0

        # Check that executions have dry_run flag
        for execution in result["executions"]:
            assert execution.get("dry_run") is True or "simulation" in str(execution)
