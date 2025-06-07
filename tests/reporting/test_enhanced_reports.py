"""
Tests for enhanced reporting functionality.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from aws_lambda_tuner.report_service import ReportGenerator, WorkloadType, TrafficPattern
from aws_lambda_tuner.exceptions import ReportGenerationError


class TestEnhancedReportGeneration:
    """Test enhanced report generation features."""

    @pytest.fixture
    def sample_results(self):
        """Sample tuning results for testing."""
        return {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
            "test_duration_seconds": 300,
            "configurations": [
                {
                    "memory_mb": 128,
                    "executions": [
                        {
                            "duration": 1500,
                            "billed_duration": 1500,
                            "cold_start": True,
                            "error": None,
                        },
                        {
                            "duration": 800,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 850,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                        },
                    ],
                },
                {
                    "memory_mb": 256,
                    "executions": [
                        {
                            "duration": 900,
                            "billed_duration": 900,
                            "cold_start": True,
                            "error": None,
                        },
                        {
                            "duration": 400,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 450,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        },
                    ],
                },
                {
                    "memory_mb": 512,
                    "executions": [
                        {
                            "duration": 600,
                            "billed_duration": 600,
                            "cold_start": True,
                            "error": None,
                        },
                        {
                            "duration": 200,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 250,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                        },
                    ],
                },
            ],
        }

    @pytest.fixture
    def report_generator(self, sample_results):
        """Report generator instance for testing."""
        return ReportGenerator(sample_results)

    def test_workload_specific_web_api_report(self, sample_results):
        """Test web API workload-specific report generation."""
        generator = ReportGenerator(sample_results, None, WorkloadType.WEB_API)
        report = generator.workload_specific_reports()

        assert report["workload_type"] == "web_api"
        assert "key_metrics" in report
        assert "p95_latency" in report["key_metrics"]
        assert "p99_latency" in report["key_metrics"]
        assert "cold_start_percentage" in report["key_metrics"]
        assert "recommendations" in report

        # Check recommendations are web API specific
        recommendations = report["recommendations"]
        assert any("latency" in rec["category"] for rec in recommendations)

    def test_workload_specific_batch_processing_report(self, sample_results):
        """Test batch processing workload-specific report generation."""
        generator = ReportGenerator(sample_results, None, WorkloadType.BATCH_PROCESSING)
        report = generator.workload_specific_reports()

        assert report["workload_type"] == "batch_processing"
        assert "key_metrics" in report
        assert "avg_duration" in report["key_metrics"]
        assert "cost_per_execution" in report["key_metrics"]
        assert "throughput_improvement" in report["key_metrics"]

        # Check recommendations are batch processing specific
        recommendations = report["recommendations"]
        assert any("throughput" in rec["category"] for rec in recommendations)

    def test_workload_specific_event_driven_report(self, sample_results):
        """Test event-driven workload-specific report generation."""
        generator = ReportGenerator(sample_results, None, WorkloadType.EVENT_DRIVEN)
        report = generator.workload_specific_reports()

        assert report["workload_type"] == "event_driven"
        assert "key_metrics" in report
        assert "avg_processing_time" in report["key_metrics"]
        assert "event_success_rate" in report["key_metrics"]

        # Check recommendations are event-driven specific
        recommendations = report["recommendations"]
        assert any("reliability" in rec["category"] for rec in recommendations)

    def test_cost_projection_reports(self, report_generator):
        """Test cost projection report generation."""
        scenarios = [
            {
                "name": "low_volume",
                "daily_invocations": 1000,
                "pattern": "steady",
                "duration_days": 30,
            },
            {
                "name": "high_volume",
                "daily_invocations": 100000,
                "pattern": "bursty",
                "duration_days": 30,
            },
        ]

        report = report_generator.cost_projection_reports(scenarios)

        assert "optimal_configuration" in report
        assert "projections" in report
        assert "comparison_baseline" in report
        assert "savings_analysis" in report
        assert "recommendations" in report

        # Check projections for each scenario
        projections = report["projections"]
        assert "low_volume" in projections
        assert "high_volume" in projections

        # Verify projection structure
        low_volume_proj = projections["low_volume"]
        assert "cost_per_invocation" in low_volume_proj
        assert "daily_cost" in low_volume_proj
        assert "monthly_cost" in low_volume_proj
        assert "yearly_cost" in low_volume_proj

        # Check savings analysis
        savings = report["savings_analysis"]
        assert "total_savings" in savings
        assert "scenario_savings" in savings
        assert "average_savings_percentage" in savings

    def test_comparative_analysis_reports(self, sample_results):
        """Test comparative analysis across workload types."""
        generator = ReportGenerator(sample_results, None, WorkloadType.WEB_API)

        # Create comparison workloads
        comparison_workloads = [
            {"type": "batch_processing", "name": "Batch Workload", "results": sample_results},
            {"type": "event_driven", "name": "Event Workload", "results": sample_results},
        ]

        report = generator.comparative_analysis_reports(comparison_workloads)

        assert "current_workload" in report
        assert "comparisons" in report
        assert "cross_workload_insights" in report
        assert "best_practices" in report

        # Check current workload
        current = report["current_workload"]
        assert current["type"] == "web_api"

        # Check comparisons
        comparisons = report["comparisons"]
        assert len(comparisons) == 2
        assert any(comp["workload_type"] == "batch_processing" for comp in comparisons)
        assert any(comp["workload_type"] == "event_driven" for comp in comparisons)

    def test_cold_start_impact_analysis(self, report_generator):
        """Test cold start impact analysis."""
        # Access private method for testing
        cold_start_impact = report_generator._analyze_cold_start_impact()

        assert "percentage" in cold_start_impact
        assert "total_cold_starts" in cold_start_impact
        assert "avg_cold_duration" in cold_start_impact
        assert "avg_warm_duration" in cold_start_impact
        assert "avg_penalty_ms" in cold_start_impact

        # Verify calculations
        assert cold_start_impact["total_cold_starts"] == 3  # One per memory size
        assert cold_start_impact["percentage"] > 0
        assert cold_start_impact["avg_penalty_ms"] >= 0

    def test_latency_percentiles_calculation(self, report_generator):
        """Test latency percentiles calculation."""
        # Access private method for testing
        percentiles = report_generator._calculate_latency_percentiles()

        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

        # Verify percentile ordering
        assert percentiles["p50"] <= percentiles["p90"]
        assert percentiles["p90"] <= percentiles["p95"]
        assert percentiles["p95"] <= percentiles["p99"]

    def test_throughput_analysis(self, report_generator):
        """Test throughput analysis calculations."""
        # Access private method for testing
        throughput = report_generator._analyze_throughput()

        assert "max_throughput" in throughput
        assert "min_throughput" in throughput
        assert "improvement_percentage" in throughput

        # Verify throughput values
        assert throughput["max_throughput"] >= throughput["min_throughput"]
        assert throughput["improvement_percentage"] >= 0

    def test_cost_efficiency_analysis(self, report_generator):
        """Test cost efficiency analysis."""
        optimal = report_generator._find_optimal_configuration()
        baseline = report_generator._get_baseline_stats()

        # Access private method for testing
        efficiency = report_generator._analyze_cost_efficiency(optimal, baseline)

        assert "cost_per_ms" in efficiency
        assert "baseline_cost_per_ms" in efficiency
        assert "ratio" in efficiency

        # Verify efficiency calculations
        assert efficiency["cost_per_ms"] > 0
        assert efficiency["baseline_cost_per_ms"] > 0
        assert efficiency["ratio"] > 0

    def test_traffic_pattern_cost_adjustments(self, report_generator):
        """Test traffic pattern cost adjustments."""
        optimal = report_generator._find_optimal_configuration()

        # Test different traffic patterns
        patterns = [
            (TrafficPattern.STEADY, 1.0),
            (TrafficPattern.BURSTY, 1.2),
            (TrafficPattern.SEASONAL, 1.1),
            (TrafficPattern.GROWTH, 1.05),
        ]

        for pattern, expected_multiplier in patterns:
            projection = report_generator._calculate_cost_projection(optimal, 1000, pattern, 30)

            assert projection["pattern_impact"] == expected_multiplier
            assert projection["cost_per_invocation"] == optimal["avg_cost"] * expected_multiplier

    def test_workload_best_practices_generation(self, report_generator):
        """Test workload-specific best practices generation."""
        # Create mock comparison data
        comparisons = [
            {"workload_type": "web_api", "analysis": {"key_metrics": {"optimal_memory": 256}}},
            {
                "workload_type": "batch_processing",
                "analysis": {"key_metrics": {"optimal_memory": 512}},
            },
        ]

        # Access private method for testing
        best_practices = report_generator._generate_workload_best_practices(comparisons)

        assert len(best_practices) > 0
        assert all("category" in practice for practice in best_practices)
        assert all("practice" in practice for practice in best_practices)

        # Check for expected categories
        categories = [practice["category"] for practice in best_practices]
        assert "memory_optimization" in categories
        assert "monitoring" in categories
        assert "cost_management" in categories

    def test_report_generation_error_handling(self, sample_results):
        """Test error handling in report generation."""
        # Test with invalid workload type
        with pytest.raises(ValueError):
            ReportGenerator(sample_results, None, "invalid_workload")

        # Test with empty results
        empty_results = {"configurations": []}
        generator = ReportGenerator(empty_results)

        # Should handle empty results gracefully
        summary = generator.get_summary()
        assert summary is not None

    def test_memory_recommendation_logic(self, report_generator):
        """Test memory recommendation logic for different strategies."""
        # Test with mock config
        mock_config = Mock()
        mock_config.strategy = "speed"

        generator_speed = ReportGenerator(report_generator.results, mock_config)
        optimal_speed = generator_speed._find_optimal_configuration()

        mock_config.strategy = "cost"
        generator_cost = ReportGenerator(report_generator.results, mock_config)
        optimal_cost = generator_cost._find_optimal_configuration()

        # Speed optimization should prefer faster execution
        # Cost optimization should prefer lower cost
        assert optimal_speed is not None
        assert optimal_cost is not None

    def test_scaling_recommendations_generation(self, sample_results):
        """Test scaling recommendations for different workload types."""
        workload_types = [
            WorkloadType.WEB_API,
            WorkloadType.BATCH_PROCESSING,
            WorkloadType.EVENT_DRIVEN,
            WorkloadType.SCHEDULED,
            WorkloadType.STREAM_PROCESSING,
        ]

        for workload_type in workload_types:
            generator = ReportGenerator(sample_results, None, workload_type)
            report = generator.workload_specific_reports()

            scaling_recs = report.get("scaling_recommendations", [])
            assert isinstance(scaling_recs, list)

            # Each scaling recommendation should have metric and recommendation
            for rec in scaling_recs:
                assert "metric" in rec
                assert "recommendation" in rec

    def test_execution_consistency_analysis(self, report_generator):
        """Test execution consistency analysis for scheduled workloads."""
        # Access private method for testing
        consistency = report_generator._analyze_execution_consistency()

        if consistency:  # Only if there are enough executions
            assert "coefficient_of_variation" in consistency
            assert "cost_variance" in consistency
            assert "cold_start_rate" in consistency

            # Coefficient of variation should be non-negative
            assert consistency["coefficient_of_variation"] >= 0


class TestCostProjectionFeatures:
    """Test cost projection specific features."""

    @pytest.fixture
    def sample_scenarios(self):
        """Sample cost projection scenarios."""
        return [
            {
                "name": "development",
                "daily_invocations": 100,
                "pattern": "steady",
                "duration_days": 30,
            },
            {
                "name": "production",
                "daily_invocations": 50000,
                "pattern": "bursty",
                "duration_days": 30,
            },
            {
                "name": "peak_season",
                "daily_invocations": 200000,
                "pattern": "seasonal",
                "duration_days": 60,
            },
        ]

    def test_cost_projection_structure(self, sample_results, sample_scenarios):
        """Test cost projection report structure."""
        generator = ReportGenerator(sample_results)
        report = generator.cost_projection_reports(sample_scenarios)

        # Verify top-level structure
        assert "optimal_configuration" in report
        assert "projections" in report
        assert "comparison_baseline" in report
        assert "savings_analysis" in report
        assert "recommendations" in report

        # Verify projections for each scenario
        projections = report["projections"]
        for scenario in sample_scenarios:
            scenario_name = scenario["name"]
            assert scenario_name in projections

            projection = projections[scenario_name]
            assert "cost_per_invocation" in projection
            assert "daily_cost" in projection
            assert "monthly_cost" in projection
            assert "yearly_cost" in projection
            assert "total_cost" in projection
            assert "pattern_impact" in projection

    def test_savings_calculation_accuracy(self, sample_results, sample_scenarios):
        """Test accuracy of savings calculations."""
        generator = ReportGenerator(sample_results)
        report = generator.cost_projection_reports(sample_scenarios)

        savings_analysis = report["savings_analysis"]
        scenario_savings = savings_analysis["scenario_savings"]

        for scenario_name, savings_data in scenario_savings.items():
            # Verify savings calculations
            assert "absolute_savings" in savings_data
            assert "percentage_savings" in savings_data
            assert "baseline_cost" in savings_data
            assert "optimized_cost" in savings_data

            # Verify mathematical consistency
            baseline = savings_data["baseline_cost"]
            optimized = savings_data["optimized_cost"]
            absolute_savings = savings_data["absolute_savings"]
            percentage_savings = savings_data["percentage_savings"]

            # Check absolute savings calculation
            assert abs(absolute_savings - (baseline - optimized)) < 0.001

            # Check percentage savings calculation
            if baseline > 0:
                expected_percentage = (absolute_savings / baseline) * 100
                assert abs(percentage_savings - expected_percentage) < 0.001

    def test_traffic_pattern_multipliers(self, sample_results):
        """Test traffic pattern cost multipliers."""
        generator = ReportGenerator(sample_results)
        optimal = generator._find_optimal_configuration()

        pattern_tests = [
            (TrafficPattern.STEADY, 1.0),
            (TrafficPattern.BURSTY, 1.2),
            (TrafficPattern.SEASONAL, 1.1),
            (TrafficPattern.GROWTH, 1.05),
        ]

        for pattern, expected_multiplier in pattern_tests:
            projection = generator._calculate_cost_projection(optimal, 1000, pattern, 30)

            assert projection["pattern_impact"] == expected_multiplier

            # Verify cost adjustment
            base_cost = optimal["avg_cost"]
            adjusted_cost = projection["cost_per_invocation"]
            assert abs(adjusted_cost - (base_cost * expected_multiplier)) < 0.000001

    def test_cost_recommendation_generation(self, sample_results, sample_scenarios):
        """Test cost recommendation generation."""
        generator = ReportGenerator(sample_results)
        report = generator.cost_projection_reports(sample_scenarios)

        recommendations = report["recommendations"]

        # Should generate cost recommendations
        assert len(recommendations) > 0

        # Each recommendation should have required fields
        for rec in recommendations:
            assert "priority" in rec
            assert "category" in rec
            assert "description" in rec
            assert rec["priority"] in ["high", "medium", "low"]

    def test_monthly_yearly_projections(self, sample_results):
        """Test monthly and yearly cost projections."""
        generator = ReportGenerator(sample_results)
        optimal = generator._find_optimal_configuration()

        # Test with known values
        daily_invocations = 1000
        pattern = TrafficPattern.STEADY
        duration_days = 30

        projection = generator._calculate_cost_projection(
            optimal, daily_invocations, pattern, duration_days
        )

        daily_cost = projection["daily_cost"]
        monthly_cost = projection["monthly_cost"]
        yearly_cost = projection["yearly_cost"]

        # Verify projections
        assert abs(monthly_cost - (daily_cost * 30)) < 0.001
        assert abs(yearly_cost - (daily_cost * 365)) < 0.001


class TestComparativeAnalysis:
    """Test comparative analysis features."""

    def test_cross_workload_insights(self, sample_results):
        """Test cross-workload insights generation."""
        generator = ReportGenerator(sample_results, None, WorkloadType.WEB_API)

        comparison_workloads = [
            {"type": "batch_processing", "name": "Batch Job", "results": sample_results},
            {"type": "event_driven", "name": "Event Handler", "results": sample_results},
        ]

        report = generator.comparative_analysis_reports(comparison_workloads)
        insights = report["cross_workload_insights"]

        # Should generate insights
        assert len(insights) > 0

        # Each insight should have category and insight
        for insight in insights:
            assert "category" in insight
            assert "insight" in insight

    def test_workload_comparison_structure(self, sample_results):
        """Test workload comparison report structure."""
        generator = ReportGenerator(sample_results, None, WorkloadType.WEB_API)

        comparison_workloads = [
            {"type": "batch_processing", "name": "Batch Workload", "results": sample_results}
        ]

        report = generator.comparative_analysis_reports(comparison_workloads)

        # Verify structure
        assert "current_workload" in report
        assert "comparisons" in report
        assert "cross_workload_insights" in report
        assert "best_practices" in report

        # Verify current workload
        current = report["current_workload"]
        assert current["type"] == "web_api"
        assert "analysis" in current

        # Verify comparisons
        comparisons = report["comparisons"]
        assert len(comparisons) == 1
        assert comparisons[0]["workload_type"] == "batch_processing"
        assert "analysis" in comparisons[0]

    def test_best_practices_generation(self, sample_results):
        """Test best practices generation."""
        generator = ReportGenerator(sample_results)

        # Create mock comparison data
        mock_comparisons = [
            {"workload_type": "web_api", "analysis": {"key_metrics": {"optimal_memory": 256}}},
            {
                "workload_type": "batch_processing",
                "analysis": {"key_metrics": {"optimal_memory": 1024}},
            },
        ]

        best_practices = generator._generate_workload_best_practices(mock_comparisons)

        # Should generate best practices
        assert len(best_practices) > 0

        # Each practice should have category and practice
        for practice in best_practices:
            assert "category" in practice
            assert "practice" in practice

        # Check for expected categories
        categories = [bp["category"] for bp in best_practices]
        expected_categories = ["memory_optimization", "monitoring", "cost_management"]
        for expected in expected_categories:
            assert expected in categories
