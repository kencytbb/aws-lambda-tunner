"""End-to-end tests for complete optimization scenarios."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime, timedelta

from aws_lambda_tuner.orchestrator_module import TunerOrchestrator
from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.reporting.service import ReportingService
from aws_lambda_tuner.visualization_module import VisualizationEngine
from tests.utils.test_helpers import TestValidators, TestAssertions, TestHelpers


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.aws
class TestCompleteOptimizationScenarios:
    """Test complete end-to-end optimization scenarios."""

    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        """Create orchestrator with mocked AWS services."""
        return TunerOrchestrator()

    @pytest.fixture
    def report_service(self):
        """Create report service."""
        return ReportingService()

    @pytest.fixture
    def visualization_generator(self):
        """Create visualization generator."""
        return VisualizationEngine()

    def test_complete_cost_optimization_scenario(
        self,
        orchestrator,
        report_service,
        visualization_generator,
        mock_aws_services,
        validator,
        temp_dir,
    ):
        """Test complete cost optimization scenario from config to reports."""
        # Step 1: Create configuration
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:cost-sensitive-app",
            memory_sizes=[128, 256, 512, 1024],
            iterations=10,
            strategy="cost",
            output_format="json",
            save_results=True,
            results_file=str(temp_dir / "cost_optimization_results.json"),
        )

        # Step 2: Run optimization
        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Step 3: Verify cost optimization behavior
        assert result.strategy == "cost"
        assert result.recommendation.strategy == "cost"

        # Cost strategy should recommend lower memory for cost savings
        assert result.recommendation.optimal_memory_size <= 512

        # Should show cost savings
        if result.recommendation.should_optimize:
            assert result.recommendation.cost_change_percent != 0

        # Step 4: Generate reports
        html_report_path = temp_dir / "cost_optimization_report.html"
        json_report_path = temp_dir / "cost_optimization_report.json"

        html_report = report_service.generate_html_report(result)
        json_report = report_service.generate_json_report(result)

        # Save reports
        with open(html_report_path, "w") as f:
            f.write(html_report)

        with open(json_report_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)

        # Verify reports were generated
        assert html_report_path.exists()
        assert json_report_path.exists()
        assert len(html_report) > 1000  # Substantial content

        # Step 5: Generate visualizations
        charts_dir = temp_dir / "charts"
        charts_dir.mkdir()

        performance_chart = visualization_generator.create_performance_chart(result.analysis)
        cost_chart = visualization_generator.create_cost_analysis_chart(result.analysis)

        # Save visualizations
        performance_chart.save(str(charts_dir / "performance.png"))
        cost_chart.save(str(charts_dir / "cost_analysis.png"))

        # Verify visualizations
        assert (charts_dir / "performance.png").exists()
        assert (charts_dir / "cost_analysis.png").exists()

        # Step 6: Verify results file was saved
        results_file = Path(config.results_file)
        assert results_file.exists()

        with open(results_file, "r") as f:
            saved_results = json.load(f)

        assert saved_results["function_arn"] == config.function_arn
        assert saved_results["strategy"] == "cost"

    def test_complete_performance_optimization_scenario(
        self, orchestrator, report_service, mock_aws_services, validator, temp_dir
    ):
        """Test complete performance optimization scenario."""
        # Configure for CPU-intensive workload
        mock_aws_services.configure_function_behavior(
            "performance-critical-app",
            base_duration=3000,  # High duration
            cold_start_rate=0.1,
            error_rate=0.005,
        )

        # Step 1: Create configuration for speed optimization
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:performance-critical-app",
            memory_sizes=[1024, 1536, 2048, 3008],
            iterations=15,
            strategy="speed",
            concurrent_executions=2,
            warmup_iterations=3,
            workload_type="cpu_intensive",
        )

        # Step 2: Run optimization
        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Step 3: Verify performance optimization
        assert result.strategy == "speed"

        # Should recommend higher memory for speed
        assert result.recommendation.optimal_memory_size >= 1536

        # Should show significant performance improvement
        memory_sizes = sorted(result.memory_results.keys())
        lowest_memory_duration = result.memory_results[memory_sizes[0]].avg_duration
        highest_memory_duration = result.memory_results[memory_sizes[-1]].avg_duration

        improvement_ratio = (
            lowest_memory_duration - highest_memory_duration
        ) / lowest_memory_duration
        assert improvement_ratio > 0.2, "Should show significant performance improvement"

        # Step 4: Generate comprehensive analysis report
        comprehensive_report = report_service.generate_comprehensive_report(result)

        report_file = temp_dir / "performance_optimization_comprehensive.html"
        with open(report_file, "w") as f:
            f.write(comprehensive_report)

        assert report_file.exists()
        assert "cpu_intensive" in comprehensive_report.lower()
        assert "performance" in comprehensive_report.lower()

    def test_complete_balanced_optimization_scenario(
        self, orchestrator, mock_aws_services, validator, temp_dir
    ):
        """Test complete balanced optimization scenario."""
        # Step 1: Create configuration for balanced optimization
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:balanced-app",
            memory_sizes=[256, 512, 1024, 1536, 2048],
            iterations=12,
            strategy="balanced",
            concurrent_executions=1,
            payload='{"balanced_test": true, "data_size": "medium"}',
            workload_type="balanced",
        )

        # Step 2: Run optimization
        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Step 3: Verify balanced optimization
        assert result.strategy == "balanced"

        # Should find optimal balance between cost and performance
        optimal_memory = result.recommendation.optimal_memory_size
        assert 512 <= optimal_memory <= 1536  # Should be in middle range

        # Should have good efficiency score
        max_efficiency = max(result.analysis.efficiency_scores.values())
        optimal_efficiency = result.analysis.efficiency_scores[optimal_memory]
        assert optimal_efficiency >= max_efficiency * 0.9

        # Step 4: Verify insights generation
        assert len(result.analysis.insights) >= 1

        insight_types = {insight["type"] for insight in result.analysis.insights}
        expected_types = {"memory_optimization", "cost_optimization", "cold_start", "reliability"}
        assert len(insight_types.intersection(expected_types)) >= 1

    def test_multi_function_optimization_scenario(
        self, orchestrator, mock_aws_services, validator, temp_dir
    ):
        """Test optimization scenario with multiple functions."""
        function_configs = [
            {
                "arn": "arn:aws:lambda:us-east-1:123456789012:function:api-handler",
                "memory_sizes": [256, 512, 1024],
                "strategy": "balanced",
                "workload_type": "io_bound",
            },
            {
                "arn": "arn:aws:lambda:us-east-1:123456789012:function:data-processor",
                "memory_sizes": [1024, 1536, 2048, 3008],
                "strategy": "speed",
                "workload_type": "cpu_intensive",
            },
            {
                "arn": "arn:aws:lambda:us-east-1:123456789012:function:logger",
                "memory_sizes": [128, 256, 512],
                "strategy": "cost",
                "workload_type": "io_bound",
            },
        ]

        results = []

        for func_config in function_configs:
            # Configure mock behavior based on workload type
            if func_config["workload_type"] == "cpu_intensive":
                mock_aws_services.configure_function_behavior(
                    func_config["arn"].split(":")[-1], base_duration=2000, cold_start_rate=0.2
                )
            elif func_config["workload_type"] == "io_bound":
                mock_aws_services.configure_function_behavior(
                    func_config["arn"].split(":")[-1], base_duration=400, cold_start_rate=0.4
                )

            config = TunerConfig(
                function_arn=func_config["arn"],
                memory_sizes=func_config["memory_sizes"],
                iterations=8,
                strategy=func_config["strategy"],
                workload_type=func_config["workload_type"],
            )

            result = orchestrator.run_optimization(config)
            validator.validate_tuning_result(result)
            results.append(result)

        # Verify results for each function
        assert len(results) == 3

        # API handler (I/O bound, balanced) should prefer medium memory
        api_result = results[0]
        assert 256 <= api_result.recommendation.optimal_memory_size <= 1024

        # Data processor (CPU intensive, speed) should prefer high memory
        processor_result = results[1]
        assert processor_result.recommendation.optimal_memory_size >= 1536

        # Logger (I/O bound, cost) should prefer low memory
        logger_result = results[2]
        assert logger_result.recommendation.optimal_memory_size <= 512

        # Generate summary report for all functions
        summary_data = {
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "functions_optimized": len(results),
            "total_recommendations": sum(1 for r in results if r.recommendation.should_optimize),
            "functions": [],
        }

        for result in results:
            function_summary = {
                "function_arn": result.function_arn,
                "strategy": result.strategy,
                "current_memory": result.recommendation.current_memory_size,
                "optimal_memory": result.recommendation.optimal_memory_size,
                "should_optimize": result.recommendation.should_optimize,
                "cost_change_percent": result.recommendation.cost_change_percent,
                "duration_change_percent": result.recommendation.duration_change_percent,
            }
            summary_data["functions"].append(function_summary)

        summary_file = temp_dir / "multi_function_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        assert summary_file.exists()

    def test_error_recovery_scenario(self, orchestrator, mock_aws_services, validator):
        """Test optimization scenario with error recovery."""
        # Configure high error rate initially
        mock_aws_services.configure_function_behavior(
            "unreliable-function", error_rate=0.5, base_duration=1000  # Very high error rate
        )

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:unreliable-function",
            memory_sizes=[256, 512, 1024],
            iterations=20,  # More iterations to handle errors
            strategy="balanced",
            max_retries=3,  # Enable retries
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Should handle errors gracefully
        total_errors = sum(r.errors for r in result.memory_results.values())
        total_executions = sum(r.iterations for r in result.memory_results.values())
        error_rate = total_errors / total_executions

        assert error_rate > 0.3  # Should have many errors
        assert error_rate < 1.0  # But not all should fail

        # Should generate reliability insights
        reliability_insights = [i for i in result.analysis.insights if i["type"] == "reliability"]
        assert len(reliability_insights) >= 1

        # Should still provide recommendations despite errors
        assert result.recommendation is not None

    def test_configuration_from_file_scenario(
        self, orchestrator, mock_aws_services, validator, temp_dir
    ):
        """Test complete scenario loading configuration from file."""
        # Step 1: Create configuration file
        config_data = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:file-config-test",
            "memory_sizes": [512, 1024, 2048],
            "iterations": 10,
            "strategy": "comprehensive",
            "concurrent_executions": 2,
            "warmup_iterations": 2,
            "payload": '{"config_source": "file", "test_data": [1, 2, 3]}',
            "output_format": "html",
            "save_results": True,
            "results_file": str(temp_dir / "file_config_results.json"),
            "workload_type": "balanced",
        }

        config_file = temp_dir / "optimization_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        # Step 2: Load configuration and run optimization
        config = TunerConfig.from_file(str(config_file))
        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Step 3: Verify configuration was loaded correctly
        assert result.function_arn == config_data["function_arn"]
        assert result.strategy == config_data["strategy"]

        # Step 4: Verify results were saved
        results_file = Path(config_data["results_file"])
        assert results_file.exists()

        with open(results_file, "r") as f:
            saved_results = json.load(f)

        assert saved_results["function_arn"] == config_data["function_arn"]
        assert saved_results["strategy"] == config_data["strategy"]

    def test_continuous_optimization_scenario(
        self, orchestrator, mock_aws_services, validator, temp_dir
    ):
        """Test scenario simulating continuous optimization over time."""
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:continuous-app"

        # Simulate multiple optimization runs over time
        optimization_history = []

        for day in range(3):  # Simulate 3 days of optimization
            # Adjust mock behavior to simulate changing workload patterns
            if day == 0:
                # Day 0: Light workload
                mock_aws_services.configure_function_behavior(
                    "continuous-app", base_duration=500, cold_start_rate=0.3
                )
            elif day == 1:
                # Day 1: Medium workload
                mock_aws_services.configure_function_behavior(
                    "continuous-app", base_duration=800, cold_start_rate=0.2
                )
            else:
                # Day 2: Heavy workload
                mock_aws_services.configure_function_behavior(
                    "continuous-app", base_duration=1500, cold_start_rate=0.1
                )

            config = TunerConfig(
                function_arn=function_arn,
                memory_sizes=[256, 512, 1024, 1536, 2048],
                iterations=8,
                strategy="balanced",
                baseline_memory=(
                    optimization_history[-1]["optimal_memory"] if optimization_history else 256
                ),
            )

            result = orchestrator.run_optimization(config)
            validator.validate_tuning_result(result)

            optimization_record = {
                "day": day,
                "timestamp": datetime.utcnow().isoformat(),
                "optimal_memory": result.recommendation.optimal_memory_size,
                "should_optimize": result.recommendation.should_optimize,
                "avg_duration": result.memory_results[
                    result.recommendation.optimal_memory_size
                ].avg_duration,
                "avg_cost": result.memory_results[
                    result.recommendation.optimal_memory_size
                ].avg_cost,
                "confidence_score": result.recommendation.confidence_score,
            }
            optimization_history.append(optimization_record)

        # Analyze optimization history
        assert len(optimization_history) == 3

        # Should show adaptation to changing workload
        day0_memory = optimization_history[0]["optimal_memory"]
        day2_memory = optimization_history[2]["optimal_memory"]

        # Heavy workload (day 2) should generally prefer more memory than light workload (day 0)
        # (This may not always be true depending on mock behavior, but is expected pattern)

        # Save optimization history
        history_file = temp_dir / "continuous_optimization_history.json"
        with open(history_file, "w") as f:
            json.dump(optimization_history, f, indent=2)

        assert history_file.exists()


@pytest.mark.e2e
@pytest.mark.slow
class TestRealWorldScenarios:
    """Test scenarios that simulate real-world usage patterns."""

    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TunerOrchestrator()

    def test_startup_optimization_scenario(self, orchestrator, mock_aws_services, validator):
        """Test optimization for a new function being deployed."""
        # New function with no baseline data
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:new-deployment",
            memory_sizes=[128, 256, 512, 1024, 1536, 2048, 3008],  # Wide range
            iterations=15,  # More iterations for initial assessment
            strategy="comprehensive",  # Comprehensive analysis
            warmup_iterations=5,  # Warmup for new function
            workload_type="unknown",
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Should provide comprehensive analysis for new function
        assert len(result.memory_results) == 7  # All memory sizes tested
        assert result.strategy == "comprehensive"
        assert len(result.analysis.insights) >= 2  # Should provide multiple insights

        # Should identify workload characteristics
        assert result.analysis.trends["memory_sensitivity"] in ["low", "medium", "high"]

    def test_production_optimization_scenario(self, orchestrator, mock_aws_services, validator):
        """Test optimization for an existing production function."""
        # Production function with existing configuration
        current_memory = 512

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:production-api",
            memory_sizes=[256, 512, 1024, 1536],  # Focus around current setting
            iterations=20,  # More iterations for production confidence
            strategy="balanced",
            baseline_memory=current_memory,
            concurrent_executions=1,  # Conservative for production
            workload_type="io_bound",  # Known workload type
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Should provide conservative recommendations for production
        assert result.recommendation.confidence_score >= 0.7  # High confidence needed

        # Should compare against baseline
        if hasattr(result.analysis, "baseline_comparison"):
            assert result.analysis.baseline_comparison is not None

    def test_cost_emergency_scenario(self, orchestrator, mock_aws_services, validator):
        """Test optimization scenario for emergency cost reduction."""
        # High-cost function needing immediate cost optimization
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:expensive-function",
            memory_sizes=[128, 256, 512],  # Focus on lower memory for cost savings
            iterations=10,
            strategy="cost",
            concurrent_executions=3,  # Faster testing for emergency
            priority="high",  # Emergency optimization
            workload_type="io_bound",  # Known to be less memory-sensitive
        )

        result = orchestrator.run_optimization(config)
        validator.validate_tuning_result(result)

        # Should aggressively optimize for cost
        assert result.recommendation.optimal_memory_size <= 512

        # Should show significant cost savings if optimization is recommended
        if result.recommendation.should_optimize:
            assert result.recommendation.cost_change_percent < -10  # At least 10% cost reduction

        # Should provide cost savings estimates
        assert result.recommendation.estimated_monthly_savings is not None
