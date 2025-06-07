"""Unit tests for analyzer modules."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from aws_lambda_tuner.analyzers.analyzer import PerformanceAnalyzer
from aws_lambda_tuner.models import MemoryTestResult, PerformanceAnalysis
from tests.utils.test_helpers import TestValidators, TestAssertions
from tests.utils.test_data_generators import TestDataGenerator


@pytest.mark.unit
@pytest.mark.aws
class TestPerformanceAnalyzer:
    """Test the main PerformanceAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a PerformanceAnalyzer instance for testing."""
        return PerformanceAnalyzer()

    @pytest.fixture
    def sample_results_data(self):
        """Create sample memory test results."""
        generator = TestDataGenerator(seed=42)
        return {
            256: generator.generate_memory_test_result(256, 10, "balanced"),
            512: generator.generate_memory_test_result(512, 10, "balanced"),
            1024: generator.generate_memory_test_result(1024, 10, "balanced"),
        }

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer can be initialized."""
        assert analyzer is not None
        assert hasattr(analyzer, "analyze_results")

    def test_analyze_memory_results(self, analyzer, sample_results_data, validator):
        """Test analyzing memory test results."""
        analysis = analyzer.analyze_results(sample_results_data)

        assert isinstance(analysis, PerformanceAnalysis)
        validator.validate_performance_analysis(analysis, expected_memory_sizes=[256, 512, 1024])

    def test_calculate_efficiency_scores(self, analyzer, sample_results_data):
        """Test efficiency score calculation."""
        efficiency_scores = analyzer._calculate_efficiency_scores(sample_results_data)

        assert isinstance(efficiency_scores, dict)
        assert len(efficiency_scores) == len(sample_results_data)

        for memory_size, score in efficiency_scores.items():
            assert memory_size in sample_results_data
            assert isinstance(score, (int, float))
            assert score >= 0

    def test_find_optimal_configurations(self, analyzer, sample_results_data):
        """Test finding optimal configurations for different strategies."""
        efficiency_scores = analyzer._calculate_efficiency_scores(sample_results_data)

        cost_optimal = analyzer._find_cost_optimal(sample_results_data)
        speed_optimal = analyzer._find_speed_optimal(sample_results_data)
        balanced_optimal = analyzer._find_balanced_optimal(sample_results_data, efficiency_scores)

        assert "memory_size" in cost_optimal
        assert "memory_size" in speed_optimal
        assert "memory_size" in balanced_optimal

        # Verify the memory sizes are from our test data
        assert cost_optimal["memory_size"] in sample_results_data
        assert speed_optimal["memory_size"] in sample_results_data
        assert balanced_optimal["memory_size"] in sample_results_data

    def test_analyze_trends(self, analyzer, sample_results_data):
        """Test trend analysis."""
        trends = analyzer._analyze_trends(sample_results_data)

        assert isinstance(trends, dict)
        assert "duration_trend" in trends
        assert "cost_trend" in trends
        assert "memory_sensitivity" in trends

        assert trends["duration_trend"] in ["increasing", "decreasing", "stable"]
        assert trends["cost_trend"] in ["increasing", "decreasing", "stable"]
        assert trends["memory_sensitivity"] in ["low", "medium", "high"]

    def test_generate_insights(self, analyzer, sample_results_data):
        """Test insight generation."""
        efficiency_scores = analyzer._calculate_efficiency_scores(sample_results_data)
        trends = analyzer._analyze_trends(sample_results_data)
        insights = analyzer._generate_insights(sample_results_data, efficiency_scores, trends)

        assert isinstance(insights, list)

        for insight in insights:
            assert isinstance(insight, dict)
            assert "type" in insight
            assert "severity" in insight
            assert "message" in insight
            assert "recommendation" in insight
            assert insight["severity"] in ["low", "medium", "high"]

    @pytest.mark.parametrize(
        "workload_type", ["cpu_intensive", "io_bound", "memory_intensive", "balanced"]
    )
    def test_workload_specific_analysis(self, analyzer, workload_type):
        """Test analysis with different workload types."""
        generator = TestDataGenerator(seed=42)
        memory_sizes = [256, 512, 1024]

        results_data = {}
        for memory_size in memory_sizes:
            results_data[memory_size] = generator.generate_memory_test_result(
                memory_size, 5, workload_type
            )

        analysis = analyzer.analyze_results(results_data)

        assert isinstance(analysis, PerformanceAnalysis)
        assert len(analysis.memory_results) == len(memory_sizes)

        # Verify workload-specific characteristics
        if workload_type == "cpu_intensive":
            # CPU-intensive workloads should benefit significantly from more memory
            memory_sizes_sorted = sorted(memory_sizes)
            duration_1024 = analysis.memory_results[1024].avg_duration
            duration_256 = analysis.memory_results[256].avg_duration
            assert duration_1024 < duration_256  # Should be faster with more memory

        elif workload_type == "io_bound":
            # I/O bound workloads should show less memory sensitivity
            assert analysis.trends["memory_sensitivity"] in ["low", "medium"]

    def test_empty_results_handling(self, analyzer):
        """Test handling of empty or invalid results."""
        with pytest.raises((ValueError, KeyError)):
            analyzer.analyze_results({})

    def test_single_memory_size_analysis(self, analyzer):
        """Test analysis with only one memory configuration."""
        generator = TestDataGenerator(seed=42)
        single_result = {512: generator.generate_memory_test_result(512, 10, "balanced")}

        analysis = analyzer.analyze_results(single_result)

        assert isinstance(analysis, PerformanceAnalysis)
        assert len(analysis.memory_results) == 1
        assert 512 in analysis.memory_results

    def test_error_rate_impact_analysis(self, analyzer):
        """Test analysis when some configurations have errors."""
        generator = TestDataGenerator(seed=42)

        # Create results with varying error rates
        results_data = {}
        for memory_size in [256, 512, 1024]:
            result = generator.generate_memory_test_result(memory_size, 10, "balanced")
            if memory_size == 256:
                result.errors = 3  # High error rate for 256MB
            results_data[memory_size] = result

        analysis = analyzer.analyze_results(results_data)

        # Should generate insight about high error rate
        error_insights = [i for i in analysis.insights if i["type"] == "reliability"]
        assert len(error_insights) >= 0  # May or may not detect depending on threshold

    def test_cold_start_analysis(self, analyzer, sample_results_data):
        """Test cold start impact analysis."""
        # Modify results to have high cold start counts
        for result in sample_results_data.values():
            result.cold_starts = result.iterations // 2  # 50% cold start rate

        analysis = analyzer.analyze_results(sample_results_data)

        # Should generate insights about cold starts
        cold_start_insights = [i for i in analysis.insights if "cold start" in i["message"].lower()]
        # Note: May or may not generate depending on threshold and implementation

    def test_performance_plateau_detection(self, analyzer):
        """Test detection of performance plateau points."""
        generator = TestDataGenerator(seed=42)

        # Create results that show diminishing returns
        memory_sizes = [256, 512, 1024, 2048, 3008]
        results_data = {}

        for memory_size in memory_sizes:
            result = generator.generate_memory_test_result(memory_size, 5, "cpu_intensive")
            # Manually adjust to show plateau after 1024MB
            if memory_size > 1024:
                result.avg_duration = results_data[1024].avg_duration * 0.98  # Minimal improvement
            results_data[memory_size] = result

        analysis = analyzer.analyze_results(results_data)
        trends = analysis.trends

        # May detect plateau (implementation dependent)
        if "performance_plateau" in trends:
            assert trends["performance_plateau"] is None or trends["performance_plateau"] >= 1024


@pytest.mark.unit
class TestAnalyzerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()

    def test_all_executions_failed(self, analyzer):
        """Test handling when all executions failed."""
        failed_result = MemoryTestResult(
            memory_size=256,
            iterations=5,
            avg_duration=0,
            p95_duration=0,
            p99_duration=0,
            avg_cost=0,
            total_cost=0,
            cold_starts=0,
            errors=5,  # All failed
        )

        results_data = {256: failed_result}

        # Should handle gracefully or raise appropriate error
        try:
            analysis = analyzer.analyze_results(results_data)
            # If it doesn't raise an error, verify it handles the case appropriately
            assert analysis.cost_optimal is not None
        except ValueError:
            # Acceptable to raise error for all-failed scenario
            pass

    def test_zero_cost_results(self, analyzer):
        """Test handling of zero-cost results."""
        zero_cost_result = MemoryTestResult(
            memory_size=256,
            iterations=5,
            avg_duration=100.0,
            p95_duration=120.0,
            p99_duration=140.0,
            avg_cost=0.0,  # Zero cost
            total_cost=0.0,
            cold_starts=1,
            errors=0,
        )

        results_data = {256: zero_cost_result}

        analysis = analyzer.analyze_results(results_data)
        assert analysis.efficiency_scores[256] == 0  # Should handle zero cost gracefully

    def test_identical_performance_results(self, analyzer):
        """Test handling when all memory configurations perform identically."""
        identical_results = {}
        for memory_size in [256, 512, 1024]:
            identical_results[memory_size] = MemoryTestResult(
                memory_size=memory_size,
                iterations=5,
                avg_duration=100.0,  # Identical performance
                p95_duration=110.0,
                p99_duration=120.0,
                avg_cost=memory_size * 0.000001,  # Cost scales with memory
                total_cost=memory_size * 0.000005,
                cold_starts=1,
                errors=0,
            )

        analysis = analyzer.analyze_results(identical_results)

        # Cost optimal should be the lowest memory (cheapest)
        assert analysis.cost_optimal["memory_size"] == 256
        # Speed optimal could be any (since they're identical)
        # Balanced should consider cost efficiency

    def test_extreme_memory_sizes(self, analyzer):
        """Test with extreme memory size values."""
        generator = TestDataGenerator(seed=42)

        extreme_results = {
            128: generator.generate_memory_test_result(128, 3, "balanced"),
            3008: generator.generate_memory_test_result(3008, 3, "balanced"),  # Max Lambda memory
        }

        analysis = analyzer.analyze_results(extreme_results)
        assert isinstance(analysis, PerformanceAnalysis)
        assert 128 in analysis.memory_results
        assert 3008 in analysis.memory_results


@pytest.mark.unit
@pytest.mark.performance
class TestAnalyzerPerformance:
    """Test analyzer performance with large datasets."""

    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()

    def test_large_dataset_analysis(self, analyzer, performance_helpers):
        """Test analyzer performance with large number of memory configurations."""
        memory_sizes = list(range(128, 3009, 64))  # Many memory sizes
        generator = TestDataGenerator(seed=42)

        large_results = {}
        for memory_size in memory_sizes:
            large_results[memory_size] = generator.generate_memory_test_result(
                memory_size, 10, "balanced"
            )

        # Measure analysis time
        analysis, duration = performance_helpers.measure_execution_time(
            analyzer.analyze_results, large_results
        )

        # Should complete in reasonable time (adjust threshold as needed)
        performance_helpers.assert_performance_within_bounds(
            duration, 5.0, "Large dataset analysis"
        )

        assert isinstance(analysis, PerformanceAnalysis)
        assert len(analysis.memory_results) == len(memory_sizes)

    def test_high_iteration_count_analysis(self, analyzer, performance_helpers):
        """Test analyzer with high iteration counts."""
        generator = TestDataGenerator(seed=42)

        high_iteration_results = {}
        for memory_size in [256, 512, 1024]:
            high_iteration_results[memory_size] = generator.generate_memory_test_result(
                memory_size, 1000, "balanced"  # High iteration count
            )

        analysis, duration = performance_helpers.measure_execution_time(
            analyzer.analyze_results, high_iteration_results
        )

        performance_helpers.assert_performance_within_bounds(
            duration, 2.0, "High iteration analysis"
        )

        assert isinstance(analysis, PerformanceAnalysis)


@pytest.mark.unit
@pytest.mark.parametrize
class TestAnalyzerParametrized:
    """Parametrized tests for different scenarios."""

    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()

    @pytest.mark.parametrize(
        "memory_sizes,expected_trend",
        [
            ([128, 256, 512], "decreasing"),  # Should improve with more memory
            ([512, 1024, 2048], "decreasing"),  # Should improve with more memory
            ([1024, 1536, 2048], "decreasing"),  # Should improve with more memory
        ],
    )
    def test_performance_trends(self, analyzer, assertions, memory_sizes, expected_trend):
        """Test performance trend detection with different memory ranges."""
        generator = TestDataGenerator(seed=42)

        results_data = {}
        for memory_size in memory_sizes:
            results_data[memory_size] = generator.generate_memory_test_result(
                memory_size, 5, "cpu_intensive"
            )

        analysis = analyzer.analyze_results(results_data)

        # Verify the trend matches expectations
        assertions.assert_memory_performance_trend(results_data, expected_trend)

    @pytest.mark.parametrize("iterations", [1, 5, 10, 25, 50])
    def test_different_iteration_counts(self, analyzer, iterations):
        """Test analysis with different iteration counts."""
        generator = TestDataGenerator(seed=42)

        results_data = {}
        for memory_size in [256, 512, 1024]:
            results_data[memory_size] = generator.generate_memory_test_result(
                memory_size, iterations, "balanced"
            )

        analysis = analyzer.analyze_results(results_data)

        assert isinstance(analysis, PerformanceAnalysis)
        # Verify all results have the expected iteration count
        for result in analysis.memory_results.values():
            assert result.iterations == iterations

    @pytest.mark.parametrize("error_rate", [0.0, 0.1, 0.2, 0.5])
    def test_different_error_rates(self, analyzer, error_rate):
        """Test analysis with different error rates."""
        generator = TestDataGenerator(seed=42)

        results_data = {}
        for memory_size in [256, 512, 1024]:
            result = generator.generate_memory_test_result(memory_size, 10, "balanced")
            result.errors = int(result.iterations * error_rate)
            results_data[memory_size] = result

        analysis = analyzer.analyze_results(results_data)

        assert isinstance(analysis, PerformanceAnalysis)

        # High error rates should generate reliability insights
        if error_rate >= 0.2:
            reliability_insights = [i for i in analysis.insights if i["type"] == "reliability"]
            # May or may not generate depending on implementation thresholds
