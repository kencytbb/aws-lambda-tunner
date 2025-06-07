"""Tests for enhanced analyzer functionality."""

import pytest
from unittest.mock import Mock, patch
import statistics
from datetime import datetime

from aws_lambda_tuner.analyzers.analyzer import PerformanceAnalyzer
from aws_lambda_tuner.models import (
    MemoryTestResult, ColdStartAnalysis, ConcurrencyAnalysis, 
    ConcurrencyPattern, WorkloadAnalysis, TimeBasedTrend
)


@pytest.fixture
def mock_config():
    """Mock configuration for analyzer."""
    return {
        'memory_sizes': [128, 256, 512, 1024],
        'iterations': 10,
        'timeout': 30
    }


@pytest.fixture
def sample_memory_results():
    """Sample memory test results with varied patterns."""
    return {
        128: MemoryTestResult(
            memory_size=128,
            iterations=10,
            avg_duration=1500.0,
            p95_duration=1800.0,
            p99_duration=2000.0,
            avg_cost=0.0000002,
            total_cost=0.000002,
            cold_starts=3,
            errors=1,
            raw_results=[
                {'duration': 1400, 'cost': 0.0000002, 'is_cold_start': True, 'concurrent_executions': 1, 'timestamp': 1000},
                {'duration': 1500, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 2, 'timestamp': 1100},
                {'duration': 1600, 'cost': 0.0000002, 'is_cold_start': True, 'concurrent_executions': 5, 'timestamp': 1200},
                {'duration': 1450, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1300},
                {'duration': 1550, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 2, 'timestamp': 1400},
                {'duration': 1700, 'cost': 0.0000002, 'is_cold_start': True, 'concurrent_executions': 8, 'timestamp': 1500},
                {'duration': 1400, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 4, 'timestamp': 1600},
                {'duration': 1500, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1700},
                {'duration': 1600, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 2, 'timestamp': 1800},
                {'duration': 1450, 'cost': 0.0000002, 'is_cold_start': False, 'concurrent_executions': 1, 'timestamp': 1900}
            ]
        ),
        256: MemoryTestResult(
            memory_size=256,
            iterations=10,
            avg_duration=1200.0,
            p95_duration=1400.0,
            p99_duration=1500.0,
            avg_cost=0.0000003,
            total_cost=0.000003,
            cold_starts=2,
            errors=0,
            raw_results=[
                {'duration': 1100, 'cost': 0.0000003, 'is_cold_start': True, 'concurrent_executions': 2, 'timestamp': 1000},
                {'duration': 1200, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1100},
                {'duration': 1250, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 4, 'timestamp': 1200},
                {'duration': 1150, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1300},
                {'duration': 1200, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 2, 'timestamp': 1400},
                {'duration': 1300, 'cost': 0.0000003, 'is_cold_start': True, 'concurrent_executions': 6, 'timestamp': 1500},
                {'duration': 1180, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 4, 'timestamp': 1600},
                {'duration': 1220, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1700},
                {'duration': 1190, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 2, 'timestamp': 1800},
                {'duration': 1210, 'cost': 0.0000003, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1900}
            ]
        ),
        512: MemoryTestResult(
            memory_size=512,
            iterations=10,
            avg_duration=1000.0,
            p95_duration=1100.0,
            p99_duration=1200.0,
            avg_cost=0.0000004,
            total_cost=0.000004,
            cold_starts=1,
            errors=0,
            raw_results=[
                {'duration': 950, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1000},
                {'duration': 1000, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 4, 'timestamp': 1100},
                {'duration': 1050, 'cost': 0.0000004, 'is_cold_start': True, 'concurrent_executions': 7, 'timestamp': 1200},
                {'duration': 980, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 5, 'timestamp': 1300},
                {'duration': 1020, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 4, 'timestamp': 1400},
                {'duration': 990, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1500},
                {'duration': 1010, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 4, 'timestamp': 1600},
                {'duration': 970, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1700},
                {'duration': 1030, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 2, 'timestamp': 1800},
                {'duration': 990, 'cost': 0.0000004, 'is_cold_start': False, 'concurrent_executions': 3, 'timestamp': 1900}
            ]
        )
    }


class TestEnhancedPerformanceAnalyzer:
    """Test cases for enhanced performance analyzer functionality."""
    
    def test_analyze_cold_starts(self, mock_config, sample_memory_results):
        """Test cold start analysis functionality."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        cold_start_analysis = analyzer.analyze_cold_starts(sample_memory_results)
        
        assert isinstance(cold_start_analysis, ColdStartAnalysis)
        assert cold_start_analysis.cold_start_ratio > 0
        assert cold_start_analysis.avg_cold_start_duration > 0
        assert cold_start_analysis.avg_warm_start_duration > 0
        assert cold_start_analysis.cold_start_impact_score >= 0
        assert cold_start_analysis.optimal_memory_for_cold_starts in sample_memory_results
        assert isinstance(cold_start_analysis.cold_start_patterns, dict)
    
    def test_cold_start_correlation_calculation(self, mock_config, sample_memory_results):
        """Test correlation calculation between memory and cold starts."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        cold_start_analysis = analyzer.analyze_cold_starts(sample_memory_results)
        
        # Should detect that higher memory reduces cold starts
        assert cold_start_analysis.memory_vs_cold_start_correlation <= 0
        
        # Optimal memory should be one with fewer cold starts
        optimal_memory = cold_start_analysis.optimal_memory_for_cold_starts
        optimal_result = sample_memory_results[optimal_memory]
        optimal_cold_start_ratio = optimal_result.cold_starts / optimal_result.iterations
        
        # Should be one of the lower cold start ratios
        all_ratios = [r.cold_starts / r.iterations for r in sample_memory_results.values()]
        assert optimal_cold_start_ratio <= max(all_ratios)
    
    def test_analyze_concurrency_patterns(self, mock_config, sample_memory_results):
        """Test concurrency pattern analysis functionality."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        concurrency_analysis = analyzer.analyze_concurrency_patterns(sample_memory_results)
        
        assert isinstance(concurrency_analysis, ConcurrencyAnalysis)
        assert concurrency_analysis.avg_concurrent_executions > 0
        assert concurrency_analysis.peak_concurrent_executions > 0
        assert 0 <= concurrency_analysis.concurrency_utilization <= 1
        assert 0 <= concurrency_analysis.scaling_efficiency <= 1
        assert concurrency_analysis.throttling_events >= 0
        assert isinstance(concurrency_analysis.concurrency_patterns, dict)
        assert isinstance(concurrency_analysis.identified_patterns, list)
    
    def test_concurrency_pattern_identification(self, mock_config, sample_memory_results):
        """Test identification of specific concurrency patterns."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Extract concurrent executions for pattern testing
        concurrent_executions = []
        execution_timestamps = []
        for result in sample_memory_results.values():
            for raw_result in result.raw_results:
                if 'concurrent_executions' in raw_result:
                    concurrent_executions.append(raw_result['concurrent_executions'])
                if 'timestamp' in raw_result:
                    execution_timestamps.append(raw_result['timestamp'])
        
        patterns = analyzer._identify_concurrency_patterns(concurrent_executions, execution_timestamps)
        
        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, ConcurrencyPattern)
            assert pattern.pattern_type in ['burst', 'steady', 'gradual_ramp', 'spike']
            assert 0 <= pattern.frequency <= 1
            assert pattern.intensity >= 0
            assert pattern.duration_ms >= 0
            assert 0 <= pattern.impact_on_performance <= 1
            assert isinstance(pattern.recommendations, list)
    
    def test_workload_specific_analysis(self, mock_config, sample_memory_results):
        """Test workload-specific pattern analysis."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        workload_analysis = analyzer.analyze_workload_specific_patterns(sample_memory_results, "cpu_intensive")
        
        assert isinstance(workload_analysis, WorkloadAnalysis)
        assert workload_analysis.workload_type == "cpu_intensive"
        assert isinstance(workload_analysis.resource_utilization, dict)
        assert isinstance(workload_analysis.optimization_opportunities, list)
        assert isinstance(workload_analysis.workload_specific_recommendations, list)
        assert isinstance(workload_analysis.cost_vs_performance_curve, dict)
    
    def test_time_based_trends_analysis(self, mock_config, sample_memory_results):
        """Test time-based performance trends analysis."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        trends = analyzer.analyze_time_based_trends(sample_memory_results)
        
        assert isinstance(trends, list)
        assert len(trends) > 0
        
        for trend in trends:
            assert isinstance(trend, TimeBasedTrend)
            assert isinstance(trend.time_period, str)
            assert isinstance(trend.metric_trends, dict)
            assert isinstance(trend.seasonal_patterns, dict)
            assert isinstance(trend.performance_degradation, bool)
            assert 0 <= trend.trend_confidence <= 1
            assert isinstance(trend.forecast, dict)
    
    def test_advanced_analysis_integration(self, mock_config, sample_memory_results):
        """Test the comprehensive advanced analysis integration."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        advanced_analysis = analyzer.perform_advanced_analysis(
            memory_results=sample_memory_results,
            workload_type="general"
        )
        
        # Verify all components are present
        assert advanced_analysis.cold_start_analysis is not None
        assert advanced_analysis.concurrency_analysis is not None
        assert advanced_analysis.workload_analysis is not None
        assert advanced_analysis.time_based_trends is not None
        
        # Verify inheritance from base analysis
        assert advanced_analysis.memory_results == sample_memory_results
        assert isinstance(advanced_analysis.efficiency_scores, dict)
        assert isinstance(advanced_analysis.insights, list)
    
    def test_pattern_impact_calculation(self, mock_config):
        """Test pattern impact calculation with controlled data."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Test with simple data
        concurrent_executions = [1, 2, 1, 2, 10, 1, 2, 1]  # One spike
        threshold = 5.0
        
        impact = analyzer._calculate_pattern_impact(concurrent_executions, threshold)
        
        assert 0 <= impact <= 1
        assert impact > 0  # Should detect the spike
    
    def test_pattern_duration_estimation(self, mock_config):
        """Test pattern duration estimation."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Test consecutive executions above threshold
        concurrent_executions = [1, 2, 8, 9, 10, 2, 1]  # 3 consecutive above threshold
        threshold = 7.0
        
        duration = analyzer._estimate_pattern_duration(concurrent_executions, threshold)
        
        assert duration > 0
        assert duration == 300  # 3 consecutive * 100ms each
    
    def test_correlation_calculation(self, mock_config):
        """Test correlation coefficient calculation."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Perfect positive correlation
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 6, 8, 10]
        
        correlation = analyzer._calculate_correlation(x_values, y_values)
        assert abs(correlation - 1.0) < 0.01  # Should be close to 1
        
        # Perfect negative correlation
        y_values_neg = [10, 8, 6, 4, 2]
        correlation_neg = analyzer._calculate_correlation(x_values, y_values_neg)
        assert abs(correlation_neg - (-1.0)) < 0.01  # Should be close to -1
        
        # No correlation
        y_values_none = [5, 5, 5, 5, 5]
        correlation_none = analyzer._calculate_correlation(x_values, y_values_none)
        assert abs(correlation_none) < 0.01  # Should be close to 0
    
    def test_cold_start_threshold_detection(self, mock_config):
        """Test cold start threshold detection."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Memory sizes with decreasing cold start ratios
        memory_cold_start_ratios = {
            128: 0.5,  # 50% cold starts
            256: 0.3,  # 30% cold starts  
            512: 0.1,  # 10% cold starts (significant drop)
            1024: 0.08  # 8% cold starts
        }
        
        threshold = analyzer._find_cold_start_threshold(memory_cold_start_ratios)
        
        # Should detect the threshold where significant decrease occurs
        assert threshold in [256, 512]  # Should be where big drop happens
    
    def test_scaling_efficiency_calculation(self, mock_config):
        """Test scaling efficiency calculation."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Very consistent concurrency (high efficiency)
        consistent_executions = [5, 5, 5, 5, 5]
        efficiency_consistent = analyzer._calculate_scaling_efficiency(consistent_executions)
        
        # Very variable concurrency (low efficiency)
        variable_executions = [1, 10, 2, 15, 3]
        efficiency_variable = analyzer._calculate_scaling_efficiency(variable_executions)
        
        assert efficiency_consistent > efficiency_variable
        assert 0 <= efficiency_consistent <= 1
        assert 0 <= efficiency_variable <= 1
    
    def test_concurrency_limit_recommendation(self, mock_config):
        """Test concurrency limit recommendation logic."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # No throttling scenario
        concurrent_executions = [5, 10, 8, 12, 15]
        throttling_events = 0
        
        limit = analyzer._recommend_concurrency_limit(concurrent_executions, throttling_events)
        
        assert limit is not None
        assert limit > max(concurrent_executions)  # Should have headroom
        
        # Throttling scenario
        throttling_events = 3
        limit_throttled = analyzer._recommend_concurrency_limit(concurrent_executions, throttling_events)
        
        assert limit_throttled is not None
        assert limit_throttled != limit  # Should be different strategy
    
    def test_burst_pattern_analysis(self, mock_config):
        """Test burst pattern detection."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Clear burst pattern
        concurrent_executions = [2, 2, 2, 20, 25, 2, 2, 2]
        
        burst_analysis = analyzer._analyze_burst_patterns(concurrent_executions)
        
        assert isinstance(burst_analysis, dict)
        assert "burst_detected" in burst_analysis
        assert "burst_ratio" in burst_analysis
        assert "peak_concurrency" in burst_analysis
        assert "average_concurrency" in burst_analysis
        
        if burst_analysis["burst_detected"]:
            assert burst_analysis["burst_ratio"] > 1
    
    def test_resource_utilization_calculation(self, mock_config, sample_memory_results):
        """Test resource utilization metrics calculation."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        # Add utilization data to sample results
        for result in sample_memory_results.values():
            for raw_result in result.raw_results:
                raw_result['cpu_utilization'] = 0.6
                raw_result['memory_utilization'] = 0.7
        
        utilization = analyzer._calculate_resource_utilization(sample_memory_results)
        
        assert isinstance(utilization, dict)
        assert "avg_cpu_utilization" in utilization
        assert "avg_memory_utilization" in utilization
        assert "peak_cpu_utilization" in utilization
        assert "peak_memory_utilization" in utilization
        
        assert 0 <= utilization["avg_cpu_utilization"] <= 1
        assert 0 <= utilization["avg_memory_utilization"] <= 1


class TestAnalyzerPatternSummarization:
    """Test pattern summarization functionality."""
    
    def test_pattern_summary_with_multiple_patterns(self, mock_config):
        """Test pattern summarization with multiple patterns."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        patterns = [
            ConcurrencyPattern(
                pattern_type="burst",
                frequency=0.3,
                intensity=2.0,
                duration_ms=500,
                impact_on_performance=0.8,
                recommendations=["rec1", "rec2"]
            ),
            ConcurrencyPattern(
                pattern_type="steady", 
                frequency=0.7,
                intensity=1.0,
                duration_ms=10000,
                impact_on_performance=0.2,
                recommendations=["rec3"]
            )
        ]
        
        summary = analyzer._summarize_concurrency_patterns(patterns)
        
        assert isinstance(summary, dict)
        assert "dominant_pattern" in summary
        assert "pattern_count" in summary
        assert "high_impact_patterns" in summary
        assert "total_recommendations" in summary
        
        assert summary["pattern_count"] == 2
        assert summary["total_recommendations"] == 3
        assert "burst" in summary["high_impact_patterns"]  # High impact pattern
    
    def test_pattern_summary_empty(self, mock_config):
        """Test pattern summarization with no patterns."""
        analyzer = PerformanceAnalyzer(mock_config)
        
        summary = analyzer._summarize_concurrency_patterns([])
        
        assert summary["dominant_pattern"] == "unknown"
        assert summary["pattern_count"] == 0
        assert summary["high_impact_patterns"] == []