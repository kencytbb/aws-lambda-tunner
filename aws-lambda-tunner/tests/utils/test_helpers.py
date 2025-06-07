"""Test helper functions for assertions and data validation."""

import json
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from unittest.mock import Mock

import pytest

from aws_lambda_tuner.models import (
    MemoryTestResult,
    Recommendation,
    PerformanceAnalysis,
    TuningResult,
    ColdStartAnalysis,
    ConcurrencyAnalysis,
    WorkloadAnalysis
)


class TestValidators:
    """Collection of validation functions for test assertions."""
    
    @staticmethod
    def validate_memory_test_result(result: MemoryTestResult, 
                                  expected_memory_size: int = None,
                                  min_iterations: int = 1) -> bool:
        """Validate MemoryTestResult structure and values."""
        assert isinstance(result, MemoryTestResult), f"Expected MemoryTestResult, got {type(result)}"
        assert result.memory_size > 0, "Memory size must be positive"
        assert result.iterations >= min_iterations, f"Iterations must be >= {min_iterations}"
        assert result.avg_duration >= 0, "Average duration must be non-negative"
        assert result.p95_duration >= result.avg_duration, "P95 duration must be >= average duration"
        assert result.p99_duration >= result.p95_duration, "P99 duration must be >= P95 duration"
        assert result.avg_cost >= 0, "Average cost must be non-negative"
        assert result.total_cost >= 0, "Total cost must be non-negative"
        assert 0 <= result.cold_starts <= result.iterations, "Cold starts must be within iterations range"
        assert 0 <= result.errors <= result.iterations, "Errors must be within iterations range"
        
        if expected_memory_size:
            assert result.memory_size == expected_memory_size, f"Expected memory size {expected_memory_size}"
        
        # Validate that total cost is approximately avg_cost * successful_iterations
        successful_iterations = result.iterations - result.errors
        if successful_iterations > 0 and result.avg_cost > 0:
            expected_total = result.avg_cost * successful_iterations
            assert abs(result.total_cost - expected_total) < 0.001, \
                f"Total cost inconsistent: {result.total_cost} vs expected {expected_total}"
        
        return True
    
    @staticmethod
    def validate_recommendation(rec: Recommendation, 
                               current_memory: int = None,
                               strategy: str = None) -> bool:
        """Validate Recommendation structure and values."""
        assert isinstance(rec, Recommendation), f"Expected Recommendation, got {type(rec)}"
        assert rec.current_memory_size > 0, "Current memory size must be positive"
        assert rec.optimal_memory_size > 0, "Optimal memory size must be positive"
        assert rec.strategy in ['cost', 'speed', 'balanced', 'comprehensive'], f"Invalid strategy: {rec.strategy}"
        assert 0 <= rec.confidence_score <= 1, "Confidence score must be between 0 and 1"
        assert isinstance(rec.should_optimize, bool), "should_optimize must be boolean"
        assert rec.reasoning, "Reasoning must not be empty"
        
        if current_memory:
            assert rec.current_memory_size == current_memory, f"Expected current memory {current_memory}"
        
        if strategy:
            assert rec.strategy == strategy, f"Expected strategy {strategy}"
        
        # If no optimization is needed, memories should be equal
        if not rec.should_optimize:
            assert rec.current_memory_size == rec.optimal_memory_size, \
                "If no optimization needed, current and optimal memory should be equal"
        
        # Validate estimated savings structure
        if rec.estimated_monthly_savings:
            for usage_level, savings_data in rec.estimated_monthly_savings.items():
                assert isinstance(savings_data, dict), f"Savings data must be dict for {usage_level}"
                required_keys = ['current_cost', 'optimized_cost', 'savings']
                for key in required_keys:
                    assert key in savings_data, f"Missing key {key} in savings data"
                    assert isinstance(savings_data[key], (int, float)), f"{key} must be numeric"
        
        return True
    
    @staticmethod
    def validate_performance_analysis(analysis: PerformanceAnalysis,
                                    expected_memory_sizes: List[int] = None) -> bool:
        """Validate PerformanceAnalysis structure and values."""
        assert isinstance(analysis, PerformanceAnalysis), f"Expected PerformanceAnalysis, got {type(analysis)}"
        assert analysis.memory_results, "Memory results must not be empty"
        assert analysis.efficiency_scores, "Efficiency scores must not be empty"
        assert analysis.cost_optimal, "Cost optimal must not be empty"
        assert analysis.speed_optimal, "Speed optimal must not be empty"
        assert analysis.balanced_optimal, "Balanced optimal must not be empty"
        assert analysis.trends, "Trends must not be empty"
        assert isinstance(analysis.insights, list), "Insights must be a list"
        
        # Validate memory results
        for memory_size, result in analysis.memory_results.items():
            TestValidators.validate_memory_test_result(result, memory_size)
        
        # Validate efficiency scores
        for memory_size, score in analysis.efficiency_scores.items():
            assert memory_size in analysis.memory_results, f"Memory size {memory_size} not in results"
            assert isinstance(score, (int, float)), f"Efficiency score must be numeric"
            assert score >= 0, f"Efficiency score must be non-negative"
        
        # Validate optimal configurations
        for optimal_type, optimal_data in [
            ('cost', analysis.cost_optimal),
            ('speed', analysis.speed_optimal),
            ('balanced', analysis.balanced_optimal)
        ]:
            if optimal_data:
                assert 'memory_size' in optimal_data, f"Missing memory_size in {optimal_type}_optimal"
                memory_size = optimal_data['memory_size']
                assert memory_size in analysis.memory_results, f"Optimal memory {memory_size} not in results"
        
        # Validate trends
        trend_keys = ['duration_trend', 'cost_trend', 'memory_sensitivity']
        for key in trend_keys:
            assert key in analysis.trends, f"Missing trend key: {key}"
        
        # Validate insights
        for insight in analysis.insights:
            assert isinstance(insight, dict), "Each insight must be a dict"
            required_keys = ['type', 'severity', 'message', 'recommendation']
            for key in required_keys:
                assert key in insight, f"Missing insight key: {key}"
        
        if expected_memory_sizes:
            actual_sizes = set(analysis.memory_results.keys())
            expected_sizes = set(expected_memory_sizes)
            assert actual_sizes == expected_sizes, f"Memory sizes mismatch: {actual_sizes} vs {expected_sizes}"
        
        return True
    
    @staticmethod
    def validate_tuning_result(result: TuningResult,
                              expected_function_arn: str = None) -> bool:
        """Validate complete TuningResult structure."""
        assert isinstance(result, TuningResult), f"Expected TuningResult, got {type(result)}"
        assert result.function_arn, "Function ARN must not be empty"
        assert isinstance(result.timestamp, datetime), "Timestamp must be datetime"
        assert result.strategy, "Strategy must not be empty"
        assert result.memory_results, "Memory results must not be empty"
        assert isinstance(result.analysis, PerformanceAnalysis), "Analysis must be PerformanceAnalysis"
        assert isinstance(result.recommendation, Recommendation), "Recommendation must be Recommendation"
        assert result.duration >= 0, "Duration must be non-negative"
        
        # Validate ARN format
        arn_pattern = r'^arn:aws:lambda:[^:]+:\d+:function:[^:]+$'
        assert re.match(arn_pattern, result.function_arn), f"Invalid ARN format: {result.function_arn}"
        
        if expected_function_arn:
            assert result.function_arn == expected_function_arn, f"Expected ARN {expected_function_arn}"
        
        # Validate consistency between components
        analysis_memory_sizes = set(result.analysis.memory_results.keys())
        result_memory_sizes = set(result.memory_results.keys())
        assert analysis_memory_sizes == result_memory_sizes, \
            "Memory sizes inconsistent between analysis and results"
        
        # Validate recommendation uses memory sizes from results
        assert result.recommendation.current_memory_size in result_memory_sizes or \
               result.recommendation.optimal_memory_size in result_memory_sizes, \
               "Recommendation memory sizes not in test results"
        
        return True


class TestAssertions:
    """Enhanced assertion helpers for testing."""
    
    @staticmethod
    def assert_memory_performance_trend(memory_results: Dict[int, MemoryTestResult],
                                      expected_trend: str = 'decreasing'):
        """Assert that performance follows expected trend with memory increases."""
        memory_sizes = sorted(memory_results.keys())
        durations = [memory_results[size].avg_duration for size in memory_sizes]
        
        if expected_trend == 'decreasing':
            # Generally, duration should decrease with more memory (but allow some variance)
            for i in range(1, len(durations)):
                improvement_ratio = (durations[0] - durations[i]) / durations[0]
                assert improvement_ratio >= -0.1, \
                    f"Performance should improve or stay similar with more memory: {durations}"
        elif expected_trend == 'increasing':
            # Duration increases (worse performance) - rare but possible for I/O bound
            assert durations[-1] >= durations[0] * 0.9, \
                f"Expected increasing duration trend: {durations}"
        elif expected_trend == 'stable':
            # Performance relatively stable across memory sizes
            max_duration = max(durations)
            min_duration = min(durations)
            variance_ratio = (max_duration - min_duration) / min_duration
            assert variance_ratio <= 0.5, \
                f"Performance should be stable across memory sizes: {durations}"
    
    @staticmethod
    def assert_cost_trend(memory_results: Dict[int, MemoryTestResult],
                         expected_trend: str = 'increasing'):
        """Assert that cost follows expected trend with memory increases."""
        memory_sizes = sorted(memory_results.keys())
        costs = [memory_results[size].avg_cost for size in memory_sizes]
        
        if expected_trend == 'increasing':
            # Cost should generally increase with memory
            for i in range(1, len(costs)):
                assert costs[i] >= costs[i-1] * 0.9, \
                    f"Cost should increase with memory: {costs}"
        elif expected_trend == 'decreasing':
            # Cost decreases due to performance improvements outweighing memory cost
            assert costs[-1] <= costs[0] * 1.1, \
                f"Expected decreasing cost trend: {costs}"
    
    @staticmethod
    def assert_efficiency_optimal(efficiency_scores: Dict[int, float],
                                optimal_memory: int,
                                tolerance: float = 0.1):
        """Assert that the specified memory size is indeed optimal or near-optimal."""
        max_efficiency = max(efficiency_scores.values())
        optimal_efficiency = efficiency_scores[optimal_memory]
        
        efficiency_ratio = optimal_efficiency / max_efficiency
        assert efficiency_ratio >= (1 - tolerance), \
            f"Optimal memory {optimal_memory} efficiency {optimal_efficiency:.2f} " \
            f"should be within {tolerance*100}% of max efficiency {max_efficiency:.2f}"
    
    @staticmethod
    def assert_cold_start_impact(memory_results: Dict[int, MemoryTestResult],
                               max_cold_start_rate: float = 0.5):
        """Assert reasonable cold start rates across memory configurations."""
        for memory_size, result in memory_results.items():
            cold_start_rate = result.cold_starts / result.iterations
            assert cold_start_rate <= max_cold_start_rate, \
                f"Cold start rate {cold_start_rate:.2%} too high for {memory_size}MB"
    
    @staticmethod
    def assert_error_rate_acceptable(memory_results: Dict[int, MemoryTestResult],
                                   max_error_rate: float = 0.1):
        """Assert that error rates are within acceptable limits."""
        for memory_size, result in memory_results.items():
            error_rate = result.errors / result.iterations
            assert error_rate <= max_error_rate, \
                f"Error rate {error_rate:.2%} too high for {memory_size}MB"
    
    @staticmethod
    def assert_recommendation_consistency(recommendation: Recommendation,
                                        analysis: PerformanceAnalysis):
        """Assert that recommendation is consistent with analysis."""
        # Strategy-specific consistency checks
        if recommendation.strategy == 'cost':
            cost_optimal_memory = analysis.cost_optimal.get('memory_size')
            if cost_optimal_memory:
                assert recommendation.optimal_memory_size == cost_optimal_memory, \
                    f"Cost strategy should recommend cost optimal memory {cost_optimal_memory}"
        
        elif recommendation.strategy == 'speed':
            speed_optimal_memory = analysis.speed_optimal.get('memory_size')
            if speed_optimal_memory:
                assert recommendation.optimal_memory_size == speed_optimal_memory, \
                    f"Speed strategy should recommend speed optimal memory {speed_optimal_memory}"
        
        elif recommendation.strategy == 'balanced':
            balanced_optimal_memory = analysis.balanced_optimal.get('memory_size')
            if balanced_optimal_memory:
                assert recommendation.optimal_memory_size == balanced_optimal_memory, \
                    f"Balanced strategy should recommend balanced optimal memory {balanced_optimal_memory}"
    
    @staticmethod
    def assert_aws_arn_format(arn: str, service: str = 'lambda', region: str = None):
        """Assert AWS ARN format is correct."""
        arn_parts = arn.split(':')
        assert len(arn_parts) >= 6, f"Invalid ARN format: {arn}"
        assert arn_parts[0] == 'arn', f"ARN must start with 'arn': {arn}"
        assert arn_parts[1] == 'aws', f"ARN must have 'aws' partition: {arn}"
        assert arn_parts[2] == service, f"ARN service must be '{service}': {arn}"
        
        if region:
            assert arn_parts[3] == region, f"Expected region '{region}' in ARN: {arn}"
        
        # Validate account ID (12 digits)
        account_id = arn_parts[4]
        assert re.match(r'^\d{12}$', account_id), f"Invalid account ID in ARN: {account_id}"


class TestHelpers:
    """Utility functions for test setup and teardown."""
    
    @staticmethod
    def create_mock_response(status_code: int = 200, 
                           payload: Dict[str, Any] = None,
                           duration: float = 100.0,
                           cold_start: bool = False,
                           error: bool = False) -> Dict[str, Any]:
        """Create a mock Lambda invocation response."""
        if payload is None:
            payload = {"result": "success"} if not error else {"errorMessage": "Test error"}
        
        billed_duration = max(1, int(duration + 0.999))
        
        response = {
            'StatusCode': status_code,
            'Payload': Mock(read=lambda: json.dumps(payload).encode()),
        }
        
        if error:
            response['FunctionError'] = 'Unhandled'
        
        # Create log result
        log_lines = ["START RequestId: test-request-id"]
        if cold_start:
            log_lines.extend([
                "INIT_START Runtime Version: python:3.9.v16",
                "INIT_REPORT Duration: 250.00 ms"
            ])
        
        if not error:
            log_lines.append("Function executed successfully")
        
        log_lines.extend([
            "END RequestId: test-request-id",
            f"REPORT RequestId: test-request-id\t"
            f"Duration: {duration:.2f} ms\t"
            f"Billed Duration: {billed_duration} ms\t"
            f"Memory Size: 256 MB\t"
            f"Max Memory Used: 128 MB"
        ])
        
        response['LogResult'] = '\n'.join(log_lines)
        return response
    
    @staticmethod
    def assert_approximately_equal(actual: float, expected: float, 
                                 tolerance: float = 0.01, 
                                 message: str = None):
        """Assert two values are approximately equal within tolerance."""
        if expected == 0:
            assert abs(actual) <= tolerance, \
                message or f"Expected ~{expected}, got {actual}"
        else:
            relative_error = abs(actual - expected) / abs(expected)
            assert relative_error <= tolerance, \
                message or f"Expected ~{expected}, got {actual} (relative error: {relative_error:.2%})"
    
    @staticmethod
    def assert_timing_reasonable(start_time: datetime, end_time: datetime,
                               min_duration: float = 0, max_duration: float = 3600):
        """Assert that timing values are reasonable."""
        assert isinstance(start_time, datetime), "Start time must be datetime"
        assert isinstance(end_time, datetime), "End time must be datetime"
        assert end_time >= start_time, "End time must be after start time"
        
        duration = (end_time - start_time).total_seconds()
        assert min_duration <= duration <= max_duration, \
            f"Duration {duration}s not in range [{min_duration}, {max_duration}]"
    
    @staticmethod
    def create_workload_specific_results(workload_type: str, 
                                       memory_sizes: List[int] = None) -> Dict[int, MemoryTestResult]:
        """Create memory test results that reflect specific workload characteristics."""
        from tests.utils.test_data_generators import TestDataGenerator
        
        if memory_sizes is None:
            memory_sizes = [256, 512, 1024, 2048]
        
        generator = TestDataGenerator(seed=42)  # Fixed seed for reproducibility
        results = {}
        
        for memory_size in memory_sizes:
            results[memory_size] = generator.generate_memory_test_result(
                memory_size=memory_size,
                iterations=10,
                workload_type=workload_type
            )
        
        return results
    
    @staticmethod
    def simulate_network_delay(min_delay: float = 0.1, max_delay: float = 0.5):
        """Simulate network delay for more realistic testing."""
        import time
        import random
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        return delay
    
    @staticmethod
    def validate_json_serializable(obj: Any, max_depth: int = 10) -> bool:
        """Validate that an object is JSON serializable."""
        try:
            json.dumps(obj, default=str)
            return True
        except (TypeError, ValueError) as e:
            pytest.fail(f"Object not JSON serializable: {e}")
    
    @staticmethod
    def create_test_environment_info() -> Dict[str, Any]:
        """Create test environment information for debugging."""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'test_timestamp': datetime.utcnow().isoformat(),
            'timezone': str(datetime.now().astimezone().tzinfo)
        }


class PerformanceTestHelpers:
    """Helpers specifically for performance testing."""
    
    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
        """Measure execution time of a function."""
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        return result, duration
    
    @staticmethod
    def assert_performance_within_bounds(duration: float, 
                                       max_duration: float,
                                       operation_name: str = "Operation"):
        """Assert that operation completed within performance bounds."""
        assert duration <= max_duration, \
            f"{operation_name} took {duration:.3f}s, expected <= {max_duration}s"
    
    @staticmethod
    def create_load_test_data(num_functions: int = 10,
                            executions_per_function: int = 100) -> List[Dict[str, Any]]:
        """Create data for load testing scenarios."""
        functions = []
        for i in range(num_functions):
            functions.append({
                'function_arn': f'arn:aws:lambda:us-east-1:123456789012:function:test-function-{i}',
                'memory_sizes': [256, 512, 1024],
                'executions': executions_per_function,
                'workload_type': ['cpu_intensive', 'io_bound', 'memory_intensive'][i % 3]
            })
        return functions


# Pytest fixtures using the helpers
@pytest.fixture
def validator():
    """Provide TestValidators instance."""
    return TestValidators()

@pytest.fixture
def assertions():
    """Provide TestAssertions instance."""
    return TestAssertions()

@pytest.fixture
def helpers():
    """Provide TestHelpers instance."""
    return TestHelpers()

@pytest.fixture
def performance_helpers():
    """Provide PerformanceTestHelpers instance."""
    return PerformanceTestHelpers()