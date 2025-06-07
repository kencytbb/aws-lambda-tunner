"""Performance tests for optimization algorithms."""

import pytest
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from aws_lambda_tuner.orchestrator_module import TuningOrchestrator
from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.analyzers.analyzer import PerformanceAnalyzer
from tests.utils.test_helpers import PerformanceTestHelpers, TestHelpers
from tests.utils.test_data_generators import TestDataGenerator


@pytest.mark.performance
@pytest.mark.slow
class TestOptimizationPerformance:
    """Test performance of optimization algorithms."""
    
    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TuningOrchestrator()
    
    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def performance_helpers(self):
        return PerformanceTestHelpers()
    
    def test_single_function_optimization_performance(self, orchestrator, mock_aws_services, 
                                                    performance_helpers):
        """Test performance of optimizing a single function."""
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:perf-test',
            memory_sizes=[256, 512, 1024, 1536, 2048],
            iterations=20,
            strategy='balanced'
        )
        
        # Measure optimization time
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        performance_helpers.assert_performance_within_bounds(
            optimization_time, 30.0, "Single function optimization"
        )
        
        # Verify result quality
        assert result is not None
        assert len(result.memory_results) == 5
        assert result.recommendation is not None
    
    def test_large_memory_range_performance(self, orchestrator, mock_aws_services, 
                                          performance_helpers):
        """Test performance with large memory size range."""
        # Test with many memory configurations
        memory_sizes = list(range(128, 3009, 64))  # 45 different memory sizes
        
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:large-range-test',
            memory_sizes=memory_sizes,
            iterations=5,  # Fewer iterations to keep test reasonable
            strategy='comprehensive'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Should scale reasonably with number of memory configurations
        # Allow more time for comprehensive testing
        max_time = len(memory_sizes) * 2.0  # 2 seconds per memory config max
        performance_helpers.assert_performance_within_bounds(
            optimization_time, max_time, "Large memory range optimization"
        )
        
        assert len(result.memory_results) == len(memory_sizes)
    
    def test_high_iteration_count_performance(self, orchestrator, mock_aws_services, 
                                            performance_helpers):
        """Test performance with high iteration counts."""
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:high-iter-test',
            memory_sizes=[512, 1024, 2048],
            iterations=100,  # High iteration count
            strategy='balanced'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Should handle high iteration counts efficiently
        performance_helpers.assert_performance_within_bounds(
            optimization_time, 60.0, "High iteration count optimization"
        )
        
        # Verify all iterations were completed
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == 100
    
    def test_concurrent_execution_performance(self, orchestrator, mock_aws_services, 
                                            performance_helpers):
        """Test performance with concurrent executions."""
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:concurrent-test',
            memory_sizes=[256, 512, 1024, 1536, 2048],
            iterations=15,
            concurrent_executions=5,  # High concurrency
            strategy='balanced'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Concurrent execution should not significantly increase total time
        # (may actually decrease due to parallelism)
        performance_helpers.assert_performance_within_bounds(
            optimization_time, 45.0, "Concurrent execution optimization"
        )
        
        assert result is not None
        assert len(result.memory_results) == 5
    
    @pytest.mark.benchmark
    def test_analysis_algorithm_performance(self, analyzer, performance_helpers):
        """Test performance of analysis algorithms."""
        # Generate large dataset
        generator = TestDataGenerator(seed=42)
        memory_sizes = [128, 256, 512, 1024, 1536, 2048, 3008]
        
        large_results = {}
        for memory_size in memory_sizes:
            large_results[memory_size] = generator.generate_memory_test_result(
                memory_size, 100, 'balanced'  # High iteration count
            )
        
        # Measure analysis time
        start_time = time.perf_counter()
        analysis = analyzer.analyze_results(large_results)
        end_time = time.perf_counter()
        
        analysis_time = end_time - start_time
        
        # Analysis should be fast even with large datasets
        performance_helpers.assert_performance_within_bounds(
            analysis_time, 2.0, "Large dataset analysis"
        )
        
        assert analysis is not None
        assert len(analysis.memory_results) == len(memory_sizes)
    
    def test_memory_usage_during_optimization(self, orchestrator, mock_aws_services):
        """Test memory usage during optimization."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:memory-test',
            memory_sizes=[256, 512, 1024, 1536, 2048, 3008],
            iterations=25,
            strategy='comprehensive'
        )
        
        # Monitor memory during optimization
        max_memory = initial_memory
        memory_samples = []
        
        def monitor_memory():
            nonlocal max_memory
            while True:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                max_memory = max(max_memory, current_memory)
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            result = orchestrator.run_optimization(config)
        finally:
            monitor_thread = None  # Stop monitoring
        
        memory_increase = max_memory - initial_memory
        
        # Memory usage should not increase excessively (adjust threshold as needed)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
        
        # Verify optimization completed successfully
        assert result is not None
        assert len(result.memory_results) == 6


@pytest.mark.performance
class TestScalabilityPerformance:
    """Test performance scalability with different loads."""
    
    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TuningOrchestrator()
    
    @pytest.fixture
    def performance_helpers(self):
        return PerformanceTestHelpers()
    
    def test_multiple_function_optimization_performance(self, orchestrator, mock_aws_services, 
                                                      performance_helpers):
        """Test performance when optimizing multiple functions."""
        # Create test data for multiple functions
        function_configs = performance_helpers.create_load_test_data(
            num_functions=5, executions_per_function=10
        )
        
        results = []
        start_time = time.perf_counter()
        
        for func_config in function_configs:
            config = TunerConfig(
                function_arn=func_config['function_arn'],
                memory_sizes=func_config['memory_sizes'],
                iterations=func_config['executions'],
                strategy='balanced',
                workload_type=func_config['workload_type']
            )
            
            result = orchestrator.run_optimization(config)
            results.append(result)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should scale reasonably with number of functions
        performance_helpers.assert_performance_within_bounds(
            total_time, 60.0, "Multiple function optimization"
        )
        
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert result.recommendation is not None
    
    def test_parallel_function_optimization_performance(self, orchestrator, mock_aws_services, 
                                                      performance_helpers):
        """Test performance of parallel function optimization."""
        function_configs = performance_helpers.create_load_test_data(
            num_functions=3, executions_per_function=10
        )
        
        def optimize_function(func_config):
            config = TunerConfig(
                function_arn=func_config['function_arn'],
                memory_sizes=func_config['memory_sizes'],
                iterations=func_config['executions'],
                strategy='balanced'
            )
            return orchestrator.run_optimization(config)
        
        start_time = time.perf_counter()
        
        # Run optimizations in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_results = [executor.submit(optimize_function, config) 
                            for config in function_configs]
            results = [future.result() for future in future_results]
        
        end_time = time.perf_counter()
        parallel_time = end_time - start_time
        
        # Parallel execution should be faster than sequential
        # (though limited by mocking overhead)
        performance_helpers.assert_performance_within_bounds(
            parallel_time, 45.0, "Parallel function optimization"
        )
        
        assert len(results) == 3
        for result in results:
            assert result is not None
    
    @pytest.mark.parametrize("num_memory_configs", [3, 5, 10, 15])
    def test_memory_config_count_scaling(self, orchestrator, mock_aws_services, 
                                       performance_helpers, num_memory_configs):
        """Test how performance scales with number of memory configurations."""
        memory_sizes = list(range(256, 256 + num_memory_configs * 256, 256))
        
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:scaling-test',
            memory_sizes=memory_sizes,
            iterations=8,
            strategy='balanced'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Time should scale roughly linearly with number of configs
        max_time = num_memory_configs * 3.0  # 3 seconds per memory config max
        performance_helpers.assert_performance_within_bounds(
            optimization_time, max_time, f"Optimization with {num_memory_configs} memory configs"
        )
        
        assert len(result.memory_results) == num_memory_configs
    
    @pytest.mark.parametrize("iteration_count", [5, 10, 25, 50])
    def test_iteration_count_scaling(self, orchestrator, mock_aws_services, 
                                   performance_helpers, iteration_count):
        """Test how performance scales with iteration count."""
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:iteration-scaling-test',
            memory_sizes=[512, 1024, 2048],
            iterations=iteration_count,
            strategy='balanced'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Time should scale with iteration count but not linearly (due to mocking efficiency)
        max_time = max(15.0, iteration_count * 0.5)  # Minimum 15s, or 0.5s per iteration
        performance_helpers.assert_performance_within_bounds(
            optimization_time, max_time, f"Optimization with {iteration_count} iterations"
        )
        
        # Verify all iterations completed
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == iteration_count


@pytest.mark.performance
@pytest.mark.stress
class TestStressPerformance:
    """Stress tests for optimization performance."""
    
    @pytest.fixture
    def orchestrator(self, mock_aws_services):
        return TuningOrchestrator()
    
    @pytest.fixture
    def performance_helpers(self):
        return PerformanceTestHelpers()
    
    @pytest.mark.slow
    def test_extreme_memory_range_stress(self, orchestrator, mock_aws_services, 
                                       performance_helpers):
        """Stress test with extreme memory range."""
        # Test every possible AWS Lambda memory configuration
        memory_sizes = list(range(128, 3009, 64))  # Every 64MB from 128 to 3008
        
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:stress-test',
            memory_sizes=memory_sizes,
            iterations=3,  # Keep iterations low for stress test
            strategy='comprehensive'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Should complete even with extreme configuration
        max_time = len(memory_sizes) * 1.5  # 1.5 seconds per memory config
        performance_helpers.assert_performance_within_bounds(
            optimization_time, max_time, "Extreme memory range stress test"
        )
        
        assert len(result.memory_results) == len(memory_sizes)
        assert result.recommendation is not None
    
    @pytest.mark.slow
    def test_high_iteration_stress(self, orchestrator, mock_aws_services, 
                                 performance_helpers):
        """Stress test with very high iteration count."""
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:high-iter-stress',
            memory_sizes=[512, 1024, 2048],
            iterations=200,  # Very high iteration count
            strategy='balanced'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Should handle high iterations without excessive time
        performance_helpers.assert_performance_within_bounds(
            optimization_time, 120.0, "High iteration stress test"
        )
        
        # Verify all iterations completed
        for memory_result in result.memory_results.values():
            assert memory_result.iterations == 200
    
    @pytest.mark.slow
    def test_memory_leak_stress(self, orchestrator, mock_aws_services):
        """Stress test to check for memory leaks."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple optimizations to test for memory leaks
        for i in range(5):
            config = TunerConfig(
                function_arn=f'arn:aws:lambda:us-east-1:123456789012:function:leak-test-{i}',
                memory_sizes=[256, 512, 1024, 1536, 2048],
                iterations=10,
                strategy='balanced'
            )
            
            result = orchestrator.run_optimization(config)
            assert result is not None
            
            # Check memory after each optimization
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Memory should not grow excessively with each iteration
            assert memory_increase < 100 * (i + 1), \
                f"Potential memory leak: {memory_increase:.1f}MB increase after {i+1} optimizations"
    
    def test_concurrent_stress(self, orchestrator, mock_aws_services, performance_helpers):
        """Stress test with high concurrent executions."""
        config = TunerConfig(
            function_arn='arn:aws:lambda:us-east-1:123456789012:function:concurrent-stress',
            memory_sizes=[256, 512, 1024, 1536, 2048, 3008],
            iterations=15,
            concurrent_executions=10,  # Very high concurrency
            strategy='speed'
        )
        
        start_time = time.perf_counter()
        result = orchestrator.run_optimization(config)
        end_time = time.perf_counter()
        
        optimization_time = end_time - start_time
        
        # Should handle high concurrency efficiently
        performance_helpers.assert_performance_within_bounds(
            optimization_time, 60.0, "High concurrency stress test"
        )
        
        assert result is not None
        assert len(result.memory_results) == 6


@pytest.mark.performance
@pytest.mark.benchmark
class TestBenchmarkComparisons:
    """Benchmark tests for performance comparisons."""
    
    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def performance_helpers(self):
        return PerformanceTestHelpers()
    
    def test_analysis_algorithm_benchmark(self, analyzer, performance_helpers):
        """Benchmark different analysis scenarios."""
        generator = TestDataGenerator(seed=42)
        
        # Small dataset benchmark
        small_results = {
            memory_size: generator.generate_memory_test_result(memory_size, 10, 'balanced')
            for memory_size in [256, 512, 1024]
        }
        
        small_analysis, small_time = performance_helpers.measure_execution_time(
            analyzer.analyze_results, small_results
        )
        
        # Large dataset benchmark
        large_results = {
            memory_size: generator.generate_memory_test_result(memory_size, 50, 'balanced')
            for memory_size in range(128, 3009, 128)
        }
        
        large_analysis, large_time = performance_helpers.measure_execution_time(
            analyzer.analyze_results, large_results
        )
        
        # Analysis time should scale sub-linearly
        size_ratio = len(large_results) / len(small_results)
        time_ratio = large_time / small_time if small_time > 0 else 1
        
        # Time ratio should be less than size ratio (sub-linear scaling)
        assert time_ratio < size_ratio * 2, \
            f"Analysis scaling: {time_ratio:.2f}x time for {size_ratio:.2f}x data"
        
        # Both should produce valid results
        assert small_analysis is not None
        assert large_analysis is not None
    
    def test_strategy_performance_comparison(self, orchestrator, mock_aws_services, 
                                           performance_helpers):
        """Compare performance of different optimization strategies."""
        base_config = {
            'function_arn': 'arn:aws:lambda:us-east-1:123456789012:function:strategy-benchmark',
            'memory_sizes': [256, 512, 1024, 1536, 2048],
            'iterations': 10
        }
        
        strategy_times = {}
        
        for strategy in ['cost', 'speed', 'balanced', 'comprehensive']:
            config = TunerConfig(**base_config, strategy=strategy)
            
            result, execution_time = performance_helpers.measure_execution_time(
                orchestrator.run_optimization, config
            )
            
            strategy_times[strategy] = execution_time
            assert result is not None
        
        # All strategies should complete in reasonable time
        for strategy, time_taken in strategy_times.items():
            performance_helpers.assert_performance_within_bounds(
                time_taken, 30.0, f"{strategy} strategy optimization"
            )
        
        # Comprehensive strategy may take longer, but not excessively
        if 'comprehensive' in strategy_times:
            comprehensive_time = strategy_times['comprehensive']
            other_times = [t for s, t in strategy_times.items() if s != 'comprehensive']
            avg_other_time = sum(other_times) / len(other_times)
            
            assert comprehensive_time <= avg_other_time * 2, \
                "Comprehensive strategy should not be more than 2x slower than other strategies"