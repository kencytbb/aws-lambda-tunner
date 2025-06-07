"""Continuous workload optimization strategy."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .workload_strategy import (
    WorkloadStrategy, WorkloadType, WorkloadCharacteristics,
    TestingStrategy, OptimizationResult
)

logger = logging.getLogger(__name__)


class ContinuousStrategy(WorkloadStrategy):
    """Strategy for optimizing continuous, long-running workloads."""

    async def analyze_workload(self) -> WorkloadCharacteristics:
        """
        Analyze continuous workload characteristics.
        
        Returns:
            WorkloadCharacteristics: Analyzed workload characteristics
        """
        self.logger.info("Analyzing continuous workload characteristics")
        
        # Get function metrics from a longer period for continuous workloads
        metrics = await self._get_function_metrics(hours_back=168)  # 7 days
        concurrency_metrics = await self._get_concurrent_executions_metrics(hours_back=168)
        
        # Determine characteristics
        frequency = self._classify_invocation_frequency(metrics)
        traffic_pattern = self._analyze_traffic_pattern(metrics)
        memory_utilization = self._estimate_memory_utilization(metrics)
        
        # Continuous workloads have specific characteristics
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.CONTINUOUS,
            invocation_frequency=frequency,
            traffic_pattern=traffic_pattern,
            cold_start_sensitivity="medium",  # Less sensitive due to sustained execution
            cost_sensitivity=self._determine_cost_sensitivity(WorkloadType.CONTINUOUS, frequency),
            response_time_requirement="moderate",  # Typically batch processing
            peak_concurrency=self._extract_peak_concurrency(concurrency_metrics),
            memory_utilization=memory_utilization
        )
        
        self.logger.info(f"Workload analysis complete: frequency={frequency}, pattern={traffic_pattern}")
        return workload

    def create_testing_strategy(self, workload: WorkloadCharacteristics) -> TestingStrategy:
        """
        Create testing strategy optimized for continuous workloads.
        
        Args:
            workload: Analyzed workload characteristics
            
        Returns:
            TestingStrategy: Strategy for testing this workload
        """
        self.logger.info("Creating continuous workload testing strategy")
        
        # Base memory sizes for continuous workloads (focus on cost-efficiency and throughput)
        if workload.cost_sensitivity == "high":
            # Focus on cost-efficient memory sizes
            memory_sizes = [128, 256, 512, 768, 1024, 1536]
        elif workload.invocation_frequency == "high":
            # Focus on performance for high-frequency workloads
            memory_sizes = [256, 512, 768, 1024, 1536, 2048, 3008]
        else:
            # Balanced approach
            memory_sizes = [256, 512, 768, 1024, 1536, 2048]
        
        # Filter to valid memory sizes
        memory_sizes = sorted(list(set([size for size in memory_sizes if 128 <= size <= 10240])))
        
        # Adjust iterations based on traffic pattern and frequency
        if workload.traffic_pattern == "steady":
            iterations = 25  # More iterations for steady workloads to get accurate measurements
            concurrent_executions = 8
        elif workload.traffic_pattern == "bursty":
            iterations = 20
            concurrent_executions = 12  # Higher concurrency to simulate bursts
        else:
            iterations = 15
            concurrent_executions = 6
        
        # Adjust based on invocation frequency
        if workload.invocation_frequency == "high":
            iterations += 10
            concurrent_executions = min(concurrent_executions + 5, 20)
        
        # Fewer warmup runs since cold starts are less critical
        warmup_runs = 2
        
        # Test duration for sustained performance testing
        test_duration_minutes = 10 if workload.traffic_pattern == "steady" else None
        
        # Determine if we should test provisioned concurrency
        provisioned_concurrency_levels = None
        if (workload.invocation_frequency == "high" and 
            workload.peak_concurrency and workload.peak_concurrency > 20):
            # For very high-frequency continuous workloads, test provisioned concurrency
            max_provisioned = min(workload.peak_concurrency, 500)
            provisioned_concurrency_levels = [
                max_provisioned // 4,
                max_provisioned // 2,
                max_provisioned
            ]
            provisioned_concurrency_levels = [pc for pc in provisioned_concurrency_levels if pc > 10]
        
        strategy = TestingStrategy(
            memory_sizes=memory_sizes,
            iterations_per_memory=iterations,
            concurrent_executions=concurrent_executions,
            warmup_runs=warmup_runs,
            test_duration_minutes=test_duration_minutes,
            time_based_testing=test_duration_minutes is not None,
            provisioned_concurrency_levels=provisioned_concurrency_levels,
            traffic_simulation=True  # Enable traffic simulation for realistic testing
        )
        
        self.logger.info(
            f"Testing strategy created: {len(memory_sizes)} memory configs, "
            f"{iterations} iterations each, "
            f"{'time-based' if test_duration_minutes else 'iteration-based'} testing, "
            f"{'with' if provisioned_concurrency_levels else 'without'} provisioned concurrency testing"
        )
        
        return strategy

    async def execute_optimization(
        self, 
        workload: WorkloadCharacteristics, 
        strategy: TestingStrategy
    ) -> OptimizationResult:
        """
        Execute optimization for continuous workloads.
        
        Args:
            workload: Workload characteristics
            strategy: Testing strategy
            
        Returns:
            OptimizationResult: Optimization results
        """
        self.logger.info("Executing continuous workload optimization")
        
        # Store original configuration
        original_config = await self.aws_provider.get_function_configuration()
        original_memory = original_config['MemorySize']
        
        test_results = {}
        baseline_results = None
        
        try:
            # Test baseline configuration first
            if original_memory in strategy.memory_sizes:
                self.logger.info(f"Testing baseline configuration: {original_memory}MB")
                baseline_results = await self._test_continuous_configuration(
                    original_memory,
                    strategy.iterations_per_memory,
                    strategy.test_duration_minutes
                )
                test_results[original_memory] = baseline_results
            
            # Test each memory configuration
            for memory_size in strategy.memory_sizes:
                if memory_size == original_memory:
                    continue  # Already tested as baseline
                
                self.logger.info(f"Testing memory configuration: {memory_size}MB")
                
                # Test standard configuration
                config_results = await self._test_continuous_configuration(
                    memory_size,
                    strategy.iterations_per_memory,
                    strategy.test_duration_minutes
                )
                test_results[memory_size] = config_results
                
                # Test with provisioned concurrency if specified
                if strategy.provisioned_concurrency_levels:
                    for pc_level in strategy.provisioned_concurrency_levels:
                        await self._test_with_provisioned_concurrency(
                            memory_size, pc_level, strategy.iterations_per_memory, test_results
                        )
            
            # Restore original configuration
            if not self.config.dry_run:
                await self.aws_provider.update_function_memory(original_memory)
            
            # Analyze results and generate recommendation
            optimal_config = self._find_optimal_configuration(test_results, workload)
            recommendation = self._generate_recommendation(
                test_results, optimal_config, workload, baseline_results
            )
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            # Ensure we restore original configuration
            if not self.config.dry_run:
                try:
                    await self.aws_provider.update_function_memory(original_memory)
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore original configuration: {restore_error}")
            raise

    async def _test_continuous_configuration(
        self, 
        memory_size: int, 
        iterations: int,
        test_duration_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test memory configuration with continuous workload patterns.
        
        Args:
            memory_size: Memory configuration to test
            iterations: Number of test iterations
            test_duration_minutes: Duration for time-based testing
            
        Returns:
            Test results with continuous workload metrics
        """
        self.logger.info(f"Testing continuous configuration: {memory_size}MB")
        
        # Standard testing
        base_results = await self._test_memory_configuration_with_monitoring(
            memory_size, iterations, test_duration_minutes
        )
        
        # Add continuous-specific testing
        sustained_performance = await self._test_sustained_performance(memory_size)
        throughput_results = await self._test_throughput_scaling(memory_size)
        
        # Combine results
        base_results['sustained_performance'] = sustained_performance
        base_results['throughput_scaling'] = throughput_results
        base_results['workload_type'] = 'continuous'
        
        return base_results

    async def _test_sustained_performance(self, memory_size: int) -> Dict[str, Any]:
        """
        Test sustained performance under continuous load.
        
        Args:
            memory_size: Memory configuration to test
            
        Returns:
            Sustained performance metrics
        """
        self.logger.info("Testing sustained performance")
        
        sustained_results = {
            'performance_waves': [],
            'performance_degradation': False,
            'stability_score': 0.0,
            'memory_leak_detected': False
        }
        
        if self.config.dry_run:
            # Simulate sustained performance results
            sustained_results['performance_waves'] = [1000, 980, 985, 990, 995]
            sustained_results['stability_score'] = 0.95
            return sustained_results
        
        # Run multiple waves of executions to test sustained performance
        wave_count = 5
        wave_size = 10
        
        for wave in range(wave_count):
            self.logger.info(f"Running sustained performance wave {wave + 1}/{wave_count}")
            
            wave_start = datetime.now()
            wave_results = []
            
            # Run wave of executions
            for i in range(wave_size):
                try:
                    result = await self.aws_provider.invoke_function(self.config.payload)
                    wave_results.append(result.get('duration', 0))
                    await asyncio.sleep(0.5)  # Brief pause between invocations
                except Exception as e:
                    self.logger.warning(f"Wave {wave} execution {i} failed: {e}")
            
            # Calculate wave average
            if wave_results:
                wave_avg = sum(wave_results) / len(wave_results)
                sustained_results['performance_waves'].append(wave_avg)
            
            # Brief pause between waves
            if wave < wave_count - 1:
                await asyncio.sleep(5)
        
        # Analyze performance stability
        if len(sustained_results['performance_waves']) > 1:
            first_wave = sustained_results['performance_waves'][0]
            last_wave = sustained_results['performance_waves'][-1]
            
            # Check for performance degradation
            if last_wave > first_wave * 1.1:  # 10% degradation threshold
                sustained_results['performance_degradation'] = True
            
            # Calculate stability score
            wave_values = sustained_results['performance_waves']
            avg_performance = sum(wave_values) / len(wave_values)
            variance = sum((x - avg_performance) ** 2 for x in wave_values) / len(wave_values)
            cv = (variance ** 0.5) / avg_performance if avg_performance > 0 else 1
            sustained_results['stability_score'] = max(0, 1 - cv)
        
        return sustained_results

    async def _test_throughput_scaling(self, memory_size: int) -> Dict[str, Any]:
        """
        Test throughput scaling characteristics.
        
        Args:
            memory_size: Memory configuration to test
            
        Returns:
            Throughput scaling metrics
        """
        self.logger.info("Testing throughput scaling")
        
        throughput_results = {
            'concurrency_levels': {},
            'optimal_concurrency': 1,
            'scaling_efficiency': 0.0,
            'bottleneck_detected': False
        }
        
        if self.config.dry_run:
            # Simulate throughput results
            throughput_results['concurrency_levels'] = {
                1: {'throughput': 2.0, 'avg_duration': 500},
                5: {'throughput': 8.5, 'avg_duration': 600},
                10: {'throughput': 15.0, 'avg_duration': 700}
            }
            throughput_results['optimal_concurrency'] = 10
            throughput_results['scaling_efficiency'] = 0.75
            return throughput_results
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 15, 20]
        
        for concurrency in concurrency_levels:
            if concurrency > self.config.concurrent_executions:
                continue
                
            self.logger.info(f"Testing concurrency level: {concurrency}")
            
            # Run concurrent executions
            start_time = datetime.now()
            execution_results = []
            
            # Create concurrent tasks
            tasks = []
            for _ in range(concurrency):
                task = asyncio.create_task(self.aws_provider.invoke_function(self.config.payload))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Process results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            if successful_results:
                avg_duration = sum(r.get('duration', 0) for r in successful_results) / len(successful_results)
                throughput = len(successful_results) / total_duration if total_duration > 0 else 0
                
                throughput_results['concurrency_levels'][concurrency] = {
                    'throughput': throughput,
                    'avg_duration': avg_duration,
                    'success_rate': len(successful_results) / concurrency * 100
                }
        
        # Analyze scaling efficiency
        if len(throughput_results['concurrency_levels']) > 1:
            concurrency_data = throughput_results['concurrency_levels']
            
            # Find optimal concurrency (highest throughput with acceptable latency)
            best_score = 0
            optimal_concurrency = 1
            
            for concurrency, metrics in concurrency_data.items():
                throughput = metrics['throughput']
                avg_duration = metrics['avg_duration']
                success_rate = metrics['success_rate']
                
                # Score = throughput / latency penalty * success rate
                latency_penalty = max(1, avg_duration / 1000)  # Normalize to seconds
                score = (throughput / latency_penalty) * (success_rate / 100)
                
                if score > best_score:
                    best_score = score
                    optimal_concurrency = concurrency
            
            throughput_results['optimal_concurrency'] = optimal_concurrency
            
            # Calculate scaling efficiency (how well throughput scales with concurrency)
            concurrency_values = sorted(concurrency_data.keys())
            if len(concurrency_values) >= 2:
                first_throughput = concurrency_data[concurrency_values[0]]['throughput']
                last_throughput = concurrency_data[concurrency_values[-1]]['throughput']
                first_concurrency = concurrency_values[0]
                last_concurrency = concurrency_values[-1]
                
                expected_scaling = last_concurrency / first_concurrency
                actual_scaling = last_throughput / first_throughput if first_throughput > 0 else 0
                
                throughput_results['scaling_efficiency'] = min(1.0, actual_scaling / expected_scaling)
        
        return throughput_results

    async def _test_with_provisioned_concurrency(
        self, 
        memory_size: int, 
        pc_level: int, 
        iterations: int,
        test_results: Dict[str, Any]
    ):
        """
        Test configuration with provisioned concurrency for continuous workloads.
        
        Args:
            memory_size: Memory configuration
            pc_level: Provisioned concurrency level
            iterations: Number of test iterations
            test_results: Dictionary to store results
        """
        self.logger.info(f"Testing {memory_size}MB with {pc_level} provisioned concurrency")
        
        if self.config.dry_run:
            # Simulate provisioned concurrency results
            results = self._simulate_provisioned_concurrency_test(memory_size, pc_level, iterations)
        else:
            # Set provisioned concurrency
            await self.aws_provider.set_provisioned_concurrency(pc_level)
            await asyncio.sleep(60)  # Wait longer for continuous workloads
            
            # Run tests with sustained performance analysis
            results = await self._test_continuous_configuration(memory_size, iterations)
            results['provisioned_concurrency'] = pc_level
            
            # Remove provisioned concurrency
            await self.aws_provider.delete_provisioned_concurrency()
        
        # Store results with provisioned concurrency suffix
        key = f"{memory_size}_pc_{pc_level}"
        test_results[key] = results

    def _find_optimal_configuration(
        self, 
        test_results: Dict[str, Any], 
        workload: WorkloadCharacteristics
    ) -> Dict[str, Any]:
        """
        Find optimal configuration for continuous workloads.
        
        Args:
            test_results: All test results
            workload: Workload characteristics
            
        Returns:
            Optimal configuration details
        """
        self.logger.info("Analyzing test results for continuous workload optimization")
        
        # Filter out failed tests
        valid_results = {
            k: v for k, v in test_results.items() 
            if v.get('success_rate', 0) >= 90  # Slightly lower threshold for continuous workloads
        }
        
        if not valid_results:
            self.logger.warning("No valid test results found")
            return {}
        
        # Calculate scores for each configuration
        scored_configs = []
        
        for config_key, results in valid_results.items():
            score = self._calculate_continuous_configuration_score(results, workload)
            
            # Parse configuration key
            if '_pc_' in config_key:
                memory_size, pc_level = config_key.split('_pc_')
                memory_size = int(memory_size)
                pc_level = int(pc_level)
            else:
                memory_size = int(config_key)
                pc_level = None
            
            scored_configs.append({
                'memory_size': memory_size,
                'provisioned_concurrency': pc_level,
                'score': score,
                'results': results,
                'config_key': config_key
            })
        
        # Sort by score (higher is better)
        scored_configs.sort(key=lambda x: x['score'], reverse=True)
        
        optimal = scored_configs[0]
        self.logger.info(
            f"Optimal configuration: {optimal['memory_size']}MB"
            f"{f' with {optimal['provisioned_concurrency']} PC' if optimal['provisioned_concurrency'] else ''}"
            f" (score: {optimal['score']:.2f})"
        )
        
        return optimal

    def _calculate_continuous_configuration_score(
        self, 
        results: Dict[str, Any], 
        workload: WorkloadCharacteristics
    ) -> float:
        """
        Calculate a score for continuous workload configuration.
        
        Args:
            results: Test results for the configuration
            workload: Workload characteristics
            
        Returns:
            Configuration score (higher is better)
        """
        # Base metrics
        avg_duration = results.get('avg_duration', float('inf'))
        success_rate = results.get('success_rate', 0)
        memory_size = results.get('memory_mb', 128)
        avg_billed_duration = results.get('avg_billed_duration', avg_duration)
        
        # Calculate cost per invocation
        cost_per_invocation = self._calculate_cost(memory_size, avg_billed_duration)
        
        # Base performance score (throughput-oriented)
        performance_score = 1000 / max(avg_duration, 1)
        
        # Stability bonus for continuous workloads
        stability_bonus = 0
        if 'sustained_performance' in results:
            stability_score = results['sustained_performance'].get('stability_score', 0)
            stability_bonus = stability_score * 200  # High weight for stability
        
        # Throughput scaling bonus
        scaling_bonus = 0
        if 'throughput_scaling' in results:
            scaling_efficiency = results['throughput_scaling'].get('scaling_efficiency', 0)
            scaling_bonus = scaling_efficiency * 150
        
        # Cost penalty (important for continuous workloads due to high volume)
        cost_penalty = 0
        if workload.cost_sensitivity == "high":
            cost_penalty = cost_per_invocation * 2000
        elif workload.cost_sensitivity == "medium":
            cost_penalty = cost_per_invocation * 1000
        else:
            cost_penalty = cost_per_invocation * 500
        
        # Success rate bonus
        success_bonus = success_rate * 5
        
        # Performance degradation penalty
        degradation_penalty = 0
        if 'sustained_performance' in results:
            if results['sustained_performance'].get('performance_degradation', False):
                degradation_penalty = 100
        
        # Calculate final score
        score = (performance_score + stability_bonus + scaling_bonus + 
                success_bonus - cost_penalty - degradation_penalty)
        
        return max(score, 0)  # Ensure non-negative score

    def _calculate_cost(self, memory_size: int, duration_ms: float) -> float:
        """
        Calculate cost per invocation.
        
        Args:
            memory_size: Memory size in MB
            duration_ms: Duration in milliseconds
            
        Returns:
            Cost per invocation in USD
        """
        # AWS Lambda pricing (simplified)
        gb_seconds = (memory_size / 1024) * (duration_ms / 1000)
        cost = gb_seconds * 0.0000166667
        
        # Add request cost
        request_cost = 0.0000002  # $0.20 per 1M requests
        
        return cost + request_cost

    def _generate_recommendation(
        self,
        test_results: Dict[str, Any],
        optimal_config: Dict[str, Any],
        workload: WorkloadCharacteristics,
        baseline_results: Optional[Dict[str, Any]]
    ) -> OptimizationResult:
        """
        Generate optimization recommendation for continuous workloads.
        
        Args:
            test_results: All test results
            optimal_config: Optimal configuration
            workload: Workload characteristics
            baseline_results: Baseline test results
            
        Returns:
            OptimizationResult with recommendation
        """
        if not optimal_config:
            return OptimizationResult(
                workload_type=workload.workload_type,
                optimal_memory=128,
                optimal_provisioned_concurrency=None,
                cost_savings_percent=0,
                performance_improvement_percent=0,
                confidence_score=0,
                reasoning="No valid test results found",
                test_results=test_results,
                recommendations=["Unable to generate recommendations due to test failures"]
            )
        
        optimal_memory = optimal_config['memory_size']
        optimal_pc = optimal_config.get('provisioned_concurrency')
        optimal_results = optimal_config['results']
        
        # Calculate improvements compared to baseline
        cost_savings_percent = 0
        performance_improvement_percent = 0
        confidence_score = 0.85  # Higher base confidence for continuous workloads
        
        if baseline_results:
            baseline_cost = self._calculate_cost(
                baseline_results.get('memory_mb', 128),
                baseline_results.get('avg_billed_duration', 1000)
            )
            optimal_cost = self._calculate_cost(
                optimal_memory,
                optimal_results.get('avg_billed_duration', 1000)
            )
            
            if baseline_cost > 0:
                cost_savings_percent = ((baseline_cost - optimal_cost) / baseline_cost) * 100
            
            baseline_duration = baseline_results.get('avg_duration', 1000)
            optimal_duration = optimal_results.get('avg_duration', 1000)
            
            if baseline_duration > 0:
                performance_improvement_percent = ((baseline_duration - optimal_duration) / baseline_duration) * 100
            
            # Adjust confidence based on stability metrics
            if 'sustained_performance' in optimal_results:
                stability_score = optimal_results['sustained_performance'].get('stability_score', 0)
                confidence_score = min(0.98, confidence_score + (stability_score * 0.1))
        
        # Generate reasoning
        reasoning_parts = []
        
        if optimal_pc:
            reasoning_parts.append(
                f"Provisioned concurrency of {optimal_pc} recommended for high-throughput continuous workload"
            )
        
        if cost_savings_percent > 0:
            reasoning_parts.append(f"Cost reduction of {cost_savings_percent:.1f}% achieved")
        elif cost_savings_percent < -5:
            reasoning_parts.append(f"Cost increase of {abs(cost_savings_percent):.1f}% justified by performance gains")
        
        if performance_improvement_percent > 0:
            reasoning_parts.append(f"Performance improvement of {performance_improvement_percent:.1f}% achieved")
        
        if 'sustained_performance' in optimal_results:
            stability_score = optimal_results['sustained_performance'].get('stability_score', 0)
            if stability_score > 0.9:
                reasoning_parts.append("Excellent performance stability under sustained load")
        
        if not reasoning_parts:
            reasoning_parts.append("Optimal configuration found based on continuous workload analysis")
        
        reasoning = ". ".join(reasoning_parts)
        
        # Generate recommendations
        recommendations = self._generate_actionable_recommendations(
            optimal_config, workload, test_results
        )
        
        return OptimizationResult(
            workload_type=workload.workload_type,
            optimal_memory=optimal_memory,
            optimal_provisioned_concurrency=optimal_pc,
            cost_savings_percent=cost_savings_percent,
            performance_improvement_percent=performance_improvement_percent,
            confidence_score=confidence_score,
            reasoning=reasoning,
            test_results=test_results,
            recommendations=recommendations
        )

    def _generate_actionable_recommendations(
        self,
        optimal_config: Dict[str, Any],
        workload: WorkloadCharacteristics,
        test_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable recommendations for continuous workloads.
        
        Args:
            optimal_config: Optimal configuration
            workload: Workload characteristics
            test_results: All test results
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        optimal_memory = optimal_config['memory_size']
        optimal_pc = optimal_config.get('provisioned_concurrency')
        optimal_results = optimal_config['results']
        
        # Memory recommendation
        recommendations.append(
            f"Update Lambda function memory to {optimal_memory}MB for optimal continuous performance"
        )
        
        # Provisioned concurrency recommendation
        if optimal_pc:
            recommendations.append(
                f"Configure {optimal_pc} provisioned concurrency instances for consistent throughput"
            )
        
        # Scaling recommendations
        if 'throughput_scaling' in optimal_results:
            optimal_concurrency = optimal_results['throughput_scaling'].get('optimal_concurrency', 1)
            recommendations.append(
                f"Consider adjusting concurrent execution settings to {optimal_concurrency} for optimal throughput"
            )
        
        # Monitoring recommendations for continuous workloads
        recommendations.append(
            "Implement enhanced CloudWatch monitoring for duration trends, memory utilization, and error rates"
        )
        
        recommendations.append(
            "Set up CloudWatch alarms for performance degradation detection in sustained workloads"
        )
        
        # Cost optimization for high-volume workloads
        if workload.cost_sensitivity == "high":
            recommendations.append(
                "Implement cost monitoring dashboards and automated cost alerts for high-volume continuous workloads"
            )
        
        # Architecture recommendations
        if workload.invocation_frequency == "high":
            recommendations.append(
                "Consider implementing message batching or connection pooling to improve efficiency"
            )
        
        # Performance stability recommendations
        if 'sustained_performance' in optimal_results:
            stability_score = optimal_results['sustained_performance'].get('stability_score', 0)
            if stability_score < 0.8:
                recommendations.append(
                    "Investigate potential memory leaks or performance degradation under sustained load"
                )
        
        return recommendations

    def _extract_peak_concurrency(self, concurrency_metrics: Dict[str, Any]) -> Optional[int]:
        """
        Extract peak concurrency from metrics.
        
        Args:
            concurrency_metrics: Concurrency metrics data
            
        Returns:
            Peak concurrency value or None
        """
        if not concurrency_metrics or 'concurrent_executions' not in concurrency_metrics:
            return None
        
        concurrent_data = concurrency_metrics['concurrent_executions']
        if not concurrent_data:
            return None
        
        max_concurrent = max(point.get('value', 0) for point in concurrent_data)
        return int(max_concurrent) if max_concurrent > 0 else None

    def _simulate_provisioned_concurrency_test(
        self, memory_size: int, pc_level: int, iterations: int
    ) -> Dict[str, Any]:
        """
        Simulate provisioned concurrency test results for continuous workloads.
        
        Args:
            memory_size: Memory configuration
            pc_level: Provisioned concurrency level
            iterations: Number of iterations
            
        Returns:
            Simulated test results
        """
        import random
        
        # Simulate improved performance with provisioned concurrency
        base_results = self._simulate_memory_test(memory_size, iterations)
        
        # Significantly reduce cold starts
        base_results['cold_starts'] = 0
        
        # Improve consistency and throughput
        for execution in base_results['executions']:
            execution['duration'] *= 0.85  # 15% improvement in duration
            execution['cold_start'] = False
        
        # Add simulated sustained performance metrics
        base_results['sustained_performance'] = {
            'performance_waves': [850, 845, 848, 847, 849],  # More consistent
            'stability_score': 0.95,
            'performance_degradation': False
        }
        
        # Add simulated throughput scaling metrics
        base_results['throughput_scaling'] = {
            'optimal_concurrency': min(pc_level, 20),
            'scaling_efficiency': 0.85
        }
        
        base_results['provisioned_concurrency'] = pc_level
        base_results.update(self._calculate_test_metrics(base_results['executions']))
        
        return base_results