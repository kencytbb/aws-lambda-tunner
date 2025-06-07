"""On-demand workload optimization strategy."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .workload_strategy import (
    WorkloadStrategy, WorkloadType, WorkloadCharacteristics,
    TestingStrategy, OptimizationResult
)

logger = logging.getLogger(__name__)


class OnDemandStrategy(WorkloadStrategy):
    """Strategy for optimizing on-demand, API-driven workloads."""

    async def analyze_workload(self) -> WorkloadCharacteristics:
        """
        Analyze on-demand workload characteristics.
        
        Returns:
            WorkloadCharacteristics: Analyzed workload characteristics
        """
        self.logger.info("Analyzing on-demand workload characteristics")
        
        # Get function metrics from the last 24 hours
        metrics = await self._get_function_metrics(hours_back=24)
        concurrency_metrics = await self._get_concurrent_executions_metrics(hours_back=24)
        
        # Determine characteristics
        frequency = self._classify_invocation_frequency(metrics)
        traffic_pattern = self._analyze_traffic_pattern(metrics)
        memory_utilization = self._estimate_memory_utilization(metrics)
        
        # On-demand workloads typically have specific characteristics
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.ON_DEMAND,
            invocation_frequency=frequency,
            traffic_pattern=traffic_pattern,
            cold_start_sensitivity="high",  # On-demand workloads are sensitive to cold starts
            cost_sensitivity=self._determine_cost_sensitivity(WorkloadType.ON_DEMAND, frequency),
            response_time_requirement="strict",  # Usually user-facing
            peak_concurrency=self._extract_peak_concurrency(concurrency_metrics),
            memory_utilization=memory_utilization
        )
        
        self.logger.info(f"Workload analysis complete: frequency={frequency}, pattern={traffic_pattern}")
        return workload

    def create_testing_strategy(self, workload: WorkloadCharacteristics) -> TestingStrategy:
        """
        Create testing strategy optimized for on-demand workloads.
        
        Args:
            workload: Analyzed workload characteristics
            
        Returns:
            TestingStrategy: Strategy for testing this workload
        """
        self.logger.info("Creating on-demand testing strategy")
        
        # Base memory sizes for on-demand workloads (focus on performance)
        if workload.cold_start_sensitivity == "high":
            # Include higher memory sizes to reduce cold start impact
            memory_sizes = [512, 768, 1024, 1536, 2048, 3008]
        else:
            memory_sizes = [256, 512, 768, 1024, 1536]
        
        # Adjust based on cost sensitivity
        if workload.cost_sensitivity == "high":
            # Focus on lower memory sizes
            memory_sizes = [size for size in memory_sizes if size <= 1536]
        elif workload.cost_sensitivity == "low":
            # Include higher memory sizes for better performance
            memory_sizes.extend([4096, 5120])
        
        # Filter to valid memory sizes and remove duplicates
        memory_sizes = sorted(list(set([size for size in memory_sizes if 128 <= size <= 10240])))
        
        # Adjust iterations based on invocation frequency
        if workload.invocation_frequency == "high":
            iterations = 20  # More iterations for high-frequency workloads
            concurrent_executions = 10
        elif workload.invocation_frequency == "medium":
            iterations = 15
            concurrent_executions = 7
        else:
            iterations = 10
            concurrent_executions = 5
        
        # More warmup for cold start sensitive workloads
        warmup_runs = 5 if workload.cold_start_sensitivity == "high" else 3
        
        # Determine if we should test provisioned concurrency
        provisioned_concurrency_levels = None
        if (workload.invocation_frequency == "high" and 
            workload.cold_start_sensitivity == "high" and
            workload.peak_concurrency and workload.peak_concurrency > 5):
            # Test provisioned concurrency for high-frequency, cold-start-sensitive workloads
            max_provisioned = min(workload.peak_concurrency, 100)
            provisioned_concurrency_levels = [
                max_provisioned // 4,
                max_provisioned // 2,
                max_provisioned
            ]
            provisioned_concurrency_levels = [pc for pc in provisioned_concurrency_levels if pc > 0]
        
        strategy = TestingStrategy(
            memory_sizes=memory_sizes,
            iterations_per_memory=iterations,
            concurrent_executions=concurrent_executions,
            warmup_runs=warmup_runs,
            provisioned_concurrency_levels=provisioned_concurrency_levels,
            traffic_simulation=True  # Enable traffic simulation for realistic testing
        )
        
        self.logger.info(
            f"Testing strategy created: {len(memory_sizes)} memory configs, "
            f"{iterations} iterations each, "
            f"{'with' if provisioned_concurrency_levels else 'without'} provisioned concurrency testing"
        )
        
        return strategy

    async def execute_optimization(
        self, 
        workload: WorkloadCharacteristics, 
        strategy: TestingStrategy
    ) -> OptimizationResult:
        """
        Execute optimization for on-demand workloads.
        
        Args:
            workload: Workload characteristics
            strategy: Testing strategy
            
        Returns:
            OptimizationResult: Optimization results
        """
        self.logger.info("Executing on-demand optimization")
        
        # Store original configuration
        original_config = await self.aws_provider.get_function_configuration()
        original_memory = original_config['MemorySize']
        
        test_results = {}
        baseline_results = None
        
        try:
            # Test baseline configuration first
            if original_memory in strategy.memory_sizes:
                self.logger.info(f"Testing baseline configuration: {original_memory}MB")
                baseline_results = await self._test_memory_configuration_with_monitoring(
                    original_memory,
                    strategy.iterations_per_memory
                )
                test_results[original_memory] = baseline_results
            
            # Test each memory configuration
            for memory_size in strategy.memory_sizes:
                if memory_size == original_memory:
                    continue  # Already tested as baseline
                
                self.logger.info(f"Testing memory configuration: {memory_size}MB")
                
                # Test standard configuration
                config_results = await self._test_memory_configuration_with_monitoring(
                    memory_size,
                    strategy.iterations_per_memory
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

    async def _test_with_provisioned_concurrency(
        self, 
        memory_size: int, 
        pc_level: int, 
        iterations: int,
        test_results: Dict[str, Any]
    ):
        """
        Test configuration with provisioned concurrency.
        
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
            await asyncio.sleep(30)  # Wait for provisioned concurrency to be ready
            
            # Run tests
            results = await self._test_memory_configuration_with_monitoring(memory_size, iterations)
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
        Find optimal configuration based on test results and workload characteristics.
        
        Args:
            test_results: All test results
            workload: Workload characteristics
            
        Returns:
            Optimal configuration details
        """
        self.logger.info("Analyzing test results to find optimal configuration")
        
        # Filter out failed tests
        valid_results = {
            k: v for k, v in test_results.items() 
            if v.get('success_rate', 0) >= 95  # At least 95% success rate
        }
        
        if not valid_results:
            self.logger.warning("No valid test results found")
            return {}
        
        # Calculate scores for each configuration
        scored_configs = []
        
        for config_key, results in valid_results.items():
            score = self._calculate_configuration_score(results, workload)
            
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

    def _calculate_configuration_score(
        self, 
        results: Dict[str, Any], 
        workload: WorkloadCharacteristics
    ) -> float:
        """
        Calculate a score for a configuration based on workload requirements.
        
        Args:
            results: Test results for the configuration
            workload: Workload characteristics
            
        Returns:
            Configuration score (higher is better)
        """
        # Base metrics
        avg_duration = results.get('avg_duration', float('inf'))
        cold_starts = results.get('cold_starts', 0)
        total_executions = len(results.get('executions', []))
        success_rate = results.get('success_rate', 0)
        
        # Calculate cost (simplified)
        memory_size = results.get('memory_mb', 128)
        avg_billed_duration = results.get('avg_billed_duration', avg_duration)
        cost_per_invocation = self._calculate_cost(memory_size, avg_billed_duration)
        
        # Performance score (lower duration is better)
        performance_score = 1000 / max(avg_duration, 1)
        
        # Cold start penalty (higher penalty for cold start sensitive workloads)
        cold_start_penalty = 0
        if total_executions > 0:
            cold_start_rate = cold_starts / total_executions
            if workload.cold_start_sensitivity == "high":
                cold_start_penalty = cold_start_rate * 500
            elif workload.cold_start_sensitivity == "medium":
                cold_start_penalty = cold_start_rate * 200
            else:
                cold_start_penalty = cold_start_rate * 50
        
        # Cost penalty (higher penalty for cost sensitive workloads)
        cost_penalty = 0
        if workload.cost_sensitivity == "high":
            cost_penalty = cost_per_invocation * 1000
        elif workload.cost_sensitivity == "medium":
            cost_penalty = cost_per_invocation * 500
        else:
            cost_penalty = cost_per_invocation * 100
        
        # Success rate bonus
        success_bonus = success_rate * 10
        
        # Calculate final score
        score = performance_score + success_bonus - cold_start_penalty - cost_penalty
        
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
        # AWS Lambda pricing (simplified, as of 2024)
        # $0.0000166667 per GB-second for x86
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
        Generate optimization recommendation.
        
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
        confidence_score = 0.8  # Base confidence
        
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
            
            # Adjust confidence based on improvement magnitude
            total_improvement = abs(cost_savings_percent) + abs(performance_improvement_percent)
            if total_improvement > 20:
                confidence_score = 0.95
            elif total_improvement > 10:
                confidence_score = 0.9
            elif total_improvement > 5:
                confidence_score = 0.85
        
        # Generate reasoning
        reasoning_parts = []
        
        if optimal_pc:
            reasoning_parts.append(
                f"Provisioned concurrency of {optimal_pc} recommended to eliminate cold starts "
                f"for this high-frequency, cold-start-sensitive workload"
            )
        
        if cost_savings_percent > 0:
            reasoning_parts.append(f"Cost reduction of {cost_savings_percent:.1f}% achieved")
        
        if performance_improvement_percent > 0:
            reasoning_parts.append(f"Performance improvement of {performance_improvement_percent:.1f}% achieved")
        
        if not reasoning_parts:
            reasoning_parts.append("Optimal configuration found based on workload analysis")
        
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
        Generate actionable recommendations.
        
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
        
        # Memory recommendation
        recommendations.append(
            f"Update Lambda function memory to {optimal_memory}MB for optimal performance"
        )
        
        # Provisioned concurrency recommendation
        if optimal_pc:
            recommendations.append(
                f"Configure {optimal_pc} provisioned concurrency instances to eliminate cold starts"
            )
        elif workload.cold_start_sensitivity == "high":
            recommendations.append(
                "Consider implementing function warming strategies or connection pooling "
                "to reduce cold start impact"
            )
        
        # Monitoring recommendations
        recommendations.append(
            "Monitor CloudWatch metrics for duration, errors, and throttles after implementation"
        )
        
        # Cost optimization recommendations
        if workload.cost_sensitivity == "high":
            recommendations.append(
                "Review function execution patterns regularly to identify further cost optimization opportunities"
            )
        
        # Architecture recommendations
        if workload.invocation_frequency == "high":
            recommendations.append(
                "Consider implementing caching strategies to reduce Lambda invocations"
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
        Simulate provisioned concurrency test results.
        
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
        
        # Reduce cold starts significantly
        base_results['cold_starts'] = max(0, base_results['cold_starts'] - pc_level)
        
        # Improve average duration slightly (no cold start penalty)
        for execution in base_results['executions']:
            if execution.get('cold_start'):
                execution['duration'] *= 0.7  # 30% improvement
                execution['cold_start'] = False
        
        base_results['provisioned_concurrency'] = pc_level
        base_results.update(self._calculate_test_metrics(base_results['executions']))
        
        return base_results