"""Scheduled workload optimization strategy."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from .workload_strategy import (
    WorkloadStrategy, WorkloadType, WorkloadCharacteristics,
    TestingStrategy, OptimizationResult
)

logger = logging.getLogger(__name__)


class ScheduledStrategy(WorkloadStrategy):
    """Strategy for optimizing scheduled, event-driven workloads."""

    async def analyze_workload(self) -> WorkloadCharacteristics:
        """
        Analyze scheduled workload characteristics.
        
        Returns:
            WorkloadCharacteristics: Analyzed workload characteristics
        """
        self.logger.info("Analyzing scheduled workload characteristics")
        
        # Get function metrics from multiple time periods to understand schedule patterns
        recent_metrics = await self._get_function_metrics(hours_back=24)
        weekly_metrics = await self._get_function_metrics(hours_back=168)  # 7 days
        concurrency_metrics = await self._get_concurrent_executions_metrics(hours_back=168)
        
        # Determine characteristics based on scheduled patterns
        frequency = self._classify_scheduled_frequency(recent_metrics, weekly_metrics)
        traffic_pattern = self._analyze_scheduled_pattern(weekly_metrics)
        memory_utilization = self._estimate_memory_utilization(recent_metrics)
        time_windows = self._identify_execution_windows(weekly_metrics)
        
        # Scheduled workloads have specific characteristics
        workload = WorkloadCharacteristics(
            workload_type=WorkloadType.SCHEDULED,
            invocation_frequency=frequency,
            traffic_pattern=traffic_pattern,
            cold_start_sensitivity="medium",  # Moderate sensitivity, depends on schedule frequency
            cost_sensitivity=self._determine_cost_sensitivity(WorkloadType.SCHEDULED, frequency),
            response_time_requirement="moderate",  # Usually not user-facing
            peak_concurrency=self._extract_peak_concurrency(concurrency_metrics),
            memory_utilization=memory_utilization,
            time_windows=time_windows
        )
        
        self.logger.info(
            f"Workload analysis complete: frequency={frequency}, pattern={traffic_pattern}, "
            f"time_windows={len(time_windows) if time_windows else 0}"
        )
        return workload

    def create_testing_strategy(self, workload: WorkloadCharacteristics) -> TestingStrategy:
        """
        Create testing strategy optimized for scheduled workloads.
        
        Args:
            workload: Analyzed workload characteristics
            
        Returns:
            TestingStrategy: Strategy for testing this workload
        """
        self.logger.info("Creating scheduled workload testing strategy")
        
        # Base memory sizes for scheduled workloads (balance of cost and performance)
        if workload.cost_sensitivity == "high":
            # Focus on cost-efficient memory sizes
            memory_sizes = [128, 256, 512, 768, 1024]
        elif workload.traffic_pattern == "bursty":
            # Higher memory for burst handling
            memory_sizes = [256, 512, 768, 1024, 1536, 2048]
        else:
            # Balanced approach
            memory_sizes = [128, 256, 512, 768, 1024, 1536]
        
        # Adjust based on execution frequency
        if workload.invocation_frequency == "high":
            # More memory options for high-frequency scheduled jobs
            memory_sizes.extend([2048, 3008])
        
        # Filter to valid memory sizes and remove duplicates
        memory_sizes = sorted(list(set([size for size in memory_sizes if 128 <= size <= 10240])))
        
        # Adjust iterations based on schedule characteristics
        if workload.traffic_pattern == "predictable":
            iterations = 12  # Fewer iterations for predictable patterns
            concurrent_executions = 4
        elif workload.traffic_pattern == "bursty":
            iterations = 18  # More iterations to capture burst behavior
            concurrent_executions = 8
        else:
            iterations = 15
            concurrent_executions = 6
        
        # Adjust based on invocation frequency
        if workload.invocation_frequency == "high":
            iterations += 5
            concurrent_executions = min(concurrent_executions + 3, 15)
        
        # Warmup considerations for scheduled workloads
        if workload.invocation_frequency == "low":
            warmup_runs = 3  # More warmup for infrequent scheduled jobs
        else:
            warmup_runs = 2
        
        # Time-based testing for scheduled workloads
        test_duration_minutes = None
        if workload.time_windows and len(workload.time_windows) > 0:
            # Enable time-based testing if we have identified execution windows
            test_duration_minutes = 5
        
        # Determine if we should test provisioned concurrency
        provisioned_concurrency_levels = None
        if (workload.invocation_frequency == "high" and 
            workload.traffic_pattern == "bursty" and
            workload.peak_concurrency and workload.peak_concurrency > 10):
            # Test provisioned concurrency for high-frequency, bursty scheduled workloads
            max_provisioned = min(workload.peak_concurrency, 200)
            provisioned_concurrency_levels = [
                max_provisioned // 3,
                max_provisioned // 2,
                max_provisioned
            ]
            provisioned_concurrency_levels = [pc for pc in provisioned_concurrency_levels if pc > 0]
        
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
        Execute optimization for scheduled workloads.
        
        Args:
            workload: Workload characteristics
            strategy: Testing strategy
            
        Returns:
            OptimizationResult: Optimization results
        """
        self.logger.info("Executing scheduled workload optimization")
        
        # Store original configuration
        original_config = await self.aws_provider.get_function_configuration()
        original_memory = original_config['MemorySize']
        
        test_results = {}
        baseline_results = None
        
        try:
            # Test baseline configuration first
            if original_memory in strategy.memory_sizes:
                self.logger.info(f"Testing baseline configuration: {original_memory}MB")
                baseline_results = await self._test_scheduled_configuration(
                    original_memory,
                    strategy.iterations_per_memory,
                    strategy.test_duration_minutes,
                    workload
                )
                test_results[original_memory] = baseline_results
            
            # Test each memory configuration
            for memory_size in strategy.memory_sizes:
                if memory_size == original_memory:
                    continue  # Already tested as baseline
                
                self.logger.info(f"Testing memory configuration: {memory_size}MB")
                
                # Test standard configuration
                config_results = await self._test_scheduled_configuration(
                    memory_size,
                    strategy.iterations_per_memory,
                    strategy.test_duration_minutes,
                    workload
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

    async def _test_scheduled_configuration(
        self, 
        memory_size: int, 
        iterations: int,
        test_duration_minutes: Optional[int] = None,
        workload: Optional[WorkloadCharacteristics] = None
    ) -> Dict[str, Any]:
        """
        Test memory configuration with scheduled workload patterns.
        
        Args:
            memory_size: Memory configuration to test
            iterations: Number of test iterations
            test_duration_minutes: Duration for time-based testing
            workload: Workload characteristics
            
        Returns:
            Test results with scheduled workload metrics
        """
        self.logger.info(f"Testing scheduled configuration: {memory_size}MB")
        
        # Standard testing
        base_results = await self._test_memory_configuration_with_monitoring(
            memory_size, iterations, test_duration_minutes
        )
        
        # Add scheduled-specific testing
        schedule_simulation = await self._test_schedule_simulation(memory_size, workload)
        resource_efficiency = await self._test_resource_efficiency(memory_size)
        idle_recovery = await self._test_idle_recovery(memory_size)
        
        # Combine results
        base_results['schedule_simulation'] = schedule_simulation
        base_results['resource_efficiency'] = resource_efficiency
        base_results['idle_recovery'] = idle_recovery
        base_results['workload_type'] = 'scheduled'
        
        return base_results

    async def _test_schedule_simulation(
        self, 
        memory_size: int, 
        workload: Optional[WorkloadCharacteristics] = None
    ) -> Dict[str, Any]:
        """
        Simulate scheduled execution patterns.
        
        Args:
            memory_size: Memory configuration to test
            workload: Workload characteristics
            
        Returns:
            Schedule simulation metrics
        """
        self.logger.info("Testing schedule simulation patterns")
        
        simulation_results = {
            'immediate_start_performance': {},
            'post_idle_performance': {},
            'burst_handling': {},
            'schedule_readiness_score': 0.0
        }
        
        if self.config.dry_run:
            # Simulate schedule results
            simulation_results['immediate_start_performance'] = {
                'avg_duration': 800,
                'cold_start_ratio': 0.8,
                'success_rate': 98
            }
            simulation_results['post_idle_performance'] = {
                'avg_duration': 850,
                'cold_start_ratio': 0.9,
                'success_rate': 97
            }
            simulation_results['schedule_readiness_score'] = 0.85
            return simulation_results
        
        # Test immediate start performance (simulating scheduled trigger)
        self.logger.info("Testing immediate start performance")
        immediate_results = []
        
        for i in range(5):
            try:
                result = await self.aws_provider.invoke_function(self.config.payload)
                immediate_results.append(result)
                if i < 4:  # Brief pause between immediate executions
                    await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Immediate start execution {i} failed: {e}")
        
        if immediate_results:
            simulation_results['immediate_start_performance'] = {
                'avg_duration': sum(r.get('duration', 0) for r in immediate_results) / len(immediate_results),
                'cold_start_ratio': sum(1 for r in immediate_results if r.get('cold_start', False)) / len(immediate_results),
                'success_rate': sum(1 for r in immediate_results if not r.get('error')) / len(immediate_results) * 100
            }
        
        # Simulate idle period (typical between scheduled executions)
        self.logger.info("Simulating idle period (30 seconds)")
        await asyncio.sleep(30)
        
        # Test post-idle performance
        self.logger.info("Testing post-idle performance")
        post_idle_results = []
        
        for i in range(3):
            try:
                result = await self.aws_provider.invoke_function(self.config.payload)
                post_idle_results.append(result)
                if i < 2:
                    await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Post-idle execution {i} failed: {e}")
        
        if post_idle_results:
            simulation_results['post_idle_performance'] = {
                'avg_duration': sum(r.get('duration', 0) for r in post_idle_results) / len(post_idle_results),
                'cold_start_ratio': sum(1 for r in post_idle_results if r.get('cold_start', False)) / len(post_idle_results),
                'success_rate': sum(1 for r in post_idle_results if not r.get('error')) / len(post_idle_results) * 100
            }
        
        # Test burst handling (simulating multiple scheduled jobs triggered simultaneously)
        if workload and workload.traffic_pattern == "bursty":
            burst_results = await self._test_burst_handling(memory_size)
            simulation_results['burst_handling'] = burst_results
        
        # Calculate schedule readiness score
        immediate_perf = simulation_results.get('immediate_start_performance', {})
        post_idle_perf = simulation_results.get('post_idle_performance', {})
        
        readiness_factors = []
        
        # Factor 1: Cold start impact
        immediate_cold_ratio = immediate_perf.get('cold_start_ratio', 1.0)
        post_idle_cold_ratio = post_idle_perf.get('cold_start_ratio', 1.0)
        cold_start_score = 1 - ((immediate_cold_ratio + post_idle_cold_ratio) / 2)
        readiness_factors.append(cold_start_score)
        
        # Factor 2: Performance consistency
        immediate_duration = immediate_perf.get('avg_duration', 1000)
        post_idle_duration = post_idle_perf.get('avg_duration', 1000)
        if immediate_duration > 0:
            consistency_score = 1 - abs(post_idle_duration - immediate_duration) / immediate_duration
            readiness_factors.append(max(0, consistency_score))
        
        # Factor 3: Success rate
        immediate_success = immediate_perf.get('success_rate', 100) / 100
        post_idle_success = post_idle_perf.get('success_rate', 100) / 100
        success_score = (immediate_success + post_idle_success) / 2
        readiness_factors.append(success_score)
        
        simulation_results['schedule_readiness_score'] = sum(readiness_factors) / len(readiness_factors) if readiness_factors else 0
        
        return simulation_results

    async def _test_burst_handling(self, memory_size: int) -> Dict[str, Any]:
        """
        Test burst handling capabilities for scheduled workloads.
        
        Args:
            memory_size: Memory configuration to test
            
        Returns:
            Burst handling metrics
        """
        self.logger.info("Testing burst handling capabilities")
        
        burst_results = {
            'concurrent_performance': {},
            'burst_success_rate': 0.0,
            'burst_latency_impact': 0.0
        }
        
        # Test concurrent execution (simulating multiple scheduled jobs)
        burst_size = min(self.config.concurrent_executions, 8)
        
        start_time = datetime.now()
        tasks = []
        
        for i in range(burst_size):
            task = asyncio.create_task(self.aws_provider.invoke_function(self.config.payload))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        total_duration = (end_time - start_time).total_seconds()
        
        # Analyze burst results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        if successful_results:
            avg_duration = sum(r.get('duration', 0) for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / burst_size * 100
            
            burst_results['concurrent_performance'] = {
                'total_duration': total_duration,
                'avg_individual_duration': avg_duration,
                'concurrent_executions': burst_size,
                'throughput': len(successful_results) / total_duration if total_duration > 0 else 0
            }
            burst_results['burst_success_rate'] = success_rate
            
            # Calculate latency impact compared to single execution
            if hasattr(self, '_single_execution_baseline'):
                baseline_duration = self._single_execution_baseline
                latency_impact = (avg_duration - baseline_duration) / baseline_duration if baseline_duration > 0 else 0
                burst_results['burst_latency_impact'] = latency_impact
        
        return burst_results

    async def _test_resource_efficiency(self, memory_size: int) -> Dict[str, Any]:
        """
        Test resource efficiency for scheduled workloads.
        
        Args:
            memory_size: Memory configuration to test
            
        Returns:
            Resource efficiency metrics
        """
        self.logger.info("Testing resource efficiency")
        
        efficiency_results = {
            'memory_utilization_ratio': 0.0,
            'cost_per_execution': 0.0,
            'resource_waste_score': 0.0,
            'efficiency_rating': 'unknown'
        }
        
        if self.config.dry_run:
            # Simulate efficiency results
            efficiency_results['memory_utilization_ratio'] = 0.65
            efficiency_results['cost_per_execution'] = 0.000025
            efficiency_results['resource_waste_score'] = 0.35
            efficiency_results['efficiency_rating'] = 'good'
            return efficiency_results
        
        # Run test executions to measure resource usage
        test_executions = []
        
        for i in range(8):
            try:
                result = await self.aws_provider.invoke_function(self.config.payload)
                test_executions.append(result)
                await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Efficiency test execution {i} failed: {e}")
        
        if test_executions:
            # Calculate memory utilization ratio
            memory_used_values = [e.get('memory_used', memory_size * 0.7) for e in test_executions]
            avg_memory_used = sum(memory_used_values) / len(memory_used_values)
            utilization_ratio = avg_memory_used / memory_size
            
            # Calculate cost per execution
            durations = [e.get('billed_duration', e.get('duration', 0)) for e in test_executions]
            avg_duration = sum(durations) / len(durations)
            cost_per_execution = self._calculate_cost(memory_size, avg_duration)
            
            # Calculate resource waste score
            waste_score = 1 - utilization_ratio
            
            # Determine efficiency rating
            if utilization_ratio > 0.8:
                rating = 'excellent'
            elif utilization_ratio > 0.6:
                rating = 'good'
            elif utilization_ratio > 0.4:
                rating = 'fair'
            else:
                rating = 'poor'
            
            efficiency_results.update({
                'memory_utilization_ratio': utilization_ratio,
                'cost_per_execution': cost_per_execution,
                'resource_waste_score': waste_score,
                'efficiency_rating': rating
            })
        
        return efficiency_results

    async def _test_idle_recovery(self, memory_size: int) -> Dict[str, Any]:
        """
        Test recovery performance after idle periods.
        
        Args:
            memory_size: Memory configuration to test
            
        Returns:
            Idle recovery metrics
        """
        self.logger.info("Testing idle recovery performance")
        
        recovery_results = {
            'pre_idle_performance': {},
            'post_idle_performance': {},
            'recovery_time_seconds': 0.0,
            'idle_impact_score': 0.0
        }
        
        if self.config.dry_run:
            # Simulate recovery results
            recovery_results['pre_idle_performance'] = {'avg_duration': 750}
            recovery_results['post_idle_performance'] = {'avg_duration': 850}
            recovery_results['recovery_time_seconds'] = 2.5
            recovery_results['idle_impact_score'] = 0.15
            return recovery_results
        
        # Test pre-idle performance
        pre_idle_result = await self.aws_provider.invoke_function(self.config.payload)
        recovery_results['pre_idle_performance'] = {
            'duration': pre_idle_result.get('duration', 0),
            'cold_start': pre_idle_result.get('cold_start', False)
        }
        
        # Simulate extended idle period (5 minutes)
        self.logger.info("Simulating extended idle period (5 minutes)")
        await asyncio.sleep(300)
        
        # Test post-idle recovery
        recovery_start = datetime.now()
        post_idle_result = await self.aws_provider.invoke_function(self.config.payload)
        recovery_end = datetime.now()
        
        recovery_time = (recovery_end - recovery_start).total_seconds()
        
        recovery_results['post_idle_performance'] = {
            'duration': post_idle_result.get('duration', 0),
            'cold_start': post_idle_result.get('cold_start', False)
        }
        recovery_results['recovery_time_seconds'] = recovery_time
        
        # Calculate idle impact score
        pre_duration = recovery_results['pre_idle_performance'].get('duration', 0)
        post_duration = recovery_results['post_idle_performance'].get('duration', 0)
        
        if pre_duration > 0:
            impact_score = (post_duration - pre_duration) / pre_duration
            recovery_results['idle_impact_score'] = max(0, impact_score)
        
        return recovery_results

    async def _test_with_provisioned_concurrency(
        self, 
        memory_size: int, 
        pc_level: int, 
        iterations: int,
        test_results: Dict[str, Any]
    ):
        """
        Test configuration with provisioned concurrency for scheduled workloads.
        
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
            await asyncio.sleep(45)  # Wait for provisioned concurrency to be ready
            
            # Run tests with scheduled workload analysis
            results = await self._test_scheduled_configuration(memory_size, iterations)
            results['provisioned_concurrency'] = pc_level
            
            # Remove provisioned concurrency
            await self.aws_provider.delete_provisioned_concurrency()
        
        # Store results with provisioned concurrency suffix
        key = f"{memory_size}_pc_{pc_level}"
        test_results[key] = results

    def _classify_scheduled_frequency(
        self, 
        recent_metrics: Dict[str, Any], 
        weekly_metrics: Dict[str, Any]
    ) -> str:
        """
        Classify invocation frequency for scheduled workloads.
        
        Args:
            recent_metrics: Recent 24-hour metrics
            weekly_metrics: Weekly metrics
            
        Returns:
            Frequency classification: "high", "medium", "low"
        """
        # Analyze invocation patterns
        recent_invocations = recent_metrics.get('invocations', [])
        weekly_invocations = weekly_metrics.get('invocations', [])
        
        if not recent_invocations or not weekly_invocations:
            return "low"
        
        # Calculate daily averages
        recent_total = sum(point.get('value', 0) for point in recent_invocations)
        weekly_total = sum(point.get('value', 0) for point in weekly_invocations)
        weekly_daily_avg = weekly_total / 7
        
        # Compare recent activity to weekly average
        activity_ratio = recent_total / max(weekly_daily_avg, 1)
        
        # Classify based on absolute numbers and activity patterns
        if recent_total > 500 or (weekly_daily_avg > 300 and activity_ratio > 0.8):
            return "high"
        elif recent_total > 100 or (weekly_daily_avg > 50 and activity_ratio > 0.5):
            return "medium"
        else:
            return "low"

    def _analyze_scheduled_pattern(self, weekly_metrics: Dict[str, Any]) -> str:
        """
        Analyze traffic pattern for scheduled workloads.
        
        Args:
            weekly_metrics: Weekly metrics data
            
        Returns:
            Traffic pattern classification
        """
        if not weekly_metrics or 'invocations' not in weekly_metrics:
            return "unpredictable"
        
        invocations = weekly_metrics['invocations']
        if not invocations:
            return "unpredictable"
        
        # Group by day/hour to detect patterns
        daily_totals = {}
        hourly_patterns = {}
        
        for point in invocations:
            if 'timestamp' in point:
                timestamp = point['timestamp']
                # Simplified pattern detection
                day = timestamp.strftime('%A') if hasattr(timestamp, 'strftime') else 'unknown'
                hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
                
                daily_totals[day] = daily_totals.get(day, 0) + point.get('value', 0)
                hourly_patterns[hour] = hourly_patterns.get(hour, 0) + point.get('value', 0)
        
        # Analyze patterns
        if len(daily_totals) > 0:
            daily_values = list(daily_totals.values())
            if len(daily_values) > 1:
                daily_variance = self._calculate_variance(daily_values)
                daily_mean = sum(daily_values) / len(daily_values)
                daily_cv = (daily_variance ** 0.5) / daily_mean if daily_mean > 0 else 1
                
                if daily_cv < 0.3:
                    return "predictable"
                elif daily_cv < 0.7:
                    return "steady"
                else:
                    return "bursty"
        
        return "unpredictable"

    def _identify_execution_windows(self, weekly_metrics: Dict[str, Any]) -> Optional[List[Tuple[datetime, datetime]]]:
        """
        Identify typical execution time windows from metrics.
        
        Args:
            weekly_metrics: Weekly metrics data
            
        Returns:
            List of execution time windows or None
        """
        if not weekly_metrics or 'invocations' not in weekly_metrics:
            return None
        
        invocations = weekly_metrics['invocations']
        if not invocations:
            return None
        
        # Simplified window detection - look for peak hours
        hourly_activity = {}
        
        for point in invocations:
            if 'timestamp' in point and hasattr(point['timestamp'], 'hour'):
                hour = point['timestamp'].hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + point.get('value', 0)
        
        if not hourly_activity:
            return None
        
        # Find peak activity hours (simplified)
        sorted_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)
        peak_hours = [hour for hour, activity in sorted_hours[:4] if activity > 0]  # Top 4 hours
        
        if peak_hours:
            # Create time windows around peak hours (simplified)
            windows = []
            for hour in peak_hours:
                start = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                end = start.replace(hour=(hour + 1) % 24)
                windows.append((start, end))
            
            return windows
        
        return None

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _find_optimal_configuration(
        self, 
        test_results: Dict[str, Any], 
        workload: WorkloadCharacteristics
    ) -> Dict[str, Any]:
        """
        Find optimal configuration for scheduled workloads.
        
        Args:
            test_results: All test results
            workload: Workload characteristics
            
        Returns:
            Optimal configuration details
        """
        self.logger.info("Analyzing test results for scheduled workload optimization")
        
        # Filter out failed tests
        valid_results = {
            k: v for k, v in test_results.items() 
            if v.get('success_rate', 0) >= 92  # Slightly lower threshold for scheduled workloads
        }
        
        if not valid_results:
            self.logger.warning("No valid test results found")
            return {}
        
        # Calculate scores for each configuration
        scored_configs = []
        
        for config_key, results in valid_results.items():
            score = self._calculate_scheduled_configuration_score(results, workload)
            
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

    def _calculate_scheduled_configuration_score(
        self, 
        results: Dict[str, Any], 
        workload: WorkloadCharacteristics
    ) -> float:
        """
        Calculate a score for scheduled workload configuration.
        
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
        
        # Base performance score
        performance_score = 1000 / max(avg_duration, 1)
        
        # Schedule readiness bonus
        readiness_bonus = 0
        if 'schedule_simulation' in results:
            readiness_score = results['schedule_simulation'].get('schedule_readiness_score', 0)
            readiness_bonus = readiness_score * 300  # High weight for schedule readiness
        
        # Resource efficiency bonus
        efficiency_bonus = 0
        if 'resource_efficiency' in results:
            utilization_ratio = results['resource_efficiency'].get('memory_utilization_ratio', 0)
            efficiency_bonus = utilization_ratio * 100
        
        # Idle recovery bonus
        recovery_bonus = 0
        if 'idle_recovery' in results:
            idle_impact = results['idle_recovery'].get('idle_impact_score', 1.0)
            recovery_bonus = max(0, (1 - idle_impact) * 150)
        
        # Cost penalty (very important for scheduled workloads)
        cost_penalty = 0
        if workload.cost_sensitivity == "high":
            cost_penalty = cost_per_invocation * 3000  # Higher penalty for scheduled workloads
        elif workload.cost_sensitivity == "medium":
            cost_penalty = cost_per_invocation * 1500
        else:
            cost_penalty = cost_per_invocation * 750
        
        # Success rate bonus
        success_bonus = success_rate * 8
        
        # Burst handling bonus (if applicable)
        burst_bonus = 0
        if 'schedule_simulation' in results and 'burst_handling' in results['schedule_simulation']:
            burst_success_rate = results['schedule_simulation']['burst_handling'].get('burst_success_rate', 0)
            burst_bonus = burst_success_rate * 2
        
        # Calculate final score
        score = (performance_score + readiness_bonus + efficiency_bonus + 
                recovery_bonus + success_bonus + burst_bonus - cost_penalty)
        
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
        Generate optimization recommendation for scheduled workloads.
        
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
        confidence_score = 0.8  # Base confidence for scheduled workloads
        
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
            
            # Adjust confidence based on schedule readiness
            if 'schedule_simulation' in optimal_results:
                readiness_score = optimal_results['schedule_simulation'].get('schedule_readiness_score', 0)
                confidence_score = min(0.95, confidence_score + (readiness_score * 0.15))
        
        # Generate reasoning
        reasoning_parts = []
        
        if optimal_pc:
            reasoning_parts.append(
                f"Provisioned concurrency of {optimal_pc} recommended for reliable scheduled execution"
            )
        
        if cost_savings_percent > 0:
            reasoning_parts.append(f"Cost reduction of {cost_savings_percent:.1f}% achieved")
        elif cost_savings_percent < -3:
            reasoning_parts.append(f"Cost increase of {abs(cost_savings_percent):.1f}% justified by reliability improvements")
        
        if performance_improvement_percent > 0:
            reasoning_parts.append(f"Performance improvement of {performance_improvement_percent:.1f}% achieved")
        
        if 'schedule_simulation' in optimal_results:
            readiness_score = optimal_results['schedule_simulation'].get('schedule_readiness_score', 0)
            if readiness_score > 0.85:
                reasoning_parts.append("Excellent schedule readiness and idle recovery performance")
        
        if 'resource_efficiency' in optimal_results:
            efficiency_rating = optimal_results['resource_efficiency'].get('efficiency_rating', 'unknown')
            if efficiency_rating in ['excellent', 'good']:
                reasoning_parts.append(f"Resource utilization rated as {efficiency_rating}")
        
        if not reasoning_parts:
            reasoning_parts.append("Optimal configuration found based on scheduled workload analysis")
        
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
        Generate actionable recommendations for scheduled workloads.
        
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
            f"Update Lambda function memory to {optimal_memory}MB for optimal scheduled execution"
        )
        
        # Provisioned concurrency recommendation
        if optimal_pc:
            recommendations.append(
                f"Configure {optimal_pc} provisioned concurrency instances to ensure consistent scheduled performance"
            )
        elif workload.invocation_frequency == "high":
            recommendations.append(
                "Consider implementing function warming strategies for high-frequency scheduled workloads"
            )
        
        # Schedule-specific monitoring
        recommendations.append(
            "Implement CloudWatch alarms for failed scheduled executions and duration spikes"
        )
        
        recommendations.append(
            "Monitor cold start ratios immediately after idle periods to validate configuration"
        )
        
        # Cost optimization for scheduled workloads
        if workload.cost_sensitivity == "high":
            recommendations.append(
                "Review scheduled execution frequency and consider batching operations to reduce costs"
            )
        
        # Resource efficiency recommendations
        if 'resource_efficiency' in optimal_results:
            efficiency_rating = optimal_results['resource_efficiency'].get('efficiency_rating', 'unknown')
            if efficiency_rating in ['fair', 'poor']:
                recommendations.append(
                    "Consider code optimization to improve memory utilization efficiency"
                )
        
        # Schedule optimization recommendations
        if workload.time_windows:
            recommendations.append(
                "Consider adjusting scheduled execution times based on identified peak performance windows"
            )
        
        # Architecture recommendations
        if workload.traffic_pattern == "bursty":
            recommendations.append(
                "Implement error handling and retry mechanisms for burst execution scenarios"
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
        Simulate provisioned concurrency test results for scheduled workloads.
        
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
        
        # Eliminate cold starts for scheduled workloads
        base_results['cold_starts'] = 0
        
        # Improve performance and consistency
        for execution in base_results['executions']:
            execution['duration'] *= 0.8  # 20% improvement
            execution['cold_start'] = False
        
        # Add simulated schedule-specific metrics
        base_results['schedule_simulation'] = {
            'immediate_start_performance': {
                'avg_duration': base_results['avg_duration'] * 0.8,
                'cold_start_ratio': 0.0,
                'success_rate': 100
            },
            'post_idle_performance': {
                'avg_duration': base_results['avg_duration'] * 0.82,
                'cold_start_ratio': 0.0,
                'success_rate': 100
            },
            'schedule_readiness_score': 0.95
        }
        
        base_results['resource_efficiency'] = {
            'memory_utilization_ratio': 0.7,
            'efficiency_rating': 'good'
        }
        
        base_results['provisioned_concurrency'] = pc_level
        base_results.update(self._calculate_test_metrics(base_results['executions']))
        
        return base_results