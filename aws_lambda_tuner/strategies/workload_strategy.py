"""Base strategy interface for workload-aware optimization."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Types of Lambda workloads."""
    ON_DEMAND = "on_demand"
    CONTINUOUS = "continuous"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    API_GATEWAY = "api_gateway"
    STREAM_PROCESSING = "stream_processing"


@dataclass
class WorkloadCharacteristics:
    """Characteristics of a Lambda workload."""
    workload_type: WorkloadType
    invocation_frequency: str  # "high", "medium", "low"
    traffic_pattern: str  # "steady", "bursty", "predictable", "unpredictable"
    cold_start_sensitivity: str  # "high", "medium", "low"
    cost_sensitivity: str  # "high", "medium", "low"
    response_time_requirement: str  # "strict", "moderate", "flexible"
    peak_concurrency: Optional[int] = None
    average_runtime: Optional[float] = None
    memory_utilization: Optional[float] = None
    time_windows: Optional[List[Tuple[datetime, datetime]]] = None


@dataclass
class TestingStrategy:
    """Strategy for testing a specific workload."""
    memory_sizes: List[int]
    iterations_per_memory: int
    concurrent_executions: int
    warmup_runs: int
    test_duration_minutes: Optional[int] = None
    time_based_testing: bool = False
    provisioned_concurrency_levels: Optional[List[int]] = None
    traffic_simulation: bool = False


@dataclass
class OptimizationResult:
    """Result of workload optimization."""
    workload_type: WorkloadType
    optimal_memory: int
    optimal_provisioned_concurrency: Optional[int]
    cost_savings_percent: float
    performance_improvement_percent: float
    confidence_score: float
    reasoning: str
    test_results: Dict[str, Any]
    recommendations: List[str]


class WorkloadStrategy(ABC):
    """Base class for workload-aware optimization strategies."""

    def __init__(self, config, aws_provider):
        """
        Initialize strategy.
        
        Args:
            config: Tuner configuration
            aws_provider: AWS provider instance
        """
        self.config = config
        self.aws_provider = aws_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def analyze_workload(self) -> WorkloadCharacteristics:
        """
        Analyze the Lambda function workload to determine characteristics.
        
        Returns:
            WorkloadCharacteristics: Analyzed workload characteristics
        """
        pass

    @abstractmethod
    def create_testing_strategy(self, workload: WorkloadCharacteristics) -> TestingStrategy:
        """
        Create a testing strategy based on workload characteristics.
        
        Args:
            workload: Analyzed workload characteristics
            
        Returns:
            TestingStrategy: Strategy for testing this workload
        """
        pass

    @abstractmethod
    async def execute_optimization(
        self, 
        workload: WorkloadCharacteristics, 
        strategy: TestingStrategy
    ) -> OptimizationResult:
        """
        Execute optimization based on workload and testing strategy.
        
        Args:
            workload: Workload characteristics
            strategy: Testing strategy
            
        Returns:
            OptimizationResult: Results of optimization
        """
        pass

    async def optimize(self) -> OptimizationResult:
        """
        Run complete optimization workflow.
        
        Returns:
            OptimizationResult: Optimization results
        """
        self.logger.info(f"Starting {self.__class__.__name__} optimization")
        
        # Analyze workload
        workload = await self.analyze_workload()
        self.logger.info(f"Detected workload type: {workload.workload_type.value}")
        
        # Create testing strategy
        testing_strategy = self.create_testing_strategy(workload)
        self.logger.info(f"Created testing strategy with {len(testing_strategy.memory_sizes)} memory configurations")
        
        # Execute optimization
        result = await self.execute_optimization(workload, testing_strategy)
        self.logger.info(f"Optimization completed with {result.confidence_score:.2f} confidence score")
        
        return result

    async def _get_function_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get CloudWatch metrics for the function.
        
        Args:
            hours_back: Number of hours to look back for metrics
            
        Returns:
            Dict containing function metrics
        """
        try:
            return await self.aws_provider.get_function_metrics(hours_back)
        except Exception as e:
            self.logger.warning(f"Could not retrieve function metrics: {e}")
            return {}

    async def _get_concurrent_executions_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get concurrent executions metrics.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Dict containing concurrency metrics
        """
        try:
            return await self.aws_provider.get_concurrent_executions_metrics(hours_back)
        except Exception as e:
            self.logger.warning(f"Could not retrieve concurrency metrics: {e}")
            return {}

    def _analyze_traffic_pattern(self, metrics: Dict[str, Any]) -> str:
        """
        Analyze traffic pattern from metrics.
        
        Args:
            metrics: CloudWatch metrics data
            
        Returns:
            Traffic pattern classification
        """
        if not metrics or 'invocations' not in metrics:
            return "unpredictable"
        
        invocations = metrics['invocations']
        if not invocations:
            return "unpredictable"
        
        # Calculate coefficient of variation
        values = [point['value'] for point in invocations if 'value' in point]
        if not values:
            return "unpredictable"
        
        if len(values) < 2:
            return "steady"
        
        mean_value = sum(values) / len(values)
        if mean_value == 0:
            return "steady"
        
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_value
        
        if cv < 0.2:
            return "steady"
        elif cv < 0.5:
            return "predictable"
        elif cv < 1.0:
            return "bursty"
        else:
            return "unpredictable"

    def _classify_invocation_frequency(self, metrics: Dict[str, Any]) -> str:
        """
        Classify invocation frequency from metrics.
        
        Args:
            metrics: CloudWatch metrics data
            
        Returns:
            Frequency classification: "high", "medium", "low"
        """
        if not metrics or 'invocations' not in metrics:
            return "low"
        
        invocations = metrics['invocations']
        if not invocations:
            return "low"
        
        # Calculate average invocations per hour
        total_invocations = sum(point.get('value', 0) for point in invocations)
        hours = len(invocations)
        avg_per_hour = total_invocations / max(hours, 1)
        
        if avg_per_hour > 1000:
            return "high"
        elif avg_per_hour > 100:
            return "medium"
        else:
            return "low"

    def _estimate_memory_utilization(self, metrics: Dict[str, Any]) -> Optional[float]:
        """
        Estimate memory utilization from metrics.
        
        Args:
            metrics: CloudWatch metrics data
            
        Returns:
            Memory utilization percentage or None if not available
        """
        if not metrics or 'memory_utilization' not in metrics:
            return None
        
        memory_data = metrics['memory_utilization']
        if not memory_data:
            return None
        
        # Calculate average memory utilization
        values = [point.get('value', 0) for point in memory_data]
        if not values:
            return None
        
        return sum(values) / len(values)

    def _determine_cold_start_sensitivity(self, workload_type: WorkloadType, metrics: Dict[str, Any]) -> str:
        """
        Determine cold start sensitivity based on workload type and metrics.
        
        Args:
            workload_type: Type of workload
            metrics: CloudWatch metrics data
            
        Returns:
            Cold start sensitivity: "high", "medium", "low"
        """
        # API Gateway and real-time workloads are more sensitive to cold starts
        if workload_type in [WorkloadType.API_GATEWAY, WorkloadType.ON_DEMAND]:
            return "high"
        elif workload_type in [WorkloadType.EVENT_DRIVEN, WorkloadType.STREAM_PROCESSING]:
            return "medium"
        else:
            return "low"

    def _determine_cost_sensitivity(self, workload_type: WorkloadType, frequency: str) -> str:
        """
        Determine cost sensitivity based on workload characteristics.
        
        Args:
            workload_type: Type of workload
            frequency: Invocation frequency
            
        Returns:
            Cost sensitivity: "high", "medium", "low"
        """
        # High frequency workloads are more cost sensitive
        if frequency == "high":
            return "high"
        elif frequency == "medium":
            return "medium"
        else:
            return "low"

    async def _test_memory_configuration_with_monitoring(
        self, 
        memory_size: int, 
        iterations: int,
        monitor_duration_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test memory configuration with enhanced monitoring.
        
        Args:
            memory_size: Memory configuration to test
            iterations: Number of test iterations
            monitor_duration_minutes: Duration to monitor (for time-based testing)
            
        Returns:
            Test results with enhanced metrics
        """
        self.logger.info(f"Testing memory configuration: {memory_size}MB")
        
        # Update function memory
        if not self.config.dry_run:
            await self.aws_provider.update_function_memory(memory_size)
            await asyncio.sleep(5)  # Wait for propagation
        
        results = {
            'memory_mb': memory_size,
            'executions': [],
            'cold_starts': 0,
            'errors': 0,
            'avg_duration': 0,
            'avg_billed_duration': 0,
            'p95_duration': 0,
            'p99_duration': 0,
            'cost_per_invocation': 0,
            'memory_utilization': [],
            'concurrent_executions_peak': 0
        }
        
        if self.config.dry_run:
            # Simulate results
            results.update(self._simulate_memory_test(memory_size, iterations))
            return results
        
        # Run warmup if specified
        if self.config.warmup_runs > 0:
            self.logger.info(f"Running {self.config.warmup_runs} warmup executions")
            await self._run_warmup_executions(memory_size, self.config.warmup_runs)
        
        # Run actual tests
        if monitor_duration_minutes:
            # Time-based testing
            end_time = datetime.now() + timedelta(minutes=monitor_duration_minutes)
            execution_count = 0
            
            while datetime.now() < end_time:
                execution_result = await self.aws_provider.invoke_function(self.config.payload)
                execution_result['execution_id'] = execution_count
                results['executions'].append(execution_result)
                execution_count += 1
                
                # Rate limiting
                await asyncio.sleep(1)
        else:
            # Iteration-based testing
            for i in range(iterations):
                execution_result = await self.aws_provider.invoke_function(self.config.payload)
                execution_result['execution_id'] = i
                results['executions'].append(execution_result)
                
                # Rate limiting
                if i < iterations - 1:
                    await asyncio.sleep(0.5)
        
        # Calculate metrics
        results.update(self._calculate_test_metrics(results['executions']))
        
        return results

    def _calculate_test_metrics(self, executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from execution results.
        
        Args:
            executions: List of execution results
            
        Returns:
            Calculated metrics
        """
        if not executions:
            return {}
        
        durations = [e.get('duration', 0) for e in executions if not e.get('error')]
        billed_durations = [e.get('billed_duration', 0) for e in executions if not e.get('error')]
        cold_starts = sum(1 for e in executions if e.get('cold_start', False))
        errors = sum(1 for e in executions if e.get('error'))
        
        if not durations:
            return {'errors': errors, 'cold_starts': cold_starts}
        
        # Sort for percentile calculations
        sorted_durations = sorted(durations)
        
        # Calculate percentiles
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)
        
        return {
            'cold_starts': cold_starts,
            'errors': errors,
            'avg_duration': sum(durations) / len(durations),
            'avg_billed_duration': sum(billed_durations) / len(billed_durations) if billed_durations else 0,
            'p95_duration': sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else sorted_durations[-1],
            'p99_duration': sorted_durations[p99_idx] if p99_idx < len(sorted_durations) else sorted_durations[-1],
            'min_duration': min(durations),
            'max_duration': max(durations),
            'success_rate': (len(executions) - errors) / len(executions) * 100
        }

    async def _run_warmup_executions(self, memory_size: int, count: int):
        """
        Run warmup executions to prepare function containers.
        
        Args:
            memory_size: Memory configuration
            count: Number of warmup executions
        """
        for i in range(count):
            try:
                await self.aws_provider.invoke_function(self.config.payload)
                await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Warmup execution {i} failed: {e}")

    def _simulate_memory_test(self, memory_size: int, iterations: int) -> Dict[str, Any]:
        """
        Simulate memory test results for dry run mode.
        
        Args:
            memory_size: Memory configuration
            iterations: Number of iterations
            
        Returns:
            Simulated test results
        """
        import random
        
        base_duration = 1000 / (memory_size / 512)  # Faster with more memory
        results = {
            'executions': [],
            'cold_starts': 1,  # First execution is typically a cold start
            'errors': max(0, int(iterations * 0.02)),  # 2% error rate
        }
        
        for i in range(iterations):
            duration = base_duration * random.uniform(0.8, 1.2)
            billed_duration = max(100, int(duration / 100) * 100)
            
            execution = {
                'execution_id': i,
                'duration': duration,
                'billed_duration': billed_duration,
                'cold_start': i == 0,
                'error': i < results['errors']
            }
            results['executions'].append(execution)
        
        # Calculate metrics
        results.update(self._calculate_test_metrics(results['executions']))
        return results