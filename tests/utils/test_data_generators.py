"""Test data generators for AWS Lambda Tuner testing."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

from aws_lambda_tuner.models import (
    MemoryTestResult,
    Recommendation,
    PerformanceAnalysis,
    TuningResult
)


class TestDataGenerator:
    """Generate realistic test data for various scenarios."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducible tests."""
        if seed is not None:
            random.seed(seed)
        
        self.function_arns = [
            'arn:aws:lambda:us-east-1:123456789012:function:test-function',
            'arn:aws:lambda:us-west-2:123456789012:function:api-handler',
            'arn:aws:lambda:eu-west-1:123456789012:function:data-processor',
        ]
        
        self.memory_sizes = [128, 256, 512, 1024, 1536, 2048, 3008]
        
        # Performance characteristics by workload type
        self.workload_profiles = {
            'cpu_intensive': {
                'base_duration': 2000,
                'memory_sensitivity': 0.8,  # High sensitivity to memory
                'cold_start_impact': 0.3,
                'error_rate': 0.02
            },
            'io_bound': {
                'base_duration': 500,
                'memory_sensitivity': 0.2,  # Low sensitivity to memory
                'cold_start_impact': 0.5,
                'error_rate': 0.01
            },
            'memory_intensive': {
                'base_duration': 1000,
                'memory_sensitivity': 0.9,  # Very high sensitivity
                'cold_start_impact': 0.4,
                'error_rate': 0.03
            },
            'balanced': {
                'base_duration': 800,
                'memory_sensitivity': 0.5,
                'cold_start_impact': 0.3,
                'error_rate': 0.015
            }
        }
    
    def generate_execution_result(self, memory_size: int, execution_id: int,
                                workload_type: str = 'balanced',
                                cold_start: Optional[bool] = None) -> Dict[str, Any]:
        """Generate a single execution result."""
        profile = self.workload_profiles[workload_type]
        
        # Determine cold start
        if cold_start is None:
            cold_start = random.random() < 0.3  # 30% cold start rate
        
        # Calculate base duration based on memory and workload
        base_duration = profile['base_duration']
        memory_factor = (256 / memory_size) ** profile['memory_sensitivity']
        duration = base_duration * memory_factor
        
        # Add cold start penalty
        if cold_start:
            cold_start_penalty = random.uniform(500, 2000)
            duration += cold_start_penalty
        
        # Add natural variance
        variance = duration * 0.15
        duration += random.uniform(-variance, variance)
        duration = max(1.0, duration)
        
        # Calculate billed duration (rounded up to nearest ms)
        billed_duration = max(1, int(duration + 0.999))
        
        # Determine if this execution had an error
        has_error = random.random() < profile['error_rate']
        status_code = 500 if has_error else 200
        
        if has_error:
            duration = random.uniform(50, 200)  # Short duration for errors
            billed_duration = max(1, int(duration + 0.999))
        
        timestamp = datetime.utcnow() - timedelta(minutes=random.randint(0, 60))
        
        return {
            'memory_mb': memory_size,
            'execution_id': execution_id,
            'duration': round(duration, 2),
            'billed_duration': billed_duration,
            'cold_start': cold_start,
            'status_code': status_code,
            'timestamp': timestamp.isoformat(),
            'max_memory_used': min(memory_size, random.randint(memory_size // 4, memory_size - 10)),
            'request_id': f'req-{random.randint(100000, 999999)}-{execution_id}'
        }
    
    def generate_memory_test_result(self, memory_size: int, iterations: int = 10,
                                  workload_type: str = 'balanced') -> MemoryTestResult:
        """Generate test results for a specific memory configuration."""
        executions = []
        durations = []
        costs = []
        cold_starts = 0
        errors = 0
        
        for i in range(iterations):
            execution = self.generate_execution_result(
                memory_size, i, workload_type
            )
            executions.append(execution)
            
            if execution['status_code'] == 200:
                durations.append(execution['duration'])
                # Calculate cost (simplified AWS Lambda pricing)
                gb_seconds = (memory_size / 1024) * (execution['billed_duration'] / 1000)
                cost = gb_seconds * 0.0000166667 + 0.0000002  # GB-second + request cost
                costs.append(cost)
            else:
                errors += 1
            
            if execution['cold_start']:
                cold_starts += 1
        
        if durations:
            durations.sort()
            avg_duration = sum(durations) / len(durations)
            p95_duration = durations[int(len(durations) * 0.95)] if durations else 0
            p99_duration = durations[int(len(durations) * 0.99)] if durations else 0
            avg_cost = sum(costs) / len(costs) if costs else 0
            total_cost = sum(costs)
        else:
            avg_duration = p95_duration = p99_duration = avg_cost = total_cost = 0
        
        return MemoryTestResult(
            memory_size=memory_size,
            iterations=iterations,
            avg_duration=round(avg_duration, 2),
            p95_duration=round(p95_duration, 2),
            p99_duration=round(p99_duration, 2),
            avg_cost=round(avg_cost, 8),
            total_cost=round(total_cost, 8),
            cold_starts=cold_starts,
            errors=errors,
            raw_results=executions
        )
    
    def generate_tuning_results(self, memory_sizes: Optional[List[int]] = None,
                              iterations: int = 10,
                              workload_type: str = 'balanced',
                              function_arn: Optional[str] = None) -> Dict[str, Any]:
        """Generate complete tuning session results."""
        if memory_sizes is None:
            memory_sizes = [256, 512, 1024, 2048]
        
        if function_arn is None:
            function_arn = random.choice(self.function_arns)
        
        # Generate memory test results
        configurations = []
        memory_results = {}
        
        for memory_size in memory_sizes:
            result = self.generate_memory_test_result(memory_size, iterations, workload_type)
            memory_results[memory_size] = result
            
            configurations.append({
                'memory_mb': memory_size,
                'executions': result.raw_results,
                'total_executions': result.iterations,
                'successful_executions': result.iterations - result.errors,
                'failed_executions': result.errors,
                'avg_duration': result.avg_duration,
                'p95_duration': result.p95_duration,
                'avg_cost': result.avg_cost,
                'total_cost': result.total_cost,
                'cold_starts': result.cold_starts
            })
        
        test_start = datetime.utcnow() - timedelta(minutes=30)
        test_end = datetime.utcnow()
        
        return {
            'function_arn': function_arn,
            'test_started': test_start.isoformat(),
            'test_completed': test_end.isoformat(),
            'test_duration_seconds': int((test_end - test_start).total_seconds()),
            'strategy': 'balanced',
            'iterations_per_memory': iterations,
            'configurations': configurations,
            'metadata': {
                'workload_type': workload_type,
                'region': function_arn.split(':')[3],
                'account_id': function_arn.split(':')[4]
            }
        }
    
    def generate_performance_analysis(self, memory_results: Dict[int, MemoryTestResult]) -> PerformanceAnalysis:
        """Generate performance analysis from memory test results."""
        # Calculate efficiency scores (performance per cost ratio)
        efficiency_scores = {}
        for memory_size, result in memory_results.items():
            if result.avg_cost > 0:
                # Higher is better (faster execution per cost unit)
                efficiency = 1000 / (result.avg_duration * result.avg_cost * 1000000)
                efficiency_scores[memory_size] = round(efficiency, 2)
            else:
                efficiency_scores[memory_size] = 0
        
        # Find optimal configurations
        cost_optimal = self._find_cost_optimal(memory_results)
        speed_optimal = self._find_speed_optimal(memory_results)
        balanced_optimal = self._find_balanced_optimal(memory_results, efficiency_scores)
        
        # Generate trends
        trends = self._analyze_trends(memory_results)
        
        # Generate insights
        insights = self._generate_insights(memory_results, efficiency_scores, trends)
        
        return PerformanceAnalysis(
            memory_results=memory_results,
            efficiency_scores=efficiency_scores,
            cost_optimal=cost_optimal,
            speed_optimal=speed_optimal,
            balanced_optimal=balanced_optimal,
            trends=trends,
            insights=insights
        )
    
    def _find_cost_optimal(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Find the most cost-effective configuration."""
        min_cost = float('inf')
        optimal = None
        
        for memory_size, result in memory_results.items():
            if result.errors == 0 and result.avg_cost < min_cost:
                min_cost = result.avg_cost
                optimal = {
                    'memory_size': memory_size,
                    'avg_cost': result.avg_cost,
                    'avg_duration': result.avg_duration,
                    'reasoning': f'Lowest average cost at {result.avg_cost:.8f} per execution'
                }
        
        return optimal or {}
    
    def _find_speed_optimal(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Find the fastest configuration."""
        min_duration = float('inf')
        optimal = None
        
        for memory_size, result in memory_results.items():
            if result.errors == 0 and result.avg_duration < min_duration:
                min_duration = result.avg_duration
                optimal = {
                    'memory_size': memory_size,
                    'avg_duration': result.avg_duration,
                    'avg_cost': result.avg_cost,
                    'reasoning': f'Fastest execution at {result.avg_duration:.2f}ms average'
                }
        
        return optimal or {}
    
    def _find_balanced_optimal(self, memory_results: Dict[int, MemoryTestResult],
                             efficiency_scores: Dict[int, float]) -> Dict[str, Any]:
        """Find the most balanced configuration."""
        max_efficiency = 0
        optimal = None
        
        for memory_size, score in efficiency_scores.items():
            if score > max_efficiency:
                max_efficiency = score
                result = memory_results[memory_size]
                optimal = {
                    'memory_size': memory_size,
                    'efficiency_score': score,
                    'avg_duration': result.avg_duration,
                    'avg_cost': result.avg_cost,
                    'reasoning': f'Best performance/cost ratio with efficiency score {score:.2f}'
                }
        
        return optimal or {}
    
    def _analyze_trends(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Analyze performance and cost trends."""
        memory_sizes = sorted(memory_results.keys())
        durations = [memory_results[size].avg_duration for size in memory_sizes]
        costs = [memory_results[size].avg_cost for size in memory_sizes]
        
        # Calculate correlation between memory and performance
        duration_trend = 'decreasing' if durations[0] > durations[-1] else 'increasing'
        cost_trend = 'increasing' if costs[0] < costs[-1] else 'decreasing'
        
        # Find performance plateau point
        plateau_point = None
        for i in range(1, len(durations)):
            improvement = (durations[i-1] - durations[i]) / durations[i-1]
            if improvement < 0.05:  # Less than 5% improvement
                plateau_point = memory_sizes[i]
                break
        
        return {
            'duration_trend': duration_trend,
            'cost_trend': cost_trend,
            'performance_plateau': plateau_point,
            'memory_sensitivity': self._calculate_memory_sensitivity(memory_sizes, durations),
            'cost_efficiency_range': {
                'min_memory': memory_sizes[0],
                'max_memory': memory_sizes[-1],
                'cost_increase_factor': costs[-1] / costs[0] if costs[0] > 0 else 1
            }
        }
    
    def _calculate_memory_sensitivity(self, memory_sizes: List[int], durations: List[float]) -> str:
        """Calculate how sensitive the function is to memory changes."""
        if len(memory_sizes) < 2:
            return 'unknown'
        
        # Calculate percentage improvement from lowest to highest memory
        improvement = (durations[0] - durations[-1]) / durations[0]
        
        if improvement > 0.5:
            return 'high'
        elif improvement > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_insights(self, memory_results: Dict[int, MemoryTestResult],
                          efficiency_scores: Dict[int, float],
                          trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from the analysis."""
        insights = []
        
        memory_sizes = sorted(memory_results.keys())
        
        # Cold start analysis
        total_cold_starts = sum(result.cold_starts for result in memory_results.values())
        total_executions = sum(result.iterations for result in memory_results.values())
        cold_start_rate = total_cold_starts / total_executions if total_executions > 0 else 0
        
        if cold_start_rate > 0.3:
            insights.append({
                'type': 'cold_start',
                'severity': 'medium',
                'message': f'High cold start rate ({cold_start_rate:.1%}). Consider using provisioned concurrency.',
                'recommendation': 'Enable provisioned concurrency for consistent performance'
            })
        
        # Memory optimization
        if trends['memory_sensitivity'] == 'high':
            insights.append({
                'type': 'memory_optimization',
                'severity': 'high',
                'message': 'Function is highly memory-sensitive. Memory increases provide significant performance gains.',
                'recommendation': f'Consider increasing memory to {memory_sizes[-1]}MB for optimal performance'
            })
        elif trends['memory_sensitivity'] == 'low':
            insights.append({
                'type': 'memory_optimization',
                'severity': 'low',
                'message': 'Function shows low memory sensitivity. Higher memory may not justify the cost.',
                'recommendation': f'Consider using {memory_sizes[0]}MB for cost optimization'
            })
        
        # Error rate analysis
        total_errors = sum(result.errors for result in memory_results.values())
        if total_errors > 0:
            error_rate = total_errors / total_executions
            insights.append({
                'type': 'reliability',
                'severity': 'high' if error_rate > 0.05 else 'medium',
                'message': f'Function has {error_rate:.1%} error rate. Investigate error causes.',
                'recommendation': 'Review function logs and add error handling'
            })
        
        # Cost optimization
        min_cost_memory = min(memory_results.keys(), key=lambda x: memory_results[x].avg_cost)
        max_efficiency_memory = max(efficiency_scores.keys(), key=lambda x: efficiency_scores[x])
        
        if min_cost_memory != max_efficiency_memory:
            insights.append({
                'type': 'cost_vs_performance',
                'severity': 'medium',
                'message': f'Most cost-effective ({min_cost_memory}MB) differs from most efficient ({max_efficiency_memory}MB).',
                'recommendation': f'Choose {max_efficiency_memory}MB for best balance of cost and performance'
            })
        
        return insights


def generate_execution_result(memory_size: int, execution_id: int = 0,
                            workload_type: str = 'balanced',
                            cold_start: Optional[bool] = None) -> Dict[str, Any]:
    """Convenience function to generate a single execution result."""
    generator = TestDataGenerator()
    return generator.generate_execution_result(memory_size, execution_id, workload_type, cold_start)


def generate_memory_test_result(memory_size: int, iterations: int = 10,
                              workload_type: str = 'balanced') -> MemoryTestResult:
    """Convenience function to generate memory test results."""
    generator = TestDataGenerator()
    return generator.generate_memory_test_result(memory_size, iterations, workload_type)


def generate_tuning_results(memory_sizes: Optional[List[int]] = None,
                          iterations: int = 10,
                          workload_type: str = 'balanced',
                          function_arn: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to generate complete tuning results."""
    generator = TestDataGenerator()
    return generator.generate_tuning_results(memory_sizes, iterations, workload_type, function_arn)


def generate_performance_analysis(memory_results: Dict[int, MemoryTestResult]) -> PerformanceAnalysis:
    """Convenience function to generate performance analysis."""
    generator = TestDataGenerator()
    return generator.generate_performance_analysis(memory_results)