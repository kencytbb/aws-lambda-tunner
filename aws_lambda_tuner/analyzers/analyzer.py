"""Performance analyzer for Lambda tuning results."""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from ..models import (
    MemoryTestResult,
    Recommendation,
    PerformanceAnalysis,
    ColdStartAnalysis,
    ConcurrencyAnalysis,
    ConcurrencyPattern,
    WorkloadAnalysis,
    TimeBasedTrend,
    AdvancedPerformanceAnalysis,
)

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes Lambda performance tuning results."""

    def __init__(self, config):
        self.config = config

    def analyze(
        self,
        memory_results: Dict[int, MemoryTestResult],
        baseline_results: Optional[List[Dict[str, Any]]] = None,
    ) -> PerformanceAnalysis:
        """Analyze performance results across memory configurations."""
        logger.info("Analyzing performance results...")

        # Calculate efficiency scores
        efficiency_scores = self._calculate_efficiency_scores(memory_results)

        # Find optimal configurations
        cost_optimal = self._find_cost_optimal(memory_results)
        speed_optimal = self._find_speed_optimal(memory_results)
        balanced_optimal = self._find_balanced_optimal(memory_results, efficiency_scores)

        # Calculate performance trends
        trends = self._calculate_trends(memory_results)

        # Generate insights
        insights = self._generate_insights(memory_results, efficiency_scores, trends)

        return PerformanceAnalysis(
            memory_results=memory_results,
            efficiency_scores=efficiency_scores,
            cost_optimal=cost_optimal,
            speed_optimal=speed_optimal,
            balanced_optimal=balanced_optimal,
            trends=trends,
            insights=insights,
            baseline_comparison=self._compare_to_baseline(memory_results, baseline_results),
        )

    def get_recommendation(self, analysis: PerformanceAnalysis, strategy: str) -> Recommendation:
        """Generate optimization recommendation based on strategy."""
        logger.info(f"Generating recommendation for strategy: {strategy}")

        if strategy == "cost":
            optimal_memory = analysis.cost_optimal["memory_size"]
        elif strategy == "speed":
            optimal_memory = analysis.speed_optimal["memory_size"]
        else:  # balanced
            optimal_memory = analysis.balanced_optimal["memory_size"]

        current_memory = self._get_current_memory_size(analysis)

        # Calculate potential improvements
        current_result = analysis.memory_results.get(current_memory)
        optimal_result = analysis.memory_results.get(optimal_memory)

        if not current_result or not optimal_result:
            return Recommendation(
                strategy=strategy,
                current_memory_size=current_memory,
                optimal_memory_size=optimal_memory,
                should_optimize=False,
                reasoning="Insufficient data for recommendation",
            )

        # Calculate improvements
        cost_change = (
            (optimal_result.avg_cost - current_result.avg_cost) / current_result.avg_cost
        ) * 100
        duration_change = (
            (current_result.avg_duration - optimal_result.avg_duration)
            / current_result.avg_duration
        ) * 100

        # Determine if optimization is worthwhile
        should_optimize = self._should_optimize(
            current_memory, optimal_memory, cost_change, duration_change, strategy
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            current_memory, optimal_memory, cost_change, duration_change, strategy
        )

        return Recommendation(
            strategy=strategy,
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=should_optimize,
            cost_change_percent=cost_change,
            duration_change_percent=duration_change,
            reasoning=reasoning,
            confidence_score=self._calculate_confidence_score(analysis, optimal_memory),
            estimated_monthly_savings=self._estimate_monthly_savings(
                current_result, optimal_result
            ),
        )

    def _calculate_efficiency_scores(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[int, float]:
        """Calculate efficiency scores for each memory configuration."""
        scores = {}

        # Normalize metrics for scoring
        durations = [result.avg_duration for result in memory_results.values()]
        costs = [result.avg_cost for result in memory_results.values()]

        min_duration = min(durations)
        max_duration = max(durations)
        min_cost = min(costs)
        max_cost = max(costs)

        for memory_size, result in memory_results.items():
            # Normalize duration (0-1, lower is better)
            duration_score = (
                (result.avg_duration - min_duration) / (max_duration - min_duration)
                if max_duration != min_duration
                else 0
            )

            # Normalize cost (0-1, lower is better)
            cost_score = (
                (result.avg_cost - min_cost) / (max_cost - min_cost) if max_cost != min_cost else 0
            )

            # Combined efficiency score (lower is better)
            efficiency_score = (duration_score * 0.6) + (cost_score * 0.4)
            scores[memory_size] = efficiency_score

        return scores

    def _find_cost_optimal(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Find the most cost-effective configuration."""
        min_cost = float("inf")
        optimal = None

        for memory_size, result in memory_results.items():
            if result.avg_cost < min_cost:
                min_cost = result.avg_cost
                optimal = {
                    "memory_size": memory_size,
                    "avg_cost": result.avg_cost,
                    "avg_duration": result.avg_duration,
                }

        return optimal

    def _find_speed_optimal(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Find the fastest configuration."""
        min_duration = float("inf")
        optimal = None

        for memory_size, result in memory_results.items():
            if result.avg_duration < min_duration:
                min_duration = result.avg_duration
                optimal = {
                    "memory_size": memory_size,
                    "avg_cost": result.avg_cost,
                    "avg_duration": result.avg_duration,
                }

        return optimal

    def _find_balanced_optimal(
        self, memory_results: Dict[int, MemoryTestResult], efficiency_scores: Dict[int, float]
    ) -> Dict[str, Any]:
        """Find the most balanced configuration."""
        min_score = float("inf")
        optimal = None

        for memory_size, score in efficiency_scores.items():
            if score < min_score:
                min_score = score
                result = memory_results[memory_size]
                optimal = {
                    "memory_size": memory_size,
                    "avg_cost": result.avg_cost,
                    "avg_duration": result.avg_duration,
                    "efficiency_score": score,
                }

        return optimal

    def _calculate_trends(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Calculate performance trends across memory sizes."""
        sorted_results = sorted(memory_results.items())

        memory_sizes = [item[0] for item in sorted_results]
        durations = [item[1].avg_duration for item in sorted_results]
        costs = [item[1].avg_cost for item in sorted_results]

        return {
            "duration_trend": self._calculate_trend_direction(memory_sizes, durations),
            "cost_trend": self._calculate_trend_direction(memory_sizes, costs),
            "diminishing_returns_point": self._find_diminishing_returns_point(sorted_results),
            "cost_efficiency_point": self._find_cost_efficiency_point(sorted_results),
        }

    def _calculate_trend_direction(self, x_values: List[int], y_values: List[float]) -> str:
        """Calculate if trend is increasing, decreasing, or mixed."""
        if len(y_values) < 2:
            return "insufficient_data"

        increases = 0
        decreases = 0

        for i in range(1, len(y_values)):
            if y_values[i] > y_values[i - 1]:
                increases += 1
            elif y_values[i] < y_values[i - 1]:
                decreases += 1

        if decreases > increases * 2:
            return "decreasing"
        elif increases > decreases * 2:
            return "increasing"
        else:
            return "mixed"

    def _generate_insights(
        self,
        memory_results: Dict[int, MemoryTestResult],
        efficiency_scores: Dict[int, float],
        trends: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from the analysis."""
        insights = []

        # Performance insights
        if trends["duration_trend"] == "decreasing":
            insights.append(
                {
                    "type": "performance",
                    "title": "Strong Performance Scaling",
                    "description": "Function shows consistent performance improvements with increased memory",
                    "impact": "high",
                }
            )

        # Cost insights
        if trends["cost_trend"] == "increasing":
            insights.append(
                {
                    "type": "cost",
                    "title": "Linear Cost Increase",
                    "description": "Costs increase proportionally with memory allocation",
                    "impact": "medium",
                }
            )

        return insights

    def _should_optimize(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        strategy: str,
    ) -> bool:
        """Determine if optimization is worthwhile."""
        # Don't optimize if already optimal
        if current_memory == optimal_memory:
            return False

        # Strategy-specific thresholds
        if strategy == "cost":
            return cost_change < -5.0  # At least 5% cost reduction
        elif strategy == "speed":
            return duration_change > 10.0  # At least 10% performance improvement
        else:  # balanced
            cost_acceptable = cost_change < 20.0  # Cost increase less than 20%
            speed_acceptable = duration_change > -10.0  # Performance degradation less than 10%
            significant_improvement = duration_change > 15.0 or cost_change < -10.0

            return cost_acceptable and speed_acceptable and significant_improvement

    def _generate_reasoning(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        strategy: str,
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        if current_memory == optimal_memory:
            return f"Current memory configuration ({current_memory}MB) is already optimal for {strategy} strategy"

        direction = "increase" if optimal_memory > current_memory else "decrease"

        reasoning = f"Recommend {direction} memory from {current_memory}MB to {optimal_memory}MB. "

        if duration_change > 0:
            reasoning += f"This improves performance by {duration_change:.1f}% "
        else:
            reasoning += f"This reduces performance by {abs(duration_change):.1f}% "

        if cost_change > 0:
            reasoning += f"with a cost increase of {cost_change:.1f}%."
        else:
            reasoning += f"while reducing costs by {abs(cost_change):.1f}%."

        return reasoning

    def _get_current_memory_size(self, analysis: PerformanceAnalysis) -> int:
        """Determine the current memory size from analysis."""
        # Use the first memory size tested as current
        return min(analysis.memory_results.keys())

    def _calculate_confidence_score(
        self, analysis: PerformanceAnalysis, optimal_memory: int
    ) -> float:
        """Calculate confidence score for the recommendation."""
        optimal_result = analysis.memory_results[optimal_memory]

        # Factor 1: Sample size (more samples = higher confidence)
        sample_factor = min(optimal_result.iterations / 20.0, 1.0)

        # Factor 2: Low error rate (fewer errors = higher confidence)
        error_factor = 1.0 - (
            optimal_result.errors / optimal_result.iterations
            if optimal_result.iterations > 0
            else 0
        )

        confidence = (sample_factor * 0.5) + (error_factor * 0.5)
        return min(max(confidence, 0.0), 1.0)

    def _estimate_monthly_savings(
        self, current_result: MemoryTestResult, optimal_result: MemoryTestResult
    ) -> Dict[str, Dict[str, float]]:
        """Estimate monthly savings for different invocation volumes."""
        cost_diff = optimal_result.avg_cost - current_result.avg_cost
        duration_diff = current_result.avg_duration - optimal_result.avg_duration

        # Calculate for different monthly volumes
        volumes = [10000, 100000, 1000000, 10000000]
        savings = {}

        for volume in volumes:
            monthly_cost_change = cost_diff * volume
            monthly_time_saved = (duration_diff / 1000) * volume  # Convert to seconds

            savings[f"{volume:,}_invocations"] = {
                "cost_change": monthly_cost_change,
                "time_saved_seconds": monthly_time_saved,
            }

        return savings

    def _compare_to_baseline(self, memory_results, baseline_results):
        """Compare results to baseline performance."""
        if not baseline_results:
            return None

        baseline_durations = [r["duration"] for r in baseline_results]
        baseline_costs = [r["cost"] for r in baseline_results]

        return {
            "baseline_avg_duration": statistics.mean(baseline_durations),
            "baseline_avg_cost": statistics.mean(baseline_costs),
            "sample_count": len(baseline_results),
        }

    def _find_diminishing_returns_point(self, sorted_results):
        """Find the point where performance gains start diminishing."""
        return None  # Simplified for this implementation

    def _find_cost_efficiency_point(self, sorted_results):
        """Find the most cost-efficient memory configuration."""
        return None  # Simplified for this implementation

    def analyze_cold_starts(self, memory_results: Dict[int, MemoryTestResult]) -> ColdStartAnalysis:
        """Analyze cold start patterns and impact on performance."""
        logger.info("Analyzing cold start patterns...")

        total_invocations = sum(result.iterations for result in memory_results.values())
        total_cold_starts = sum(result.cold_starts for result in memory_results.values())

        # Calculate overall cold start ratio
        cold_start_ratio = total_cold_starts / total_invocations if total_invocations > 0 else 0

        # Analyze durations for cold vs warm starts
        cold_start_durations = []
        warm_start_durations = []
        memory_cold_start_ratios = {}

        for memory_size, result in memory_results.items():
            memory_cold_start_ratios[memory_size] = (
                result.cold_starts / result.iterations if result.iterations > 0 else 0
            )

            # Extract cold start and warm start durations from raw results
            for raw_result in result.raw_results:
                if raw_result.get("is_cold_start", False):
                    cold_start_durations.append(raw_result["duration"])
                else:
                    warm_start_durations.append(raw_result["duration"])

        avg_cold_start_duration = (
            statistics.mean(cold_start_durations) if cold_start_durations else 0
        )
        avg_warm_start_duration = (
            statistics.mean(warm_start_durations) if warm_start_durations else 0
        )

        # Calculate cold start impact score
        if avg_warm_start_duration > 0:
            cold_start_impact_score = (
                avg_cold_start_duration - avg_warm_start_duration
            ) / avg_warm_start_duration
        else:
            cold_start_impact_score = 0

        # Find correlation between memory and cold start frequency
        memory_sizes = list(memory_cold_start_ratios.keys())
        cold_start_ratios = list(memory_cold_start_ratios.values())
        memory_vs_cold_start_correlation = self._calculate_correlation(
            memory_sizes, cold_start_ratios
        )

        # Find optimal memory for reducing cold starts
        optimal_memory_for_cold_starts = min(
            memory_cold_start_ratios, key=memory_cold_start_ratios.get
        )

        # Identify patterns
        cold_start_patterns = {
            "memory_impact": {
                "high_memory_reduces_cold_starts": memory_vs_cold_start_correlation < -0.3,
                "memory_size_threshold": self._find_cold_start_threshold(memory_cold_start_ratios),
            },
            "frequency_analysis": {
                "frequent_cold_starts": cold_start_ratio > 0.3,
                "memory_distribution": memory_cold_start_ratios,
            },
        }

        return ColdStartAnalysis(
            cold_start_ratio=cold_start_ratio,
            avg_cold_start_duration=avg_cold_start_duration,
            avg_warm_start_duration=avg_warm_start_duration,
            cold_start_impact_score=cold_start_impact_score,
            memory_vs_cold_start_correlation=memory_vs_cold_start_correlation,
            optimal_memory_for_cold_starts=optimal_memory_for_cold_starts,
            cold_start_patterns=cold_start_patterns,
        )

    def analyze_concurrency_patterns(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> ConcurrencyAnalysis:
        """Analyze concurrency utilization and scaling patterns."""
        logger.info("Analyzing concurrency patterns...")

        # Extract concurrency data from raw results
        concurrent_executions = []
        throttling_events = 0
        execution_timestamps = []

        for result in memory_results.values():
            for raw_result in result.raw_results:
                if "concurrent_executions" in raw_result:
                    concurrent_executions.append(raw_result["concurrent_executions"])
                if "timestamp" in raw_result:
                    execution_timestamps.append(raw_result["timestamp"])
                if raw_result.get("throttled", False):
                    throttling_events += 1

        # Calculate concurrency metrics
        avg_concurrent_executions = (
            statistics.mean(concurrent_executions) if concurrent_executions else 0
        )
        peak_concurrent_executions = max(concurrent_executions) if concurrent_executions else 0

        # Calculate utilization based on AWS Lambda concurrent execution limits
        default_concurrent_limit = 1000  # AWS default
        concurrency_utilization = (
            peak_concurrent_executions / default_concurrent_limit
            if default_concurrent_limit > 0
            else 0
        )

        # Analyze scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(concurrent_executions)

        # Recommend optimal concurrency limit
        recommended_concurrency_limit = self._recommend_concurrency_limit(
            concurrent_executions, throttling_events
        )

        # Identify specific concurrency patterns
        identified_patterns = self._identify_concurrency_patterns(
            concurrent_executions, execution_timestamps
        )

        # Identify patterns
        concurrency_patterns = {
            "burst_behavior": self._analyze_burst_patterns(concurrent_executions),
            "scaling_latency": self._analyze_scaling_latency(memory_results),
            "resource_contention": throttling_events > 0,
            "pattern_summary": self._summarize_concurrency_patterns(identified_patterns),
        }

        return ConcurrencyAnalysis(
            avg_concurrent_executions=avg_concurrent_executions,
            peak_concurrent_executions=peak_concurrent_executions,
            concurrency_utilization=concurrency_utilization,
            scaling_efficiency=scaling_efficiency,
            throttling_events=throttling_events,
            recommended_concurrency_limit=recommended_concurrency_limit,
            concurrency_patterns=concurrency_patterns,
            identified_patterns=identified_patterns,
        )

    def analyze_workload_specific_patterns(
        self, memory_results: Dict[int, MemoryTestResult], workload_type: str
    ) -> WorkloadAnalysis:
        """Perform workload-specific optimization analysis."""
        logger.info(f"Analyzing workload-specific patterns for type: {workload_type}")

        # Calculate resource utilization
        resource_utilization = self._calculate_resource_utilization(memory_results)

        # Identify optimization opportunities based on workload type
        optimization_opportunities = self._identify_optimization_opportunities(
            memory_results, workload_type
        )

        # Generate workload-specific recommendations
        workload_specific_recommendations = self._generate_workload_recommendations(
            memory_results, workload_type
        )

        # Create cost vs performance curve
        cost_vs_performance_curve = self._create_cost_performance_curve(memory_results)

        return WorkloadAnalysis(
            workload_type=workload_type,
            resource_utilization=resource_utilization,
            optimization_opportunities=optimization_opportunities,
            workload_specific_recommendations=workload_specific_recommendations,
            cost_vs_performance_curve=cost_vs_performance_curve,
        )

    def analyze_time_based_trends(
        self,
        memory_results: Dict[int, MemoryTestResult],
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TimeBasedTrend]:
        """Analyze time-based performance trends and patterns."""
        logger.info("Analyzing time-based performance trends...")

        trends = []

        # Current session trend analysis
        current_trend = self._analyze_current_session_trends(memory_results)
        trends.append(current_trend)

        # Historical trend analysis if data is available
        if historical_data:
            historical_trends = self._analyze_historical_trends(historical_data)
            trends.extend(historical_trends)

        return trends

    def perform_advanced_analysis(
        self,
        memory_results: Dict[int, MemoryTestResult],
        baseline_results: Optional[List[Dict[str, Any]]] = None,
        workload_type: str = "general",
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> AdvancedPerformanceAnalysis:
        """Perform comprehensive advanced analysis including all new features."""
        logger.info("Performing advanced performance analysis...")

        # Perform base analysis
        base_analysis = self.analyze(memory_results, baseline_results)

        # Perform advanced analyses
        cold_start_analysis = self.analyze_cold_starts(memory_results)
        concurrency_analysis = self.analyze_concurrency_patterns(memory_results)
        workload_analysis = self.analyze_workload_specific_patterns(memory_results, workload_type)
        time_based_trends = self.analyze_time_based_trends(memory_results, historical_data)

        return AdvancedPerformanceAnalysis(
            memory_results=base_analysis.memory_results,
            efficiency_scores=base_analysis.efficiency_scores,
            cost_optimal=base_analysis.cost_optimal,
            speed_optimal=base_analysis.speed_optimal,
            balanced_optimal=base_analysis.balanced_optimal,
            trends=base_analysis.trends,
            insights=base_analysis.insights,
            baseline_comparison=base_analysis.baseline_comparison,
            cold_start_analysis=cold_start_analysis,
            concurrency_analysis=concurrency_analysis,
            workload_analysis=workload_analysis,
            time_based_trends=time_based_trends,
        )

    # Helper methods for advanced analysis

    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation coefficient between two variables."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        try:
            return float(np.corrcoef(x_values, y_values)[0, 1])
        except Exception:
            return 0.0

    def _find_cold_start_threshold(
        self, memory_cold_start_ratios: Dict[int, float]
    ) -> Optional[int]:
        """Find the memory threshold where cold starts significantly decrease."""
        sorted_items = sorted(memory_cold_start_ratios.items())

        for i in range(1, len(sorted_items)):
            current_ratio = sorted_items[i][1]
            previous_ratio = sorted_items[i - 1][1]

            # If cold start ratio decreases by more than 20%
            if previous_ratio > 0 and (previous_ratio - current_ratio) / previous_ratio > 0.2:
                return sorted_items[i][0]

        return None

    def _calculate_scaling_efficiency(self, concurrent_executions: List[int]) -> float:
        """Calculate how efficiently the function scales with concurrency."""
        if len(concurrent_executions) < 2:
            return 1.0

        # Calculate the coefficient of variation (stability metric)
        mean_concurrency = statistics.mean(concurrent_executions)
        std_concurrency = statistics.stdev(concurrent_executions)

        if mean_concurrency == 0:
            return 0.0

        cv = std_concurrency / mean_concurrency
        # Convert to efficiency score (lower CV = higher efficiency)
        return max(0, 1 - cv)

    def _recommend_concurrency_limit(
        self, concurrent_executions: List[int], throttling_events: int
    ) -> Optional[int]:
        """Recommend optimal concurrency limit based on usage patterns."""
        if not concurrent_executions:
            return None

        peak_concurrency = max(concurrent_executions)
        avg_concurrency = statistics.mean(concurrent_executions)

        # If no throttling, allow for some headroom
        if throttling_events == 0:
            return int(peak_concurrency * 1.2)

        # If throttling occurred, recommend based on average usage
        return int(avg_concurrency * 1.5)

    def _analyze_burst_patterns(self, concurrent_executions: List[int]) -> Dict[str, Any]:
        """Analyze burst traffic patterns."""
        if not concurrent_executions:
            return {"burst_detected": False}

        mean_concurrency = statistics.mean(concurrent_executions)
        peak_concurrency = max(concurrent_executions)

        # Detect burst if peak is significantly higher than average
        burst_ratio = peak_concurrency / mean_concurrency if mean_concurrency > 0 else 0
        burst_detected = burst_ratio > 3.0

        return {
            "burst_detected": burst_detected,
            "burst_ratio": burst_ratio,
            "peak_concurrency": peak_concurrency,
            "average_concurrency": mean_concurrency,
        }

    def _analyze_scaling_latency(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze how quickly the function scales up."""
        # This would typically analyze timestamps of invocations
        # For now, return basic metrics
        return {
            "scaling_analysis": "requires_timestamp_data",
            "recommendation": "monitor_scaling_metrics",
        }

    def _calculate_resource_utilization(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        # Extract CPU and memory utilization from raw results if available
        cpu_utilizations = []
        memory_utilizations = []

        for result in memory_results.values():
            for raw_result in result.raw_results:
                if "cpu_utilization" in raw_result:
                    cpu_utilizations.append(raw_result["cpu_utilization"])
                if "memory_utilization" in raw_result:
                    memory_utilizations.append(raw_result["memory_utilization"])

        return {
            "avg_cpu_utilization": statistics.mean(cpu_utilizations) if cpu_utilizations else 0.0,
            "avg_memory_utilization": (
                statistics.mean(memory_utilizations) if memory_utilizations else 0.0
            ),
            "peak_cpu_utilization": max(cpu_utilizations) if cpu_utilizations else 0.0,
            "peak_memory_utilization": max(memory_utilizations) if memory_utilizations else 0.0,
        }

    def _identify_optimization_opportunities(
        self, memory_results: Dict[int, MemoryTestResult], workload_type: str
    ) -> List[Dict[str, Any]]:
        """Identify workload-specific optimization opportunities."""
        opportunities = []

        # Analyze based on workload type
        if workload_type == "cpu_intensive":
            opportunities.extend(self._cpu_intensive_optimizations(memory_results))
        elif workload_type == "memory_intensive":
            opportunities.extend(self._memory_intensive_optimizations(memory_results))
        elif workload_type == "io_intensive":
            opportunities.extend(self._io_intensive_optimizations(memory_results))

        # Common optimizations
        opportunities.extend(self._common_optimizations(memory_results))

        return opportunities

    def _cpu_intensive_optimizations(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> List[Dict[str, Any]]:
        """Identify optimizations for CPU-intensive workloads."""
        opportunities = []

        # Check if higher memory significantly improves performance
        sorted_results = sorted(memory_results.items())
        if len(sorted_results) >= 2:
            lowest_memory = sorted_results[0][1]
            highest_memory = sorted_results[-1][1]

            performance_improvement = (
                lowest_memory.avg_duration - highest_memory.avg_duration
            ) / lowest_memory.avg_duration

            if performance_improvement > 0.3:  # 30% improvement
                opportunities.append(
                    {
                        "type": "memory_scaling",
                        "description": "Significant performance gains with higher memory allocation",
                        "impact": "high",
                        "recommendation": "Consider allocating more memory to improve CPU performance",
                    }
                )

        return opportunities

    def _memory_intensive_optimizations(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> List[Dict[str, Any]]:
        """Identify optimizations for memory-intensive workloads."""
        return [
            {
                "type": "memory_efficiency",
                "description": "Monitor memory utilization patterns",
                "impact": "medium",
                "recommendation": "Optimize memory allocation based on actual usage",
            }
        ]

    def _io_intensive_optimizations(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> List[Dict[str, Any]]:
        """Identify optimizations for I/O-intensive workloads."""
        return [
            {
                "type": "connection_pooling",
                "description": "Consider connection pooling for I/O operations",
                "impact": "medium",
                "recommendation": "Implement connection reuse to reduce I/O overhead",
            }
        ]

    def _common_optimizations(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> List[Dict[str, Any]]:
        """Identify common optimization opportunities."""
        opportunities = []

        # Check for high error rates
        total_errors = sum(result.errors for result in memory_results.values())
        total_iterations = sum(result.iterations for result in memory_results.values())
        error_rate = total_errors / total_iterations if total_iterations > 0 else 0

        if error_rate > 0.05:  # 5% error rate
            opportunities.append(
                {
                    "type": "error_reduction",
                    "description": f"High error rate detected: {error_rate:.2%}",
                    "impact": "high",
                    "recommendation": "Investigate and fix causes of function errors",
                }
            )

        return opportunities

    def _generate_workload_recommendations(
        self, memory_results: Dict[int, MemoryTestResult], workload_type: str
    ) -> List[Dict[str, Any]]:
        """Generate workload-specific recommendations."""
        recommendations = []

        if workload_type == "on_demand":
            recommendations.append(
                {
                    "type": "provisioned_concurrency",
                    "priority": "high",
                    "description": "Consider provisioned concurrency to reduce cold starts",
                    "rationale": "On-demand workloads benefit from reduced latency",
                }
            )
        elif workload_type == "continuous":
            recommendations.append(
                {
                    "type": "memory_optimization",
                    "priority": "medium",
                    "description": "Optimize for cost efficiency in continuous workloads",
                    "rationale": "Long-running workloads should prioritize cost optimization",
                }
            )
        elif workload_type == "scheduled":
            recommendations.append(
                {
                    "type": "balanced_approach",
                    "priority": "medium",
                    "description": "Balance cost and performance for scheduled workloads",
                    "rationale": "Predictable workloads allow for balanced optimization",
                }
            )

        return recommendations

    def _create_cost_performance_curve(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Create cost vs performance curve data."""
        sorted_results = sorted(memory_results.items())

        memory_sizes = [item[0] for item in sorted_results]
        costs = [item[1].avg_cost for item in sorted_results]
        durations = [item[1].avg_duration for item in sorted_results]

        return {
            "memory_sizes": memory_sizes,
            "costs": costs,
            "durations": durations,
            "efficiency_frontier": self._calculate_efficiency_frontier(costs, durations),
        }

    def _calculate_efficiency_frontier(
        self, costs: List[float], durations: List[float]
    ) -> List[int]:
        """Calculate the efficiency frontier (Pareto optimal points)."""
        # Find points that are not dominated by any other point
        frontier_indices = []

        for i, (cost_i, duration_i) in enumerate(zip(costs, durations)):
            is_dominated = False
            for j, (cost_j, duration_j) in enumerate(zip(costs, durations)):
                if (
                    i != j
                    and cost_j <= cost_i
                    and duration_j <= duration_i
                    and (cost_j < cost_i or duration_j < duration_i)
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                frontier_indices.append(i)

        return frontier_indices

    def _analyze_current_session_trends(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> TimeBasedTrend:
        """Analyze trends within the current testing session."""
        # Extract timing data from raw results
        execution_times = []
        for result in memory_results.values():
            for raw_result in result.raw_results:
                if "timestamp" in raw_result:
                    execution_times.append(raw_result["timestamp"])

        # Basic trend analysis
        metric_trends = {
            "duration": [result.avg_duration for result in memory_results.values()],
            "cost": [result.avg_cost for result in memory_results.values()],
        }

        return TimeBasedTrend(
            time_period="current_session",
            metric_trends=metric_trends,
            seasonal_patterns={},
            performance_degradation=False,
            trend_confidence=0.8,
            forecast={},
        )

    def _analyze_historical_trends(
        self, historical_data: List[Dict[str, Any]]
    ) -> List[TimeBasedTrend]:
        """Analyze historical performance trends."""
        # This would analyze historical data for long-term trends
        # For now, return a placeholder
        return [
            TimeBasedTrend(
                time_period="historical",
                metric_trends={},
                seasonal_patterns={},
                performance_degradation=False,
                trend_confidence=0.5,
                forecast={},
            )
        ]

    def _identify_concurrency_patterns(
        self, concurrent_executions: List[int], execution_timestamps: List[float]
    ) -> List[ConcurrencyPattern]:
        """Identify specific concurrency patterns from execution data."""
        patterns = []

        if not concurrent_executions or len(concurrent_executions) < 5:
            return patterns

        # Calculate basic statistics
        mean_concurrency = statistics.mean(concurrent_executions)
        std_concurrency = (
            statistics.stdev(concurrent_executions) if len(concurrent_executions) > 1 else 0
        )
        max_concurrency = max(concurrent_executions)

        # Identify burst pattern
        burst_threshold = mean_concurrency + (2 * std_concurrency)
        burst_count = sum(1 for c in concurrent_executions if c > burst_threshold)
        if burst_count > 0:
            burst_frequency = burst_count / len(concurrent_executions)
            patterns.append(
                ConcurrencyPattern(
                    pattern_type="burst",
                    frequency=burst_frequency,
                    intensity=(
                        (max_concurrency - mean_concurrency) / mean_concurrency
                        if mean_concurrency > 0
                        else 0
                    ),
                    duration_ms=self._estimate_pattern_duration(
                        concurrent_executions, burst_threshold
                    ),
                    impact_on_performance=self._calculate_pattern_impact(
                        concurrent_executions, burst_threshold
                    ),
                    recommendations=[
                        "Consider provisioned concurrency for burst workloads",
                        "Monitor scaling behavior during bursts",
                        "Implement backpressure mechanisms if needed",
                    ],
                )
            )

        # Identify steady pattern
        cv = std_concurrency / mean_concurrency if mean_concurrency > 0 else 0
        if cv < 0.3:  # Low coefficient of variation indicates steady pattern
            patterns.append(
                ConcurrencyPattern(
                    pattern_type="steady",
                    frequency=1.0 - cv,  # Higher frequency for lower variation
                    intensity=mean_concurrency / max_concurrency if max_concurrency > 0 else 0,
                    duration_ms=len(concurrent_executions) * 1000,  # Approximate total duration
                    impact_on_performance=0.1,  # Steady patterns have low impact
                    recommendations=[
                        "Optimize for cost efficiency with steady workloads",
                        "Consider reserved concurrency for predictable load",
                        "Focus on memory optimization over concurrency tuning",
                    ],
                )
            )

        # Identify gradual ramp pattern
        if len(concurrent_executions) >= 10:
            # Check for gradual increase over time
            first_half = concurrent_executions[: len(concurrent_executions) // 2]
            second_half = concurrent_executions[len(concurrent_executions) // 2 :]
            first_half_avg = statistics.mean(first_half)
            second_half_avg = statistics.mean(second_half)

            if second_half_avg > first_half_avg * 1.5:  # 50% increase
                patterns.append(
                    ConcurrencyPattern(
                        pattern_type="gradual_ramp",
                        frequency=0.5,  # Moderate frequency
                        intensity=(
                            (second_half_avg - first_half_avg) / first_half_avg
                            if first_half_avg > 0
                            else 0
                        ),
                        duration_ms=len(concurrent_executions) * 500,  # Approximate ramp duration
                        impact_on_performance=0.3,  # Moderate impact
                        recommendations=[
                            "Monitor scaling latency during ramp-up periods",
                            "Consider gradual warm-up strategies",
                            "Implement predictive scaling if pattern is regular",
                        ],
                    )
                )

        # Identify spike pattern
        spike_threshold = mean_concurrency + (3 * std_concurrency)
        spike_count = sum(1 for c in concurrent_executions if c > spike_threshold)
        if (
            spike_count > 0 and spike_count < len(concurrent_executions) * 0.1
        ):  # Less than 10% of samples
            spike_frequency = spike_count / len(concurrent_executions)
            patterns.append(
                ConcurrencyPattern(
                    pattern_type="spike",
                    frequency=spike_frequency,
                    intensity=max_concurrency / mean_concurrency if mean_concurrency > 0 else 0,
                    duration_ms=spike_count * 100,  # Estimated spike duration
                    impact_on_performance=0.7,  # High impact
                    recommendations=[
                        "Implement circuit breakers for spike protection",
                        "Consider auto-scaling policies with aggressive scaling",
                        "Monitor error rates during spikes",
                        "Implement queue-based processing for spike handling",
                    ],
                )
            )

        return patterns

    def _estimate_pattern_duration(
        self, concurrent_executions: List[int], threshold: float
    ) -> float:
        """Estimate the duration of a pattern above threshold."""
        # Simple estimation - count consecutive periods above threshold
        consecutive_count = 0
        max_consecutive = 0

        for execution in concurrent_executions:
            if execution > threshold:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0

        # Estimate duration (assuming each sample represents ~100ms)
        return max_consecutive * 100

    def _calculate_pattern_impact(
        self, concurrent_executions: List[int], threshold: float
    ) -> float:
        """Calculate the performance impact of a pattern."""
        above_threshold_count = sum(1 for c in concurrent_executions if c > threshold)
        total_count = len(concurrent_executions)

        if total_count == 0:
            return 0.0

        # Impact score based on frequency and magnitude
        frequency_impact = above_threshold_count / total_count
        magnitude_impact = (
            max(concurrent_executions) / statistics.mean(concurrent_executions)
            if statistics.mean(concurrent_executions) > 0
            else 0
        )

        return min(1.0, (frequency_impact + magnitude_impact) / 2)

    def _summarize_concurrency_patterns(self, patterns: List[ConcurrencyPattern]) -> Dict[str, Any]:
        """Summarize identified concurrency patterns."""
        if not patterns:
            return {"dominant_pattern": "unknown", "pattern_count": 0, "high_impact_patterns": []}

        # Find dominant pattern (highest frequency * impact)
        dominant_pattern = max(patterns, key=lambda p: p.frequency * p.impact_on_performance)

        # Find high impact patterns
        high_impact_patterns = [p.pattern_type for p in patterns if p.impact_on_performance > 0.5]

        return {
            "dominant_pattern": dominant_pattern.pattern_type,
            "pattern_count": len(patterns),
            "high_impact_patterns": high_impact_patterns,
            "total_recommendations": sum(len(p.recommendations) for p in patterns),
        }
