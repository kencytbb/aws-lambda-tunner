"""Continuous workload analyzer for Lambda tuning - optimizes for sustained throughput."""

import logging
import statistics
from typing import Dict, List, Optional, Any

from ..models import MemoryTestResult, Recommendation, ConcurrencyAnalysis
from .analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class ContinuousAnalyzer(PerformanceAnalyzer):
    """Specialized analyzer for continuous workloads that prioritize sustained throughput and cost efficiency."""

    def __init__(self, config):
        """Initialize the continuous workload analyzer with configuration."""
        super().__init__(config)
        self.workload_type = "continuous"

        # Continuous workload specific thresholds
        self.cost_sensitivity = "high"  # High sensitivity to cost for long-running workloads
        self.throughput_priority = "high"  # High priority on sustained throughput
        self.latency_tolerance = "medium"  # Medium tolerance for latency in favor of cost

    def analyze_for_sustained_throughput(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze and recommend configuration for sustained throughput."""
        logger.info("Analyzing for sustained throughput in continuous workload...")

        # Perform concurrency analysis
        concurrency_analysis = self.analyze_concurrency_patterns(memory_results)

        # Find configuration that maximizes throughput efficiency
        throughput_optimal_config = self._find_throughput_optimal_configuration(
            memory_results, concurrency_analysis
        )

        # Analyze cost efficiency for long-running workloads
        cost_efficiency_analysis = self._analyze_cost_efficiency_for_continuous(memory_results)

        # Calculate sustained performance metrics
        sustained_performance = self._calculate_sustained_performance_metrics(memory_results)

        # Analyze resource utilization efficiency
        resource_efficiency = self._analyze_resource_utilization_efficiency(memory_results)

        return {
            "throughput_optimal_configuration": throughput_optimal_config,
            "cost_efficiency_analysis": cost_efficiency_analysis,
            "sustained_performance_metrics": sustained_performance,
            "resource_utilization_efficiency": resource_efficiency,
            "scaling_recommendations": self._generate_scaling_recommendations(concurrency_analysis),
        }

    def get_continuous_recommendation(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Recommendation:
        """Generate specialized recommendation for continuous workloads."""
        logger.info("Generating continuous workload recommendation...")

        # Perform comprehensive analysis
        base_analysis = self.analyze(memory_results)
        concurrency_analysis = self.analyze_concurrency_patterns(memory_results)
        continuous_analysis = self.analyze_for_sustained_throughput(memory_results)

        # Determine optimal memory size prioritizing cost efficiency and sustained performance
        optimal_memory = self._determine_continuous_optimal_memory(
            memory_results, continuous_analysis, concurrency_analysis
        )

        # Calculate current memory (baseline)
        current_memory = self._get_current_memory_size(base_analysis)

        # Get performance and cost metrics
        current_result = memory_results.get(current_memory)
        optimal_result = memory_results.get(optimal_memory)

        if not current_result or not optimal_result:
            return self._create_insufficient_data_recommendation(current_memory, optimal_memory)

        # Calculate improvements
        cost_change = (
            (optimal_result.avg_cost - current_result.avg_cost) / current_result.avg_cost
        ) * 100
        duration_change = (
            (current_result.avg_duration - optimal_result.avg_duration)
            / current_result.avg_duration
        ) * 100

        # Calculate throughput improvement (inverse of duration)
        current_throughput = 1000 / current_result.avg_duration  # Requests per second equivalent
        optimal_throughput = 1000 / optimal_result.avg_duration
        throughput_improvement = (
            (optimal_throughput - current_throughput) / current_throughput
        ) * 100

        # Determine if optimization is worthwhile for continuous workloads
        should_optimize = self._should_optimize_for_continuous(
            current_memory, optimal_memory, cost_change, duration_change, throughput_improvement
        )

        # Generate continuous workload specific reasoning
        reasoning = self._generate_continuous_reasoning(
            current_memory,
            optimal_memory,
            cost_change,
            duration_change,
            throughput_improvement,
            continuous_analysis,
        )

        # Calculate confidence score with continuous workload considerations
        confidence_score = self._calculate_continuous_confidence_score(
            base_analysis, optimal_memory, concurrency_analysis
        )

        # Estimate monthly savings with throughput benefits
        monthly_savings = self._estimate_continuous_monthly_savings(
            current_result, optimal_result, throughput_improvement
        )

        return Recommendation(
            strategy="continuous_optimized",
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=should_optimize,
            cost_change_percent=cost_change,
            duration_change_percent=duration_change,
            reasoning=reasoning,
            confidence_score=confidence_score,
            estimated_monthly_savings=monthly_savings,
        )

    def _find_throughput_optimal_configuration(
        self, memory_results: Dict[int, MemoryTestResult], concurrency_analysis: ConcurrencyAnalysis
    ) -> Dict[str, Any]:
        """Find the configuration that maximizes sustained throughput efficiency."""
        throughput_efficiency_scores = {}

        for memory_size, result in memory_results.items():
            # Calculate throughput (requests per second equivalent)
            throughput = 1000 / result.avg_duration if result.avg_duration > 0 else 0

            # Calculate cost per request
            cost_per_request = result.avg_cost

            # Calculate throughput efficiency (throughput per dollar)
            throughput_efficiency = throughput / cost_per_request if cost_per_request > 0 else 0

            # Factor in error rate (reduce efficiency for high error rates)
            error_rate = result.errors / result.iterations if result.iterations > 0 else 0
            error_penalty = 1 - error_rate

            # Final efficiency score
            final_efficiency = throughput_efficiency * error_penalty
            throughput_efficiency_scores[memory_size] = final_efficiency

        # Find the configuration with the highest efficiency score
        optimal_memory = max(throughput_efficiency_scores, key=throughput_efficiency_scores.get)
        optimal_result = memory_results[optimal_memory]

        return {
            "memory_size": optimal_memory,
            "throughput_efficiency_score": throughput_efficiency_scores[optimal_memory],
            "estimated_throughput": (
                1000 / optimal_result.avg_duration if optimal_result.avg_duration > 0 else 0
            ),
            "cost_per_request": optimal_result.avg_cost,
            "error_rate": (
                optimal_result.errors / optimal_result.iterations
                if optimal_result.iterations > 0
                else 0
            ),
            "reasoning": f"Selected {optimal_memory}MB for optimal throughput efficiency",
        }

    def _analyze_cost_efficiency_for_continuous(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze cost efficiency specifically for continuous/long-running workloads."""
        # Calculate cost per time unit and cost per successful request
        cost_efficiency_metrics = {}

        for memory_size, result in memory_results.items():
            successful_requests = result.iterations - result.errors

            cost_per_successful_request = (
                result.avg_cost if successful_requests > 0 else float("inf")
            )
            cost_per_minute = (
                (result.avg_cost * 60000) / result.avg_duration
                if result.avg_duration > 0
                else float("inf")
            )

            cost_efficiency_metrics[memory_size] = {
                "cost_per_successful_request": cost_per_successful_request,
                "cost_per_minute": cost_per_minute,
                "success_rate": (
                    successful_requests / result.iterations if result.iterations > 0 else 0
                ),
            }

        # Find the most cost-efficient configuration
        best_config = min(
            cost_efficiency_metrics.items(), key=lambda x: x[1]["cost_per_successful_request"]
        )

        return {
            "most_cost_efficient_memory": best_config[0],
            "cost_efficiency_metrics": cost_efficiency_metrics,
            "monthly_cost_projections": self._project_monthly_costs(memory_results),
            "cost_optimization_recommendations": self._generate_cost_optimization_recommendations(
                cost_efficiency_metrics
            ),
        }

    def _calculate_sustained_performance_metrics(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Calculate metrics related to sustained performance over time."""
        sustained_metrics = {}

        for memory_size, result in memory_results.items():
            # Calculate performance consistency
            if len(result.raw_results) > 1:
                durations = [r["duration"] for r in result.raw_results if "duration" in r]
                if durations:
                    performance_consistency = 1 - (
                        statistics.stdev(durations) / statistics.mean(durations)
                    )
                else:
                    performance_consistency = 0
            else:
                performance_consistency = 0

            # Calculate sustained throughput capacity
            sustained_throughput = (
                (1000 / result.avg_duration) * performance_consistency
                if result.avg_duration > 0
                else 0
            )

            sustained_metrics[memory_size] = {
                "performance_consistency": max(0, performance_consistency),
                "sustained_throughput_capacity": sustained_throughput,
                "performance_degradation_risk": self._assess_performance_degradation_risk(result),
            }

        return sustained_metrics

    def _analyze_resource_utilization_efficiency(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze how efficiently resources are being utilized."""
        utilization_analysis = {}

        for memory_size, result in memory_results.items():
            # Analyze memory utilization from raw results if available
            memory_utilizations = []
            cpu_utilizations = []

            for raw_result in result.raw_results:
                if "memory_utilization" in raw_result:
                    memory_utilizations.append(raw_result["memory_utilization"])
                if "cpu_utilization" in raw_result:
                    cpu_utilizations.append(raw_result["cpu_utilization"])

            avg_memory_util = statistics.mean(memory_utilizations) if memory_utilizations else 0
            avg_cpu_util = statistics.mean(cpu_utilizations) if cpu_utilizations else 0

            # Calculate resource efficiency score
            memory_efficiency = avg_memory_util  # Higher utilization is better for cost efficiency
            cpu_efficiency = avg_cpu_util

            # Calculate over-provisioning risk
            over_provisioning_risk = self._calculate_over_provisioning_risk(
                memory_size, avg_memory_util
            )

            utilization_analysis[memory_size] = {
                "memory_utilization": avg_memory_util,
                "cpu_utilization": avg_cpu_util,
                "resource_efficiency_score": (memory_efficiency + cpu_efficiency) / 2,
                "over_provisioning_risk": over_provisioning_risk,
            }

        return utilization_analysis

    def _generate_scaling_recommendations(
        self, concurrency_analysis: ConcurrencyAnalysis
    ) -> Dict[str, Any]:
        """Generate recommendations for scaling in continuous workloads."""
        recommendations = []

        # Analyze current scaling patterns
        if concurrency_analysis.concurrency_utilization < 0.3:
            recommendations.append(
                {
                    "type": "underutilization",
                    "priority": "medium",
                    "description": "Low concurrency utilization detected",
                    "recommendation": "Consider reducing reserved concurrency or optimizing invocation patterns",
                }
            )
        elif concurrency_analysis.concurrency_utilization > 0.8:
            recommendations.append(
                {
                    "type": "high_utilization",
                    "priority": "high",
                    "description": "High concurrency utilization may lead to throttling",
                    "recommendation": "Consider increasing concurrency limits or implementing backpressure",
                }
            )

        # Analyze scaling efficiency
        if concurrency_analysis.scaling_efficiency < 0.7:
            recommendations.append(
                {
                    "type": "scaling_inefficiency",
                    "priority": "medium",
                    "description": "Inefficient scaling patterns detected",
                    "recommendation": "Investigate scaling behavior and consider provisioned concurrency",
                }
            )

        return {
            "recommendations": recommendations,
            "optimal_concurrency_limit": concurrency_analysis.recommended_concurrency_limit,
            "current_utilization": concurrency_analysis.concurrency_utilization,
            "scaling_efficiency": concurrency_analysis.scaling_efficiency,
        }

    def _determine_continuous_optimal_memory(
        self,
        memory_results: Dict[int, MemoryTestResult],
        continuous_analysis: Dict[str, Any],
        concurrency_analysis: ConcurrencyAnalysis,
    ) -> int:
        """Determine optimal memory size for continuous workloads."""
        # Start with the throughput optimal configuration
        throughput_optimal = continuous_analysis["throughput_optimal_configuration"]["memory_size"]
        cost_optimal = continuous_analysis["cost_efficiency_analysis"]["most_cost_efficient_memory"]

        # For continuous workloads, balance throughput and cost
        # If they're the same, use that
        if throughput_optimal == cost_optimal:
            return throughput_optimal

        # Otherwise, evaluate trade-offs
        throughput_result = memory_results[throughput_optimal]
        cost_result = memory_results[cost_optimal]

        # Calculate the trade-off: if cost difference is significant but performance gain is minimal
        cost_diff = abs(throughput_result.avg_cost - cost_result.avg_cost) / cost_result.avg_cost
        perf_diff = (
            abs(throughput_result.avg_duration - cost_result.avg_duration)
            / cost_result.avg_duration
        )

        # If cost increase is more than 25% but performance gain is less than 10%, choose cost optimal
        if cost_diff > 0.25 and perf_diff < 0.1:
            return cost_optimal
        else:
            return throughput_optimal

    def _should_optimize_for_continuous(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        throughput_improvement: float,
    ) -> bool:
        """Determine if optimization is worthwhile for continuous workloads."""
        # Don't optimize if already optimal
        if current_memory == optimal_memory:
            return False

        # For continuous workloads, prioritize cost efficiency
        significant_cost_savings = cost_change < -10.0  # At least 10% cost reduction
        acceptable_performance_trade_off = (
            duration_change > -15.0
        )  # Performance degradation less than 15%
        significant_throughput_improvement = (
            throughput_improvement > 15.0
        )  # 15% throughput improvement

        # Optimize if either significant cost savings or throughput improvement with acceptable trade-offs
        return (
            significant_cost_savings and acceptable_performance_trade_off
        ) or significant_throughput_improvement

    def _generate_continuous_reasoning(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        throughput_improvement: float,
        continuous_analysis: Dict[str, Any],
    ) -> str:
        """Generate reasoning for continuous workload optimization."""
        if current_memory == optimal_memory:
            return f"Current memory configuration ({current_memory}MB) is already optimal for continuous workloads"

        direction = "increase" if optimal_memory > current_memory else "decrease"
        reasoning = f"Recommend {direction} memory from {current_memory}MB to {optimal_memory}MB for continuous workload optimization. "

        # Add throughput impact
        if throughput_improvement > 0:
            reasoning += f"This improves sustained throughput by {throughput_improvement:.1f}% "

        # Add performance impact
        if duration_change > 0:
            reasoning += f"and reduces latency by {duration_change:.1f}% "
        elif duration_change < 0:
            reasoning += f"with {abs(duration_change):.1f}% latency trade-off "

        # Add cost impact (critical for continuous workloads)
        if cost_change > 0:
            reasoning += f"at a cost increase of {cost_change:.1f}%. "
        else:
            reasoning += f"while reducing costs by {abs(cost_change):.1f}%. "

        # Add continuous workload specific context
        efficiency_score = continuous_analysis["throughput_optimal_configuration"][
            "throughput_efficiency_score"
        ]
        reasoning += f"This configuration provides the best throughput efficiency score ({efficiency_score:.2f}) for sustained operations."

        return reasoning

    def _calculate_continuous_confidence_score(
        self, analysis, optimal_memory: int, concurrency_analysis: ConcurrencyAnalysis
    ) -> float:
        """Calculate confidence score with continuous workload considerations."""
        base_confidence = self._calculate_confidence_score(analysis, optimal_memory)

        # Adjust confidence based on concurrency data quality and scaling efficiency
        scaling_confidence = concurrency_analysis.scaling_efficiency

        # Combine base confidence with scaling confidence
        return (base_confidence * 0.7) + (scaling_confidence * 0.3)

    def _estimate_continuous_monthly_savings(
        self,
        current_result: MemoryTestResult,
        optimal_result: MemoryTestResult,
        throughput_improvement: float,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate monthly savings including throughput benefits for continuous workloads."""
        base_savings = self._estimate_monthly_savings(current_result, optimal_result)

        # Add throughput benefits (important for continuous workloads)
        for volume_key in base_savings:
            base_savings[volume_key]["throughput_improvement_percent"] = throughput_improvement
            # Calculate potential capacity gains
            volume = int(volume_key.split("_")[0].replace(",", ""))
            potential_additional_capacity = volume * (throughput_improvement / 100)
            base_savings[volume_key]["additional_capacity_requests"] = potential_additional_capacity

        return base_savings

    def _project_monthly_costs(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Project monthly costs for different configurations."""
        monthly_projections = {}
        monthly_volumes = [1000000, 10000000, 100000000]  # Higher volumes for continuous workloads

        for memory_size, result in memory_results.items():
            projections = {}
            for volume in monthly_volumes:
                monthly_cost = result.avg_cost * volume
                projections[f"{volume:,}_requests"] = monthly_cost
            monthly_projections[f"{memory_size}MB"] = projections

        return monthly_projections

    def _generate_cost_optimization_recommendations(
        self, cost_efficiency_metrics: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Find configurations with poor cost efficiency
        for memory_size, metrics in cost_efficiency_metrics.items():
            if metrics["success_rate"] < 0.95:  # Less than 95% success rate
                recommendations.append(
                    {
                        "type": "reliability_issue",
                        "memory_size": memory_size,
                        "description": f"Low success rate ({metrics['success_rate']:.2%}) increases effective cost",
                        "recommendation": "Investigate and fix error causes to improve cost efficiency",
                    }
                )

        return recommendations

    def _assess_performance_degradation_risk(self, result: MemoryTestResult) -> str:
        """Assess the risk of performance degradation over time."""
        # This would analyze trends in the raw results over time
        # For now, use error rate as a proxy
        error_rate = result.errors / result.iterations if result.iterations > 0 else 0

        if error_rate > 0.1:
            return "high"
        elif error_rate > 0.05:
            return "medium"
        else:
            return "low"

    def _calculate_over_provisioning_risk(self, memory_size: int, memory_utilization: float) -> str:
        """Calculate the risk of over-provisioning resources."""
        if memory_utilization < 0.3:  # Less than 30% utilization
            return "high"
        elif memory_utilization < 0.6:  # Less than 60% utilization
            return "medium"
        else:
            return "low"

    def _create_insufficient_data_recommendation(
        self, current_memory: int, optimal_memory: int
    ) -> Recommendation:
        """Create recommendation when insufficient data is available."""
        return Recommendation(
            strategy="continuous_optimized",
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=False,
            reasoning="Insufficient data for continuous workload recommendation. Consider running longer duration tests with sustained load.",
        )
