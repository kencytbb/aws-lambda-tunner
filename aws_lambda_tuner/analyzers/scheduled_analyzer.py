"""Scheduled workload analyzer for Lambda tuning - optimizes for cost-performance balance."""

import logging
import statistics
from typing import Dict, List, Optional, Any

from ..models import MemoryTestResult, Recommendation, WorkloadAnalysis
from .analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class ScheduledAnalyzer(PerformanceAnalyzer):
    """Specialized analyzer for scheduled workloads that balance cost and performance."""

    def __init__(self, config):
        """Initialize the scheduled workload analyzer with configuration."""
        super().__init__(config)
        self.workload_type = "scheduled"

        # Scheduled workload specific thresholds
        self.cost_weight = 0.6  # Higher weight on cost for scheduled workloads
        self.performance_weight = 0.4  # Moderate weight on performance
        self.predictability_factor = 0.8  # High predictability for scheduled workloads
        self.optimization_frequency = "weekly"  # How often to re-optimize

    def analyze_for_cost_performance_balance(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze and recommend configuration for optimal cost-performance balance."""
        logger.info("Analyzing for cost-performance balance in scheduled workload...")

        # Calculate balanced efficiency scores
        balanced_efficiency = self._calculate_balanced_efficiency_scores(memory_results)

        # Find sweet spot configuration
        sweet_spot_config = self._find_cost_performance_sweet_spot(
            memory_results, balanced_efficiency
        )

        # Analyze predictable performance patterns
        predictability_analysis = self._analyze_predictable_patterns(memory_results)

        # Calculate optimization stability
        optimization_stability = self._calculate_optimization_stability(memory_results)

        # Generate resource allocation recommendations
        resource_recommendations = self._generate_resource_allocation_recommendations(
            memory_results, sweet_spot_config
        )

        return {
            "sweet_spot_configuration": sweet_spot_config,
            "balanced_efficiency_scores": balanced_efficiency,
            "predictability_analysis": predictability_analysis,
            "optimization_stability": optimization_stability,
            "resource_allocation_recommendations": resource_recommendations,
            "cost_performance_trade_offs": self._analyze_cost_performance_trade_offs(
                memory_results
            ),
        }

    def get_scheduled_recommendation(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Recommendation:
        """Generate specialized recommendation for scheduled workloads."""
        logger.info("Generating scheduled workload recommendation...")

        # Perform comprehensive analysis
        base_analysis = self.analyze(memory_results)
        scheduled_analysis = self.analyze_for_cost_performance_balance(memory_results)

        # Determine optimal memory size balancing cost and performance
        optimal_memory = self._determine_scheduled_optimal_memory(
            memory_results, scheduled_analysis
        )

        # Calculate current memory (baseline)
        current_memory = self._get_current_memory_size(base_analysis)

        # Get performance and cost metrics
        current_result = memory_results.get(current_memory)
        optimal_result = memory_results.get(optimal_memory)

        if not current_result or optimal_result:
            return self._create_insufficient_data_recommendation(current_memory, optimal_memory)

        # Calculate improvements
        cost_change = (
            (optimal_result.avg_cost - current_result.avg_cost) / current_result.avg_cost
        ) * 100
        duration_change = (
            (current_result.avg_duration - optimal_result.avg_duration)
            / current_result.avg_duration
        ) * 100

        # Calculate efficiency improvement (balanced score)
        current_efficiency = scheduled_analysis["balanced_efficiency_scores"][current_memory]
        optimal_efficiency = scheduled_analysis["balanced_efficiency_scores"][optimal_memory]
        efficiency_improvement = (
            ((optimal_efficiency - current_efficiency) / current_efficiency) * 100
            if current_efficiency > 0
            else 0
        )

        # Determine if optimization is worthwhile for scheduled workloads
        should_optimize = self._should_optimize_for_scheduled(
            current_memory, optimal_memory, cost_change, duration_change, efficiency_improvement
        )

        # Generate scheduled workload specific reasoning
        reasoning = self._generate_scheduled_reasoning(
            current_memory,
            optimal_memory,
            cost_change,
            duration_change,
            efficiency_improvement,
            scheduled_analysis,
        )

        # Calculate confidence score with scheduled workload considerations
        confidence_score = self._calculate_scheduled_confidence_score(
            base_analysis, optimal_memory, scheduled_analysis
        )

        # Estimate monthly savings with balance considerations
        monthly_savings = self._estimate_scheduled_monthly_savings(
            current_result, optimal_result, efficiency_improvement
        )

        return Recommendation(
            strategy="scheduled_balanced",
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=should_optimize,
            cost_change_percent=cost_change,
            duration_change_percent=duration_change,
            reasoning=reasoning,
            confidence_score=confidence_score,
            estimated_monthly_savings=monthly_savings,
        )

    def _calculate_balanced_efficiency_scores(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[int, float]:
        """Calculate efficiency scores with balanced cost-performance weighting."""
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
            if max_duration != min_duration:
                duration_score = (result.avg_duration - min_duration) / (
                    max_duration - min_duration
                )
            else:
                duration_score = 0

            # Normalize cost (0-1, lower is better)
            if max_cost != min_cost:
                cost_score = (result.avg_cost - min_cost) / (max_cost - min_cost)
            else:
                cost_score = 0

            # Factor in error rate (penalty for errors)
            error_rate = result.errors / result.iterations if result.iterations > 0 else 0
            error_penalty = error_rate * 0.5  # Errors add to the score (worse)

            # Balanced efficiency score using scheduled workload weights
            balanced_score = (
                (duration_score * self.performance_weight)
                + (cost_score * self.cost_weight)
                + error_penalty
            )

            # Convert to efficiency (lower score = higher efficiency)
            efficiency_score = 1.0 - balanced_score
            scores[memory_size] = max(0.0, efficiency_score)

        return scores

    def _find_cost_performance_sweet_spot(
        self, memory_results: Dict[int, MemoryTestResult], balanced_efficiency: Dict[int, float]
    ) -> Dict[str, Any]:
        """Find the configuration that provides the best cost-performance balance."""
        # Find the configuration with the highest balanced efficiency score
        optimal_memory = max(balanced_efficiency, key=balanced_efficiency.get)
        optimal_result = memory_results[optimal_memory]
        optimal_score = balanced_efficiency[optimal_memory]

        # Calculate how this compares to pure cost and pure performance optimizations
        cost_optimal_memory = min(memory_results.keys(), key=lambda k: memory_results[k].avg_cost)
        speed_optimal_memory = min(
            memory_results.keys(), key=lambda k: memory_results[k].avg_duration
        )

        cost_optimal_result = memory_results[cost_optimal_memory]
        speed_optimal_result = memory_results[speed_optimal_memory]

        # Calculate trade-off metrics
        cost_vs_optimal = (
            (optimal_result.avg_cost - cost_optimal_result.avg_cost) / cost_optimal_result.avg_cost
        ) * 100
        speed_vs_optimal = (
            (optimal_result.avg_duration - speed_optimal_result.avg_duration)
            / speed_optimal_result.avg_duration
        ) * 100

        return {
            "memory_size": optimal_memory,
            "balanced_efficiency_score": optimal_score,
            "avg_cost": optimal_result.avg_cost,
            "avg_duration": optimal_result.avg_duration,
            "cost_trade_off_vs_cheapest": cost_vs_optimal,
            "speed_trade_off_vs_fastest": speed_vs_optimal,
            "reasoning": f"Selected {optimal_memory}MB for optimal cost-performance balance",
            "is_cost_optimal": optimal_memory == cost_optimal_memory,
            "is_speed_optimal": optimal_memory == speed_optimal_memory,
        }

    def _analyze_predictable_patterns(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze predictable performance patterns in scheduled workloads."""
        predictability_metrics = {}

        for memory_size, result in memory_results.items():
            if len(result.raw_results) > 1:
                # Calculate performance consistency
                durations = [r["duration"] for r in result.raw_results if "duration" in r]
                costs = [r["cost"] for r in result.raw_results if "cost" in r]

                if durations:
                    duration_cv = (
                        statistics.stdev(durations) / statistics.mean(durations)
                        if statistics.mean(durations) > 0
                        else 0
                    )
                    duration_predictability = max(0, 1 - duration_cv)
                else:
                    duration_predictability = 0

                if costs:
                    cost_cv = (
                        statistics.stdev(costs) / statistics.mean(costs)
                        if statistics.mean(costs) > 0
                        else 0
                    )
                    cost_predictability = max(0, 1 - cost_cv)
                else:
                    cost_predictability = 0

                # Overall predictability score
                overall_predictability = (duration_predictability + cost_predictability) / 2

                predictability_metrics[memory_size] = {
                    "duration_predictability": duration_predictability,
                    "cost_predictability": cost_predictability,
                    "overall_predictability": overall_predictability,
                    "consistency_rating": self._rate_consistency(overall_predictability),
                }
            else:
                predictability_metrics[memory_size] = {
                    "duration_predictability": 0,
                    "cost_predictability": 0,
                    "overall_predictability": 0,
                    "consistency_rating": "insufficient_data",
                }

        # Find the most predictable configuration
        most_predictable = max(
            predictability_metrics.items(), key=lambda x: x[1]["overall_predictability"]
        )

        return {
            "predictability_by_memory": predictability_metrics,
            "most_predictable_config": {
                "memory_size": most_predictable[0],
                "predictability_score": most_predictable[1]["overall_predictability"],
            },
            "predictability_recommendations": self._generate_predictability_recommendations(
                predictability_metrics
            ),
        }

    def _rate_consistency(self, predictability_score: float) -> str:
        """Rate the consistency of performance based on predictability score."""
        if predictability_score >= 0.9:
            return "very_high"
        elif predictability_score >= 0.8:
            return "high"
        elif predictability_score >= 0.7:
            return "medium"
        elif predictability_score >= 0.5:
            return "low"
        else:
            return "very_low"

    def _calculate_optimization_stability(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Calculate how stable the optimization recommendations are."""
        # Analyze how close different configurations are in terms of efficiency
        efficiency_scores = self._calculate_balanced_efficiency_scores(memory_results)
        sorted_configs = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)

        top_configs = sorted_configs[:3]  # Top 3 configurations

        if len(top_configs) > 1:
            best_score = top_configs[0][1]
            second_best_score = top_configs[1][1]
            stability_margin = best_score - second_best_score

            # Calculate sensitivity to configuration changes
            sensitivity = self._calculate_configuration_sensitivity(
                memory_results, efficiency_scores
            )
        else:
            stability_margin = 1.0
            sensitivity = 0.0

        return {
            "stability_margin": stability_margin,
            "configuration_sensitivity": sensitivity,
            "top_configurations": [
                {"memory_size": config[0], "efficiency_score": config[1]} for config in top_configs
            ],
            "optimization_confidence": self._calculate_optimization_confidence(
                stability_margin, sensitivity
            ),
            "recommendation": self._generate_stability_recommendation(
                stability_margin, sensitivity
            ),
        }

    def _calculate_configuration_sensitivity(
        self, memory_results: Dict[int, MemoryTestResult], efficiency_scores: Dict[int, float]
    ) -> float:
        """Calculate how sensitive the performance is to configuration changes."""
        if len(efficiency_scores) < 2:
            return 0.0

        # Calculate the standard deviation of efficiency scores
        scores = list(efficiency_scores.values())
        sensitivity = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return sensitivity

    def _calculate_optimization_confidence(
        self, stability_margin: float, sensitivity: float
    ) -> float:
        """Calculate confidence in the optimization recommendation."""
        # High stability margin and low sensitivity = high confidence
        margin_confidence = min(1.0, stability_margin * 2)  # Normalize to 0-1
        sensitivity_confidence = max(0.0, 1.0 - sensitivity * 2)  # Lower sensitivity is better

        return (margin_confidence + sensitivity_confidence) / 2

    def _generate_stability_recommendation(
        self, stability_margin: float, sensitivity: float
    ) -> str:
        """Generate stability-based recommendation."""
        if stability_margin > 0.1 and sensitivity < 0.2:
            return "Highly stable optimization - safe to implement"
        elif stability_margin > 0.05 and sensitivity < 0.3:
            return "Moderately stable optimization - monitor after implementation"
        else:
            return "Less stable optimization - consider A/B testing before full implementation"

    def _generate_resource_allocation_recommendations(
        self, memory_results: Dict[int, MemoryTestResult], sweet_spot_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate resource allocation recommendations for scheduled workloads."""
        recommendations = []

        optimal_memory = sweet_spot_config["memory_size"]

        # Memory allocation recommendation
        recommendations.append(
            {
                "type": "memory_allocation",
                "priority": "high",
                "current_recommendation": f"{optimal_memory}MB",
                "rationale": "Optimal balance of cost and performance for scheduled workloads",
                "implementation": "Update Lambda function memory configuration",
            }
        )

        # Concurrent execution recommendation
        total_iterations = sum(result.iterations for result in memory_results.values())
        avg_duration = memory_results[optimal_memory].avg_duration

        # Estimate concurrent executions needed for scheduled workload
        if avg_duration > 0:
            estimated_concurrency = max(
                1, int(total_iterations / (30000 / avg_duration))
            )  # Assume 30-second windows
            recommendations.append(
                {
                    "type": "concurrency_allocation",
                    "priority": "medium",
                    "current_recommendation": f"{estimated_concurrency} concurrent executions",
                    "rationale": "Estimated concurrency for typical scheduled workload patterns",
                    "implementation": "Configure reserved concurrency if needed",
                }
            )

        # Cost optimization recommendation
        if not sweet_spot_config["is_cost_optimal"]:
            cost_difference = sweet_spot_config["cost_trade_off_vs_cheapest"]
            if cost_difference > 20:  # More than 20% cost increase
                recommendations.append(
                    {
                        "type": "cost_consideration",
                        "priority": "medium",
                        "current_recommendation": "Monitor cost impact",
                        "rationale": f"Balanced configuration costs {cost_difference:.1f}% more than cheapest option",
                        "implementation": "Set up cost alerts and regular review",
                    }
                )

        # Performance monitoring recommendation
        recommendations.append(
            {
                "type": "monitoring",
                "priority": "low",
                "current_recommendation": "Implement performance monitoring",
                "rationale": "Track performance consistency for scheduled workloads",
                "implementation": "Set up CloudWatch alarms for duration and error rates",
            }
        )

        return recommendations

    def _analyze_cost_performance_trade_offs(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze the trade-offs between cost and performance across configurations."""
        trade_offs = {}

        # Create cost vs performance curve
        for memory_size, result in memory_results.items():
            trade_offs[memory_size] = {
                "cost": result.avg_cost,
                "duration": result.avg_duration,
                "cost_efficiency": 1 / result.avg_cost if result.avg_cost > 0 else 0,
                "speed_efficiency": 1000 / result.avg_duration if result.avg_duration > 0 else 0,
                "error_rate": result.errors / result.iterations if result.iterations > 0 else 0,
            }

        # Find configurations on the Pareto frontier
        pareto_efficient = self._find_pareto_efficient_configurations(trade_offs)

        # Calculate trade-off ratios
        trade_off_analysis = self._calculate_trade_off_ratios(trade_offs)

        return {
            "trade_off_matrix": trade_offs,
            "pareto_efficient_configs": pareto_efficient,
            "trade_off_analysis": trade_off_analysis,
            "recommendations": self._generate_trade_off_recommendations(
                pareto_efficient, trade_off_analysis
            ),
        }

    def _find_pareto_efficient_configurations(
        self, trade_offs: Dict[int, Dict[str, float]]
    ) -> List[int]:
        """Find configurations that are Pareto efficient (optimal trade-offs)."""
        pareto_efficient = []

        for memory_size, metrics in trade_offs.items():
            is_dominated = False
            cost = metrics["cost"]
            duration = metrics["duration"]

            # Check if this configuration is dominated by any other
            for other_memory, other_metrics in trade_offs.items():
                if memory_size != other_memory:
                    other_cost = other_metrics["cost"]
                    other_duration = other_metrics["duration"]

                    # If other configuration is better or equal in both dimensions and strictly better in at least one
                    if (
                        other_cost <= cost
                        and other_duration <= duration
                        and (other_cost < cost or other_duration < duration)
                    ):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_efficient.append(memory_size)

        return sorted(pareto_efficient)

    def _calculate_trade_off_ratios(
        self, trade_offs: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate trade-off ratios between cost and performance."""
        costs = [metrics["cost"] for metrics in trade_offs.values()]
        durations = [metrics["duration"] for metrics in trade_offs.values()]

        cost_range = max(costs) - min(costs) if len(costs) > 1 else 0
        duration_range = max(durations) - min(durations) if len(durations) > 1 else 0

        return {
            "cost_range": cost_range,
            "duration_range": duration_range,
            "cost_sensitivity": cost_range / min(costs) if min(costs) > 0 else 0,
            "duration_sensitivity": duration_range / min(durations) if min(durations) > 0 else 0,
            "trade_off_ratio": (
                (cost_range / min(costs)) / (duration_range / min(durations))
                if min(durations) > 0 and duration_range > 0
                else 0
            ),
        }

    def _generate_trade_off_recommendations(
        self, pareto_efficient: List[int], trade_off_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on trade-off analysis."""
        recommendations = []

        if len(pareto_efficient) == 1:
            recommendations.append(f"Clear optimal choice: {pareto_efficient[0]}MB memory")
        elif len(pareto_efficient) > 1:
            recommendations.append(f"Multiple efficient options available: {pareto_efficient}")
            recommendations.append(
                "Consider business requirements to choose between cost and performance"
            )

        if trade_off_analysis["cost_sensitivity"] > trade_off_analysis["duration_sensitivity"]:
            recommendations.append("Cost varies significantly - focus on cost optimization")
        elif trade_off_analysis["duration_sensitivity"] > trade_off_analysis["cost_sensitivity"]:
            recommendations.append(
                "Performance varies significantly - focus on performance optimization"
            )
        else:
            recommendations.append(
                "Balanced cost-performance trade-offs - current balanced approach is optimal"
            )

        return recommendations

    def _generate_predictability_recommendations(
        self, predictability_metrics: Dict[int, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on predictability analysis."""
        recommendations = []

        # Find configurations with high predictability
        highly_predictable = [
            memory
            for memory, metrics in predictability_metrics.items()
            if metrics["overall_predictability"] > 0.8
        ]

        if highly_predictable:
            recommendations.append(f"Highly predictable configurations: {highly_predictable}")
            recommendations.append("These configurations are suitable for scheduled workloads")

        # Check for inconsistent configurations
        inconsistent = [
            memory
            for memory, metrics in predictability_metrics.items()
            if metrics["overall_predictability"] < 0.5
        ]

        if inconsistent:
            recommendations.append(f"Avoid inconsistent configurations: {inconsistent}")
            recommendations.append("These may cause unpredictable costs for scheduled workloads")

        return recommendations

    def _determine_scheduled_optimal_memory(
        self, memory_results: Dict[int, MemoryTestResult], scheduled_analysis: Dict[str, Any]
    ) -> int:
        """Determine optimal memory size for scheduled workloads."""
        # Use the sweet spot configuration as the primary choice
        sweet_spot_memory = scheduled_analysis["sweet_spot_configuration"]["memory_size"]

        # Validate with predictability analysis
        predictability_analysis = scheduled_analysis["predictability_analysis"]
        sweet_spot_predictability = predictability_analysis["predictability_by_memory"][
            sweet_spot_memory
        ]["overall_predictability"]

        # If sweet spot has low predictability, consider the most predictable option
        if sweet_spot_predictability < 0.6:
            most_predictable = predictability_analysis["most_predictable_config"]["memory_size"]

            # Compare efficiency scores
            sweet_spot_efficiency = scheduled_analysis["balanced_efficiency_scores"][
                sweet_spot_memory
            ]
            predictable_efficiency = scheduled_analysis["balanced_efficiency_scores"][
                most_predictable
            ]

            # If predictable option is within 10% efficiency, prefer it for scheduled workloads
            if predictable_efficiency >= sweet_spot_efficiency * 0.9:
                return most_predictable

        return sweet_spot_memory

    def _should_optimize_for_scheduled(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        efficiency_improvement: float,
    ) -> bool:
        """Determine if optimization is worthwhile for scheduled workloads."""
        # Don't optimize if already optimal
        if current_memory == optimal_memory:
            return False

        # For scheduled workloads, focus on significant efficiency improvements
        significant_efficiency_improvement = (
            efficiency_improvement > 5.0
        )  # 5% efficiency improvement
        acceptable_cost_impact = abs(cost_change) < 30.0  # Cost change less than 30%
        acceptable_performance_impact = (
            abs(duration_change) < 25.0
        )  # Performance change less than 25%

        # Optimize if efficiency improvement is significant and impacts are acceptable
        return (
            significant_efficiency_improvement
            and acceptable_cost_impact
            and acceptable_performance_impact
        )

    def _generate_scheduled_reasoning(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        efficiency_improvement: float,
        scheduled_analysis: Dict[str, Any],
    ) -> str:
        """Generate reasoning for scheduled workload optimization."""
        if current_memory == optimal_memory:
            return f"Current memory configuration ({current_memory}MB) is already optimal for scheduled workloads"

        direction = "increase" if optimal_memory > current_memory else "decrease"
        reasoning = f"Recommend {direction} memory from {current_memory}MB to {optimal_memory}MB for scheduled workload optimization. "

        # Add efficiency improvement
        reasoning += f"This improves overall efficiency by {efficiency_improvement:.1f}% "

        # Add cost and performance impacts
        if cost_change > 0:
            reasoning += f"with a cost increase of {cost_change:.1f}% "
        else:
            reasoning += f"while reducing costs by {abs(cost_change):.1f}% "

        if duration_change > 0:
            reasoning += f"and improving performance by {duration_change:.1f}%. "
        else:
            reasoning += f"and {abs(duration_change):.1f}% performance trade-off. "

        # Add scheduled workload specific context
        sweet_spot = scheduled_analysis["sweet_spot_configuration"]
        reasoning += f"This configuration provides the optimal cost-performance balance (efficiency score: {sweet_spot['balanced_efficiency_score']:.3f}) "
        reasoning += "for predictable scheduled workloads."

        return reasoning

    def _calculate_scheduled_confidence_score(
        self, analysis, optimal_memory: int, scheduled_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score with scheduled workload considerations."""
        base_confidence = self._calculate_confidence_score(analysis, optimal_memory)

        # Adjust confidence based on predictability and stability
        predictability = scheduled_analysis["predictability_analysis"]["predictability_by_memory"][
            optimal_memory
        ]["overall_predictability"]
        stability = scheduled_analysis["optimization_stability"]["optimization_confidence"]

        # Combine base confidence with scheduled workload factors
        scheduled_confidence = (base_confidence * 0.5) + (predictability * 0.3) + (stability * 0.2)

        return min(1.0, max(0.0, scheduled_confidence))

    def _estimate_scheduled_monthly_savings(
        self,
        current_result: MemoryTestResult,
        optimal_result: MemoryTestResult,
        efficiency_improvement: float,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate monthly savings for scheduled workloads."""
        base_savings = self._estimate_monthly_savings(current_result, optimal_result)

        # Add efficiency benefits specific to scheduled workloads
        for volume_key in base_savings:
            base_savings[volume_key]["efficiency_improvement_percent"] = efficiency_improvement
            # For scheduled workloads, add predictability value
            base_savings[volume_key]["predictability_benefit"] = "improved_cost_forecasting"
            # Add operational benefits
            base_savings[volume_key]["operational_benefit"] = "reduced_monitoring_overhead"

        return base_savings

    def _create_insufficient_data_recommendation(
        self, current_memory: int, optimal_memory: int
    ) -> Recommendation:
        """Create recommendation when insufficient data is available."""
        return Recommendation(
            strategy="scheduled_balanced",
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=False,
            reasoning="Insufficient data for scheduled workload recommendation. Consider running tests with representative scheduled workload patterns.",
        )
