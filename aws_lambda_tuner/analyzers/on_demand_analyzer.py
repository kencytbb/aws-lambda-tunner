"""On-demand workload analyzer for Lambda tuning - optimizes for minimal cold starts."""

import logging
import statistics
from typing import Dict, List, Optional, Any

from ..models import MemoryTestResult, Recommendation, ColdStartAnalysis
from .analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class OnDemandAnalyzer(PerformanceAnalyzer):
    """Specialized analyzer for on-demand workloads that prioritize minimal cold starts."""

    def __init__(self, config):
        """Initialize the on-demand analyzer with configuration."""
        super().__init__(config)
        self.workload_type = "on_demand"

        # On-demand specific thresholds
        self.cold_start_threshold = 0.1  # Max 10% cold start ratio acceptable
        self.latency_sensitivity = "high"  # High sensitivity to latency
        self.cost_tolerance = "medium"  # Medium tolerance for cost increases

    def analyze_for_minimal_cold_starts(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Dict[str, Any]:
        """Analyze and recommend configuration for minimal cold starts."""
        logger.info("Analyzing for minimal cold starts in on-demand workload...")

        # Perform cold start analysis
        cold_start_analysis = self.analyze_cold_starts(memory_results)

        # Find configuration that minimizes cold starts
        optimal_config = self._find_cold_start_optimal_configuration(
            memory_results, cold_start_analysis
        )

        # Analyze provisioned concurrency benefits
        provisioned_concurrency_recommendation = self._analyze_provisioned_concurrency_benefits(
            memory_results, cold_start_analysis
        )

        # Calculate latency impact of cold starts
        latency_impact = self._calculate_cold_start_latency_impact(
            memory_results, cold_start_analysis
        )

        return {
            "optimal_configuration": optimal_config,
            "provisioned_concurrency_recommendation": provisioned_concurrency_recommendation,
            "latency_impact_analysis": latency_impact,
            "cold_start_metrics": {
                "current_cold_start_ratio": cold_start_analysis.cold_start_ratio,
                "target_cold_start_ratio": self.cold_start_threshold,
                "avg_cold_start_penalty": cold_start_analysis.avg_cold_start_duration
                - cold_start_analysis.avg_warm_start_duration,
            },
        }

    def get_on_demand_recommendation(
        self, memory_results: Dict[int, MemoryTestResult]
    ) -> Recommendation:
        """Generate specialized recommendation for on-demand workloads."""
        logger.info("Generating on-demand workload recommendation...")

        # Perform comprehensive analysis
        base_analysis = self.analyze(memory_results)
        cold_start_analysis = self.analyze_cold_starts(memory_results)
        on_demand_analysis = self.analyze_for_minimal_cold_starts(memory_results)

        # Determine optimal memory size prioritizing cold start reduction
        optimal_memory = self._determine_on_demand_optimal_memory(
            memory_results, cold_start_analysis, on_demand_analysis
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

        # Calculate cold start improvement
        current_cold_start_ratio = (
            current_result.cold_starts / current_result.iterations
            if current_result.iterations > 0
            else 0
        )
        optimal_cold_start_ratio = (
            optimal_result.cold_starts / optimal_result.iterations
            if optimal_result.iterations > 0
            else 0
        )
        cold_start_improvement = (
            ((current_cold_start_ratio - optimal_cold_start_ratio) / current_cold_start_ratio) * 100
            if current_cold_start_ratio > 0
            else 0
        )

        # Determine if optimization is worthwhile for on-demand workloads
        should_optimize = self._should_optimize_for_on_demand(
            current_memory, optimal_memory, cost_change, duration_change, cold_start_improvement
        )

        # Generate on-demand specific reasoning
        reasoning = self._generate_on_demand_reasoning(
            current_memory,
            optimal_memory,
            cost_change,
            duration_change,
            cold_start_improvement,
            on_demand_analysis,
        )

        # Calculate confidence score with cold start consideration
        confidence_score = self._calculate_on_demand_confidence_score(
            base_analysis, optimal_memory, cold_start_analysis
        )

        # Estimate monthly savings including cold start impact
        monthly_savings = self._estimate_on_demand_monthly_savings(
            current_result, optimal_result, cold_start_improvement
        )

        return Recommendation(
            strategy="on_demand_optimized",
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=should_optimize,
            cost_change_percent=cost_change,
            duration_change_percent=duration_change,
            reasoning=reasoning,
            confidence_score=confidence_score,
            estimated_monthly_savings=monthly_savings,
        )

    def _find_cold_start_optimal_configuration(
        self, memory_results: Dict[int, MemoryTestResult], cold_start_analysis: ColdStartAnalysis
    ) -> Dict[str, Any]:
        """Find the configuration that best minimizes cold starts."""
        memory_cold_start_scores = {}

        for memory_size, result in memory_results.items():
            cold_start_ratio = (
                result.cold_starts / result.iterations if result.iterations > 0 else 0
            )

            # Score based on cold start ratio (lower is better) and performance
            cold_start_score = cold_start_ratio * 100  # Convert to penalty score
            performance_score = result.avg_duration / 1000  # Convert to seconds for scoring

            # Combined score (weighted toward cold start reduction)
            combined_score = (cold_start_score * 0.7) + (performance_score * 0.3)
            memory_cold_start_scores[memory_size] = combined_score

        # Find the configuration with the lowest combined score
        optimal_memory = min(memory_cold_start_scores, key=memory_cold_start_scores.get)
        optimal_result = memory_results[optimal_memory]

        return {
            "memory_size": optimal_memory,
            "cold_start_ratio": (
                optimal_result.cold_starts / optimal_result.iterations
                if optimal_result.iterations > 0
                else 0
            ),
            "avg_duration": optimal_result.avg_duration,
            "avg_cost": optimal_result.avg_cost,
            "score": memory_cold_start_scores[optimal_memory],
            "reasoning": f"Selected {optimal_memory}MB for optimal cold start performance",
        }

    def _analyze_provisioned_concurrency_benefits(
        self, memory_results: Dict[int, MemoryTestResult], cold_start_analysis: ColdStartAnalysis
    ) -> Dict[str, Any]:
        """Analyze potential benefits of using provisioned concurrency."""
        # Calculate potential cost of provisioned concurrency vs cold start impact
        avg_cold_start_duration = cold_start_analysis.avg_cold_start_duration
        avg_warm_start_duration = cold_start_analysis.avg_warm_start_duration
        cold_start_penalty = avg_cold_start_duration - avg_warm_start_duration

        # Estimate if provisioned concurrency would be beneficial
        should_use_provisioned = cold_start_analysis.cold_start_ratio > self.cold_start_threshold

        # Calculate provisioned concurrency cost estimate
        # (This would require AWS pricing data, using approximation)
        estimated_provisioned_cost_per_hour = 0.0000041667  # Approximate cost per GB-hour

        return {
            "recommended": should_use_provisioned,
            "reason": (
                f"Cold start ratio of {cold_start_analysis.cold_start_ratio:.2%} exceeds threshold of {self.cold_start_threshold:.2%}"
                if should_use_provisioned
                else "Cold start ratio is acceptable"
            ),
            "cold_start_penalty_ms": cold_start_penalty,
            "potential_latency_reduction": f"{cold_start_penalty:.0f}ms average latency reduction",
            "provisioned_concurrency_considerations": {
                "cost_impact": "Adds base cost but eliminates cold start latency",
                "use_cases": "Recommended for latency-sensitive applications",
                "monitoring_required": "Monitor actual usage patterns",
            },
        }

    def _calculate_cold_start_latency_impact(
        self, memory_results: Dict[int, MemoryTestResult], cold_start_analysis: ColdStartAnalysis
    ) -> Dict[str, Any]:
        """Calculate the latency impact of cold starts on user experience."""
        total_requests = sum(result.iterations for result in memory_results.values())
        total_cold_starts = sum(result.cold_starts for result in memory_results.values())

        # Calculate weighted average latency including cold start impact
        cold_start_penalty = (
            cold_start_analysis.avg_cold_start_duration
            - cold_start_analysis.avg_warm_start_duration
        )

        # Estimate impact on different percentiles
        p95_impact = cold_start_penalty * 0.8  # Cold starts likely impact upper percentiles more
        p99_impact = cold_start_penalty * 0.9

        return {
            "total_affected_requests": total_cold_starts,
            "percentage_affected": (
                (total_cold_starts / total_requests) * 100 if total_requests > 0 else 0
            ),
            "average_penalty_ms": cold_start_penalty,
            "estimated_p95_impact_ms": p95_impact,
            "estimated_p99_impact_ms": p99_impact,
            "user_experience_rating": self._rate_user_experience_impact(
                cold_start_analysis.cold_start_ratio, cold_start_penalty
            ),
        }

    def _rate_user_experience_impact(
        self, cold_start_ratio: float, cold_start_penalty: float
    ) -> str:
        """Rate the impact of cold starts on user experience."""
        # Combine frequency and severity of cold starts
        impact_score = cold_start_ratio * (cold_start_penalty / 1000)  # Convert to seconds

        if impact_score < 0.05:  # Less than 50ms average impact
            return "minimal"
        elif impact_score < 0.15:  # Less than 150ms average impact
            return "moderate"
        elif impact_score < 0.3:  # Less than 300ms average impact
            return "significant"
        else:
            return "severe"

    def _determine_on_demand_optimal_memory(
        self,
        memory_results: Dict[int, MemoryTestResult],
        cold_start_analysis: ColdStartAnalysis,
        on_demand_analysis: Dict[str, Any],
    ) -> int:
        """Determine optimal memory size for on-demand workloads."""
        # Start with the cold start optimal configuration
        optimal_memory = on_demand_analysis["optimal_configuration"]["memory_size"]

        # Validate that this choice makes sense for on-demand workloads
        optimal_result = memory_results[optimal_memory]
        cold_start_ratio = (
            optimal_result.cold_starts / optimal_result.iterations
            if optimal_result.iterations > 0
            else 0
        )

        # If cold start ratio is still too high, look for alternatives
        if cold_start_ratio > self.cold_start_threshold:
            # Find the highest memory configuration that reduces cold starts further
            sorted_by_memory = sorted(memory_results.items(), reverse=True)
            for memory_size, result in sorted_by_memory:
                test_cold_start_ratio = (
                    result.cold_starts / result.iterations if result.iterations > 0 else 0
                )
                if test_cold_start_ratio <= self.cold_start_threshold:
                    optimal_memory = memory_size
                    break

        return optimal_memory

    def _should_optimize_for_on_demand(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        cold_start_improvement: float,
    ) -> bool:
        """Determine if optimization is worthwhile for on-demand workloads."""
        # Don't optimize if already optimal
        if current_memory == optimal_memory:
            return False

        # For on-demand workloads, prioritize cold start reduction
        significant_cold_start_improvement = (
            cold_start_improvement > 20.0
        )  # 20% reduction in cold starts
        acceptable_cost_increase = (
            cost_change < 50.0
        )  # Accept up to 50% cost increase for cold start reduction
        performance_not_degraded = duration_change > -20.0  # Performance degradation less than 20%

        # Optimize if cold start improvement is significant and costs are acceptable
        return (
            significant_cold_start_improvement
            and acceptable_cost_increase
            and performance_not_degraded
        )

    def _generate_on_demand_reasoning(
        self,
        current_memory: int,
        optimal_memory: int,
        cost_change: float,
        duration_change: float,
        cold_start_improvement: float,
        on_demand_analysis: Dict[str, Any],
    ) -> str:
        """Generate reasoning for on-demand workload optimization."""
        if current_memory == optimal_memory:
            return f"Current memory configuration ({current_memory}MB) is already optimal for on-demand workloads"

        direction = "increase" if optimal_memory > current_memory else "decrease"
        reasoning = f"Recommend {direction} memory from {current_memory}MB to {optimal_memory}MB for on-demand optimization. "

        # Add cold start impact
        if cold_start_improvement > 0:
            reasoning += f"This reduces cold starts by {cold_start_improvement:.1f}% "

        # Add performance impact
        if duration_change > 0:
            reasoning += f"and improves performance by {duration_change:.1f}% "
        else:
            reasoning += f"with {abs(duration_change):.1f}% performance trade-off "

        # Add cost impact
        if cost_change > 0:
            reasoning += f"at a cost increase of {cost_change:.1f}%. "
        else:
            reasoning += f"while reducing costs by {abs(cost_change):.1f}%. "

        # Add on-demand specific context
        cold_start_ratio = on_demand_analysis["cold_start_metrics"]["current_cold_start_ratio"]
        if cold_start_ratio > self.cold_start_threshold:
            reasoning += f"Current cold start ratio of {cold_start_ratio:.2%} exceeds acceptable threshold for latency-sensitive on-demand workloads."

        return reasoning

    def _calculate_on_demand_confidence_score(
        self, analysis, optimal_memory: int, cold_start_analysis: ColdStartAnalysis
    ) -> float:
        """Calculate confidence score with cold start considerations."""
        base_confidence = self._calculate_confidence_score(analysis, optimal_memory)

        # Adjust confidence based on cold start data quality
        cold_start_data_quality = 1.0
        if cold_start_analysis.cold_start_ratio == 0:
            cold_start_data_quality = 0.5  # Lower confidence if no cold starts observed
        elif cold_start_analysis.cold_start_ratio < 0.05:
            cold_start_data_quality = 0.7  # Medium confidence with few cold starts

        # Combine base confidence with cold start data quality
        return (base_confidence * 0.7) + (cold_start_data_quality * 0.3)

    def _estimate_on_demand_monthly_savings(
        self,
        current_result: MemoryTestResult,
        optimal_result: MemoryTestResult,
        cold_start_improvement: float,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate monthly savings including cold start latency benefits."""
        base_savings = self._estimate_monthly_savings(current_result, optimal_result)

        # Add cold start latency value (this would depend on business impact)
        # For now, we'll add it as a qualitative benefit
        for volume_key in base_savings:
            base_savings[volume_key]["cold_start_improvement_percent"] = cold_start_improvement
            base_savings[volume_key]["latency_benefit"] = "reduced_cold_start_impact"

        return base_savings

    def _create_insufficient_data_recommendation(
        self, current_memory: int, optimal_memory: int
    ) -> Recommendation:
        """Create recommendation when insufficient data is available."""
        return Recommendation(
            strategy="on_demand_optimized",
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=False,
            reasoning="Insufficient data for on-demand workload recommendation. Consider running more comprehensive tests.",
        )
