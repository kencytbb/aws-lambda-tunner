"""Performance analyzer for Lambda tuning results."""

import logging
import statistics
from typing import Dict, List, Optional, Any

from ..models import MemoryTestResult, Recommendation, PerformanceAnalysis

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes Lambda performance tuning results."""
    
    def __init__(self, config):
        self.config = config
    
    def analyze(self, memory_results: Dict[int, MemoryTestResult], 
               baseline_results: Optional[List[Dict[str, Any]]] = None) -> PerformanceAnalysis:
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
            baseline_comparison=self._compare_to_baseline(memory_results, baseline_results)
        )
    
    def get_recommendation(self, analysis: PerformanceAnalysis, strategy: str) -> Recommendation:
        """Generate optimization recommendation based on strategy."""
        logger.info(f"Generating recommendation for strategy: {strategy}")
        
        if strategy == 'cost':
            optimal_memory = analysis.cost_optimal['memory_size']
        elif strategy == 'speed':
            optimal_memory = analysis.speed_optimal['memory_size']
        else:  # balanced
            optimal_memory = analysis.balanced_optimal['memory_size']
        
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
                reasoning="Insufficient data for recommendation"
            )
        
        # Calculate improvements
        cost_change = ((optimal_result.avg_cost - current_result.avg_cost) / current_result.avg_cost) * 100
        duration_change = ((current_result.avg_duration - optimal_result.avg_duration) / current_result.avg_duration) * 100
        
        # Determine if optimization is worthwhile
        should_optimize = self._should_optimize(current_memory, optimal_memory, cost_change, duration_change, strategy)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(current_memory, optimal_memory, cost_change, duration_change, strategy)
        
        return Recommendation(
            strategy=strategy,
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=should_optimize,
            cost_change_percent=cost_change,
            duration_change_percent=duration_change,
            reasoning=reasoning,
            confidence_score=self._calculate_confidence_score(analysis, optimal_memory),
            estimated_monthly_savings=self._estimate_monthly_savings(current_result, optimal_result)
        )
    
    def _calculate_efficiency_scores(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[int, float]:
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
            duration_score = (result.avg_duration - min_duration) / (max_duration - min_duration) if max_duration != min_duration else 0
            
            # Normalize cost (0-1, lower is better)
            cost_score = (result.avg_cost - min_cost) / (max_cost - min_cost) if max_cost != min_cost else 0
            
            # Combined efficiency score (lower is better)
            efficiency_score = (duration_score * 0.6) + (cost_score * 0.4)
            scores[memory_size] = efficiency_score
        
        return scores
    
    def _find_cost_optimal(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Find the most cost-effective configuration."""
        min_cost = float('inf')
        optimal = None
        
        for memory_size, result in memory_results.items():
            if result.avg_cost < min_cost:
                min_cost = result.avg_cost
                optimal = {
                    'memory_size': memory_size,
                    'avg_cost': result.avg_cost,
                    'avg_duration': result.avg_duration
                }
        
        return optimal
    
    def _find_speed_optimal(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Find the fastest configuration."""
        min_duration = float('inf')
        optimal = None
        
        for memory_size, result in memory_results.items():
            if result.avg_duration < min_duration:
                min_duration = result.avg_duration
                optimal = {
                    'memory_size': memory_size,
                    'avg_cost': result.avg_cost,
                    'avg_duration': result.avg_duration
                }
        
        return optimal
    
    def _find_balanced_optimal(self, memory_results: Dict[int, MemoryTestResult], 
                              efficiency_scores: Dict[int, float]) -> Dict[str, Any]:
        """Find the most balanced configuration."""
        min_score = float('inf')
        optimal = None
        
        for memory_size, score in efficiency_scores.items():
            if score < min_score:
                min_score = score
                result = memory_results[memory_size]
                optimal = {
                    'memory_size': memory_size,
                    'avg_cost': result.avg_cost,
                    'avg_duration': result.avg_duration,
                    'efficiency_score': score
                }
        
        return optimal
    
    def _calculate_trends(self, memory_results: Dict[int, MemoryTestResult]) -> Dict[str, Any]:
        """Calculate performance trends across memory sizes."""
        sorted_results = sorted(memory_results.items())
        
        memory_sizes = [item[0] for item in sorted_results]
        durations = [item[1].avg_duration for item in sorted_results]
        costs = [item[1].avg_cost for item in sorted_results]
        
        return {
            'duration_trend': self._calculate_trend_direction(memory_sizes, durations),
            'cost_trend': self._calculate_trend_direction(memory_sizes, costs),
            'diminishing_returns_point': self._find_diminishing_returns_point(sorted_results),
            'cost_efficiency_point': self._find_cost_efficiency_point(sorted_results)
        }
    
    def _calculate_trend_direction(self, x_values: List[int], y_values: List[float]) -> str:
        """Calculate if trend is increasing, decreasing, or mixed."""
        if len(y_values) < 2:
            return 'insufficient_data'
        
        increases = 0
        decreases = 0
        
        for i in range(1, len(y_values)):
            if y_values[i] > y_values[i-1]:
                increases += 1
            elif y_values[i] < y_values[i-1]:
                decreases += 1
        
        if decreases > increases * 2:
            return 'decreasing'
        elif increases > decreases * 2:
            return 'increasing'
        else:
            return 'mixed'
    
    def _generate_insights(self, memory_results: Dict[int, MemoryTestResult], 
                          efficiency_scores: Dict[int, float], 
                          trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from the analysis."""
        insights = []
        
        # Performance insights
        if trends['duration_trend'] == 'decreasing':
            insights.append({
                'type': 'performance',
                'title': 'Strong Performance Scaling',
                'description': 'Function shows consistent performance improvements with increased memory',
                'impact': 'high'
            })
        
        # Cost insights
        if trends['cost_trend'] == 'increasing':
            insights.append({
                'type': 'cost',
                'title': 'Linear Cost Increase',
                'description': 'Costs increase proportionally with memory allocation',
                'impact': 'medium'
            })
        
        return insights
    
    def _should_optimize(self, current_memory: int, optimal_memory: int, 
                        cost_change: float, duration_change: float, strategy: str) -> bool:
        """Determine if optimization is worthwhile."""
        # Don't optimize if already optimal
        if current_memory == optimal_memory:
            return False
        
        # Strategy-specific thresholds
        if strategy == 'cost':
            return cost_change < -5.0  # At least 5% cost reduction
        elif strategy == 'speed':
            return duration_change > 10.0  # At least 10% performance improvement
        else:  # balanced
            cost_acceptable = cost_change < 20.0  # Cost increase less than 20%
            speed_acceptable = duration_change > -10.0  # Performance degradation less than 10%
            significant_improvement = duration_change > 15.0 or cost_change < -10.0
            
            return cost_acceptable and speed_acceptable and significant_improvement
    
    def _generate_reasoning(self, current_memory: int, optimal_memory: int, 
                          cost_change: float, duration_change: float, strategy: str) -> str:
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
    
    def _calculate_confidence_score(self, analysis: PerformanceAnalysis, optimal_memory: int) -> float:
        """Calculate confidence score for the recommendation."""
        optimal_result = analysis.memory_results[optimal_memory]
        
        # Factor 1: Sample size (more samples = higher confidence)
        sample_factor = min(optimal_result.iterations / 20.0, 1.0)
        
        # Factor 2: Low error rate (fewer errors = higher confidence)
        error_factor = 1.0 - (optimal_result.errors / optimal_result.iterations if optimal_result.iterations > 0 else 0)
        
        confidence = (sample_factor * 0.5) + (error_factor * 0.5)
        return min(max(confidence, 0.0), 1.0)
    
    def _estimate_monthly_savings(self, current_result: MemoryTestResult, 
                                 optimal_result: MemoryTestResult) -> Dict[str, Dict[str, float]]:
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
                'cost_change': monthly_cost_change,
                'time_saved_seconds': monthly_time_saved
            }
        
        return savings
    
    def _compare_to_baseline(self, memory_results, baseline_results):
        """Compare results to baseline performance."""
        if not baseline_results:
            return None
        
        baseline_durations = [r['duration'] for r in baseline_results]
        baseline_costs = [r['cost'] for r in baseline_results]
        
        return {
            'baseline_avg_duration': statistics.mean(baseline_durations),
            'baseline_avg_cost': statistics.mean(baseline_costs),
            'sample_count': len(baseline_results)
        }
    
    def _find_diminishing_returns_point(self, sorted_results):
        """Find the point where performance gains start diminishing."""
        return None  # Simplified for this implementation
    
    def _find_cost_efficiency_point(self, sorted_results):
        """Find the most cost-efficient memory configuration."""
        return None  # Simplified for this implementation