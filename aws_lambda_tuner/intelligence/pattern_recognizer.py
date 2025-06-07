"""
Pattern recognizer for analyzing Lambda usage patterns and performance trends.
Identifies patterns in execution data to improve optimization decisions.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
from enum import Enum

from ..models import MemoryTestResult
from ..config import TunerConfig

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be recognized."""
    TEMPORAL = "temporal"
    PERFORMANCE = "performance"
    COST_EFFICIENCY = "cost_efficiency"
    ERROR = "error"
    COLD_START = "cold_start"
    CONCURRENCY = "concurrency"


@dataclass
class Pattern:
    """Represents a recognized pattern."""
    pattern_type: PatternType
    description: str
    confidence: float
    impact_score: float
    evidence: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class UsagePattern:
    """Usage pattern analysis results."""
    pattern_id: str
    pattern_type: str
    frequency: str  # daily, weekly, monthly
    peak_hours: List[int]
    baseline_metrics: Dict[str, float]
    variance_metrics: Dict[str, float]
    seasonality: Dict[str, Any]
    anomalies: List[Dict[str, Any]]


@dataclass
class PerformancePattern:
    """Performance pattern analysis results."""
    memory_efficiency_trend: str
    duration_stability: float
    cold_start_pattern: Dict[str, Any]
    error_clustering: Dict[str, Any]
    performance_degradation: Optional[Dict[str, Any]]
    optimal_memory_ranges: List[Tuple[int, int]]


class PatternRecognizer:
    """
    Advanced pattern recognition system for Lambda performance analysis.
    Identifies trends, anomalies, and optimization opportunities.
    """
    
    def __init__(self, config: TunerConfig, data_dir: Optional[str] = None):
        """
        Initialize the pattern recognizer.
        
        Args:
            config: Tuner configuration
            data_dir: Directory for storing pattern data
        """
        self.config = config
        self.data_dir = Path(data_dir or "./pattern_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Pattern recognition parameters
        self.confidence_threshold = 0.7
        self.anomaly_threshold = 2.0  # Standard deviations
        self.min_pattern_samples = 5
        
        # Load historical patterns
        self.historical_patterns = self._load_historical_patterns()
        self.baseline_metrics = self._load_baseline_metrics()
        
        logger.info("Pattern recognizer initialized")
    
    def analyze_performance_patterns(
        self,
        memory_results: Dict[int, MemoryTestResult],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Pattern]:
        """
        Analyze performance patterns in memory test results.
        
        Args:
            memory_results: Memory test results
            historical_data: Optional historical performance data
            
        Returns:
            List of recognized patterns
        """
        logger.info("Analyzing performance patterns")
        
        patterns = []
        
        try:
            # Analyze memory efficiency patterns
            memory_patterns = self._analyze_memory_efficiency_patterns(memory_results)
            patterns.extend(memory_patterns)
            
            # Analyze duration patterns
            duration_patterns = self._analyze_duration_patterns(memory_results)
            patterns.extend(duration_patterns)
            
            # Analyze cold start patterns
            cold_start_patterns = self._analyze_cold_start_patterns(memory_results)
            patterns.extend(cold_start_patterns)
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(memory_results)
            patterns.extend(error_patterns)
            
            # Analyze cost efficiency patterns
            cost_patterns = self._analyze_cost_efficiency_patterns(memory_results)
            patterns.extend(cost_patterns)
            
            # Analyze temporal patterns if historical data is available
            if historical_data:
                temporal_patterns = self._analyze_temporal_patterns(historical_data)
                patterns.extend(temporal_patterns)
            
            # Store patterns for future analysis
            self._store_patterns(patterns)
            
            logger.info(f"Identified {len(patterns)} performance patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze performance patterns: {e}")
            return []
    
    def detect_usage_patterns(
        self,
        execution_data: List[Dict[str, Any]],
        time_window: timedelta = timedelta(days=30)
    ) -> UsagePattern:
        """
        Detect usage patterns from execution data.
        
        Args:
            execution_data: List of execution records
            time_window: Time window for pattern analysis
            
        Returns:
            Usage pattern analysis
        """
        logger.debug("Detecting usage patterns")
        
        try:
            # Group executions by time
            hourly_counts = defaultdict(int)
            daily_counts = defaultdict(int)
            weekly_counts = defaultdict(int)
            
            duration_by_hour = defaultdict(list)
            error_by_hour = defaultdict(list)
            
            for execution in execution_data:
                timestamp = datetime.fromisoformat(execution.get('timestamp', ''))
                
                hour_key = timestamp.hour
                day_key = timestamp.strftime('%Y-%m-%d')
                week_key = timestamp.strftime('%Y-W%U')
                
                hourly_counts[hour_key] += 1
                daily_counts[day_key] += 1
                weekly_counts[week_key] += 1
                
                duration_by_hour[hour_key].append(execution.get('duration', 0))
                error_by_hour[hour_key].append(1 if execution.get('error') else 0)
            
            # Identify peak hours
            peak_hours = self._identify_peak_hours(hourly_counts)
            
            # Calculate baseline metrics
            baseline_metrics = self._calculate_baseline_metrics(execution_data)
            
            # Calculate variance metrics
            variance_metrics = self._calculate_variance_metrics(
                hourly_counts, daily_counts, weekly_counts
            )
            
            # Detect seasonality
            seasonality = self._detect_seasonality(daily_counts, weekly_counts)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(execution_data)
            
            pattern_id = f"usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return UsagePattern(
                pattern_id=pattern_id,
                pattern_type="temporal",
                frequency=self._determine_frequency(daily_counts, weekly_counts),
                peak_hours=peak_hours,
                baseline_metrics=baseline_metrics,
                variance_metrics=variance_metrics,
                seasonality=seasonality,
                anomalies=anomalies
            )
            
        except Exception as e:
            logger.error(f"Failed to detect usage patterns: {e}")
            return UsagePattern(
                pattern_id="error",
                pattern_type="unknown",
                frequency="unknown",
                peak_hours=[],
                baseline_metrics={},
                variance_metrics={},
                seasonality={},
                anomalies=[]
            )
    
    def identify_optimization_opportunities(
        self,
        patterns: List[Pattern],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities based on recognized patterns.
        
        Args:
            patterns: List of recognized patterns
            current_config: Current Lambda configuration
            
        Returns:
            List of optimization opportunities
        """
        logger.debug("Identifying optimization opportunities")
        
        opportunities = []
        
        # Group patterns by type
        patterns_by_type = defaultdict(list)
        for pattern in patterns:
            patterns_by_type[pattern.pattern_type].append(pattern)
        
        # Memory optimization opportunities
        if PatternType.PERFORMANCE in patterns_by_type:
            memory_opportunities = self._identify_memory_opportunities(
                patterns_by_type[PatternType.PERFORMANCE], current_config
            )
            opportunities.extend(memory_opportunities)
        
        # Cold start optimization opportunities
        if PatternType.COLD_START in patterns_by_type:
            cold_start_opportunities = self._identify_cold_start_opportunities(
                patterns_by_type[PatternType.COLD_START], current_config
            )
            opportunities.extend(cold_start_opportunities)
        
        # Cost optimization opportunities
        if PatternType.COST_EFFICIENCY in patterns_by_type:
            cost_opportunities = self._identify_cost_opportunities(
                patterns_by_type[PatternType.COST_EFFICIENCY], current_config
            )
            opportunities.extend(cost_opportunities)
        
        # Error reduction opportunities
        if PatternType.ERROR in patterns_by_type:
            error_opportunities = self._identify_error_opportunities(
                patterns_by_type[PatternType.ERROR], current_config
            )
            opportunities.extend(error_opportunities)
        
        # Temporal optimization opportunities
        if PatternType.TEMPORAL in patterns_by_type:
            temporal_opportunities = self._identify_temporal_opportunities(
                patterns_by_type[PatternType.TEMPORAL], current_config
            )
            opportunities.extend(temporal_opportunities)
        
        # Sort by impact score
        opportunities.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
        
        return opportunities
    
    def _analyze_memory_efficiency_patterns(
        self,
        memory_results: Dict[int, MemoryTestResult]
    ) -> List[Pattern]:
        """Analyze memory efficiency patterns."""
        patterns = []
        
        if len(memory_results) < 2:
            return patterns
        
        memory_sizes = sorted(memory_results.keys())
        durations = [memory_results[m].avg_duration for m in memory_sizes]
        costs = [memory_results[m].avg_cost for m in memory_sizes]
        
        # Calculate efficiency metrics
        duration_improvements = []
        cost_increases = []
        
        for i in range(1, len(memory_sizes)):
            prev_duration = durations[i-1]
            curr_duration = durations[i]
            prev_cost = costs[i-1]
            curr_cost = costs[i]
            
            duration_improvement = (prev_duration - curr_duration) / prev_duration if prev_duration > 0 else 0
            cost_increase = (curr_cost - prev_cost) / prev_cost if prev_cost > 0 else 0
            
            duration_improvements.append(duration_improvement)
            cost_increases.append(cost_increase)
        
        # Identify diminishing returns pattern
        if len(duration_improvements) >= 3:
            diminishing_threshold = 0.05  # 5% improvement threshold
            diminishing_point = None
            
            for i in range(1, len(duration_improvements)):
                if duration_improvements[i] < diminishing_threshold and duration_improvements[i-1] >= diminishing_threshold:
                    diminishing_point = memory_sizes[i+1]
                    break
            
            if diminishing_point:
                patterns.append(Pattern(
                    pattern_type=PatternType.PERFORMANCE,
                    description=f"Diminishing returns observed beyond {diminishing_point}MB",
                    confidence=0.8,
                    impact_score=0.7,
                    evidence={
                        'diminishing_point': diminishing_point,
                        'duration_improvements': duration_improvements
                    },
                    recommendations=[
                        f"Consider memory sizes up to {diminishing_point}MB for optimal efficiency",
                        "Evaluate cost-benefit ratio for higher memory allocations"
                    ],
                    metadata={'pattern_subtype': 'diminishing_returns'}
                ))
        
        # Identify sweet spot pattern
        efficiency_scores = []
        for i in range(len(memory_sizes)):
            if i < len(duration_improvements) and i < len(cost_increases):
                # Balance duration improvement and cost increase
                efficiency = duration_improvements[i] - (cost_increases[i] * 0.5)
                efficiency_scores.append((memory_sizes[i+1], efficiency))
        
        if efficiency_scores:
            best_efficiency = max(efficiency_scores, key=lambda x: x[1])
            if best_efficiency[1] > 0.1:  # Significant efficiency
                patterns.append(Pattern(
                    pattern_type=PatternType.PERFORMANCE,
                    description=f"Performance sweet spot identified at {best_efficiency[0]}MB",
                    confidence=0.9,
                    impact_score=0.8,
                    evidence={
                        'sweet_spot_memory': best_efficiency[0],
                        'efficiency_score': best_efficiency[1],
                        'efficiency_scores': efficiency_scores
                    },
                    recommendations=[
                        f"Consider {best_efficiency[0]}MB as optimal memory configuration",
                        "Monitor performance at this configuration"
                    ],
                    metadata={'pattern_subtype': 'sweet_spot'}
                ))
        
        return patterns
    
    def _analyze_duration_patterns(
        self,
        memory_results: Dict[int, MemoryTestResult]
    ) -> List[Pattern]:
        """Analyze duration patterns."""
        patterns = []
        
        durations = [result.avg_duration for result in memory_results.values()]
        p95_durations = [result.p95_duration for result in memory_results.values()]
        
        if not durations:
            return patterns
        
        # Analyze duration stability
        duration_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
        
        if duration_cv < 0.1:  # Low coefficient of variation
            patterns.append(Pattern(
                pattern_type=PatternType.PERFORMANCE,
                description="Stable duration performance across memory configurations",
                confidence=0.8,
                impact_score=0.5,
                evidence={
                    'coefficient_of_variation': duration_cv,
                    'duration_range': (min(durations), max(durations))
                },
                recommendations=[
                    "Duration is stable - optimize for cost efficiency",
                    "Consider lower memory configurations"
                ],
                metadata={'pattern_subtype': 'duration_stability'}
            ))
        elif duration_cv > 0.3:  # High variation
            patterns.append(Pattern(
                pattern_type=PatternType.PERFORMANCE,
                description="High duration variability across memory configurations",
                confidence=0.9,
                impact_score=0.8,
                evidence={
                    'coefficient_of_variation': duration_cv,
                    'duration_range': (min(durations), max(durations))
                },
                recommendations=[
                    "Investigate causes of duration variability",
                    "Consider memory optimization for consistency"
                ],
                metadata={'pattern_subtype': 'duration_variability'}
            ))
        
        # Analyze p95 vs average duration patterns
        tail_latency_ratios = []
        for result in memory_results.values():
            if result.avg_duration > 0:
                ratio = result.p95_duration / result.avg_duration
                tail_latency_ratios.append(ratio)
        
        if tail_latency_ratios:
            avg_tail_ratio = np.mean(tail_latency_ratios)
            if avg_tail_ratio > 2.0:  # High tail latency
                patterns.append(Pattern(
                    pattern_type=PatternType.PERFORMANCE,
                    description="High tail latency detected",
                    confidence=0.8,
                    impact_score=0.7,
                    evidence={
                        'avg_tail_latency_ratio': avg_tail_ratio,
                        'tail_latency_ratios': tail_latency_ratios
                    },
                    recommendations=[
                        "Investigate causes of tail latency",
                        "Consider higher memory to reduce variability"
                    ],
                    metadata={'pattern_subtype': 'tail_latency'}
                ))
        
        return patterns
    
    def _analyze_cold_start_patterns(
        self,
        memory_results: Dict[int, MemoryTestResult]
    ) -> List[Pattern]:
        """Analyze cold start patterns."""
        patterns = []
        
        memory_sizes = sorted(memory_results.keys())
        cold_start_ratios = []
        
        for memory_size in memory_sizes:
            result = memory_results[memory_size]
            if result.iterations > 0:
                cold_start_ratio = result.cold_starts / result.iterations
                cold_start_ratios.append((memory_size, cold_start_ratio))
        
        if len(cold_start_ratios) < 2:
            return patterns
        
        # Analyze cold start reduction with memory
        memory_values = [x[0] for x in cold_start_ratios]
        ratio_values = [x[1] for x in cold_start_ratios]
        
        # Calculate correlation between memory and cold start reduction
        if len(memory_values) > 2:
            correlation = np.corrcoef(memory_values, ratio_values)[0, 1]
            
            if correlation < -0.5:  # Strong negative correlation
                patterns.append(Pattern(
                    pattern_type=PatternType.COLD_START,
                    description="Higher memory significantly reduces cold starts",
                    confidence=abs(correlation),
                    impact_score=0.8,
                    evidence={
                        'correlation': correlation,
                        'cold_start_ratios': cold_start_ratios
                    },
                    recommendations=[
                        "Consider higher memory allocation to reduce cold starts",
                        "Monitor cold start impact on user experience"
                    ],
                    metadata={'pattern_subtype': 'memory_cold_start_correlation'}
                ))
        
        # Identify high cold start configurations
        high_cold_start_threshold = 0.3  # 30% cold starts
        high_cold_start_configs = [
            (memory, ratio) for memory, ratio in cold_start_ratios
            if ratio > high_cold_start_threshold
        ]
        
        if high_cold_start_configs:
            patterns.append(Pattern(
                pattern_type=PatternType.COLD_START,
                description=f"High cold start rates detected in {len(high_cold_start_configs)} configurations",
                confidence=0.9,
                impact_score=0.7,
                evidence={
                    'high_cold_start_configs': high_cold_start_configs,
                    'threshold': high_cold_start_threshold
                },
                recommendations=[
                    "Optimize memory configuration to reduce cold starts",
                    "Consider provisioned concurrency for critical workloads"
                ],
                metadata={'pattern_subtype': 'high_cold_start_rate'}
            ))
        
        return patterns
    
    def _analyze_error_patterns(
        self,
        memory_results: Dict[int, MemoryTestResult]
    ) -> List[Pattern]:
        """Analyze error patterns."""
        patterns = []
        
        memory_sizes = sorted(memory_results.keys())
        error_rates = []
        
        for memory_size in memory_sizes:
            result = memory_results[memory_size]
            if result.iterations > 0:
                error_rate = result.errors / result.iterations
                error_rates.append((memory_size, error_rate))
        
        if not error_rates:
            return patterns
        
        # Identify configurations with high error rates
        high_error_threshold = 0.05  # 5% error rate
        high_error_configs = [
            (memory, rate) for memory, rate in error_rates
            if rate > high_error_threshold
        ]
        
        if high_error_configs:
            patterns.append(Pattern(
                pattern_type=PatternType.ERROR,
                description=f"High error rates detected in {len(high_error_configs)} configurations",
                confidence=0.9,
                impact_score=0.9,
                evidence={
                    'high_error_configs': high_error_configs,
                    'threshold': high_error_threshold
                },
                recommendations=[
                    "Investigate error causes in problematic configurations",
                    "Consider avoiding configurations with high error rates"
                ],
                metadata={'pattern_subtype': 'high_error_rate'}
            ))
        
        # Analyze error correlation with memory
        memory_values = [x[0] for x in error_rates]
        rate_values = [x[1] for x in error_rates]
        
        if len(memory_values) > 2 and max(rate_values) > 0:
            correlation = np.corrcoef(memory_values, rate_values)[0, 1]
            
            if abs(correlation) > 0.5:  # Strong correlation
                pattern_desc = "Higher memory reduces errors" if correlation < 0 else "Higher memory increases errors"
                patterns.append(Pattern(
                    pattern_type=PatternType.ERROR,
                    description=pattern_desc,
                    confidence=abs(correlation),
                    impact_score=0.8,
                    evidence={
                        'correlation': correlation,
                        'error_rates': error_rates
                    },
                    recommendations=[
                        "Consider memory impact on error rates",
                        "Investigate memory-related error patterns"
                    ],
                    metadata={'pattern_subtype': 'memory_error_correlation'}
                ))
        
        return patterns
    
    def _analyze_cost_efficiency_patterns(
        self,
        memory_results: Dict[int, MemoryTestResult]
    ) -> List[Pattern]:
        """Analyze cost efficiency patterns."""
        patterns = []
        
        memory_sizes = sorted(memory_results.keys())
        cost_efficiency_scores = []
        
        for memory_size in memory_sizes:
            result = memory_results[memory_size]
            if result.avg_cost > 0 and result.avg_duration > 0:
                # Cost efficiency = 1 / (cost * duration)
                efficiency = 1.0 / (result.avg_cost * result.avg_duration)
                cost_efficiency_scores.append((memory_size, efficiency))
        
        if len(cost_efficiency_scores) < 2:
            return patterns
        
        # Find most cost-efficient configuration
        best_efficiency = max(cost_efficiency_scores, key=lambda x: x[1])
        
        patterns.append(Pattern(
            pattern_type=PatternType.COST_EFFICIENCY,
            description=f"Most cost-efficient configuration: {best_efficiency[0]}MB",
            confidence=0.8,
            impact_score=0.7,
            evidence={
                'best_config': best_efficiency,
                'efficiency_scores': cost_efficiency_scores
            },
            recommendations=[
                f"Consider {best_efficiency[0]}MB for optimal cost efficiency",
                "Balance cost efficiency with performance requirements"
            ],
            metadata={'pattern_subtype': 'cost_optimal'}
        ))
        
        return patterns
    
    def _analyze_temporal_patterns(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Analyze temporal patterns in historical data."""
        patterns = []
        
        # This would analyze time-based patterns in actual implementation
        # For now, return placeholder patterns
        
        if len(historical_data) > 10:
            patterns.append(Pattern(
                pattern_type=PatternType.TEMPORAL,
                description="Sufficient historical data available for trend analysis",
                confidence=0.7,
                impact_score=0.5,
                evidence={'data_points': len(historical_data)},
                recommendations=[
                    "Continue monitoring for temporal patterns",
                    "Consider seasonal optimization strategies"
                ],
                metadata={'pattern_subtype': 'data_availability'}
            ))
        
        return patterns
    
    def _identify_peak_hours(self, hourly_counts: Dict[int, int]) -> List[int]:
        """Identify peak usage hours."""
        if not hourly_counts:
            return []
        
        counts = list(hourly_counts.values())
        if not counts:
            return []
        
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        threshold = mean_count + std_count
        
        peak_hours = [
            hour for hour, count in hourly_counts.items()
            if count > threshold
        ]
        
        return sorted(peak_hours)
    
    def _calculate_baseline_metrics(self, execution_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate baseline performance metrics."""
        if not execution_data:
            return {}
        
        durations = [exec_data.get('duration', 0) for exec_data in execution_data]
        errors = [1 if exec_data.get('error') else 0 for exec_data in execution_data]
        
        return {
            'avg_duration': np.mean(durations) if durations else 0.0,
            'p95_duration': np.percentile(durations, 95) if durations else 0.0,
            'error_rate': np.mean(errors) if errors else 0.0,
            'total_executions': len(execution_data)
        }
    
    def _calculate_variance_metrics(
        self,
        hourly_counts: Dict[int, int],
        daily_counts: Dict[int, int],
        weekly_counts: Dict[int, int]
    ) -> Dict[str, float]:
        """Calculate variance metrics."""
        metrics = {}
        
        if hourly_counts:
            hourly_values = list(hourly_counts.values())
            metrics['hourly_cv'] = np.std(hourly_values) / np.mean(hourly_values) if np.mean(hourly_values) > 0 else 0
        
        if daily_counts:
            daily_values = list(daily_counts.values())
            metrics['daily_cv'] = np.std(daily_values) / np.mean(daily_values) if np.mean(daily_values) > 0 else 0
        
        if weekly_counts:
            weekly_values = list(weekly_counts.values())
            metrics['weekly_cv'] = np.std(weekly_values) / np.mean(weekly_values) if np.mean(weekly_values) > 0 else 0
        
        return metrics
    
    def _detect_seasonality(
        self,
        daily_counts: Dict[str, int],
        weekly_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Detect seasonal patterns."""
        seasonality = {
            'daily_pattern': False,
            'weekly_pattern': False,
            'patterns': []
        }
        
        # Simple seasonality detection (would be more sophisticated in real implementation)
        if len(daily_counts) >= 7:
            daily_values = list(daily_counts.values())
            daily_cv = np.std(daily_values) / np.mean(daily_values) if np.mean(daily_values) > 0 else 0
            
            if daily_cv > 0.3:  # High variation suggests patterns
                seasonality['daily_pattern'] = True
                seasonality['patterns'].append('daily_variation')
        
        if len(weekly_counts) >= 4:
            weekly_values = list(weekly_counts.values())
            weekly_cv = np.std(weekly_values) / np.mean(weekly_values) if np.mean(weekly_values) > 0 else 0
            
            if weekly_cv > 0.2:
                seasonality['weekly_pattern'] = True
                seasonality['patterns'].append('weekly_variation')
        
        return seasonality
    
    def _detect_anomalies(self, execution_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in execution data."""
        anomalies = []
        
        if len(execution_data) < 10:
            return anomalies
        
        durations = [exec_data.get('duration', 0) for exec_data in execution_data]
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        threshold = mean_duration + (self.anomaly_threshold * std_duration)
        
        for i, exec_data in enumerate(execution_data):
            duration = exec_data.get('duration', 0)
            if duration > threshold:
                anomalies.append({
                    'index': i,
                    'timestamp': exec_data.get('timestamp'),
                    'duration': duration,
                    'anomaly_type': 'high_duration',
                    'severity': (duration - threshold) / std_duration if std_duration > 0 else 0
                })
        
        return anomalies
    
    def _determine_frequency(
        self,
        daily_counts: Dict[str, int],
        weekly_counts: Dict[str, int]
    ) -> str:
        """Determine execution frequency pattern."""
        if len(daily_counts) >= 7:
            daily_values = list(daily_counts.values())
            if max(daily_values) > 100:
                return "high"
            elif max(daily_values) > 10:
                return "medium"
            else:
                return "low"
        
        return "unknown"
    
    def _identify_memory_opportunities(
        self,
        performance_patterns: List[Pattern],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify memory optimization opportunities."""
        opportunities = []
        
        for pattern in performance_patterns:
            if pattern.metadata.get('pattern_subtype') == 'sweet_spot':
                opportunities.append({
                    'type': 'memory_optimization',
                    'title': 'Memory Sweet Spot Identified',
                    'description': pattern.description,
                    'impact_score': pattern.impact_score,
                    'confidence': pattern.confidence,
                    'action': f"Update memory to {pattern.evidence.get('sweet_spot_memory')}MB",
                    'expected_benefit': 'Improved cost-performance ratio'
                })
            
            elif pattern.metadata.get('pattern_subtype') == 'diminishing_returns':
                opportunities.append({
                    'type': 'memory_optimization',
                    'title': 'Diminishing Returns Detected',
                    'description': pattern.description,
                    'impact_score': pattern.impact_score,
                    'confidence': pattern.confidence,
                    'action': f"Avoid memory above {pattern.evidence.get('diminishing_point')}MB",
                    'expected_benefit': 'Cost savings without performance loss'
                })
        
        return opportunities
    
    def _identify_cold_start_opportunities(
        self,
        cold_start_patterns: List[Pattern],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify cold start optimization opportunities."""
        opportunities = []
        
        for pattern in cold_start_patterns:
            if pattern.metadata.get('pattern_subtype') == 'high_cold_start_rate':
                opportunities.append({
                    'type': 'cold_start_optimization',
                    'title': 'High Cold Start Rate Detected',
                    'description': pattern.description,
                    'impact_score': pattern.impact_score,
                    'confidence': pattern.confidence,
                    'action': 'Consider higher memory or provisioned concurrency',
                    'expected_benefit': 'Reduced cold start latency'
                })
        
        return opportunities
    
    def _identify_cost_opportunities(
        self,
        cost_patterns: List[Pattern],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        for pattern in cost_patterns:
            if pattern.metadata.get('pattern_subtype') == 'cost_optimal':
                opportunities.append({
                    'type': 'cost_optimization',
                    'title': 'Cost-Optimal Configuration Found',
                    'description': pattern.description,
                    'impact_score': pattern.impact_score,
                    'confidence': pattern.confidence,
                    'action': f"Consider memory configuration from pattern analysis",
                    'expected_benefit': 'Reduced operational costs'
                })
        
        return opportunities
    
    def _identify_error_opportunities(
        self,
        error_patterns: List[Pattern],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify error reduction opportunities."""
        opportunities = []
        
        for pattern in error_patterns:
            if pattern.metadata.get('pattern_subtype') == 'high_error_rate':
                opportunities.append({
                    'type': 'reliability_improvement',
                    'title': 'High Error Rate Detected',
                    'description': pattern.description,
                    'impact_score': pattern.impact_score,
                    'confidence': pattern.confidence,
                    'action': 'Investigate and resolve error causes',
                    'expected_benefit': 'Improved reliability and user experience'
                })
        
        return opportunities
    
    def _identify_temporal_opportunities(
        self,
        temporal_patterns: List[Pattern],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify temporal optimization opportunities."""
        opportunities = []
        
        for pattern in temporal_patterns:
            opportunities.append({
                'type': 'temporal_optimization',
                'title': 'Temporal Pattern Identified',
                'description': pattern.description,
                'impact_score': pattern.impact_score,
                'confidence': pattern.confidence,
                'action': 'Monitor for seasonal optimization opportunities',
                'expected_benefit': 'Time-based performance optimization'
            })
        
        return opportunities
    
    def _store_patterns(self, patterns: List[Pattern]):
        """Store recognized patterns for future analysis."""
        try:
            pattern_records = []
            for pattern in patterns:
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'function_arn': self.config.function_arn,
                    'pattern_type': pattern.pattern_type.value,
                    'description': pattern.description,
                    'confidence': pattern.confidence,
                    'impact_score': pattern.impact_score,
                    'evidence': pattern.evidence,
                    'metadata': pattern.metadata
                }
                pattern_records.append(record)
            
            patterns_file = self.data_dir / f"patterns_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(patterns_file, 'a') as f:
                for record in pattern_records:
                    f.write(json.dumps(record) + '\n')
                    
        except Exception as e:
            logger.warning(f"Failed to store patterns: {e}")
    
    def _load_historical_patterns(self) -> List[Dict[str, Any]]:
        """Load historical pattern data."""
        historical_patterns = []
        
        try:
            for pattern_file in self.data_dir.glob("patterns_*.jsonl"):
                with open(pattern_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            historical_patterns.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load historical patterns: {e}")
        
        return historical_patterns
    
    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics for comparison."""
        baseline_file = self.data_dir / "baseline_metrics.json"
        
        try:
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load baseline metrics: {e}")
        
        return {}