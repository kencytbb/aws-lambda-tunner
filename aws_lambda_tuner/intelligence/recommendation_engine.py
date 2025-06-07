"""
Intelligent recommendation engine using ML-based pattern recognition.
Provides advanced recommendations based on historical data and learned patterns.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..models import (
    MemoryTestResult, Recommendation, PerformanceAnalysis,
    ColdStartAnalysis, ConcurrencyAnalysis, WorkloadAnalysis
)
from ..config import TunerConfig
from .pattern_recognizer import PatternRecognizer
from .cost_predictor import CostPredictor

logger = logging.getLogger(__name__)


@dataclass
class MLRecommendation:
    """Enhanced recommendation with ML insights."""
    base_recommendation: Recommendation
    confidence_score: float
    pattern_match_score: float
    similar_functions: List[str]
    predicted_performance: Dict[str, float]
    risk_assessment: Dict[str, Any]
    optimization_timeline: Dict[str, Any]


@dataclass
class LearningFeatures:
    """Features extracted for ML model."""
    workload_type: str
    traffic_pattern: str
    cold_start_sensitivity: str
    avg_duration: float
    p95_duration: float
    memory_efficiency: float
    cost_efficiency: float
    concurrency_ratio: float
    error_rate: float
    seasonal_variance: float


class IntelligentRecommendationEngine:
    """
    ML-based recommendation engine that learns from historical data
    and provides intelligent optimization suggestions.
    """
    
    def __init__(self, config: TunerConfig, data_dir: Optional[str] = None):
        """
        Initialize the intelligent recommendation engine.
        
        Args:
            config: Tuner configuration
            data_dir: Directory to store learning data
        """
        self.config = config
        self.data_dir = Path(data_dir or "./ml_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.pattern_recognizer = PatternRecognizer(config)
        self.cost_predictor = CostPredictor(config)
        
        # Initialize learning components
        self.historical_data = self._load_historical_data()
        self.function_profiles = self._load_function_profiles()
        self.performance_models = self._initialize_performance_models()
        
        logger.info("Intelligent recommendation engine initialized")
    
    def generate_intelligent_recommendation(
        self,
        analysis: PerformanceAnalysis,
        memory_results: Dict[int, MemoryTestResult]
    ) -> MLRecommendation:
        """
        Generate intelligent recommendations using ML insights.
        
        Args:
            analysis: Performance analysis results
            memory_results: Memory test results
            
        Returns:
            Enhanced ML-based recommendation
        """
        logger.info("Generating intelligent recommendation with ML insights")
        
        try:
            # Extract features for ML model
            features = self._extract_features(analysis, memory_results)
            
            # Find similar functions for pattern matching
            similar_functions = self._find_similar_functions(features)
            
            # Generate base recommendation
            base_rec = self._generate_base_recommendation(analysis, memory_results)
            
            # Enhance with ML insights
            ml_insights = self._apply_ml_insights(features, similar_functions, base_rec)
            
            # Calculate confidence and risk assessment
            confidence = self._calculate_confidence_score(features, similar_functions)
            risk_assessment = self._assess_optimization_risks(features, base_rec)
            
            # Predict performance outcomes
            predicted_performance = self._predict_performance_outcomes(
                features, base_rec.optimal_memory_size
            )
            
            # Generate optimization timeline
            timeline = self._generate_optimization_timeline(features, base_rec)
            
            # Create enhanced recommendation
            ml_recommendation = MLRecommendation(
                base_recommendation=base_rec,
                confidence_score=confidence,
                pattern_match_score=ml_insights.get('pattern_score', 0.0),
                similar_functions=similar_functions,
                predicted_performance=predicted_performance,
                risk_assessment=risk_assessment,
                optimization_timeline=timeline
            )
            
            # Store learning data for future improvements
            self._store_learning_data(features, ml_recommendation)
            
            logger.info(f"Generated ML recommendation with {confidence:.2f} confidence")
            return ml_recommendation
            
        except Exception as e:
            logger.error(f"Failed to generate intelligent recommendation: {e}")
            # Fallback to basic recommendation
            base_rec = self._generate_base_recommendation(analysis, memory_results)
            return MLRecommendation(
                base_recommendation=base_rec,
                confidence_score=0.5,
                pattern_match_score=0.0,
                similar_functions=[],
                predicted_performance={},
                risk_assessment={'level': 'unknown'},
                optimization_timeline={}
            )
    
    def _extract_features(
        self,
        analysis: PerformanceAnalysis,
        memory_results: Dict[int, MemoryTestResult]
    ) -> LearningFeatures:
        """Extract features for ML model."""
        # Calculate aggregate metrics
        durations = [result.avg_duration for result in memory_results.values()]
        p95_durations = [result.p95_duration for result in memory_results.values()]
        costs = [result.avg_cost for result in memory_results.values()]
        
        avg_duration = np.mean(durations) if durations else 0.0
        p95_duration = np.mean(p95_durations) if p95_durations else 0.0
        
        # Calculate efficiency metrics
        memory_efficiency = self._calculate_memory_efficiency(memory_results)
        cost_efficiency = self._calculate_cost_efficiency(memory_results)
        
        # Extract concurrency and error metrics
        total_executions = sum(result.iterations for result in memory_results.values())
        total_errors = sum(result.errors for result in memory_results.values())
        error_rate = total_errors / total_executions if total_executions > 0 else 0.0
        
        concurrency_ratio = self.config.expected_concurrency / 100.0  # Normalize
        
        # Calculate seasonal variance (placeholder - would use historical data)
        seasonal_variance = 0.1  # Default low variance
        
        return LearningFeatures(
            workload_type=self.config.workload_type,
            traffic_pattern=self.config.traffic_pattern,
            cold_start_sensitivity=self.config.cold_start_sensitivity,
            avg_duration=avg_duration,
            p95_duration=p95_duration,
            memory_efficiency=memory_efficiency,
            cost_efficiency=cost_efficiency,
            concurrency_ratio=concurrency_ratio,
            error_rate=error_rate,
            seasonal_variance=seasonal_variance
        )
    
    def _calculate_memory_efficiency(self, memory_results: Dict[int, MemoryTestResult]) -> float:
        """Calculate memory efficiency score."""
        if not memory_results:
            return 0.0
        
        # Calculate efficiency as performance improvement per MB
        efficiencies = []
        memory_sizes = sorted(memory_results.keys())
        
        for i in range(1, len(memory_sizes)):
            prev_memory = memory_sizes[i-1]
            curr_memory = memory_sizes[i]
            
            prev_result = memory_results[prev_memory]
            curr_result = memory_results[curr_memory]
            
            memory_increase = curr_memory - prev_memory
            duration_improvement = prev_result.avg_duration - curr_result.avg_duration
            
            if memory_increase > 0:
                efficiency = duration_improvement / memory_increase
                efficiencies.append(max(0, efficiency))  # Only positive improvements
        
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def _calculate_cost_efficiency(self, memory_results: Dict[int, MemoryTestResult]) -> float:
        """Calculate cost efficiency score."""
        if not memory_results:
            return 0.0
        
        # Find the configuration with best cost-performance ratio
        best_ratio = 0.0
        for result in memory_results.values():
            if result.avg_cost > 0:
                # Lower duration and cost is better
                ratio = 1.0 / (result.avg_duration * result.avg_cost)
                best_ratio = max(best_ratio, ratio)
        
        return min(best_ratio, 1.0)  # Normalize to 0-1
    
    def _find_similar_functions(self, features: LearningFeatures) -> List[str]:
        """Find similar functions based on feature matching."""
        similar_functions = []
        
        for function_arn, profile in self.function_profiles.items():
            if function_arn == self.config.function_arn:
                continue
            
            similarity_score = self._calculate_similarity_score(features, profile)
            if similarity_score > 0.7:  # High similarity threshold
                similar_functions.append(function_arn)
        
        return similar_functions[:5]  # Top 5 similar functions
    
    def _calculate_similarity_score(self, features: LearningFeatures, profile: Dict[str, Any]) -> float:
        """Calculate similarity score between current function and profile."""
        score = 0.0
        total_weight = 0.0
        
        # Workload type similarity (high weight)
        if features.workload_type == profile.get('workload_type'):
            score += 0.3
        total_weight += 0.3
        
        # Traffic pattern similarity (high weight)
        if features.traffic_pattern == profile.get('traffic_pattern'):
            score += 0.25
        total_weight += 0.25
        
        # Performance similarity (medium weight)
        profile_duration = profile.get('avg_duration', 0)
        if profile_duration > 0:
            duration_similarity = 1.0 - abs(features.avg_duration - profile_duration) / max(features.avg_duration, profile_duration)
            score += 0.2 * max(0, duration_similarity)
        total_weight += 0.2
        
        # Efficiency similarity (medium weight)
        profile_efficiency = profile.get('memory_efficiency', 0)
        if profile_efficiency > 0:
            efficiency_similarity = 1.0 - abs(features.memory_efficiency - profile_efficiency) / max(features.memory_efficiency, profile_efficiency)
            score += 0.15 * max(0, efficiency_similarity)
        total_weight += 0.15
        
        # Cold start sensitivity (low weight)
        if features.cold_start_sensitivity == profile.get('cold_start_sensitivity'):
            score += 0.1
        total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _generate_base_recommendation(
        self,
        analysis: PerformanceAnalysis,
        memory_results: Dict[int, MemoryTestResult]
    ) -> Recommendation:
        """Generate base recommendation from analysis."""
        strategy_optimal = None
        current_memory = 1024  # Default assumption
        
        if self.config.strategy == "cost":
            strategy_optimal = analysis.cost_optimal
        elif self.config.strategy == "speed":
            strategy_optimal = analysis.speed_optimal
        else:
            strategy_optimal = analysis.balanced_optimal
        
        optimal_memory = strategy_optimal.get('memory_size', current_memory)
        
        # Calculate improvements
        current_result = memory_results.get(current_memory)
        optimal_result = memory_results.get(optimal_memory)
        
        cost_change = 0.0
        duration_change = 0.0
        
        if current_result and optimal_result:
            if current_result.avg_cost > 0:
                cost_change = ((optimal_result.avg_cost - current_result.avg_cost) / current_result.avg_cost) * 100
            if current_result.avg_duration > 0:
                duration_change = ((optimal_result.avg_duration - current_result.avg_duration) / current_result.avg_duration) * 100
        
        return Recommendation(
            strategy=self.config.strategy,
            current_memory_size=current_memory,
            optimal_memory_size=optimal_memory,
            should_optimize=optimal_memory != current_memory,
            cost_change_percent=cost_change,
            duration_change_percent=duration_change,
            reasoning=f"Optimal configuration for {self.config.strategy} strategy",
            confidence_score=0.8
        )
    
    def _apply_ml_insights(
        self,
        features: LearningFeatures,
        similar_functions: List[str],
        base_rec: Recommendation
    ) -> Dict[str, Any]:
        """Apply ML insights to enhance recommendation."""
        insights = {
            'pattern_score': 0.0,
            'adjustments': [],
            'confidence_factors': []
        }
        
        # Pattern matching insights
        if similar_functions:
            pattern_score = len(similar_functions) / 10.0  # Normalize by max similar functions
            insights['pattern_score'] = min(pattern_score, 1.0)
            insights['confidence_factors'].append(f"Found {len(similar_functions)} similar functions")
        
        # Workload-specific adjustments
        if features.workload_type == "continuous" and features.error_rate > 0.01:
            insights['adjustments'].append("Consider higher memory for continuous workloads with errors")
        
        if features.cold_start_sensitivity == "high" and base_rec.optimal_memory_size < 512:
            insights['adjustments'].append("Consider higher memory to reduce cold starts")
        
        # Traffic pattern adjustments
        if features.traffic_pattern == "burst" and features.concurrency_ratio > 0.5:
            insights['adjustments'].append("High concurrency burst pattern detected")
        
        return insights
    
    def _calculate_confidence_score(
        self,
        features: LearningFeatures,
        similar_functions: List[str]
    ) -> float:
        """Calculate confidence score for recommendation."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on similar functions
        if similar_functions:
            confidence += min(len(similar_functions) * 0.1, 0.3)
        
        # Increase confidence for stable patterns
        if features.error_rate < 0.01:
            confidence += 0.1
        
        # Increase confidence for well-defined workload types
        if features.workload_type in ["continuous", "scheduled"]:
            confidence += 0.1
        
        # Decrease confidence for high variance
        if features.seasonal_variance > 0.3:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _assess_optimization_risks(
        self,
        features: LearningFeatures,
        recommendation: Recommendation
    ) -> Dict[str, Any]:
        """Assess risks of implementing the recommendation."""
        risk_assessment = {
            'level': 'low',
            'factors': [],
            'mitigation': []
        }
        
        # High memory changes are riskier
        memory_change_ratio = abs(recommendation.optimal_memory_size - recommendation.current_memory_size) / recommendation.current_memory_size
        if memory_change_ratio > 0.5:
            risk_assessment['level'] = 'medium'
            risk_assessment['factors'].append('Large memory configuration change')
            risk_assessment['mitigation'].append('Gradual rollout recommended')
        
        # High error rates indicate instability
        if features.error_rate > 0.05:
            risk_assessment['level'] = 'high'
            risk_assessment['factors'].append('High error rate detected')
            risk_assessment['mitigation'].append('Address errors before optimization')
        
        # Burst traffic patterns can be unpredictable
        if features.traffic_pattern == "burst" and features.concurrency_ratio > 0.7:
            if risk_assessment['level'] == 'low':
                risk_assessment['level'] = 'medium'
            risk_assessment['factors'].append('High concurrency burst pattern')
            risk_assessment['mitigation'].append('Monitor during peak hours')
        
        return risk_assessment
    
    def _predict_performance_outcomes(
        self,
        features: LearningFeatures,
        optimal_memory: int
    ) -> Dict[str, float]:
        """Predict performance outcomes using cost predictor."""
        try:
            # Use cost predictor for performance predictions
            predictions = self.cost_predictor.predict_costs_for_memory(
                optimal_memory,
                features.avg_duration,
                1000  # Sample invocations
            )
            
            return {
                'predicted_duration': features.avg_duration * 0.9,  # Assume 10% improvement
                'predicted_cost_per_invocation': predictions.get('cost_per_invocation', 0.0),
                'predicted_monthly_cost': predictions.get('monthly_cost', 0.0),
                'confidence': 0.7
            }
        except Exception as e:
            logger.warning(f"Failed to predict performance outcomes: {e}")
            return {}
    
    def _generate_optimization_timeline(
        self,
        features: LearningFeatures,
        recommendation: Recommendation
    ) -> Dict[str, Any]:
        """Generate optimization implementation timeline."""
        timeline = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        # Immediate actions (0-24 hours)
        timeline['immediate'].append("Update memory configuration")
        timeline['immediate'].append("Monitor initial performance")
        
        # Short-term actions (1-7 days)
        timeline['short_term'].append("Validate performance improvements")
        timeline['short_term'].append("Monitor error rates and cold starts")
        
        # Long-term actions (1-4 weeks)
        timeline['long_term'].append("Analyze cost savings")
        timeline['long_term'].append("Consider additional optimizations")
        
        # Add workload-specific timeline items
        if features.workload_type == "continuous":
            timeline['short_term'].append("Monitor sustained performance")
        
        if features.cold_start_sensitivity == "high":
            timeline['immediate'].append("Monitor cold start frequency")
        
        return timeline
    
    def _store_learning_data(self, features: LearningFeatures, recommendation: MLRecommendation):
        """Store learning data for future model improvements."""
        try:
            learning_record = {
                'timestamp': datetime.now().isoformat(),
                'function_arn': self.config.function_arn,
                'features': {
                    'workload_type': features.workload_type,
                    'traffic_pattern': features.traffic_pattern,
                    'cold_start_sensitivity': features.cold_start_sensitivity,
                    'avg_duration': features.avg_duration,
                    'memory_efficiency': features.memory_efficiency,
                    'cost_efficiency': features.cost_efficiency,
                    'error_rate': features.error_rate
                },
                'recommendation': {
                    'strategy': recommendation.base_recommendation.strategy,
                    'optimal_memory_size': recommendation.base_recommendation.optimal_memory_size,
                    'confidence_score': recommendation.confidence_score,
                    'pattern_match_score': recommendation.pattern_match_score
                }
            }
            
            # Store in historical data
            learning_file = self.data_dir / f"learning_data_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(learning_file, 'a') as f:
                f.write(json.dumps(learning_record) + '\n')
            
            # Update function profile
            self._update_function_profile(features)
            
        except Exception as e:
            logger.warning(f"Failed to store learning data: {e}")
    
    def _update_function_profile(self, features: LearningFeatures):
        """Update function profile with new data."""
        profile = {
            'workload_type': features.workload_type,
            'traffic_pattern': features.traffic_pattern,
            'cold_start_sensitivity': features.cold_start_sensitivity,
            'avg_duration': features.avg_duration,
            'memory_efficiency': features.memory_efficiency,
            'cost_efficiency': features.cost_efficiency,
            'last_updated': datetime.now().isoformat()
        }
        
        self.function_profiles[self.config.function_arn] = profile
        
        # Save to file
        profiles_file = self.data_dir / "function_profiles.json"
        with open(profiles_file, 'w') as f:
            json.dump(self.function_profiles, f, indent=2)
    
    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical learning data."""
        historical_data = []
        
        try:
            for learning_file in self.data_dir.glob("learning_data_*.jsonl"):
                with open(learning_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            historical_data.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
        
        return historical_data
    
    def _load_function_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load function profiles."""
        profiles_file = self.data_dir / "function_profiles.json"
        
        try:
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load function profiles: {e}")
        
        return {}
    
    def _initialize_performance_models(self) -> Dict[str, Any]:
        """Initialize performance prediction models."""
        # Placeholder for actual ML models
        # In a real implementation, this would load trained models
        return {
            'duration_model': None,
            'cost_model': None,
            'error_model': None
        }