"""
Cost predictor for modeling costs across different workload types.
Uses statistical models and machine learning to predict Lambda costs.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from enum import Enum

from ..models import MemoryTestResult
from ..config import TunerConfig

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Workload types for cost modeling."""
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"


class TrafficPattern(Enum):
    """Traffic patterns for cost modeling."""
    BURST = "burst"
    STEADY = "steady"
    VARIABLE = "variable"


@dataclass
class CostPrediction:
    """Cost prediction results."""
    memory_size: int
    predicted_cost_per_invocation: float
    predicted_monthly_cost: float
    confidence_interval: Tuple[float, float]
    cost_breakdown: Dict[str, float]
    savings_potential: float
    prediction_confidence: float


@dataclass
class WorkloadCostModel:
    """Cost model for specific workload characteristics."""
    workload_type: WorkloadType
    traffic_pattern: TrafficPattern
    base_cost_factor: float
    duration_sensitivity: float
    concurrency_factor: float
    cold_start_penalty: float
    memory_efficiency_curve: List[float]


class CostPredictor:
    """
    Advanced cost predictor that models Lambda costs across different
    workload types using historical data and performance patterns.
    """
    
    def __init__(self, config: TunerConfig, data_dir: Optional[str] = None):
        """
        Initialize the cost predictor.
        
        Args:
            config: Tuner configuration
            data_dir: Directory for storing cost models and data
        """
        self.config = config
        self.data_dir = Path(data_dir or "./cost_models")
        self.data_dir.mkdir(exist_ok=True)
        
        # AWS Lambda pricing (as of 2024)
        self.cost_per_gb_second = config.cost_per_gb_second
        self.cost_per_request = config.cost_per_request
        
        # Initialize cost models
        self.workload_models = self._initialize_workload_models()
        self.historical_costs = self._load_historical_costs()
        self.pricing_trends = self._load_pricing_trends()
        
        logger.info("Cost predictor initialized with workload-specific models")
    
    def predict_costs_for_memory(
        self,
        memory_size: int,
        avg_duration: float,
        monthly_invocations: int,
        workload_characteristics: Optional[Dict[str, Any]] = None
    ) -> CostPrediction:
        """
        Predict costs for a specific memory configuration.
        
        Args:
            memory_size: Memory size in MB
            avg_duration: Average duration in seconds
            monthly_invocations: Expected monthly invocations
            workload_characteristics: Additional workload info
            
        Returns:
            Detailed cost prediction
        """
        logger.debug(f"Predicting costs for {memory_size}MB memory configuration")
        
        try:
            # Get workload model
            workload_model = self._get_workload_model()
            
            # Calculate base costs
            base_costs = self._calculate_base_costs(
                memory_size, avg_duration, monthly_invocations
            )
            
            # Apply workload-specific adjustments
            adjusted_costs = self._apply_workload_adjustments(
                base_costs, workload_model, workload_characteristics or {}
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                adjusted_costs['total_cost'], memory_size, avg_duration
            )
            
            # Calculate savings potential
            savings_potential = self._calculate_savings_potential(
                memory_size, adjusted_costs['total_cost']
            )
            
            # Determine prediction confidence
            prediction_confidence = self._calculate_prediction_confidence(
                memory_size, avg_duration, monthly_invocations
            )
            
            prediction = CostPrediction(
                memory_size=memory_size,
                predicted_cost_per_invocation=adjusted_costs['cost_per_invocation'],
                predicted_monthly_cost=adjusted_costs['total_cost'],
                confidence_interval=confidence_interval,
                cost_breakdown=adjusted_costs['breakdown'],
                savings_potential=savings_potential,
                prediction_confidence=prediction_confidence
            )
            
            # Store prediction for learning
            self._store_prediction(prediction, workload_characteristics)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to predict costs: {e}")
            # Return basic prediction as fallback
            basic_cost = self._calculate_basic_cost(memory_size, avg_duration, monthly_invocations)
            return CostPrediction(
                memory_size=memory_size,
                predicted_cost_per_invocation=basic_cost / monthly_invocations if monthly_invocations > 0 else 0,
                predicted_monthly_cost=basic_cost,
                confidence_interval=(basic_cost * 0.8, basic_cost * 1.2),
                cost_breakdown={'compute': basic_cost, 'requests': 0},
                savings_potential=0.0,
                prediction_confidence=0.5
            )
    
    def predict_costs_across_memories(
        self,
        memory_sizes: List[int],
        performance_data: Dict[int, MemoryTestResult],
        monthly_invocations: int
    ) -> Dict[int, CostPrediction]:
        """
        Predict costs across multiple memory configurations.
        
        Args:
            memory_sizes: List of memory sizes to evaluate
            performance_data: Performance test results
            monthly_invocations: Expected monthly invocations
            
        Returns:
            Cost predictions for each memory size
        """
        logger.info(f"Predicting costs across {len(memory_sizes)} memory configurations")
        
        predictions = {}
        
        for memory_size in memory_sizes:
            if memory_size in performance_data:
                result = performance_data[memory_size]
                prediction = self.predict_costs_for_memory(
                    memory_size,
                    result.avg_duration,
                    monthly_invocations,
                    self._extract_workload_characteristics(result)
                )
                predictions[memory_size] = prediction
            else:
                # Estimate duration for missing data points
                estimated_duration = self._estimate_duration_for_memory(
                    memory_size, performance_data
                )
                prediction = self.predict_costs_for_memory(
                    memory_size,
                    estimated_duration,
                    monthly_invocations
                )
                predictions[memory_size] = prediction
        
        return predictions
    
    def find_cost_optimal_configuration(
        self,
        cost_predictions: Dict[int, CostPrediction],
        performance_constraints: Optional[Dict[str, float]] = None
    ) -> Tuple[int, CostPrediction]:
        """
        Find the most cost-optimal configuration considering constraints.
        
        Args:
            cost_predictions: Cost predictions for different memory sizes
            performance_constraints: Optional performance constraints
            
        Returns:
            Optimal memory size and its cost prediction
        """
        logger.debug("Finding cost-optimal configuration")
        
        constraints = performance_constraints or {}
        max_duration = constraints.get('max_duration')
        min_reliability = constraints.get('min_reliability', 0.95)
        
        # Filter configurations that meet constraints
        valid_configs = {}
        for memory_size, prediction in cost_predictions.items():
            # Check duration constraint
            if max_duration and prediction.predicted_cost_per_invocation > max_duration:
                continue
            
            # Check reliability (placeholder - would use actual reliability data)
            reliability = max(0.99 - prediction.memory_size / 10000, 0.95)
            if reliability < min_reliability:
                continue
            
            valid_configs[memory_size] = prediction
        
        if not valid_configs:
            logger.warning("No configurations meet performance constraints")
            valid_configs = cost_predictions
        
        # Find minimum cost configuration
        optimal_memory = min(
            valid_configs.keys(),
            key=lambda m: valid_configs[m].predicted_monthly_cost
        )
        
        return optimal_memory, valid_configs[optimal_memory]
    
    def analyze_cost_trends(
        self,
        cost_predictions: Dict[int, CostPrediction]
    ) -> Dict[str, Any]:
        """
        Analyze cost trends across memory configurations.
        
        Args:
            cost_predictions: Cost predictions for different memory sizes
            
        Returns:
            Cost trend analysis
        """
        if len(cost_predictions) < 2:
            return {'trend': 'insufficient_data'}
        
        memory_sizes = sorted(cost_predictions.keys())
        costs = [cost_predictions[m].predicted_monthly_cost for m in memory_sizes]
        
        # Calculate cost efficiency (cost per MB)
        cost_per_mb = [costs[i] / memory_sizes[i] for i in range(len(costs))]
        
        # Find inflection points
        cost_derivatives = np.diff(costs)
        efficiency_derivatives = np.diff(cost_per_mb)
        
        # Identify sweet spots
        sweet_spots = []
        for i in range(1, len(cost_derivatives)):
            if cost_derivatives[i-1] < 0 and cost_derivatives[i] > 0:
                sweet_spots.append(memory_sizes[i])
        
        # Calculate marginal cost effectiveness
        marginal_effectiveness = {}
        for i in range(1, len(memory_sizes)):
            prev_memory = memory_sizes[i-1]
            curr_memory = memory_sizes[i]
            
            memory_increase = curr_memory - prev_memory
            cost_increase = costs[i] - costs[i-1]
            
            if memory_increase > 0:
                marginal_effectiveness[curr_memory] = cost_increase / memory_increase
        
        return {
            'trend': 'analyzed',
            'optimal_range': {
                'min_memory': memory_sizes[np.argmin(costs)],
                'min_cost': min(costs)
            },
            'sweet_spots': sweet_spots,
            'marginal_effectiveness': marginal_effectiveness,
            'cost_efficiency_trend': 'decreasing' if efficiency_derivatives[-1] < 0 else 'increasing',
            'recommendations': self._generate_cost_trend_recommendations(
                memory_sizes, costs, sweet_spots
            )
        }
    
    def _initialize_workload_models(self) -> Dict[str, WorkloadCostModel]:
        """Initialize workload-specific cost models."""
        models = {}
        
        # On-demand workload model
        models['on_demand'] = WorkloadCostModel(
            workload_type=WorkloadType.ON_DEMAND,
            traffic_pattern=TrafficPattern.BURST,
            base_cost_factor=1.0,
            duration_sensitivity=0.8,
            concurrency_factor=1.2,
            cold_start_penalty=0.15,
            memory_efficiency_curve=[0.6, 0.75, 0.85, 0.92, 0.96, 0.98, 0.99]
        )
        
        # Scheduled workload model
        models['scheduled'] = WorkloadCostModel(
            workload_type=WorkloadType.SCHEDULED,
            traffic_pattern=TrafficPattern.STEADY,
            base_cost_factor=0.9,
            duration_sensitivity=0.9,
            concurrency_factor=1.0,
            cold_start_penalty=0.05,
            memory_efficiency_curve=[0.7, 0.8, 0.88, 0.94, 0.97, 0.99, 1.0]
        )
        
        # Continuous workload model
        models['continuous'] = WorkloadCostModel(
            workload_type=WorkloadType.CONTINUOUS,
            traffic_pattern=TrafficPattern.VARIABLE,
            base_cost_factor=0.85,
            duration_sensitivity=1.0,
            concurrency_factor=0.9,
            cold_start_penalty=0.02,
            memory_efficiency_curve=[0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
        )
        
        return models
    
    def _get_workload_model(self) -> WorkloadCostModel:
        """Get the appropriate workload model for current configuration."""
        workload_type = self.config.workload_type
        return self.workload_models.get(workload_type, self.workload_models['on_demand'])
    
    def _calculate_base_costs(
        self,
        memory_size: int,
        avg_duration: float,
        monthly_invocations: int
    ) -> Dict[str, float]:
        """Calculate base AWS Lambda costs."""
        # Convert memory to GB
        memory_gb = memory_size / 1024.0
        
        # Compute cost (memory * duration * price per GB-second)
        compute_cost_per_invocation = memory_gb * avg_duration * self.cost_per_gb_second
        
        # Request cost
        request_cost_per_invocation = self.cost_per_request
        
        # Total per invocation
        total_cost_per_invocation = compute_cost_per_invocation + request_cost_per_invocation
        
        # Monthly costs
        monthly_compute_cost = compute_cost_per_invocation * monthly_invocations
        monthly_request_cost = request_cost_per_invocation * monthly_invocations
        monthly_total_cost = monthly_compute_cost + monthly_request_cost
        
        return {
            'cost_per_invocation': total_cost_per_invocation,
            'total_cost': monthly_total_cost,
            'breakdown': {
                'compute': monthly_compute_cost,
                'requests': monthly_request_cost
            }
        }
    
    def _apply_workload_adjustments(
        self,
        base_costs: Dict[str, float],
        workload_model: WorkloadCostModel,
        workload_characteristics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply workload-specific cost adjustments."""
        adjusted_costs = base_costs.copy()
        
        # Apply base cost factor
        adjusted_costs['total_cost'] *= workload_model.base_cost_factor
        adjusted_costs['cost_per_invocation'] *= workload_model.base_cost_factor
        
        # Apply cold start penalty
        cold_start_ratio = workload_characteristics.get('cold_start_ratio', 0.1)
        cold_start_adjustment = 1.0 + (cold_start_ratio * workload_model.cold_start_penalty)
        adjusted_costs['total_cost'] *= cold_start_adjustment
        adjusted_costs['cost_per_invocation'] *= cold_start_adjustment
        
        # Apply concurrency factor
        concurrency_ratio = workload_characteristics.get('concurrency_ratio', 0.5)
        if concurrency_ratio > 0.7:  # High concurrency
            concurrency_adjustment = workload_model.concurrency_factor
            adjusted_costs['total_cost'] *= concurrency_adjustment
            adjusted_costs['cost_per_invocation'] *= concurrency_adjustment
        
        # Update breakdown
        total_adjustment = adjusted_costs['total_cost'] / base_costs['total_cost']
        for category in adjusted_costs['breakdown']:
            adjusted_costs['breakdown'][category] *= total_adjustment
        
        return adjusted_costs
    
    def _calculate_confidence_interval(
        self,
        predicted_cost: float,
        memory_size: int,
        avg_duration: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for cost prediction."""
        # Base confidence interval (±10%)
        base_uncertainty = 0.10
        
        # Increase uncertainty for extreme configurations
        if memory_size < 256 or memory_size > 3008:
            base_uncertainty += 0.05
        
        # Increase uncertainty for very short or long durations
        if avg_duration < 0.1 or avg_duration > 10.0:
            base_uncertainty += 0.05
        
        # Historical variance (placeholder - would use actual data)
        historical_variance = 0.08
        
        total_uncertainty = min(base_uncertainty + historical_variance, 0.25)
        
        lower_bound = predicted_cost * (1 - total_uncertainty)
        upper_bound = predicted_cost * (1 + total_uncertainty)
        
        return (lower_bound, upper_bound)
    
    def _calculate_savings_potential(self, memory_size: int, predicted_cost: float) -> float:
        """Calculate potential savings compared to default configuration."""
        # Default to 1024MB as baseline
        baseline_memory = 1024
        
        if memory_size == baseline_memory:
            return 0.0
        
        # Estimate baseline cost (simplified)
        baseline_cost = predicted_cost * (baseline_memory / memory_size) * 1.1  # Assume 10% efficiency loss
        
        savings = baseline_cost - predicted_cost
        savings_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        
        return max(savings_percentage, 0.0)
    
    def _calculate_prediction_confidence(
        self,
        memory_size: int,
        avg_duration: float,
        monthly_invocations: int
    ) -> float:
        """Calculate confidence in the prediction."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on data availability
        common_memory_sizes = [128, 256, 512, 1024, 1536, 2048, 3008]
        if memory_size in common_memory_sizes:
            confidence += 0.1
        
        # Adjust based on duration range
        if 0.1 <= avg_duration <= 5.0:  # Typical range
            confidence += 0.05
        
        # Adjust based on invocation volume
        if monthly_invocations >= 1000:  # Sufficient data
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _calculate_basic_cost(
        self,
        memory_size: int,
        avg_duration: float,
        monthly_invocations: int
    ) -> float:
        """Calculate basic cost without adjustments (fallback)."""
        memory_gb = memory_size / 1024.0
        compute_cost = memory_gb * avg_duration * self.cost_per_gb_second * monthly_invocations
        request_cost = self.cost_per_request * monthly_invocations
        return compute_cost + request_cost
    
    def _extract_workload_characteristics(self, result: MemoryTestResult) -> Dict[str, Any]:
        """Extract workload characteristics from test result."""
        total_executions = result.iterations
        return {
            'cold_start_ratio': result.cold_starts / total_executions if total_executions > 0 else 0.0,
            'error_ratio': result.errors / total_executions if total_executions > 0 else 0.0,
            'duration_variance': (result.p95_duration - result.avg_duration) / result.avg_duration if result.avg_duration > 0 else 0.0,
            'concurrency_ratio': min(self.config.expected_concurrency / 100.0, 1.0)
        }
    
    def _estimate_duration_for_memory(
        self,
        memory_size: int,
        performance_data: Dict[int, MemoryTestResult]
    ) -> float:
        """Estimate duration for a memory size using interpolation."""
        if not performance_data:
            return 1.0  # Default duration
        
        memory_sizes = sorted(performance_data.keys())
        durations = [performance_data[m].avg_duration for m in memory_sizes]
        
        # Simple linear interpolation
        if memory_size <= memory_sizes[0]:
            return durations[0]
        elif memory_size >= memory_sizes[-1]:
            return durations[-1]
        else:
            # Find surrounding points
            for i in range(len(memory_sizes) - 1):
                if memory_sizes[i] <= memory_size <= memory_sizes[i + 1]:
                    # Linear interpolation
                    x1, x2 = memory_sizes[i], memory_sizes[i + 1]
                    y1, y2 = durations[i], durations[i + 1]
                    return y1 + (y2 - y1) * (memory_size - x1) / (x2 - x1)
        
        return np.mean(durations)  # Fallback to average
    
    def _generate_cost_trend_recommendations(
        self,
        memory_sizes: List[int],
        costs: List[float],
        sweet_spots: List[int]
    ) -> List[str]:
        """Generate recommendations based on cost trends."""
        recommendations = []
        
        if sweet_spots:
            recommendations.append(f"Consider memory sizes around {sweet_spots[0]}MB for optimal cost efficiency")
        
        # Find diminishing returns point
        cost_reductions = np.diff(costs)
        if len(cost_reductions) > 1:
            diminishing_point = None
            for i in range(1, len(cost_reductions)):
                if abs(cost_reductions[i]) < abs(cost_reductions[i-1]) * 0.5:
                    diminishing_point = memory_sizes[i+1]
                    break
            
            if diminishing_point:
                recommendations.append(f"Diminishing returns observed beyond {diminishing_point}MB")
        
        # Cost stability recommendation
        cost_variance = np.var(costs)
        mean_cost = np.mean(costs)
        if cost_variance / mean_cost < 0.1:  # Low variance
            recommendations.append("Costs are relatively stable across memory configurations")
        
        return recommendations
    
    def _store_prediction(self, prediction: CostPrediction, characteristics: Optional[Dict[str, Any]]):
        """Store cost prediction for model learning."""
        try:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'function_arn': self.config.function_arn,
                'prediction': {
                    'memory_size': prediction.memory_size,
                    'predicted_cost': prediction.predicted_monthly_cost,
                    'confidence': prediction.prediction_confidence
                },
                'characteristics': characteristics or {}
            }
            
            predictions_file = self.data_dir / f"cost_predictions_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(predictions_file, 'a') as f:
                f.write(json.dumps(prediction_record) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to store cost prediction: {e}")
    
    def _load_historical_costs(self) -> List[Dict[str, Any]]:
        """Load historical cost data."""
        historical_costs = []
        
        try:
            for cost_file in self.data_dir.glob("cost_predictions_*.jsonl"):
                with open(cost_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            historical_costs.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load historical costs: {e}")
        
        return historical_costs
    
    def _load_pricing_trends(self) -> Dict[str, Any]:
        """Load AWS pricing trends (placeholder)."""
        # In a real implementation, this would load actual pricing trend data
        return {
            'trend': 'stable',
            'last_updated': datetime.now().isoformat(),
            'regional_variations': {}
        }