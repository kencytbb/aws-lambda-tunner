"""
Continuous optimizer for ongoing Lambda optimization.
Monitors performance and automatically triggers re-optimization when needed.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from ..config import TunerConfig
from ..models import Recommendation
from ..intelligence.recommendation_engine import IntelligentRecommendationEngine
from ..intelligence.pattern_recognizer import PatternRecognizer
from .performance_monitor import PerformanceMonitor
from .alert_manager import AlertManager

logger = logging.getLogger(__name__)


class OptimizationTrigger(Enum):
    """Types of triggers for re-optimization."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COST_THRESHOLD = "cost_threshold"
    ERROR_RATE_INCREASE = "error_rate_increase"
    TRAFFIC_PATTERN_CHANGE = "traffic_pattern_change"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class OptimizationEvent:
    """Represents an optimization event."""
    trigger: OptimizationTrigger
    timestamp: datetime
    function_arn: str
    trigger_data: Dict[str, Any]
    severity: str = "medium"  # low, medium, high, critical
    auto_approve: bool = False


@dataclass
class OptimizationResult:
    """Results from an optimization run."""
    event: OptimizationEvent
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    previous_config: Dict[str, Any] = field(default_factory=dict)
    new_config: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[Recommendation] = None
    performance_impact: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class ContinuousOptimizer:
    """
    Continuous optimization engine that monitors Lambda performance
    and automatically triggers re-optimization when needed.
    """
    
    def __init__(
        self,
        config: TunerConfig,
        performance_monitor: PerformanceMonitor,
        alert_manager: AlertManager,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the continuous optimizer.
        
        Args:
            config: Tuner configuration
            performance_monitor: Performance monitoring component
            alert_manager: Alert management component
            data_dir: Directory for storing optimization data
        """
        self.config = config
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        self.data_dir = Path(data_dir or "./optimization_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # AWS clients
        self.lambda_client = self._create_lambda_client()
        self.cloudwatch_client = self._create_cloudwatch_client()
        
        # Optimization components
        self.recommendation_engine = IntelligentRecommendationEngine(config, str(self.data_dir))
        self.pattern_recognizer = PatternRecognizer(config, str(self.data_dir))
        
        # State management
        self.is_running = False
        self.optimization_history: List[OptimizationResult] = []
        self.pending_optimizations: List[OptimizationEvent] = []
        
        # Configuration
        self.optimization_cooldown = timedelta(hours=6)  # Minimum time between optimizations
        self.max_pending_optimizations = 10
        
        # Callbacks
        self.optimization_callbacks: List[Callable[[OptimizationResult], None]] = []
        
        logger.info("Continuous optimizer initialized")
    
    async def start_continuous_optimization(self):
        """Start the continuous optimization process."""
        if self.is_running:
            logger.warning("Continuous optimization is already running")
            return
        
        logger.info("Starting continuous optimization")
        self.is_running = True
        
        try:
            # Start monitoring components
            await self.performance_monitor.start_monitoring()
            
            # Main optimization loop
            await self._optimization_loop()
            
        except Exception as e:
            logger.error(f"Error in continuous optimization: {e}")
            self.is_running = False
            raise
    
    async def stop_continuous_optimization(self):
        """Stop the continuous optimization process."""
        logger.info("Stopping continuous optimization")
        self.is_running = False
        
        # Stop monitoring components
        await self.performance_monitor.stop_monitoring()
        
        # Save state
        await self._save_optimization_state()
    
    async def trigger_optimization(
        self,
        trigger: OptimizationTrigger,
        trigger_data: Dict[str, Any],
        severity: str = "medium",
        auto_approve: bool = False
    ) -> OptimizationEvent:
        """
        Trigger an optimization event.
        
        Args:
            trigger: Type of trigger
            trigger_data: Additional trigger data
            severity: Severity level
            auto_approve: Whether to auto-approve the optimization
            
        Returns:
            Created optimization event
        """
        event = OptimizationEvent(
            trigger=trigger,
            timestamp=datetime.now(),
            function_arn=self.config.function_arn,
            trigger_data=trigger_data,
            severity=severity,
            auto_approve=auto_approve
        )
        
        logger.info(f"Optimization triggered: {trigger.value} (severity: {severity})")
        
        # Add to pending optimizations
        if len(self.pending_optimizations) < self.max_pending_optimizations:
            self.pending_optimizations.append(event)
        else:
            logger.warning("Maximum pending optimizations reached, dropping oldest")
            self.pending_optimizations.pop(0)
            self.pending_optimizations.append(event)
        
        # Send alert
        await self.alert_manager.send_optimization_alert(event)
        
        return event
    
    async def approve_optimization(self, event: OptimizationEvent) -> bool:
        """
        Approve a pending optimization.
        
        Args:
            event: Optimization event to approve
            
        Returns:
            Success status
        """
        if event not in self.pending_optimizations:
            logger.warning("Optimization event not found in pending list")
            return False
        
        logger.info(f"Approving optimization: {event.trigger.value}")
        
        # Execute optimization
        result = await self._execute_optimization(event)
        
        # Remove from pending
        self.pending_optimizations.remove(event)
        
        # Store result
        self.optimization_history.append(result)
        
        # Notify callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Optimization callback failed: {e}")
        
        return result.success
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get current optimization status.
        
        Returns:
            Status information
        """
        recent_optimizations = [
            result for result in self.optimization_history
            if result.started_at > datetime.now() - timedelta(days=7)
        ]
        
        status = {
            'is_running': self.is_running,
            'pending_optimizations': len(self.pending_optimizations),
            'recent_optimizations': len(recent_optimizations),
            'last_optimization': None,
            'next_scheduled_check': None,
            'performance_status': await self.performance_monitor.get_current_status()
        }
        
        if self.optimization_history:
            last_opt = self.optimization_history[-1]
            status['last_optimization'] = {
                'timestamp': last_opt.started_at.isoformat(),
                'trigger': last_opt.event.trigger.value,
                'success': last_opt.success,
                'performance_impact': last_opt.performance_impact
            }
        
        return status
    
    def add_optimization_callback(self, callback: Callable[[OptimizationResult], None]):
        """Add a callback to be called after optimizations."""
        self.optimization_callbacks.append(callback)
    
    async def _optimization_loop(self):
        """Main optimization monitoring loop."""
        logger.info("Starting optimization monitoring loop")
        
        while self.is_running:
            try:
                # Check for performance degradation
                await self._check_performance_degradation()
                
                # Check for cost threshold breaches
                await self._check_cost_thresholds()
                
                # Check for error rate increases
                await self._check_error_rate_increases()
                
                # Check for traffic pattern changes
                await self._check_traffic_pattern_changes()
                
                # Process pending optimizations
                await self._process_pending_optimizations()
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _check_performance_degradation(self):
        """Check for performance degradation that triggers optimization."""
        try:
            current_metrics = await self.performance_monitor.get_current_metrics()
            baseline_metrics = await self.performance_monitor.get_baseline_metrics()
            
            if not baseline_metrics:
                return
            
            # Check duration degradation
            current_duration = current_metrics.get('avg_duration', 0)
            baseline_duration = baseline_metrics.get('avg_duration', 0)
            
            if baseline_duration > 0:
                duration_increase = (current_duration - baseline_duration) / baseline_duration
                
                if duration_increase > 0.2:  # 20% degradation
                    await self.trigger_optimization(
                        OptimizationTrigger.PERFORMANCE_DEGRADATION,
                        {
                            'metric': 'duration',
                            'current_value': current_duration,
                            'baseline_value': baseline_duration,
                            'degradation_percent': duration_increase * 100
                        },
                        severity="high",
                        auto_approve=False
                    )
            
            # Check error rate increase
            current_error_rate = current_metrics.get('error_rate', 0)
            baseline_error_rate = baseline_metrics.get('error_rate', 0)
            
            if current_error_rate > baseline_error_rate + 0.05:  # 5% absolute increase
                await self.trigger_optimization(
                    OptimizationTrigger.ERROR_RATE_INCREASE,
                    {
                        'current_error_rate': current_error_rate,
                        'baseline_error_rate': baseline_error_rate,
                        'increase': current_error_rate - baseline_error_rate
                    },
                    severity="critical",
                    auto_approve=False
                )
                
        except Exception as e:
            logger.warning(f"Failed to check performance degradation: {e}")
    
    async def _check_cost_thresholds(self):
        """Check for cost threshold breaches."""
        try:
            # Get current cost metrics
            cost_metrics = await self.performance_monitor.get_cost_metrics()
            
            if not cost_metrics:
                return
            
            monthly_cost = cost_metrics.get('monthly_cost', 0)
            cost_threshold = getattr(self.config, 'cost_threshold', None)
            
            if cost_threshold and monthly_cost > cost_threshold:
                await self.trigger_optimization(
                    OptimizationTrigger.COST_THRESHOLD,
                    {
                        'monthly_cost': monthly_cost,
                        'threshold': cost_threshold,
                        'overage': monthly_cost - cost_threshold
                    },
                    severity="medium",
                    auto_approve=True  # Auto-approve cost optimizations
                )
                
        except Exception as e:
            logger.warning(f"Failed to check cost thresholds: {e}")
    
    async def _check_error_rate_increases(self):
        """Check for sudden error rate increases."""
        try:
            error_trends = await self.performance_monitor.get_error_trends()
            
            if not error_trends:
                return
            
            recent_error_rate = error_trends.get('recent_error_rate', 0)
            historical_error_rate = error_trends.get('historical_error_rate', 0)
            
            # Trigger if error rate doubles or exceeds 5%
            if (recent_error_rate > historical_error_rate * 2) or (recent_error_rate > 0.05):
                await self.trigger_optimization(
                    OptimizationTrigger.ERROR_RATE_INCREASE,
                    {
                        'recent_error_rate': recent_error_rate,
                        'historical_error_rate': historical_error_rate,
                        'trend': error_trends.get('trend', 'unknown')
                    },
                    severity="high",
                    auto_approve=False
                )
                
        except Exception as e:
            logger.warning(f"Failed to check error rate increases: {e}")
    
    async def _check_traffic_pattern_changes(self):
        """Check for significant traffic pattern changes."""
        try:
            traffic_analysis = await self.performance_monitor.analyze_traffic_patterns()
            
            if not traffic_analysis:
                return
            
            pattern_change_score = traffic_analysis.get('pattern_change_score', 0)
            
            if pattern_change_score > 0.7:  # Significant pattern change
                await self.trigger_optimization(
                    OptimizationTrigger.TRAFFIC_PATTERN_CHANGE,
                    {
                        'pattern_change_score': pattern_change_score,
                        'new_pattern': traffic_analysis.get('detected_pattern'),
                        'previous_pattern': traffic_analysis.get('previous_pattern')
                    },
                    severity="medium",
                    auto_approve=True
                )
                
        except Exception as e:
            logger.warning(f"Failed to check traffic pattern changes: {e}")
    
    async def _process_pending_optimizations(self):
        """Process pending optimizations that are auto-approved."""
        auto_approved = [
            event for event in self.pending_optimizations
            if event.auto_approve and self._can_execute_optimization()
        ]
        
        for event in auto_approved:
            logger.info(f"Auto-executing optimization: {event.trigger.value}")
            await self.approve_optimization(event)
    
    async def _execute_optimization(self, event: OptimizationEvent) -> OptimizationResult:
        """
        Execute an optimization event.
        
        Args:
            event: Optimization event to execute
            
        Returns:
            Optimization result
        """
        result = OptimizationResult(
            event=event,
            started_at=datetime.now()
        )
        
        try:
            logger.info(f"Executing optimization for trigger: {event.trigger.value}")
            
            # Get current configuration
            current_config = await self._get_current_function_config()
            result.previous_config = current_config
            
            # Collect recent performance data
            performance_data = await self.performance_monitor.get_recent_performance_data()
            
            # Generate recommendations using intelligence engine
            if performance_data:
                recommendation = await self._generate_intelligent_recommendation(performance_data)
                result.recommendation = recommendation
                
                if recommendation and recommendation.should_optimize:
                    # Apply optimization
                    success = await self._apply_optimization(recommendation)
                    result.success = success
                    
                    if success:
                        result.new_config = await self._get_current_function_config()
                        
                        # Monitor impact
                        impact = await self._measure_optimization_impact(
                            result.previous_config, result.new_config
                        )
                        result.performance_impact = impact
                        
                        logger.info(f"Optimization completed successfully: {recommendation.optimal_memory_size}MB")
                    else:
                        result.error_message = "Failed to apply optimization"
                else:
                    result.success = True  # No optimization needed
                    result.error_message = "No optimization needed"
            else:
                result.error_message = "Insufficient performance data"
                
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
            result.error_message = str(e)
            result.success = False
        
        result.completed_at = datetime.now()
        
        # Send completion alert
        await self.alert_manager.send_optimization_completion_alert(result)
        
        # Store result
        await self._store_optimization_result(result)
        
        return result
    
    async def _generate_intelligent_recommendation(
        self,
        performance_data: Dict[str, Any]
    ) -> Optional[Recommendation]:
        """Generate intelligent recommendation from performance data."""
        try:
            # Convert performance data to expected format
            memory_results = performance_data.get('memory_results', {})
            analysis = performance_data.get('analysis')
            
            if not memory_results or not analysis:
                return None
            
            # Use recommendation engine
            ml_recommendation = self.recommendation_engine.generate_intelligent_recommendation(
                analysis, memory_results
            )
            
            return ml_recommendation.base_recommendation
            
        except Exception as e:
            logger.warning(f"Failed to generate intelligent recommendation: {e}")
            return None
    
    async def _apply_optimization(self, recommendation: Recommendation) -> bool:
        """
        Apply optimization recommendation.
        
        Args:
            recommendation: Optimization recommendation
            
        Returns:
            Success status
        """
        try:
            # Update Lambda function configuration
            await self._update_function_memory(recommendation.optimal_memory_size)
            
            # Wait for configuration to propagate
            await asyncio.sleep(10)
            
            # Verify configuration was applied
            current_config = await self._get_current_function_config()
            current_memory = current_config.get('MemorySize', 0)
            
            return current_memory == recommendation.optimal_memory_size
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
            return False
    
    async def _measure_optimization_impact(
        self,
        previous_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Measure the impact of optimization.
        
        Args:
            previous_config: Previous function configuration
            new_config: New function configuration
            
        Returns:
            Performance impact metrics
        """
        # Wait for some execution data
        await asyncio.sleep(30)
        
        try:
            # Get metrics before and after optimization
            impact_metrics = await self.performance_monitor.measure_configuration_impact(
                previous_config, new_config
            )
            
            return impact_metrics
            
        except Exception as e:
            logger.warning(f"Failed to measure optimization impact: {e}")
            return {}
    
    def _can_execute_optimization(self) -> bool:
        """Check if optimization can be executed (cooldown, etc.)."""
        if not self.optimization_history:
            return True
        
        last_optimization = self.optimization_history[-1]
        time_since_last = datetime.now() - last_optimization.started_at
        
        return time_since_last >= self.optimization_cooldown
    
    async def _get_current_function_config(self) -> Dict[str, Any]:
        """Get current Lambda function configuration."""
        try:
            response = self.lambda_client.get_function_configuration(
                FunctionName=self.config.function_arn
            )
            
            return {
                'MemorySize': response['MemorySize'],
                'Timeout': response['Timeout'],
                'Runtime': response['Runtime'],
                'LastModified': response['LastModified']
            }
            
        except ClientError as e:
            logger.error(f"Failed to get function configuration: {e}")
            return {}
    
    async def _update_function_memory(self, memory_size: int):
        """Update Lambda function memory size."""
        try:
            self.lambda_client.update_function_configuration(
                FunctionName=self.config.function_arn,
                MemorySize=memory_size
            )
            
            logger.info(f"Updated function memory to {memory_size}MB")
            
        except ClientError as e:
            logger.error(f"Failed to update function memory: {e}")
            raise
    
    async def _store_optimization_result(self, result: OptimizationResult):
        """Store optimization result for analysis."""
        try:
            result_record = {
                'timestamp': result.started_at.isoformat(),
                'function_arn': self.config.function_arn,
                'trigger': result.event.trigger.value,
                'success': result.success,
                'previous_memory': result.previous_config.get('MemorySize'),
                'new_memory': result.new_config.get('MemorySize'),
                'performance_impact': result.performance_impact,
                'error_message': result.error_message
            }
            
            results_file = self.data_dir / f"optimization_results_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(results_file, 'a') as f:
                f.write(json.dumps(result_record) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to store optimization result: {e}")
    
    async def _save_optimization_state(self):
        """Save current optimization state."""
        try:
            state = {
                'pending_optimizations': [
                    {
                        'trigger': event.trigger.value,
                        'timestamp': event.timestamp.isoformat(),
                        'trigger_data': event.trigger_data,
                        'severity': event.severity,
                        'auto_approve': event.auto_approve
                    }
                    for event in self.pending_optimizations
                ],
                'last_saved': datetime.now().isoformat()
            }
            
            state_file = self.data_dir / "optimization_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save optimization state: {e}")
    
    def _create_lambda_client(self):
        """Create AWS Lambda client."""
        try:
            return boto3.client('lambda', region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create Lambda client: {e}")
            raise
    
    def _create_cloudwatch_client(self):
        """Create AWS CloudWatch client."""
        try:
            return boto3.client('cloudwatch', region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create CloudWatch client: {e}")
            raise