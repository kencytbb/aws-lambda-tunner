"""
Performance monitor for CloudWatch integration and real-time monitoring.
Provides continuous monitoring of Lambda performance metrics.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import numpy as np

from ..config import TunerConfig

logger = logging.getLogger(__name__)


@dataclass
class MetricDataPoint:
    """CloudWatch metric data point."""
    timestamp: datetime
    value: float
    unit: str
    metric_name: str


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    function_arn: str
    memory_size: int
    avg_duration: float
    p95_duration: float
    p99_duration: float
    error_rate: float
    throttle_rate: float
    cold_start_rate: float
    concurrent_executions: int
    invocation_count: int
    cost_per_invocation: float


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric_name: str
    time_period: str
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0-1, strength of trend
    data_points: List[MetricDataPoint]
    statistical_summary: Dict[str, float]


class PerformanceMonitor:
    """
    Real-time performance monitor that integrates with CloudWatch
    to track Lambda function performance continuously.
    """
    
    def __init__(self, config: TunerConfig, data_dir: Optional[str] = None):
        """
        Initialize the performance monitor.
        
        Args:
            config: Tuner configuration
            data_dir: Directory for storing monitoring data
        """
        self.config = config
        self.data_dir = Path(data_dir or "./monitoring_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # AWS clients
        self.cloudwatch = self._create_cloudwatch_client()
        self.lambda_client = self._create_lambda_client()
        self.logs_client = self._create_logs_client()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 60  # seconds
        self.baseline_metrics: Optional[Dict[str, float]] = None
        
        # Performance history
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.max_snapshots = 1000
        
        # Metric configuration
        self.monitored_metrics = [
            'Duration',
            'Errors',
            'Throttles',
            'Invocations',
            'ConcurrentExecutions',
            'DeadLetterErrors'
        ]
        
        logger.info("Performance monitor initialized")
    
    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        logger.info("Starting performance monitoring")
        self.is_monitoring = True
        
        # Load historical data
        await self._load_historical_data()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        logger.info("Stopping performance monitoring")
        self.is_monitoring = False
        
        # Save current data
        await self._save_monitoring_data()
    
    async def get_current_metrics(self, time_window: timedelta = timedelta(minutes=15)) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Args:
            time_window: Time window for metrics collection
            
        Returns:
            Current performance metrics
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            metrics = {}
            
            for metric_name in self.monitored_metrics:
                metric_data = await self._get_cloudwatch_metric(
                    metric_name, start_time, end_time
                )
                
                if metric_data:
                    if metric_name == 'Duration':
                        metrics['avg_duration'] = np.mean([dp.value for dp in metric_data]) / 1000.0  # Convert to seconds
                        metrics['p95_duration'] = np.percentile([dp.value for dp in metric_data], 95) / 1000.0
                        metrics['p99_duration'] = np.percentile([dp.value for dp in metric_data], 99) / 1000.0
                    elif metric_name == 'Errors':
                        error_count = sum(dp.value for dp in metric_data)
                        invocation_data = await self._get_cloudwatch_metric('Invocations', start_time, end_time)
                        total_invocations = sum(dp.value for dp in invocation_data) if invocation_data else 1
                        metrics['error_rate'] = error_count / total_invocations if total_invocations > 0 else 0
                    elif metric_name == 'Throttles':
                        throttle_count = sum(dp.value for dp in metric_data)
                        invocation_data = await self._get_cloudwatch_metric('Invocations', start_time, end_time)
                        total_invocations = sum(dp.value for dp in invocation_data) if invocation_data else 1
                        metrics['throttle_rate'] = throttle_count / total_invocations if total_invocations > 0 else 0
                    elif metric_name == 'Invocations':
                        metrics['invocation_count'] = sum(dp.value for dp in metric_data)
                    elif metric_name == 'ConcurrentExecutions':
                        metrics['concurrent_executions'] = np.max([dp.value for dp in metric_data])
            
            # Calculate derived metrics
            metrics['cold_start_rate'] = await self._estimate_cold_start_rate(start_time, end_time)
            metrics['cost_per_invocation'] = await self._calculate_current_cost_per_invocation(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    async def get_baseline_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get baseline performance metrics.
        
        Returns:
            Baseline metrics or None if not available
        """
        if self.baseline_metrics:
            return self.baseline_metrics.copy()
        
        # Calculate baseline from historical data if available
        if len(self.performance_snapshots) >= 10:
            baseline = self._calculate_baseline_from_snapshots()
            self.baseline_metrics = baseline
            return baseline
        
        return None
    
    async def get_performance_trends(
        self,
        time_window: timedelta = timedelta(hours=24),
        metrics: Optional[List[str]] = None
    ) -> List[PerformanceTrend]:
        """
        Analyze performance trends over time.
        
        Args:
            time_window: Time window for trend analysis
            metrics: Specific metrics to analyze
            
        Returns:
            List of performance trends
        """
        trends = []
        metrics_to_analyze = metrics or ['avg_duration', 'error_rate', 'invocation_count']
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            for metric_name in metrics_to_analyze:
                if metric_name == 'avg_duration':
                    cloudwatch_metric = 'Duration'
                elif metric_name == 'error_rate':
                    cloudwatch_metric = 'Errors'
                elif metric_name == 'invocation_count':
                    cloudwatch_metric = 'Invocations'
                else:
                    continue
                
                # Get metric data
                metric_data = await self._get_cloudwatch_metric(
                    cloudwatch_metric, start_time, end_time, period=300
                )
                
                if not metric_data or len(metric_data) < 3:
                    continue
                
                # Analyze trend
                trend = self._analyze_metric_trend(metric_name, metric_data)
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            return []
    
    async def get_cost_metrics(self, time_window: timedelta = timedelta(days=1)) -> Dict[str, float]:
        """
        Calculate cost metrics.
        
        Args:
            time_window: Time window for cost calculation
            
        Returns:
            Cost metrics
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            # Get invocation and duration data
            invocation_data = await self._get_cloudwatch_metric('Invocations', start_time, end_time)
            duration_data = await self._get_cloudwatch_metric('Duration', start_time, end_time)
            
            if not invocation_data or not duration_data:
                return {}
            
            total_invocations = sum(dp.value for dp in invocation_data)
            total_duration_ms = sum(dp.value for dp in duration_data)
            
            # Get current memory size
            current_config = await self._get_current_function_config()
            memory_size = current_config.get('MemorySize', 1024)
            memory_gb = memory_size / 1024.0
            
            # Calculate costs
            compute_cost = (total_duration_ms / 1000.0) * memory_gb * self.config.cost_per_gb_second
            request_cost = total_invocations * self.config.cost_per_request
            total_cost = compute_cost + request_cost
            
            # Extrapolate to monthly
            days_in_window = time_window.total_seconds() / (24 * 3600)
            monthly_cost = (total_cost / days_in_window) * 30 if days_in_window > 0 else 0
            
            return {
                'total_cost': total_cost,
                'compute_cost': compute_cost,
                'request_cost': request_cost,
                'monthly_cost': monthly_cost,
                'cost_per_invocation': total_cost / total_invocations if total_invocations > 0 else 0,
                'invocations': total_invocations,
                'memory_size': memory_size
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate cost metrics: {e}")
            return {}
    
    async def get_error_trends(self, time_window: timedelta = timedelta(hours=6)) -> Dict[str, Any]:
        """
        Analyze error trends.
        
        Args:
            time_window: Time window for error analysis
            
        Returns:
            Error trend analysis
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            # Get error and invocation data
            error_data = await self._get_cloudwatch_metric('Errors', start_time, end_time)
            invocation_data = await self._get_cloudwatch_metric('Invocations', start_time, end_time)
            
            if not error_data or not invocation_data:
                return {}
            
            # Calculate error rates over time
            error_rates = []
            for i in range(len(error_data)):
                if i < len(invocation_data) and invocation_data[i].value > 0:
                    error_rate = error_data[i].value / invocation_data[i].value
                    error_rates.append(error_rate)
            
            if not error_rates:
                return {}
            
            recent_error_rate = np.mean(error_rates[-3:]) if len(error_rates) >= 3 else np.mean(error_rates)
            historical_error_rate = np.mean(error_rates[:-3]) if len(error_rates) > 3 else recent_error_rate
            
            # Determine trend
            if len(error_rates) >= 5:
                recent_avg = np.mean(error_rates[-5:])
                earlier_avg = np.mean(error_rates[:-5]) if len(error_rates) > 5 else recent_avg
                
                if recent_avg > earlier_avg * 1.5:
                    trend = "increasing"
                elif recent_avg < earlier_avg * 0.5:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            return {
                'recent_error_rate': recent_error_rate,
                'historical_error_rate': historical_error_rate,
                'trend': trend,
                'error_rate_history': error_rates,
                'total_errors': sum(dp.value for dp in error_data),
                'total_invocations': sum(dp.value for dp in invocation_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze error trends: {e}")
            return {}
    
    async def analyze_traffic_patterns(self, time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """
        Analyze traffic patterns for pattern change detection.
        
        Args:
            time_window: Time window for pattern analysis
            
        Returns:
            Traffic pattern analysis
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            # Get invocation data with hourly granularity
            invocation_data = await self._get_cloudwatch_metric(
                'Invocations', start_time, end_time, period=3600
            )
            
            if not invocation_data or len(invocation_data) < 24:
                return {'pattern_change_score': 0.0}
            
            # Analyze patterns
            hourly_patterns = self._extract_hourly_patterns(invocation_data)
            daily_patterns = self._extract_daily_patterns(invocation_data)
            
            # Compare with historical patterns
            pattern_change_score = await self._calculate_pattern_change_score(
                hourly_patterns, daily_patterns
            )
            
            # Detect current pattern type
            detected_pattern = self._classify_traffic_pattern(hourly_patterns)
            
            return {
                'pattern_change_score': pattern_change_score,
                'detected_pattern': detected_pattern,
                'hourly_patterns': hourly_patterns,
                'daily_patterns': daily_patterns,
                'analysis_period': time_window.total_seconds() / 3600  # hours
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze traffic patterns: {e}")
            return {'pattern_change_score': 0.0}
    
    async def get_current_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.
        
        Returns:
            Current status information
        """
        return {
            'is_monitoring': self.is_monitoring,
            'snapshots_collected': len(self.performance_snapshots),
            'baseline_available': self.baseline_metrics is not None,
            'last_snapshot': self.performance_snapshots[-1].timestamp.isoformat() if self.performance_snapshots else None,
            'monitoring_interval': self.monitoring_interval,
            'data_directory': str(self.data_dir)
        }
    
    async def get_recent_performance_data(
        self,
        time_window: timedelta = timedelta(hours=1)
    ) -> Optional[Dict[str, Any]]:
        """
        Get recent performance data for optimization decisions.
        
        Args:
            time_window: Time window for data collection
            
        Returns:
            Recent performance data
        """
        try:
            current_metrics = await self.get_current_metrics(time_window)
            
            if not current_metrics:
                return None
            
            # Convert to format expected by optimization components
            memory_results = {
                current_metrics.get('memory_size', 1024): type('MemoryTestResult', (), {
                    'memory_size': current_metrics.get('memory_size', 1024),
                    'iterations': int(current_metrics.get('invocation_count', 0)),
                    'avg_duration': current_metrics.get('avg_duration', 0),
                    'p95_duration': current_metrics.get('p95_duration', 0),
                    'p99_duration': current_metrics.get('p99_duration', 0),
                    'avg_cost': current_metrics.get('cost_per_invocation', 0),
                    'total_cost': 0,
                    'cold_starts': int(current_metrics.get('cold_start_rate', 0) * current_metrics.get('invocation_count', 0)),
                    'errors': int(current_metrics.get('error_rate', 0) * current_metrics.get('invocation_count', 0)),
                    'raw_results': []
                })()
            }
            
            return {
                'memory_results': memory_results,
                'analysis': current_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get recent performance data: {e}")
            return None
    
    async def measure_configuration_impact(
        self,
        previous_config: Dict[str, Any],
        new_config: Dict[str, Any],
        measurement_window: timedelta = timedelta(minutes=10)
    ) -> Dict[str, float]:
        """
        Measure impact of configuration changes.
        
        Args:
            previous_config: Previous configuration
            new_config: New configuration
            measurement_window: Time window for impact measurement
            
        Returns:
            Impact metrics
        """
        try:
            # Wait for new configuration to take effect
            await asyncio.sleep(30)
            
            # Get metrics after change
            end_time = datetime.utcnow()
            start_time = end_time - measurement_window
            
            post_change_metrics = await self.get_current_metrics(measurement_window)
            
            # Compare with baseline
            baseline = await self.get_baseline_metrics()
            
            impact = {}
            
            if baseline and post_change_metrics:
                for metric in ['avg_duration', 'error_rate', 'cost_per_invocation']:
                    if metric in baseline and metric in post_change_metrics:
                        baseline_value = baseline[metric]
                        current_value = post_change_metrics[metric]
                        
                        if baseline_value > 0:
                            improvement = (baseline_value - current_value) / baseline_value
                            impact[f'{metric}_improvement'] = improvement
            
            # Add configuration change info
            impact['memory_change'] = new_config.get('MemorySize', 0) - previous_config.get('MemorySize', 0)
            impact['measurement_window_minutes'] = measurement_window.total_seconds() / 60
            
            return impact
            
        except Exception as e:
            logger.warning(f"Failed to measure configuration impact: {e}")
            return {}
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting performance monitoring loop")
        
        while self.is_monitoring:
            try:
                # Collect performance snapshot
                snapshot = await self._collect_performance_snapshot()
                
                if snapshot:
                    self.performance_snapshots.append(snapshot)
                    
                    # Maintain snapshot limit
                    if len(self.performance_snapshots) > self.max_snapshots:
                        self.performance_snapshots.pop(0)
                    
                    # Update baseline if needed
                    await self._update_baseline_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _collect_performance_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Collect a performance snapshot."""
        try:
            current_metrics = await self.get_current_metrics(timedelta(minutes=5))
            
            if not current_metrics:
                return None
            
            # Get current function configuration
            config = await self._get_current_function_config()
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                function_arn=self.config.function_arn,
                memory_size=config.get('MemorySize', 1024),
                avg_duration=current_metrics.get('avg_duration', 0),
                p95_duration=current_metrics.get('p95_duration', 0),
                p99_duration=current_metrics.get('p99_duration', 0),
                error_rate=current_metrics.get('error_rate', 0),
                throttle_rate=current_metrics.get('throttle_rate', 0),
                cold_start_rate=current_metrics.get('cold_start_rate', 0),
                concurrent_executions=int(current_metrics.get('concurrent_executions', 0)),
                invocation_count=int(current_metrics.get('invocation_count', 0)),
                cost_per_invocation=current_metrics.get('cost_per_invocation', 0)
            )
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Failed to collect performance snapshot: {e}")
            return None
    
    async def _get_cloudwatch_metric(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 60
    ) -> List[MetricDataPoint]:
        """Get CloudWatch metric data."""
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName=metric_name,
                Dimensions=[
                    {
                        'Name': 'FunctionName',
                        'Value': self.config.function_arn.split(':')[-1]  # Extract function name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=['Average', 'Sum', 'Maximum'],
                Unit='None'
            )
            
            data_points = []
            for point in response.get('Datapoints', []):
                # Use appropriate statistic based on metric
                if metric_name in ['Invocations', 'Errors', 'Throttles']:
                    value = point.get('Sum', 0)
                else:
                    value = point.get('Average', 0)
                
                data_point = MetricDataPoint(
                    timestamp=point['Timestamp'],
                    value=value,
                    unit=point.get('Unit', 'None'),
                    metric_name=metric_name
                )
                data_points.append(data_point)
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x.timestamp)
            
            return data_points
            
        except ClientError as e:
            logger.warning(f"Failed to get CloudWatch metric {metric_name}: {e}")
            return []
    
    async def _estimate_cold_start_rate(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """Estimate cold start rate from CloudWatch logs."""
        try:
            # This is a simplified estimation
            # In a real implementation, you would parse CloudWatch logs for INIT_START events
            
            # For now, return a default estimate based on invocation patterns
            invocation_data = await self._get_cloudwatch_metric('Invocations', start_time, end_time)
            
            if not invocation_data:
                return 0.0
            
            # Simple heuristic: more invocations = lower cold start rate
            total_invocations = sum(dp.value for dp in invocation_data)
            
            if total_invocations < 10:
                return 0.5  # High cold start rate for low traffic
            elif total_invocations < 100:
                return 0.2  # Medium cold start rate
            else:
                return 0.05  # Low cold start rate for high traffic
                
        except Exception as e:
            logger.warning(f"Failed to estimate cold start rate: {e}")
            return 0.1  # Default estimate
    
    async def _calculate_current_cost_per_invocation(self, metrics: Dict[str, float]) -> float:
        """Calculate current cost per invocation."""
        try:
            # Get current function configuration
            config = await self._get_current_function_config()
            memory_size = config.get('MemorySize', 1024)
            memory_gb = memory_size / 1024.0
            
            avg_duration = metrics.get('avg_duration', 0)
            
            # Calculate costs
            compute_cost = memory_gb * avg_duration * self.config.cost_per_gb_second
            request_cost = self.config.cost_per_request
            
            return compute_cost + request_cost
            
        except Exception as e:
            logger.warning(f"Failed to calculate cost per invocation: {e}")
            return 0.0
    
    def _calculate_baseline_from_snapshots(self) -> Dict[str, float]:
        """Calculate baseline metrics from historical snapshots."""
        if len(self.performance_snapshots) < 10:
            return {}
        
        # Use the oldest 50% of snapshots as baseline
        baseline_snapshots = self.performance_snapshots[:len(self.performance_snapshots)//2]
        
        durations = [s.avg_duration for s in baseline_snapshots]
        error_rates = [s.error_rate for s in baseline_snapshots]
        costs = [s.cost_per_invocation for s in baseline_snapshots]
        
        return {
            'avg_duration': np.mean(durations),
            'p95_duration': np.percentile(durations, 95),
            'error_rate': np.mean(error_rates),
            'cost_per_invocation': np.mean(costs)
        }
    
    def _analyze_metric_trend(self, metric_name: str, data_points: List[MetricDataPoint]) -> PerformanceTrend:
        """Analyze trend for a specific metric."""
        values = [dp.value for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        # Calculate trend using linear regression
        if len(values) >= 3:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # Determine trend direction and strength
            if abs(slope) < np.std(values) * 0.1:
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(abs(slope) / np.std(values), 1.0)
            else:
                trend_direction = "decreasing"
                trend_strength = min(abs(slope) / np.std(values), 1.0)
        else:
            trend_direction = "insufficient_data"
            trend_strength = 0.0
        
        # Statistical summary
        statistical_summary = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
        
        return PerformanceTrend(
            metric_name=metric_name,
            time_period=f"{len(data_points)} data points",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            data_points=data_points,
            statistical_summary=statistical_summary
        )
    
    def _extract_hourly_patterns(self, invocation_data: List[MetricDataPoint]) -> Dict[int, float]:
        """Extract hourly usage patterns."""
        hourly_sums = {}
        
        for dp in invocation_data:
            hour = dp.timestamp.hour
            if hour not in hourly_sums:
                hourly_sums[hour] = 0
            hourly_sums[hour] += dp.value
        
        return hourly_sums
    
    def _extract_daily_patterns(self, invocation_data: List[MetricDataPoint]) -> Dict[int, float]:
        """Extract daily usage patterns."""
        daily_sums = {}
        
        for dp in invocation_data:
            day = dp.timestamp.weekday()  # 0=Monday, 6=Sunday
            if day not in daily_sums:
                daily_sums[day] = 0
            daily_sums[day] += dp.value
        
        return daily_sums
    
    async def _calculate_pattern_change_score(
        self,
        current_hourly: Dict[int, float],
        current_daily: Dict[int, float]
    ) -> float:
        """Calculate pattern change score compared to historical patterns."""
        # This would compare with stored historical patterns
        # For now, return a simple score based on variance
        
        if not current_hourly:
            return 0.0
        
        hourly_values = list(current_hourly.values())
        hourly_cv = np.std(hourly_values) / np.mean(hourly_values) if np.mean(hourly_values) > 0 else 0
        
        # High coefficient of variation suggests pattern changes
        return min(hourly_cv, 1.0)
    
    def _classify_traffic_pattern(self, hourly_patterns: Dict[int, float]) -> str:
        """Classify the current traffic pattern."""
        if not hourly_patterns:
            return "unknown"
        
        values = list(hourly_patterns.values())
        if not values:
            return "unknown"
        
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        if cv < 0.3:
            return "steady"
        elif cv > 0.8:
            return "burst"
        else:
            return "variable"
    
    async def _update_baseline_metrics(self):
        """Update baseline metrics if needed."""
        if len(self.performance_snapshots) >= 50 and not self.baseline_metrics:
            self.baseline_metrics = self._calculate_baseline_from_snapshots()
            logger.info("Baseline metrics calculated from snapshots")
    
    async def _get_current_function_config(self) -> Dict[str, Any]:
        """Get current Lambda function configuration."""
        try:
            response = self.lambda_client.get_function_configuration(
                FunctionName=self.config.function_arn
            )
            
            return {
                'MemorySize': response['MemorySize'],
                'Timeout': response['Timeout'],
                'Runtime': response['Runtime']
            }
            
        except ClientError as e:
            logger.error(f"Failed to get function configuration: {e}")
            return {}
    
    async def _load_historical_data(self):
        """Load historical monitoring data."""
        try:
            data_file = self.data_dir / "performance_snapshots.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert to PerformanceSnapshot objects
                for item in data:
                    snapshot = PerformanceSnapshot(
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        function_arn=item['function_arn'],
                        memory_size=item['memory_size'],
                        avg_duration=item['avg_duration'],
                        p95_duration=item['p95_duration'],
                        p99_duration=item['p99_duration'],
                        error_rate=item['error_rate'],
                        throttle_rate=item['throttle_rate'],
                        cold_start_rate=item['cold_start_rate'],
                        concurrent_executions=item['concurrent_executions'],
                        invocation_count=item['invocation_count'],
                        cost_per_invocation=item['cost_per_invocation']
                    )
                    self.performance_snapshots.append(snapshot)
                    
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
    
    async def _save_monitoring_data(self):
        """Save current monitoring data."""
        try:
            # Save performance snapshots
            data = []
            for snapshot in self.performance_snapshots[-100:]:  # Save last 100 snapshots
                data.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'function_arn': snapshot.function_arn,
                    'memory_size': snapshot.memory_size,
                    'avg_duration': snapshot.avg_duration,
                    'p95_duration': snapshot.p95_duration,
                    'p99_duration': snapshot.p99_duration,
                    'error_rate': snapshot.error_rate,
                    'throttle_rate': snapshot.throttle_rate,
                    'cold_start_rate': snapshot.cold_start_rate,
                    'concurrent_executions': snapshot.concurrent_executions,
                    'invocation_count': snapshot.invocation_count,
                    'cost_per_invocation': snapshot.cost_per_invocation
                })
            
            data_file = self.data_dir / "performance_snapshots.json"
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Save baseline metrics
            if self.baseline_metrics:
                baseline_file = self.data_dir / "baseline_metrics.json"
                with open(baseline_file, 'w') as f:
                    json.dump(self.baseline_metrics, f, indent=2)
                    
        except Exception as e:
            logger.warning(f"Failed to save monitoring data: {e}")
    
    def _create_cloudwatch_client(self):
        """Create CloudWatch client."""
        try:
            return boto3.client('cloudwatch', region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create CloudWatch client: {e}")
            raise
    
    def _create_lambda_client(self):
        """Create Lambda client."""
        try:
            return boto3.client('lambda', region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create Lambda client: {e}")
            raise
    
    def _create_logs_client(self):
        """Create CloudWatch Logs client."""
        try:
            return boto3.client('logs', region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create Logs client: {e}")
            raise