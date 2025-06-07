"""
Tests for the Performance Monitor component.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from aws_lambda_tuner.config import TunerConfig
from aws_lambda_tuner.monitoring.performance_monitor import (
    PerformanceMonitor, MetricDataPoint, PerformanceSnapshot, PerformanceTrend
)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
        region="us-east-1",
        monitoring_enabled=True,
        monitoring_interval_minutes=5,
        baseline_window_hours=24,
        performance_window_minutes=15
    )


@pytest.fixture
def mock_cloudwatch_client():
    """Create a mock CloudWatch client."""
    mock_client = Mock()
    mock_client.get_metric_statistics.return_value = {
        'Datapoints': [
            {
                'Timestamp': datetime.utcnow(),
                'Average': 1500.0,
                'Sum': 15000.0,
                'Maximum': 2000.0,
                'Unit': 'Milliseconds'
            },
            {
                'Timestamp': datetime.utcnow() - timedelta(minutes=5),
                'Average': 1600.0,
                'Sum': 16000.0,
                'Maximum': 2100.0,
                'Unit': 'Milliseconds'
            }
        ]
    }
    return mock_client


@pytest.fixture
def mock_lambda_client():
    """Create a mock Lambda client."""
    mock_client = Mock()
    mock_client.get_function_configuration.return_value = {
        'MemorySize': 512,
        'Timeout': 30,
        'Runtime': 'python3.9'
    }
    return mock_client


class TestPerformanceMonitor:
    """Test cases for the Performance Monitor."""
    
    def test_initialization(self, sample_config):
        """Test performance monitor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            assert monitor.config == sample_config
            assert monitor.data_dir.exists()
            assert monitor.monitoring_interval == 60  # seconds
            assert not monitor.is_monitoring
            assert len(monitor.performance_snapshots) == 0
            assert len(monitor.monitored_metrics) > 0
    
    @patch('aws_lambda_tuner.monitoring.performance_monitor.boto3')
    def test_client_creation(self, mock_boto3, sample_config):
        """Test AWS client creation."""
        mock_boto3.client.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Should create CloudWatch, Lambda, and Logs clients
            assert mock_boto3.client.call_count >= 3
            
            # Check that clients were created with correct service names
            call_args = [call[0][0] for call in mock_boto3.client.call_args_list]
            assert 'cloudwatch' in call_args
            assert 'lambda' in call_args
            assert 'logs' in call_args
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, sample_config):
        """Test starting and stopping monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Mock the monitoring loop to prevent infinite loop
            monitor._monitoring_loop = AsyncMock()
            monitor._load_historical_data = AsyncMock()
            
            # Start monitoring
            await monitor.start_monitoring()
            assert monitor.is_monitoring
            
            # Stop monitoring
            monitor._save_monitoring_data = AsyncMock()
            await monitor.stop_monitoring()
            assert not monitor.is_monitoring
    
    @pytest.mark.asyncio
    async def test_get_cloudwatch_metric(self, sample_config, mock_cloudwatch_client):
        """Test getting CloudWatch metric data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            
            start_time = datetime.utcnow() - timedelta(hours=1)
            end_time = datetime.utcnow()
            
            data_points = await monitor._get_cloudwatch_metric(
                'Duration', start_time, end_time
            )
            
            assert isinstance(data_points, list)
            assert len(data_points) == 2  # Based on mock data
            
            for point in data_points:
                assert isinstance(point, MetricDataPoint)
                assert point.metric_name == 'Duration'
                assert point.value > 0
                assert isinstance(point.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_get_current_metrics(self, sample_config, mock_cloudwatch_client):
        """Test getting current performance metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            monitor._get_current_function_config = AsyncMock(return_value={'MemorySize': 512})
            monitor._estimate_cold_start_rate = AsyncMock(return_value=0.1)
            monitor._calculate_current_cost_per_invocation = AsyncMock(return_value=0.000002)
            
            metrics = await monitor.get_current_metrics()
            
            assert isinstance(metrics, dict)
            assert 'avg_duration' in metrics
            assert 'error_rate' in metrics
            assert 'throttle_rate' in metrics
            assert 'invocation_count' in metrics
            assert 'cold_start_rate' in metrics
            assert 'cost_per_invocation' in metrics
            
            # Check that durations are converted to seconds
            assert metrics['avg_duration'] == 1.5  # 1500ms / 1000
    
    @pytest.mark.asyncio
    async def test_get_baseline_metrics(self, sample_config):
        """Test getting baseline metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Test with no baseline
            baseline = await monitor.get_baseline_metrics()
            assert baseline is None
            
            # Add some snapshots
            for i in range(15):
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.utcnow() - timedelta(minutes=i*5),
                    function_arn=sample_config.function_arn,
                    memory_size=512,
                    avg_duration=1.5 + i*0.1,
                    p95_duration=2.0 + i*0.1,
                    p99_duration=2.5 + i*0.1,
                    error_rate=0.01,
                    throttle_rate=0.0,
                    cold_start_rate=0.1,
                    concurrent_executions=5,
                    invocation_count=100,
                    cost_per_invocation=0.000002
                )
                monitor.performance_snapshots.append(snapshot)
            
            baseline = await monitor.get_baseline_metrics()
            assert baseline is not None
            assert 'avg_duration' in baseline
            assert 'error_rate' in baseline
            assert 'cost_per_invocation' in baseline
    
    @pytest.mark.asyncio
    async def test_get_performance_trends(self, sample_config, mock_cloudwatch_client):
        """Test performance trend analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            
            trends = await monitor.get_performance_trends()
            
            assert isinstance(trends, list)
            
            for trend in trends:
                assert isinstance(trend, PerformanceTrend)
                assert trend.metric_name in ['avg_duration', 'error_rate', 'invocation_count']
                assert trend.trend_direction in ['increasing', 'decreasing', 'stable', 'insufficient_data']
                assert 0 <= trend.trend_strength <= 1
                assert isinstance(trend.statistical_summary, dict)
    
    @pytest.mark.asyncio
    async def test_get_cost_metrics(self, sample_config, mock_cloudwatch_client, mock_lambda_client):
        """Test cost metrics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            monitor.lambda_client = mock_lambda_client
            
            cost_metrics = await monitor.get_cost_metrics()
            
            assert isinstance(cost_metrics, dict)
            assert 'total_cost' in cost_metrics
            assert 'compute_cost' in cost_metrics
            assert 'request_cost' in cost_metrics
            assert 'monthly_cost' in cost_metrics
            assert 'cost_per_invocation' in cost_metrics
            assert 'invocations' in cost_metrics
            assert 'memory_size' in cost_metrics
            
            # All costs should be non-negative
            for key in ['total_cost', 'compute_cost', 'request_cost', 'monthly_cost']:
                assert cost_metrics[key] >= 0
    
    @pytest.mark.asyncio
    async def test_get_error_trends(self, sample_config, mock_cloudwatch_client):
        """Test error trend analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            
            error_trends = await monitor.get_error_trends()
            
            assert isinstance(error_trends, dict)
            assert 'recent_error_rate' in error_trends
            assert 'historical_error_rate' in error_trends
            assert 'trend' in error_trends
            assert 'total_errors' in error_trends
            assert 'total_invocations' in error_trends
            
            assert error_trends['trend'] in ['increasing', 'decreasing', 'stable', 'insufficient_data']
    
    @pytest.mark.asyncio
    async def test_analyze_traffic_patterns(self, sample_config, mock_cloudwatch_client):
        """Test traffic pattern analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            
            # Mock traffic pattern analysis methods
            monitor._extract_hourly_patterns = Mock(return_value={9: 100, 10: 150, 11: 120})
            monitor._extract_daily_patterns = Mock(return_value={0: 500, 1: 600, 2: 550})
            monitor._calculate_pattern_change_score = AsyncMock(return_value=0.3)
            monitor._classify_traffic_pattern = Mock(return_value='steady')
            
            traffic_analysis = await monitor.analyze_traffic_patterns()
            
            assert isinstance(traffic_analysis, dict)
            assert 'pattern_change_score' in traffic_analysis
            assert 'detected_pattern' in traffic_analysis
            assert 'hourly_patterns' in traffic_analysis
            assert 'daily_patterns' in traffic_analysis
            
            assert 0 <= traffic_analysis['pattern_change_score'] <= 1
            assert traffic_analysis['detected_pattern'] in ['burst', 'steady', 'variable', 'unknown']
    
    @pytest.mark.asyncio
    async def test_get_current_status(self, sample_config):
        """Test getting current monitoring status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            status = await monitor.get_current_status()
            
            assert isinstance(status, dict)
            assert 'is_monitoring' in status
            assert 'snapshots_collected' in status
            assert 'baseline_available' in status
            assert 'monitoring_interval' in status
            assert 'data_directory' in status
            
            assert status['is_monitoring'] == monitor.is_monitoring
            assert status['snapshots_collected'] == len(monitor.performance_snapshots)
    
    @pytest.mark.asyncio
    async def test_measure_configuration_impact(self, sample_config):
        """Test measuring configuration change impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Mock dependencies
            monitor.get_current_metrics = AsyncMock(return_value={
                'avg_duration': 1.2,
                'error_rate': 0.01,
                'cost_per_invocation': 0.000002
            })
            monitor.get_baseline_metrics = AsyncMock(return_value={
                'avg_duration': 1.5,
                'error_rate': 0.02,
                'cost_per_invocation': 0.000003
            })
            
            previous_config = {'MemorySize': 256}
            new_config = {'MemorySize': 512}
            
            impact = await monitor.measure_configuration_impact(
                previous_config, new_config
            )
            
            assert isinstance(impact, dict)
            assert 'memory_change' in impact
            assert impact['memory_change'] == 256  # 512 - 256
            
            # Should include improvement metrics if baseline available
            if 'avg_duration_improvement' in impact:
                assert isinstance(impact['avg_duration_improvement'], float)
    
    def test_metric_trend_analysis(self, sample_config):
        """Test trend analysis for individual metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Create sample data points with increasing trend
            data_points = []
            for i in range(10):
                point = MetricDataPoint(
                    timestamp=datetime.utcnow() - timedelta(minutes=i*5),
                    value=100.0 + i*10,  # Increasing trend
                    unit='Milliseconds',
                    metric_name='Duration'
                )
                data_points.append(point)
            
            trend = monitor._analyze_metric_trend('avg_duration', data_points)
            
            assert isinstance(trend, PerformanceTrend)
            assert trend.metric_name == 'avg_duration'
            assert trend.trend_direction in ['increasing', 'decreasing', 'stable']
            assert 0 <= trend.trend_strength <= 1
            assert isinstance(trend.statistical_summary, dict)
            assert 'mean' in trend.statistical_summary
            assert 'std' in trend.statistical_summary
    
    def test_hourly_pattern_extraction(self, sample_config):
        """Test extraction of hourly usage patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Create sample invocation data
            invocation_data = []
            base_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            for hour in range(24):
                point = MetricDataPoint(
                    timestamp=base_time + timedelta(hours=hour),
                    value=100 + hour*5,  # Varying by hour
                    unit='Count',
                    metric_name='Invocations'
                )
                invocation_data.append(point)
            
            hourly_patterns = monitor._extract_hourly_patterns(invocation_data)
            
            assert isinstance(hourly_patterns, dict)
            assert len(hourly_patterns) == 24
            
            for hour in range(24):
                assert hour in hourly_patterns
                assert hourly_patterns[hour] > 0
    
    def test_daily_pattern_extraction(self, sample_config):
        """Test extraction of daily usage patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Create sample invocation data for a week
            invocation_data = []
            base_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            for day in range(7):
                point = MetricDataPoint(
                    timestamp=base_time + timedelta(days=day),
                    value=1000 + day*100,  # Varying by day
                    unit='Count',
                    metric_name='Invocations'
                )
                invocation_data.append(point)
            
            daily_patterns = monitor._extract_daily_patterns(invocation_data)
            
            assert isinstance(daily_patterns, dict)
            assert len(daily_patterns) == 7
            
            for day in range(7):
                assert day in daily_patterns
                assert daily_patterns[day] > 0
    
    def test_traffic_pattern_classification(self, sample_config):
        """Test traffic pattern classification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Test steady pattern (low variance)
            steady_pattern = {hour: 100 for hour in range(24)}
            classification = monitor._classify_traffic_pattern(steady_pattern)
            assert classification == 'steady'
            
            # Test burst pattern (high variance)
            burst_pattern = {hour: 100 if hour not in [9, 17] else 1000 for hour in range(24)}
            classification = monitor._classify_traffic_pattern(burst_pattern)
            assert classification == 'burst'
            
            # Test empty pattern
            empty_pattern = {}
            classification = monitor._classify_traffic_pattern(empty_pattern)
            assert classification == 'unknown'
    
    @pytest.mark.asyncio
    async def test_collect_performance_snapshot(self, sample_config):
        """Test collection of performance snapshots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Mock dependencies
            monitor.get_current_metrics = AsyncMock(return_value={
                'avg_duration': 1.5,
                'p95_duration': 2.0,
                'p99_duration': 2.5,
                'error_rate': 0.01,
                'throttle_rate': 0.0,
                'cold_start_rate': 0.1,
                'concurrent_executions': 5,
                'invocation_count': 100,
                'cost_per_invocation': 0.000002
            })
            monitor._get_current_function_config = AsyncMock(return_value={'MemorySize': 512})
            
            snapshot = await monitor._collect_performance_snapshot()
            
            assert isinstance(snapshot, PerformanceSnapshot)
            assert snapshot.function_arn == sample_config.function_arn
            assert snapshot.memory_size == 512
            assert snapshot.avg_duration == 1.5
            assert snapshot.error_rate == 0.01
            assert snapshot.invocation_count == 100
    
    @pytest.mark.asyncio
    async def test_cold_start_rate_estimation(self, sample_config, mock_cloudwatch_client):
        """Test cold start rate estimation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.cloudwatch = mock_cloudwatch_client
            
            start_time = datetime.utcnow() - timedelta(hours=1)
            end_time = datetime.utcnow()
            
            cold_start_rate = await monitor._estimate_cold_start_rate(start_time, end_time)
            
            assert isinstance(cold_start_rate, float)
            assert 0 <= cold_start_rate <= 1
    
    @pytest.mark.asyncio
    async def test_cost_per_invocation_calculation(self, sample_config, mock_lambda_client):
        """Test cost per invocation calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            monitor.lambda_client = mock_lambda_client
            
            metrics = {'avg_duration': 1.5}
            
            cost = await monitor._calculate_current_cost_per_invocation(metrics)
            
            assert isinstance(cost, float)
            assert cost > 0
            
            # Should include both compute and request costs
            expected_compute = (512 / 1024) * 1.5 * sample_config.cost_per_gb_second
            expected_total = expected_compute + sample_config.cost_per_request
            
            assert abs(cost - expected_total) < 0.000001
    
    @pytest.mark.asyncio
    async def test_historical_data_loading_and_saving(self, sample_config):
        """Test loading and saving of historical monitoring data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(sample_config, temp_dir)
            
            # Add some snapshots
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                function_arn=sample_config.function_arn,
                memory_size=512,
                avg_duration=1.5,
                p95_duration=2.0,
                p99_duration=2.5,
                error_rate=0.01,
                throttle_rate=0.0,
                cold_start_rate=0.1,
                concurrent_executions=5,
                invocation_count=100,
                cost_per_invocation=0.000002
            )
            monitor.performance_snapshots.append(snapshot)
            
            # Save data
            await monitor._save_monitoring_data()
            
            # Create new monitor and load data
            new_monitor = PerformanceMonitor(sample_config, temp_dir)
            await new_monitor._load_historical_data()
            
            assert len(new_monitor.performance_snapshots) == 1
            loaded_snapshot = new_monitor.performance_snapshots[0]
            assert loaded_snapshot.function_arn == snapshot.function_arn
            assert loaded_snapshot.memory_size == snapshot.memory_size