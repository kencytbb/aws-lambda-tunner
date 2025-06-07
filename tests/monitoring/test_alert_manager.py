"""
Tests for the Alert Manager component.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from aws_lambda_tuner.config import TunerConfig
from aws_lambda_tuner.monitoring.alert_manager import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertChannel,
    NotificationConfig,
)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
        region="us-east-1",
        alerts_enabled=True,
        alert_channels=["email", "sns"],
        alert_cooldown_minutes=30,
        email_notifications={
            "from_address": "alerts@example.com",
            "to_address": "admin@example.com",
        },
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:lambda-alerts",
    )


@pytest.fixture
def notification_config():
    """Create a sample notification configuration."""
    return NotificationConfig(
        email_config={
            "from_address": "alerts@example.com",
            "to_address": "admin@example.com",
            "smtp_server": "localhost",
            "smtp_port": "587",
            "use_tls": True,
        },
        sns_config={"topic_arn": "arn:aws:sns:us-east-1:123456789012:lambda-alerts"},
    )


@pytest.fixture
def mock_sns_client():
    """Create a mock SNS client."""
    mock_client = Mock()
    mock_client.publish.return_value = {"MessageId": "test-message-id"}
    return mock_client


@pytest.fixture
def mock_cloudwatch_client():
    """Create a mock CloudWatch client."""
    mock_client = Mock()
    mock_client.put_metric_data.return_value = {}
    return mock_client


class TestAlertManager:
    """Test cases for the Alert Manager."""

    def test_initialization(self, sample_config, notification_config):
        """Test alert manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            assert alert_manager.config == sample_config
            assert alert_manager.notification_config == notification_config
            assert alert_manager.data_dir.exists()
            assert len(alert_manager.active_alerts) == 0
            assert len(alert_manager.alert_history) == 0
            assert len(alert_manager.alert_rules) > 0  # Default rules should be initialized
            assert len(alert_manager.last_alert_times) == 0

    def test_default_alert_rules_initialization(self, sample_config):
        """Test that default alert rules are properly initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, data_dir=temp_dir)

            rules = alert_manager.alert_rules

            assert len(rules) >= 3  # Should have at least 3 default rules

            rule_types = [rule.alert_type for rule in rules]
            assert "performance_degradation" in rule_types
            assert "cost_threshold" in rule_types
            assert "error_rate_increase" in rule_types

            # Check rule structure
            for rule in rules:
                assert hasattr(rule, "rule_id")
                assert hasattr(rule, "name")
                assert hasattr(rule, "alert_type")
                assert hasattr(rule, "severity")
                assert hasattr(rule, "channels")
                assert isinstance(rule.enabled, bool)

    @pytest.mark.asyncio
    async def test_send_performance_degradation_alert(self, sample_config, notification_config):
        """Test sending performance degradation alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)
            alert_manager._send_alert = AsyncMock()

            await alert_manager.send_performance_degradation_alert(
                metric_name="duration",
                current_value=2.5,
                baseline_value=1.5,
                degradation_percent=66.7,
            )

            # Should have called _send_alert
            alert_manager._send_alert.assert_called_once()

            # Get the alert that was sent
            sent_alert = alert_manager._send_alert.call_args[0][0]
            assert isinstance(sent_alert, Alert)
            assert sent_alert.alert_type == "performance_degradation"
            assert sent_alert.severity == AlertSeverity.CRITICAL  # High degradation
            assert "duration" in sent_alert.metadata["metric_name"]
            assert sent_alert.metadata["degradation_percent"] == 66.7

    @pytest.mark.asyncio
    async def test_send_cost_threshold_alert(self, sample_config, notification_config):
        """Test sending cost threshold alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)
            alert_manager._send_alert = AsyncMock()

            await alert_manager.send_cost_threshold_alert(
                monthly_cost=150.0, threshold=100.0, overage=50.0
            )

            alert_manager._send_alert.assert_called_once()

            sent_alert = alert_manager._send_alert.call_args[0][0]
            assert sent_alert.alert_type == "cost_threshold"
            assert sent_alert.severity == AlertSeverity.ERROR  # 50% overage
            assert sent_alert.metadata["monthly_cost"] == 150.0
            assert sent_alert.metadata["threshold"] == 100.0
            assert sent_alert.metadata["overage"] == 50.0

    @pytest.mark.asyncio
    async def test_send_error_rate_alert(self, sample_config, notification_config):
        """Test sending error rate alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)
            alert_manager._send_alert = AsyncMock()

            await alert_manager.send_error_rate_alert(
                current_error_rate=0.12,  # 12% error rate
                baseline_error_rate=0.02,
                total_errors=120,
            )

            alert_manager._send_alert.assert_called_once()

            sent_alert = alert_manager._send_alert.call_args[0][0]
            assert sent_alert.alert_type == "error_rate_increase"
            assert sent_alert.severity == AlertSeverity.CRITICAL  # High error rate
            assert sent_alert.metadata["current_error_rate"] == 0.12
            assert sent_alert.metadata["total_errors"] == 120

    @pytest.mark.asyncio
    async def test_send_custom_alert(self, sample_config, notification_config):
        """Test sending custom alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)
            alert_manager._send_alert = AsyncMock()

            await alert_manager.send_custom_alert(
                title="Custom Test Alert",
                message="This is a test message",
                severity=AlertSeverity.WARNING,
                alert_type="custom_test",
                metadata={"key": "value"},
                channels=[AlertChannel.EMAIL],
            )

            alert_manager._send_alert.assert_called_once()

            sent_alert = alert_manager._send_alert.call_args[0][0]
            assert sent_alert.title == "Custom Test Alert"
            assert sent_alert.message == "This is a test message"
            assert sent_alert.severity == AlertSeverity.WARNING
            assert sent_alert.alert_type == "custom_test"
            assert sent_alert.metadata["key"] == "value"
            assert sent_alert.channels == [AlertChannel.EMAIL]

    def test_severity_determination(self, sample_config):
        """Test severity determination methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, data_dir=temp_dir)

            # Test degradation severity
            assert alert_manager._determine_degradation_severity(10.0) == AlertSeverity.INFO
            assert alert_manager._determine_degradation_severity(25.0) == AlertSeverity.WARNING
            assert alert_manager._determine_degradation_severity(35.0) == AlertSeverity.ERROR
            assert alert_manager._determine_degradation_severity(60.0) == AlertSeverity.CRITICAL

            # Test error severity
            assert alert_manager._determine_error_severity(0.01) == AlertSeverity.INFO
            assert alert_manager._determine_error_severity(0.03) == AlertSeverity.WARNING
            assert alert_manager._determine_error_severity(0.07) == AlertSeverity.ERROR
            assert alert_manager._determine_error_severity(0.15) == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, sample_config, notification_config):
        """Test acknowledging alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Create an alert
            alert = Alert(
                alert_id="test_alert_123",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="Test Alert",
                message="Test message",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={},
                channels=[AlertChannel.EMAIL],
            )

            alert_manager.active_alerts.append(alert)

            # Acknowledge the alert
            success = await alert_manager.acknowledge_alert("test_alert_123")

            assert success
            assert alert.acknowledged

            # Try to acknowledge non-existent alert
            success = await alert_manager.acknowledge_alert("non_existent")
            assert not success

    @pytest.mark.asyncio
    async def test_resolve_alert(self, sample_config, notification_config):
        """Test resolving alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Create an alert
            alert = Alert(
                alert_id="test_alert_456",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="Test Alert",
                message="Test message",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={},
                channels=[AlertChannel.EMAIL],
            )

            alert_manager.active_alerts.append(alert)

            # Resolve the alert
            success = await alert_manager.resolve_alert("test_alert_456")

            assert success
            assert alert.resolved
            assert len(alert_manager.active_alerts) == 0
            assert len(alert_manager.alert_history) == 1
            assert alert_manager.alert_history[0] == alert

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, sample_config, notification_config):
        """Test getting active alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Should start with no active alerts
            active_alerts = await alert_manager.get_active_alerts()
            assert len(active_alerts) == 0

            # Add some alerts
            for i in range(3):
                alert = Alert(
                    alert_id=f"test_alert_{i}",
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    title=f"Test Alert {i}",
                    message="Test message",
                    function_arn=sample_config.function_arn,
                    alert_type="test",
                    metadata={},
                    channels=[AlertChannel.EMAIL],
                )
                alert_manager.active_alerts.append(alert)

            active_alerts = await alert_manager.get_active_alerts()
            assert len(active_alerts) == 3

    @pytest.mark.asyncio
    async def test_get_alert_history(self, sample_config, notification_config):
        """Test getting alert history with filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Add some historical alerts
            base_time = datetime.now()
            for i in range(5):
                alert = Alert(
                    alert_id=f"hist_alert_{i}",
                    timestamp=base_time - timedelta(hours=i),
                    severity=AlertSeverity.WARNING,
                    title=f"Historical Alert {i}",
                    message="Test message",
                    function_arn=sample_config.function_arn,
                    alert_type="performance_degradation" if i % 2 == 0 else "cost_threshold",
                    metadata={},
                    channels=[AlertChannel.EMAIL],
                    resolved=True,
                )
                alert_manager.alert_history.append(alert)

            # Get all history
            all_history = await alert_manager.get_alert_history()
            assert len(all_history) == 5

            # Filter by time window
            recent_history = await alert_manager.get_alert_history(time_window=timedelta(hours=2))
            assert len(recent_history) == 3  # 0, 1, 2 hours ago

            # Filter by alert type
            perf_history = await alert_manager.get_alert_history(
                alert_type="performance_degradation"
            )
            assert len(perf_history) == 3  # indices 0, 2, 4

    @pytest.mark.asyncio
    async def test_get_alert_statistics(self, sample_config, notification_config):
        """Test getting alert statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Add some alerts for statistics
            base_time = datetime.now()
            severities = [
                AlertSeverity.INFO,
                AlertSeverity.WARNING,
                AlertSeverity.ERROR,
                AlertSeverity.WARNING,
            ]
            types = [
                "performance_degradation",
                "cost_threshold",
                "error_rate_increase",
                "performance_degradation",
            ]

            for i in range(4):
                alert = Alert(
                    alert_id=f"stat_alert_{i}",
                    timestamp=base_time - timedelta(hours=i),
                    severity=severities[i],
                    title=f"Stat Alert {i}",
                    message="Test message",
                    function_arn=sample_config.function_arn,
                    alert_type=types[i],
                    metadata={},
                    channels=[AlertChannel.EMAIL],
                    resolved=i > 1,  # First 2 are active, last 2 are resolved
                )

                if i <= 1:
                    alert_manager.active_alerts.append(alert)
                else:
                    alert_manager.alert_history.append(alert)

            stats = await alert_manager.get_alert_statistics()

            assert isinstance(stats, dict)
            assert "total_alerts" in stats
            assert "active_alerts" in stats
            assert "resolved_alerts" in stats
            assert "alerts_by_severity" in stats
            assert "alerts_by_type" in stats
            assert "most_common_alert_type" in stats

            assert stats["total_alerts"] == 4
            assert stats["active_alerts"] == 2
            assert stats["resolved_alerts"] == 2
            assert stats["alerts_by_severity"]["warning"] == 2
            assert stats["alerts_by_type"]["performance_degradation"] == 2
            assert stats["most_common_alert_type"] == "performance_degradation"

    def test_alert_cooldown(self, sample_config, notification_config):
        """Test alert cooldown functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Set a recent alert time
            cooldown_key = "test_alert_test-function"
            alert_manager.last_alert_times[cooldown_key] = datetime.now() - timedelta(minutes=10)

            # Mock the actual alert sending
            alert_manager._send_email_alert = AsyncMock()
            alert_manager._send_sns_alert = AsyncMock()
            alert_manager._send_cloudwatch_alert = AsyncMock()
            alert_manager._store_alert = AsyncMock()

            # Create an alert
            alert = Alert(
                alert_id="test_cooldown",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="Test Alert",
                message="Test message",
                function_arn=sample_config.function_arn,
                alert_type="test_alert",
                metadata={},
                channels=[AlertChannel.EMAIL],
            )

            # This should be suppressed due to cooldown
            alert_manager._send_alert = alert_manager._send_alert.__wrapped__  # Get original method
            # We need to test this differently since it's an async method

    @pytest.mark.asyncio
    async def test_send_email_alert(self, sample_config, notification_config):
        """Test sending email alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            # Mock email sending
            alert_manager._send_email_via_smtp = AsyncMock()
            alert_manager._create_email_body = Mock(return_value="<html>Test email</html>")

            alert = Alert(
                alert_id="email_test",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="Email Test Alert",
                message="Test message",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={},
                channels=[AlertChannel.EMAIL],
            )

            await alert_manager._send_email_alert(alert)

            # Should have attempted to send email
            alert_manager._send_email_via_smtp.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_sns_alert(self, sample_config, notification_config, mock_sns_client):
        """Test sending SNS alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)
            alert_manager.sns_client = mock_sns_client

            alert = Alert(
                alert_id="sns_test",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="SNS Test Alert",
                message="Test message",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={},
                channels=[AlertChannel.SNS],
            )

            await alert_manager._send_sns_alert(alert)

            # Should have called SNS publish
            mock_sns_client.publish.assert_called_once()
            call_args = mock_sns_client.publish.call_args[1]
            assert "TopicArn" in call_args
            assert "Subject" in call_args
            assert "Message" in call_args

    @pytest.mark.asyncio
    async def test_send_cloudwatch_alert(
        self, sample_config, notification_config, mock_cloudwatch_client
    ):
        """Test sending CloudWatch alerts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)
            alert_manager.cloudwatch_client = mock_cloudwatch_client

            alert = Alert(
                alert_id="cw_test",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="CloudWatch Test Alert",
                message="Test message",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={},
                channels=[AlertChannel.CLOUDWATCH],
            )

            await alert_manager._send_cloudwatch_alert(alert)

            # Should have called CloudWatch put_metric_data
            mock_cloudwatch_client.put_metric_data.assert_called_once()
            call_args = mock_cloudwatch_client.put_metric_data.call_args[1]
            assert "Namespace" in call_args
            assert "MetricData" in call_args

    def test_message_formatting(self, sample_config):
        """Test alert message formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, data_dir=temp_dir)

            # Test degradation message
            degradation_msg = alert_manager._format_degradation_message("duration", 2.5, 1.5, 66.7)
            assert "duration" in degradation_msg
            assert "2.5" in degradation_msg
            assert "1.5" in degradation_msg
            assert "66.7" in degradation_msg

            # Test cost threshold message
            cost_msg = alert_manager._format_cost_threshold_message(150.0, 100.0, 50.0)
            assert "$150.00" in cost_msg
            assert "$100.00" in cost_msg
            assert "$50.00" in cost_msg

            # Test error rate message
            error_msg = alert_manager._format_error_rate_message(0.12, 0.02, 120)
            assert "12.00%" in error_msg
            assert "2.00%" in error_msg
            assert "120" in error_msg

    def test_email_body_creation(self, sample_config):
        """Test HTML email body creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, data_dir=temp_dir)

            alert = Alert(
                alert_id="email_body_test",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                title="Email Body Test",
                message="Test message with details",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={"key1": "value1", "key2": "value2"},
                channels=[AlertChannel.EMAIL],
            )

            html_body = alert_manager._create_email_body(alert)

            assert isinstance(html_body, str)
            assert "<!DOCTYPE html>" in html_body
            assert alert.title in html_body
            assert alert.message in html_body
            assert alert.function_arn in html_body
            assert "WARNING" in html_body  # Severity
            assert "key1" in html_body  # Metadata
            assert "value1" in html_body

    def test_sns_message_creation(self, sample_config):
        """Test SNS message creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, data_dir=temp_dir)

            alert = Alert(
                alert_id="sns_message_test",
                timestamp=datetime.now(),
                severity=AlertSeverity.ERROR,
                title="SNS Message Test",
                message="Test SNS message",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={"detail": "important"},
                channels=[AlertChannel.SNS],
            )

            sns_message = alert_manager._create_sns_message(alert)

            assert isinstance(sns_message, str)
            assert alert.title in sns_message
            assert alert.message in sns_message
            assert alert.function_arn in sns_message
            assert "ERROR" in sns_message
            assert "important" in sns_message

    @pytest.mark.asyncio
    async def test_alert_storage(self, sample_config, notification_config):
        """Test alert storage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, notification_config, temp_dir)

            alert = Alert(
                alert_id="storage_test",
                timestamp=datetime.now(),
                severity=AlertSeverity.INFO,
                title="Storage Test",
                message="Test alert storage",
                function_arn=sample_config.function_arn,
                alert_type="test",
                metadata={"stored": True},
                channels=[AlertChannel.EMAIL],
            )

            await alert_manager._store_alert(alert)

            # Check that alert file was created
            alert_files = list(alert_manager.data_dir.glob("alerts_*.jsonl"))
            assert len(alert_files) > 0

            # Read and verify the stored alert
            with open(alert_files[0], "r") as f:
                stored_data = f.read()
                assert "storage_test" in stored_data
                assert "Storage Test" in stored_data

    def test_add_alert_callback(self, sample_config):
        """Test adding alert callbacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_manager = AlertManager(sample_config, data_dir=temp_dir)

            callback_called = False
            callback_alert = None

            def test_callback(alert):
                nonlocal callback_called, callback_alert
                callback_called = True
                callback_alert = alert

            alert_manager.add_alert_callback(test_callback)

            # Verify callback was added
            assert len(alert_manager.alert_callbacks) == 1
            assert alert_manager.alert_callbacks[0] == test_callback


class TestNotificationConfig:
    """Test cases for NotificationConfig."""

    def test_notification_config_creation(self):
        """Test creating notification configuration."""
        config = NotificationConfig(
            email_config={"from": "test@example.com"},
            sns_config={"topic_arn": "arn:aws:sns:us-east-1:123:test"},
            slack_config={"webhook_url": "https://hooks.slack.com/test"},
            webhook_config={"url": "https://api.example.com/webhook"},
        )

        assert config.email_config["from"] == "test@example.com"
        assert config.sns_config["topic_arn"] == "arn:aws:sns:us-east-1:123:test"
        assert config.slack_config["webhook_url"] == "https://hooks.slack.com/test"
        assert config.webhook_config["url"] == "https://api.example.com/webhook"


class TestAlert:
    """Test cases for Alert data structure."""

    def test_alert_creation(self, sample_config):
        """Test creating an alert."""
        alert = Alert(
            alert_id="test_creation",
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING,
            title="Test Alert Creation",
            message="Testing alert creation",
            function_arn=sample_config.function_arn,
            alert_type="test",
            metadata={"test": True},
            channels=[AlertChannel.EMAIL, AlertChannel.SNS],
        )

        assert alert.alert_id == "test_creation"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert Creation"
        assert alert.message == "Testing alert creation"
        assert alert.function_arn == sample_config.function_arn
        assert alert.alert_type == "test"
        assert alert.metadata["test"] is True
        assert AlertChannel.EMAIL in alert.channels
        assert AlertChannel.SNS in alert.channels
        assert not alert.acknowledged
        assert not alert.resolved
