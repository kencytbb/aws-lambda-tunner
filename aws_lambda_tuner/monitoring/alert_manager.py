"""
Alert manager for performance degradation alerts and notifications.
Manages alerts for optimization events and performance issues.
"""

import asyncio
import logging
import smtplib
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from ..config import TunerConfig

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    SNS = "sns"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CLOUDWATCH = "cloudwatch"


@dataclass
class Alert:
    """Represents an alert."""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    function_arn: str
    alert_type: str
    metadata: Dict[str, Any]
    channels: List[AlertChannel]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration."""

    rule_id: str
    name: str
    description: str
    alert_type: str
    severity: AlertSeverity
    conditions: Dict[str, Any]
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown: timedelta = timedelta(minutes=30)


@dataclass
class NotificationConfig:
    """Notification configuration for different channels."""

    email_config: Optional[Dict[str, str]] = None
    sns_config: Optional[Dict[str, str]] = None
    slack_config: Optional[Dict[str, str]] = None
    webhook_config: Optional[Dict[str, str]] = None


class AlertManager:
    """
    Manages alerts and notifications for Lambda performance issues
    and optimization events.
    """

    def __init__(
        self,
        config: TunerConfig,
        notification_config: Optional[NotificationConfig] = None,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize the alert manager.

        Args:
            config: Tuner configuration
            notification_config: Notification channel configurations
            data_dir: Directory for storing alert data
        """
        self.config = config
        self.notification_config = notification_config or NotificationConfig()
        self.data_dir = Path(data_dir or "./alert_data")
        self.data_dir.mkdir(exist_ok=True)

        # AWS clients
        self.sns_client = self._create_sns_client()
        self.cloudwatch_client = self._create_cloudwatch_client()

        # Alert management
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Alert cooldowns (prevent spam)
        self.last_alert_times: Dict[str, datetime] = {}

        # Initialize default alert rules
        self._initialize_default_alert_rules()

        logger.info("Alert manager initialized")

    async def send_performance_degradation_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        degradation_percent: float,
    ):
        """
        Send alert for performance degradation.

        Args:
            metric_name: Name of the degraded metric
            current_value: Current metric value
            baseline_value: Baseline metric value
            degradation_percent: Degradation percentage
        """
        severity = self._determine_degradation_severity(degradation_percent)

        alert = Alert(
            alert_id=f"perf_degradation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            title=f"Performance Degradation Detected: {metric_name}",
            message=self._format_degradation_message(
                metric_name, current_value, baseline_value, degradation_percent
            ),
            function_arn=self.config.function_arn,
            alert_type="performance_degradation",
            metadata={
                "metric_name": metric_name,
                "current_value": current_value,
                "baseline_value": baseline_value,
                "degradation_percent": degradation_percent,
            },
            channels=[AlertChannel.EMAIL, AlertChannel.SNS, AlertChannel.CLOUDWATCH],
        )

        await self._send_alert(alert)

    async def send_cost_threshold_alert(
        self, monthly_cost: float, threshold: float, overage: float
    ):
        """
        Send alert for cost threshold breach.

        Args:
            monthly_cost: Current monthly cost
            threshold: Cost threshold
            overage: Amount over threshold
        """
        severity = AlertSeverity.WARNING if overage < threshold * 0.5 else AlertSeverity.ERROR

        alert = Alert(
            alert_id=f"cost_threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            title="Cost Threshold Exceeded",
            message=self._format_cost_threshold_message(monthly_cost, threshold, overage),
            function_arn=self.config.function_arn,
            alert_type="cost_threshold",
            metadata={"monthly_cost": monthly_cost, "threshold": threshold, "overage": overage},
            channels=[AlertChannel.EMAIL, AlertChannel.SNS],
        )

        await self._send_alert(alert)

    async def send_error_rate_alert(
        self, current_error_rate: float, baseline_error_rate: float, total_errors: int
    ):
        """
        Send alert for error rate increase.

        Args:
            current_error_rate: Current error rate
            baseline_error_rate: Baseline error rate
            total_errors: Total number of errors
        """
        severity = self._determine_error_severity(current_error_rate)

        alert = Alert(
            alert_id=f"error_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            title="Error Rate Increase Detected",
            message=self._format_error_rate_message(
                current_error_rate, baseline_error_rate, total_errors
            ),
            function_arn=self.config.function_arn,
            alert_type="error_rate_increase",
            metadata={
                "current_error_rate": current_error_rate,
                "baseline_error_rate": baseline_error_rate,
                "total_errors": total_errors,
            },
            channels=[AlertChannel.EMAIL, AlertChannel.SNS, AlertChannel.CLOUDWATCH],
        )

        await self._send_alert(alert)

    async def send_optimization_alert(self, optimization_event):
        """
        Send alert for optimization trigger.

        Args:
            optimization_event: Optimization event that was triggered
        """
        alert = Alert(
            alert_id=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=AlertSeverity.INFO,
            title=f"Optimization Triggered: {optimization_event.trigger.value}",
            message=self._format_optimization_alert_message(optimization_event),
            function_arn=self.config.function_arn,
            alert_type="optimization_triggered",
            metadata={
                "trigger": optimization_event.trigger.value,
                "severity": optimization_event.severity,
                "auto_approve": optimization_event.auto_approve,
                "trigger_data": optimization_event.trigger_data,
            },
            channels=(
                [AlertChannel.EMAIL]
                if not optimization_event.auto_approve
                else [AlertChannel.CLOUDWATCH]
            ),
        )

        await self._send_alert(alert)

    async def send_optimization_completion_alert(self, optimization_result):
        """
        Send alert for optimization completion.

        Args:
            optimization_result: Optimization result
        """
        severity = AlertSeverity.INFO if optimization_result.success else AlertSeverity.WARNING

        alert = Alert(
            alert_id=f"optimization_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            title=f"Optimization {'Completed' if optimization_result.success else 'Failed'}",
            message=self._format_optimization_completion_message(optimization_result),
            function_arn=self.config.function_arn,
            alert_type="optimization_completed",
            metadata={
                "success": optimization_result.success,
                "trigger": optimization_result.event.trigger.value,
                "previous_memory": optimization_result.previous_config.get("MemorySize"),
                "new_memory": optimization_result.new_config.get("MemorySize"),
                "performance_impact": optimization_result.performance_impact,
                "error_message": optimization_result.error_message,
            },
            channels=[AlertChannel.EMAIL, AlertChannel.SNS],
        )

        await self._send_alert(alert)

    async def send_custom_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        alert_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[AlertChannel]] = None,
    ):
        """
        Send a custom alert.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            alert_type: Type of alert
            metadata: Additional metadata
            channels: Delivery channels
        """
        alert = Alert(
            alert_id=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            title=title,
            message=message,
            function_arn=self.config.function_arn,
            alert_type=alert_type,
            metadata=metadata or {},
            channels=channels or [AlertChannel.EMAIL],
        )

        await self._send_alert(alert)

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge

        Returns:
            Success status
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True

        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: ID of the alert to resolve

        Returns:
            Success status
        """
        for i, alert in enumerate(self.active_alerts):
            if alert.alert_id == alert_id:
                alert.resolved = True
                resolved_alert = self.active_alerts.pop(i)
                self.alert_history.append(resolved_alert)

                logger.info(f"Alert {alert_id} resolved")
                return True

        return False

    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return self.active_alerts.copy()

    async def get_alert_history(
        self, time_window: Optional[timedelta] = None, alert_type: Optional[str] = None
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            time_window: Time window for history
            alert_type: Filter by alert type

        Returns:
            List of historical alerts
        """
        alerts = self.alert_history.copy()

        if time_window:
            cutoff_time = datetime.now() - time_window
            alerts = [alert for alert in alerts if alert.timestamp >= cutoff_time]

        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]

        return alerts

    async def get_alert_statistics(
        self, time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Get alert statistics.

        Args:
            time_window: Time window for statistics

        Returns:
            Alert statistics
        """
        cutoff_time = datetime.now() - time_window

        # Get recent alerts
        recent_alerts = [
            alert
            for alert in self.alert_history + self.active_alerts
            if alert.timestamp >= cutoff_time
        ]

        # Calculate statistics
        total_alerts = len(recent_alerts)
        alerts_by_severity = {}
        alerts_by_type = {}

        for alert in recent_alerts:
            severity = alert.severity.value
            alert_type = alert.alert_type

            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1

        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_alerts": total_alerts,
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len([a for a in recent_alerts if a.resolved]),
            "alerts_by_severity": alerts_by_severity,
            "alerts_by_type": alerts_by_type,
            "most_common_alert_type": (
                max(alerts_by_type.items(), key=lambda x: x[1])[0] if alerts_by_type else None
            ),
        }

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback to be called when alerts are sent."""
        self.alert_callbacks.append(callback)

    async def _send_alert(self, alert: Alert):
        """
        Send an alert through configured channels.

        Args:
            alert: Alert to send
        """
        # Check cooldown
        cooldown_key = f"{alert.alert_type}_{alert.function_arn}"
        if cooldown_key in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[cooldown_key]
            if time_since_last < timedelta(minutes=30):  # 30-minute cooldown
                logger.debug(f"Alert {alert.alert_id} suppressed due to cooldown")
                return

        self.last_alert_times[cooldown_key] = datetime.now()

        # Add to active alerts
        self.active_alerts.append(alert)

        # Send through channels
        for channel in alert.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SNS:
                    await self._send_sns_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.CLOUDWATCH:
                    await self._send_cloudwatch_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")

        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

        # Store alert
        await self._store_alert(alert)

        logger.info(f"Alert sent: {alert.title} (severity: {alert.severity.value})")

    async def _send_email_alert(self, alert: Alert):
        """Send alert via email."""
        if not self.notification_config.email_config:
            logger.warning("Email configuration not available")
            return

        email_config = self.notification_config.email_config

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = email_config["from_address"]
            msg["To"] = email_config["to_address"]
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, "html"))

            # Send email
            if email_config.get("use_ses"):
                await self._send_email_via_ses(msg)
            else:
                await self._send_email_via_smtp(msg, email_config)

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_sns_alert(self, alert: Alert):
        """Send alert via SNS."""
        if not self.notification_config.sns_config:
            logger.warning("SNS configuration not available")
            return

        sns_config = self.notification_config.sns_config
        topic_arn = sns_config.get("topic_arn")

        if not topic_arn:
            logger.warning("SNS topic ARN not configured")
            return

        try:
            message = self._create_sns_message(alert)

            self.sns_client.publish(TopicArn=topic_arn, Subject=alert.title, Message=message)

        except ClientError as e:
            logger.error(f"Failed to send SNS alert: {e}")

    async def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack webhook."""
        if not self.notification_config.slack_config:
            logger.warning("Slack configuration not available")
            return

        # Implementation would depend on Slack webhook integration
        logger.info(f"Slack alert would be sent: {alert.title}")

    async def _send_webhook_alert(self, alert: Alert):
        """Send alert via webhook."""
        if not self.notification_config.webhook_config:
            logger.warning("Webhook configuration not available")
            return

        # Implementation would depend on webhook endpoint
        logger.info(f"Webhook alert would be sent: {alert.title}")

    async def _send_cloudwatch_alert(self, alert: Alert):
        """Send alert to CloudWatch as custom metric."""
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace="LambdaTuner/Alerts",
                MetricData=[
                    {
                        "MetricName": "AlertCount",
                        "Dimensions": [
                            {
                                "Name": "FunctionName",
                                "Value": self.config.function_arn.split(":")[-1],
                            },
                            {"Name": "AlertType", "Value": alert.alert_type},
                            {"Name": "Severity", "Value": alert.severity.value},
                        ],
                        "Value": 1,
                        "Unit": "Count",
                        "Timestamp": alert.timestamp,
                    }
                ],
            )

        except ClientError as e:
            logger.error(f"Failed to send CloudWatch alert: {e}")

    def _determine_degradation_severity(self, degradation_percent: float) -> AlertSeverity:
        """Determine severity based on degradation percentage."""
        if degradation_percent >= 50:
            return AlertSeverity.CRITICAL
        elif degradation_percent >= 30:
            return AlertSeverity.ERROR
        elif degradation_percent >= 20:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _determine_error_severity(self, error_rate: float) -> AlertSeverity:
        """Determine severity based on error rate."""
        if error_rate >= 0.1:  # 10% error rate
            return AlertSeverity.CRITICAL
        elif error_rate >= 0.05:  # 5% error rate
            return AlertSeverity.ERROR
        elif error_rate >= 0.02:  # 2% error rate
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _format_degradation_message(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        degradation_percent: float,
    ) -> str:
        """Format performance degradation message."""
        return f"""
        Performance degradation detected for Lambda function {self.config.function_arn}:
        
        Metric: {metric_name}
        Current Value: {current_value:.4f}
        Baseline Value: {baseline_value:.4f}
        Degradation: {degradation_percent:.1f}%
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        This may indicate a need for optimization or investigation into the cause of performance degradation.
        """

    def _format_cost_threshold_message(
        self, monthly_cost: float, threshold: float, overage: float
    ) -> str:
        """Format cost threshold breach message."""
        return f"""
        Cost threshold exceeded for Lambda function {self.config.function_arn}:
        
        Current Monthly Cost: ${monthly_cost:.2f}
        Threshold: ${threshold:.2f}
        Overage: ${overage:.2f}
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        Consider optimization to reduce costs.
        """

    def _format_error_rate_message(
        self, current_error_rate: float, baseline_error_rate: float, total_errors: int
    ) -> str:
        """Format error rate increase message."""
        return f"""
        Error rate increase detected for Lambda function {self.config.function_arn}:
        
        Current Error Rate: {current_error_rate:.2%}
        Baseline Error Rate: {baseline_error_rate:.2%}
        Total Errors: {total_errors}
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        Please investigate the cause of increased error rates.
        """

    def _format_optimization_alert_message(self, optimization_event) -> str:
        """Format optimization trigger message."""
        return f"""
        Optimization triggered for Lambda function {self.config.function_arn}:
        
        Trigger: {optimization_event.trigger.value}
        Severity: {optimization_event.severity}
        Auto-approve: {optimization_event.auto_approve}
        
        Trigger Data: {json.dumps(optimization_event.trigger_data, indent=2)}
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        {'Optimization will be executed automatically.' if optimization_event.auto_approve else 'Manual approval required for optimization.'}
        """

    def _format_optimization_completion_message(self, optimization_result) -> str:
        """Format optimization completion message."""
        status = "completed successfully" if optimization_result.success else "failed"

        message = f"""
        Optimization {status} for Lambda function {self.config.function_arn}:
        
        Trigger: {optimization_result.event.trigger.value}
        Status: {'Success' if optimization_result.success else 'Failed'}
        """

        if optimization_result.previous_config and optimization_result.new_config:
            prev_memory = optimization_result.previous_config.get("MemorySize", "Unknown")
            new_memory = optimization_result.new_config.get("MemorySize", "Unknown")
            message += f"""
        Memory Change: {prev_memory}MB → {new_memory}MB
        """

        if optimization_result.performance_impact:
            message += f"""
        Performance Impact: {json.dumps(optimization_result.performance_impact, indent=2)}
        """

        if optimization_result.error_message:
            message += f"""
        Error: {optimization_result.error_message}
        """

        message += f"""
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        return message

    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#6f42c1",
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .alert-body {{ padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
                .metadata {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; margin-top: 10px; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>{alert.title}</h2>
                <div class="timestamp">Severity: {alert.severity.value.upper()} | {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
            <div class="alert-body">
                <p><strong>Function:</strong> {alert.function_arn}</p>
                <p><strong>Alert Type:</strong> {alert.alert_type}</p>
                <div>{alert.message}</div>
                {f'<div class="metadata"><strong>Additional Data:</strong><pre>{json.dumps(alert.metadata, indent=2)}</pre></div>' if alert.metadata else ''}
            </div>
        </body>
        </html>
        """

    def _create_sns_message(self, alert: Alert) -> str:
        """Create SNS message."""
        return f"""
{alert.title}

Function: {alert.function_arn}
Severity: {alert.severity.value.upper()}
Type: {alert.alert_type}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}

{f'Metadata: {json.dumps(alert.metadata)}' if alert.metadata else ''}
        """.strip()

    def _initialize_default_alert_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="performance_degradation",
                name="Performance Degradation",
                description="Alert when performance degrades significantly",
                alert_type="performance_degradation",
                severity=AlertSeverity.WARNING,
                conditions={"degradation_threshold": 20.0},
                channels=[AlertChannel.EMAIL, AlertChannel.SNS],
            ),
            AlertRule(
                rule_id="cost_threshold",
                name="Cost Threshold Breach",
                description="Alert when costs exceed threshold",
                alert_type="cost_threshold",
                severity=AlertSeverity.WARNING,
                conditions={"threshold_multiplier": 1.2},
                channels=[AlertChannel.EMAIL],
            ),
            AlertRule(
                rule_id="error_rate_spike",
                name="Error Rate Spike",
                description="Alert when error rates spike",
                alert_type="error_rate_increase",
                severity=AlertSeverity.ERROR,
                conditions={"error_rate_threshold": 0.05},
                channels=[AlertChannel.EMAIL, AlertChannel.SNS, AlertChannel.CLOUDWATCH],
            ),
        ]

        self.alert_rules.extend(default_rules)

    async def _send_email_via_ses(self, msg: MIMEMultipart):
        """Send email via AWS SES."""
        # Implementation would use boto3 SES client
        logger.info("Email would be sent via SES")

    async def _send_email_via_smtp(self, msg: MIMEMultipart, email_config: Dict[str, str]):
        """Send email via SMTP."""
        try:
            smtp_server = email_config.get("smtp_server", "localhost")
            smtp_port = int(email_config.get("smtp_port", 587))
            username = email_config.get("username")
            password = email_config.get("password")

            server = smtplib.SMTP(smtp_server, smtp_port)

            if email_config.get("use_tls", True):
                server.starttls()

            if username and password:
                server.login(username, password)

            text = msg.as_string()
            server.sendmail(msg["From"], msg["To"], text)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")

    async def _store_alert(self, alert: Alert):
        """Store alert for history."""
        try:
            alert_record = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "function_arn": alert.function_arn,
                "alert_type": alert.alert_type,
                "metadata": alert.metadata,
                "channels": [ch.value for ch in alert.channels],
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
            }

            alerts_file = self.data_dir / f"alerts_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(alerts_file, "a") as f:
                f.write(json.dumps(alert_record) + "\n")

        except Exception as e:
            logger.warning(f"Failed to store alert: {e}")

    def _create_sns_client(self):
        """Create SNS client."""
        try:
            return boto3.client("sns", region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create SNS client: {e}")
            return None

    def _create_cloudwatch_client(self):
        """Create CloudWatch client."""
        try:
            return boto3.client("cloudwatch", region_name=self.config.region)
        except Exception as e:
            logger.error(f"Failed to create CloudWatch client: {e}")
            return None
