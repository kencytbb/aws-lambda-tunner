"""Mock AWS services for testing."""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock
from io import BytesIO

import boto3
from moto import mock_aws


class MockLambdaClient:
    """Mock AWS Lambda client for testing."""

    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.functions = {}
        self.invocation_count = 0
        self.error_rate = 0.0
        self.cold_start_rate = 0.3
        self.base_duration = 100.0
        self.duration_variance = 0.2

    def create_function(self, **kwargs) -> Dict[str, Any]:
        """Create a mock Lambda function."""
        function_name = kwargs["FunctionName"]
        self.functions[function_name] = {
            "FunctionName": function_name,
            "FunctionArn": f"arn:aws:lambda:{self.region_name}:123456789012:function:{function_name}",
            "Runtime": kwargs.get("Runtime", "python3.9"),
            "Role": kwargs.get("Role", "arn:aws:iam::123456789012:role/lambda-role"),
            "Handler": kwargs.get("Handler", "index.handler"),
            "MemorySize": kwargs.get("MemorySize", 256),
            "Timeout": kwargs.get("Timeout", 30),
            "CodeSize": 1024,
            "LastModified": datetime.utcnow().isoformat() + "Z",
            "State": "Active",
        }
        return self.functions[function_name]

    def get_function(self, FunctionName: str) -> Dict[str, Any]:
        """Get function configuration."""
        if FunctionName not in self.functions:
            raise Exception("ResourceNotFoundException")

        return {
            "Configuration": self.functions[FunctionName],
            "Code": {"Location": "https://s3.amazonaws.com/bucket/key.zip"},
        }

    def get_function_configuration(self, FunctionName: str) -> Dict[str, Any]:
        """Get function configuration only."""
        if FunctionName not in self.functions:
            raise Exception("ResourceNotFoundException")

        return self.functions[FunctionName]

    def update_function_configuration(self, FunctionName: str, **kwargs) -> Dict[str, Any]:
        """Update function configuration."""
        if FunctionName not in self.functions:
            raise Exception("ResourceNotFoundException")

        config = self.functions[FunctionName]
        for key, value in kwargs.items():
            if key in config:
                config[key] = value

        return config

    def invoke(self, FunctionName: str, Payload: bytes = b"{}", **kwargs) -> Dict[str, Any]:
        """Mock function invocation."""
        self.invocation_count += 1

        # Simulate errors
        if random.random() < self.error_rate:
            return {
                "StatusCode": 500,
                "FunctionError": "Unhandled",
                "Payload": BytesIO(b'{"errorMessage": "Test error"}'),
                "LogResult": self._generate_log_result(error=True),
            }

        # Get function config
        if FunctionName not in self.functions:
            raise Exception("ResourceNotFoundException")

        config = self.functions[FunctionName]
        memory_size = config["MemorySize"]

        # Simulate duration based on memory size (higher memory = faster execution)
        base_duration = self.base_duration
        memory_factor = 256 / memory_size  # Inverse relationship
        duration = base_duration * memory_factor

        # Add variance
        variance = duration * self.duration_variance
        duration += random.uniform(-variance, variance)
        duration = max(1.0, duration)  # Minimum 1ms

        # Simulate cold start
        is_cold_start = random.random() < self.cold_start_rate
        if is_cold_start:
            duration += random.uniform(500, 2000)  # Cold start penalty

        # Calculate billed duration (rounded up to nearest 1ms, minimum 1ms)
        billed_duration = max(1, int(duration + 0.999))

        return {
            "StatusCode": 200,
            "Payload": BytesIO(b'{"result": "success"}'),
            "LogResult": self._generate_log_result(
                duration=duration,
                billed_duration=billed_duration,
                memory_size=memory_size,
                cold_start=is_cold_start,
            ),
        }

    def _generate_log_result(
        self,
        duration: float = None,
        billed_duration: int = None,
        memory_size: int = None,
        cold_start: bool = False,
        error: bool = False,
    ) -> str:
        """Generate mock CloudWatch log result."""
        if error:
            log_lines = [
                "START RequestId: 12345678-1234-1234-1234-123456789012 Version: $LATEST",
                "[ERROR] Runtime.UserCodeSyntaxError: Syntax error in module 'lambda_function'",
                "END RequestId: 12345678-1234-1234-1234-123456789012",
                "REPORT RequestId: 12345678-1234-1234-1234-123456789012\tDuration: 100.00 ms\tBilled Duration: 100 ms\tMemory Size: 256 MB\tMax Memory Used: 50 MB",
            ]
        else:
            log_lines = ["START RequestId: 12345678-1234-1234-1234-123456789012 Version: $LATEST"]

            if cold_start:
                log_lines.append("INIT_START Runtime Version: python:3.9.v16")
                log_lines.append("INIT_REPORT Duration: 250.00 ms")

            log_lines.extend(
                [
                    "Function executed successfully",
                    "END RequestId: 12345678-1234-1234-1234-123456789012",
                    f"REPORT RequestId: 12345678-1234-1234-1234-123456789012\t"
                    f"Duration: {duration:.2f} ms\t"
                    f"Billed Duration: {billed_duration} ms\t"
                    f"Memory Size: {memory_size} MB\t"
                    f"Max Memory Used: {memory_size // 2} MB",
                ]
            )

        return "\n".join(log_lines)

    def set_error_rate(self, rate: float):
        """Set the error rate for testing."""
        self.error_rate = max(0.0, min(1.0, rate))

    def set_cold_start_rate(self, rate: float):
        """Set the cold start rate for testing."""
        self.cold_start_rate = max(0.0, min(1.0, rate))

    def set_base_duration(self, duration: float):
        """Set the base duration for testing."""
        self.base_duration = max(1.0, duration)


class MockCloudWatchClient:
    """Mock CloudWatch client for testing."""

    def __init__(self):
        self.metrics = {}
        self.log_groups = {}

    def put_metric_data(self, Namespace: str, MetricData: List[Dict[str, Any]]):
        """Mock putting metric data."""
        if Namespace not in self.metrics:
            self.metrics[Namespace] = []

        self.metrics[Namespace].extend(MetricData)

    def get_metric_statistics(self, **kwargs) -> Dict[str, Any]:
        """Mock getting metric statistics."""
        # Return mock statistics
        start_time = kwargs.get("StartTime", datetime.utcnow() - timedelta(hours=1))
        end_time = kwargs.get("EndTime", datetime.utcnow())

        # Generate some mock data points
        datapoints = []
        current_time = start_time
        while current_time < end_time:
            datapoints.append(
                {
                    "Timestamp": current_time,
                    "Sum": random.uniform(100, 1000),
                    "Average": random.uniform(50, 500),
                    "Maximum": random.uniform(200, 2000),
                    "Minimum": random.uniform(10, 100),
                    "SampleCount": random.randint(10, 100),
                    "Unit": "Milliseconds",
                }
            )
            current_time += timedelta(minutes=5)

        return {"Label": kwargs.get("MetricName", "Duration"), "Datapoints": datapoints}


class MockCloudWatchLogsClient:
    """Mock CloudWatch Logs client for testing."""

    def __init__(self):
        self.log_groups = {}
        self.log_streams = {}

    def describe_log_groups(self, **kwargs) -> Dict[str, Any]:
        """Mock describing log groups."""
        log_group_name = kwargs.get("logGroupNamePrefix", "")

        groups = []
        for name in self.log_groups:
            if name.startswith(log_group_name):
                groups.append(
                    {
                        "logGroupName": name,
                        "creationTime": int(time.time() * 1000),
                        "retentionInDays": 30,
                        "metricFilterCount": 0,
                        "arn": f"arn:aws:logs:us-east-1:123456789012:log-group:{name}:*",
                        "storedBytes": 1024,
                    }
                )

        return {"logGroups": groups}

    def filter_log_events(self, **kwargs) -> Dict[str, Any]:
        """Mock filtering log events."""
        log_group_name = kwargs["logGroupName"]

        # Generate mock log events
        events = []
        for i in range(10):
            events.append(
                {
                    "logStreamName": f"2024/01/01/[$LATEST]{random.randint(1000, 9999)}",
                    "timestamp": int((datetime.utcnow() - timedelta(minutes=i)).timestamp() * 1000),
                    "message": f"REPORT RequestId: {random.randint(10000, 99999)}\t"
                    f"Duration: {random.uniform(50, 500):.2f} ms\t"
                    f"Billed Duration: {random.randint(100, 1000)} ms\t"
                    f"Memory Size: 256 MB\tMax Memory Used: {random.randint(50, 200)} MB",
                    "ingestionTime": int(datetime.utcnow().timestamp() * 1000),
                    "eventId": f"event_{i}",
                }
            )

        return {
            "events": events,
            "searchedLogStreams": [
                {"logStreamName": "2024/01/01/[$LATEST]123456789", "searchedCompletely": True}
            ],
        }


class MockAWSProvider:
    """Mock AWS provider for comprehensive testing."""

    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.lambda_client = MockLambdaClient(region_name)
        self.cloudwatch_client = MockCloudWatchClient()
        self.logs_client = MockCloudWatchLogsClient()

        # Create default test function
        self.lambda_client.create_function(
            FunctionName="test-function",
            Runtime="python3.9",
            Role="arn:aws:iam::123456789012:role/lambda-role",
            Handler="index.handler",
            MemorySize=256,
            Timeout=30,
        )

    def get_lambda_client(self):
        """Get mock Lambda client."""
        return self.lambda_client

    def get_cloudwatch_client(self):
        """Get mock CloudWatch client."""
        return self.cloudwatch_client

    def get_logs_client(self):
        """Get mock CloudWatch Logs client."""
        return self.logs_client

    def configure_function_behavior(self, function_name: str, **kwargs):
        """Configure mock function behavior."""
        if "error_rate" in kwargs:
            self.lambda_client.set_error_rate(kwargs["error_rate"])
        if "cold_start_rate" in kwargs:
            self.lambda_client.set_cold_start_rate(kwargs["cold_start_rate"])
        if "base_duration" in kwargs:
            self.lambda_client.set_base_duration(kwargs["base_duration"])

    def reset_invocation_count(self):
        """Reset invocation counter."""
        self.lambda_client.invocation_count = 0


def create_mock_boto3_session(mock_provider: MockAWSProvider):
    """Create a mock boto3 session that returns mock clients."""
    session = Mock()

    def mock_client(service_name, **kwargs):
        if service_name == "lambda":
            return mock_provider.get_lambda_client()
        elif service_name == "cloudwatch":
            return mock_provider.get_cloudwatch_client()
        elif service_name == "logs":
            return mock_provider.get_logs_client()
        else:
            return Mock()

    session.client = mock_client
    return session


def patch_boto3_with_mock(monkeypatch, mock_provider: MockAWSProvider):
    """Patch boto3 to use mock provider."""
    mock_session = create_mock_boto3_session(mock_provider)

    # Patch boto3.Session
    monkeypatch.setattr("boto3.Session", lambda **kwargs: mock_session)

    # Patch boto3.client for direct usage
    def mock_client(service_name, **kwargs):
        return mock_session.client(service_name, **kwargs)

    monkeypatch.setattr("boto3.client", mock_client)

    return mock_provider
