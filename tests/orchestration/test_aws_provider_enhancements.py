"""Integration tests for AWS provider enhancements."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
from moto import mock_aws

import boto3
from aws_lambda_tuner.providers.aws import AWSLambdaProvider


class TestAWSProviderEnhancements:
    """Test enhanced AWS provider functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock()
        config.region = "us-east-1"
        config.function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
        return config

    @pytest.fixture
    def provider(self, config):
        """Create AWS provider instance."""
        return AWSLambdaProvider(config)

    @mock_aws
    @pytest.mark.asyncio
    async def test_get_provisioned_concurrency_not_configured(self, provider):
        """Test getting provisioned concurrency when not configured."""
        # Mock the AWS client to return not found error
        with patch.object(provider.lambda_client, "get_provisioned_concurrency_config") as mock_get:
            from botocore.exceptions import ClientError

            mock_get.side_effect = ClientError(
                error_response={"Error": {"Code": "ProvisionedConcurrencyConfigNotFoundException"}},
                operation_name="GetProvisionedConcurrencyConfig",
            )

            result = await provider.get_provisioned_concurrency()
            assert result is None

    @mock_aws
    @pytest.mark.asyncio
    async def test_get_provisioned_concurrency_configured(self, provider):
        """Test getting provisioned concurrency when configured."""
        mock_response = {
            "AllocatedConcurrencyUnits": 10,
            "AvailableConcurrencyUnits": 8,
            "Status": "READY",
            "LastModified": datetime.utcnow(),
        }

        with patch.object(provider.lambda_client, "get_provisioned_concurrency_config") as mock_get:
            mock_get.return_value = mock_response

            result = await provider.get_provisioned_concurrency()

            assert result is not None
            assert result["allocated_concurrency"] == 10
            assert result["available_concurrency"] == 8
            assert result["status"] == "READY"

    @mock_aws
    @pytest.mark.asyncio
    async def test_set_provisioned_concurrency(self, provider):
        """Test setting provisioned concurrency."""
        mock_response = {
            "AllocatedConcurrencyUnits": 5,
            "Status": "IN_PROGRESS",
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }

        with patch.object(
            provider.lambda_client, "put_provisioned_concurrency_config"
        ) as mock_put, patch.object(
            provider, "_wait_for_provisioned_concurrency_ready"
        ) as mock_wait:

            mock_put.return_value = mock_response
            mock_wait.return_value = None

            result = await provider.set_provisioned_concurrency(5)

            assert result["allocated_concurrency"] == 5
            assert result["status"] == "IN_PROGRESS"
            assert result["request_id"] == "test-request-id"

            mock_put.assert_called_once()
            mock_wait.assert_called_once_with(5)

    @mock_aws
    @pytest.mark.asyncio
    async def test_delete_provisioned_concurrency(self, provider):
        """Test deleting provisioned concurrency."""
        with patch.object(
            provider.lambda_client, "delete_provisioned_concurrency_config"
        ) as mock_delete:
            mock_delete.return_value = {}

            result = await provider.delete_provisioned_concurrency()
            assert result is True
            mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_provisioned_concurrency_ready(self, provider):
        """Test waiting for provisioned concurrency to be ready."""
        # Mock sequence: IN_PROGRESS -> READY
        with patch.object(provider, "get_provisioned_concurrency") as mock_get:
            mock_get.side_effect = [
                {"status": "IN_PROGRESS", "allocated_concurrency": 5},
                {"status": "READY", "allocated_concurrency": 5},
            ]

            # Should complete without exception
            await provider._wait_for_provisioned_concurrency_ready(5)

            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_wait_for_provisioned_concurrency_failed(self, provider):
        """Test handling failed provisioned concurrency setup."""
        with patch.object(provider, "get_provisioned_concurrency") as mock_get:
            mock_get.return_value = {"status": "FAILED", "allocated_concurrency": 5}

            with pytest.raises(Exception, match="Provisioned concurrency setup failed"):
                await provider._wait_for_provisioned_concurrency_ready(5)

    @mock_aws
    @pytest.mark.asyncio
    async def test_cloudwatch_metrics_integration(self, provider):
        """Test CloudWatch metrics integration."""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        mock_datapoints = [
            {"Timestamp": start_time, "Sum": 100, "Average": 50},
            {"Timestamp": start_time + timedelta(minutes=30), "Sum": 120, "Average": 60},
        ]

        with patch.object(provider, "_get_cloudwatch_metric") as mock_get_metric:
            mock_get_metric.return_value = mock_datapoints

            result = await provider.cloudwatch_metrics_integration(start_time, end_time)

            assert "invocations" in result
            assert "duration" in result
            assert "errors" in result
            assert "analysis" in result

            # Should call _get_cloudwatch_metric multiple times for different metrics
            assert mock_get_metric.call_count >= 4

    @mock_aws
    @pytest.mark.asyncio
    async def test_get_cloudwatch_metric(self, provider):
        """Test getting specific CloudWatch metric."""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        mock_response = {
            "Datapoints": [
                {"Timestamp": start_time, "Sum": 100, "Average": 50, "Maximum": 80, "Minimum": 20},
                {
                    "Timestamp": start_time + timedelta(minutes=30),
                    "Sum": 120,
                    "Average": 60,
                    "Maximum": 90,
                    "Minimum": 30,
                },
            ]
        }

        with patch.object(provider.cloudwatch_client, "get_metric_statistics") as mock_get:
            mock_get.return_value = mock_response

            result = await provider._get_cloudwatch_metric(
                "AWS/Lambda", "Invocations", start_time, end_time
            )

            assert len(result) == 2
            assert result[0]["Sum"] == 100
            assert result[1]["Average"] == 60

    def test_analyze_cloudwatch_metrics(self, provider):
        """Test CloudWatch metrics analysis."""
        metrics_data = {
            "invocations": [{"Sum": 100}, {"Sum": 120}, {"Sum": 80}],
            "duration": [{"Average": 500}, {"Average": 520}, {"Average": 480}],
            "errors": [{"Sum": 5}, {"Sum": 2}, {"Sum": 3}],
            "throttles": [{"Sum": 0}, {"Sum": 1}, {"Sum": 0}],
            "concurrent_executions": [{"Average": 5}, {"Average": 8}, {"Average": 6}],
        }

        analysis = provider._analyze_cloudwatch_metrics(metrics_data)

        assert analysis["total_invocations"] == 300  # 100 + 120 + 80
        assert analysis["avg_duration"] == 500  # (500 + 520 + 480) / 3
        assert analysis["error_rate"] == 10 / 300  # 10 errors out of 300 invocations
        assert analysis["throttle_rate"] == 1 / 300  # 1 throttle out of 300 invocations
        assert analysis["avg_concurrency"] == 6.33  # (5 + 8 + 6) / 3, rounded
        assert analysis["peak_concurrency"] == 8

    @pytest.mark.asyncio
    async def test_get_historical_performance_data(self, provider):
        """Test getting historical performance data."""
        with patch.object(provider, "cloudwatch_metrics_integration") as mock_metrics:
            mock_metrics.return_value = {
                "invocations": [],
                "duration": [],
                "analysis": {"total_invocations": 1000},
            }

            result = await provider.get_historical_performance_data(7)

            assert "time_period" in result
            assert result["time_period"]["days"] == 7
            assert "start" in result["time_period"]
            assert "end" in result["time_period"]
            mock_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_traffic_simulation_steady(self, provider):
        """Test steady traffic simulation."""
        simulation_config = {
            "pattern": "steady",
            "duration_minutes": 0.1,  # Very short for testing
            "peak_rps": 5,
            "payload": {"test": "data"},
        }

        with patch.object(provider, "invoke_function") as mock_invoke:
            mock_invoke.return_value = {
                "duration": 500,
                "status_code": 200,
                "timestamp": datetime.utcnow(),
            }

            result = await provider.traffic_simulation(simulation_config)

            assert result["pattern"] == "steady"
            assert result["peak_rps"] == 5
            assert "executions" in result
            assert "metrics" in result

            # Should have made some invocations
            assert mock_invoke.call_count > 0

    @pytest.mark.asyncio
    async def test_traffic_simulation_burst(self, provider):
        """Test burst traffic simulation."""
        simulation_config = {
            "pattern": "burst",
            "duration_minutes": 0.1,  # Very short for testing
            "peak_rps": 10,
            "payload": {"test": "data"},
        }

        with patch.object(provider, "_execute_simulation_request") as mock_execute:
            mock_execute.return_value = {
                "duration": 600,
                "simulation_timestamp": datetime.utcnow().timestamp(),
                "pattern": "burst",
            }

            with patch("asyncio.sleep"):  # Speed up the test
                result = await provider.traffic_simulation(simulation_config)

            assert result["pattern"] == "burst"
            assert "executions" in result

    @pytest.mark.asyncio
    async def test_traffic_simulation_gradual_ramp(self, provider):
        """Test gradual ramp traffic simulation."""
        simulation_config = {
            "pattern": "gradual_ramp",
            "duration_minutes": 0.1,  # Very short for testing
            "peak_rps": 10,
            "payload": {"test": "data"},
        }

        with patch.object(provider, "invoke_function") as mock_invoke:
            mock_invoke.return_value = {
                "duration": 700,
                "simulation_timestamp": datetime.utcnow().timestamp(),
                "pattern": "gradual_ramp",
            }

            result = await provider.traffic_simulation(simulation_config)

            assert result["pattern"] == "gradual_ramp"
            assert "executions" in result

    @pytest.mark.asyncio
    async def test_traffic_simulation_spike(self, provider):
        """Test spike traffic simulation."""
        simulation_config = {
            "pattern": "spike",
            "duration_minutes": 0.1,  # Very short for testing
            "peak_rps": 15,
            "payload": {"test": "data"},
        }

        with patch.object(provider, "invoke_function") as mock_invoke, patch.object(
            provider, "_execute_simulation_request"
        ) as mock_execute:

            mock_invoke.return_value = {
                "duration": 800,
                "simulation_timestamp": datetime.utcnow().timestamp(),
                "pattern": "spike",
                "phase": "normal",
            }

            mock_execute.return_value = {
                "duration": 900,
                "simulation_timestamp": datetime.utcnow().timestamp(),
                "pattern": "spike",
            }

            with patch("asyncio.sleep"):  # Speed up the test
                result = await provider.traffic_simulation(simulation_config)

            assert result["pattern"] == "spike"
            assert "executions" in result

    def test_analyze_simulation_results(self, provider):
        """Test simulation results analysis."""
        executions = [
            {"duration": 500, "simulation_timestamp": 1000.0},
            {"duration": 520, "simulation_timestamp": 1001.0},
            {"duration": 480, "simulation_timestamp": 1002.0},
            {"error": "timeout", "simulation_timestamp": 1003.0},
            {"duration": 510, "simulation_timestamp": 1004.0},
        ]

        analysis = provider._analyze_simulation_results(executions)

        assert analysis["total_requests"] == 5
        assert analysis["successful_requests"] == 4
        assert analysis["failed_requests"] == 1
        assert analysis["success_rate"] == 0.8
        assert analysis["error_rate"] == 0.2
        assert analysis["avg_duration"] == 502.5  # (500 + 520 + 480 + 510) / 4
        assert analysis["min_duration"] == 480
        assert analysis["max_duration"] == 520

    @pytest.mark.asyncio
    async def test_run_comprehensive_load_test(self, provider):
        """Test comprehensive load test."""
        test_config = {
            "patterns": ["steady", "burst"],
            "base_rps": 5,
            "duration_per_pattern": 0.05,  # Very short for testing
            "payload": {"test": "data"},
        }

        with patch.object(provider, "traffic_simulation") as mock_simulation:
            mock_simulation.side_effect = [
                # Steady pattern results
                {
                    "pattern": "steady",
                    "metrics": {"success_rate": 0.95, "avg_duration": 500, "total_requests": 10},
                },
                # Burst pattern results
                {
                    "pattern": "burst",
                    "metrics": {"success_rate": 0.90, "avg_duration": 600, "total_requests": 12},
                },
            ]

            with patch("asyncio.sleep"):  # Speed up the test
                result = await provider.run_comprehensive_load_test(test_config)

            assert "pattern_results" in result
            assert "overall_analysis" in result
            assert "steady" in result["pattern_results"]
            assert "burst" in result["pattern_results"]

            # Should have called traffic_simulation for each pattern
            assert mock_simulation.call_count == 2

    def test_analyze_comprehensive_results(self, provider):
        """Test comprehensive results analysis."""
        pattern_results = {
            "steady": {
                "metrics": {"success_rate": 0.95, "avg_duration": 500, "total_requests": 10}
            },
            "burst": {"metrics": {"success_rate": 0.85, "avg_duration": 700, "total_requests": 8}},
            "spike": {"error": "Test error"},
        }

        analysis = provider._analyze_comprehensive_results(pattern_results)

        assert "pattern_performance" in analysis
        assert "best_performing_pattern" in analysis
        assert "worst_performing_pattern" in analysis
        assert "recommendations" in analysis

        # Should have performance data for steady and burst (not spike due to error)
        assert "steady" in analysis["pattern_performance"]
        assert "burst" in analysis["pattern_performance"]
        assert "spike" not in analysis["pattern_performance"]

        # Best performing should be steady (higher success rate and lower latency)
        assert analysis["best_performing_pattern"] == "steady"

    @pytest.mark.asyncio
    async def test_execute_simulation_request(self, provider):
        """Test executing a single simulation request."""
        payload = {"test": "data"}
        pattern = "test_pattern"

        with patch.object(provider, "invoke_function") as mock_invoke:
            mock_invoke.return_value = {"duration": 500, "status_code": 200}

            result = await provider._execute_simulation_request(payload, pattern)

            assert result["pattern"] == pattern
            assert result["duration"] == 500
            assert "simulation_timestamp" in result

    @pytest.mark.asyncio
    async def test_execute_simulation_request_error(self, provider):
        """Test executing simulation request with error."""
        payload = {"test": "data"}
        pattern = "test_pattern"

        with patch.object(provider, "invoke_function") as mock_invoke:
            mock_invoke.side_effect = Exception("Test error")

            result = await provider._execute_simulation_request(payload, pattern)

            assert result["pattern"] == pattern
            assert result["error"] == "Test error"
            assert "simulation_timestamp" in result

    @pytest.mark.asyncio
    async def test_simulate_steady_traffic(self, provider):
        """Test steady traffic simulation implementation."""
        end_time = datetime.utcnow().timestamp() + 0.1  # Very short duration
        rps = 5
        payload = {"test": "data"}
        results = {"executions": []}

        with patch.object(provider, "invoke_function") as mock_invoke:
            mock_invoke.return_value = {"duration": 500, "status_code": 200}

            await provider._simulate_steady_traffic(end_time, rps, payload, results)

            # Should have made some invocations
            assert len(results["executions"]) > 0

            # All executions should have steady pattern
            for execution in results["executions"]:
                assert execution["pattern"] == "steady"

    @pytest.mark.asyncio
    async def test_error_handling_in_cloudwatch_metrics(self, provider):
        """Test error handling in CloudWatch metrics integration."""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        with patch.object(provider, "_get_cloudwatch_metric") as mock_get_metric:
            mock_get_metric.side_effect = Exception("CloudWatch error")

            with pytest.raises(Exception, match="CloudWatch error"):
                await provider.cloudwatch_metrics_integration(start_time, end_time)

    @pytest.mark.asyncio
    async def test_traffic_simulation_invalid_pattern(self, provider):
        """Test traffic simulation with invalid pattern."""
        simulation_config = {
            "pattern": "invalid_pattern",
            "duration_minutes": 1,
            "peak_rps": 5,
            "payload": {},
        }

        with pytest.raises(ValueError, match="Unknown traffic pattern"):
            await provider.traffic_simulation(simulation_config)
