"""AWS Lambda provider for function tuning."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import random

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AWSLambdaProvider:
    """AWS Lambda provider for function tuning operations."""

    def __init__(self, config):
        self.config = config
        self.lambda_client = boto3.client("lambda", region_name=config.region)
        self.logs_client = boto3.client("logs", region_name=config.region)
        self.cloudwatch_client = boto3.client("cloudwatch", region_name=config.region)
        self._function_name = self._extract_function_name(config.function_arn)

    def _extract_function_name(self, function_arn: str) -> str:
        """Extract function name from ARN."""
        # ARN format: arn:aws:lambda:region:account:function:function-name
        return function_arn.split(":")[-1]

    async def get_function_configuration(self) -> Dict[str, Any]:
        """Get current function configuration."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self.lambda_client.get_function, {"FunctionName": self._function_name}
            )
            return response["Configuration"]
        except ClientError as e:
            logger.error(f"Failed to get function configuration: {e}")
            raise

    async def update_function_memory(self, memory_size: int) -> Dict[str, Any]:
        """Update function memory configuration."""
        try:
            logger.info(f"Updating function memory to {memory_size}MB")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.update_function_configuration,
                {"FunctionName": self._function_name, "MemorySize": memory_size},
            )

            # Wait for update to complete
            await self._wait_for_function_update()

            return response

        except ClientError as e:
            logger.error(f"Failed to update function memory: {e}")
            raise

    async def _wait_for_function_update(self, max_wait: int = 60):
        """Wait for function update to complete."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                config = await self.get_function_configuration()
                if config["LastUpdateStatus"] == "Successful":
                    return
                elif config["LastUpdateStatus"] == "Failed":
                    raise Exception(
                        f"Function update failed: {config.get('LastUpdateStatusReason', 'Unknown')}"
                    )
            except Exception as e:
                logger.warning(f"Error checking function status: {e}")

            await asyncio.sleep(2)

        raise TimeoutError("Function update timed out")

    async def invoke_function(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the Lambda function and collect performance metrics."""
        try:
            start_time = time.time()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.invoke,
                {
                    "FunctionName": self._function_name,
                    "Payload": json.dumps(payload),
                    "LogType": "Tail",
                },
            )

            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds

            # Parse response
            result = {
                "duration": duration,
                "status_code": response["StatusCode"],
                "timestamp": datetime.utcnow(),
                "request_id": response["ResponseMetadata"]["RequestId"],
            }

            # Check for errors
            if "FunctionError" in response:
                result["error"] = True
                result["error_type"] = response["FunctionError"]
                logger.warning(f"Function error: {response['FunctionError']}")
            else:
                result["error"] = False

            # Parse CloudWatch logs if available
            if "LogResult" in response:
                log_data = self._parse_log_result(response["LogResult"])
                result.update(log_data)

            return result

        except ClientError as e:
            logger.error(f"Failed to invoke function: {e}")
            raise

    def _parse_log_result(self, log_result: str) -> Dict[str, Any]:
        """Parse CloudWatch log result from Lambda response."""
        import base64

        try:
            # Decode base64 log data
            log_data = base64.b64decode(log_result).decode("utf-8")

            # Parse log lines
            lines = log_data.strip().split("\n")

            result = {
                "cold_start": False,
                "billed_duration": None,
                "memory_used": None,
                "max_memory_used": None,
            }

            for line in lines:
                # Check for cold start indicators
                if "INIT_START" in line or "Runtime.ImportModuleError" in line:
                    result["cold_start"] = True

                # Parse REPORT line for detailed metrics
                if line.startswith("REPORT"):
                    report_data = self._parse_report_line(line)
                    result.update(report_data)

            return result

        except Exception as e:
            logger.warning(f"Failed to parse log result: {e}")
            return {"cold_start": False}

    def _parse_report_line(self, report_line: str) -> Dict[str, Any]:
        """Parse Lambda REPORT log line."""
        result = {}

        try:
            # Example REPORT line:
            # REPORT RequestId: 12345 Duration: 1234.56 ms Billed Duration: 1300 ms
            # Memory Size: 512 MB Max Memory Used: 256 MB

            parts = report_line.split("\t")

            for part in parts:
                part = part.strip()

                if part.startswith("Duration:"):
                    # Extract duration in ms
                    duration_str = part.split(":")[1].strip().replace("ms", "").strip()
                    result["actual_duration"] = float(duration_str)

                elif part.startswith("Billed Duration:"):
                    # Extract billed duration in ms
                    duration_str = part.split(":")[1].strip().replace("ms", "").strip()
                    result["billed_duration"] = float(duration_str)

                elif part.startswith("Memory Size:"):
                    # Extract memory size in MB
                    memory_str = part.split(":")[1].strip().replace("MB", "").strip()
                    result["memory_size"] = int(memory_str)

                elif part.startswith("Max Memory Used:"):
                    # Extract max memory used in MB
                    memory_str = part.split(":")[1].strip().replace("MB", "").strip()
                    result["memory_used"] = int(memory_str)

        except Exception as e:
            logger.warning(f"Failed to parse REPORT line: {e}")

        return result

    # Provisioned Concurrency Operations

    async def get_provisioned_concurrency(self) -> Optional[Dict[str, Any]]:
        """Get current provisioned concurrency configuration."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.get_provisioned_concurrency_config,
                {"FunctionName": self._function_name},
            )

            return {
                "allocated_concurrency": response.get("AllocatedConcurrencyUnits", 0),
                "available_concurrency": response.get("AvailableConcurrencyUnits", 0),
                "status": response.get("Status", "Unknown"),
                "last_modified": response.get("LastModified"),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "ProvisionedConcurrencyConfigNotFoundException":
                logger.info("No provisioned concurrency configured")
                return None
            else:
                logger.error(f"Failed to get provisioned concurrency config: {e}")
                raise

    async def set_provisioned_concurrency(self, concurrency_units: int) -> Dict[str, Any]:
        """Set provisioned concurrency for the function."""
        try:
            logger.info(f"Setting provisioned concurrency to {concurrency_units} units")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.put_provisioned_concurrency_config,
                {
                    "FunctionName": self._function_name,
                    "ProvisionedConcurrencyUnits": concurrency_units,
                },
            )

            # Wait for provisioned concurrency to be ready
            await self._wait_for_provisioned_concurrency_ready(concurrency_units)

            return {
                "allocated_concurrency": response.get("AllocatedConcurrencyUnits", 0),
                "status": response.get("Status", "Unknown"),
                "request_id": response["ResponseMetadata"]["RequestId"],
            }

        except ClientError as e:
            logger.error(f"Failed to set provisioned concurrency: {e}")
            raise

    async def delete_provisioned_concurrency(self) -> bool:
        """Delete provisioned concurrency configuration."""
        try:
            logger.info("Deleting provisioned concurrency configuration")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.lambda_client.delete_provisioned_concurrency_config,
                {"FunctionName": self._function_name},
            )

            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ProvisionedConcurrencyConfigNotFoundException":
                logger.info("No provisioned concurrency to delete")
                return True
            else:
                logger.error(f"Failed to delete provisioned concurrency: {e}")
                raise

    async def _wait_for_provisioned_concurrency_ready(
        self, expected_units: int, max_wait: int = 300
    ):
        """Wait for provisioned concurrency to be ready."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                config = await self.get_provisioned_concurrency()
                if (
                    config
                    and config["status"] == "READY"
                    and config["allocated_concurrency"] == expected_units
                ):
                    logger.info("Provisioned concurrency is ready")
                    return
                elif config and config["status"] == "FAILED":
                    raise Exception(f"Provisioned concurrency setup failed")

                logger.info(
                    f"Waiting for provisioned concurrency... Status: {config['status'] if config else 'None'}"
                )
                await asyncio.sleep(10)

            except Exception as e:
                logger.warning(f"Error checking provisioned concurrency status: {e}")
                await asyncio.sleep(5)

        raise TimeoutError("Provisioned concurrency setup timed out")

    # CloudWatch Metrics Integration

    async def cloudwatch_metrics_integration(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get CloudWatch metrics for the function over a specified time period.

        Args:
            start_time: Start time for metrics collection
            end_time: End time for metrics collection

        Returns:
            Dictionary containing CloudWatch metrics
        """
        logger.info(f"Fetching CloudWatch metrics from {start_time} to {end_time}")

        metrics_data = {
            "invocations": [],
            "duration": [],
            "errors": [],
            "throttles": [],
            "concurrent_executions": [],
            "cold_starts": [],
            "provisioned_concurrency_utilization": [],
            "memory_utilization": [],
        }

        try:
            # Get invocation metrics
            invocations = await self._get_cloudwatch_metric(
                "AWS/Lambda", "Invocations", start_time, end_time
            )
            metrics_data["invocations"] = invocations

            # Get duration metrics
            duration = await self._get_cloudwatch_metric(
                "AWS/Lambda", "Duration", start_time, end_time
            )
            metrics_data["duration"] = duration

            # Get error metrics
            errors = await self._get_cloudwatch_metric("AWS/Lambda", "Errors", start_time, end_time)
            metrics_data["errors"] = errors

            # Get throttle metrics
            throttles = await self._get_cloudwatch_metric(
                "AWS/Lambda", "Throttles", start_time, end_time
            )
            metrics_data["throttles"] = throttles

            # Get concurrent execution metrics
            concurrent = await self._get_cloudwatch_metric(
                "AWS/Lambda", "ConcurrentExecutions", start_time, end_time
            )
            metrics_data["concurrent_executions"] = concurrent

            # Get provisioned concurrency utilization if available
            pc_utilization = await self._get_cloudwatch_metric(
                "AWS/Lambda", "ProvisionedConcurrencyUtilization", start_time, end_time
            )
            metrics_data["provisioned_concurrency_utilization"] = pc_utilization

            # Calculate derived metrics
            metrics_data["analysis"] = self._analyze_cloudwatch_metrics(metrics_data)

            return metrics_data

        except Exception as e:
            logger.error(f"Failed to fetch CloudWatch metrics: {e}")
            raise

    async def _get_cloudwatch_metric(
        self, namespace: str, metric_name: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get specific CloudWatch metric data."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.cloudwatch_client.get_metric_statistics,
                {
                    "Namespace": namespace,
                    "MetricName": metric_name,
                    "Dimensions": [{"Name": "FunctionName", "Value": self._function_name}],
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "Period": 60,  # 1 minute periods
                    "Statistics": ["Sum", "Average", "Maximum", "Minimum"],
                },
            )

            return response.get("Datapoints", [])

        except ClientError as e:
            logger.warning(f"Failed to get metric {metric_name}: {e}")
            return []

    def _analyze_cloudwatch_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CloudWatch metrics and generate insights."""
        analysis = {
            "total_invocations": 0,
            "avg_duration": 0,
            "error_rate": 0,
            "throttle_rate": 0,
            "peak_concurrency": 0,
            "avg_concurrency": 0,
            "performance_trends": {},
            "cost_analysis": {},
        }

        # Calculate total invocations
        if metrics_data["invocations"]:
            analysis["total_invocations"] = sum(
                dp.get("Sum", 0) for dp in metrics_data["invocations"]
            )

        # Calculate average duration
        if metrics_data["duration"]:
            durations = [
                dp.get("Average", 0) for dp in metrics_data["duration"] if dp.get("Average", 0) > 0
            ]
            analysis["avg_duration"] = sum(durations) / len(durations) if durations else 0

        # Calculate error rate
        if metrics_data["errors"] and analysis["total_invocations"] > 0:
            total_errors = sum(dp.get("Sum", 0) for dp in metrics_data["errors"])
            analysis["error_rate"] = total_errors / analysis["total_invocations"]

        # Calculate throttle rate
        if metrics_data["throttles"] and analysis["total_invocations"] > 0:
            total_throttles = sum(dp.get("Sum", 0) for dp in metrics_data["throttles"])
            analysis["throttle_rate"] = total_throttles / analysis["total_invocations"]

        # Calculate concurrency metrics
        if metrics_data["concurrent_executions"]:
            concurrency_values = [
                dp.get("Average", 0)
                for dp in metrics_data["concurrent_executions"]
                if dp.get("Average", 0) > 0
            ]
            if concurrency_values:
                analysis["avg_concurrency"] = sum(concurrency_values) / len(concurrency_values)
                analysis["peak_concurrency"] = max(concurrency_values)

        # Analyze provisioned concurrency utilization
        if metrics_data["provisioned_concurrency_utilization"]:
            utilization_values = [
                dp.get("Average", 0)
                for dp in metrics_data["provisioned_concurrency_utilization"]
                if dp.get("Average", 0) > 0
            ]
            if utilization_values:
                analysis["avg_pc_utilization"] = sum(utilization_values) / len(utilization_values)
                analysis["peak_pc_utilization"] = max(utilization_values)

        return analysis

    async def get_historical_performance_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Get historical performance data for trend analysis."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)

        logger.info(f"Fetching {days_back} days of historical performance data")

        historical_data = await self.cloudwatch_metrics_integration(start_time, end_time)
        historical_data["time_period"] = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "days": days_back,
        }

        return historical_data

    # Traffic Simulation Capabilities

    async def traffic_simulation(self, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate different traffic patterns for testing.

        Args:
            simulation_config: Configuration for traffic simulation
                - pattern: 'steady', 'burst', 'gradual_ramp', 'spike'
                - duration_minutes: Total simulation duration
                - peak_rps: Peak requests per second
                - payload: Request payload

        Returns:
            Dictionary containing simulation results
        """
        pattern = simulation_config.get("pattern", "steady")
        duration_minutes = simulation_config.get("duration_minutes", 5)
        peak_rps = simulation_config.get("peak_rps", 10)
        payload = simulation_config.get("payload", {})

        logger.info(
            f"Starting traffic simulation: {pattern} pattern for {duration_minutes} minutes"
        )

        simulation_results = {
            "pattern": pattern,
            "duration_minutes": duration_minutes,
            "peak_rps": peak_rps,
            "start_time": datetime.utcnow().isoformat(),
            "executions": [],
            "metrics": {},
        }

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        try:
            if pattern == "steady":
                await self._simulate_steady_traffic(end_time, peak_rps, payload, simulation_results)
            elif pattern == "burst":
                await self._simulate_burst_traffic(end_time, peak_rps, payload, simulation_results)
            elif pattern == "gradual_ramp":
                await self._simulate_gradual_ramp_traffic(
                    end_time, peak_rps, payload, simulation_results
                )
            elif pattern == "spike":
                await self._simulate_spike_traffic(end_time, peak_rps, payload, simulation_results)
            else:
                raise ValueError(f"Unknown traffic pattern: {pattern}")

            simulation_results["end_time"] = datetime.utcnow().isoformat()
            simulation_results["actual_duration"] = time.time() - start_time
            simulation_results["metrics"] = self._analyze_simulation_results(
                simulation_results["executions"]
            )

            return simulation_results

        except Exception as e:
            logger.error(f"Traffic simulation failed: {e}")
            simulation_results["error"] = str(e)
            raise

    async def _simulate_steady_traffic(
        self, end_time: float, rps: int, payload: Dict[str, Any], results: Dict[str, Any]
    ):
        """Simulate steady traffic pattern."""
        interval = 1.0 / rps  # Time between requests

        while time.time() < end_time:
            request_start = time.time()

            try:
                result = await self.invoke_function(payload)
                result["simulation_timestamp"] = time.time()
                result["pattern"] = "steady"
                results["executions"].append(result)

            except Exception as e:
                logger.warning(f"Simulation request failed: {e}")
                results["executions"].append(
                    {"error": str(e), "simulation_timestamp": time.time(), "pattern": "steady"}
                )

            # Maintain steady rate
            elapsed = time.time() - request_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _simulate_burst_traffic(
        self, end_time: float, peak_rps: int, payload: Dict[str, Any], results: Dict[str, Any]
    ):
        """Simulate burst traffic pattern."""
        burst_duration = 30  # 30 seconds of burst
        quiet_duration = 60  # 60 seconds of quiet

        while time.time() < end_time:
            # Burst period
            logger.info("Starting burst period...")
            burst_end = min(time.time() + burst_duration, end_time)

            while time.time() < burst_end:
                # Send multiple concurrent requests during burst
                tasks = []
                for _ in range(min(peak_rps, 10)):  # Limit concurrency
                    task = asyncio.create_task(self._execute_simulation_request(payload, "burst"))
                    tasks.append(task)

                # Wait for all requests to complete
                request_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in request_results:
                    if isinstance(result, Exception):
                        results["executions"].append(
                            {
                                "error": str(result),
                                "simulation_timestamp": time.time(),
                                "pattern": "burst",
                            }
                        )
                    else:
                        results["executions"].append(result)

                await asyncio.sleep(1)  # Brief pause between bursts

            # Quiet period
            if time.time() < end_time:
                logger.info("Starting quiet period...")
                await asyncio.sleep(min(quiet_duration, end_time - time.time()))

    async def _simulate_gradual_ramp_traffic(
        self, end_time: float, peak_rps: int, payload: Dict[str, Any], results: Dict[str, Any]
    ):
        """Simulate gradual ramp-up traffic pattern."""
        total_duration = end_time - time.time()
        ramp_steps = 10
        step_duration = total_duration / ramp_steps

        for step in range(ramp_steps):
            if time.time() >= end_time:
                break

            # Calculate current RPS (linear ramp)
            current_rps = int((step + 1) * peak_rps / ramp_steps)
            step_end = min(time.time() + step_duration, end_time)

            logger.info(f"Ramp step {step + 1}/{ramp_steps}: {current_rps} RPS")

            while time.time() < step_end:
                request_start = time.time()

                try:
                    result = await self.invoke_function(payload)
                    result["simulation_timestamp"] = time.time()
                    result["pattern"] = "gradual_ramp"
                    result["ramp_step"] = step + 1
                    result["rps_at_step"] = current_rps
                    results["executions"].append(result)

                except Exception as e:
                    results["executions"].append(
                        {
                            "error": str(e),
                            "simulation_timestamp": time.time(),
                            "pattern": "gradual_ramp",
                            "ramp_step": step + 1,
                        }
                    )

                # Maintain current rate
                if current_rps > 0:
                    interval = 1.0 / current_rps
                    elapsed = time.time() - request_start
                    sleep_time = max(0, interval - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

    async def _simulate_spike_traffic(
        self, end_time: float, peak_rps: int, payload: Dict[str, Any], results: Dict[str, Any]
    ):
        """Simulate spike traffic pattern."""
        spike_duration = 10  # 10 seconds of spike
        normal_rps = max(1, peak_rps // 4)  # Normal traffic is 25% of peak

        while time.time() < end_time:
            # Normal traffic period
            normal_duration = random.randint(60, 120)  # 1-2 minutes of normal traffic
            normal_end = min(time.time() + normal_duration, end_time)

            logger.info(f"Normal traffic period: {normal_rps} RPS")
            while time.time() < normal_end:
                try:
                    result = await self.invoke_function(payload)
                    result["simulation_timestamp"] = time.time()
                    result["pattern"] = "spike"
                    result["phase"] = "normal"
                    results["executions"].append(result)

                except Exception as e:
                    results["executions"].append(
                        {
                            "error": str(e),
                            "simulation_timestamp": time.time(),
                            "pattern": "spike",
                            "phase": "normal",
                        }
                    )

                await asyncio.sleep(1.0 / normal_rps)

            # Spike period
            if time.time() < end_time:
                spike_end = min(time.time() + spike_duration, end_time)
                logger.info(f"Traffic spike: {peak_rps} RPS")

                while time.time() < spike_end:
                    # Send concurrent requests during spike
                    tasks = []
                    for _ in range(min(peak_rps, 20)):
                        task = asyncio.create_task(
                            self._execute_simulation_request(payload, "spike")
                        )
                        tasks.append(task)

                    request_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in request_results:
                        if isinstance(result, Exception):
                            results["executions"].append(
                                {
                                    "error": str(result),
                                    "simulation_timestamp": time.time(),
                                    "pattern": "spike",
                                    "phase": "spike",
                                }
                            )
                        else:
                            result["phase"] = "spike"
                            results["executions"].append(result)

                    await asyncio.sleep(1)

    async def _execute_simulation_request(
        self, payload: Dict[str, Any], pattern: str
    ) -> Dict[str, Any]:
        """Execute a single simulation request."""
        try:
            result = await self.invoke_function(payload)
            result["simulation_timestamp"] = time.time()
            result["pattern"] = pattern
            return result
        except Exception as e:
            return {"error": str(e), "simulation_timestamp": time.time(), "pattern": pattern}

    def _analyze_simulation_results(self, executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze traffic simulation results."""
        if not executions:
            return {}

        successful_executions = [e for e in executions if not e.get("error")]
        failed_executions = [e for e in executions if e.get("error")]

        analysis = {
            "total_requests": len(executions),
            "successful_requests": len(successful_executions),
            "failed_requests": len(failed_executions),
            "success_rate": len(successful_executions) / len(executions) if executions else 0,
            "error_rate": len(failed_executions) / len(executions) if executions else 0,
        }

        if successful_executions:
            durations = [e.get("duration", 0) for e in successful_executions]
            analysis.update(
                {
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p95_duration": (
                        sorted(durations)[int(len(durations) * 0.95)] if durations else 0
                    ),
                    "p99_duration": (
                        sorted(durations)[int(len(durations) * 0.99)] if durations else 0
                    ),
                }
            )

            # Calculate throughput over time
            if len(successful_executions) > 1:
                timestamps = [e.get("simulation_timestamp", 0) for e in successful_executions]
                time_span = max(timestamps) - min(timestamps)
                analysis["actual_rps"] = (
                    len(successful_executions) / time_span if time_span > 0 else 0
                )

        return analysis

    async def run_comprehensive_load_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a comprehensive load test with multiple traffic patterns."""
        logger.info("Starting comprehensive load test")

        patterns = test_config.get("patterns", ["steady", "burst", "gradual_ramp", "spike"])
        base_rps = test_config.get("base_rps", 5)
        duration_per_pattern = test_config.get("duration_per_pattern", 3)
        payload = test_config.get("payload", {})

        comprehensive_results = {
            "test_started": datetime.utcnow().isoformat(),
            "patterns_tested": patterns,
            "base_rps": base_rps,
            "duration_per_pattern": duration_per_pattern,
            "pattern_results": {},
            "overall_analysis": {},
        }

        for pattern in patterns:
            logger.info(f"Testing pattern: {pattern}")

            simulation_config = {
                "pattern": pattern,
                "duration_minutes": duration_per_pattern,
                "peak_rps": base_rps * (2 if pattern in ["burst", "spike"] else 1),
                "payload": payload,
            }

            try:
                pattern_result = await self.traffic_simulation(simulation_config)
                comprehensive_results["pattern_results"][pattern] = pattern_result

                # Brief pause between patterns
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Pattern {pattern} failed: {e}")
                comprehensive_results["pattern_results"][pattern] = {"error": str(e)}

        # Analyze overall results
        comprehensive_results["overall_analysis"] = self._analyze_comprehensive_results(
            comprehensive_results["pattern_results"]
        )
        comprehensive_results["test_completed"] = datetime.utcnow().isoformat()

        return comprehensive_results

    def _analyze_comprehensive_results(self, pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results across all traffic patterns."""
        analysis = {
            "pattern_performance": {},
            "best_performing_pattern": None,
            "worst_performing_pattern": None,
            "performance_consistency": 0.0,
            "recommendations": [],
        }

        pattern_scores = {}

        for pattern, results in pattern_results.items():
            if "error" in results:
                continue

            metrics = results.get("metrics", {})
            if not metrics:
                continue

            # Calculate performance score for this pattern
            success_rate = metrics.get("success_rate", 0)
            avg_duration = metrics.get("avg_duration", float("inf"))

            # Simple scoring: prioritize success rate and low latency
            score = success_rate * (1000 / (avg_duration + 1))  # Higher is better
            pattern_scores[pattern] = score

            analysis["pattern_performance"][pattern] = {
                "score": score,
                "success_rate": success_rate,
                "avg_duration": avg_duration,
                "total_requests": metrics.get("total_requests", 0),
            }

        if pattern_scores:
            analysis["best_performing_pattern"] = max(
                pattern_scores.keys(), key=lambda k: pattern_scores[k]
            )
            analysis["worst_performing_pattern"] = min(
                pattern_scores.keys(), key=lambda k: pattern_scores[k]
            )

            # Calculate performance consistency (lower variance is better)
            scores = list(pattern_scores.values())
            if len(scores) > 1:
                avg_score = sum(scores) / len(scores)
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                analysis["performance_consistency"] = 1 / (1 + variance)  # Normalize to 0-1

        # Generate recommendations
        if analysis["best_performing_pattern"]:
            analysis["recommendations"].append(
                f"Function performs best under {analysis['best_performing_pattern']} traffic patterns"
            )

        if analysis["performance_consistency"] < 0.7:
            analysis["recommendations"].append(
                "Performance varies significantly across traffic patterns - consider optimization"
            )

        return analysis
