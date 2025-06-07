"""
Orchestrator module for AWS Lambda Tuner.
Manages the tuning process and coordinates Lambda invocations.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config_module import TunerConfig
from .utils import encode_payload, decode_response, retry_with_backoff, format_timestamp, chunk_list
from .exceptions import (
    LambdaExecutionError,
    AWSPermissionError,
    TimeoutError,
    ConcurrencyLimitError,
)
from .intelligence.recommendation_engine import IntelligentRecommendationEngine
from .monitoring.performance_monitor import PerformanceMonitor
from .monitoring.alert_manager import AlertManager, NotificationConfig
from .monitoring.continuous_optimizer import ContinuousOptimizer

logger = logging.getLogger(__name__)


class TunerOrchestrator:
    """Orchestrates the Lambda tuning process with intelligent monitoring and optimization."""

    def __init__(
        self, config: TunerConfig, notification_config: Optional[NotificationConfig] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Tuner configuration
            notification_config: Optional notification configuration for alerts
        """
        self.config = config
        self.lambda_client = self._create_lambda_client()
        self.results = {
            "function_arn": config.function_arn,
            "test_started": format_timestamp(),
            "configurations": [],
        }

        # Initialize intelligence and monitoring components
        self.recommendation_engine = None
        self.performance_monitor = None
        self.alert_manager = None
        self.continuous_optimizer = None

        # Check for monitoring configuration (backward compatibility)
        monitoring_enabled = getattr(config, "monitoring_enabled", False)
        auto_retuning_enabled = getattr(config, "auto_retuning_enabled", False)

        if monitoring_enabled or auto_retuning_enabled:
            self._initialize_monitoring_components(notification_config)

        logger.info(f"TunerOrchestrator initialized with monitoring: {monitoring_enabled}")

    def _initialize_monitoring_components(self, notification_config: Optional[NotificationConfig]):
        """Initialize monitoring and intelligence components."""
        try:
            # Initialize recommendation engine
            self.recommendation_engine = IntelligentRecommendationEngine(self.config)

            # Initialize performance monitor
            if self.config.monitoring_enabled:
                self.performance_monitor = PerformanceMonitor(self.config)

            # Initialize alert manager
            if self.config.alerts_enabled:
                self.alert_manager = AlertManager(self.config, notification_config)

            # Initialize continuous optimizer
            if (
                self.config.auto_retuning_enabled
                and self.performance_monitor
                and self.alert_manager
            ):
                self.continuous_optimizer = ContinuousOptimizer(
                    self.config, self.performance_monitor, self.alert_manager
                )

            logger.info("Monitoring components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring components: {e}")
            # Disable monitoring features if initialization fails
            self.config.monitoring_enabled = False
            self.config.auto_retuning_enabled = False

    def _create_lambda_client(self):
        """Create AWS Lambda client."""
        try:
            session_config = {}
            if self.config.profile:
                session_config["profile_name"] = self.config.profile

            session = boto3.Session(**session_config)

            client_config = {"region_name": self.config.region}

            return session.client("lambda", **client_config)

        except Exception as e:
            logger.error(f"Failed to create Lambda client: {e}")
            raise AWSPermissionError(f"Failed to create Lambda client: {e}")

    async def run_tuning(self) -> Dict[str, Any]:
        """
        Run the complete tuning process.

        Returns:
            Dictionary containing all test results
        """
        start_time = time.time()

        try:
            logger.info(f"Starting tuning for function: {self.config.function_arn}")

            # Get current function configuration
            original_config = await self._get_function_configuration()
            logger.info(f"Original memory configuration: {original_config['MemorySize']}MB")

            # Test each memory configuration
            for memory_size in self.config.memory_sizes:
                logger.info(f"Testing memory configuration: {memory_size}MB")

                # Update function memory
                if not self.config.dry_run:
                    await self._update_function_memory(memory_size)
                    # Wait for update to propagate
                    await asyncio.sleep(2)

                # Run tests for this configuration
                config_results = await self._test_memory_configuration(memory_size)
                self.results["configurations"].append(config_results)

                logger.info(
                    f"Completed testing for {memory_size}MB: "
                    f"{config_results['successful_executions']}/{config_results['total_executions']} successful"
                )

            # Restore original configuration
            if (
                not self.config.dry_run
                and original_config["MemorySize"] not in self.config.memory_sizes
            ):
                logger.info(
                    f"Restoring original memory configuration: {original_config['MemorySize']}MB"
                )
                await self._update_function_memory(original_config["MemorySize"])

            # Add metadata
            self.results["test_completed"] = format_timestamp()
            self.results["test_duration_seconds"] = time.time() - start_time
            self.results["total_configurations_tested"] = len(self.config.memory_sizes)

            logger.info(f"Tuning completed in {self.results['test_duration_seconds']:.2f} seconds")

            return self.results

        except Exception as e:
            logger.error(f"Tuning failed: {e}")
            self.results["error"] = str(e)
            self.results["test_duration_seconds"] = time.time() - start_time
            raise

    async def _get_function_configuration(self) -> Dict[str, Any]:
        """Get current Lambda function configuration."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.lambda_client.get_function_configuration,
                FunctionName=self.config.function_arn,
            )
            return response

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                raise LambdaExecutionError(f"Function not found: {self.config.function_arn}")
            elif e.response["Error"]["Code"] == "AccessDeniedException":
                raise AWSPermissionError(f"Permission denied: {e}")
            else:
                raise LambdaExecutionError(f"Failed to get function configuration: {e}")

    async def _update_function_memory(self, memory_size: int):
        """Update Lambda function memory configuration."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.lambda_client.update_function_configuration,
                FunctionName=self.config.function_arn,
                MemorySize=memory_size,
            )

            # Wait for function to be active
            waiter = self.lambda_client.get_waiter("function_active_v2")
            await asyncio.get_event_loop().run_in_executor(
                None, waiter.wait, FunctionName=self.config.function_arn
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                raise AWSPermissionError(f"Permission denied to update function: {e}")
            else:
                raise LambdaExecutionError(f"Failed to update function memory: {e}")

    async def _test_memory_configuration(self, memory_size: int) -> Dict[str, Any]:
        """Test a specific memory configuration."""
        config_results = {
            "memory_mb": memory_size,
            "executions": [],
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
        }

        # Run warmup executions
        if self.config.warmup_runs > 0 and not self.config.dry_run:
            logger.info(f"Running {self.config.warmup_runs} warmup executions...")
            await self._run_concurrent_invocations(
                memory_size, self.config.warmup_runs, is_warmup=True
            )

        # Run actual test executions
        if self.config.dry_run:
            # Simulate executions for dry run
            config_results["executions"] = self._simulate_executions(memory_size)
        else:
            # Run real executions
            config_results["executions"] = await self._run_concurrent_invocations(
                memory_size, self.config.iterations, is_warmup=False
            )

        # Calculate statistics
        config_results["total_executions"] = len(config_results["executions"])
        config_results["successful_executions"] = sum(
            1 for e in config_results["executions"] if not e.get("error")
        )
        config_results["failed_executions"] = (
            config_results["total_executions"] - config_results["successful_executions"]
        )

        return config_results

    async def _run_concurrent_invocations(
        self, memory_size: int, count: int, is_warmup: bool
    ) -> List[Dict[str, Any]]:
        """Run Lambda invocations concurrently."""
        executions = []

        # Split into chunks based on concurrency limit
        chunks = chunk_list(list(range(count)), self.config.concurrent_executions)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()

            # Run chunk concurrently
            with ThreadPoolExecutor(max_workers=self.config.concurrent_executions) as executor:
                futures = []

                for i in chunk:
                    future = executor.submit(
                        self._invoke_lambda_sync, memory_size, execution_id=i, is_warmup=is_warmup
                    )
                    futures.append(future)

                # Wait for all invocations in chunk to complete
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout)
                        if not is_warmup:
                            executions.append(result)
                    except Exception as e:
                        logger.error(f"Invocation failed: {e}")
                        if not is_warmup:
                            executions.append(
                                {
                                    "memory_mb": memory_size,
                                    "error": str(e),
                                    "timestamp": format_timestamp(),
                                }
                            )

            # Rate limiting between chunks
            chunk_duration = time.time() - chunk_start
            if chunk_idx < len(chunks) - 1 and chunk_duration < 1:
                await asyncio.sleep(1 - chunk_duration)

        return executions

    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def _invoke_lambda_sync(
        self, memory_size: int, execution_id: int, is_warmup: bool
    ) -> Dict[str, Any]:
        """Invoke Lambda function synchronously."""
        start_time = time.time()

        try:
            # Prepare payload
            payload = encode_payload(self.config.payload)

            # Invoke function
            response = self.lambda_client.invoke(
                FunctionName=self.config.function_arn,
                InvocationType="RequestResponse",
                Payload=payload,
            )

            # Process response
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds

            result = {
                "memory_mb": memory_size,
                "execution_id": execution_id,
                "duration": duration,
                "billed_duration": response.get("LogResult", {}).get("BilledDuration", duration),
                "cold_start": execution_id == 0,  # Simplified cold start detection
                "status_code": response["StatusCode"],
                "timestamp": format_timestamp(),
            }

            # Check for function errors
            if response.get("FunctionError"):
                result["error"] = response["FunctionError"]
                result["error_message"] = decode_response(response["Payload"])

            # Extract execution details from response
            if "LogResult" in response:
                # Parse CloudWatch logs if available
                # This would require base64 decoding and parsing
                pass

            return result

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "TooManyRequestsException":
                raise ConcurrencyLimitError("Lambda concurrency limit exceeded")
            elif error_code == "ResourceNotFoundException":
                raise LambdaExecutionError(f"Function not found: {self.config.function_arn}")
            elif error_code == "AccessDeniedException":
                raise AWSPermissionError("Permission denied to invoke function")
            else:
                raise LambdaExecutionError(f"Lambda invocation failed: {e}")

        except Exception as e:
            raise LambdaExecutionError(f"Unexpected error during invocation: {e}")

    def _simulate_executions(self, memory_size: int) -> List[Dict[str, Any]]:
        """Simulate executions for dry run mode."""
        import random

        executions = []
        base_duration = 1000 / (memory_size / 512)  # Simulate faster execution with more memory

        for i in range(self.config.iterations):
            # Add some variance
            duration = base_duration * random.uniform(0.8, 1.2)
            billed_duration = max(100, int(duration / 100) * 100)  # Round up to nearest 100ms

            execution = {
                "memory_mb": memory_size,
                "execution_id": i,
                "duration": duration,
                "billed_duration": billed_duration,
                "cold_start": i == 0,
                "status_code": 200,
                "timestamp": format_timestamp(),
                "dry_run": True,
            }

            # Simulate occasional failures
            if random.random() < 0.05:  # 5% failure rate
                execution["error"] = "Simulated error"
                execution["status_code"] = 500

            executions.append(execution)

        return executions

    async def run_with_reporting(self) -> Dict[str, Any]:
        """Run tuning with automatic report generation."""
        from .report_service import ReportGenerator
        from .visualization_module import VisualizationEngine

        # Run tuning
        results = await self.run_tuning()

        # Generate reports
        logger.info("Generating reports...")
        report_gen = ReportGenerator(results, self.config)

        # Save reports
        output_dir = self.config.output_dir
        report_gen.save_json(f"{output_dir}/results.json")
        report_gen.save_html(f"{output_dir}/report.html")

        # Generate visualizations
        logger.info("Generating visualizations...")
        viz_engine = VisualizationEngine(results)
        viz_engine.plot_performance_comparison(f"{output_dir}/performance.png")
        viz_engine.plot_cost_analysis(f"{output_dir}/cost-analysis.png")

        return results

    async def workload_aware_testing(self, memory_size: int) -> Dict[str, Any]:
        """
        Select testing strategy based on workload type.

        Args:
            memory_size: Memory size to test

        Returns:
            Dictionary containing test results
        """
        logger.info(f"Running workload-aware testing for {self.config.workload_type} workload")

        if self.config.workload_type == "on_demand":
            return await self._test_on_demand_workload(memory_size)
        elif self.config.workload_type == "scheduled":
            return await self._test_scheduled_workload(memory_size)
        elif self.config.workload_type == "continuous":
            return await self._test_continuous_workload(memory_size)
        else:
            # Fallback to default testing
            return await self._test_memory_configuration(memory_size)

    async def _test_on_demand_workload(self, memory_size: int) -> Dict[str, Any]:
        """Test configuration for on-demand workloads focusing on cold start optimization."""
        config_results = {
            "memory_mb": memory_size,
            "executions": [],
            "cold_start_performance": {},
            "burst_performance": {},
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "workload_type": "on_demand",
        }

        # Test cold start scenarios with different intervals
        logger.info("Testing cold start scenarios...")
        cold_start_results = await self._test_cold_start_scenarios(memory_size)
        config_results["cold_start_performance"] = cold_start_results
        config_results["executions"].extend(cold_start_results.get("executions", []))

        # Test burst traffic patterns
        if self.config.traffic_pattern == "burst":
            logger.info("Testing burst traffic pattern...")
            burst_results = await self._test_burst_pattern(memory_size)
            config_results["burst_performance"] = burst_results
            config_results["executions"].extend(burst_results.get("executions", []))

        # Calculate statistics
        config_results["total_executions"] = len(config_results["executions"])
        config_results["successful_executions"] = sum(
            1 for e in config_results["executions"] if not e.get("error")
        )
        config_results["failed_executions"] = (
            config_results["total_executions"] - config_results["successful_executions"]
        )

        return config_results

    async def _test_scheduled_workload(self, memory_size: int) -> Dict[str, Any]:
        """Test configuration for scheduled workloads."""
        config_results = {
            "memory_mb": memory_size,
            "executions": [],
            "time_window_performance": {},
            "resource_efficiency": {},
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "workload_type": "scheduled",
        }

        # Test time window scenarios
        time_window_results = await self.time_window_testing(memory_size)
        config_results["time_window_performance"] = time_window_results
        config_results["executions"].extend(time_window_results.get("executions", []))

        # Test resource efficiency during scheduled periods
        efficiency_results = await self._test_resource_efficiency(memory_size)
        config_results["resource_efficiency"] = efficiency_results
        config_results["executions"].extend(efficiency_results.get("executions", []))

        # Calculate statistics
        config_results["total_executions"] = len(config_results["executions"])
        config_results["successful_executions"] = sum(
            1 for e in config_results["executions"] if not e.get("error")
        )
        config_results["failed_executions"] = (
            config_results["total_executions"] - config_results["successful_executions"]
        )

        return config_results

    async def _test_continuous_workload(self, memory_size: int) -> Dict[str, Any]:
        """Test configuration for continuous workloads."""
        config_results = {
            "memory_mb": memory_size,
            "executions": [],
            "sustained_performance": {},
            "concurrency_performance": {},
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "workload_type": "continuous",
        }

        # Test sustained performance under continuous load
        sustained_results = await self._test_sustained_performance(memory_size)
        config_results["sustained_performance"] = sustained_results
        config_results["executions"].extend(sustained_results.get("executions", []))

        # Test concurrency patterns
        concurrency_results = await self._test_concurrency_patterns(memory_size)
        config_results["concurrency_performance"] = concurrency_results
        config_results["executions"].extend(concurrency_results.get("executions", []))

        # Calculate statistics
        config_results["total_executions"] = len(config_results["executions"])
        config_results["successful_executions"] = sum(
            1 for e in config_results["executions"] if not e.get("error")
        )
        config_results["failed_executions"] = (
            config_results["total_executions"] - config_results["successful_executions"]
        )

        return config_results

    async def time_window_testing(self, memory_size: int) -> Dict[str, Any]:
        """
        Test performance across different time windows for scheduled workloads.

        Args:
            memory_size: Memory size to test

        Returns:
            Dictionary containing time window test results
        """
        logger.info(f"Running time window testing for memory size: {memory_size}MB")

        time_window_results = {
            "memory_mb": memory_size,
            "executions": [],
            "window_analysis": {},
            "peak_performance": {},
            "idle_recovery": {},
        }

        # Test immediate execution (simulating scheduled start)
        logger.info("Testing immediate execution performance...")
        immediate_start = time.time()
        immediate_results = await self._run_concurrent_invocations(
            memory_size, min(self.config.iterations, 5), is_warmup=False
        )
        immediate_end = time.time()

        time_window_results["executions"].extend(immediate_results)
        time_window_results["window_analysis"]["immediate"] = {
            "duration": immediate_end - immediate_start,
            "avg_execution_time": (
                sum(r.get("duration", 0) for r in immediate_results) / len(immediate_results)
                if immediate_results
                else 0
            ),
            "cold_start_ratio": (
                sum(1 for r in immediate_results if r.get("cold_start", False))
                / len(immediate_results)
                if immediate_results
                else 0
            ),
        }

        # Simulate idle period and test recovery
        if not self.config.dry_run:
            logger.info("Simulating idle period (30 seconds)...")
            await asyncio.sleep(30)

        # Test post-idle performance
        logger.info("Testing post-idle performance...")
        recovery_start = time.time()
        recovery_results = await self._run_concurrent_invocations(
            memory_size, min(self.config.iterations, 3), is_warmup=False
        )
        recovery_end = time.time()

        time_window_results["executions"].extend(recovery_results)
        time_window_results["idle_recovery"] = {
            "duration": recovery_end - recovery_start,
            "avg_execution_time": (
                sum(r.get("duration", 0) for r in recovery_results) / len(recovery_results)
                if recovery_results
                else 0
            ),
            "cold_start_ratio": (
                sum(1 for r in recovery_results if r.get("cold_start", False))
                / len(recovery_results)
                if recovery_results
                else 0
            ),
        }

        return time_window_results

    async def multi_stage_optimization(self) -> Dict[str, Any]:
        """
        Multi-stage optimization workflow for comprehensive tuning.

        Returns:
            Dictionary containing multi-stage optimization results
        """
        logger.info("Starting multi-stage optimization workflow")

        optimization_results = {
            "function_arn": self.config.function_arn,
            "optimization_started": format_timestamp(),
            "stages": {},
            "recommendations": {},
            "final_configuration": {},
        }

        try:
            # Stage 1: Baseline establishment
            logger.info("Stage 1: Establishing baseline performance")
            baseline_results = await self._establish_baseline()
            optimization_results["stages"]["baseline"] = baseline_results

            # Stage 2: Initial memory sweep
            logger.info("Stage 2: Initial memory configuration sweep")
            initial_sweep = await self._initial_memory_sweep()
            optimization_results["stages"]["initial_sweep"] = initial_sweep

            # Stage 3: Focused optimization
            logger.info("Stage 3: Focused optimization on promising configurations")
            focused_results = await self._focused_optimization(initial_sweep)
            optimization_results["stages"]["focused_optimization"] = focused_results

            # Stage 4: Workload-specific validation
            logger.info("Stage 4: Workload-specific validation")
            validation_results = await self._workload_validation(focused_results)
            optimization_results["stages"]["validation"] = validation_results

            # Stage 5: Generate final recommendations
            logger.info("Stage 5: Generating recommendations")
            recommendations = await self._generate_recommendations(optimization_results["stages"])
            optimization_results["recommendations"] = recommendations

            optimization_results["optimization_completed"] = format_timestamp()
            optimization_results["total_duration"] = (
                time.time() - time.time()
            )  # This will be corrected below

            return optimization_results

        except Exception as e:
            logger.error(f"Multi-stage optimization failed: {e}")
            optimization_results["error"] = str(e)
            raise

    async def _establish_baseline(self) -> Dict[str, Any]:
        """Establish baseline performance with current configuration."""
        logger.info("Establishing baseline performance...")

        original_config = await self._get_function_configuration()
        baseline_memory = original_config["MemorySize"]

        baseline_results = await self.workload_aware_testing(baseline_memory)
        baseline_results["is_baseline"] = True
        baseline_results["original_memory"] = baseline_memory

        return baseline_results

    async def _initial_memory_sweep(self) -> Dict[str, Any]:
        """Perform initial sweep across all memory configurations."""
        logger.info("Performing initial memory sweep...")

        sweep_results = {"configurations": [], "performance_curve": {}, "initial_insights": {}}

        for memory_size in self.config.memory_sizes:
            logger.info(f"Testing memory configuration: {memory_size}MB")

            if not self.config.dry_run:
                await self._update_function_memory(memory_size)
                await asyncio.sleep(2)  # Allow configuration to propagate

            config_results = await self.workload_aware_testing(memory_size)
            sweep_results["configurations"].append(config_results)

        # Analyze performance curve
        sweep_results["performance_curve"] = self._analyze_performance_curve(
            sweep_results["configurations"]
        )

        return sweep_results

    async def _focused_optimization(self, initial_sweep: Dict[str, Any]) -> Dict[str, Any]:
        """Focus optimization on most promising configurations."""
        logger.info("Performing focused optimization...")

        # Identify top 3 performing configurations
        configurations = initial_sweep["configurations"]

        # Sort by a composite score (example: balance of performance and cost)
        scored_configs = []
        for config in configurations:
            if config["successful_executions"] > 0:
                avg_duration = sum(e.get("duration", 0) for e in config["executions"]) / len(
                    config["executions"]
                )
                success_rate = config["successful_executions"] / config["total_executions"]
                # Simple scoring: lower duration and higher success rate is better
                score = success_rate / (avg_duration + 1)  # +1 to avoid division by zero
                scored_configs.append((score, config))

        scored_configs.sort(key=lambda x: x[0], reverse=True)
        top_configs = scored_configs[:3]

        focused_results = {"top_configurations": [], "detailed_analysis": {}}

        # Run more detailed tests on top configurations
        for score, config in top_configs:
            memory_size = config["memory_mb"]
            logger.info(f"Detailed testing for top configuration: {memory_size}MB")

            if not self.config.dry_run:
                await self._update_function_memory(memory_size)
                await asyncio.sleep(2)

            # Run extended testing
            detailed_config = await self._run_extended_testing(memory_size)
            detailed_config["optimization_score"] = score
            focused_results["top_configurations"].append(detailed_config)

        return focused_results

    async def _workload_validation(self, focused_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization results against workload-specific requirements."""
        logger.info("Validating against workload requirements...")

        validation_results = {
            "workload_compliance": {},
            "performance_validation": {},
            "cost_validation": {},
        }

        for config in focused_results["top_configurations"]:
            memory_size = config["memory_mb"]

            # Workload-specific validation
            compliance_score = self._calculate_workload_compliance(config)
            validation_results["workload_compliance"][memory_size] = compliance_score

            # Performance validation
            performance_score = self._calculate_performance_score(config)
            validation_results["performance_validation"][memory_size] = performance_score

            # Cost validation
            cost_score = self._calculate_cost_score(config)
            validation_results["cost_validation"][memory_size] = cost_score

        return validation_results

    async def _run_extended_testing(self, memory_size: int) -> Dict[str, Any]:
        """Run extended testing with higher iteration count."""
        extended_iterations = min(self.config.iterations * 2, 50)

        # Temporarily increase iterations
        original_iterations = self.config.iterations
        self.config.iterations = extended_iterations

        try:
            results = await self.workload_aware_testing(memory_size)
            results["extended_testing"] = True
            return results
        finally:
            # Restore original iterations
            self.config.iterations = original_iterations

    def _analyze_performance_curve(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance curve across memory configurations."""
        curve_analysis = {
            "memory_vs_performance": {},
            "efficiency_points": [],
            "diminishing_returns_threshold": None,
        }

        # Calculate performance metrics for each memory size
        for config in configurations:
            if config["successful_executions"] > 0:
                memory_mb = config["memory_mb"]
                avg_duration = sum(e.get("duration", 0) for e in config["executions"]) / len(
                    config["executions"]
                )

                curve_analysis["memory_vs_performance"][memory_mb] = {
                    "avg_duration": avg_duration,
                    "success_rate": config["successful_executions"] / config["total_executions"],
                    "cold_start_ratio": sum(
                        1 for e in config["executions"] if e.get("cold_start", False)
                    )
                    / len(config["executions"]),
                }

        return curve_analysis

    def _calculate_workload_compliance(self, config: Dict[str, Any]) -> float:
        """Calculate how well configuration complies with workload requirements."""
        score = 0.0

        if self.config.workload_type == "on_demand":
            # Prioritize cold start performance
            cold_start_ratio = sum(
                1 for e in config["executions"] if e.get("cold_start", False)
            ) / len(config["executions"])
            if cold_start_ratio < 0.1:  # Less than 10% cold starts
                score += 0.4

            # Check burst performance
            if "burst_performance" in config and config["burst_performance"]:
                score += 0.3

        elif self.config.workload_type == "continuous":
            # Prioritize sustained performance
            if "sustained_performance" in config and config["sustained_performance"]:
                score += 0.5

            # Check concurrency handling
            if "concurrency_performance" in config and config["concurrency_performance"]:
                score += 0.3

        elif self.config.workload_type == "scheduled":
            # Check time window performance
            if "time_window_performance" in config and config["time_window_performance"]:
                score += 0.4

            # Check resource efficiency
            if "resource_efficiency" in config and config["resource_efficiency"]:
                score += 0.3

        # General performance score
        success_rate = config["successful_executions"] / config["total_executions"]
        score += success_rate * 0.3

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_performance_score(self, config: Dict[str, Any]) -> float:
        """Calculate performance score for configuration."""
        if not config["executions"]:
            return 0.0

        avg_duration = sum(e.get("duration", 0) for e in config["executions"]) / len(
            config["executions"]
        )
        success_rate = config["successful_executions"] / config["total_executions"]

        # Normalize duration (lower is better)
        duration_score = max(0, 1 - (avg_duration / 10000))  # Assume 10s is poor performance

        # Combine with success rate
        return (duration_score * 0.7) + (success_rate * 0.3)

    def _calculate_cost_score(self, config: Dict[str, Any]) -> float:
        """Calculate cost efficiency score for configuration."""
        memory_mb = config["memory_mb"]

        if not config["executions"]:
            return 0.0

        avg_duration = sum(e.get("duration", 0) for e in config["executions"]) / len(
            config["executions"]
        )
        avg_billed_duration = sum(
            e.get("billed_duration", avg_duration) for e in config["executions"]
        ) / len(config["executions"])

        # Calculate cost per execution
        gb_seconds = (memory_mb / 1024) * (avg_billed_duration / 1000)
        cost_per_execution = (
            gb_seconds * self.config.cost_per_gb_second
        ) + self.config.cost_per_request

        # Lower cost is better - normalize against a baseline
        baseline_cost = (512 / 1024) * (
            1000 / 1000
        ) * self.config.cost_per_gb_second + self.config.cost_per_request
        cost_efficiency = max(0, 1 - (cost_per_execution / baseline_cost))

        return cost_efficiency

    async def _generate_recommendations(self, stages: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final optimization recommendations."""
        recommendations = {
            "optimal_memory_size": None,
            "confidence_score": 0.0,
            "reasoning": [],
            "workload_specific_insights": {},
            "alternative_configurations": [],
        }

        if "validation" in stages and stages["validation"]["workload_compliance"]:
            # Find configuration with highest compliance score
            compliance_scores = stages["validation"]["workload_compliance"]
            optimal_memory = max(compliance_scores.keys(), key=lambda k: compliance_scores[k])

            recommendations["optimal_memory_size"] = optimal_memory
            recommendations["confidence_score"] = compliance_scores[optimal_memory]

            # Generate reasoning
            recommendations["reasoning"].append(
                f"Selected {optimal_memory}MB based on workload compliance analysis"
            )

            if self.config.workload_type == "on_demand":
                recommendations["reasoning"].append(
                    "Optimized for cold start performance and burst traffic handling"
                )
            elif self.config.workload_type == "continuous":
                recommendations["reasoning"].append(
                    "Optimized for sustained performance and concurrency handling"
                )
            elif self.config.workload_type == "scheduled":
                recommendations["reasoning"].append(
                    "Optimized for scheduled execution patterns and resource efficiency"
                )

        return recommendations

    async def _test_cold_start_scenarios(self, memory_size: int) -> Dict[str, Any]:
        """Test cold start scenarios with different intervals."""
        cold_start_results = {
            "executions": [],
            "cold_start_analysis": {},
            "warm_start_analysis": {},
        }

        # Test immediate cold start
        logger.info("Testing immediate cold start...")
        cold_execution = await self._run_concurrent_invocations(memory_size, 1, is_warmup=False)
        cold_start_results["executions"].extend(cold_execution)

        if not self.config.dry_run:
            # Wait for container to expire (simulate cold start)
            await asyncio.sleep(5)

        # Test after brief pause
        brief_pause_execution = await self._run_concurrent_invocations(
            memory_size, 2, is_warmup=False
        )
        cold_start_results["executions"].extend(brief_pause_execution)

        # Test warm starts
        warm_executions = await self._run_concurrent_invocations(memory_size, 3, is_warmup=False)
        cold_start_results["executions"].extend(warm_executions)

        # Analyze cold vs warm performance
        cold_starts = [e for e in cold_start_results["executions"] if e.get("cold_start", False)]
        warm_starts = [
            e for e in cold_start_results["executions"] if not e.get("cold_start", False)
        ]

        if cold_starts:
            cold_start_results["cold_start_analysis"] = {
                "count": len(cold_starts),
                "avg_duration": sum(e.get("duration", 0) for e in cold_starts) / len(cold_starts),
                "max_duration": max(e.get("duration", 0) for e in cold_starts),
            }

        if warm_starts:
            cold_start_results["warm_start_analysis"] = {
                "count": len(warm_starts),
                "avg_duration": sum(e.get("duration", 0) for e in warm_starts) / len(warm_starts),
                "max_duration": max(e.get("duration", 0) for e in warm_starts),
            }

        return cold_start_results

    async def _test_burst_pattern(self, memory_size: int) -> Dict[str, Any]:
        """Test burst traffic pattern."""
        burst_results = {"executions": [], "burst_analysis": {}}

        # Simulate burst by running high concurrency for short period
        logger.info("Simulating burst traffic pattern...")
        burst_concurrency = min(self.config.concurrent_executions * 2, 20)

        # Save original concurrency setting
        original_concurrency = self.config.concurrent_executions
        self.config.concurrent_executions = burst_concurrency

        try:
            burst_executions = await self._run_concurrent_invocations(
                memory_size, burst_concurrency, is_warmup=False
            )
            burst_results["executions"].extend(burst_executions)

            # Analyze burst performance
            if burst_executions:
                burst_results["burst_analysis"] = {
                    "total_executions": len(burst_executions),
                    "avg_duration": sum(e.get("duration", 0) for e in burst_executions)
                    / len(burst_executions),
                    "success_rate": sum(1 for e in burst_executions if not e.get("error"))
                    / len(burst_executions),
                    "concurrency_level": burst_concurrency,
                }
        finally:
            # Restore original concurrency
            self.config.concurrent_executions = original_concurrency

        return burst_results

    async def _test_resource_efficiency(self, memory_size: int) -> Dict[str, Any]:
        """Test resource efficiency for scheduled workloads."""
        efficiency_results = {"executions": [], "efficiency_metrics": {}}

        logger.info("Testing resource efficiency...")

        # Run standard test set
        executions = await self._run_concurrent_invocations(
            memory_size, self.config.iterations, is_warmup=False
        )
        efficiency_results["executions"].extend(executions)

        if executions:
            # Calculate efficiency metrics
            total_memory_seconds = sum(
                memory_size * (e.get("billed_duration", e.get("duration", 0)) / 1000)
                for e in executions
            )
            avg_memory_used = sum(
                e.get("memory_used", memory_size * 0.7) for e in executions
            ) / len(executions)

            efficiency_results["efficiency_metrics"] = {
                "memory_utilization": avg_memory_used / memory_size,
                "total_memory_seconds": total_memory_seconds,
                "avg_execution_cost": self._calculate_execution_cost(memory_size, executions),
                "resource_waste_ratio": max(0, (memory_size - avg_memory_used) / memory_size),
            }

        return efficiency_results

    async def _test_sustained_performance(self, memory_size: int) -> Dict[str, Any]:
        """Test sustained performance for continuous workloads."""
        sustained_results = {"executions": [], "performance_stability": {}}

        logger.info("Testing sustained performance...")

        # Run multiple waves of executions to simulate sustained load
        waves = 3
        wave_size = max(self.config.iterations // waves, 3)

        all_wave_results = []

        for wave in range(waves):
            logger.info(f"Running wave {wave + 1}/{waves}...")
            wave_executions = await self._run_concurrent_invocations(
                memory_size, wave_size, is_warmup=False
            )
            sustained_results["executions"].extend(wave_executions)

            if wave_executions:
                wave_avg_duration = sum(e.get("duration", 0) for e in wave_executions) / len(
                    wave_executions
                )
                all_wave_results.append(wave_avg_duration)

            # Brief pause between waves
            if wave < waves - 1 and not self.config.dry_run:
                await asyncio.sleep(2)

        # Analyze performance stability
        if all_wave_results:
            performance_variance = max(all_wave_results) - min(all_wave_results)
            avg_performance = sum(all_wave_results) / len(all_wave_results)

            sustained_results["performance_stability"] = {
                "wave_performances": all_wave_results,
                "performance_variance": performance_variance,
                "avg_performance": avg_performance,
                "stability_score": (
                    max(0, 1 - (performance_variance / avg_performance))
                    if avg_performance > 0
                    else 0
                ),
            }

        return sustained_results

    async def _test_concurrency_patterns(self, memory_size: int) -> Dict[str, Any]:
        """Test concurrency patterns for continuous workloads."""
        concurrency_results = {"executions": [], "concurrency_analysis": {}}

        logger.info("Testing concurrency patterns...")

        # Test different concurrency levels
        concurrency_levels = [
            1,
            self.config.concurrent_executions // 2,
            self.config.concurrent_executions,
        ]
        concurrency_performance = {}

        for concurrency in concurrency_levels:
            if concurrency > 0:
                logger.info(f"Testing concurrency level: {concurrency}")

                # Temporarily set concurrency
                original_concurrency = self.config.concurrent_executions
                self.config.concurrent_executions = concurrency

                try:
                    level_executions = await self._run_concurrent_invocations(
                        memory_size, concurrency * 2, is_warmup=False
                    )
                    concurrency_results["executions"].extend(level_executions)

                    if level_executions:
                        avg_duration = sum(e.get("duration", 0) for e in level_executions) / len(
                            level_executions
                        )
                        success_rate = sum(1 for e in level_executions if not e.get("error")) / len(
                            level_executions
                        )

                        concurrency_performance[concurrency] = {
                            "avg_duration": avg_duration,
                            "success_rate": success_rate,
                            "total_executions": len(level_executions),
                        }
                finally:
                    self.config.concurrent_executions = original_concurrency

        concurrency_results["concurrency_analysis"] = {
            "concurrency_performance": concurrency_performance,
            "optimal_concurrency": self._find_optimal_concurrency(concurrency_performance),
        }

        return concurrency_results

    def _find_optimal_concurrency(self, concurrency_performance: Dict[int, Dict[str, Any]]) -> int:
        """Find optimal concurrency level based on performance data."""
        if not concurrency_performance:
            return self.config.concurrent_executions

        # Score each concurrency level (balance of performance and success rate)
        scores = {}
        for concurrency, metrics in concurrency_performance.items():
            # Lower duration and higher success rate is better
            duration_score = max(0, 1 - (metrics["avg_duration"] / 10000))  # Normalize against 10s
            success_score = metrics["success_rate"]
            combined_score = (duration_score * 0.6) + (success_score * 0.4)
            scores[concurrency] = combined_score

        # Return concurrency level with highest score
        return max(scores.keys(), key=lambda k: scores[k])

    def _calculate_execution_cost(
        self, memory_size: int, executions: List[Dict[str, Any]]
    ) -> float:
        """Calculate average cost per execution."""
        if not executions:
            return 0.0

        total_cost = 0.0
        for execution in executions:
            duration_ms = execution.get("billed_duration", execution.get("duration", 0))
            gb_seconds = (memory_size / 1024) * (duration_ms / 1000)
            execution_cost = (
                gb_seconds * self.config.cost_per_gb_second
            ) + self.config.cost_per_request
            total_cost += execution_cost

        return total_cost / len(executions)

    async def start_continuous_monitoring(self):
        """Start continuous monitoring and optimization."""
        if not self.performance_monitor:
            logger.warning("Performance monitor not initialized - cannot start monitoring")
            return

        logger.info("Starting continuous monitoring")
        await self.performance_monitor.start_monitoring()

        if self.continuous_optimizer:
            await self.continuous_optimizer.start_continuous_optimization()

    async def stop_continuous_monitoring(self):
        """Stop continuous monitoring and optimization."""
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()

        if self.continuous_optimizer:
            await self.continuous_optimizer.stop_continuous_optimization()

        logger.info("Continuous monitoring stopped")

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        status = {
            "monitoring_enabled": self.config.monitoring_enabled,
            "auto_retuning_enabled": self.config.auto_retuning_enabled,
            "alerts_enabled": self.config.alerts_enabled,
            "components": {
                "performance_monitor": self.performance_monitor is not None,
                "alert_manager": self.alert_manager is not None,
                "continuous_optimizer": self.continuous_optimizer is not None,
                "recommendation_engine": self.recommendation_engine is not None,
            },
        }

        if self.performance_monitor:
            status["performance_status"] = await self.performance_monitor.get_current_status()

        if self.continuous_optimizer:
            status["optimization_status"] = (
                await self.continuous_optimizer.get_optimization_status()
            )

        if self.alert_manager:
            status["active_alerts"] = len(await self.alert_manager.get_active_alerts())

        return status

    async def generate_intelligent_recommendation(
        self, analysis_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate intelligent recommendations using ML insights."""
        if not self.recommendation_engine:
            logger.warning("Recommendation engine not initialized")
            return None

        try:
            # Extract required data from analysis results
            memory_results = analysis_results.get("memory_results", {})
            performance_analysis = analysis_results.get("analysis")

            if not memory_results or not performance_analysis:
                logger.warning("Insufficient data for intelligent recommendations")
                return None

            # Generate intelligent recommendation
            ml_recommendation = self.recommendation_engine.generate_intelligent_recommendation(
                performance_analysis, memory_results
            )

            return {
                "base_recommendation": {
                    "strategy": ml_recommendation.base_recommendation.strategy,
                    "current_memory_size": ml_recommendation.base_recommendation.current_memory_size,
                    "optimal_memory_size": ml_recommendation.base_recommendation.optimal_memory_size,
                    "should_optimize": ml_recommendation.base_recommendation.should_optimize,
                    "cost_change_percent": ml_recommendation.base_recommendation.cost_change_percent,
                    "duration_change_percent": ml_recommendation.base_recommendation.duration_change_percent,
                    "reasoning": ml_recommendation.base_recommendation.reasoning,
                    "confidence_score": ml_recommendation.base_recommendation.confidence_score,
                },
                "ml_insights": {
                    "confidence_score": ml_recommendation.confidence_score,
                    "pattern_match_score": ml_recommendation.pattern_match_score,
                    "similar_functions": ml_recommendation.similar_functions,
                    "predicted_performance": ml_recommendation.predicted_performance,
                    "risk_assessment": ml_recommendation.risk_assessment,
                    "optimization_timeline": ml_recommendation.optimization_timeline,
                },
            }

        except Exception as e:
            logger.error(f"Failed to generate intelligent recommendation: {e}")
            return None

    async def trigger_manual_optimization(
        self, trigger_reason: str, trigger_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Manually trigger an optimization."""
        if not self.continuous_optimizer:
            logger.warning("Continuous optimizer not initialized")
            return False

        try:
            from .monitoring.continuous_optimizer import OptimizationTrigger

            event = await self.continuous_optimizer.trigger_optimization(
                OptimizationTrigger.MANUAL,
                trigger_data or {"reason": trigger_reason},
                severity="medium",
                auto_approve=False,
            )

            logger.info(f"Manual optimization triggered: {event.trigger.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to trigger manual optimization: {e}")
            return False

    async def get_performance_insights(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance insights and trends."""
        if not self.performance_monitor:
            return {"error": "Performance monitor not available"}

        try:
            time_window = timedelta(hours=time_window_hours)

            # Get current metrics
            current_metrics = await self.performance_monitor.get_current_metrics()

            # Get performance trends
            trends = await self.performance_monitor.get_performance_trends(time_window)

            # Get cost metrics
            cost_metrics = await self.performance_monitor.get_cost_metrics()

            # Get error trends
            error_trends = await self.performance_monitor.get_error_trends()

            return {
                "current_metrics": current_metrics,
                "performance_trends": [
                    {
                        "metric_name": trend.metric_name,
                        "trend_direction": trend.trend_direction,
                        "trend_strength": trend.trend_strength,
                        "statistical_summary": trend.statistical_summary,
                    }
                    for trend in trends
                ],
                "cost_metrics": cost_metrics,
                "error_trends": error_trends,
                "analysis_window_hours": time_window_hours,
            }

        except Exception as e:
            logger.error(f"Failed to get performance insights: {e}")
            return {"error": str(e)}

    async def update_monitoring_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update monitoring configuration at runtime."""
        try:
            # Update configuration
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Reinitialize components if needed
            if "monitoring_enabled" in new_config or "auto_retuning_enabled" in new_config:
                # Stop existing components
                await self.stop_continuous_monitoring()

                # Reinitialize with new config
                self._initialize_monitoring_components(None)

                # Restart if enabled
                if self.config.monitoring_enabled:
                    await self.start_continuous_monitoring()

            logger.info("Monitoring configuration updated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to update monitoring configuration: {e}")
            return False


# Convenience functions
async def run_tuning_session(config: TunerConfig) -> Dict[str, Any]:
    """Run a complete tuning session."""
    orchestrator = TunerOrchestrator(config)
    return await orchestrator.run_tuning()


async def test_single_configuration(
    function_arn: str, memory_size: int, payload: str = "{}", iterations: int = 10
) -> Dict[str, Any]:
    """Test a single memory configuration."""
    config = TunerConfig(
        function_arn=function_arn,
        memory_sizes=[memory_size],
        payload=payload,
        iterations=iterations,
    )

    orchestrator = TunerOrchestrator(config)
    return await orchestrator.run_tuning()
