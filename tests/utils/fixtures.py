"""Common test fixtures and data creation utilities."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.models import (
    MemoryTestResult,
    Recommendation,
    PerformanceAnalysis,
    TuningResult,
)


def create_test_config(
    function_arn: str = None,
    memory_sizes: List[int] = None,
    iterations: int = 5,
    strategy: str = "balanced",
    **kwargs,
) -> TunerConfig:
    """Create a test configuration with sensible defaults."""
    if function_arn is None:
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"

    if memory_sizes is None:
        memory_sizes = [256, 512, 1024]

    config_data = {
        "function_arn": function_arn,
        "memory_sizes": memory_sizes,
        "iterations": iterations,
        "strategy": strategy,
        "concurrent_executions": kwargs.get("concurrent_executions", 2),
        "timeout": kwargs.get("timeout", 60),
        "dry_run": kwargs.get("dry_run", True),
        "payload": kwargs.get("payload", '{"test": "data"}'),
        "warmup_iterations": kwargs.get("warmup_iterations", 0),
        "output_format": kwargs.get("output_format", "json"),
        "save_results": kwargs.get("save_results", False),
    }

    return TunerConfig.from_dict(config_data)


def create_test_results(
    function_arn: str = None, memory_sizes: List[int] = None, executions_per_memory: int = 5
) -> Dict[str, Any]:
    """Create test results data structure."""
    if function_arn is None:
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"

    if memory_sizes is None:
        memory_sizes = [256, 512, 1024]

    test_start = datetime.utcnow() - timedelta(minutes=20)
    test_end = datetime.utcnow()

    configurations = []

    for memory_size in memory_sizes:
        executions = []
        successful_count = 0

        for i in range(executions_per_memory):
            # Simulate performance: higher memory = faster execution
            base_duration = 200 * (256 / memory_size)
            duration = base_duration + (i * 10)  # Add some variance

            # Simulate cold starts (first execution more likely)
            cold_start = i == 0
            if cold_start:
                duration += 500  # Cold start penalty

            execution = {
                "memory_mb": memory_size,
                "execution_id": i,
                "duration": round(duration, 2),
                "billed_duration": max(100, int(duration + 99) // 100 * 100),  # Round up to 100ms
                "cold_start": cold_start,
                "status_code": 200,
                "timestamp": (test_start + timedelta(minutes=i)).isoformat(),
                "max_memory_used": min(memory_size, memory_size // 2 + 20),
                "request_id": f"req-{memory_size}-{i:03d}",
            }
            executions.append(execution)
            successful_count += 1

        # Calculate aggregated metrics
        durations = [e["duration"] for e in executions]
        billed_durations = [e["billed_duration"] for e in executions]
        cold_starts = sum(1 for e in executions if e["cold_start"])

        # Simple cost calculation (AWS Lambda pricing approximation)
        gb_seconds = sum((memory_size / 1024) * (bd / 1000) for bd in billed_durations)
        total_cost = gb_seconds * 0.0000166667 + len(executions) * 0.0000002

        configuration = {
            "memory_mb": memory_size,
            "executions": executions,
            "total_executions": len(executions),
            "successful_executions": successful_count,
            "failed_executions": len(executions) - successful_count,
            "avg_duration": sum(durations) / len(durations),
            "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
            "p99_duration": sorted(durations)[int(len(durations) * 0.99)],
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_billed_duration": sum(billed_durations) / len(billed_durations),
            "total_cost": round(total_cost, 8),
            "avg_cost": round(total_cost / len(executions), 8),
            "cold_starts": cold_starts,
            "cold_start_rate": cold_starts / len(executions),
        }
        configurations.append(configuration)

    return {
        "function_arn": function_arn,
        "test_started": test_start.isoformat(),
        "test_completed": test_end.isoformat(),
        "test_duration_seconds": int((test_end - test_start).total_seconds()),
        "strategy": "balanced",
        "iterations_per_memory": executions_per_memory,
        "total_executions": len(memory_sizes) * executions_per_memory,
        "configurations": configurations,
        "summary": {
            "memory_sizes_tested": memory_sizes,
            "total_test_time": int((test_end - test_start).total_seconds()),
            "total_cost": sum(config["total_cost"] for config in configurations),
            "fastest_memory": min(configurations, key=lambda x: x["avg_duration"])["memory_mb"],
            "cheapest_memory": min(configurations, key=lambda x: x["avg_cost"])["memory_mb"],
        },
    }


def create_test_analysis(memory_results: Dict[int, MemoryTestResult] = None) -> PerformanceAnalysis:
    """Create a test performance analysis."""
    if memory_results is None:
        memory_results = {
            256: MemoryTestResult(
                memory_size=256,
                iterations=5,
                avg_duration=180.5,
                p95_duration=200.0,
                p99_duration=220.0,
                avg_cost=0.00000834,
                total_cost=0.00004170,
                cold_starts=1,
                errors=0,
            ),
            512: MemoryTestResult(
                memory_size=512,
                iterations=5,
                avg_duration=120.3,
                p95_duration=140.0,
                p99_duration=150.0,
                avg_cost=0.00001125,
                total_cost=0.00005625,
                cold_starts=1,
                errors=0,
            ),
            1024: MemoryTestResult(
                memory_size=1024,
                iterations=5,
                avg_duration=95.7,
                p95_duration=110.0,
                p99_duration=115.0,
                avg_cost=0.00001668,
                total_cost=0.00008340,
                cold_starts=1,
                errors=0,
            ),
        }

    # Calculate efficiency scores
    efficiency_scores = {}
    for memory_size, result in memory_results.items():
        if result.avg_cost > 0:
            # Performance per cost ratio
            efficiency = 1000 / (result.avg_duration * result.avg_cost * 1000000)
            efficiency_scores[memory_size] = round(efficiency, 2)
        else:
            efficiency_scores[memory_size] = 0

    # Find optimal configurations
    cost_optimal = {
        "memory_size": min(memory_results.keys(), key=lambda x: memory_results[x].avg_cost),
        "reasoning": "Lowest average execution cost",
    }

    speed_optimal = {
        "memory_size": min(memory_results.keys(), key=lambda x: memory_results[x].avg_duration),
        "reasoning": "Fastest average execution time",
    }

    balanced_optimal = {
        "memory_size": max(efficiency_scores.keys(), key=lambda x: efficiency_scores[x]),
        "reasoning": "Best performance/cost ratio",
    }

    trends = {
        "duration_trend": "decreasing",
        "cost_trend": "increasing",
        "performance_plateau": None,
        "memory_sensitivity": "medium",
    }

    insights = [
        {
            "type": "memory_optimization",
            "severity": "medium",
            "message": "Function shows moderate memory sensitivity.",
            "recommendation": "Consider 512MB for balanced performance and cost",
        },
        {
            "type": "cost_optimization",
            "severity": "low",
            "message": "Current memory allocation appears reasonable.",
            "recommendation": "Monitor usage patterns for potential optimization",
        },
    ]

    return PerformanceAnalysis(
        memory_results=memory_results,
        efficiency_scores=efficiency_scores,
        cost_optimal=cost_optimal,
        speed_optimal=speed_optimal,
        balanced_optimal=balanced_optimal,
        trends=trends,
        insights=insights,
    )


def create_test_recommendation(
    strategy: str = "balanced", current_memory: int = 256, optimal_memory: int = 512
) -> Recommendation:
    """Create a test recommendation."""
    cost_change = ((optimal_memory / current_memory) - 1) * 100
    duration_change = (
        -((optimal_memory / current_memory) ** 0.5 - 1) * 100
    )  # Negative = improvement

    monthly_savings = {
        "low_usage": {"current_cost": 1.50, "optimized_cost": 1.20, "savings": 0.30},
        "medium_usage": {"current_cost": 15.00, "optimized_cost": 12.00, "savings": 3.00},
        "high_usage": {"current_cost": 150.00, "optimized_cost": 120.00, "savings": 30.00},
    }

    return Recommendation(
        strategy=strategy,
        current_memory_size=current_memory,
        optimal_memory_size=optimal_memory,
        should_optimize=optimal_memory != current_memory,
        cost_change_percent=round(cost_change, 1),
        duration_change_percent=round(duration_change, 1),
        reasoning=f"Optimal memory for {strategy} strategy based on performance analysis",
        confidence_score=0.85,
        estimated_monthly_savings=monthly_savings,
    )


def create_complete_test_results(
    function_arn: str = None, memory_sizes: List[int] = None
) -> TuningResult:
    """Create a complete TuningResult object for testing."""
    if function_arn is None:
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"

    if memory_sizes is None:
        memory_sizes = [256, 512, 1024]

    # Create memory test results
    memory_results = {}
    for memory_size in memory_sizes:
        memory_results[memory_size] = MemoryTestResult(
            memory_size=memory_size,
            iterations=5,
            avg_duration=200 * (256 / memory_size),  # Inverse relationship
            p95_duration=220 * (256 / memory_size),
            p99_duration=240 * (256 / memory_size),
            avg_cost=memory_size * 0.000000033,  # Proportional to memory
            total_cost=memory_size * 0.000000165,
            cold_starts=1,
            errors=0,
        )

    # Create analysis
    analysis = create_test_analysis(memory_results)

    # Create recommendation
    optimal_memory = min(
        memory_results.keys(),
        key=lambda x: memory_results[x].avg_duration * memory_results[x].avg_cost,
    )
    recommendation = create_test_recommendation(
        current_memory=memory_sizes[0], optimal_memory=optimal_memory
    )

    return TuningResult(
        function_arn=function_arn,
        timestamp=datetime.utcnow(),
        strategy="balanced",
        memory_results=memory_results,
        baseline_results=None,
        analysis=analysis,
        recommendation=recommendation,
        duration=1200.0,  # 20 minutes
    )


def create_config_file(temp_dir: Path, config_data: Dict[str, Any] = None) -> Path:
    """Create a temporary configuration file."""
    if config_data is None:
        config_data = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
            "memory_sizes": [256, 512, 1024],
            "iterations": 5,
            "strategy": "balanced",
            "payload": '{"test": "data"}',
        }

    config_file = temp_dir / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_file


def create_results_file(temp_dir: Path, results_data: Dict[str, Any] = None) -> Path:
    """Create a temporary results file."""
    if results_data is None:
        results_data = create_test_results()

    results_file = temp_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    return results_file


def create_payload_file(temp_dir: Path, payload: Any = None) -> Path:
    """Create a temporary payload file."""
    if payload is None:
        payload = {"test": "data", "timestamp": datetime.utcnow().isoformat()}

    payload_file = temp_dir / "test_payload.json"
    with open(payload_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload_file


# Test data constants for reuse across tests
DEFAULT_FUNCTION_ARN = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
DEFAULT_MEMORY_SIZES = [256, 512, 1024]
DEFAULT_ITERATIONS = 5
DEFAULT_STRATEGY = "balanced"
DEFAULT_PAYLOAD = '{"test": "data"}'

# Common test scenarios
TEST_SCENARIOS = {
    "cpu_intensive": {
        "description": "CPU-intensive workload",
        "memory_sizes": [512, 1024, 2048, 3008],
        "expected_memory_sensitivity": "high",
        "expected_optimal_memory": 2048,
    },
    "io_bound": {
        "description": "I/O-bound workload",
        "memory_sizes": [128, 256, 512],
        "expected_memory_sensitivity": "low",
        "expected_optimal_memory": 256,
    },
    "memory_intensive": {
        "description": "Memory-intensive workload",
        "memory_sizes": [1024, 1536, 2048, 3008],
        "expected_memory_sensitivity": "high",
        "expected_optimal_memory": 3008,
    },
    "cost_sensitive": {
        "description": "Cost-sensitive optimization",
        "memory_sizes": [128, 256, 512],
        "strategy": "cost",
        "expected_optimal_memory": 128,
    },
    "speed_critical": {
        "description": "Speed-critical optimization",
        "memory_sizes": [1024, 1536, 2048, 3008],
        "strategy": "speed",
        "expected_optimal_memory": 3008,
    },
}
