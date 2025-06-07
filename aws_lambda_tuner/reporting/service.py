"""
Reporting Service for AWS Lambda Tuner.

Handles generation of reports in various formats (JSON, CSV, HTML).
"""

import json
import csv
import os
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from ..models import TuningResult
from ..config import TunerConfig
from .visualizer import ReportVisualizer


class ReportingService:
    """Service for generating performance tuning reports."""

    def __init__(self, output_dir: str = "reports"):
        """Initialize the reporting service.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.visualizer = ReportVisualizer()

    def generate_json_report(self, results: List[TuningResult], config: TunerConfig) -> str:
        """Generate a JSON report of tuning results.
        
        Args:
            results: List of tuning results
            config: Tuner configuration
            
        Returns:
            Path to generated JSON file
        """
        report_data = {
            "metadata": {
                "function_arn": config.function_arn,
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": config.strategy,
                "iterations": config.iterations,
                "memory_sizes_tested": [r.memory_size for r in results]
            },
            "results": []
        }
        
        for result in results:
            result_data = {
                "memory_size": result.memory_size,
                "test_runs": [
                    {
                        "iteration": run.iteration,
                        "duration_ms": run.duration_ms,
                        "billed_duration_ms": run.billed_duration_ms,
                        "memory_used_mb": run.memory_used_mb,
                        "cold_start": run.cold_start,
                        "status_code": run.status_code,
                        "error": run.error
                    }
                    for run in result.test_runs
                ],
                "analysis": result.analysis,
                "cost_analysis": result.cost_analysis
            }
            report_data["results"].append(result_data)
        
        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"lambda_tuning_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(filepath)

    def generate_csv_report(self, results: List[TuningResult], config: TunerConfig) -> str:
        """Generate a CSV report of tuning results.
        
        Args:
            results: List of tuning results
            config: Tuner configuration
            
        Returns:
            Path to generated CSV file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"lambda_tuning_report_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Memory Size (MB)",
                "Iteration",
                "Duration (ms)",
                "Billed Duration (ms)",
                "Memory Used (MB)",
                "Cold Start",
                "Status Code",
                "Error",
                "Cost per Invocation ($)"
            ])
            
            # Write data rows
            for result in results:
                for run in result.test_runs:
                    cost_per_invocation = self._calculate_invocation_cost(
                        run.billed_duration_ms,
                        result.memory_size,
                        config.region
                    )
                    
                    writer.writerow([
                        result.memory_size,
                        run.iteration,
                        f"{run.duration_ms:.2f}",
                        run.billed_duration_ms,
                        run.memory_used_mb,
                        run.cold_start,
                        run.status_code,
                        run.error or "",
                        f"{cost_per_invocation:.8f}"
                    ])
        
        return str(filepath)

    def generate_html_report(self, results: List[TuningResult], config: TunerConfig) -> str:
        """Generate an HTML report with visualizations.
        
        Args:
            results: List of tuning results
            config: Tuner configuration
            
        Returns:
            Path to generated HTML file
        """
        # Generate charts
        performance_chart = self.visualizer.create_performance_chart(results)
        cost_chart = self.visualizer.create_cost_chart(results)
        comparison_chart = self.visualizer.create_comparison_chart(results)
        
        # Find optimal configuration
        optimal = self._find_optimal(results, config.strategy)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AWS Lambda Tuning Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .optimal {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #343a40;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>AWS Lambda Performance Tuning Report</h1>
        
        <div class="metadata">
            <h2>Test Configuration</h2>
            <p><strong>Function ARN:</strong> {config.function_arn}</p>
            <p><strong>Date:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <p><strong>Strategy:</strong> {config.strategy}</p>
            <p><strong>Iterations per Memory Size:</strong> {config.iterations}</p>
            <p><strong>Memory Sizes Tested:</strong> {', '.join(str(r.memory_size) + 'MB' for r in results)}</p>
        </div>
        
        <div class="optimal">
            <h2>🏆 Optimal Configuration</h2>
            <p><strong>Memory Size:</strong> {optimal.memory_size}MB</p>
            <p><strong>Average Duration:</strong> {optimal.analysis['avg_duration']:.2f}ms</p>
            <p><strong>Average Cost per Invocation:</strong> ${optimal.cost_analysis['avg_cost_per_invocation']:.8f}</p>
            <p><strong>Estimated Monthly Cost (1M invocations):</strong> ${optimal.cost_analysis['monthly_cost_1m_invocations']:.2f}</p>
        </div>
        
        <div class="chart-container">
            <h2>Performance Analysis</h2>
            <div id="performance-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Cost Analysis</h2>
            <div id="cost-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Performance vs Cost Comparison</h2>
            <div id="comparison-chart"></div>
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Memory Size</th>
                    <th>Avg Duration (ms)</th>
                    <th>Min Duration (ms)</th>
                    <th>Max Duration (ms)</th>
                    <th>Avg Cost/Invocation</th>
                    <th>Cold Starts</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for result in results:
            html_content += f"""
                <tr>
                    <td>{result.memory_size}MB</td>
                    <td>{result.analysis['avg_duration']:.2f}</td>
                    <td>{result.analysis['min_duration']:.2f}</td>
                    <td>{result.analysis['max_duration']:.2f}</td>
                    <td>${result.cost_analysis['avg_cost_per_invocation']:.8f}</td>
                    <td>{result.analysis['cold_start_count']}</td>
                    <td>{result.analysis['error_count']}</td>
                </tr>
"""
        
        html_content += f"""
            </tbody>
        </table>
    </div>
    
    <script>
        {performance_chart}
        {cost_chart}
        {comparison_chart}
    </script>
</body>
</html>
"""
        
        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"lambda_tuning_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return str(filepath)

    def _calculate_invocation_cost(self, billed_duration_ms: int, memory_size: int, region: str) -> float:
        """Calculate the cost of a single invocation.
        
        Args:
            billed_duration_ms: Billed duration in milliseconds
            memory_size: Memory size in MB
            region: AWS region
            
        Returns:
            Cost in USD
        """
        # AWS Lambda pricing (as of 2024)
        # Price per GB-second varies by region
        price_per_gb_second = 0.0000166667  # Default US East pricing
        
        # Calculate GB-seconds
        gb_seconds = (memory_size / 1024) * (billed_duration_ms / 1000)
        
        # Calculate cost
        cost = gb_seconds * price_per_gb_second
        
        # Add request charge ($0.20 per 1M requests)
        cost += 0.0000002
        
        return cost

    def _find_optimal(self, results: List[TuningResult], strategy: str) -> TuningResult:
        """Find the optimal configuration based on strategy.
        
        Args:
            results: List of tuning results
            strategy: Optimization strategy
            
        Returns:
            Optimal TuningResult
        """
        if strategy == 'speed':
            return min(results, key=lambda r: r.analysis['avg_duration'])
        elif strategy == 'cost':
            return min(results, key=lambda r: r.cost_analysis['avg_cost_per_invocation'])
        else:  # balanced
            # Calculate a balanced score (normalized speed + normalized cost)
            min_duration = min(r.analysis['avg_duration'] for r in results)
            max_duration = max(r.analysis['avg_duration'] for r in results)
            min_cost = min(r.cost_analysis['avg_cost_per_invocation'] for r in results)
            max_cost = max(r.cost_analysis['avg_cost_per_invocation'] for r in results)
            
            def balanced_score(result):
                norm_duration = (result.analysis['avg_duration'] - min_duration) / (max_duration - min_duration) if max_duration > min_duration else 0
                norm_cost = (result.cost_analysis['avg_cost_per_invocation'] - min_cost) / (max_cost - min_cost) if max_cost > min_cost else 0
                return norm_duration + norm_cost
            
            return min(results, key=balanced_score)
