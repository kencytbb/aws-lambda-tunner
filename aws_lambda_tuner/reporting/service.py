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

# from .visualizer import ReportVisualizer  # Module not yet implemented
from ..intelligence.recommendation_engine import IntelligentRecommendationEngine, MLRecommendation
from ..intelligence.pattern_recognizer import PatternRecognizer


class ReportingService:
    """Service for generating performance tuning reports with intelligent insights."""

    def __init__(self, output_dir: str = "reports", config: TunerConfig = None):
        """Initialize the reporting service.

        Args:
            output_dir: Directory to save reports
            config: Tuner configuration for intelligent features
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.visualizer = ReportVisualizer()
        self.config = config

        # Initialize intelligence components if config is provided
        self.recommendation_engine = None
        self.pattern_recognizer = None

        if config:
            try:
                self.recommendation_engine = IntelligentRecommendationEngine(config)
                self.pattern_recognizer = PatternRecognizer(config)
            except Exception as e:
                # Fall back to basic reporting if intelligence components fail
                print(f"Warning: Could not initialize intelligence components: {e}")
                self.recommendation_engine = None
                self.pattern_recognizer = None

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
                "memory_sizes_tested": [r.memory_size for r in results],
            },
            "results": [],
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
                        "error": run.error,
                    }
                    for run in result.test_runs
                ],
                "analysis": result.analysis,
                "cost_analysis": result.cost_analysis,
            }
            report_data["results"].append(result_data)

        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"lambda_tuning_report_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
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

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Memory Size (MB)",
                    "Iteration",
                    "Duration (ms)",
                    "Billed Duration (ms)",
                    "Memory Used (MB)",
                    "Cold Start",
                    "Status Code",
                    "Error",
                    "Cost per Invocation ($)",
                ]
            )

            # Write data rows
            for result in results:
                for run in result.test_runs:
                    cost_per_invocation = self._calculate_invocation_cost(
                        run.billed_duration_ms, result.memory_size, config.region
                    )

                    writer.writerow(
                        [
                            result.memory_size,
                            run.iteration,
                            f"{run.duration_ms:.2f}",
                            run.billed_duration_ms,
                            run.memory_used_mb,
                            run.cold_start,
                            run.status_code,
                            run.error or "",
                            f"{cost_per_invocation:.8f}",
                        ]
                    )

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

        with open(filepath, "w") as f:
            f.write(html_content)

        return str(filepath)

    def _calculate_invocation_cost(
        self, billed_duration_ms: int, memory_size: int, region: str
    ) -> float:
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
        if strategy == "speed":
            return min(results, key=lambda r: r.analysis["avg_duration"])
        elif strategy == "cost":
            return min(results, key=lambda r: r.cost_analysis["avg_cost_per_invocation"])
        else:  # balanced
            # Calculate a balanced score (normalized speed + normalized cost)
            min_duration = min(r.analysis["avg_duration"] for r in results)
            max_duration = max(r.analysis["avg_duration"] for r in results)
            min_cost = min(r.cost_analysis["avg_cost_per_invocation"] for r in results)
            max_cost = max(r.cost_analysis["avg_cost_per_invocation"] for r in results)

            def balanced_score(result):
                norm_duration = (
                    (result.analysis["avg_duration"] - min_duration) / (max_duration - min_duration)
                    if max_duration > min_duration
                    else 0
                )
                norm_cost = (
                    (result.cost_analysis["avg_cost_per_invocation"] - min_cost)
                    / (max_cost - min_cost)
                    if max_cost > min_cost
                    else 0
                )
                return norm_duration + norm_cost

            return min(results, key=balanced_score)

    def generate_intelligent_report(self, results: List[TuningResult], config: TunerConfig) -> str:
        """Generate an intelligent report with ML insights and pattern analysis.

        Args:
            results: List of tuning results
            config: Tuner configuration

        Returns:
            Path to generated intelligent report file
        """
        # Base report data
        report_data = {
            "metadata": {
                "function_arn": config.function_arn,
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": config.strategy,
                "iterations": config.iterations,
                "memory_sizes_tested": [r.memory_size for r in results],
                "report_type": "intelligent_analysis",
            },
            "results": [],
            "intelligent_insights": {},
            "performance_patterns": [],
            "optimization_opportunities": [],
            "recommendations": {},
        }

        # Add basic results
        for result in results:
            result_data = {
                "memory_size": result.memory_size,
                "analysis": result.analysis,
                "cost_analysis": result.cost_analysis,
                "performance_summary": {
                    "avg_duration": result.analysis.get("avg_duration", 0),
                    "p95_duration": result.analysis.get("p95_duration", 0),
                    "avg_cost": result.cost_analysis.get("avg_cost_per_invocation", 0),
                    "cold_start_ratio": result.analysis.get("cold_start_ratio", 0),
                    "error_rate": result.analysis.get("error_rate", 0),
                },
            }
            report_data["results"].append(result_data)

        # Generate intelligent insights if components are available
        if self.recommendation_engine and self.pattern_recognizer:
            try:
                # Convert results to format expected by intelligence components
                memory_results = self._convert_results_to_memory_test_format(results)
                analysis = self._create_performance_analysis_from_results(results)

                # Generate ML recommendations
                ml_recommendation = self.recommendation_engine.generate_intelligent_recommendation(
                    analysis, memory_results
                )

                report_data["recommendations"] = {
                    "ml_recommendation": {
                        "optimal_memory_size": ml_recommendation.base_recommendation.optimal_memory_size,
                        "confidence_score": ml_recommendation.confidence_score,
                        "pattern_match_score": ml_recommendation.pattern_match_score,
                        "similar_functions": ml_recommendation.similar_functions,
                        "predicted_performance": ml_recommendation.predicted_performance,
                        "risk_assessment": ml_recommendation.risk_assessment,
                        "optimization_timeline": ml_recommendation.optimization_timeline,
                        "reasoning": ml_recommendation.base_recommendation.reasoning,
                    }
                }

                # Analyze performance patterns
                patterns = self.pattern_recognizer.analyze_performance_patterns(memory_results)

                report_data["performance_patterns"] = [
                    {
                        "pattern_type": pattern.pattern_type.value,
                        "description": pattern.description,
                        "confidence": pattern.confidence,
                        "impact_score": pattern.impact_score,
                        "recommendations": pattern.recommendations,
                        "evidence": pattern.evidence,
                    }
                    for pattern in patterns
                ]

                # Identify optimization opportunities
                current_config = {
                    "memory_size": config.memory_sizes[0] if config.memory_sizes else 1024,
                    "strategy": config.strategy,
                }

                opportunities = self.pattern_recognizer.identify_optimization_opportunities(
                    patterns, current_config
                )

                report_data["optimization_opportunities"] = opportunities

                # Generate intelligent insights summary
                report_data["intelligent_insights"] = self._generate_insights_summary(
                    ml_recommendation, patterns, opportunities
                )

            except Exception as e:
                report_data["intelligent_insights"] = {
                    "error": f"Failed to generate intelligent insights: {str(e)}",
                    "fallback_message": "Basic analysis completed successfully",
                }

        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligent_tuning_report_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2)

        return str(filepath)

    def generate_comprehensive_html_report(
        self, results: List[TuningResult], config: TunerConfig
    ) -> str:
        """Generate a comprehensive HTML report with intelligent insights.

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

        # Get intelligent insights
        intelligent_data = {}
        if self.recommendation_engine and self.pattern_recognizer:
            try:
                memory_results = self._convert_results_to_memory_test_format(results)
                analysis = self._create_performance_analysis_from_results(results)

                ml_recommendation = self.recommendation_engine.generate_intelligent_recommendation(
                    analysis, memory_results
                )
                patterns = self.pattern_recognizer.analyze_performance_patterns(memory_results)

                intelligent_data = {
                    "ml_recommendation": ml_recommendation,
                    "patterns": patterns,
                    "has_intelligence": True,
                }
            except:
                intelligent_data = {"has_intelligence": False}
        else:
            intelligent_data = {"has_intelligence": False}

        # Create HTML content
        html_content = self._create_comprehensive_html_content(
            results, config, performance_chart, cost_chart, comparison_chart, intelligent_data
        )

        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_report_{timestamp}.html"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(html_content)

        return str(filepath)

    def _convert_results_to_memory_test_format(self, results: List[TuningResult]) -> Dict[int, Any]:
        """Convert TuningResult objects to MemoryTestResult format."""
        from ..models import MemoryTestResult

        memory_results = {}

        for result in results:
            # Calculate aggregate metrics
            durations = [run.duration_ms for run in result.test_runs if not run.error]
            costs = [
                self._calculate_invocation_cost(
                    run.billed_duration_ms,
                    result.memory_size,
                    self.config.region if self.config else "us-east-1",
                )
                for run in result.test_runs
                if not run.error
            ]

            if durations:
                avg_duration = sum(durations) / len(durations) / 1000.0  # Convert to seconds
                p95_duration = sorted(durations)[int(len(durations) * 0.95)] / 1000.0
                p99_duration = sorted(durations)[int(len(durations) * 0.99)] / 1000.0
            else:
                avg_duration = p95_duration = p99_duration = 0.0

            avg_cost = sum(costs) / len(costs) if costs else 0.0
            total_cost = sum(costs)

            cold_starts = sum(1 for run in result.test_runs if run.cold_start)
            errors = sum(1 for run in result.test_runs if run.error)

            # Create MemoryTestResult-like object
            memory_result = type(
                "MemoryTestResult",
                (),
                {
                    "memory_size": result.memory_size,
                    "iterations": len(result.test_runs),
                    "avg_duration": avg_duration,
                    "p95_duration": p95_duration,
                    "p99_duration": p99_duration,
                    "avg_cost": avg_cost,
                    "total_cost": total_cost,
                    "cold_starts": cold_starts,
                    "errors": errors,
                    "raw_results": [
                        {
                            "duration": run.duration_ms / 1000.0,
                            "cost": self._calculate_invocation_cost(
                                run.billed_duration_ms,
                                result.memory_size,
                                self.config.region if self.config else "us-east-1",
                            ),
                            "cold_start": run.cold_start,
                            "error": run.error,
                        }
                        for run in result.test_runs
                    ],
                },
            )()

            memory_results[result.memory_size] = memory_result

        return memory_results

    def _create_performance_analysis_from_results(self, results: List[TuningResult]) -> Any:
        """Create PerformanceAnalysis object from results."""
        from ..models import PerformanceAnalysis

        memory_results = self._convert_results_to_memory_test_format(results)

        # Calculate efficiency scores
        efficiency_scores = {}
        for memory_size, result in memory_results.items():
            if result.avg_cost > 0 and result.avg_duration > 0:
                efficiency_scores[memory_size] = 1.0 / (result.avg_cost * result.avg_duration)
            else:
                efficiency_scores[memory_size] = 0.0

        # Find optimal configurations
        cost_optimal = min(memory_results.items(), key=lambda x: x[1].avg_cost)
        speed_optimal = min(memory_results.items(), key=lambda x: x[1].avg_duration)

        # Balanced optimal (simple scoring)
        balanced_scores = {
            memory: (1.0 / result.avg_duration if result.avg_duration > 0 else 0)
            + (1.0 / result.avg_cost if result.avg_cost > 0 else 0)
            for memory, result in memory_results.items()
        }
        balanced_optimal = max(balanced_scores.items(), key=lambda x: x[1])

        analysis = type(
            "PerformanceAnalysis",
            (),
            {
                "memory_results": memory_results,
                "efficiency_scores": efficiency_scores,
                "cost_optimal": {"memory_size": cost_optimal[0], "cost": cost_optimal[1].avg_cost},
                "speed_optimal": {
                    "memory_size": speed_optimal[0],
                    "duration": speed_optimal[1].avg_duration,
                },
                "balanced_optimal": {
                    "memory_size": balanced_optimal[0],
                    "score": balanced_optimal[1],
                },
                "trends": {},
                "insights": [],
            },
        )()

        return analysis

    def _generate_insights_summary(
        self,
        ml_recommendation: MLRecommendation,
        patterns: List[Any],
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a summary of intelligent insights."""
        return {
            "confidence_assessment": {
                "overall_confidence": ml_recommendation.confidence_score,
                "pattern_match_confidence": ml_recommendation.pattern_match_score,
                "recommendation_reliability": (
                    "high"
                    if ml_recommendation.confidence_score > 0.8
                    else "medium" if ml_recommendation.confidence_score > 0.6 else "low"
                ),
            },
            "key_findings": [
                f"Optimal memory configuration: {ml_recommendation.base_recommendation.optimal_memory_size}MB",
                f"Performance patterns identified: {len(patterns)}",
                f"Optimization opportunities found: {len(opportunities)}",
                f"Similar functions in database: {len(ml_recommendation.similar_functions)}",
            ],
            "risk_factors": ml_recommendation.risk_assessment.get("factors", []),
            "implementation_priority": ml_recommendation.risk_assessment.get("level", "unknown"),
            "estimated_impact": {
                "cost_change": ml_recommendation.base_recommendation.cost_change_percent,
                "duration_change": ml_recommendation.base_recommendation.duration_change_percent,
            },
        }

    def _create_comprehensive_html_content(
        self,
        results: List[TuningResult],
        config: TunerConfig,
        performance_chart: str,
        cost_chart: str,
        comparison_chart: str,
        intelligent_data: Dict[str, Any],
    ) -> str:
        """Create comprehensive HTML report content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AWS Lambda Tuning Report - Intelligent Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #232f3e; color: white; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .intelligence-section { background-color: #f8f9fa; }
                .pattern { margin: 10px 0; padding: 10px; background-color: #e9ecef; border-radius: 3px; }
                .recommendation { background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .high-confidence { border-left: 5px solid #28a745; }
                .medium-confidence { border-left: 5px solid #ffc107; }
                .low-confidence { border-left: 5px solid #dc3545; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                .metric { text-align: center; padding: 10px; background-color: #f1f3f4; border-radius: 5px; }
                .chart { text-align: center; margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AWS Lambda Performance Tuning Report</h1>
                <p>Function: {function_arn}</p>
                <p>Generated: {timestamp}</p>
                <p>Strategy: {strategy}</p>
            </div>
            
            {intelligence_section}
            
            <div class="section">
                <h2>Performance Analysis</h2>
                <div class="chart">{performance_chart}</div>
                <div class="chart">{cost_chart}</div>
                <div class="chart">{comparison_chart}</div>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                {results_table}
            </div>
            
            <div class="section">
                <h2>Configuration Summary</h2>
                <p><strong>Memory sizes tested:</strong> {memory_sizes}</p>
                <p><strong>Iterations per configuration:</strong> {iterations}</p>
                <p><strong>Total test runs:</strong> {total_runs}</p>
            </div>
        </body>
        </html>
        """

        # Build intelligence section
        intelligence_section = ""
        if intelligent_data.get("has_intelligence"):
            ml_rec = intelligent_data.get("ml_recommendation")
            patterns = intelligent_data.get("patterns", [])

            confidence_class = (
                "high-confidence"
                if ml_rec.confidence_score > 0.8
                else "medium-confidence" if ml_rec.confidence_score > 0.6 else "low-confidence"
            )

            intelligence_section = f"""
            <div class="section intelligence-section">
                <h2>🧠 Intelligent Analysis</h2>
                
                <div class="recommendation {confidence_class}">
                    <h3>ML-Based Recommendation</h3>
                    <p><strong>Optimal Memory Size:</strong> {ml_rec.base_recommendation.optimal_memory_size}MB</p>
                    <p><strong>Confidence Score:</strong> {ml_rec.confidence_score:.2%}</p>
                    <p><strong>Expected Cost Change:</strong> {ml_rec.base_recommendation.cost_change_percent:+.1f}%</p>
                    <p><strong>Expected Duration Change:</strong> {ml_rec.base_recommendation.duration_change_percent:+.1f}%</p>
                    <p><strong>Reasoning:</strong> {ml_rec.base_recommendation.reasoning}</p>
                </div>
                
                <h3>Performance Patterns Detected</h3>
                {"".join([f'<div class="pattern"><strong>{pattern.pattern_type.value.title()}:</strong> {pattern.description} (Confidence: {pattern.confidence:.2%})</div>' for pattern in patterns[:5]])}
                
                <h3>Risk Assessment</h3>
                <p><strong>Risk Level:</strong> {ml_rec.risk_assessment.get('level', 'Unknown')}</p>
                <p><strong>Risk Factors:</strong> {', '.join(ml_rec.risk_assessment.get('factors', ['None identified']))}</p>
                
                <h3>Similar Functions</h3>
                <p>Found {len(ml_rec.similar_functions)} similar function(s) for pattern matching</p>
            </div>
            """
        else:
            intelligence_section = """
            <div class="section">
                <h2>Basic Analysis</h2>
                <p>Intelligent analysis features not available. Showing basic performance metrics.</p>
            </div>
            """

        # Build results table
        results_table = "<table><tr><th>Memory (MB)</th><th>Avg Duration (ms)</th><th>P95 Duration (ms)</th><th>Cost per Invocation</th><th>Cold Starts</th><th>Error Rate</th></tr>"

        for result in results:
            avg_duration = result.analysis.get("avg_duration", 0)
            p95_duration = result.analysis.get("p95_duration", 0)
            avg_cost = result.cost_analysis.get("avg_cost_per_invocation", 0)
            cold_start_ratio = result.analysis.get("cold_start_ratio", 0)
            error_rate = result.analysis.get("error_rate", 0)

            results_table += f"""
            <tr>
                <td>{result.memory_size}</td>
                <td>{avg_duration:.2f}</td>
                <td>{p95_duration:.2f}</td>
                <td>${avg_cost:.6f}</td>
                <td>{cold_start_ratio:.1%}</td>
                <td>{error_rate:.1%}</td>
            </tr>
            """

        results_table += "</table>"

        # Fill template
        total_runs = sum(len(result.test_runs) for result in results)

        return html_template.format(
            function_arn=config.function_arn,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            strategy=config.strategy,
            intelligence_section=intelligence_section,
            performance_chart=performance_chart,
            cost_chart=cost_chart,
            comparison_chart=comparison_chart,
            results_table=results_table,
            memory_sizes=", ".join(str(r.memory_size) for r in results),
            iterations=config.iterations,
            total_runs=total_runs,
        )
