"""
Report generation service for AWS Lambda Tuner.
Enhanced with workload-specific reporting and cost projections.
"""

import json
import csv
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import logging
from enum import Enum

from .utils import (
    calculate_statistics, format_duration, calculate_cost,
    save_json_file, format_timestamp, safe_divide
)
from .exceptions import ReportGenerationError

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Supported workload types for specialized reporting."""
    WEB_API = "web_api"
    BATCH_PROCESSING = "batch_processing"  
    EVENT_DRIVEN = "event_driven"
    SCHEDULED = "scheduled"
    STREAM_PROCESSING = "stream_processing"
    

class TrafficPattern(Enum):
    """Traffic patterns for cost projection."""
    STEADY = "steady"
    BURSTY = "bursty"
    SEASONAL = "seasonal"
    GROWTH = "growth"


class ReportGenerator:
    """Generates various reports from tuning results."""
    
    def __init__(self, results: Dict[str, Any], config: Optional[Any] = None, 
                 workload_type: Optional[WorkloadType] = None):
        """
        Initialize report generator.
        
        Args:
            results: Tuning results dictionary
            config: Optional tuner configuration
            workload_type: Optional workload type for specialized reporting
        """
        self.results = results
        self.config = config
        self.workload_type = workload_type
        self.timestamp = format_timestamp()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary report.
        
        Returns:
            Summary statistics dictionary
        """
        try:
            # Find optimal configuration
            optimal = self._find_optimal_configuration()
            
            # Calculate baseline (first memory size tested)
            baseline = self._get_baseline_stats()
            
            # Calculate improvements
            performance_gain = safe_divide(
                baseline['avg_duration'] - optimal['avg_duration'],
                baseline['avg_duration']
            ) * 100
            
            cost_savings = safe_divide(
                baseline['avg_cost'] - optimal['avg_cost'],
                baseline['avg_cost']
            ) * 100
            
            return {
                'optimal_memory': optimal['memory_mb'],
                'optimal_duration': optimal['avg_duration'],
                'optimal_cost': optimal['avg_cost'],
                'performance_gain': performance_gain,
                'cost_savings': cost_savings,
                'total_invocations': self._count_total_invocations(),
                'test_duration': self.results.get('test_duration_seconds', 0),
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise ReportGenerationError(f"Failed to generate summary: {e}")
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Generate a detailed report with all metrics.
        
        Returns:
            Detailed report dictionary
        """
        try:
            report = {
                'metadata': {
                    'timestamp': self.timestamp,
                    'function_arn': self.results.get('function_arn', 'unknown'),
                    'test_configuration': self._get_test_configuration()
                },
                'summary': self.get_summary(),
                'configurations': []
            }
            
            # Process each memory configuration
            for memory_config in self.results.get('configurations', []):
                config_report = self._process_configuration(memory_config)
                report['configurations'].append(config_report)
            
            # Add recommendations
            report['recommendations'] = self._generate_recommendations()
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
            raise ReportGenerationError(f"Failed to generate detailed report: {e}")
    
    def save_json(self, filepath: str):
        """Save report as JSON."""
        report = self.get_detailed_report()
        save_json_file(report, filepath)
        logger.info(f"JSON report saved to {filepath}")
    
    def save_csv(self, filepath: str):
        """Save report as CSV."""
        try:
            rows = []
            
            # Process each configuration
            for config in self.results.get('configurations', []):
                memory_mb = config['memory_mb']
                
                for execution in config.get('executions', []):
                    row = {
                        'memory_mb': memory_mb,
                        'duration_ms': execution.get('duration', 0),
                        'billed_duration_ms': execution.get('billed_duration', 0),
                        'cost_usd': calculate_cost(memory_mb, execution.get('billed_duration', 0)),
                        'cold_start': execution.get('cold_start', False),
                        'error': execution.get('error', ''),
                        'timestamp': execution.get('timestamp', '')
                    }
                    rows.append(row)
            
            # Write CSV
            if rows:
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                
                logger.info(f"CSV report saved to {filepath}")
            else:
                logger.warning("No data to save in CSV report")
                
        except Exception as e:
            logger.error(f"Error saving CSV report: {e}")
            raise ReportGenerationError(f"Failed to save CSV report: {e}")
    
    def save_html(self, filepath: str):
        """Save report as HTML."""
        try:
            report = self.get_detailed_report()
            summary = report['summary']
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Lambda Tuning Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .optimal {{ background-color: #d4edda; }}
        .recommendation {{ background: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>AWS Lambda Performance Tuning Report</h1>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric">
            <span class="metric-label">Optimal Memory Configuration:</span> {summary['optimal_memory']}MB
        </div>
        <div class="metric">
            <span class="metric-label">Average Duration:</span> {summary['optimal_duration']:.2f}ms
        </div>
        <div class="metric">
            <span class="metric-label">Cost per Invocation:</span> ${summary['optimal_cost']:.6f}
        </div>
        <div class="metric">
            <span class="metric-label">Performance Improvement:</span> {summary['performance_gain']:.1f}%
        </div>
        <div class="metric">
            <span class="metric-label">Cost Savings:</span> {summary['cost_savings']:.1f}%
        </div>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Memory (MB)</th>
            <th>Avg Duration (ms)</th>
            <th>Min Duration (ms)</th>
            <th>Max Duration (ms)</th>
            <th>P95 Duration (ms)</th>
            <th>Avg Cost ($)</th>
            <th>Success Rate (%)</th>
        </tr>
"""
            
            # Add configuration rows
            optimal_memory = summary['optimal_memory']
            for config in report['configurations']:
                is_optimal = config['memory_mb'] == optimal_memory
                row_class = 'class="optimal"' if is_optimal else ''
                
                html_content += f"""
        <tr {row_class}>
            <td>{config['memory_mb']}</td>
            <td>{config['statistics']['duration']['mean']:.2f}</td>
            <td>{config['statistics']['duration']['min']:.2f}</td>
            <td>{config['statistics']['duration']['max']:.2f}</td>
            <td>{config['statistics']['duration']['p95']:.2f}</td>
            <td>${config['statistics']['cost']['mean']:.6f}</td>
            <td>{config['success_rate']:.1f}</td>
        </tr>
"""
            
            html_content += """
    </table>
    
    <h2>Recommendations</h2>
"""
            
            # Add recommendations
            for rec in report['recommendations']:
                html_content += f"""
    <div class="recommendation">
        <strong>{rec['title']}:</strong> {rec['description']}
    </div>
"""
            
            html_content += f"""
    <hr>
    <p><small>Generated on {self.timestamp}</small></p>
</body>
</html>
"""
            
            # Save HTML
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving HTML report: {e}")
            raise ReportGenerationError(f"Failed to save HTML report: {e}")
    
    def _find_optimal_configuration(self) -> Dict[str, Any]:
        """Find the optimal memory configuration based on strategy."""
        configurations = self.results.get('configurations', [])
        if not configurations:
            raise ReportGenerationError("No configurations found in results")
        
        strategy = self.config.strategy if self.config else 'balanced'
        
        # Calculate scores for each configuration
        scored_configs = []
        for config in configurations:
            stats = self._calculate_config_statistics(config)
            
            if strategy == 'speed':
                # Optimize for speed (lower duration is better)
                score = -stats['avg_duration']
            elif strategy == 'cost':
                # Optimize for cost (lower cost is better)
                score = -stats['avg_cost']
            else:  # balanced
                # Balance between speed and cost
                # Normalize both metrics and combine
                duration_score = 1 / (1 + stats['avg_duration'] / 1000)  # Convert to seconds
                cost_score = 1 / (1 + stats['avg_cost'] * 1000000)  # Scale up cost
                score = duration_score + cost_score
            
            scored_configs.append({
                'memory_mb': config['memory_mb'],
                'avg_duration': stats['avg_duration'],
                'avg_cost': stats['avg_cost'],
                'score': score
            })
        
        # Return configuration with highest score
        return max(scored_configs, key=lambda x: x['score'])
    
    def _get_baseline_stats(self) -> Dict[str, Any]:
        """Get baseline statistics (first configuration)."""
        configurations = self.results.get('configurations', [])
        if not configurations:
            return {'avg_duration': 0, 'avg_cost': 0}
        
        first_config = configurations[0]
        stats = self._calculate_config_statistics(first_config)
        
        return {
            'memory_mb': first_config['memory_mb'],
            'avg_duration': stats['avg_duration'],
            'avg_cost': stats['avg_cost']
        }
    
    def _calculate_config_statistics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for a memory configuration."""
        executions = config.get('executions', [])
        
        if not executions:
            return {'avg_duration': 0, 'avg_cost': 0}
        
        # Extract successful executions
        successful = [e for e in executions if not e.get('error')]
        
        if not successful:
            return {'avg_duration': 0, 'avg_cost': 0}
        
        durations = [e['duration'] for e in successful]
        costs = [calculate_cost(config['memory_mb'], e['billed_duration']) for e in successful]
        
        return {
            'avg_duration': statistics.mean(durations),
            'avg_cost': statistics.mean(costs)
        }
    
    def _count_total_invocations(self) -> int:
        """Count total number of invocations."""
        total = 0
        for config in self.results.get('configurations', []):
            total += len(config.get('executions', []))
        return total
    
    def _get_test_configuration(self) -> Dict[str, Any]:
        """Get test configuration details."""
        if self.config:
            return {
                'strategy': self.config.strategy,
                'iterations': self.config.iterations,
                'memory_sizes': self.config.memory_sizes,
                'concurrent_executions': self.config.concurrent_executions
            }
        return {}
    
    def _process_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single memory configuration."""
        executions = config.get('executions', [])
        
        # Separate successful and failed executions
        successful = [e for e in executions if not e.get('error')]
        failed = [e for e in executions if e.get('error')]
        
        # Calculate statistics
        if successful:
            durations = [e['duration'] for e in successful]
            billed_durations = [e['billed_duration'] for e in successful]
            costs = [calculate_cost(config['memory_mb'], bd) for bd in billed_durations]
            
            duration_stats = calculate_statistics(durations)
            cost_stats = calculate_statistics(costs)
        else:
            duration_stats = calculate_statistics([])
            cost_stats = calculate_statistics([])
        
        return {
            'memory_mb': config['memory_mb'],
            'total_executions': len(executions),
            'successful_executions': len(successful),
            'failed_executions': len(failed),
            'success_rate': safe_divide(len(successful), len(executions)) * 100,
            'statistics': {
                'duration': duration_stats,
                'cost': cost_stats
            },
            'cold_starts': sum(1 for e in successful if e.get('cold_start', False))
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Get optimal configuration
        optimal = self._find_optimal_configuration()
        
        # Memory recommendation
        recommendations.append({
            'title': 'Optimal Memory Configuration',
            'description': f"Set your Lambda function memory to {optimal['memory_mb']}MB for the best {self.config.strategy if self.config else 'balanced'} performance."
        })
        
        # Cost impact
        baseline = self._get_baseline_stats()
        if baseline['avg_cost'] > optimal['avg_cost']:
            monthly_savings = (baseline['avg_cost'] - optimal['avg_cost']) * 30 * 24 * 60 * 2  # Assume 2 invocations per minute
            recommendations.append({
                'title': 'Cost Savings',
                'description': f"This configuration could save approximately ${monthly_savings:.2f} per month (assuming 2 invocations per minute)."
            })
        
        # Performance impact
        if baseline['avg_duration'] > optimal['avg_duration']:
            time_saved = baseline['avg_duration'] - optimal['avg_duration']
            recommendations.append({
                'title': 'Performance Improvement',
                'description': f"This configuration reduces average execution time by {time_saved:.2f}ms ({safe_divide(time_saved, baseline['avg_duration']) * 100:.1f}% improvement)."
            })
        
        # Cold start analysis
        total_cold_starts = sum(
            c.get('cold_starts', 0) 
            for c in self.results.get('configurations', [])
        )
        if total_cold_starts > 0:
            recommendations.append({
                'title': 'Cold Start Optimization',
                'description': f"Consider using provisioned concurrency or scheduled warming to reduce cold starts (observed {total_cold_starts} cold starts during testing)."
            })
        
        return recommendations

    def workload_specific_reports(self) -> Dict[str, Any]:
        """
        Generate workload-specific performance reports and recommendations.
        
        Returns:
            Workload-specific analysis and recommendations
        """
        try:
            if not self.workload_type:
                logger.warning("No workload type specified, using generic analysis")
                return self._generic_workload_analysis()
            
            # Get base metrics
            optimal = self._find_optimal_configuration()
            baseline = self._get_baseline_stats()
            
            # Workload-specific analysis
            if self.workload_type == WorkloadType.WEB_API:
                return self._web_api_analysis(optimal, baseline)
            elif self.workload_type == WorkloadType.BATCH_PROCESSING:
                return self._batch_processing_analysis(optimal, baseline)
            elif self.workload_type == WorkloadType.EVENT_DRIVEN:
                return self._event_driven_analysis(optimal, baseline)
            elif self.workload_type == WorkloadType.SCHEDULED:
                return self._scheduled_analysis(optimal, baseline)
            elif self.workload_type == WorkloadType.STREAM_PROCESSING:
                return self._stream_processing_analysis(optimal, baseline)
            else:
                return self._generic_workload_analysis()
                
        except Exception as e:
            logger.error(f"Error generating workload-specific report: {e}")
            raise ReportGenerationError(f"Failed to generate workload-specific report: {e}")

    def cost_projection_reports(self, traffic_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate cost projections for different traffic scenarios.
        
        Args:
            traffic_scenarios: List of scenarios with invocations/day, pattern, duration
            
        Returns:
            Cost projections for each scenario
        """
        try:
            optimal = self._find_optimal_configuration()
            projections = {}
            
            for scenario in traffic_scenarios:
                scenario_name = scenario.get('name', f"scenario_{len(projections) + 1}")
                daily_invocations = scenario.get('daily_invocations', 10000)
                pattern = TrafficPattern(scenario.get('pattern', 'steady'))
                duration_days = scenario.get('duration_days', 30)
                
                projection = self._calculate_cost_projection(
                    optimal, daily_invocations, pattern, duration_days
                )
                
                projections[scenario_name] = projection
                
            return {
                'optimal_configuration': optimal,
                'projections': projections,
                'comparison_baseline': self._get_baseline_cost_projection(traffic_scenarios),
                'savings_analysis': self._calculate_savings_analysis(projections),
                'recommendations': self._generate_cost_recommendations(projections)
            }
            
        except Exception as e:
            logger.error(f"Error generating cost projections: {e}")
            raise ReportGenerationError(f"Failed to generate cost projections: {e}")

    def comparative_analysis_reports(self, comparison_workloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparative analysis across different workload types.
        
        Args:
            comparison_workloads: List of workload configurations to compare
            
        Returns:
            Comparative analysis report
        """
        try:
            current_analysis = self.workload_specific_reports()
            comparisons = []
            
            for workload in comparison_workloads:
                workload_type = WorkloadType(workload.get('type', 'web_api'))
                workload_results = workload.get('results', self.results)
                
                # Create temporary generator for comparison
                temp_generator = ReportGenerator(
                    workload_results, 
                    self.config, 
                    workload_type
                )
                
                workload_analysis = temp_generator.workload_specific_reports()
                comparisons.append({
                    'workload_type': workload_type.value,
                    'name': workload.get('name', workload_type.value),
                    'analysis': workload_analysis
                })
            
            return {
                'current_workload': {
                    'type': self.workload_type.value if self.workload_type else 'generic',
                    'analysis': current_analysis
                },
                'comparisons': comparisons,
                'cross_workload_insights': self._generate_cross_workload_insights(comparisons),
                'best_practices': self._generate_workload_best_practices(comparisons)
            }
            
        except Exception as e:
            logger.error(f"Error generating comparative analysis: {e}")
            raise ReportGenerationError(f"Failed to generate comparative analysis: {e}")

    def _web_api_analysis(self, optimal: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis specific to web API workloads."""
        cold_start_impact = self._analyze_cold_start_impact()
        latency_percentiles = self._calculate_latency_percentiles()
        
        return {
            'workload_type': 'web_api',
            'key_metrics': {
                'p95_latency': latency_percentiles.get('p95', 0),
                'p99_latency': latency_percentiles.get('p99', 0),
                'cold_start_percentage': cold_start_impact.get('percentage', 0),
                'cold_start_avg_penalty': cold_start_impact.get('avg_penalty_ms', 0)
            },
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'latency',
                    'description': f"For web APIs, consider memory setting of {optimal['memory_mb']}MB to achieve p95 latency of {latency_percentiles.get('p95', 0):.1f}ms"
                },
                {
                    'priority': 'medium' if cold_start_impact.get('percentage', 0) < 5 else 'high',
                    'category': 'cold_starts',
                    'description': f"Cold starts affect {cold_start_impact.get('percentage', 0):.1f}% of requests. Consider provisioned concurrency if this exceeds 10%"
                }
            ],
            'scaling_recommendations': self._get_web_api_scaling_recommendations(optimal)
        }

    def _batch_processing_analysis(self, optimal: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis specific to batch processing workloads."""
        throughput_analysis = self._analyze_throughput()
        cost_efficiency = self._analyze_cost_efficiency(optimal, baseline)
        
        return {
            'workload_type': 'batch_processing',
            'key_metrics': {
                'avg_duration': optimal['avg_duration'],
                'cost_per_execution': optimal['avg_cost'],
                'throughput_improvement': throughput_analysis.get('improvement_percentage', 0),
                'cost_efficiency_ratio': cost_efficiency.get('ratio', 0)
            },
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'throughput',
                    'description': f"For batch processing, {optimal['memory_mb']}MB provides optimal cost-throughput balance"
                },
                {
                    'priority': 'medium',
                    'category': 'parallelization',
                    'description': "Consider increasing concurrent executions for large batch jobs to maximize throughput"
                }
            ],
            'scaling_recommendations': self._get_batch_scaling_recommendations(optimal)
        }

    def _event_driven_analysis(self, optimal: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis specific to event-driven workloads."""
        event_processing_stats = self._analyze_event_processing()
        
        return {
            'workload_type': 'event_driven',
            'key_metrics': {
                'avg_processing_time': optimal['avg_duration'],
                'event_success_rate': event_processing_stats.get('success_rate', 0),
                'memory_utilization': event_processing_stats.get('memory_efficiency', 0)
            },
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'reliability',
                    'description': f"Optimal memory of {optimal['memory_mb']}MB ensures reliable event processing"
                },
                {
                    'priority': 'medium',
                    'category': 'error_handling',
                    'description': "Consider implementing dead letter queues for failed event processing"
                }
            ],
            'scaling_recommendations': self._get_event_driven_scaling_recommendations(optimal)
        }

    def _scheduled_analysis(self, optimal: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis specific to scheduled workloads."""
        execution_consistency = self._analyze_execution_consistency()
        
        return {
            'workload_type': 'scheduled',
            'key_metrics': {
                'execution_consistency': execution_consistency.get('coefficient_of_variation', 0),
                'cost_predictability': execution_consistency.get('cost_variance', 0),
                'cold_start_frequency': execution_consistency.get('cold_start_rate', 0)
            },
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'consistency',
                    'description': f"Memory setting of {optimal['memory_mb']}MB provides consistent execution times for scheduled tasks"
                },
                {
                    'priority': 'low',
                    'category': 'warming',
                    'description': "For critical scheduled jobs, consider warming strategies to reduce cold starts"
                }
            ],
            'scaling_recommendations': self._get_scheduled_scaling_recommendations(optimal)
        }

    def _stream_processing_analysis(self, optimal: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis specific to stream processing workloads."""
        stream_metrics = self._analyze_stream_processing()
        
        return {
            'workload_type': 'stream_processing',
            'key_metrics': {
                'processing_rate': stream_metrics.get('records_per_second', 0),
                'latency_stability': stream_metrics.get('latency_variance', 0),
                'memory_efficiency': stream_metrics.get('memory_utilization', 0)
            },
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'throughput',
                    'description': f"Memory configuration of {optimal['memory_mb']}MB optimizes stream processing throughput"
                },
                {
                    'priority': 'medium',
                    'category': 'parallelization',
                    'description': "Consider shard-level parallelization for high-volume streams"
                }
            ],
            'scaling_recommendations': self._get_stream_scaling_recommendations(optimal)
        }

    def _generic_workload_analysis(self) -> Dict[str, Any]:
        """Generate generic workload analysis when type is not specified."""
        optimal = self._find_optimal_configuration()
        baseline = self._get_baseline_stats()
        
        return {
            'workload_type': 'generic',
            'key_metrics': {
                'optimal_memory': optimal['memory_mb'],
                'performance_gain': safe_divide(baseline['avg_duration'] - optimal['avg_duration'], baseline['avg_duration']) * 100,
                'cost_savings': safe_divide(baseline['avg_cost'] - optimal['avg_cost'], baseline['avg_cost']) * 100
            },
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'optimization',
                    'description': f"Use {optimal['memory_mb']}MB for optimal performance"
                }
            ],
            'scaling_recommendations': []
        }

    def _calculate_cost_projection(self, optimal: Dict[str, Any], daily_invocations: int, 
                                 pattern: TrafficPattern, duration_days: int) -> Dict[str, Any]:
        """Calculate cost projection for a traffic scenario."""
        base_cost_per_invocation = optimal['avg_cost']
        
        # Adjust for traffic pattern
        pattern_multiplier = {
            TrafficPattern.STEADY: 1.0,
            TrafficPattern.BURSTY: 1.2,  # Higher costs due to cold starts
            TrafficPattern.SEASONAL: 1.1,
            TrafficPattern.GROWTH: 1.05
        }.get(pattern, 1.0)
        
        adjusted_cost = base_cost_per_invocation * pattern_multiplier
        daily_cost = daily_invocations * adjusted_cost
        total_cost = daily_cost * duration_days
        
        # Calculate monthly and yearly projections
        monthly_cost = daily_cost * 30
        yearly_cost = daily_cost * 365
        
        return {
            'cost_per_invocation': adjusted_cost,
            'daily_cost': daily_cost,
            'monthly_cost': monthly_cost,
            'yearly_cost': yearly_cost,
            'total_cost': total_cost,
            'pattern_impact': pattern_multiplier,
            'invocations_per_day': daily_invocations,
            'duration_days': duration_days
        }

    def _get_baseline_cost_projection(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate baseline cost projections using the first memory configuration."""
        baseline = self._get_baseline_stats()
        projections = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', f"baseline_scenario_{len(projections) + 1}")
            daily_invocations = scenario.get('daily_invocations', 10000)
            pattern = TrafficPattern(scenario.get('pattern', 'steady'))
            duration_days = scenario.get('duration_days', 30)
            
            projection = self._calculate_cost_projection(
                baseline, daily_invocations, pattern, duration_days
            )
            projections[scenario_name] = projection
            
        return projections

    def _calculate_savings_analysis(self, projections: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate savings analysis across all projections."""
        baseline_projections = self._get_baseline_cost_projection([
            {'name': name, 'daily_invocations': proj['invocations_per_day'], 
             'pattern': 'steady', 'duration_days': proj['duration_days']}
            for name, proj in projections.items()
        ])
        
        total_savings = 0
        scenario_savings = {}
        
        for name, optimal_proj in projections.items():
            if name in baseline_projections:
                baseline_proj = baseline_projections[name]
                savings = baseline_proj['total_cost'] - optimal_proj['total_cost']
                savings_percentage = safe_divide(savings, baseline_proj['total_cost']) * 100
                
                scenario_savings[name] = {
                    'absolute_savings': savings,
                    'percentage_savings': savings_percentage,
                    'baseline_cost': baseline_proj['total_cost'],
                    'optimized_cost': optimal_proj['total_cost']
                }
                total_savings += savings
        
        return {
            'total_savings': total_savings,
            'scenario_savings': scenario_savings,
            'average_savings_percentage': statistics.mean([
                s['percentage_savings'] for s in scenario_savings.values()
            ]) if scenario_savings else 0
        }

    def _generate_cost_recommendations(self, projections: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        total_monthly_cost = sum(proj['monthly_cost'] for proj in projections.values())
        
        if total_monthly_cost > 1000:
            recommendations.append({
                'priority': 'high',
                'category': 'cost_management',
                'description': f"High monthly cost projection (${total_monthly_cost:.2f}). Consider implementing cost monitoring and alerts."
            })
        
        # Check for expensive scenarios
        for name, proj in projections.items():
            if proj['cost_per_invocation'] > 0.001:  # $0.001 per invocation
                recommendations.append({
                    'priority': 'medium',
                    'category': 'optimization',
                    'description': f"Scenario '{name}' has high per-invocation cost (${proj['cost_per_invocation']:.6f}). Review memory allocation."
                })
        
        return recommendations

    def _analyze_cold_start_impact(self) -> Dict[str, Any]:
        """Analyze cold start impact across configurations."""
        total_executions = 0
        total_cold_starts = 0
        cold_start_durations = []
        warm_start_durations = []
        
        for config in self.results.get('configurations', []):
            executions = config.get('executions', [])
            for execution in executions:
                if not execution.get('error'):
                    total_executions += 1
                    if execution.get('cold_start'):
                        total_cold_starts += 1
                        cold_start_durations.append(execution['duration'])
                    else:
                        warm_start_durations.append(execution['duration'])
        
        cold_start_percentage = safe_divide(total_cold_starts, total_executions) * 100
        avg_cold_duration = statistics.mean(cold_start_durations) if cold_start_durations else 0
        avg_warm_duration = statistics.mean(warm_start_durations) if warm_start_durations else 0
        
        return {
            'percentage': cold_start_percentage,
            'total_cold_starts': total_cold_starts,
            'avg_cold_duration': avg_cold_duration,
            'avg_warm_duration': avg_warm_duration,
            'avg_penalty_ms': max(0, avg_cold_duration - avg_warm_duration)
        }

    def _calculate_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles across all configurations."""
        all_durations = []
        
        for config in self.results.get('configurations', []):
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            durations = [e['duration'] for e in successful]
            all_durations.extend(durations)
        
        if not all_durations:
            return {}
        
        all_durations.sort()
        n = len(all_durations)
        
        return {
            'p50': all_durations[int(0.5 * n)] if n > 0 else 0,
            'p90': all_durations[int(0.9 * n)] if n > 0 else 0,
            'p95': all_durations[int(0.95 * n)] if n > 0 else 0,
            'p99': all_durations[int(0.99 * n)] if n > 0 else 0
        }

    def _analyze_throughput(self) -> Dict[str, Any]:
        """Analyze throughput characteristics."""
        # Calculate invocations per second for each configuration
        throughput_data = []
        
        for config in self.results.get('configurations', []):
            executions = config.get('executions', [])
            if executions:
                # Estimate throughput based on average duration
                successful = [e for e in executions if not e.get('error')]
                if successful:
                    avg_duration_seconds = statistics.mean([e['duration'] for e in successful]) / 1000
                    estimated_throughput = 1 / avg_duration_seconds if avg_duration_seconds > 0 else 0
                    throughput_data.append(estimated_throughput)
        
        if not throughput_data:
            return {}
        
        max_throughput = max(throughput_data)
        min_throughput = min(throughput_data)
        improvement = safe_divide(max_throughput - min_throughput, min_throughput) * 100
        
        return {
            'max_throughput': max_throughput,
            'min_throughput': min_throughput,
            'improvement_percentage': improvement
        }

    def _analyze_cost_efficiency(self, optimal: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost efficiency metrics."""
        cost_per_ms = safe_divide(optimal['avg_cost'], optimal['avg_duration'])
        baseline_cost_per_ms = safe_divide(baseline['avg_cost'], baseline['avg_duration'])
        
        efficiency_ratio = safe_divide(baseline_cost_per_ms, cost_per_ms)
        
        return {
            'cost_per_ms': cost_per_ms,
            'baseline_cost_per_ms': baseline_cost_per_ms,
            'ratio': efficiency_ratio
        }

    def _analyze_event_processing(self) -> Dict[str, Any]:
        """Analyze event processing characteristics."""
        total_executions = 0
        successful_executions = 0
        
        for config in self.results.get('configurations', []):
            executions = config.get('executions', [])
            total_executions += len(executions)
            successful_executions += len([e for e in executions if not e.get('error')])
        
        success_rate = safe_divide(successful_executions, total_executions) * 100
        
        return {
            'success_rate': success_rate,
            'total_events': total_executions,
            'memory_efficiency': 85.0  # Placeholder - would need memory metrics
        }

    def _analyze_execution_consistency(self) -> Dict[str, Any]:
        """Analyze execution consistency for scheduled workloads."""
        all_durations = []
        
        for config in self.results.get('configurations', []):
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            durations = [e['duration'] for e in successful]
            all_durations.extend(durations)
        
        if len(all_durations) < 2:
            return {}
        
        mean_duration = statistics.mean(all_durations)
        stdev_duration = statistics.stdev(all_durations)
        coefficient_of_variation = safe_divide(stdev_duration, mean_duration)
        
        return {
            'coefficient_of_variation': coefficient_of_variation,
            'cost_variance': coefficient_of_variation * 0.8,  # Estimate
            'cold_start_rate': 5.0  # Placeholder
        }

    def _analyze_stream_processing(self) -> Dict[str, Any]:
        """Analyze stream processing characteristics."""
        # Placeholder implementation - would need stream-specific metrics
        return {
            'records_per_second': 1000,  # Estimated
            'latency_variance': 0.1,
            'memory_utilization': 75.0
        }

    def _get_web_api_scaling_recommendations(self, optimal: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get scaling recommendations for web API workloads."""
        return [
            {
                'metric': 'Concurrent Executions',
                'recommendation': 'Set reserved concurrency based on expected peak traffic'
            },
            {
                'metric': 'Provisioned Concurrency',
                'recommendation': 'Use provisioned concurrency for consistent sub-100ms response times'
            }
        ]

    def _get_batch_scaling_recommendations(self, optimal: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get scaling recommendations for batch processing workloads."""
        return [
            {
                'metric': 'Parallel Processing',
                'recommendation': 'Increase concurrent executions for large batch jobs'
            },
            {
                'metric': 'Memory Allocation',
                'recommendation': f"Use {optimal['memory_mb']}MB for optimal cost-performance balance"
            }
        ]

    def _get_event_driven_scaling_recommendations(self, optimal: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get scaling recommendations for event-driven workloads."""
        return [
            {
                'metric': 'Event Source Mapping',
                'recommendation': 'Tune batch size and maximum batching window for optimal throughput'
            },
            {
                'metric': 'Error Handling',
                'recommendation': 'Configure dead letter queues and retry policies'
            }
        ]

    def _get_scheduled_scaling_recommendations(self, optimal: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get scaling recommendations for scheduled workloads."""
        return [
            {
                'metric': 'Schedule Optimization',
                'recommendation': 'Distribute scheduled jobs to avoid resource contention'
            },
            {
                'metric': 'Timeout Settings',
                'recommendation': 'Set appropriate timeout values based on execution patterns'
            }
        ]

    def _get_stream_scaling_recommendations(self, optimal: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get scaling recommendations for stream processing workloads."""
        return [
            {
                'metric': 'Shard Processing',
                'recommendation': 'Configure parallelization factor based on shard count'
            },
            {
                'metric': 'Batch Processing',
                'recommendation': 'Optimize batch size for stream processing efficiency'
            }
        ]

    def _generate_cross_workload_insights(self, comparisons: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate insights from cross-workload analysis."""
        insights = []
        
        # Analyze memory requirements across workloads
        memory_configs = []
        for comp in comparisons:
            analysis = comp.get('analysis', {})
            key_metrics = analysis.get('key_metrics', {})
            if 'optimal_memory' in key_metrics:
                memory_configs.append(key_metrics['optimal_memory'])
        
        if memory_configs:
            avg_memory = statistics.mean(memory_configs)
            insights.append({
                'category': 'memory_patterns',
                'insight': f"Average optimal memory across workloads: {avg_memory:.0f}MB"
            })
        
        return insights

    def _generate_workload_best_practices(self, comparisons: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate best practices based on workload comparisons."""
        return [
            {
                'category': 'memory_optimization',
                'practice': 'Test memory configurations specific to your workload type for optimal results'
            },
            {
                'category': 'monitoring',
                'practice': 'Implement workload-specific monitoring and alerting strategies'
            },
            {
                'category': 'cost_management',
                'practice': 'Regular cost reviews and optimization cycles based on usage patterns'
            }
        ]


# Convenience functions for standalone usage
def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary report from results."""
    generator = ReportGenerator(results)
    return generator.get_summary()


def generate_detailed_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a detailed report from results."""
    generator = ReportGenerator(results)
    return generator.get_detailed_report()


def export_to_json(results: Dict[str, Any], filepath: str):
    """Export results to JSON file."""
    generator = ReportGenerator(results)
    generator.save_json(filepath)


def export_to_csv(results: Dict[str, Any], filepath: str):
    """Export results to CSV file."""
    generator = ReportGenerator(results)
    generator.save_csv(filepath)


def export_to_html(results: Dict[str, Any], filepath: str):
    """Export results to HTML file."""
    generator = ReportGenerator(results)
    generator.save_html(filepath)
