"""
Report generation service for AWS Lambda Tuner.
"""

import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import statistics
import logging

from .utils import (
    calculate_statistics, format_duration, calculate_cost,
    save_json_file, format_timestamp, safe_divide
)
from .exceptions import ReportGenerationError

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates various reports from tuning results."""
    
    def __init__(self, results: Dict[str, Any], config: Optional[Any] = None):
        """
        Initialize report generator.
        
        Args:
            results: Tuning results dictionary
            config: Optional tuner configuration
        """
        self.results = results
        self.config = config
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
