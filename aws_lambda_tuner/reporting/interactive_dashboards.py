"""
Interactive dashboard components for AWS Lambda Tuner.
Provides web-based interactive visualizations and dashboards.
"""

import json
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..utils import calculate_cost, calculate_statistics
from ..exceptions import VisualizationError

logger = logging.getLogger(__name__)


class InteractiveDashboard:
    """Creates interactive web-based dashboards using Plotly."""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize interactive dashboard.
        
        Args:
            results: Tuning results dictionary
        """
        self.results = results
        self.configurations = results.get('configurations', [])
        
        if not self.configurations:
            raise VisualizationError("No configurations found in results")

    def create_performance_dashboard(self, output_path: str, workload_type: str = "generic"):
        """
        Create an interactive performance dashboard.
        
        Args:
            output_path: Path to save the HTML dashboard
            workload_type: Type of workload for specialized metrics
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Performance by Memory', 'Cost Analysis', 
                              'Cold Start Impact', 'Duration Distribution',
                              'Optimization Curve', 'Key Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]],
                row_heights=[0.35, 0.35, 0.3]
            )
            
            # Extract data
            memory_sizes = []
            avg_durations = []
            avg_costs = []
            cold_start_rates = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                memory_sizes.append(memory_mb)
                
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                
                if successful:
                    durations = [e['duration'] for e in successful]
                    costs = [calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                            for e in successful]
                    
                    avg_durations.append(sum(durations) / len(durations))
                    avg_costs.append(sum(costs) / len(costs))
                    
                    cold_starts = sum(1 for e in successful if e.get('cold_start', False))
                    cold_start_rates.append((cold_starts / len(successful)) * 100)
                else:
                    avg_durations.append(0)
                    avg_costs.append(0)
                    cold_start_rates.append(0)
            
            # Performance chart
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=avg_durations, 
                      name="Avg Duration", marker_color='lightblue'),
                row=1, col=1
            )
            
            # Cost chart
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=avg_costs, 
                      name="Avg Cost", marker_color='orange'),
                row=1, col=2
            )
            
            # Cold start chart
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=cold_start_rates, 
                      name="Cold Start Rate", marker_color='lightcoral'),
                row=2, col=1
            )
            
            # Duration distribution (box plot)
            for i, config in enumerate(self.configurations):
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                durations = [e['duration'] for e in successful] if successful else [0]
                
                fig.add_trace(
                    go.Box(y=durations, name=f"{config['memory_mb']}MB",
                          boxpoints='outliers'),
                    row=2, col=2
                )
            
            # Optimization curve
            fig.add_trace(
                go.Scatter(x=avg_durations, y=avg_costs, mode='markers+lines',
                          name="Cost vs Performance", 
                          text=[f"{m}MB" for m in memory_sizes],
                          textposition="top center",
                          marker=dict(size=10, color=memory_sizes, 
                                    colorscale='viridis', showscale=True)),
                row=3, col=1
            )
            
            # Key metrics table
            best_performance_idx = avg_durations.index(min(avg_durations)) if avg_durations else 0
            best_cost_idx = avg_costs.index(min(avg_costs)) if avg_costs else 0
            
            table_data = [
                ["Best Performance", f"{memory_sizes[best_performance_idx]}MB", 
                 f"{avg_durations[best_performance_idx]:.1f}ms"],
                ["Best Cost", f"{memory_sizes[best_cost_idx]}MB", 
                 f"${avg_costs[best_cost_idx]:.6f}"],
                ["Lowest Cold Start", f"{memory_sizes[cold_start_rates.index(min(cold_start_rates))]}MB",
                 f"{min(cold_start_rates):.1f}%"],
                ["Total Configurations", str(len(self.configurations)), ""],
                ["Total Executions", str(sum(len(c.get('executions', [])) for c in self.configurations)), ""]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Memory Size', 'Value'],
                              fill_color='lightblue',
                              font=dict(size=12, color='white')),
                    cells=dict(values=list(zip(*table_data)),
                             fill_color='lightgray',
                             font=dict(size=11))
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"{workload_type.title()} Lambda Performance Dashboard",
                height=1000,
                showlegend=False,
                template="plotly_white"
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Duration (ms)", row=1, col=1)
            fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
            fig.update_yaxes(title_text="Cold Start Rate (%)", row=2, col=1)
            fig.update_yaxes(title_text="Duration (ms)", row=2, col=2)
            fig.update_yaxes(title_text="Cost ($)", row=3, col=1)
            fig.update_xaxes(title_text="Duration (ms)", row=3, col=1)
            
            # Save interactive HTML
            self._save_interactive_html(fig, output_path, f"{workload_type.title()} Performance Dashboard")
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            raise VisualizationError(f"Failed to create performance dashboard: {e}")

    def create_cost_analysis_dashboard(self, output_path: str, scenarios: List[Dict[str, Any]] = None):
        """
        Create an interactive cost analysis dashboard.
        
        Args:
            output_path: Path to save the HTML dashboard
            scenarios: Optional cost projection scenarios
        """
        try:
            # Create multi-tab dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cost per Invocation', 'Monthly Cost Projection',
                              'Cost Efficiency Trend', 'Savings Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Extract cost data
            memory_sizes = []
            avg_costs = []
            total_costs = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                memory_sizes.append(memory_mb)
                
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                
                if successful:
                    costs = [calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                            for e in successful]
                    avg_costs.append(sum(costs) / len(costs))
                    total_costs.append(sum(costs))
                else:
                    avg_costs.append(0)
                    total_costs.append(0)
            
            # Cost per invocation
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=avg_costs,
                      name="Cost per Invocation", marker_color='green',
                      text=[f"${c:.6f}" for c in avg_costs],
                      textposition='outside'),
                row=1, col=1
            )
            
            # Monthly projection (example: 1M invocations/month)
            monthly_costs = [cost * 1000000 for cost in avg_costs]
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=monthly_costs,
                      name="Monthly Cost (1M invocations)", marker_color='orange',
                      text=[f"${c:.0f}" for c in monthly_costs],
                      textposition='outside'),
                row=1, col=2
            )
            
            # Cost efficiency trend
            if avg_costs:
                baseline_cost = avg_costs[0]
                efficiency_ratios = [baseline_cost / cost if cost > 0 else 1 for cost in avg_costs]
                
                fig.add_trace(
                    go.Scatter(x=memory_sizes, y=efficiency_ratios, mode='lines+markers',
                              name="Efficiency Ratio", line=dict(width=3),
                              marker=dict(size=8, color='purple')),
                    row=2, col=1
                )
                
                # Add baseline line
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                             annotation_text="Baseline", row=2, col=1)
            
            # Savings analysis
            if scenarios:
                scenario_names = []
                scenario_savings = []
                
                for scenario in scenarios:
                    name = scenario.get('name', 'Scenario')
                    daily_invocations = scenario.get('daily_invocations', 10000)
                    baseline_monthly = avg_costs[0] * daily_invocations * 30 if avg_costs else 0
                    optimized_monthly = min(avg_costs) * daily_invocations * 30 if avg_costs else 0
                    savings = baseline_monthly - optimized_monthly
                    
                    scenario_names.append(name)
                    scenario_savings.append(savings)
                
                fig.add_trace(
                    go.Bar(x=scenario_names, y=scenario_savings,
                          name="Monthly Savings", marker_color='lightgreen',
                          text=[f"${s:.2f}" for s in scenario_savings],
                          textposition='outside'),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Lambda Cost Analysis Dashboard",
                height=800,
                showlegend=False,
                template="plotly_white"
            )
            
            # Update axes
            fig.update_yaxes(title_text="Cost per Invocation ($)", row=1, col=1)
            fig.update_yaxes(title_text="Monthly Cost ($)", row=1, col=2)
            fig.update_yaxes(title_text="Efficiency Ratio", row=2, col=1)
            fig.update_yaxes(title_text="Monthly Savings ($)", row=2, col=2)
            
            # Save interactive HTML
            self._save_interactive_html(fig, output_path, "Cost Analysis Dashboard")
            
        except Exception as e:
            logger.error(f"Error creating cost analysis dashboard: {e}")
            raise VisualizationError(f"Failed to create cost analysis dashboard: {e}")

    def create_cold_start_dashboard(self, output_path: str):
        """
        Create an interactive cold start analysis dashboard.
        
        Args:
            output_path: Path to save the HTML dashboard
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cold Start Frequency', 'Cold vs Warm Duration',
                              'Cold Start Timeline', 'Memory Impact on Cold Starts'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Collect cold start data
            memory_data = {}
            all_executions = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                executions = config.get('executions', [])
                
                cold_starts = [e for e in executions if e.get('cold_start') and not e.get('error')]
                warm_starts = [e for e in executions if not e.get('cold_start') and not e.get('error')]
                
                memory_data[memory_mb] = {
                    'cold_count': len(cold_starts),
                    'warm_count': len(warm_starts),
                    'cold_durations': [e['duration'] for e in cold_starts],
                    'warm_durations': [e['duration'] for e in warm_starts],
                    'total': len(cold_starts) + len(warm_starts)
                }
                
                # Collect for timeline
                for i, execution in enumerate(executions):
                    if not execution.get('error'):
                        all_executions.append({
                            'memory': memory_mb,
                            'duration': execution['duration'],
                            'cold_start': execution.get('cold_start', False),
                            'index': i  # Simulate time
                        })
            
            # Cold start frequency
            memory_sizes = sorted(memory_data.keys())
            frequencies = []
            
            for memory in memory_sizes:
                data = memory_data[memory]
                frequency = (data['cold_count'] / data['total'] * 100) if data['total'] > 0 else 0
                frequencies.append(frequency)
            
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=frequencies,
                      name="Cold Start Frequency", marker_color='lightcoral',
                      text=[f"{f:.1f}%" for f in frequencies],
                      textposition='outside'),
                row=1, col=1
            )
            
            # Cold vs Warm comparison
            for memory in memory_sizes:
                data = memory_data[memory]
                if data['cold_durations'] and data['warm_durations']:
                    avg_cold = sum(data['cold_durations']) / len(data['cold_durations'])
                    avg_warm = sum(data['warm_durations']) / len(data['warm_durations'])
                    
                    fig.add_trace(
                        go.Bar(x=[f"{memory}MB Cold", f"{memory}MB Warm"], 
                              y=[avg_cold, avg_warm],
                              name=f"{memory}MB",
                              marker_color=['lightcoral', 'lightgreen']),
                        row=1, col=2
                    )
            
            # Cold start timeline
            if all_executions:
                cold_executions = [e for e in all_executions if e['cold_start']]
                warm_executions = [e for e in all_executions if not e['cold_start']]
                
                if cold_executions:
                    fig.add_trace(
                        go.Scatter(x=[e['index'] for e in cold_executions],
                                  y=[e['duration'] for e in cold_executions],
                                  mode='markers', name='Cold Start',
                                  marker=dict(color='red', size=8, symbol='circle')),
                        row=2, col=1
                    )
                
                if warm_executions:
                    fig.add_trace(
                        go.Scatter(x=[e['index'] for e in warm_executions],
                                  y=[e['duration'] for e in warm_executions],
                                  mode='markers', name='Warm Start',
                                  marker=dict(color='blue', size=6, symbol='circle')),
                        row=2, col=1
                    )
            
            # Memory impact analysis
            cold_averages = []
            warm_averages = []
            penalties = []
            
            for memory in memory_sizes:
                data = memory_data[memory]
                cold_avg = sum(data['cold_durations']) / len(data['cold_durations']) if data['cold_durations'] else 0
                warm_avg = sum(data['warm_durations']) / len(data['warm_durations']) if data['warm_durations'] else 0
                penalty = max(0, cold_avg - warm_avg)
                
                cold_averages.append(cold_avg)
                warm_averages.append(warm_avg)
                penalties.append(penalty)
            
            fig.add_trace(
                go.Bar(x=[f"{m}MB" for m in memory_sizes], y=penalties,
                      name="Cold Start Penalty", marker_color='orange',
                      text=[f"{p:.1f}ms" for p in penalties],
                      textposition='outside'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Cold Start Analysis Dashboard",
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            # Update axes
            fig.update_yaxes(title_text="Frequency (%)", row=1, col=1)
            fig.update_yaxes(title_text="Duration (ms)", row=1, col=2)
            fig.update_yaxes(title_text="Duration (ms)", row=2, col=1)
            fig.update_yaxes(title_text="Penalty (ms)", row=2, col=2)
            fig.update_xaxes(title_text="Execution Index", row=2, col=1)
            
            # Save interactive HTML
            self._save_interactive_html(fig, output_path, "Cold Start Analysis Dashboard")
            
        except Exception as e:
            logger.error(f"Error creating cold start dashboard: {e}")
            raise VisualizationError(f"Failed to create cold start dashboard: {e}")

    def create_workload_comparison_dashboard(self, output_path: str, 
                                           comparison_results: List[Dict[str, Any]]):
        """
        Create a dashboard comparing multiple workload types.
        
        Args:
            output_path: Path to save the HTML dashboard
            comparison_results: List of workload analysis results
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Memory Recommendations by Workload', 'Performance Metrics',
                              'Cost Efficiency Comparison', 'Workload Characteristics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            workload_names = []
            optimal_memories = []
            performance_scores = []
            cost_scores = []
            
            # Extract comparison data
            for result in comparison_results:
                workload_type = result.get('workload_type', 'Unknown')
                analysis = result.get('analysis', {})
                key_metrics = analysis.get('key_metrics', {})
                
                workload_names.append(workload_type.replace('_', ' ').title())
                optimal_memories.append(key_metrics.get('optimal_memory', 128))
                
                # Calculate normalized scores (0-100)
                perf_gain = key_metrics.get('performance_gain', 0)
                cost_savings = key_metrics.get('cost_savings', 0)
                performance_scores.append(max(0, min(100, 50 + perf_gain)))
                cost_scores.append(max(0, min(100, 50 + cost_savings)))
            
            # Memory recommendations
            fig.add_trace(
                go.Bar(x=workload_names, y=optimal_memories,
                      name="Optimal Memory", marker_color='lightblue',
                      text=[f"{m}MB" for m in optimal_memories],
                      textposition='outside'),
                row=1, col=1
            )
            
            # Performance metrics radar-like comparison
            fig.add_trace(
                go.Bar(x=workload_names, y=performance_scores,
                      name="Performance Score", marker_color='lightgreen'),
                row=1, col=2
            )
            
            # Cost efficiency
            fig.add_trace(
                go.Bar(x=workload_names, y=cost_scores,
                      name="Cost Efficiency Score", marker_color='gold'),
                row=2, col=1
            )
            
            # Characteristics table
            table_data = []
            for i, result in enumerate(comparison_results):
                workload_type = result.get('workload_type', 'Unknown')
                analysis = result.get('analysis', {})
                key_metrics = analysis.get('key_metrics', {})
                
                characteristics = []
                if workload_type == 'web_api':
                    characteristics = ['Low Latency', 'Cold Start Sensitive', 'High Concurrency']
                elif workload_type == 'batch_processing':
                    characteristics = ['High Throughput', 'Cost Optimized', 'Long Running']
                elif workload_type == 'event_driven':
                    characteristics = ['Event Processing', 'Scalable', 'Reactive']
                else:
                    characteristics = ['General Purpose', 'Balanced', 'Flexible']
                
                table_data.append([
                    workload_names[i],
                    f"{optimal_memories[i]}MB",
                    f"{performance_scores[i]:.0f}",
                    f"{cost_scores[i]:.0f}",
                    ', '.join(characteristics)
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Workload', 'Memory', 'Performance', 'Cost Efficiency', 'Characteristics'],
                              fill_color='lightblue',
                              font=dict(size=11)),
                    cells=dict(values=list(zip(*table_data)),
                             fill_color='lightgray',
                             font=dict(size=10))
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Workload Comparison Dashboard",
                height=800,
                showlegend=False,
                template="plotly_white"
            )
            
            # Update axes
            fig.update_yaxes(title_text="Memory (MB)", row=1, col=1)
            fig.update_yaxes(title_text="Performance Score", row=1, col=2)
            fig.update_yaxes(title_text="Cost Efficiency Score", row=2, col=1)
            
            # Save interactive HTML
            self._save_interactive_html(fig, output_path, "Workload Comparison Dashboard")
            
        except Exception as e:
            logger.error(f"Error creating workload comparison dashboard: {e}")
            raise VisualizationError(f"Failed to create workload comparison dashboard: {e}")

    def create_real_time_monitoring_dashboard(self, output_path: str):
        """
        Create a real-time monitoring dashboard template.
        
        Args:
            output_path: Path to save the HTML dashboard
        """
        try:
            # Create a template for real-time monitoring
            html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Lambda Real-Time Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; text-transform: uppercase; }}
        .chart-container {{ height: 400px; }}
        .status-indicator {{ width: 20px; height: 20px; border-radius: 50%; display: inline-block; }}
        .status-healthy {{ background: #27ae60; }}
        .status-warning {{ background: #f39c12; }}
        .status-critical {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <h1>🔄 Lambda Real-Time Monitoring</h1>
    
    <div class="dashboard">
        <div class="metric-card">
            <div class="metric-label">Current Invocations/Min</div>
            <div class="metric-value" id="invocations-rate">0</div>
            <span class="status-indicator status-healthy" id="invocations-status"></span>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Average Duration</div>
            <div class="metric-value" id="avg-duration">0ms</div>
            <span class="status-indicator status-healthy" id="duration-status"></span>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Error Rate</div>
            <div class="metric-value" id="error-rate">0%</div>
            <span class="status-indicator status-healthy" id="error-status"></span>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Cold Start Rate</div>
            <div class="metric-value" id="cold-start-rate">0%</div>
            <span class="status-indicator status-healthy" id="cold-start-status"></span>
        </div>
    </div>
    
    <div class="chart-container">
        <div id="performance-chart"></div>
    </div>
    
    <div class="chart-container">
        <div id="cost-chart"></div>
    </div>
    
    <script>
        // Initialize charts
        var performanceTrace = {{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Duration (ms)',
            line: {{color: 'blue'}}
        }};
        
        var performanceLayout = {{
            title: 'Real-Time Performance',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Duration (ms)'}}
        }};
        
        Plotly.newPlot('performance-chart', [performanceTrace], performanceLayout);
        
        var costTrace = {{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Cost ($)',
            line: {{color: 'orange'}}
        }};
        
        var costLayout = {{
            title: 'Real-Time Cost Tracking',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Cost ($)'}}
        }};
        
        Plotly.newPlot('cost-chart', [costTrace], costLayout);
        
        // Simulate real-time updates
        function updateMetrics() {{
            // Simulate random data
            var now = new Date();
            var invocations = Math.floor(Math.random() * 100) + 50;
            var duration = Math.floor(Math.random() * 200) + 100;
            var errorRate = Math.random() * 5;
            var coldStartRate = Math.random() * 10;
            var cost = (Math.random() * 0.001) + 0.0001;
            
            // Update metric cards
            document.getElementById('invocations-rate').textContent = invocations;
            document.getElementById('avg-duration').textContent = duration + 'ms';
            document.getElementById('error-rate').textContent = errorRate.toFixed(1) + '%';
            document.getElementById('cold-start-rate').textContent = coldStartRate.toFixed(1) + '%';
            
            // Update status indicators
            updateStatus('invocations-status', invocations, [30, 80]);
            updateStatus('duration-status', duration, [150, 300]);
            updateStatus('error-status', errorRate, [2, 5]);
            updateStatus('cold-start-status', coldStartRate, [5, 15]);
            
            // Update charts
            Plotly.extendTraces('performance-chart', {{
                x: [[now]],
                y: [[duration]]
            }}, [0]);
            
            Plotly.extendTraces('cost-chart', {{
                x: [[now]],
                y: [[cost]]
            }}, [0]);
            
            // Keep only last 50 points
            if (performanceTrace.x.length > 50) {{
                Plotly.relayout('performance-chart', {{
                    'xaxis.range': [performanceTrace.x[performanceTrace.x.length-50], 
                                   performanceTrace.x[performanceTrace.x.length-1]]
                }});
                Plotly.relayout('cost-chart', {{
                    'xaxis.range': [costTrace.x[costTrace.x.length-50], 
                                   costTrace.x[costTrace.x.length-1]]
                }});
            }}
        }}
        
        function updateStatus(elementId, value, thresholds) {{
            var element = document.getElementById(elementId);
            if (value < thresholds[0]) {{
                element.className = 'status-indicator status-healthy';
            }} else if (value < thresholds[1]) {{
                element.className = 'status-indicator status-warning';
            }} else {{
                element.className = 'status-indicator status-critical';
            }}
        }}
        
        // Update every 5 seconds
        setInterval(updateMetrics, 5000);
        updateMetrics(); // Initial update
    </script>
</body>
</html>
"""
            
            # Save HTML file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Real-time monitoring dashboard saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating real-time monitoring dashboard: {e}")
            raise VisualizationError(f"Failed to create real-time monitoring dashboard: {e}")

    def _save_interactive_html(self, fig: go.Figure, output_path: str, title: str):
        """Save Plotly figure as interactive HTML."""
        try:
            # Create directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Configure Plotly HTML
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'lambda_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
            
            # Generate HTML with custom template
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>{title}</h1>
            <p>Interactive AWS Lambda Performance Analysis</p>
            <p><small>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
        </div>
        
        <div id="plotly-div" style="width:100%;height:800px;"></div>
        
        <div class="footer">
            <p>🚀 AWS Lambda Performance Tuner - Interactive Dashboard</p>
            <p><small>Use the toolbar above the chart to interact with the visualization</small></p>
        </div>
    </div>
    
    <script>
        {pyo.plot(fig, output_type='div', config=config, div_id='plotly-div', include_plotlyjs=False)}
    </script>
</body>
</html>
"""
            
            with open(output_path, 'w') as f:
                f.write(html_template)
            
            logger.info(f"Interactive dashboard saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving interactive HTML: {e}")
            raise VisualizationError(f"Failed to save interactive HTML: {e}")


# Convenience functions
def create_interactive_performance_dashboard(results: Dict[str, Any], output_path: str, 
                                           workload_type: str = "generic"):
    """Create interactive performance dashboard."""
    dashboard = InteractiveDashboard(results)
    dashboard.create_performance_dashboard(output_path, workload_type)


def create_interactive_cost_dashboard(results: Dict[str, Any], output_path: str, 
                                    scenarios: List[Dict[str, Any]] = None):
    """Create interactive cost analysis dashboard."""
    dashboard = InteractiveDashboard(results)
    dashboard.create_cost_analysis_dashboard(output_path, scenarios)


def create_interactive_cold_start_dashboard(results: Dict[str, Any], output_path: str):
    """Create interactive cold start analysis dashboard."""
    dashboard = InteractiveDashboard(results)
    dashboard.create_cold_start_dashboard(output_path)


def create_workload_comparison_dashboard(comparison_results: List[Dict[str, Any]], output_path: str):
    """Create workload comparison dashboard."""
    # Use first result for dashboard initialization
    if comparison_results:
        dashboard = InteractiveDashboard(comparison_results[0].get('results', {}))
        dashboard.create_workload_comparison_dashboard(output_path, comparison_results)


def create_real_time_monitoring_dashboard(output_path: str):
    """Create real-time monitoring dashboard template."""
    # Create dummy results for initialization
    dummy_results = {'configurations': []}
    dashboard = InteractiveDashboard(dummy_results)
    dashboard.create_real_time_monitoring_dashboard(output_path)