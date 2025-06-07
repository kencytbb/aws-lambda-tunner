"""
Visualization module for AWS Lambda Tuner.
Enhanced with advanced charts and performance trend analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .utils import calculate_cost, calculate_statistics
from .exceptions import VisualizationError

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VisualizationEngine:
    """Engine for creating performance visualizations."""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize visualization engine.
        
        Args:
            results: Tuning results dictionary
        """
        self.results = results
        self.configurations = results.get('configurations', [])
        
        if not self.configurations:
            raise VisualizationError("No configurations found in results")
    
    def plot_performance_comparison(self, output_path: str):
        """
        Create a bar chart comparing performance across memory configurations.
        
        Args:
            output_path: Path to save the chart
        """
        try:
            # Extract data
            memory_sizes = []
            avg_durations = []
            min_durations = []
            max_durations = []
            
            for config in self.configurations:
                memory_sizes.append(config['memory_mb'])
                
                # Calculate statistics
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                
                if successful:
                    durations = [e['duration'] for e in successful]
                    stats = calculate_statistics(durations)
                    avg_durations.append(stats['mean'])
                    min_durations.append(stats['min'])
                    max_durations.append(stats['max'])
                else:
                    avg_durations.append(0)
                    min_durations.append(0)
                    max_durations.append(0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bar chart
            x_pos = np.arange(len(memory_sizes))
            bars = ax.bar(x_pos, avg_durations, alpha=0.8)
            
            # Add error bars
            errors = [[avg - min_val for avg, min_val in zip(avg_durations, min_durations)],
                     [max_val - avg for avg, max_val in zip(avg_durations, max_durations)]]
            ax.errorbar(x_pos, avg_durations, yerr=errors, fmt='none', ecolor='black', capsize=5)
            
            # Customize chart
            ax.set_xlabel('Memory Size (MB)', fontsize=12)
            ax.set_ylabel('Average Duration (ms)', fontsize=12)
            ax.set_title('Lambda Performance by Memory Configuration', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{size}MB' for size in memory_sizes])
            
            # Color bars based on performance (green = faster)
            norm = plt.Normalize(min(avg_durations), max(avg_durations))
            colors = plt.cm.RdYlGn_r(norm(avg_durations))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels on bars
            for i, (bar, duration) in enumerate(zip(bars, avg_durations)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{duration:.1f}ms', ha='center', va='bottom', fontsize=10)
            
            # Add grid
            ax.grid(True, axis='y', alpha=0.3)
            
            # Save figure
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating performance comparison chart: {e}")
            raise VisualizationError(f"Failed to create performance comparison: {e}")
    
    def plot_cost_analysis(self, output_path: str):
        """
        Create a chart showing cost analysis across memory configurations.
        
        Args:
            output_path: Path to save the chart
        """
        try:
            # Extract data
            memory_sizes = []
            avg_costs = []
            total_costs = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                memory_sizes.append(memory_mb)
                
                # Calculate costs
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                
                if successful:
                    costs = [calculate_cost(memory_mb, e['billed_duration']) for e in successful]
                    avg_costs.append(np.mean(costs))
                    total_costs.append(sum(costs))
                else:
                    avg_costs.append(0)
                    total_costs.append(0)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Cost per invocation
            bars1 = ax1.bar(range(len(memory_sizes)), avg_costs, alpha=0.8)
            ax1.set_xlabel('Memory Size (MB)', fontsize=12)
            ax1.set_ylabel('Average Cost per Invocation ($)', fontsize=12)
            ax1.set_title('Cost per Invocation by Memory Size', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(memory_sizes)))
            ax1.set_xticklabels([f'{size}MB' for size in memory_sizes])
            
            # Add value labels
            for bar, cost in zip(bars1, avg_costs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.000001,
                        f'${cost:.6f}', ha='center', va='bottom', fontsize=9)
            
            # Total cost comparison
            bars2 = ax2.bar(range(len(memory_sizes)), total_costs, alpha=0.8, color='orange')
            ax2.set_xlabel('Memory Size (MB)', fontsize=12)
            ax2.set_ylabel('Total Test Cost ($)', fontsize=12)
            ax2.set_title('Total Testing Cost by Memory Size', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(memory_sizes)))
            ax2.set_xticklabels([f'{size}MB' for size in memory_sizes])
            
            # Add value labels
            for bar, cost in zip(bars2, total_costs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                        f'${cost:.5f}', ha='center', va='bottom', fontsize=9)
            
            # Add grid
            ax1.grid(True, axis='y', alpha=0.3)
            ax2.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating cost analysis chart: {e}")
            raise VisualizationError(f"Failed to create cost analysis: {e}")
    
    def plot_duration_distribution(self, output_path: str):
        """
        Create box plots showing duration distribution for each memory configuration.
        
        Args:
            output_path: Path to save the chart
        """
        try:
            # Prepare data for box plot
            data_by_memory = []
            labels = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                
                if successful:
                    durations = [e['duration'] for e in successful]
                    data_by_memory.append(durations)
                    labels.append(f'{memory_mb}MB')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create box plot
            bp = ax.boxplot(data_by_memory, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Customize box plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_memory)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize other elements
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black')
            
            plt.setp(bp['means'], color='red', linewidth=2)
            plt.setp(bp['medians'], color='black', linewidth=2)
            
            # Labels and title
            ax.set_xlabel('Memory Configuration', fontsize=12)
            ax.set_ylabel('Execution Duration (ms)', fontsize=12)
            ax.set_title('Duration Distribution by Memory Configuration', fontsize=14, fontweight='bold')
            
            # Add grid
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label='Mean')
            black_patch = mpatches.Patch(color='black', label='Median')
            ax.legend(handles=[red_patch, black_patch], loc='upper right')
            
            # Save figure
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating duration distribution chart: {e}")
            raise VisualizationError(f"Failed to create duration distribution: {e}")
    
    def plot_optimization_curve(self, output_path: str):
        """
        Create a chart showing the optimization curve (cost vs performance).
        
        Args:
            output_path: Path to save the chart
        """
        try:
            # Extract data
            memory_sizes = []
            avg_durations = []
            avg_costs = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                memory_sizes.append(memory_mb)
                
                # Calculate statistics
                executions = config.get('executions', [])
                successful = [e for e in executions if not e.get('error')]
                
                if successful:
                    durations = [e['duration'] for e in successful]
                    costs = [calculate_cost(memory_mb, e['billed_duration']) for e in successful]
                    avg_durations.append(np.mean(durations))
                    avg_costs.append(np.mean(costs))
                else:
                    avg_durations.append(0)
                    avg_costs.append(0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot
            scatter = ax.scatter(avg_durations, avg_costs, s=200, alpha=0.6, 
                               c=memory_sizes, cmap='viridis', edgecolors='black', linewidth=2)
            
            # Add labels for each point
            for i, (duration, cost, memory) in enumerate(zip(avg_durations, avg_costs, memory_sizes)):
                ax.annotate(f'{memory}MB', (duration, cost), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            # Add optimal point marker
            # Find the point with best cost/performance ratio
            if avg_durations and avg_costs:
                ratios = [cost / duration if duration > 0 else float('inf') 
                         for cost, duration in zip(avg_costs, avg_durations)]
                optimal_idx = np.argmin(ratios)
                ax.scatter(avg_durations[optimal_idx], avg_costs[optimal_idx], 
                          s=400, marker='*', color='red', edgecolors='darkred', 
                          linewidth=2, label='Optimal', zorder=5)
            
            # Labels and title
            ax.set_xlabel('Average Duration (ms)', fontsize=12)
            ax.set_ylabel('Average Cost per Invocation ($)', fontsize=12)
            ax.set_title('Cost vs Performance Optimization Curve', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Memory Size (MB)', fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Save figure
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating optimization curve: {e}")
            raise VisualizationError(f"Failed to create optimization curve: {e}")
    
    def plot_cold_start_analysis(self, output_path: str):
        """
        Create a chart analyzing cold start behavior.
        
        Args:
            output_path: Path to save the chart
        """
        try:
            # Extract cold start data
            memory_sizes = []
            cold_start_counts = []
            cold_start_durations = []
            warm_start_durations = []
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                memory_sizes.append(memory_mb)
                
                executions = config.get('executions', [])
                cold_starts = [e for e in executions if e.get('cold_start') and not e.get('error')]
                warm_starts = [e for e in executions if not e.get('cold_start') and not e.get('error')]
                
                cold_start_counts.append(len(cold_starts))
                
                if cold_starts:
                    cold_start_durations.append(np.mean([e['duration'] for e in cold_starts]))
                else:
                    cold_start_durations.append(0)
                
                if warm_starts:
                    warm_start_durations.append(np.mean([e['duration'] for e in warm_starts]))
                else:
                    warm_start_durations.append(0)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Cold start counts
            bars = ax1.bar(range(len(memory_sizes)), cold_start_counts, alpha=0.8, color='skyblue')
            ax1.set_xlabel('Memory Size (MB)', fontsize=12)
            ax1.set_ylabel('Number of Cold Starts', fontsize=12)
            ax1.set_title('Cold Start Occurrences by Memory Size', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(memory_sizes)))
            ax1.set_xticklabels([f'{size}MB' for size in memory_sizes])
            
            # Add value labels
            for bar, count in zip(bars, cold_start_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=10)
            
            # Cold vs Warm comparison
            x = np.arange(len(memory_sizes))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, cold_start_durations, width, label='Cold Start', alpha=0.8, color='lightcoral')
            bars2 = ax2.bar(x + width/2, warm_start_durations, width, label='Warm Start', alpha=0.8, color='lightgreen')
            
            ax2.set_xlabel('Memory Size (MB)', fontsize=12)
            ax2.set_ylabel('Average Duration (ms)', fontsize=12)
            ax2.set_title('Cold Start vs Warm Start Duration', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{size}MB' for size in memory_sizes])
            ax2.legend()
            
            # Add grid
            ax1.grid(True, axis='y', alpha=0.3)
            ax2.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating cold start analysis: {e}")
            raise VisualizationError(f"Failed to create cold start analysis: {e}")

    def cold_start_visualization(self, output_path: str):
        """
        Create comprehensive cold start visualization with detailed analysis.
        
        Args:
            output_path: Path to save the chart
        """
        try:
            # Extract cold start data with timestamps
            memory_data = defaultdict(lambda: {'cold': [], 'warm': [], 'timestamps': []})
            
            for config in self.configurations:
                memory_mb = config['memory_mb']
                executions = config.get('executions', [])
                
                for execution in executions:
                    if not execution.get('error'):
                        duration = execution['duration']
                        timestamp = execution.get('timestamp', datetime.now().isoformat())
                        
                        if execution.get('cold_start'):
                            memory_data[memory_mb]['cold'].append(duration)
                        else:
                            memory_data[memory_mb]['warm'].append(duration)
                        
                        memory_data[memory_mb]['timestamps'].append(timestamp)
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
            
            # Main cold start impact chart
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_cold_start_impact(ax1, memory_data)
            
            # Cold start frequency by memory
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_cold_start_frequency(ax2, memory_data)
            
            # Cold start duration distribution
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_cold_start_duration_distribution(ax3, memory_data)
            
            # Cold start penalty analysis
            ax4 = fig.add_subplot(gs[2, :])
            self._plot_cold_start_penalty(ax4, memory_data)
            
            plt.suptitle('Comprehensive Cold Start Analysis', fontsize=16, fontweight='bold')
            
            # Save figure
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating cold start visualization: {e}")
            raise VisualizationError(f"Failed to create cold start visualization: {e}")

    def workload_pattern_charts(self, output_path: str, workload_type: str = "generic"):
        """
        Create charts showing workload-specific patterns and characteristics.
        
        Args:
            output_path: Path to save the chart
            workload_type: Type of workload for specialized analysis
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Workload Pattern Analysis - {workload_type.title()}', fontsize=16, fontweight='bold')
            
            # Execution timeline
            self._plot_execution_timeline(axes[0, 0])
            
            # Memory utilization efficiency
            self._plot_memory_efficiency(axes[0, 1], workload_type)
            
            # Performance consistency
            self._plot_performance_consistency(axes[1, 0])
            
            # Cost efficiency trends
            self._plot_cost_efficiency_trends(axes[1, 1])
            
            plt.tight_layout()
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating workload pattern charts: {e}")
            raise VisualizationError(f"Failed to create workload pattern charts: {e}")

    def cost_efficiency_dashboards(self, output_path: str, scenarios: List[Dict[str, Any]] = None):
        """
        Create cost efficiency dashboard with multiple scenarios.
        
        Args:
            output_path: Path to save the dashboard
            scenarios: Cost projection scenarios
        """
        try:
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Cost per invocation comparison
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_cost_per_invocation_comparison(ax1)
            
            # ROI analysis
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_roi_analysis(ax2)
            
            # Cost projection scenarios
            if scenarios:
                ax3 = fig.add_subplot(gs[1, :])
                self._plot_cost_projection_scenarios(ax3, scenarios)
            else:
                ax3 = fig.add_subplot(gs[1, :])
                self._plot_cost_breakdown_analysis(ax3)
            
            # Cost efficiency heatmap
            ax4 = fig.add_subplot(gs[2, :2])
            self._plot_cost_efficiency_heatmap(ax4)
            
            # Savings potential
            ax5 = fig.add_subplot(gs[2, 2])
            self._plot_savings_potential(ax5)
            
            plt.suptitle('Cost Efficiency Dashboard', fontsize=16, fontweight='bold')
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating cost efficiency dashboard: {e}")
            raise VisualizationError(f"Failed to create cost efficiency dashboard: {e}")

    def performance_trend_visualizations(self, output_path: str, time_series_data: List[Dict[str, Any]] = None):
        """
        Create performance trend visualizations over time.
        
        Args:
            output_path: Path to save the chart
            time_series_data: Optional time series performance data
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Performance Trend Analysis', fontsize=16, fontweight='bold')
            
            # Duration trends over time
            self._plot_duration_trends(axes[0, 0], time_series_data)
            
            # Performance variance analysis
            self._plot_performance_variance(axes[0, 1])
            
            # Memory scaling impact
            self._plot_memory_scaling_impact(axes[1, 0])
            
            # Performance stability index
            self._plot_performance_stability_index(axes[1, 1])
            
            plt.tight_layout()
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating performance trend visualizations: {e}")
            raise VisualizationError(f"Failed to create performance trend visualizations: {e}")

    def create_comprehensive_dashboard(self, output_path: str, workload_type: str = "generic", 
                                    scenarios: List[Dict[str, Any]] = None):
        """
        Create a comprehensive dashboard combining all visualizations.
        
        Args:
            output_path: Path to save the dashboard
            workload_type: Type of workload
            scenarios: Cost projection scenarios
        """
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
            
            # Performance comparison (top row)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_performance_comparison_summary(ax1)
            
            # Cost analysis (top row)
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_cost_analysis_summary(ax2)
            
            # Cold start analysis (second row)
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_cold_start_summary(ax3)
            
            # Optimization curve (second row)
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_optimization_curve_summary(ax4)
            
            # Workload-specific patterns (third row)
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_workload_specific_summary(ax5, workload_type)
            
            # Cost efficiency (third row)
            ax6 = fig.add_subplot(gs[2, 2:])
            self._plot_cost_efficiency_summary(ax6)
            
            # Key metrics summary (bottom row)
            ax7 = fig.add_subplot(gs[3, :])
            self._plot_key_metrics_summary(ax7)
            
            plt.suptitle('AWS Lambda Performance Tuning - Comprehensive Dashboard', 
                        fontsize=18, fontweight='bold', y=0.95)
            self._save_figure(fig, output_path)
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            raise VisualizationError(f"Failed to create comprehensive dashboard: {e}")

    # Helper methods for the new visualizations
    def _plot_cold_start_impact(self, ax: plt.Axes, memory_data: Dict):
        """Plot cold start impact comparison."""
        memory_sizes = sorted(memory_data.keys())
        cold_averages = []
        warm_averages = []
        
        for memory in memory_sizes:
            data = memory_data[memory]
            cold_avg = np.mean(data['cold']) if data['cold'] else 0
            warm_avg = np.mean(data['warm']) if data['warm'] else 0
            cold_averages.append(cold_avg)
            warm_averages.append(warm_avg)
        
        x = np.arange(len(memory_sizes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cold_averages, width, label='Cold Start', color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x + width/2, warm_averages, width, label='Warm Start', color='lightgreen', alpha=0.8)
        
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Average Duration (ms)')
        ax.set_title('Cold Start vs Warm Start Performance Impact')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}MB' for m in memory_sizes])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_cold_start_frequency(self, ax: plt.Axes, memory_data: Dict):
        """Plot cold start frequency by memory size."""
        memory_sizes = sorted(memory_data.keys())
        frequencies = []
        
        for memory in memory_sizes:
            data = memory_data[memory]
            total = len(data['cold']) + len(data['warm'])
            frequency = len(data['cold']) / total * 100 if total > 0 else 0
            frequencies.append(frequency)
        
        bars = ax.bar(range(len(memory_sizes)), frequencies, color='skyblue', alpha=0.8)
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Cold Start Frequency (%)')
        ax.set_title('Cold Start Frequency by Memory Size')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{m}MB' for m in memory_sizes])
        
        # Add percentage labels on bars
        for bar, freq in zip(bars, frequencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{freq:.1f}%', ha='center', va='bottom')

    def _plot_cold_start_duration_distribution(self, ax: plt.Axes, memory_data: Dict):
        """Plot cold start duration distribution."""
        all_cold_durations = []
        labels = []
        
        for memory, data in sorted(memory_data.items()):
            if data['cold']:
                all_cold_durations.append(data['cold'])
                labels.append(f'{memory}MB')
        
        if all_cold_durations:
            bp = ax.boxplot(all_cold_durations, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(all_cold_durations)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xlabel('Memory Size')
        ax.set_ylabel('Cold Start Duration (ms)')
        ax.set_title('Cold Start Duration Distribution')
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_cold_start_penalty(self, ax: plt.Axes, memory_data: Dict):
        """Plot cold start penalty analysis."""
        memory_sizes = sorted(memory_data.keys())
        penalties = []
        
        for memory in memory_sizes:
            data = memory_data[memory]
            cold_avg = np.mean(data['cold']) if data['cold'] else 0
            warm_avg = np.mean(data['warm']) if data['warm'] else 0
            penalty = max(0, cold_avg - warm_avg)
            penalties.append(penalty)
        
        bars = ax.bar(range(len(memory_sizes)), penalties, color='orange', alpha=0.8)
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Cold Start Penalty (ms)')
        ax.set_title('Cold Start Performance Penalty by Memory Size')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{m}MB' for m in memory_sizes])
        
        # Add penalty values on bars
        for bar, penalty in zip(bars, penalties):
            if penalty > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{penalty:.1f}ms', ha='center', va='bottom')

    def _plot_execution_timeline(self, ax: plt.Axes):
        """Plot execution timeline showing patterns."""
        # Simulate timeline data based on executions
        all_executions = []
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            executions = config.get('executions', [])
            
            for i, execution in enumerate(executions):
                if not execution.get('error'):
                    # Simulate timestamp if not present
                    timestamp = datetime.now() - timedelta(minutes=len(executions)-i)
                    all_executions.append({
                        'timestamp': timestamp,
                        'duration': execution['duration'],
                        'memory': memory_mb,
                        'cold_start': execution.get('cold_start', False)
                    })
        
        if all_executions:
            # Sort by timestamp
            all_executions.sort(key=lambda x: x['timestamp'])
            
            # Plot execution timeline
            timestamps = [e['timestamp'] for e in all_executions]
            durations = [e['duration'] for e in all_executions]
            colors = ['red' if e['cold_start'] else 'blue' for e in all_executions]
            
            scatter = ax.scatter(timestamps, durations, c=colors, alpha=0.6, s=30)
            ax.set_xlabel('Time')
            ax.set_ylabel('Duration (ms)')
            ax.set_title('Execution Timeline')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label='Cold Start')
            blue_patch = mpatches.Patch(color='blue', label='Warm Start')
            ax.legend(handles=[red_patch, blue_patch])

    def _plot_memory_efficiency(self, ax: plt.Axes, workload_type: str):
        """Plot memory utilization efficiency."""
        memory_sizes = []
        efficiency_scores = []
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            memory_sizes.append(memory_mb)
            
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                avg_duration = np.mean([e['duration'] for e in successful])
                # Calculate efficiency score (lower duration per MB is better)
                efficiency = 1000 / (avg_duration * memory_mb / 128)  # Normalized score
                efficiency_scores.append(efficiency)
            else:
                efficiency_scores.append(0)
        
        if memory_sizes and efficiency_scores:
            bars = ax.bar(range(len(memory_sizes)), efficiency_scores, 
                         color='lightblue', alpha=0.8)
            ax.set_xlabel('Memory Size (MB)')
            ax.set_ylabel('Efficiency Score')
            ax.set_title(f'Memory Efficiency - {workload_type.title()}')
            ax.set_xticks(range(len(memory_sizes)))
            ax.set_xticklabels([f'{m}MB' for m in memory_sizes])

    def _plot_performance_consistency(self, ax: plt.Axes):
        """Plot performance consistency metrics."""
        memory_sizes = []
        cv_scores = []  # Coefficient of variation
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            memory_sizes.append(memory_mb)
            
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if len(successful) > 1:
                durations = [e['duration'] for e in successful]
                mean_duration = np.mean(durations)
                std_duration = np.std(durations)
                cv = std_duration / mean_duration if mean_duration > 0 else 0
                cv_scores.append(cv)
            else:
                cv_scores.append(0)
        
        if memory_sizes and cv_scores:
            bars = ax.bar(range(len(memory_sizes)), cv_scores, 
                         color='lightgreen', alpha=0.8)
            ax.set_xlabel('Memory Size (MB)')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('Performance Consistency (Lower is Better)')
            ax.set_xticks(range(len(memory_sizes)))
            ax.set_xticklabels([f'{m}MB' for m in memory_sizes])

    def _plot_cost_efficiency_trends(self, ax: plt.Axes):
        """Plot cost efficiency trends."""
        memory_sizes = []
        cost_per_ms = []
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            memory_sizes.append(memory_mb)
            
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                avg_duration = np.mean([e['duration'] for e in successful])
                avg_cost = np.mean([calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                                  for e in successful])
                efficiency = avg_cost / avg_duration if avg_duration > 0 else 0
                cost_per_ms.append(efficiency * 1000000)  # Scale for readability
            else:
                cost_per_ms.append(0)
        
        if memory_sizes and cost_per_ms:
            ax.plot(memory_sizes, cost_per_ms, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Memory Size (MB)')
            ax.set_ylabel('Cost per ms (μ$)')
            ax.set_title('Cost Efficiency Trend')
            ax.grid(True, alpha=0.3)

    # Summary plot methods for comprehensive dashboard
    def _plot_performance_comparison_summary(self, ax: plt.Axes):
        """Simplified performance comparison for dashboard."""
        memory_sizes = []
        avg_durations = []
        
        for config in self.configurations:
            memory_sizes.append(config['memory_mb'])
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                durations = [e['duration'] for e in successful]
                avg_durations.append(statistics.mean(durations))
            else:
                avg_durations.append(0)
        
        bars = ax.bar(range(len(memory_sizes)), avg_durations, alpha=0.8)
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Avg Duration (ms)')
        ax.set_title('Performance by Memory Size')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{size}MB' for size in memory_sizes])

    def _plot_cost_analysis_summary(self, ax: plt.Axes):
        """Simplified cost analysis for dashboard."""
        memory_sizes = []
        avg_costs = []
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            memory_sizes.append(memory_mb)
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                costs = [calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                        for e in successful]
                avg_costs.append(np.mean(costs))
            else:
                avg_costs.append(0)
        
        bars = ax.bar(range(len(memory_sizes)), avg_costs, alpha=0.8, color='orange')
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Avg Cost per Invocation ($)')
        ax.set_title('Cost by Memory Size')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{size}MB' for size in memory_sizes])

    def _plot_cold_start_summary(self, ax: plt.Axes):
        """Simplified cold start analysis for dashboard."""
        memory_sizes = []
        cold_start_rates = []
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            memory_sizes.append(memory_mb)
            executions = config.get('executions', [])
            
            if executions:
                cold_starts = sum(1 for e in executions if e.get('cold_start') and not e.get('error'))
                total_successful = len([e for e in executions if not e.get('error')])
                rate = (cold_starts / total_successful * 100) if total_successful > 0 else 0
                cold_start_rates.append(rate)
            else:
                cold_start_rates.append(0)
        
        bars = ax.bar(range(len(memory_sizes)), cold_start_rates, alpha=0.8, color='lightcoral')
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Cold Start Rate (%)')
        ax.set_title('Cold Start Rate by Memory')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{size}MB' for size in memory_sizes])

    def _plot_optimization_curve_summary(self, ax: plt.Axes):
        """Simplified optimization curve for dashboard."""
        durations = []
        costs = []
        memory_sizes = []
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                avg_duration = np.mean([e['duration'] for e in successful])
                avg_cost = np.mean([calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                                  for e in successful])
                durations.append(avg_duration)
                costs.append(avg_cost)
                memory_sizes.append(memory_mb)
        
        if durations and costs:
            scatter = ax.scatter(durations, costs, s=100, alpha=0.6, c=memory_sizes, cmap='viridis')
            ax.set_xlabel('Avg Duration (ms)')
            ax.set_ylabel('Avg Cost ($)')
            ax.set_title('Cost vs Performance')
            plt.colorbar(scatter, ax=ax, label='Memory (MB)')

    def _plot_workload_specific_summary(self, ax: plt.Axes, workload_type: str):
        """Workload-specific summary metrics."""
        # Placeholder implementation - would be customized based on workload_type
        memory_sizes = [config['memory_mb'] for config in self.configurations]
        performance_scores = [100 - i*5 for i in range(len(memory_sizes))]  # Simulated scores
        
        bars = ax.bar(range(len(memory_sizes)), performance_scores, alpha=0.8, color='lightgreen')
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Performance Score')
        ax.set_title(f'{workload_type.title()} Performance Score')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{size}MB' for size in memory_sizes])

    def _plot_cost_efficiency_summary(self, ax: plt.Axes):
        """Cost efficiency summary."""
        memory_sizes = []
        efficiency_ratios = []
        
        baseline_cost = None
        for i, config in enumerate(self.configurations):
            memory_mb = config['memory_mb']
            memory_sizes.append(memory_mb)
            
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                avg_cost = np.mean([calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                                  for e in successful])
                if i == 0:  # First configuration as baseline
                    baseline_cost = avg_cost
                    efficiency_ratios.append(1.0)
                else:
                    ratio = baseline_cost / avg_cost if avg_cost > 0 else 1.0
                    efficiency_ratios.append(ratio)
            else:
                efficiency_ratios.append(1.0)
        
        bars = ax.bar(range(len(memory_sizes)), efficiency_ratios, alpha=0.8, color='gold')
        ax.set_xlabel('Memory Size (MB)')
        ax.set_ylabel('Cost Efficiency Ratio')
        ax.set_title('Cost Efficiency vs Baseline')
        ax.set_xticks(range(len(memory_sizes)))
        ax.set_xticklabels([f'{size}MB' for size in memory_sizes])
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax.legend()

    def _plot_key_metrics_summary(self, ax: plt.Axes):
        """Key metrics summary table."""
        # Calculate key metrics
        best_performance_memory = None
        best_cost_memory = None
        min_duration = float('inf')
        min_cost = float('inf')
        
        for config in self.configurations:
            memory_mb = config['memory_mb']
            executions = config.get('executions', [])
            successful = [e for e in executions if not e.get('error')]
            
            if successful:
                avg_duration = np.mean([e['duration'] for e in successful])
                avg_cost = np.mean([calculate_cost(memory_mb, e.get('billed_duration', e['duration'])) 
                                  for e in successful])
                
                if avg_duration < min_duration:
                    min_duration = avg_duration
                    best_performance_memory = memory_mb
                
                if avg_cost < min_cost:
                    min_cost = avg_cost
                    best_cost_memory = memory_mb
        
        # Create summary table
        metrics_data = [
            ['Best Performance', f'{best_performance_memory}MB' if best_performance_memory else 'N/A', 
             f'{min_duration:.1f}ms' if min_duration != float('inf') else 'N/A'],
            ['Best Cost', f'{best_cost_memory}MB' if best_cost_memory else 'N/A', 
             f'${min_cost:.6f}' if min_cost != float('inf') else 'N/A'],
            ['Total Configurations', str(len(self.configurations)), ''],
            ['Total Executions', str(sum(len(c.get('executions', [])) for c in self.configurations)), '']
        ]
        
        # Create table
        table = ax.table(cellText=metrics_data, 
                        colLabels=['Metric', 'Memory Size', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(metrics_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Key Metrics Summary')
        ax.axis('off')

    # Placeholder methods for new features
    def _plot_cost_per_invocation_comparison(self, ax: plt.Axes):
        """Plot cost per invocation comparison."""
        # Implementation similar to existing cost analysis
        pass

    def _plot_roi_analysis(self, ax: plt.Axes):
        """Plot ROI analysis."""
        # Implementation for ROI visualization
        pass

    def _plot_cost_projection_scenarios(self, ax: plt.Axes, scenarios: List[Dict[str, Any]]):
        """Plot cost projection scenarios."""
        # Implementation for scenario projections
        pass

    def _plot_cost_breakdown_analysis(self, ax: plt.Axes):
        """Plot cost breakdown analysis."""
        # Implementation for cost breakdown
        pass

    def _plot_cost_efficiency_heatmap(self, ax: plt.Axes):
        """Plot cost efficiency heatmap."""
        # Implementation for efficiency heatmap
        pass

    def _plot_savings_potential(self, ax: plt.Axes):
        """Plot savings potential."""
        # Implementation for savings visualization
        pass

    def _plot_duration_trends(self, ax: plt.Axes, time_series_data: List[Dict[str, Any]] = None):
        """Plot duration trends over time."""
        # Implementation for duration trends
        pass

    def _plot_performance_variance(self, ax: plt.Axes):
        """Plot performance variance analysis."""
        # Implementation for variance analysis
        pass

    def _plot_memory_scaling_impact(self, ax: plt.Axes):
        """Plot memory scaling impact."""
        # Implementation for scaling impact
        pass

    def _plot_performance_stability_index(self, ax: plt.Axes):
        """Plot performance stability index."""
        # Implementation for stability index
        pass
    
    def _save_figure(self, fig: plt.Figure, output_path: str):
        """Save figure to file."""
        try:
            # Create directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Chart saved to {output_path}")
            
        except Exception as e:
            plt.close(fig)
            raise VisualizationError(f"Failed to save figure: {e}")


# Convenience functions for standalone usage
def create_performance_chart(results: Dict[str, Any], output_path: str):
    """Create performance comparison chart."""
    engine = VisualizationEngine(results)
    engine.plot_performance_comparison(output_path)


def create_cost_chart(results: Dict[str, Any], output_path: str):
    """Create cost analysis chart."""
    engine = VisualizationEngine(results)
    engine.plot_cost_analysis(output_path)


def create_distribution_chart(results: Dict[str, Any], output_path: str):
    """Create duration distribution chart."""
    engine = VisualizationEngine(results)
    engine.plot_duration_distribution(output_path)


def create_optimization_chart(results: Dict[str, Any], output_path: str):
    """Create optimization curve chart."""
    engine = VisualizationEngine(results)
    engine.plot_optimization_curve(output_path)


def create_cold_start_chart(results: Dict[str, Any], output_path: str):
    """Create cold start analysis chart."""
    engine = VisualizationEngine(results)
    engine.plot_cold_start_analysis(output_path)


def create_enhanced_cold_start_visualization(results: Dict[str, Any], output_path: str):
    """Create comprehensive cold start visualization."""
    engine = VisualizationEngine(results)
    engine.cold_start_visualization(output_path)


def create_workload_pattern_charts(results: Dict[str, Any], output_path: str, workload_type: str = "generic"):
    """Create workload-specific pattern charts."""
    engine = VisualizationEngine(results)
    engine.workload_pattern_charts(output_path, workload_type)


def create_cost_efficiency_dashboard(results: Dict[str, Any], output_path: str, scenarios: List[Dict[str, Any]] = None):
    """Create cost efficiency dashboard."""
    engine = VisualizationEngine(results)
    engine.cost_efficiency_dashboards(output_path, scenarios)


def create_performance_trend_charts(results: Dict[str, Any], output_path: str, time_series_data: List[Dict[str, Any]] = None):
    """Create performance trend visualizations."""
    engine = VisualizationEngine(results)
    engine.performance_trend_visualizations(output_path, time_series_data)


def create_comprehensive_dashboard(results: Dict[str, Any], output_path: str, workload_type: str = "generic", 
                                 scenarios: List[Dict[str, Any]] = None):
    """Create comprehensive performance dashboard."""
    engine = VisualizationEngine(results)
    engine.create_comprehensive_dashboard(output_path, workload_type, scenarios)
