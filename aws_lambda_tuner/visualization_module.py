"""
Visualization module for AWS Lambda Tuner.
Generates charts and graphs for performance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

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
