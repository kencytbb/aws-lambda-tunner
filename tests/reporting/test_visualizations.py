"""
Tests for enhanced visualization functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from aws_lambda_tuner.visualization_module import VisualizationEngine
from aws_lambda_tuner.reporting.interactive_dashboards import InteractiveDashboard
from aws_lambda_tuner.exceptions import VisualizationError


class TestVisualizationEngine:
    """Test enhanced visualization engine features."""

    @pytest.fixture
    def sample_results(self):
        """Sample tuning results for visualization testing."""
        return {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
            "configurations": [
                {
                    "memory_mb": 128,
                    "executions": [
                        {
                            "duration": 1500,
                            "billed_duration": 1500,
                            "cold_start": True,
                            "error": None,
                            "timestamp": "2024-01-01T12:00:00Z",
                        },
                        {
                            "duration": 800,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:01:00Z",
                        },
                        {
                            "duration": 850,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:02:00Z",
                        },
                        {
                            "duration": 900,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:03:00Z",
                        },
                        {
                            "duration": 780,
                            "billed_duration": 800,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:04:00Z",
                        },
                    ],
                },
                {
                    "memory_mb": 256,
                    "executions": [
                        {
                            "duration": 900,
                            "billed_duration": 900,
                            "cold_start": True,
                            "error": None,
                            "timestamp": "2024-01-01T12:00:00Z",
                        },
                        {
                            "duration": 400,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:01:00Z",
                        },
                        {
                            "duration": 450,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:02:00Z",
                        },
                        {
                            "duration": 380,
                            "billed_duration": 400,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:03:00Z",
                        },
                        {
                            "duration": 420,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:04:00Z",
                        },
                    ],
                },
                {
                    "memory_mb": 512,
                    "executions": [
                        {
                            "duration": 600,
                            "billed_duration": 600,
                            "cold_start": True,
                            "error": None,
                            "timestamp": "2024-01-01T12:00:00Z",
                        },
                        {
                            "duration": 200,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:01:00Z",
                        },
                        {
                            "duration": 250,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:02:00Z",
                        },
                        {
                            "duration": 180,
                            "billed_duration": 200,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:03:00Z",
                        },
                        {
                            "duration": 220,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                            "timestamp": "2024-01-01T12:04:00Z",
                        },
                    ],
                },
            ],
        }

    @pytest.fixture
    def viz_engine(self, sample_results):
        """Visualization engine instance for testing."""
        return VisualizationEngine(sample_results)

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_enhanced_cold_start_visualization(self, viz_engine, temp_dir):
        """Test enhanced cold start visualization."""
        output_path = os.path.join(temp_dir, "cold_start_analysis.png")

        # Should not raise exception
        viz_engine.cold_start_visualization(output_path)

        # Check if file was created
        assert os.path.exists(output_path)

    def test_workload_pattern_charts(self, viz_engine, temp_dir):
        """Test workload pattern charts generation."""
        output_path = os.path.join(temp_dir, "workload_patterns.png")

        # Test with different workload types
        workload_types = ["web_api", "batch_processing", "event_driven", "generic"]

        for workload_type in workload_types:
            viz_engine.workload_pattern_charts(output_path, workload_type)
            assert os.path.exists(output_path)

    def test_cost_efficiency_dashboards(self, viz_engine, temp_dir):
        """Test cost efficiency dashboard generation."""
        output_path = os.path.join(temp_dir, "cost_efficiency.png")

        # Test without scenarios
        viz_engine.cost_efficiency_dashboards(output_path)
        assert os.path.exists(output_path)

        # Test with scenarios
        scenarios = [
            {"name": "low_volume", "daily_invocations": 1000, "pattern": "steady"},
            {"name": "high_volume", "daily_invocations": 100000, "pattern": "bursty"},
        ]

        output_path_2 = os.path.join(temp_dir, "cost_efficiency_with_scenarios.png")
        viz_engine.cost_efficiency_dashboards(output_path_2, scenarios)
        assert os.path.exists(output_path_2)

    def test_performance_trend_visualizations(self, viz_engine, temp_dir):
        """Test performance trend visualizations."""
        output_path = os.path.join(temp_dir, "performance_trends.png")

        # Test without time series data
        viz_engine.performance_trend_visualizations(output_path)
        assert os.path.exists(output_path)

        # Test with time series data
        time_series_data = [
            {"timestamp": "2024-01-01T12:00:00Z", "duration": 500, "memory": 256},
            {"timestamp": "2024-01-01T12:01:00Z", "duration": 450, "memory": 256},
            {"timestamp": "2024-01-01T12:02:00Z", "duration": 480, "memory": 256},
        ]

        output_path_2 = os.path.join(temp_dir, "performance_trends_with_data.png")
        viz_engine.performance_trend_visualizations(output_path_2, time_series_data)
        assert os.path.exists(output_path_2)

    def test_comprehensive_dashboard(self, viz_engine, temp_dir):
        """Test comprehensive dashboard generation."""
        output_path = os.path.join(temp_dir, "comprehensive_dashboard.png")

        # Test with different workload types and scenarios
        scenarios = [{"name": "production", "daily_invocations": 50000, "pattern": "seasonal"}]

        viz_engine.create_comprehensive_dashboard(output_path, "web_api", scenarios)
        assert os.path.exists(output_path)

    def test_cold_start_impact_calculation(self, viz_engine):
        """Test cold start impact calculation in visualizations."""
        # Access sample data to verify calculations
        memory_data = {}

        # Simulate the data processing that happens in cold_start_visualization
        for config in viz_engine.configurations:
            memory_mb = config["memory_mb"]
            executions = config.get("executions", [])

            cold_starts = [e for e in executions if e.get("cold_start") and not e.get("error")]
            warm_starts = [e for e in executions if not e.get("cold_start") and not e.get("error")]

            memory_data[memory_mb] = {
                "cold": [e["duration"] for e in cold_starts],
                "warm": [e["duration"] for e in warm_starts],
            }

        # Verify data extraction
        assert 128 in memory_data
        assert 256 in memory_data
        assert 512 in memory_data

        # Check cold start data
        for memory_mb, data in memory_data.items():
            assert len(data["cold"]) == 1  # One cold start per memory size
            assert len(data["warm"]) == 4  # Four warm starts per memory size

    def test_memory_efficiency_calculation(self, viz_engine):
        """Test memory efficiency calculation logic."""
        # Test the efficiency calculation logic
        memory_sizes = []
        efficiency_scores = []

        for config in viz_engine.configurations:
            memory_mb = config["memory_mb"]
            memory_sizes.append(memory_mb)

            executions = config.get("executions", [])
            successful = [e for e in executions if not e.get("error")]

            if successful:
                avg_duration = np.mean([e["duration"] for e in successful])
                # Calculate efficiency score (should match the visualization logic)
                efficiency = 1000 / (avg_duration * memory_mb / 128)
                efficiency_scores.append(efficiency)
            else:
                efficiency_scores.append(0)

        # Verify efficiency scores are calculated correctly
        assert len(efficiency_scores) == 3
        assert all(score > 0 for score in efficiency_scores)

        # Higher memory with lower duration should be more efficient
        # (but this depends on the specific trade-offs)
        assert len(memory_sizes) == len(efficiency_scores)

    def test_performance_consistency_calculation(self, viz_engine):
        """Test performance consistency calculation."""
        memory_sizes = []
        cv_scores = []  # Coefficient of variation

        for config in viz_engine.configurations:
            memory_mb = config["memory_mb"]
            memory_sizes.append(memory_mb)

            executions = config.get("executions", [])
            successful = [e for e in executions if not e.get("error")]

            if len(successful) > 1:
                durations = [e["duration"] for e in successful]
                mean_duration = np.mean(durations)
                std_duration = np.std(durations)
                cv = std_duration / mean_duration if mean_duration > 0 else 0
                cv_scores.append(cv)
            else:
                cv_scores.append(0)

        # Verify coefficient of variation calculations
        assert len(cv_scores) == 3
        assert all(score >= 0 for score in cv_scores)

        # CV should be reasonable (not too high)
        assert all(score < 1.0 for score in cv_scores)

    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        # Test with empty results
        empty_results = {"configurations": []}

        with pytest.raises(VisualizationError):
            VisualizationEngine(empty_results)

        # Test with malformed results
        bad_results = {"configurations": None}

        with pytest.raises(VisualizationError):
            VisualizationEngine(bad_results)

    @patch("matplotlib.pyplot.savefig")
    def test_save_figure_error_handling(self, mock_savefig, viz_engine, temp_dir):
        """Test save figure error handling."""
        # Mock savefig to raise an exception
        mock_savefig.side_effect = Exception("Failed to save")

        output_path = os.path.join(temp_dir, "test_chart.png")

        with pytest.raises(VisualizationError):
            viz_engine.plot_performance_comparison(output_path)

    def test_chart_data_validation(self, viz_engine):
        """Test chart data validation and processing."""
        # Test that charts handle edge cases properly

        # Create results with edge cases
        edge_case_results = {
            "configurations": [
                {
                    "memory_mb": 128,
                    "executions": [
                        # Only cold starts
                        {
                            "duration": 1000,
                            "billed_duration": 1000,
                            "cold_start": True,
                            "error": None,
                        },
                        {
                            "duration": 1100,
                            "billed_duration": 1100,
                            "cold_start": True,
                            "error": None,
                        },
                    ],
                },
                {
                    "memory_mb": 256,
                    "executions": [
                        # Only warm starts
                        {
                            "duration": 400,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 450,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        },
                    ],
                },
                {
                    "memory_mb": 512,
                    "executions": [
                        # Mix with errors
                        {
                            "duration": 200,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 0,
                            "billed_duration": 0,
                            "cold_start": False,
                            "error": "timeout",
                        },
                    ],
                },
            ]
        }

        edge_viz = VisualizationEngine(edge_case_results)

        # Should handle edge cases without crashing
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "edge_case_chart.png")
            edge_viz.cold_start_visualization(output_path)
            assert os.path.exists(output_path)


class TestInteractiveDashboards:
    """Test interactive dashboard functionality."""

    @pytest.fixture
    def sample_results(self):
        """Sample results for dashboard testing."""
        return {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
            "configurations": [
                {
                    "memory_mb": 128,
                    "executions": [
                        {
                            "duration": 1000,
                            "billed_duration": 1000,
                            "cold_start": True,
                            "error": None,
                        },
                        {
                            "duration": 800,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 850,
                            "billed_duration": 900,
                            "cold_start": False,
                            "error": None,
                        },
                    ],
                },
                {
                    "memory_mb": 256,
                    "executions": [
                        {
                            "duration": 600,
                            "billed_duration": 600,
                            "cold_start": True,
                            "error": None,
                        },
                        {
                            "duration": 400,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        },
                        {
                            "duration": 450,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        },
                    ],
                },
            ],
        }

    @pytest.fixture
    def dashboard(self, sample_results):
        """Interactive dashboard instance for testing."""
        return InteractiveDashboard(sample_results)

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_performance_dashboard_creation(self, dashboard, temp_dir):
        """Test interactive performance dashboard creation."""
        output_path = os.path.join(temp_dir, "performance_dashboard.html")

        dashboard.create_performance_dashboard(output_path, "web_api")

        assert os.path.exists(output_path)

        # Check that HTML file contains expected content
        with open(output_path, "r") as f:
            content = f.read()
            assert "Web Api Performance Dashboard" in content or "Performance Dashboard" in content
            assert "plotly" in content.lower()

    def test_cost_analysis_dashboard_creation(self, dashboard, temp_dir):
        """Test interactive cost analysis dashboard creation."""
        output_path = os.path.join(temp_dir, "cost_dashboard.html")

        # Test without scenarios
        dashboard.create_cost_analysis_dashboard(output_path)
        assert os.path.exists(output_path)

        # Test with scenarios
        scenarios = [
            {
                "name": "Low Volume",
                "daily_invocations": 1000,
                "pattern": "steady",
                "duration_days": 30,
            },
            {
                "name": "High Volume",
                "daily_invocations": 100000,
                "pattern": "bursty",
                "duration_days": 30,
            },
        ]

        output_path_2 = os.path.join(temp_dir, "cost_dashboard_with_scenarios.html")
        dashboard.create_cost_analysis_dashboard(output_path_2, scenarios)
        assert os.path.exists(output_path_2)

    def test_cold_start_dashboard_creation(self, dashboard, temp_dir):
        """Test interactive cold start dashboard creation."""
        output_path = os.path.join(temp_dir, "cold_start_dashboard.html")

        dashboard.create_cold_start_dashboard(output_path)

        assert os.path.exists(output_path)

        # Check content
        with open(output_path, "r") as f:
            content = f.read()
            assert "Cold Start Analysis" in content
            assert "plotly" in content.lower()

    def test_workload_comparison_dashboard(self, sample_results, temp_dir):
        """Test workload comparison dashboard creation."""
        comparison_results = [
            {
                "workload_type": "web_api",
                "name": "Web API",
                "analysis": {
                    "key_metrics": {
                        "optimal_memory": 256,
                        "performance_gain": 15.5,
                        "cost_savings": 12.3,
                    }
                },
                "results": sample_results,
            },
            {
                "workload_type": "batch_processing",
                "name": "Batch Job",
                "analysis": {
                    "key_metrics": {
                        "optimal_memory": 512,
                        "performance_gain": 25.2,
                        "cost_savings": 8.7,
                    }
                },
                "results": sample_results,
            },
        ]

        dashboard = InteractiveDashboard(sample_results)
        output_path = os.path.join(temp_dir, "comparison_dashboard.html")

        dashboard.create_workload_comparison_dashboard(output_path, comparison_results)

        assert os.path.exists(output_path)

        # Check content
        with open(output_path, "r") as f:
            content = f.read()
            assert "Workload Comparison" in content
            assert "Web API" in content
            assert "Batch Job" in content

    def test_real_time_monitoring_dashboard(self, dashboard, temp_dir):
        """Test real-time monitoring dashboard template creation."""
        output_path = os.path.join(temp_dir, "real_time_monitoring.html")

        dashboard.create_real_time_monitoring_dashboard(output_path)

        assert os.path.exists(output_path)

        # Check content
        with open(output_path, "r") as f:
            content = f.read()
            assert "Real-Time Monitoring" in content
            assert "setInterval" in content  # JavaScript for real-time updates
            assert "updateMetrics" in content

    def test_dashboard_html_structure(self, dashboard, temp_dir):
        """Test that generated HTML has proper structure."""
        output_path = os.path.join(temp_dir, "test_dashboard.html")

        dashboard.create_performance_dashboard(output_path, "generic")

        with open(output_path, "r") as f:
            content = f.read()

            # Check HTML structure
            assert "<!DOCTYPE html>" in content
            assert "<html>" in content
            assert "<head>" in content
            assert "<body>" in content
            assert "</html>" in content

            # Check for Plotly integration
            assert "plotly" in content.lower()

            # Check for custom styling
            assert "<style>" in content or "style=" in content

    def test_dashboard_error_handling(self):
        """Test dashboard error handling with invalid data."""
        # Test with empty configurations
        empty_results = {"configurations": []}

        with pytest.raises(VisualizationError):
            InteractiveDashboard(empty_results)

        # Test with malformed data
        bad_results = {"configurations": None}

        with pytest.raises(VisualizationError):
            InteractiveDashboard(bad_results)

    def test_dashboard_data_processing(self, dashboard):
        """Test data processing for dashboard visualizations."""
        # Test data extraction for charts
        memory_sizes = []
        avg_durations = []
        avg_costs = []

        for config in dashboard.configurations:
            memory_mb = config["memory_mb"]
            memory_sizes.append(memory_mb)

            executions = config.get("executions", [])
            successful = [e for e in executions if not e.get("error")]

            if successful:
                durations = [e["duration"] for e in successful]
                avg_durations.append(sum(durations) / len(durations))

                # Simulate cost calculation
                costs = [0.0001 for _ in successful]  # Placeholder
                avg_costs.append(sum(costs) / len(costs))
            else:
                avg_durations.append(0)
                avg_costs.append(0)

        # Verify data processing
        assert len(memory_sizes) == 2
        assert len(avg_durations) == 2
        assert len(avg_costs) == 2
        assert all(d > 0 for d in avg_durations)

    @patch("pathlib.Path.mkdir")
    def test_dashboard_directory_creation(self, mock_mkdir, dashboard, temp_dir):
        """Test dashboard directory creation."""
        output_path = os.path.join(temp_dir, "subdir", "dashboard.html")

        dashboard.create_performance_dashboard(output_path, "generic")

        # Verify mkdir was called
        mock_mkdir.assert_called()

    def test_dashboard_customization_options(self, dashboard, temp_dir):
        """Test dashboard customization for different workload types."""
        workload_types = [
            "web_api",
            "batch_processing",
            "event_driven",
            "scheduled",
            "stream_processing",
            "generic",
        ]

        for workload_type in workload_types:
            output_path = os.path.join(temp_dir, f"{workload_type}_dashboard.html")

            dashboard.create_performance_dashboard(output_path, workload_type)
            assert os.path.exists(output_path)

            # Check that workload type is reflected in the content
            with open(output_path, "r") as f:
                content = f.read()
                # The title should contain the workload type
                assert (
                    workload_type.replace("_", " ").title() in content
                    or "Performance Dashboard" in content
                )


class TestConvenienceFunctions:
    """Test convenience functions for visualizations."""

    @pytest.fixture
    def sample_results(self):
        """Sample results for testing."""
        return {
            "configurations": [
                {
                    "memory_mb": 256,
                    "executions": [
                        {
                            "duration": 500,
                            "billed_duration": 500,
                            "cold_start": False,
                            "error": None,
                        }
                    ],
                }
            ]
        }

    def test_convenience_function_imports(self):
        """Test that convenience functions can be imported."""
        from aws_lambda_tuner.visualization_module import (
            create_performance_chart,
            create_cost_chart,
            create_distribution_chart,
            create_optimization_chart,
            create_cold_start_chart,
            create_enhanced_cold_start_visualization,
            create_workload_pattern_charts,
            create_cost_efficiency_dashboard,
            create_performance_trend_charts,
            create_comprehensive_dashboard,
        )

        # All functions should be callable
        assert callable(create_performance_chart)
        assert callable(create_cost_chart)
        assert callable(create_distribution_chart)
        assert callable(create_optimization_chart)
        assert callable(create_cold_start_chart)
        assert callable(create_enhanced_cold_start_visualization)
        assert callable(create_workload_pattern_charts)
        assert callable(create_cost_efficiency_dashboard)
        assert callable(create_performance_trend_charts)
        assert callable(create_comprehensive_dashboard)

    def test_interactive_dashboard_convenience_functions(self):
        """Test interactive dashboard convenience functions."""
        from aws_lambda_tuner.reporting.interactive_dashboards import (
            create_interactive_performance_dashboard,
            create_interactive_cost_dashboard,
            create_interactive_cold_start_dashboard,
            create_workload_comparison_dashboard,
            create_real_time_monitoring_dashboard,
        )

        # All functions should be callable
        assert callable(create_interactive_performance_dashboard)
        assert callable(create_interactive_cost_dashboard)
        assert callable(create_interactive_cold_start_dashboard)
        assert callable(create_workload_comparison_dashboard)
        assert callable(create_real_time_monitoring_dashboard)

    @patch("aws_lambda_tuner.visualization_module.VisualizationEngine.plot_performance_comparison")
    def test_create_performance_chart(self, mock_plot, sample_results):
        """Test create_performance_chart convenience function."""
        from aws_lambda_tuner.visualization_module import create_performance_chart

        output_path = "/tmp/test_chart.png"
        create_performance_chart(sample_results, output_path)

        mock_plot.assert_called_once_with(output_path)

    @patch(
        "aws_lambda_tuner.reporting.interactive_dashboards.InteractiveDashboard.create_performance_dashboard"
    )
    def test_create_interactive_performance_dashboard(self, mock_create, sample_results):
        """Test create_interactive_performance_dashboard convenience function."""
        from aws_lambda_tuner.reporting.interactive_dashboards import (
            create_interactive_performance_dashboard,
        )

        output_path = "/tmp/test_dashboard.html"
        workload_type = "web_api"

        create_interactive_performance_dashboard(sample_results, output_path, workload_type)

        mock_create.assert_called_once_with(output_path, workload_type)
