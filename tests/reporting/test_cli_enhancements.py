"""
Tests for CLI enhancements and workload-guided workflows.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from aws_lambda_tuner.cli_module import (
    cli,
    workload_wizard,
    cost_projection,
    compare_workloads,
    dashboard,
    export,
    cost_explorer,
    _interactive_workload_selection,
    _get_workload_config,
    _customize_config_for_workload,
)
from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.report_service import WorkloadType


class TestWorkloadWizard:
    """Test workload wizard functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Sample tuning results."""
        return {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
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
                },
                {
                    "memory_mb": 512,
                    "executions": [
                        {
                            "duration": 300,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                        }
                    ],
                },
            ],
        }

    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.asyncio.run")
    def test_workload_wizard_web_api(
        self, mock_asyncio_run, mock_orchestrator, runner, sample_results
    ):
        """Test workload wizard for web API workload."""
        # Mock the orchestrator to return sample results
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance
        mock_asyncio_run.return_value = sample_results

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                workload_wizard,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test-function",
                    "--workload-type",
                    "web_api",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0
            assert "Web API optimization" in result.output
            assert "Workload analysis complete" in result.output

            # Check that output files were created
            assert os.path.exists(os.path.join(tmpdir, "web_api_report.html"))
            assert os.path.exists(os.path.join(tmpdir, "web_api_analysis.json"))
            assert os.path.exists(os.path.join(tmpdir, "web_api_dashboard.html"))

    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.asyncio.run")
    def test_workload_wizard_batch_processing(
        self, mock_asyncio_run, mock_orchestrator, runner, sample_results
    ):
        """Test workload wizard for batch processing workload."""
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance
        mock_asyncio_run.return_value = sample_results

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                workload_wizard,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test-function",
                    "--workload-type",
                    "batch_processing",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0
            assert "Batch Processing optimization" in result.output

    @patch("aws_lambda_tuner.cli_module.click.prompt")
    def test_interactive_workload_selection(self, mock_prompt):
        """Test interactive workload selection."""
        # Mock user input
        mock_prompt.return_value = "1"  # Select web_api

        workload_type = _interactive_workload_selection()

        assert workload_type == "web_api"
        mock_prompt.assert_called_once()

    @patch("aws_lambda_tuner.cli_module.click.prompt")
    @patch("aws_lambda_tuner.cli_module.click.confirm")
    def test_get_workload_config_web_api(self, mock_confirm, mock_prompt):
        """Test workload configuration for web API."""
        # Mock user inputs
        mock_prompt.side_effect = [
            100,
            10,
            5.0,
        ]  # target_latency, expected_concurrency, cold_start_tolerance

        config = _get_workload_config("web_api", True)

        assert config["target_latency"] == 100
        assert config["expected_concurrency"] == 10
        assert config["cold_start_tolerance"] == 5.0

    @patch("aws_lambda_tuner.cli_module.click.prompt")
    @patch("aws_lambda_tuner.cli_module.click.confirm")
    def test_get_workload_config_batch_processing(self, mock_confirm, mock_prompt):
        """Test workload configuration for batch processing."""
        # Mock user inputs
        mock_prompt.side_effect = [1000, 10]  # batch_size, processing_time
        mock_confirm.return_value = True  # cost_priority

        config = _get_workload_config("batch_processing", True)

        assert config["batch_size"] == 1000
        assert config["processing_time"] == 10
        assert config["cost_priority"] is True

    def test_customize_config_for_workload_web_api(self):
        """Test configuration customization for web API workload."""
        # Create base config
        base_config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            payload="{}",
            memory_sizes=[128, 256, 512],
            iterations=10,
            strategy="balanced",
            concurrent_executions=5,
            timeout=300,
            dry_run=False,
            output_dir="./output",
        )

        workload_config = {"target_latency": 100}
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"

        customized_config = _customize_config_for_workload(
            base_config, "web_api", workload_config, function_arn
        )

        # Web API should focus on lower memory sizes
        assert customized_config.memory_sizes == [128, 256, 512, 1024, 1536]
        assert customized_config.iterations == 15  # More iterations for statistical significance
        assert customized_config.concurrent_executions == 10  # Higher concurrency

    def test_customize_config_for_workload_batch_processing(self):
        """Test configuration customization for batch processing workload."""
        base_config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            payload="{}",
            memory_sizes=[128, 256, 512],
            iterations=10,
            strategy="balanced",
            concurrent_executions=5,
            timeout=300,
            dry_run=False,
            output_dir="./output",
        )

        workload_config = {"batch_size": 1000}
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"

        customized_config = _customize_config_for_workload(
            base_config, "batch_processing", workload_config, function_arn
        )

        # Batch processing should test higher memory sizes
        assert customized_config.memory_sizes == [512, 1024, 2048, 3008]
        assert customized_config.concurrent_executions == 5  # Lower concurrency

    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.asyncio.run")
    @patch("aws_lambda_tuner.cli_module._interactive_workload_selection")
    def test_workload_wizard_interactive_mode(
        self, mock_selection, mock_asyncio_run, mock_orchestrator, runner, sample_results
    ):
        """Test workload wizard in interactive mode."""
        mock_selection.return_value = "event_driven"
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance
        mock_asyncio_run.return_value = sample_results

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                workload_wizard,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--interactive",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0
            mock_selection.assert_called_once()


class TestCostProjection:
    """Test cost projection functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Sample results file for testing."""
        results = {
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            yield f.name

        os.unlink(f.name)

    def test_cost_projection_default_scenarios(self, runner, sample_results_file):
        """Test cost projection with default scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                cost_projection,
                [
                    sample_results_file,
                    "--monthly-invocations",
                    "1000000",
                    "--pattern",
                    "steady",
                    "--output-dir",
                    tmpdir,
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "Cost Projection Analysis" in result.output
            assert "Cost projection analysis saved" in result.output

            # Check output files exist
            output_files = os.listdir(tmpdir)
            assert any("cost_projection_" in f for f in output_files)

    def test_cost_projection_custom_scenarios(self, runner, sample_results_file):
        """Test cost projection with custom scenarios."""
        scenarios = {
            "scenarios": [
                {
                    "name": "dev_environment",
                    "daily_invocations": 100,
                    "pattern": "steady",
                    "duration_days": 30,
                },
                {
                    "name": "prod_environment",
                    "daily_invocations": 50000,
                    "pattern": "bursty",
                    "duration_days": 30,
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as scenarios_file:
            json.dump(scenarios, scenarios_file)
            scenarios_file_path = scenarios_file.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    cost_projection,
                    [
                        sample_results_file,
                        "--scenarios-file",
                        scenarios_file_path,
                        "--output-dir",
                        tmpdir,
                        "--format",
                        "html",
                    ],
                )

                assert result.exit_code == 0

                # Check output files exist
                output_files = os.listdir(tmpdir)
                assert any(f.endswith(".html") for f in output_files)
        finally:
            os.unlink(scenarios_file_path)

    def test_cost_projection_different_formats(self, runner, sample_results_file):
        """Test cost projection with different export formats."""
        formats = ["html", "json", "pdf", "excel"]

        for format_type in formats:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    cost_projection,
                    [sample_results_file, "--output-dir", tmpdir, "--format", format_type],
                )

                # PDF and Excel might fail due to missing dependencies, that's ok
                if format_type in ["html", "json"]:
                    assert result.exit_code == 0

                if result.exit_code == 0:
                    output_files = os.listdir(tmpdir)
                    assert len(output_files) > 0


class TestWorkloadComparison:
    """Test workload comparison functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_files(self):
        """Multiple sample results files for comparison."""
        results1 = {
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

        results2 = {
            "configurations": [
                {
                    "memory_mb": 512,
                    "executions": [
                        {
                            "duration": 300,
                            "billed_duration": 300,
                            "cold_start": False,
                            "error": None,
                        }
                    ],
                }
            ]
        }

        files = []
        for i, results in enumerate([results1, results2]):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(results, f)
                files.append(f.name)

        yield files

        for f in files:
            os.unlink(f)

    def test_compare_workloads_basic(self, runner, sample_results_files):
        """Test basic workload comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                compare_workloads,
                [
                    sample_results_files[0],
                    sample_results_files[1],
                    "--workload-types",
                    "web_api,batch_processing",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0
            assert "Multi-Workload Performance Comparison" in result.output
            assert "Workload comparison saved" in result.output

            # Check output files
            assert os.path.exists(os.path.join(tmpdir, "workload_comparison.html"))
            assert os.path.exists(os.path.join(tmpdir, "comparison_analysis.json"))

    def test_compare_workloads_interactive_dashboard(self, runner, sample_results_files):
        """Test workload comparison with interactive dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                compare_workloads,
                [
                    sample_results_files[0],
                    sample_results_files[1],
                    "--workload-types",
                    "web_api,event_driven",
                    "--output-dir",
                    tmpdir,
                    "--interactive-dashboard",
                ],
            )

            assert result.exit_code == 0

            # Check for interactive dashboard file
            assert os.path.exists(os.path.join(tmpdir, "interactive_comparison.html"))

    def test_compare_workloads_insufficient_files(self, runner):
        """Test workload comparison with insufficient files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"configurations": []}, f)
            file_path = f.name

        try:
            result = runner.invoke(compare_workloads, [file_path])

            assert result.exit_code == 1
            assert "At least 2 results files are required" in result.output
        finally:
            os.unlink(file_path)

    def test_compare_workloads_mismatched_types(self, runner, sample_results_files):
        """Test workload comparison with mismatched workload types."""
        result = runner.invoke(
            compare_workloads,
            [
                sample_results_files[0],
                sample_results_files[1],
                "--workload-types",
                "web_api",  # Only one type for two files
            ],
        )

        assert result.exit_code == 1
        assert "Number of workload types" in result.output
        assert "must match number of result files" in result.output


class TestDashboardGeneration:
    """Test dashboard generation functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Sample results file for testing."""
        results = {
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            yield f.name

        os.unlink(f.name)

    def test_dashboard_generation_basic(self, runner, sample_results_file):
        """Test basic dashboard generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                dashboard,
                [sample_results_file, "--output-dir", tmpdir, "--workload-type", "web_api"],
            )

            assert result.exit_code == 0
            assert "Generating Interactive Dashboard" in result.output
            assert "Dashboard suite ready" in result.output

            # Check that dashboard files were created
            expected_files = [
                "performance_dashboard.html",
                "cost_dashboard.html",
                "cold_start_dashboard.html",
                "comprehensive_dashboard.html",
                "index.html",
            ]

            for file_name in expected_files:
                assert os.path.exists(os.path.join(tmpdir, file_name))

    def test_dashboard_generation_with_cost_scenarios(self, runner, sample_results_file):
        """Test dashboard generation with cost scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                dashboard, [sample_results_file, "--output-dir", tmpdir, "--include-cost-scenarios"]
            )

            assert result.exit_code == 0

            # Check that all files exist
            assert os.path.exists(os.path.join(tmpdir, "cost_dashboard.html"))

    def test_dashboard_generation_with_real_time(self, runner, sample_results_file):
        """Test dashboard generation with real-time monitoring template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                dashboard, [sample_results_file, "--output-dir", tmpdir, "--real-time-template"]
            )

            assert result.exit_code == 0

            # Check for real-time monitoring file
            assert os.path.exists(os.path.join(tmpdir, "real_time_monitoring.html"))


class TestExportFunctionality:
    """Test export functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Sample results file for testing."""
        results = {
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            yield f.name

        os.unlink(f.name)

    def test_export_json_format(self, runner, sample_results_file):
        """Test export in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "export_test.json")

            result = runner.invoke(
                export,
                [
                    sample_results_file,
                    "--format",
                    "json",
                    "--workload-type",
                    "web_api",
                    "--output",
                    output_file,
                ],
            )

            assert result.exit_code == 0
            assert "Report exported to" in result.output
            assert os.path.exists(output_file)

    def test_export_csv_format(self, runner, sample_results_file):
        """Test export in CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "export_test.csv")

            result = runner.invoke(
                export, [sample_results_file, "--format", "csv", "--output", output_file]
            )

            assert result.exit_code == 0
            assert os.path.exists(output_file)

    def test_export_auto_filename(self, runner, sample_results_file):
        """Test export with automatic filename generation."""
        result = runner.invoke(
            export, [sample_results_file, "--format", "json", "--workload-type", "batch_processing"]
        )

        # Should not fail even without explicit output path
        assert result.exit_code == 0
        assert "Report exported to" in result.output


class TestCostExplorerIntegration:
    """Test Cost Explorer integration."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_cost_explorer_template_generation(self, runner):
        """Test Cost Explorer template generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                cost_explorer,
                [
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--days",
                    "14",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0
            assert "AWS Cost Explorer Integration" in result.output
            assert "Cost Explorer template saved" in result.output

            # Check that template file was created
            assert os.path.exists(os.path.join(tmpdir, "cost_explorer_analysis.html"))

            # Check template content
            with open(os.path.join(tmpdir, "cost_explorer_analysis.html"), "r") as f:
                content = f.read()
                assert "Cost Explorer Integration" in content
                assert "arn:aws:lambda:us-east-1:123456789012:function:test" in content
                assert "Last 14 days" in content


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_workload_wizard_missing_function_arn(self, runner):
        """Test workload wizard with missing function ARN."""
        result = runner.invoke(workload_wizard, ["--workload-type", "web_api"])

        assert result.exit_code != 0
        # Should fail due to missing required function ARN

    def test_cost_projection_invalid_file(self, runner):
        """Test cost projection with invalid results file."""
        result = runner.invoke(cost_projection, ["/nonexistent/file.json"])

        assert result.exit_code != 0
        # Should fail due to file not existing

    def test_dashboard_invalid_file(self, runner):
        """Test dashboard generation with invalid file."""
        result = runner.invoke(dashboard, ["/nonexistent/file.json"])

        assert result.exit_code != 0
        # Should fail due to file not existing

    def test_export_invalid_format(self, runner):
        """Test export with invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"configurations": []}, f)
            file_path = f.name

        try:
            result = runner.invoke(export, [file_path, "--format", "invalid_format"])

            assert result.exit_code != 0
            # Should fail due to invalid format choice
        finally:
            os.unlink(file_path)


class TestCLIHelperFunctions:
    """Test CLI helper functions."""

    def test_workload_config_non_interactive(self):
        """Test workload configuration in non-interactive mode."""
        config = _get_workload_config("web_api", False)

        # Should return empty config for non-interactive mode
        assert config == {}

    def test_customize_config_unknown_workload(self):
        """Test configuration customization for unknown workload type."""
        base_config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            payload="{}",
            memory_sizes=[128, 256, 512],
            iterations=10,
            strategy="balanced",
            concurrent_executions=5,
            timeout=300,
            dry_run=False,
            output_dir="./output",
        )

        workload_config = {}
        function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"

        customized_config = _customize_config_for_workload(
            base_config, "unknown_workload", workload_config, function_arn
        )

        # Should fall back to default configuration
        assert customized_config.memory_sizes == base_config.memory_sizes
        assert customized_config.iterations == base_config.iterations
        assert customized_config.concurrent_executions == base_config.concurrent_executions

    def test_all_workload_types_have_configs(self):
        """Test that all workload types have configuration customization."""
        workload_types = [
            "web_api",
            "batch_processing",
            "event_driven",
            "scheduled",
            "stream_processing",
        ]

        base_config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            payload="{}",
            memory_sizes=[128, 256, 512],
            iterations=10,
            strategy="balanced",
            concurrent_executions=5,
            timeout=300,
            dry_run=False,
            output_dir="./output",
        )

        for workload_type in workload_types:
            customized_config = _customize_config_for_workload(
                base_config,
                workload_type,
                {},
                "arn:aws:lambda:us-east-1:123456789012:function:test",
            )

            # Each workload type should produce a valid configuration
            assert customized_config.function_arn is not None
            assert customized_config.memory_sizes is not None
            assert len(customized_config.memory_sizes) > 0
            assert customized_config.iterations > 0
            assert customized_config.concurrent_executions > 0
