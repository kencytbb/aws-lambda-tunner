"""
Tests for core CLI commands (init, tune, report, visualize, templates).
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from click.testing import CliRunner

from aws_lambda_tuner.cli_module import cli, init, tune, report, visualize, templates
from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.exceptions import TunerException, ConfigurationError


class TestInitCommand:
    """Test the init command functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_init_command_default_template(self, runner):
        """Test init command with default template."""
        with runner.isolated_filesystem():
            result = runner.invoke(init)

            assert result.exit_code == 0
            assert "Configuration file created: tuner.config.json" in result.output
            assert "Template used: balanced" in result.output
            assert "Next steps:" in result.output

            # Check if config file was created
            assert os.path.exists("tuner.config.json")

            # Verify config file content
            with open("tuner.config.json", "r") as f:
                config = json.load(f)

            assert (
                config["function_arn"]
                == "arn:aws:lambda:us-east-1:123456789012:function:my-function"
            )
            assert "memory_sizes" in config
            assert "iterations" in config
            assert "strategy" in config

    def test_init_command_with_template(self, runner):
        """Test init command with specific template."""
        with runner.isolated_filesystem():
            result = runner.invoke(init, ["--template", "speed"])

            assert result.exit_code == 0
            assert "Template used: speed" in result.output
            assert os.path.exists("tuner.config.json")

    def test_init_command_custom_output(self, runner):
        """Test init command with custom output path."""
        with runner.isolated_filesystem():
            result = runner.invoke(init, ["--output", "custom.config.json"])

            assert result.exit_code == 0
            assert "Configuration file created: custom.config.json" in result.output
            assert os.path.exists("custom.config.json")

    def test_init_command_invalid_template(self, runner):
        """Test init command with invalid template."""
        result = runner.invoke(init, ["--template", "invalid"])

        assert result.exit_code != 0
        # Click should handle the invalid choice

    @patch("aws_lambda_tuner.cli_module.ConfigManager")
    def test_init_command_config_error(self, mock_config_manager, runner):
        """Test init command with configuration error."""
        mock_config_manager.return_value.create_from_template.side_effect = Exception(
            "Config error"
        )

        result = runner.invoke(init)

        assert result.exit_code == 1
        assert "Error creating configuration" in result.output


class TestTuneCommand:
    """Test the tune command functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_config(self):
        """Sample configuration data."""
        return {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
            "payload": "{}",
            "memory_sizes": [256, 512, 1024],
            "iterations": 5,
            "strategy": "balanced",
            "concurrent_executions": 3,
            "timeout": 300,
            "dry_run": False,
            "output_dir": "./tuning-results",
        }

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

    def test_tune_command_missing_function_arn(self, runner):
        """Test tune command without function ARN."""
        result = runner.invoke(tune)

        assert result.exit_code == 1
        assert "Function ARN is required" in result.output

    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.ReportGenerator")
    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    @patch("asyncio.run")
    def test_tune_command_with_cli_args(
        self,
        mock_asyncio,
        mock_validate,
        mock_report_gen,
        mock_orchestrator,
        runner,
        sample_results,
    ):
        """Test tune command with CLI arguments."""
        # Setup mocks
        mock_orchestrator.return_value.run_tuning = Mock(return_value=sample_results)
        mock_asyncio.return_value = sample_results
        mock_report_instance = Mock()
        mock_report_gen.return_value = mock_report_instance
        mock_report_instance.get_summary.return_value = {
            "optimal_memory": 512,
            "optimal_duration": 300.0,
            "optimal_cost": 0.000001,
            "performance_gain": 40.0,
            "cost_savings": 20.0,
        }

        with runner.isolated_filesystem():
            result = runner.invoke(
                tune,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--memory-sizes",
                    "256,512,1024",
                    "--iterations",
                    "5",
                    "--strategy",
                    "balanced",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            assert "Starting Lambda performance tuning" in result.output
            assert "DRY RUN MODE" in result.output
            assert "TUNING SUMMARY" in result.output

            # Verify orchestrator was called
            mock_orchestrator.assert_called_once()

            # Verify config was properly constructed
            config_call = mock_orchestrator.call_args[0][0]
            assert config_call.function_arn == "arn:aws:lambda:us-east-1:123456789012:function:test"
            assert config_call.memory_sizes == [256, 512, 1024]
            assert config_call.iterations == 5
            assert config_call.strategy == "balanced"
            assert config_call.dry_run is True

    @patch("aws_lambda_tuner.cli_module.load_json_file")
    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.ReportGenerator")
    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    @patch("asyncio.run")
    def test_tune_command_with_config_file(
        self,
        mock_asyncio,
        mock_validate,
        mock_report_gen,
        mock_orchestrator,
        mock_load_json,
        runner,
        sample_config,
        sample_results,
    ):
        """Test tune command with configuration file."""
        # Setup mocks
        mock_load_json.return_value = sample_config
        mock_orchestrator.return_value.run_tuning = Mock(return_value=sample_results)
        mock_asyncio.return_value = sample_results
        mock_report_instance = Mock()
        mock_report_gen.return_value = mock_report_instance
        mock_report_instance.get_summary.return_value = {
            "optimal_memory": 512,
            "optimal_duration": 300.0,
            "optimal_cost": 0.000001,
            "performance_gain": 40.0,
            "cost_savings": 20.0,
        }

        with runner.isolated_filesystem():
            # Create a dummy config file
            Path("test.config.json").touch()

            result = runner.invoke(tune, ["--config", "test.config.json"])

            assert result.exit_code == 0
            assert "Starting Lambda performance tuning" in result.output

            # Verify config file was loaded
            mock_load_json.assert_called_once_with("test.config.json")

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=False)
    def test_tune_command_invalid_arn(self, mock_validate, runner):
        """Test tune command with invalid ARN."""
        result = runner.invoke(tune, ["--function-arn", "invalid-arn"])

        assert result.exit_code == 1
        assert "Invalid Lambda ARN" in result.output

    def test_tune_command_with_payload_file(self, runner):
        """Test tune command with payload file."""
        with runner.isolated_filesystem():
            # Create payload file
            with open("payload.json", "w") as f:
                json.dump({"key": "value"}, f)

            with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True), patch(
                "aws_lambda_tuner.cli_module.TunerOrchestrator"
            ) as mock_orchestrator, patch("aws_lambda_tuner.cli_module.ReportGenerator"), patch(
                "asyncio.run"
            ):

                result = runner.invoke(
                    tune,
                    [
                        "--function-arn",
                        "arn:aws:lambda:us-east-1:123456789012:function:test",
                        "--payload-file",
                        "payload.json",
                    ],
                )

                # Verify payload was loaded from file
                config_call = mock_orchestrator.call_args[0][0]
                assert '{"key": "value"}' in config_call.payload


class TestReportCommand:
    """Test the report command functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Create a sample results file."""
        results = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "configurations": [
                {"memory_mb": 256, "avg_duration": 500, "avg_cost": 0.000001},
                {"memory_mb": 512, "avg_duration": 300, "avg_cost": 0.000002},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            return f.name

    def test_report_command_summary_format(self, runner, sample_results_file):
        """Test report command with summary format."""
        with patch("aws_lambda_tuner.cli_module.ReportGenerator") as mock_report_gen:
            mock_report_instance = Mock()
            mock_report_gen.return_value = mock_report_instance
            mock_report_instance.get_summary.return_value = {
                "optimal_memory": 512,
                "optimal_duration": 300.0,
                "performance_gain": 40.0,
            }

            result = runner.invoke(report, [sample_results_file, "--format", "summary"])

            assert result.exit_code == 0
            assert "PERFORMANCE SUMMARY" in result.output
            assert "Optimal Memory:" in result.output

        # Clean up
        os.unlink(sample_results_file)

    def test_report_command_detailed_format(self, runner, sample_results_file):
        """Test report command with detailed format."""
        with patch("aws_lambda_tuner.cli_module.ReportGenerator") as mock_report_gen:
            mock_report_instance = Mock()
            mock_report_gen.return_value = mock_report_instance
            mock_report_instance.get_detailed_report.return_value = {"detailed": "data"}

            result = runner.invoke(report, [sample_results_file, "--format", "detailed"])

            assert result.exit_code == 0
            assert '"detailed": "data"' in result.output

        # Clean up
        os.unlink(sample_results_file)

    def test_report_command_json_format(self, runner, sample_results_file):
        """Test report command with JSON format."""
        result = runner.invoke(report, [sample_results_file, "--format", "json"])

        assert result.exit_code == 0
        assert '"function_arn"' in result.output

        # Clean up
        os.unlink(sample_results_file)

    def test_report_command_file_not_found(self, runner):
        """Test report command with non-existent file."""
        result = runner.invoke(report, ["nonexistent.json"])

        assert result.exit_code != 0


class TestVisualizeCommand:
    """Test the visualize command functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Create a sample results file."""
        results = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "configurations": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            return f.name

    @patch("aws_lambda_tuner.cli_module.VisualizationEngine")
    def test_visualize_command_all_charts(self, mock_viz_engine, runner, sample_results_file):
        """Test visualize command with all charts."""
        mock_viz_instance = Mock()
        mock_viz_engine.return_value = mock_viz_instance

        with runner.isolated_filesystem():
            result = runner.invoke(visualize, [sample_results_file, "--charts", "all"])

            assert result.exit_code == 0
            assert "Generated: performance-comparison.png" in result.output
            assert "Generated: cost-analysis.png" in result.output
            assert "Generated: duration-distribution.png" in result.output
            assert "Generated: optimization-curve.png" in result.output

            # Verify all chart methods were called
            mock_viz_instance.plot_performance_comparison.assert_called()
            mock_viz_instance.plot_cost_analysis.assert_called()
            mock_viz_instance.plot_duration_distribution.assert_called()
            mock_viz_instance.plot_optimization_curve.assert_called()

        # Clean up
        os.unlink(sample_results_file)

    @patch("aws_lambda_tuner.cli_module.VisualizationEngine")
    def test_visualize_command_specific_charts(self, mock_viz_engine, runner, sample_results_file):
        """Test visualize command with specific charts."""
        mock_viz_instance = Mock()
        mock_viz_engine.return_value = mock_viz_instance

        with runner.isolated_filesystem():
            result = runner.invoke(
                visualize, [sample_results_file, "--charts", "performance", "--charts", "cost"]
            )

            assert result.exit_code == 0
            assert "Generated: performance-comparison.png" in result.output
            assert "Generated: cost-analysis.png" in result.output

            # Verify only specified charts were called
            mock_viz_instance.plot_performance_comparison.assert_called()
            mock_viz_instance.plot_cost_analysis.assert_called()
            mock_viz_instance.plot_duration_distribution.assert_not_called()
            mock_viz_instance.plot_optimization_curve.assert_not_called()

        # Clean up
        os.unlink(sample_results_file)

    def test_visualize_command_custom_output_dir(self, runner, sample_results_file):
        """Test visualize command with custom output directory."""
        with patch("aws_lambda_tuner.cli_module.VisualizationEngine"), runner.isolated_filesystem():

            result = runner.invoke(
                visualize,
                [sample_results_file, "--output-dir", "./custom-charts", "--charts", "performance"],
            )

            assert result.exit_code == 0
            assert "All visualizations saved to: ./custom-charts" in result.output

        # Clean up
        os.unlink(sample_results_file)


class TestTemplatesCommand:
    """Test the templates command functionality."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_templates_command(self, runner):
        """Test templates command output."""
        result = runner.invoke(templates)

        assert result.exit_code == 0
        assert "Available Configuration Templates" in result.output
        assert "speed" in result.output
        assert "cost" in result.output
        assert "balanced" in result.output
        assert "comprehensive" in result.output
        assert "Optimized for fastest execution time" in result.output
        assert "Optimized for lowest cost" in result.output
        assert "Balance between speed and cost" in result.output
        assert "aws-lambda-tuner init --template" in result.output


class TestCLIValidation:
    """Test CLI argument validation and error handling."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_memory_sizes_parsing(self, runner):
        """Test memory sizes parsing from CLI."""
        with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True), patch(
            "aws_lambda_tuner.cli_module.TunerOrchestrator"
        ) as mock_orchestrator, patch("aws_lambda_tuner.cli_module.ReportGenerator"), patch(
            "asyncio.run"
        ):

            result = runner.invoke(
                tune,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--memory-sizes",
                    "256,512,1024,1536",
                ],
            )

            # Verify memory sizes were parsed correctly
            config_call = mock_orchestrator.call_args[0][0]
            assert config_call.memory_sizes == [256, 512, 1024, 1536]

    def test_invalid_memory_sizes_parsing(self, runner):
        """Test invalid memory sizes parsing."""
        with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True):

            result = runner.invoke(
                tune,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--memory-sizes",
                    "invalid,sizes",
                ],
            )

            assert result.exit_code == 1

    def test_verbose_logging(self, runner):
        """Test verbose logging flag."""
        with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True), patch(
            "aws_lambda_tuner.cli_module.TunerOrchestrator"
        ), patch("aws_lambda_tuner.cli_module.ReportGenerator"), patch("asyncio.run"), patch(
            "logging.getLogger"
        ) as mock_logger:

            runner.invoke(
                tune,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--verbose",
                ],
            )

            # Verify logger level was set to DEBUG
            mock_logger.return_value.setLevel.assert_called()


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    def test_tuner_exception_handling(self, mock_orchestrator, mock_validate, runner):
        """Test handling of TunerException."""
        mock_orchestrator.return_value.run_tuning.side_effect = TunerException("Test error")

        with patch("asyncio.run", side_effect=TunerException("Test error")):
            result = runner.invoke(
                tune, ["--function-arn", "arn:aws:lambda:us-east-1:123456789012:function:test"]
            )

            assert result.exit_code == 1
            assert "Tuner error: Test error" in result.output

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    def test_unexpected_exception_handling(self, mock_orchestrator, mock_validate, runner):
        """Test handling of unexpected exceptions."""
        mock_orchestrator.return_value.run_tuning.side_effect = Exception("Unexpected error")

        with patch("asyncio.run", side_effect=Exception("Unexpected error")):
            result = runner.invoke(
                tune, ["--function-arn", "arn:aws:lambda:us-east-1:123456789012:function:test"]
            )

            assert result.exit_code == 1
            assert "Unexpected error: Unexpected error" in result.output

    def test_configuration_error_handling(self, runner):
        """Test handling of ConfigurationError."""
        result = runner.invoke(
            tune, ["--function-arn", ""]  # Empty ARN should trigger validation error
        )

        assert result.exit_code == 1
        assert "Function ARN is required" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
