"""
Integration tests for complete CLI workflows and configuration precedence.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from aws_lambda_tuner.cli_module import cli, init, tune
from aws_lambda_tuner.config_module import TunerConfig


class TestConfigurationPrecedence:
    """Test configuration loading and CLI argument precedence."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:config-function",
            "payload": '{"config": "file"}',
            "memory_sizes": [256, 512],
            "iterations": 8,
            "strategy": "cost",
            "concurrent_executions": 3,
            "timeout": 300,
            "dry_run": False,
            "output_dir": "./config-results",
        }

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.ReportGenerator")
    @patch("asyncio.run")
    def test_cli_args_override_config_file(
        self, mock_asyncio, mock_report_gen, mock_orchestrator, mock_validate, runner, base_config
    ):
        """Test that CLI arguments override config file values."""
        # Mock the orchestrator and report generator
        mock_orchestrator.return_value.run_tuning = Mock(return_value={})
        mock_asyncio.return_value = {}
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
            # Create config file
            with open("test.config.json", "w") as f:
                json.dump(base_config, f)

            # Run tune command with CLI args that should override config
            result = runner.invoke(
                tune,
                [
                    "--config",
                    "test.config.json",
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:override-function",
                    "--memory-sizes",
                    "1024,1536,2048",
                    "--iterations",
                    "15",
                    "--strategy",
                    "speed",
                    "--concurrent",
                    "10",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0

            # Verify that CLI arguments took precedence
            config_call = mock_orchestrator.call_args[0][0]
            assert (
                config_call.function_arn
                == "arn:aws:lambda:us-east-1:123456789012:function:override-function"
            )
            assert config_call.memory_sizes == [1024, 1536, 2048]
            assert config_call.iterations == 15
            assert config_call.strategy == "speed"
            assert config_call.concurrent_executions == 10
            assert config_call.dry_run is True

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    @patch("aws_lambda_tuner.cli_module.TunerOrchestrator")
    @patch("aws_lambda_tuner.cli_module.ReportGenerator")
    @patch("asyncio.run")
    def test_config_file_used_when_no_cli_override(
        self, mock_asyncio, mock_report_gen, mock_orchestrator, mock_validate, runner, base_config
    ):
        """Test that config file values are used when no CLI override."""
        # Mock the orchestrator and report generator
        mock_orchestrator.return_value.run_tuning = Mock(return_value={})
        mock_asyncio.return_value = {}
        mock_report_instance = Mock()
        mock_report_gen.return_value = mock_report_instance
        mock_report_instance.get_summary.return_value = {
            "optimal_memory": 256,
            "optimal_duration": 400.0,
            "optimal_cost": 0.000001,
            "performance_gain": 30.0,
            "cost_savings": 15.0,
        }

        with runner.isolated_filesystem():
            # Create config file
            with open("test.config.json", "w") as f:
                json.dump(base_config, f)

            # Run tune command with only config file
            result = runner.invoke(tune, ["--config", "test.config.json"])

            assert result.exit_code == 0

            # Verify that config file values were used
            config_call = mock_orchestrator.call_args[0][0]
            assert (
                config_call.function_arn
                == "arn:aws:lambda:us-east-1:123456789012:function:config-function"
            )
            assert config_call.memory_sizes == [256, 512]
            assert config_call.iterations == 8
            assert config_call.strategy == "cost"
            assert config_call.concurrent_executions == 3
            assert config_call.dry_run is False

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    def test_partial_cli_override(self, mock_validate, runner, base_config):
        """Test partial CLI argument override of config file."""
        with patch("aws_lambda_tuner.cli_module.TunerOrchestrator") as mock_orchestrator, patch(
            "aws_lambda_tuner.cli_module.ReportGenerator"
        ), patch("asyncio.run"):

            mock_orchestrator.return_value.run_tuning = Mock(return_value={})

            with runner.isolated_filesystem():
                # Create config file
                with open("test.config.json", "w") as f:
                    json.dump(base_config, f)

                # Run tune command with only some CLI overrides
                result = runner.invoke(
                    tune,
                    [
                        "--config",
                        "test.config.json",
                        "--iterations",
                        "12",  # Override only iterations
                        "--verbose",  # Add verbose flag
                    ],
                )

                assert result.exit_code == 0

                # Verify mixed configuration
                config_call = mock_orchestrator.call_args[0][0]
                # Config file values should be preserved
                assert (
                    config_call.function_arn
                    == "arn:aws:lambda:us-east-1:123456789012:function:config-function"
                )
                assert config_call.memory_sizes == [256, 512]
                assert config_call.strategy == "cost"
                # CLI override should take effect
                assert config_call.iterations == 12

    def test_invalid_config_file_handling(self, runner):
        """Test handling of invalid configuration files."""
        with runner.isolated_filesystem():
            # Create invalid JSON file
            with open("invalid.config.json", "w") as f:
                f.write("invalid json content")

            result = runner.invoke(tune, ["--config", "invalid.config.json"])

            assert result.exit_code == 1

    def test_missing_config_file_handling(self, runner):
        """Test handling of missing configuration files."""
        result = runner.invoke(tune, ["--config", "nonexistent.config.json"])

        assert result.exit_code != 0

    @patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True)
    def test_payload_file_vs_direct_payload_precedence(self, mock_validate, runner):
        """Test precedence between payload file and direct payload."""
        with patch("aws_lambda_tuner.cli_module.TunerOrchestrator") as mock_orchestrator, patch(
            "aws_lambda_tuner.cli_module.ReportGenerator"
        ), patch("asyncio.run"):

            mock_orchestrator.return_value.run_tuning = Mock(return_value={})

            with runner.isolated_filesystem():
                # Create payload file
                with open("payload.json", "w") as f:
                    json.dump({"from": "file"}, f)

                # Run with both payload file and direct payload (payload file should win)
                result = runner.invoke(
                    tune,
                    [
                        "--function-arn",
                        "arn:aws:lambda:us-east-1:123456789012:function:test",
                        "--payload-file",
                        "payload.json",
                        "--payload",
                        '{"from": "direct"}',
                    ],
                )

                assert result.exit_code == 0

                # Verify payload file took precedence
                config_call = mock_orchestrator.call_args[0][0]
                assert '{"from": "file"}' in config_call.payload


class TestCompleteWorkflows:
    """Test complete end-to-end CLI workflows."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_init_to_tune_workflow(self, runner):
        """Test complete workflow from init to tune."""
        with runner.isolated_filesystem():
            # Step 1: Initialize configuration
            init_result = runner.invoke(init, ["--template", "balanced"])
            assert init_result.exit_code == 0
            assert os.path.exists("tuner.config.json")

            # Step 2: Modify config for testing
            with open("tuner.config.json", "r") as f:
                config = json.load(f)

            config["function_arn"] = "arn:aws:lambda:us-east-1:123456789012:function:workflow-test"
            config["dry_run"] = True  # Use dry run for testing

            with open("tuner.config.json", "w") as f:
                json.dump(config, f)

            # Step 3: Run tuning
            with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True), patch(
                "aws_lambda_tuner.cli_module.TunerOrchestrator"
            ) as mock_orchestrator, patch(
                "aws_lambda_tuner.cli_module.ReportGenerator"
            ) as mock_report_gen, patch(
                "asyncio.run"
            ):

                mock_orchestrator.return_value.run_tuning = Mock(return_value={})
                mock_report_instance = Mock()
                mock_report_gen.return_value = mock_report_instance
                mock_report_instance.get_summary.return_value = {
                    "optimal_memory": 512,
                    "optimal_duration": 300.0,
                    "optimal_cost": 0.000001,
                    "performance_gain": 40.0,
                    "cost_savings": 20.0,
                }

                tune_result = runner.invoke(tune, ["--config", "tuner.config.json"])

                assert tune_result.exit_code == 0
                assert "Starting Lambda performance tuning" in tune_result.output
                assert "TUNING SUMMARY" in tune_result.output

    @patch("aws_lambda_tuner.cli_module.VisualizationEngine")
    def test_tune_to_visualize_workflow(self, mock_viz_engine, runner):
        """Test workflow from tune to visualize."""
        # Sample results that would be generated by tune command
        sample_results = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "configurations": [
                {
                    "memory_mb": 256,
                    "executions": [{"duration": 500, "billed_duration": 500, "cold_start": False}],
                },
                {
                    "memory_mb": 512,
                    "executions": [{"duration": 300, "billed_duration": 300, "cold_start": False}],
                },
            ],
        }

        mock_viz_instance = Mock()
        mock_viz_engine.return_value = mock_viz_instance

        with runner.isolated_filesystem():
            # Step 1: Create results file (simulating tune command output)
            os.makedirs("tuning-results", exist_ok=True)
            with open("tuning-results/tuning-results.json", "w") as f:
                json.dump(sample_results, f)

            # Step 2: Generate visualizations
            viz_result = runner.invoke(
                visualize,
                [
                    "tuning-results/tuning-results.json",
                    "--charts",
                    "all",
                    "--output-dir",
                    "./charts",
                ],
            )

            assert viz_result.exit_code == 0
            assert "Generated: performance-comparison.png" in viz_result.output
            assert "Generated: cost-analysis.png" in viz_result.output

    def test_tune_to_report_workflow(self, runner):
        """Test workflow from tune to report generation."""
        sample_results = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "configurations": [
                {"memory_mb": 256, "avg_duration": 500, "avg_cost": 0.000001},
                {"memory_mb": 512, "avg_duration": 300, "avg_cost": 0.000002},
            ],
        }

        with runner.isolated_filesystem():
            # Step 1: Create results file
            with open("results.json", "w") as f:
                json.dump(sample_results, f)

            # Step 2: Generate summary report
            with patch("aws_lambda_tuner.cli_module.ReportGenerator") as mock_report_gen:
                mock_report_instance = Mock()
                mock_report_gen.return_value = mock_report_instance
                mock_report_instance.get_summary.return_value = {
                    "optimal_memory": 512,
                    "performance_gain": 40.0,
                    "cost_savings": 20.0,
                }

                report_result = runner.invoke(report, ["results.json", "--format", "summary"])

                assert report_result.exit_code == 0
                assert "PERFORMANCE SUMMARY" in report_result.output
                assert "Optimal Memory:" in report_result.output

    def test_multiple_output_formats_workflow(self, runner):
        """Test generating multiple output formats from same results."""
        sample_results = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "configurations": [],
        }

        with runner.isolated_filesystem():
            # Create results file
            with open("results.json", "w") as f:
                json.dump(sample_results, f)

            # Test different report formats
            with patch("aws_lambda_tuner.cli_module.ReportGenerator"):
                # Summary format
                summary_result = runner.invoke(report, ["results.json", "--format", "summary"])
                assert summary_result.exit_code == 0

                # JSON format
                json_result = runner.invoke(report, ["results.json", "--format", "json"])
                assert json_result.exit_code == 0

                # Detailed format
                detailed_result = runner.invoke(report, ["results.json", "--format", "detailed"])
                assert detailed_result.exit_code == 0


class TestCLIIntegrationErrorScenarios:
    """Test error scenarios in complete workflows."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_invalid_arn_in_workflow(self, runner):
        """Test workflow with invalid ARN."""
        with runner.isolated_filesystem():
            # Create config with invalid ARN
            config = {
                "function_arn": "invalid-arn-format",
                "memory_sizes": [256, 512],
                "iterations": 5,
            }

            with open("invalid.config.json", "w") as f:
                json.dump(config, f)

            # Try to run tune - should fail validation
            with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=False):
                result = runner.invoke(tune, ["--config", "invalid.config.json"])

                assert result.exit_code == 1
                assert "Invalid Lambda ARN" in result.output

    def test_missing_required_config_fields(self, runner):
        """Test workflow with missing required configuration fields."""
        with runner.isolated_filesystem():
            # Create incomplete config
            incomplete_config = {
                "memory_sizes": [256, 512]
                # Missing function_arn
            }

            with open("incomplete.config.json", "w") as f:
                json.dump(incomplete_config, f)

            result = runner.invoke(tune, ["--config", "incomplete.config.json"])

            assert result.exit_code == 1

    def test_output_directory_creation_failure(self, runner):
        """Test handling of output directory creation failure."""
        with patch("aws_lambda_tuner.cli_module.validate_arn", return_value=True), patch(
            "aws_lambda_tuner.cli_module.TunerOrchestrator"
        ), patch("aws_lambda_tuner.cli_module.ReportGenerator"), patch("asyncio.run"), patch(
            "pathlib.Path.mkdir", side_effect=PermissionError("Cannot create directory")
        ):

            result = runner.invoke(
                tune,
                [
                    "--function-arn",
                    "arn:aws:lambda:us-east-1:123456789012:function:test",
                    "--output-dir",
                    "/invalid/path",
                ],
            )

            assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__])
