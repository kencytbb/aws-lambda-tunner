"""Unit tests for configuration modules."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from aws_lambda_tuner.config_module import TunerConfig
from aws_lambda_tuner.exceptions import ConfigurationError
from tests.utils.test_helpers import TestValidators, TestAssertions, TestHelpers


@pytest.mark.unit
class TestTunerConfig:
    """Test TunerConfig class."""

    def test_config_creation_with_defaults(self):
        """Test creating config with minimal required parameters."""
        config = TunerConfig(function_arn="arn:aws:lambda:us-east-1:123456789012:function:test")

        assert config.function_arn == "arn:aws:lambda:us-east-1:123456789012:function:test"
        assert config.memory_sizes == [256, 512, 1024, 1536, 2048]  # Default values
        assert config.iterations == 10
        assert config.strategy == "balanced"
        assert config.concurrent_executions == 1
        assert config.timeout == 300
        assert config.dry_run is False

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom parameters."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function",
            memory_sizes=[128, 256, 512],
            iterations=5,
            strategy="cost",
            concurrent_executions=3,
            timeout=120,
            dry_run=True,
            payload='{"custom": "data"}',
            warmup_iterations=2,
        )

        assert config.function_arn == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        assert config.memory_sizes == [128, 256, 512]
        assert config.iterations == 5
        assert config.strategy == "cost"
        assert config.concurrent_executions == 3
        assert config.timeout == 120
        assert config.dry_run is True
        assert config.payload == '{"custom": "data"}'
        assert config.warmup_iterations == 2

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "memory_sizes": [256, 512, 1024],
            "iterations": 15,
            "strategy": "speed",
            "payload": '{"test": true}',
            "concurrent_executions": 2,
            "timeout": 600,
            "dry_run": False,
            "warmup_iterations": 1,
            "output_format": "json",
            "save_results": True,
            "results_file": "results.json",
        }

        config = TunerConfig.from_dict(config_dict)

        assert config.function_arn == config_dict["function_arn"]
        assert config.memory_sizes == config_dict["memory_sizes"]
        assert config.iterations == config_dict["iterations"]
        assert config.strategy == config_dict["strategy"]
        assert config.payload == config_dict["payload"]
        assert config.output_format == config_dict["output_format"]
        assert config.save_results == config_dict["save_results"]
        assert config.results_file == config_dict["results_file"]

    def test_config_from_file(self, temp_dir):
        """Test loading config from JSON file."""
        config_data = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "memory_sizes": [512, 1024],
            "iterations": 8,
            "strategy": "balanced",
        }

        config_file = temp_dir / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = TunerConfig.from_file(str(config_file))

        assert config.function_arn == config_data["function_arn"]
        assert config.memory_sizes == config_data["memory_sizes"]
        assert config.iterations == config_data["iterations"]
        assert config.strategy == config_data["strategy"]

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512],
            iterations=5,
            strategy="cost",
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["function_arn"] == config.function_arn
        assert config_dict["memory_sizes"] == config.memory_sizes
        assert config_dict["iterations"] == config.iterations
        assert config_dict["strategy"] == config.strategy

    def test_config_save_to_file(self, temp_dir):
        """Test saving config to file."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512, 1024],
            iterations=10,
        )

        config_file = temp_dir / "saved_config.json"
        config.save_to_file(str(config_file))

        assert config_file.exists()

        # Verify the saved content
        with open(config_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["function_arn"] == config.function_arn
        assert saved_data["memory_sizes"] == config.memory_sizes
        assert saved_data["iterations"] == config.iterations


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_function_arn(self):
        """Test validation with invalid function ARN."""
        with pytest.raises(ConfigurationError):
            TunerConfig(function_arn="invalid-arn")

    def test_empty_function_arn(self):
        """Test validation with empty function ARN."""
        with pytest.raises(ConfigurationError):
            TunerConfig(function_arn="")

    def test_invalid_memory_sizes(self):
        """Test validation with invalid memory sizes."""
        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                memory_sizes=[],  # Empty list
            )

        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                memory_sizes=[0, 256],  # Invalid memory size
            )

        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                memory_sizes=[4000],  # Above AWS limit
            )

    def test_invalid_strategy(self):
        """Test validation with invalid strategy."""
        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                strategy="invalid_strategy",
            )

    def test_invalid_iterations(self):
        """Test validation with invalid iteration count."""
        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test", iterations=0
            )

        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test", iterations=-1
            )

    def test_invalid_concurrent_executions(self):
        """Test validation with invalid concurrent execution count."""
        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                concurrent_executions=0,
            )

        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                concurrent_executions=1001,  # AWS Lambda limit is 1000
            )

    def test_invalid_timeout(self):
        """Test validation with invalid timeout."""
        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test", timeout=0
            )

        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                timeout=901,  # AWS Lambda max is 900 seconds
            )

    def test_valid_memory_sizes(self):
        """Test validation with valid memory sizes."""
        # Test AWS Lambda valid memory sizes
        valid_memory_sizes = [128, 256, 512, 1024, 1536, 2048, 3008]

        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=valid_memory_sizes,
        )

        assert config.memory_sizes == valid_memory_sizes

    def test_payload_validation(self):
        """Test payload validation."""
        # Valid JSON payload
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            payload='{"valid": "json"}',
        )
        assert config.payload == '{"valid": "json"}'

        # Invalid JSON should raise error
        with pytest.raises(ConfigurationError):
            TunerConfig(
                function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                payload="invalid json",
            )


@pytest.mark.unit
class TestConfigFileHandling:
    """Test configuration file handling."""

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            TunerConfig.from_file("nonexistent_file.json")

    def test_load_invalid_json_file(self, temp_dir):
        """Test loading from file with invalid JSON."""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            TunerConfig.from_file(str(invalid_file))

    def test_load_file_missing_required_fields(self, temp_dir):
        """Test loading from file missing required fields."""
        incomplete_config = {"memory_sizes": [256, 512]}  # Missing function_arn

        config_file = temp_dir / "incomplete.json"
        with open(config_file, "w") as f:
            json.dump(incomplete_config, f)

        with pytest.raises(ConfigurationError):
            TunerConfig.from_file(str(config_file))

    def test_save_to_readonly_location(self, temp_dir):
        """Test saving to read-only location."""
        config = TunerConfig(function_arn="arn:aws:lambda:us-east-1:123456789012:function:test")

        # Create a read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        readonly_file = readonly_dir / "config.json"

        with pytest.raises(PermissionError):
            config.save_to_file(str(readonly_file))

    def test_load_file_with_extra_fields(self, temp_dir):
        """Test loading file with extra unknown fields."""
        config_data = {
            "function_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "memory_sizes": [256, 512],
            "iterations": 5,
            "unknown_field": "should_be_ignored",
        }

        config_file = temp_dir / "extra_fields.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = TunerConfig.from_file(str(config_file))

        # Should load successfully, ignoring unknown fields
        assert config.function_arn == config_data["function_arn"]
        assert config.memory_sizes == config_data["memory_sizes"]
        assert config.iterations == config_data["iterations"]


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["cost", "speed", "balanced", "comprehensive"])
def test_valid_strategies(strategy):
    """Test all valid optimization strategies."""
    config = TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:test", strategy=strategy
    )

    assert config.strategy == strategy


@pytest.mark.unit
@pytest.mark.parametrize("memory_size", [128, 256, 320, 512, 1024, 1536, 2048, 3008])
def test_valid_individual_memory_sizes(memory_size):
    """Test individual valid memory sizes."""
    config = TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
        memory_sizes=[memory_size],
    )

    assert memory_size in config.memory_sizes


@pytest.mark.unit
@pytest.mark.parametrize("invalid_memory", [0, -1, 127, 3009, 4096])
def test_invalid_individual_memory_sizes(invalid_memory):
    """Test individual invalid memory sizes."""
    with pytest.raises(ConfigurationError):
        TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[invalid_memory],
        )


@pytest.mark.unit
class TestConfigUtilities:
    """Test configuration utility methods."""

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512],
            iterations=5,
        )

        config2 = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512],
            iterations=5,
        )

        config3 = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512],
            iterations=10,  # Different
        )

        assert config1 == config2
        assert config1 != config3

    def test_config_string_representation(self):
        """Test config string representation."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512],
            strategy="cost",
        )

        config_str = str(config)
        assert "test" in config_str
        assert "cost" in config_str
        assert "256" in config_str

    def test_config_repr(self):
        """Test config repr."""
        config = TunerConfig(function_arn="arn:aws:lambda:us-east-1:123456789012:function:test")

        config_repr = repr(config)
        assert "TunerConfig" in config_repr
        assert "test" in config_repr

    def test_config_copy(self):
        """Test config copying."""
        original = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            memory_sizes=[256, 512],
            iterations=5,
        )

        copied = original.copy()

        assert copied == original
        assert copied is not original
        assert copied.memory_sizes is not original.memory_sizes  # Deep copy of lists

    def test_config_update(self):
        """Test config updating."""
        config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test", iterations=5
        )

        config.update({"iterations": 10, "strategy": "speed", "dry_run": True})

        assert config.iterations == 10
        assert config.strategy == "speed"
        assert config.dry_run is True

    def test_config_merge(self):
        """Test config merging."""
        base_config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            iterations=5,
            strategy="balanced",
        )

        override_config = TunerConfig(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            iterations=10,
            dry_run=True,
        )

        merged = base_config.merge(override_config)

        assert merged.iterations == 10  # Overridden
        assert merged.strategy == "balanced"  # From base
        assert merged.dry_run is True  # From override


@pytest.mark.unit
class TestConfigEnvironmentVariables:
    """Test configuration from environment variables."""

    @patch.dict(
        "os.environ",
        {
            "AWS_LAMBDA_TUNER_FUNCTION_ARN": "arn:aws:lambda:us-east-1:123456789012:function:env-test",
            "AWS_LAMBDA_TUNER_STRATEGY": "cost",
            "AWS_LAMBDA_TUNER_ITERATIONS": "15",
        },
    )
    def test_config_from_environment(self):
        """Test loading config from environment variables."""
        config = TunerConfig.from_environment()

        assert config.function_arn == "arn:aws:lambda:us-east-1:123456789012:function:env-test"
        assert config.strategy == "cost"
        assert config.iterations == 15

    @patch.dict("os.environ", {}, clear=True)
    def test_config_from_environment_missing_required(self):
        """Test loading config from environment without required variables."""
        with pytest.raises(ConfigurationError):
            TunerConfig.from_environment()

    @patch.dict(
        "os.environ",
        {
            "AWS_LAMBDA_TUNER_FUNCTION_ARN": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "AWS_LAMBDA_TUNER_MEMORY_SIZES": "256,512,1024",
        },
    )
    def test_config_from_environment_list_parsing(self):
        """Test parsing list values from environment."""
        config = TunerConfig.from_environment()

        assert config.memory_sizes == [256, 512, 1024]
