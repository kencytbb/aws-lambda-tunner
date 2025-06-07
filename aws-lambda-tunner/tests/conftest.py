"""
Pytest configuration and fixtures for AWS Lambda Tuner tests.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import boto3
from moto import mock_lambda

from aws_lambda_tuner import TunerConfig
from aws_lambda_tuner.models import MemoryTestResult, PerformanceAnalysis, Recommendation

# Import our test utilities
from tests.utils.mock_aws import MockAWSProvider, patch_boto3_with_mock
from tests.utils.test_data_generators import TestDataGenerator
from tests.utils.fixtures import (
    create_test_config,
    create_test_results,
    create_test_analysis,
    create_test_recommendation,
    create_complete_test_results
)


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return TunerConfig(
        function_arn='arn:aws:lambda:us-east-1:123456789012:function:test-function',
        payload='{"test": "data"}',
        memory_sizes=[256, 512, 1024],
        iterations=5,
        strategy='balanced',
        concurrent_executions=2,
        timeout=60,
        dry_run=True
    )


@pytest.fixture
def sample_results():
    """Provide sample tuning results for testing."""
    return {
        'function_arn': 'arn:aws:lambda:us-east-1:123456789012:function:test-function',
        'test_started': '2024-01-01T00:00:00',
        'test_completed': '2024-01-01T00:10:00',
        'test_duration_seconds': 600,
        'configurations': [
            {
                'memory_mb': 256,
                'executions': [
                    {
                        'memory_mb': 256,
                        'execution_id': 0,
                        'duration': 150.5,
                        'billed_duration': 200,
                        'cold_start': True,
                        'status_code': 200,
                        'timestamp': '2024-01-01T00:01:00'
                    },
                    {
                        'memory_mb': 256,
                        'execution_id': 1,
                        'duration': 120.3,
                        'billed_duration': 200,
                        'cold_start': False,
                        'status_code': 200,
                        'timestamp': '2024-01-01T00:02:00'
                    }
                ],
                'total_executions': 2,
                'successful_executions': 2,
                'failed_executions': 0
            },
            {
                'memory_mb': 512,
                'executions': [
                    {
                        'memory_mb': 512,
                        'execution_id': 0,
                        'duration': 100.2,
                        'billed_duration': 200,
                        'cold_start': True,
                        'status_code': 200,
                        'timestamp': '2024-01-01T00:03:00'
                    },
                    {
                        'memory_mb': 512,
                        'execution_id': 1,
                        'duration': 85.7,
                        'billed_duration': 100,
                        'cold_start': False,
                        'status_code': 200,
                        'timestamp': '2024-01-01T00:04:00'
                    }
                ],
                'total_executions': 2,
                'successful_executions': 2,
                'failed_executions': 0
            }
        ]
    }


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_lambda_client():
    """Provide a mocked Lambda client."""
    with mock_lambda():
        client = boto3.client('lambda', region_name='us-east-1')
        
        # Create a test function
        client.create_function(
            FunctionName='test-function',
            Runtime='python3.9',
            Role='arn:aws:iam::123456789012:role/test-role',
            Handler='index.handler',
            Code={'ZipFile': b'fake code'},
            MemorySize=256,
            Timeout=60
        )
        
        yield client


@pytest.fixture
def mock_boto3_session(monkeypatch):
    """Mock boto3 session creation."""
    mock_session = Mock()
    mock_client = Mock()
    
    # Configure mock client
    mock_client.get_function_configuration.return_value = {
        'MemorySize': 256,
        'Timeout': 60,
        'Runtime': 'python3.9'
    }
    
    mock_client.invoke.return_value = {
        'StatusCode': 200,
        'Payload': Mock(read=lambda: b'{"result": "success"}'),
        'LogResult': {'BilledDuration': 100}
    }
    
    mock_session.client.return_value = mock_client
    
    # Patch boto3.Session
    monkeypatch.setattr('boto3.Session', lambda **kwargs: mock_session)
    
    return mock_client


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample configuration file."""
    config_data = {
        'function_arn': 'arn:aws:lambda:us-east-1:123456789012:function:test-function',
        'payload': '{"test": "data"}',
        'memory_sizes': [256, 512],
        'iterations': 3,
        'strategy': 'cost'
    }
    
    config_path = temp_dir / 'test_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    return config_path


@pytest.fixture
def mock_visualization_backend(monkeypatch):
    """Mock matplotlib to avoid GUI issues in tests."""
    import matplotlib
    monkeypatch.setattr(matplotlib, 'use', lambda backend: None)


# New enhanced fixtures using our test utilities

@pytest.fixture
def mock_aws_provider():
    """Provide a comprehensive mock AWS provider."""
    return MockAWSProvider()


@pytest.fixture
def mock_aws_services(monkeypatch, mock_aws_provider):
    """Mock all AWS services using our provider."""
    patch_boto3_with_mock(monkeypatch, mock_aws_provider)
    return mock_aws_provider


@pytest.fixture
def test_data_generator():
    """Provide a test data generator with fixed seed for reproducible tests."""
    return TestDataGenerator(seed=42)


@pytest.fixture
def enhanced_test_config():
    """Enhanced test configuration fixture."""
    return create_test_config()


@pytest.fixture
def sample_memory_results():
    """Sample memory test results for testing."""
    return {
        256: MemoryTestResult(
            memory_size=256,
            iterations=5,
            avg_duration=180.5,
            p95_duration=200.0,
            p99_duration=220.0,
            avg_cost=0.00000834,
            total_cost=0.00004170,
            cold_starts=1,
            errors=0
        ),
        512: MemoryTestResult(
            memory_size=512,
            iterations=5,
            avg_duration=120.3,
            p95_duration=140.0,
            p99_duration=150.0,
            avg_cost=0.00001125,
            total_cost=0.00005625,
            cold_starts=1,
            errors=0
        ),
        1024: MemoryTestResult(
            memory_size=1024,
            iterations=5,
            avg_duration=95.7,
            p95_duration=110.0,
            p99_duration=115.0,
            avg_cost=0.00001668,
            total_cost=0.00008340,
            cold_starts=1,
            errors=0
        )
    }


@pytest.fixture
def sample_performance_analysis(sample_memory_results):
    """Sample performance analysis for testing."""
    return create_test_analysis(sample_memory_results)


@pytest.fixture
def sample_recommendation():
    """Sample recommendation for testing."""
    return create_test_recommendation()


@pytest.fixture
def complete_tuning_results():
    """Complete tuning results for integration testing."""
    return create_complete_test_results()


@pytest.fixture(params=['cpu_intensive', 'io_bound', 'memory_intensive', 'balanced'])
def workload_type(request):
    """Parametrized fixture for different workload types."""
    return request.param


@pytest.fixture(params=[128, 256, 512, 1024, 2048, 3008])
def memory_size(request):
    """Parametrized fixture for different memory sizes."""
    return request.param


@pytest.fixture(params=['cost', 'speed', 'balanced', 'comprehensive'])
def strategy(request):
    """Parametrized fixture for different optimization strategies."""
    return request.param


@pytest.fixture
def aws_credentials_env(monkeypatch):
    """Set up AWS credentials in environment for testing."""
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
    monkeypatch.setenv('AWS_SECURITY_TOKEN', 'testing')
    monkeypatch.setenv('AWS_SESSION_TOKEN', 'testing')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')


@pytest.fixture
def disable_aws_retry(monkeypatch):
    """Disable AWS retry for faster test execution."""
    import botocore.config
    config = botocore.config.Config(
        retries={'max_attempts': 1},
        connect_timeout=1,
        read_timeout=1
    )
    monkeypatch.setattr('botocore.config.Config', lambda **kwargs: config)
