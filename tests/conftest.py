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
