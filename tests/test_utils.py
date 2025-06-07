"""
Tests for utility functions.
"""

import pytest
import json
import tempfile
from pathlib import Path
from io import BytesIO

from aws_lambda_tuner.utils import (
    validate_arn,
    encode_payload,
    decode_response,
    calculate_statistics,
    format_duration,
    calculate_cost,
    retry_with_backoff,
    load_json_file,
    save_json_file,
    format_timestamp,
    safe_divide,
    chunk_list,
    get_memory_sizes
)


class TestValidateArn:
    """Test ARN validation."""
    
    def test_valid_arn(self):
        """Test valid ARN formats."""
        valid_arns = [
            'arn:aws:lambda:us-east-1:123456789012:function:my-function',
            'arn:aws:lambda:eu-west-1:999999999999:function:test-func',
            'arn:aws:lambda:ap-southeast-2:111111111111:function:func-name-123'
        ]
        
        for arn in valid_arns:
            assert validate_arn(arn) is True
    
    def test_invalid_arn(self):
        """Test invalid ARN formats."""
        invalid_arns = [
            '',
            None,
            'not-an-arn',
            'arn:aws:s3:::my-bucket',  # S3 ARN, not Lambda
            'arn:aws:lambda:us-east-1:123456789012',  # Incomplete
            'arn:aws:lambda:us-east-1:not-a-number:function:my-func'
        ]
        
        for arn in invalid_arns:
            assert validate_arn(arn) is False


class TestPayloadHandling:
    """Test payload encoding and decoding."""
    
    def test_encode_payload_dict(self):
        """Test encoding dictionary payload."""
        payload = {'test': 'data', 'number': 123}
        encoded = encode_payload(payload)
        assert json.loads(encoded) == payload
    
    def test_encode_payload_string(self):
        """Test encoding string payload."""
        payload = '{"test": "data"}'
        encoded = encode_payload(payload)
        assert encoded == payload
    
    def test_encode_payload_invalid(self):
        """Test encoding invalid payload."""
        with pytest.raises(ValueError):
            encode_payload('not valid json')
        
        with pytest.raises(TypeError):
            encode_payload(123)
    
    def test_decode_response(self):
        """Test decoding Lambda response."""
        data = {'result': 'success'}
        response = BytesIO(json.dumps(data).encode('utf-8'))
        
        decoded = decode_response(response)
        assert decoded == data


class TestStatistics:
    """Test statistical calculations."""
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        values = [100, 200, 300, 400, 500]
        stats = calculate_statistics(values)
        
        assert stats['min'] == 100
        assert stats['max'] == 500
        assert stats['mean'] == 300
        assert stats['median'] == 300
        assert stats['p95'] == 500
        assert stats['p99'] == 500
    
    def test_calculate_statistics_empty(self):
        """Test statistics with empty list."""
        stats = calculate_statistics([])
        
        assert stats['min'] == 0
        assert stats['max'] == 0
        assert stats['mean'] == 0


class TestFormatting:
    """Test formatting functions."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(500) == "500.00ms"
        assert format_duration(1500) == "1.50s"
        assert format_duration(65000) == "1m 5.00s"
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        timestamp = format_timestamp()
        assert isinstance(timestamp, str)
        assert 'T' in timestamp  # ISO format


class TestCostCalculation:
    """Test cost calculation."""
    
    def test_calculate_cost(self):
        """Test Lambda cost calculation."""
        # 1GB for 1 second
        cost = calculate_cost(1024, 1000, 1)
        expected_compute = (1024 / 1024) * (1000 / 1000) * 0.0000166667
        expected_request = (1 / 1_000_000) * 0.20
        assert cost == pytest.approx(expected_compute + expected_request)


class TestRetryDecorator:
    """Test retry decorator."""
    
    def test_retry_success(self):
        """Test successful function after retries."""
        call_count = 0
        
        @retry_with_backoff(retries=3, backoff_in_seconds=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_failure(self):
        """Test function that fails all retries."""
        @retry_with_backoff(retries=2, backoff_in_seconds=0.1)
        def always_fails():
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError):
            always_fails()


class TestFileOperations:
    """Test file operations."""
    
    def test_load_save_json(self, temp_dir):
        """Test loading and saving JSON files."""
        data = {'test': 'data', 'number': 123}
        filepath = temp_dir / 'test.json'
        
        # Save
        save_json_file(data, str(filepath))
        assert filepath.exists()
        
        # Load
        loaded = load_json_file(str(filepath))
        assert loaded == data
    
    def test_load_json_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_json_file('/non/existent/file.json')


class TestUtilityFunctions:
    """Test other utility functions."""
    
    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, default=999) == 999
    
    def test_chunk_list(self):
        """Test list chunking."""
        items = list(range(10))
        chunks = chunk_list(items, 3)
        
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]
    
    def test_get_memory_sizes(self):
        """Test memory size strategies."""
        speed_sizes = get_memory_sizes('speed')
        assert 2048 in speed_sizes
        assert 3008 in speed_sizes
        
        cost_sizes = get_memory_sizes('cost')
        assert 128 in cost_sizes
        assert max(cost_sizes) <= 1024
        
        balanced_sizes = get_memory_sizes('balanced')
        assert len(balanced_sizes) > 0
        
        # Unknown strategy should return balanced
        default_sizes = get_memory_sizes('unknown')
        assert default_sizes == balanced_sizes
