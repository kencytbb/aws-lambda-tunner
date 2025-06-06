"""
Utility functions for the AWS Lambda Tuner package.
"""

import json
import time
import statistics
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import base64
import logging
from functools import wraps
import os

logger = logging.getLogger(__name__)


def validate_arn(arn: str) -> bool:
    """
    Validate AWS ARN format.
    
    Args:
        arn: AWS ARN string
        
    Returns:
        bool: True if valid ARN format
    """
    if not arn or not isinstance(arn, str):
        return False
    
    parts = arn.split(':')
    if len(parts) < 6:
        return False
        
    return parts[0] == 'arn' and parts[1] == 'aws' and parts[2] == 'lambda'


def encode_payload(payload: Union[str, dict]) -> str:
    """
    Encode payload for Lambda invocation.
    
    Args:
        payload: Payload as string or dict
        
    Returns:
        str: JSON encoded payload
    """
    if isinstance(payload, dict):
        return json.dumps(payload)
    elif isinstance(payload, str):
        # Validate it's valid JSON
        try:
            json.loads(payload)
            return payload
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON payload")
    else:
        raise TypeError("Payload must be a string or dictionary")


def decode_response(response_payload: bytes) -> Dict[str, Any]:
    """
    Decode Lambda response payload.
    
    Args:
        response_payload: Response payload bytes
        
    Returns:
        dict: Decoded response
    """
    try:
        payload_str = response_payload.read()
        if isinstance(payload_str, bytes):
            payload_str = payload_str.decode('utf-8')
        return json.loads(payload_str)
    except Exception as e:
        logger.error(f"Failed to decode response: {e}")
        return {"error": str(e), "raw": str(payload_str)}


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        dict: Statistical metrics
    """
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "stddev": 0,
            "p95": 0,
            "p99": 0
        }
    
    sorted_values = sorted(values)
    
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0,
        "p95": sorted_values[int(len(sorted_values) * 0.95)],
        "p99": sorted_values[int(len(sorted_values) * 0.99)]
    }


def format_duration(milliseconds: float) -> str:
    """
    Format duration in milliseconds to human-readable string.
    
    Args:
        milliseconds: Duration in milliseconds
        
    Returns:
        str: Formatted duration
    """
    if milliseconds < 1000:
        return f"{milliseconds:.2f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.2f}s"
    else:
        minutes = int(milliseconds / 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.2f}s"


def calculate_cost(memory_mb: int, duration_ms: float, requests: int = 1) -> float:
    """
    Calculate AWS Lambda cost.
    
    Args:
        memory_mb: Memory allocation in MB
        duration_ms: Duration in milliseconds
        requests: Number of requests
        
    Returns:
        float: Estimated cost in USD
    """
    # AWS Lambda pricing (as of 2024)
    # $0.0000166667 per GB-second
    # $0.20 per 1M requests
    
    gb_seconds = (memory_mb / 1024) * (duration_ms / 1000)
    compute_cost = gb_seconds * 0.0000166667
    request_cost = (requests / 1_000_000) * 0.20
    
    return compute_cost + request_cost


def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        retries: Number of retry attempts
        backoff_in_seconds: Initial backoff time
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = backoff_in_seconds
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise
                    logger.warning(f"Attempt {i+1} failed: {e}. Retrying in {x}s...")
                    time.sleep(x)
                    x *= 2
            return func(*args, **kwargs)
        return wrapper
    return decorator


def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file safely.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        dict: Parsed JSON content
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def save_json_file(data: Dict[str, Any], filepath: str, pretty: bool = True):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        pretty: Whether to format JSON
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2, default=str)
        else:
            json.dump(data, f, default=str)


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp to ISO format.
    
    Args:
        timestamp: Datetime object (default: current time)
        
    Returns:
        str: ISO formatted timestamp
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.isoformat()


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division by zero
        
    Returns:
        float: Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_memory_sizes(strategy: str = 'balanced') -> List[int]:
    """
    Get recommended memory sizes based on strategy.
    
    Args:
        strategy: Tuning strategy
        
    Returns:
        List of memory sizes in MB
    """
    strategies = {
        'speed': [512, 1024, 1536, 2048, 3008],
        'cost': [128, 256, 512, 1024],
        'balanced': [256, 512, 1024, 1536, 2048],
        'comprehensive': [128, 256, 512, 768, 1024, 1536, 2048, 2560, 3008]
    }
    
    return strategies.get(strategy, strategies['balanced'])
