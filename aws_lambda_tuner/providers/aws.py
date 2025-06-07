"""AWS Lambda provider for function tuning."""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AWSLambdaProvider:
    """AWS Lambda provider for function tuning operations."""
    
    def __init__(self, config):
        self.config = config
        self.lambda_client = boto3.client('lambda', region_name=config.region)
        self.logs_client = boto3.client('logs', region_name=config.region)
        self._function_name = self._extract_function_name(config.function_arn)
        
    def _extract_function_name(self, function_arn: str) -> str:
        """Extract function name from ARN."""
        # ARN format: arn:aws:lambda:region:account:function:function-name
        return function_arn.split(':')[-1]
    
    async def get_function_configuration(self) -> Dict[str, Any]:
        """Get current function configuration."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.get_function,
                {'FunctionName': self._function_name}
            )
            return response['Configuration']
        except ClientError as e:
            logger.error(f"Failed to get function configuration: {e}")
            raise
    
    async def update_function_memory(self, memory_size: int) -> Dict[str, Any]:
        """Update function memory configuration."""
        try:
            logger.info(f"Updating function memory to {memory_size}MB")
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.update_function_configuration,
                {
                    'FunctionName': self._function_name,
                    'MemorySize': memory_size
                }
            )
            
            # Wait for update to complete
            await self._wait_for_function_update()
            
            return response
            
        except ClientError as e:
            logger.error(f"Failed to update function memory: {e}")
            raise
    
    async def _wait_for_function_update(self, max_wait: int = 60):
        """Wait for function update to complete."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                config = await self.get_function_configuration()
                if config['LastUpdateStatus'] == 'Successful':
                    return
                elif config['LastUpdateStatus'] == 'Failed':
                    raise Exception(f"Function update failed: {config.get('LastUpdateStatusReason', 'Unknown')}")
            except Exception as e:
                logger.warning(f"Error checking function status: {e}")
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Function update timed out")
    
    async def invoke_function(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the Lambda function and collect performance metrics."""
        try:
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.lambda_client.invoke,
                {
                    'FunctionName': self._function_name,
                    'Payload': json.dumps(payload),
                    'LogType': 'Tail'
                }
            )
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Parse response
            result = {
                'duration': duration,
                'status_code': response['StatusCode'],
                'timestamp': datetime.utcnow(),
                'request_id': response['ResponseMetadata']['RequestId']
            }
            
            # Check for errors
            if 'FunctionError' in response:
                result['error'] = True
                result['error_type'] = response['FunctionError']
                logger.warning(f"Function error: {response['FunctionError']}")
            else:
                result['error'] = False
            
            # Parse CloudWatch logs if available
            if 'LogResult' in response:
                log_data = self._parse_log_result(response['LogResult'])
                result.update(log_data)
            
            return result
            
        except ClientError as e:
            logger.error(f"Failed to invoke function: {e}")
            raise
    
    def _parse_log_result(self, log_result: str) -> Dict[str, Any]:
        """Parse CloudWatch log result from Lambda response."""
        import base64
        
        try:
            # Decode base64 log data
            log_data = base64.b64decode(log_result).decode('utf-8')
            
            # Parse log lines
            lines = log_data.strip().split('\n')
            
            result = {
                'cold_start': False,
                'billed_duration': None,
                'memory_used': None,
                'max_memory_used': None
            }
            
            for line in lines:
                # Check for cold start indicators
                if 'INIT_START' in line or 'Runtime.ImportModuleError' in line:
                    result['cold_start'] = True
                
                # Parse REPORT line for detailed metrics
                if line.startswith('REPORT'):
                    report_data = self._parse_report_line(line)
                    result.update(report_data)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to parse log result: {e}")
            return {'cold_start': False}
    
    def _parse_report_line(self, report_line: str) -> Dict[str, Any]:
        """Parse Lambda REPORT log line."""
        result = {}
        
        try:
            # Example REPORT line:
            # REPORT RequestId: 12345 Duration: 1234.56 ms Billed Duration: 1300 ms 
            # Memory Size: 512 MB Max Memory Used: 256 MB
            
            parts = report_line.split('\t')
            
            for part in parts:
                part = part.strip()
                
                if part.startswith('Duration:'):
                    # Extract duration in ms
                    duration_str = part.split(':')[1].strip().replace('ms', '').strip()
                    result['actual_duration'] = float(duration_str)
                
                elif part.startswith('Billed Duration:'):
                    # Extract billed duration in ms
                    duration_str = part.split(':')[1].strip().replace('ms', '').strip()
                    result['billed_duration'] = float(duration_str)
                
                elif part.startswith('Memory Size:'):
                    # Extract memory size in MB
                    memory_str = part.split(':')[1].strip().replace('MB', '').strip()
                    result['memory_size'] = int(memory_str)
                
                elif part.startswith('Max Memory Used:'):
                    # Extract max memory used in MB
                    memory_str = part.split(':')[1].strip().replace('MB', '').strip()
                    result['memory_used'] = int(memory_str)
        
        except Exception as e:
            logger.warning(f"Failed to parse REPORT line: {e}")
        
        return result