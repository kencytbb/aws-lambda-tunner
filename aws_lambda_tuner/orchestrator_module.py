"""
Orchestrator module for AWS Lambda Tuner.
Manages the tuning process and coordinates Lambda invocations.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config_module import TunerConfig
from .utils import (
    encode_payload, decode_response, retry_with_backoff,
    format_timestamp, chunk_list
)
from .exceptions import (
    LambdaExecutionError, AWSPermissionError,
    TimeoutError, ConcurrencyLimitError
)

logger = logging.getLogger(__name__)


class TunerOrchestrator:
    """Orchestrates the Lambda tuning process."""
    
    def __init__(self, config: TunerConfig):
        """
        Initialize the orchestrator.
        
        Args:
            config: Tuner configuration
        """
        self.config = config
        self.lambda_client = self._create_lambda_client()
        self.results = {
            'function_arn': config.function_arn,
            'test_started': format_timestamp(),
            'configurations': []
        }
    
    def _create_lambda_client(self):
        """Create AWS Lambda client."""
        try:
            session_config = {}
            if self.config.profile:
                session_config['profile_name'] = self.config.profile
            
            session = boto3.Session(**session_config)
            
            client_config = {'region_name': self.config.region}
            
            return session.client('lambda', **client_config)
            
        except Exception as e:
            logger.error(f"Failed to create Lambda client: {e}")
            raise AWSPermissionError(f"Failed to create Lambda client: {e}")
    
    async def run_tuning(self) -> Dict[str, Any]:
        """
        Run the complete tuning process.
        
        Returns:
            Dictionary containing all test results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting tuning for function: {self.config.function_arn}")
            
            # Get current function configuration
            original_config = await self._get_function_configuration()
            logger.info(f"Original memory configuration: {original_config['MemorySize']}MB")
            
            # Test each memory configuration
            for memory_size in self.config.memory_sizes:
                logger.info(f"Testing memory configuration: {memory_size}MB")
                
                # Update function memory
                if not self.config.dry_run:
                    await self._update_function_memory(memory_size)
                    # Wait for update to propagate
                    await asyncio.sleep(2)
                
                # Run tests for this configuration
                config_results = await self._test_memory_configuration(memory_size)
                self.results['configurations'].append(config_results)
                
                logger.info(f"Completed testing for {memory_size}MB: "
                          f"{config_results['successful_executions']}/{config_results['total_executions']} successful")
            
            # Restore original configuration
            if not self.config.dry_run and original_config['MemorySize'] not in self.config.memory_sizes:
                logger.info(f"Restoring original memory configuration: {original_config['MemorySize']}MB")
                await self._update_function_memory(original_config['MemorySize'])
            
            # Add metadata
            self.results['test_completed'] = format_timestamp()
            self.results['test_duration_seconds'] = time.time() - start_time
            self.results['total_configurations_tested'] = len(self.config.memory_sizes)
            
            logger.info(f"Tuning completed in {self.results['test_duration_seconds']:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Tuning failed: {e}")
            self.results['error'] = str(e)
            self.results['test_duration_seconds'] = time.time() - start_time
            raise
    
    async def _get_function_configuration(self) -> Dict[str, Any]:
        """Get current Lambda function configuration."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.lambda_client.get_function_configuration,
                FunctionName=self.config.function_arn
            )
            return response
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                raise LambdaExecutionError(f"Function not found: {self.config.function_arn}")
            elif e.response['Error']['Code'] == 'AccessDeniedException':
                raise AWSPermissionError(f"Permission denied: {e}")
            else:
                raise LambdaExecutionError(f"Failed to get function configuration: {e}")
    
    async def _update_function_memory(self, memory_size: int):
        """Update Lambda function memory configuration."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.lambda_client.update_function_configuration,
                FunctionName=self.config.function_arn,
                MemorySize=memory_size
            )
            
            # Wait for function to be active
            waiter = self.lambda_client.get_waiter('function_active_v2')
            await asyncio.get_event_loop().run_in_executor(
                None,
                waiter.wait,
                FunctionName=self.config.function_arn
            )
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                raise AWSPermissionError(f"Permission denied to update function: {e}")
            else:
                raise LambdaExecutionError(f"Failed to update function memory: {e}")
    
    async def _test_memory_configuration(self, memory_size: int) -> Dict[str, Any]:
        """Test a specific memory configuration."""
        config_results = {
            'memory_mb': memory_size,
            'executions': [],
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }
        
        # Run warmup executions
        if self.config.warmup_runs > 0 and not self.config.dry_run:
            logger.info(f"Running {self.config.warmup_runs} warmup executions...")
            await self._run_concurrent_invocations(
                memory_size,
                self.config.warmup_runs,
                is_warmup=True
            )
        
        # Run actual test executions
        if self.config.dry_run:
            # Simulate executions for dry run
            config_results['executions'] = self._simulate_executions(memory_size)
        else:
            # Run real executions
            config_results['executions'] = await self._run_concurrent_invocations(
                memory_size,
                self.config.iterations,
                is_warmup=False
            )
        
        # Calculate statistics
        config_results['total_executions'] = len(config_results['executions'])
        config_results['successful_executions'] = sum(
            1 for e in config_results['executions'] if not e.get('error')
        )
        config_results['failed_executions'] = config_results['total_executions'] - config_results['successful_executions']
        
        return config_results
    
    async def _run_concurrent_invocations(self, memory_size: int, count: int, is_warmup: bool) -> List[Dict[str, Any]]:
        """Run Lambda invocations concurrently."""
        executions = []
        
        # Split into chunks based on concurrency limit
        chunks = chunk_list(list(range(count)), self.config.concurrent_executions)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # Run chunk concurrently
            with ThreadPoolExecutor(max_workers=self.config.concurrent_executions) as executor:
                futures = []
                
                for i in chunk:
                    future = executor.submit(
                        self._invoke_lambda_sync,
                        memory_size,
                        execution_id=i,
                        is_warmup=is_warmup
                    )
                    futures.append(future)
                
                # Wait for all invocations in chunk to complete
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout)
                        if not is_warmup:
                            executions.append(result)
                    except Exception as e:
                        logger.error(f"Invocation failed: {e}")
                        if not is_warmup:
                            executions.append({
                                'memory_mb': memory_size,
                                'error': str(e),
                                'timestamp': format_timestamp()
                            })
            
            # Rate limiting between chunks
            chunk_duration = time.time() - chunk_start
            if chunk_idx < len(chunks) - 1 and chunk_duration < 1:
                await asyncio.sleep(1 - chunk_duration)
        
        return executions
    
    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def _invoke_lambda_sync(self, memory_size: int, execution_id: int, is_warmup: bool) -> Dict[str, Any]:
        """Invoke Lambda function synchronously."""
        start_time = time.time()
        
        try:
            # Prepare payload
            payload = encode_payload(self.config.payload)
            
            # Invoke function
            response = self.lambda_client.invoke(
                FunctionName=self.config.function_arn,
                InvocationType='RequestResponse',
                Payload=payload
            )
            
            # Process response
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            result = {
                'memory_mb': memory_size,
                'execution_id': execution_id,
                'duration': duration,
                'billed_duration': response.get('LogResult', {}).get('BilledDuration', duration),
                'cold_start': execution_id == 0,  # Simplified cold start detection
                'status_code': response['StatusCode'],
                'timestamp': format_timestamp()
            }
            
            # Check for function errors
            if response.get('FunctionError'):
                result['error'] = response['FunctionError']
                result['error_message'] = decode_response(response['Payload'])
            
            # Extract execution details from response
            if 'LogResult' in response:
                # Parse CloudWatch logs if available
                # This would require base64 decoding and parsing
                pass
            
            return result
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'TooManyRequestsException':
                raise ConcurrencyLimitError("Lambda concurrency limit exceeded")
            elif error_code == 'ResourceNotFoundException':
                raise LambdaExecutionError(f"Function not found: {self.config.function_arn}")
            elif error_code == 'AccessDeniedException':
                raise AWSPermissionError("Permission denied to invoke function")
            else:
                raise LambdaExecutionError(f"Lambda invocation failed: {e}")
            
        except Exception as e:
            raise LambdaExecutionError(f"Unexpected error during invocation: {e}")
    
    def _simulate_executions(self, memory_size: int) -> List[Dict[str, Any]]:
        """Simulate executions for dry run mode."""
        import random
        
        executions = []
        base_duration = 1000 / (memory_size / 512)  # Simulate faster execution with more memory
        
        for i in range(self.config.iterations):
            # Add some variance
            duration = base_duration * random.uniform(0.8, 1.2)
            billed_duration = max(100, int(duration / 100) * 100)  # Round up to nearest 100ms
            
            execution = {
                'memory_mb': memory_size,
                'execution_id': i,
                'duration': duration,
                'billed_duration': billed_duration,
                'cold_start': i == 0,
                'status_code': 200,
                'timestamp': format_timestamp(),
                'dry_run': True
            }
            
            # Simulate occasional failures
            if random.random() < 0.05:  # 5% failure rate
                execution['error'] = 'Simulated error'
                execution['status_code'] = 500
            
            executions.append(execution)
        
        return executions
    
    async def run_with_reporting(self) -> Dict[str, Any]:
        """Run tuning with automatic report generation."""
        from .report_service import ReportGenerator
        from .visualization_module import VisualizationEngine
        
        # Run tuning
        results = await self.run_tuning()
        
        # Generate reports
        logger.info("Generating reports...")
        report_gen = ReportGenerator(results, self.config)
        
        # Save reports
        output_dir = self.config.output_dir
        report_gen.save_json(f"{output_dir}/results.json")
        report_gen.save_html(f"{output_dir}/report.html")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        viz_engine = VisualizationEngine(results)
        viz_engine.plot_performance_comparison(f"{output_dir}/performance.png")
        viz_engine.plot_cost_analysis(f"{output_dir}/cost-analysis.png")
        
        return results


# Convenience functions
async def run_tuning_session(config: TunerConfig) -> Dict[str, Any]:
    """Run a complete tuning session."""
    orchestrator = TunerOrchestrator(config)
    return await orchestrator.run_tuning()


async def test_single_configuration(
    function_arn: str,
    memory_size: int,
    payload: str = '{}',
    iterations: int = 10
) -> Dict[str, Any]:
    """Test a single memory configuration."""
    config = TunerConfig(
        function_arn=function_arn,
        memory_sizes=[memory_size],
        payload=payload,
        iterations=iterations
    )
    
    orchestrator = TunerOrchestrator(config)
    return await orchestrator.run_tuning()
