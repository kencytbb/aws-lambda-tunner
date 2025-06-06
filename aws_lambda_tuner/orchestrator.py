"""
AWS Lambda Tuner Orchestrator.

Main orchestration logic for running performance tests across different
memory configurations and analyzing the results.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .config import TunerConfig
from .providers.aws import AWSLambdaProvider
from .analyzers.analyzer import PerformanceAnalyzer
from .models import TuningResult, TestRun
from .reporting.service import ReportingService

logger = logging.getLogger(__name__)


class TunerOrchestrator:
    """Orchestrates the Lambda function tuning process."""

    def __init__(self, config: TunerConfig):
        """Initialize the orchestrator with configuration.
        
        Args:
            config: Tuner configuration object
        """
        self.config = config
        self.aws_provider = AWSLambdaProvider(config)
        self.analyzer = PerformanceAnalyzer()
        self.reporting = ReportingService()
        self.results: List[TuningResult] = []

    async def run(self) -> List[TuningResult]:
        """Run the tuning process for all memory configurations.
        
        Returns:
            List of tuning results for each memory configuration
        """
        logger.info(f"Starting tuning for function: {self.config.function_arn}")
        logger.info(f"Memory sizes to test: {self.config.memory_sizes}")
        logger.info(f"Iterations per size: {self.config.iterations}")
        
        # Store original memory configuration
        original_config = await self.aws_provider.get_function_configuration()
        original_memory = original_config.get('MemorySize', 512)
        
        try:
            # Test each memory configuration
            for memory_size in self.config.memory_sizes:
                logger.info(f"Testing memory size: {memory_size}MB")
                result = await self._test_memory_configuration(memory_size)
                self.results.append(result)
                
                # Add delay between configurations to avoid throttling
                if memory_size != self.config.memory_sizes[-1]:
                    await asyncio.sleep(self.config.delay_between_tests)
            
            # Analyze all results
            self._analyze_results()
            
            return self.results
            
        finally:
            # Restore original memory configuration
            logger.info(f"Restoring original memory size: {original_memory}MB")
            await self.aws_provider.update_function_configuration(original_memory)

    async def _test_memory_configuration(self, memory_size: int) -> TuningResult:
        """Test a specific memory configuration.
        
        Args:
            memory_size: Memory size in MB
            
        Returns:
            TuningResult for this configuration
        """
        # Update function memory
        await self.aws_provider.update_function_configuration(memory_size)
        
        # Wait for configuration to propagate
        await asyncio.sleep(5)
        
        # Run test iterations
        test_runs: List[TestRun] = []
        
        for i in range(self.config.iterations):
            logger.debug(f"Running iteration {i+1}/{self.config.iterations}")
            
            # Execute function
            start_time = time.time()
            response = await self.aws_provider.invoke_function(self.config.payload)
            end_time = time.time()
            
            # Create test run record
            test_run = TestRun(
                iteration=i + 1,
                duration_ms=(end_time - start_time) * 1000,
                billed_duration_ms=response.get('BilledDuration', 0),
                memory_used_mb=response.get('MemoryUsed', 0),
                cold_start=response.get('ColdStart', False),
                status_code=response.get('StatusCode', 0),
                error=response.get('FunctionError'),
                log_result=response.get('LogResult'),
                timestamp=datetime.utcnow()
            )
            
            test_runs.append(test_run)
            
            # Add delay between invocations
            if i < self.config.iterations - 1:
                await asyncio.sleep(self.config.delay_between_invocations)
        
        # Create tuning result
        return TuningResult(
            memory_size=memory_size,
            test_runs=test_runs,
            timestamp=datetime.utcnow()
        )

    def _analyze_results(self):
        """Analyze all test results and add analysis data."""
        for result in self.results:
            # Analyze performance metrics
            result.analysis = self.analyzer.analyze(result)
            
            # Calculate cost metrics
            result.cost_analysis = self.analyzer.calculate_cost(
                result,
                self.config.region
            )

    async def run_with_reporting(self) -> Dict[str, Any]:
        """Run tuning with full reporting.
        
        Returns:
            Dictionary containing results and report paths
        """
        # Run the tuning process
        results = await self.run()
        
        # Generate reports
        report_paths = {}
        
        if 'json' in self.config.output_formats:
            report_paths['json'] = self.reporting.generate_json_report(
                results, self.config
            )
        
        if 'csv' in self.config.output_formats:
            report_paths['csv'] = self.reporting.generate_csv_report(
                results, self.config
            )
            
        if 'html' in self.config.output_formats:
            report_paths['html'] = self.reporting.generate_html_report(
                results, self.config
            )
        
        # Find optimal configuration
        optimal = self.analyzer.find_optimal_configuration(
            results,
            self.config.strategy
        )
        
        # Print summary
        self._print_summary(results, optimal)
        
        return {
            'results': results,
            'optimal_configuration': optimal,
            'report_paths': report_paths
        }

    def _print_summary(self, results: List[TuningResult], optimal: TuningResult):
        """Print a summary of the tuning results.
        
        Args:
            results: All tuning results
            optimal: The optimal configuration
        """
        print("\n" + "="*60)
        print("AWS Lambda Tuning Results Summary")
        print("="*60)
        print(f"Function: {self.config.function_arn}")
        print(f"Strategy: {self.config.strategy}")
        print(f"Tested memory sizes: {[r.memory_size for r in results]}")
        print("\nOptimal Configuration:")
        print(f"  Memory: {optimal.memory_size}MB")
        print(f"  Avg Duration: {optimal.analysis['avg_duration']:.2f}ms")
        print(f"  Avg Cost: ${optimal.cost_analysis['avg_cost_per_invocation']:.6f}")
        print(f"  Cost/Month (1M invocations): ${optimal.cost_analysis['monthly_cost_1m_invocations']:.2f}")
        
        if self.config.strategy == 'speed':
            improvement = (results[0].analysis['avg_duration'] - optimal.analysis['avg_duration']) / results[0].analysis['avg_duration'] * 100
            print(f"  Speed Improvement: {improvement:.1f}%")
        elif self.config.strategy == 'cost':
            savings = (results[-1].cost_analysis['avg_cost_per_invocation'] - optimal.cost_analysis['avg_cost_per_invocation']) / results[-1].cost_analysis['avg_cost_per_invocation'] * 100
            print(f"  Cost Savings: {savings:.1f}%")
        
        print("="*60 + "\n")
