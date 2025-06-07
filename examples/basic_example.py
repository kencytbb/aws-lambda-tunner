"""
Basic example of using AWS Lambda Tuner.

This example demonstrates how to tune a Lambda function
using the command line interface.
"""

import asyncio
from aws_lambda_tuner import TunerConfig, TunerOrchestrator, ReportGenerator


async def main():
    """Run basic Lambda tuning example."""
    
    # Create configuration
    config = TunerConfig(
        function_arn='arn:aws:lambda:us-east-1:123456789012:function:my-api-handler',
        payload={
            "httpMethod": "GET",
            "path": "/users",
            "headers": {
                "Content-Type": "application/json"
            }
        },
        memory_sizes=[256, 512, 1024, 1536],
        iterations=10,
        strategy='balanced',
        concurrent_executions=5,
        timeout=300
    )
    
    print(f"Starting Lambda tuning for: {config.function_arn}")
    print(f"Memory configurations to test: {config.memory_sizes}")
    print(f"Iterations per configuration: {config.iterations}")
    
    # Create orchestrator and run tuning
    orchestrator = TunerOrchestrator(config)
    results = await orchestrator.run_tuning()
    
    print("\nTuning completed!")
    print(f"Total time: {results['test_duration_seconds']:.2f} seconds")
    
    # Generate report
    report_gen = ReportGenerator(results, config)
    summary = report_gen.get_summary()
    
    print("\n=== TUNING RESULTS ===")
    print(f"Optimal Memory: {summary['optimal_memory']}MB")
    print(f"Average Duration: {summary['optimal_duration']:.2f}ms")
    print(f"Cost per Invocation: ${summary['optimal_cost']:.6f}")
    print(f"Performance Improvement: {summary['performance_gain']:.1f}%")
    print(f"Cost Savings: {summary['cost_savings']:.1f}%")
    
    # Save reports
    report_gen.save_json('tuning-results.json')
    report_gen.save_html('tuning-report.html')
    
    print("\nReports saved:")
    print("- tuning-results.json")
    print("- tuning-report.html")


if __name__ == '__main__':
    # Run the example
    asyncio.run(main())
