#!/usr/bin/env python3
"""
Example: I/O-Bound Workload Optimization

This example demonstrates how to optimize I/O-bound Lambda functions
that primarily wait for database queries, API calls, or file operations.
"""

import asyncio
import json
from aws_lambda_tuner import TunerConfig, TunerOrchestrator


async def optimize_io_bound_function():
    """Optimize an I/O-bound Lambda function."""
    print("🔧 Optimizing I/O-bound Lambda function...")
    
    # Configure for I/O-bound workloads
    config = TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:database-processor",
        workload_type="io_bound",
        optimization_strategy="cost_optimized",
        memory_range=(128, 1024),  # Lower memory for I/O workloads
        step_size=128,
        iterations_per_memory=25,
        concurrent_executions=20,  # Higher concurrency for I/O
        payload_template={
            "operation": "database_query",
            "table": "users",
            "query_type": "complex_join",
            "limit": 1000
        }
    )
    
    # Initialize orchestrator
    orchestrator = TunerOrchestrator(config)
    
    try:
        # Run optimization
        result = await orchestrator.run_comprehensive_analysis()
        
        print(f"\n✅ Optimization completed!")
        print(f"📊 Current memory: {result.recommendation.current_memory_size}MB")
        print(f"🎯 Optimal memory: {result.recommendation.optimal_memory_size}MB")
        print(f"⚡ Performance improvement: {result.recommendation.duration_change_percent:.1f}%")
        print(f"💰 Cost savings: {-result.recommendation.cost_change_percent:.1f}%")
        print(f"🤔 Recommendation: {result.recommendation.reasoning}")
        
        # Display I/O-specific insights
        if hasattr(result.analysis, 'workload_analysis'):
            workload = result.analysis.workload_analysis
            print(f"\n🔍 I/O Workload Analysis:")
            print(f"   • Network I/O: {workload.resource_utilization.get('network_io', 0):.1f}%")
            print(f"   • Disk I/O: {workload.resource_utilization.get('disk_io', 0):.1f}%")
            print(f"   • Wait Time: {workload.resource_utilization.get('wait_time', 0):.1f}%")
            
            for opportunity in workload.optimization_opportunities:
                print(f"   • {opportunity['description']}: {opportunity['impact']}")
        
        # Display concurrency analysis
        if hasattr(result.analysis, 'concurrency_analysis'):
            concurrency = result.analysis.concurrency_analysis
            print(f"\n🔀 Concurrency Analysis:")
            print(f"   • Avg Concurrent Executions: {concurrency.avg_concurrent_executions:.1f}")
            print(f"   • Peak Concurrent Executions: {concurrency.peak_concurrent_executions}")
            print(f"   • Scaling Efficiency: {concurrency.scaling_efficiency:.1f}%")
            
            if concurrency.recommended_concurrency_limit:
                print(f"   • Recommended Limit: {concurrency.recommended_concurrency_limit}")
        
        return result
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None


async def main():
    """Main execution function."""
    print("🚀 I/O-Bound Lambda Optimization Example")
    print("=" * 45)
    
    result = await optimize_io_bound_function()
    
    if result:
        print(f"\n💡 I/O-Bound Optimization Tips:")
        print(f"   • Lower memory often sufficient for I/O-bound tasks")
        print(f"   • Focus on reducing cold starts with provisioned concurrency")
        print(f"   • Optimize database connections and connection pooling")
        print(f"   • Consider async/await patterns for concurrent I/O")
        print(f"   • Use VPC endpoints to reduce network latency")


if __name__ == "__main__":
    asyncio.run(main())