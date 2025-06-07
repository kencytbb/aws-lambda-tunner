#!/usr/bin/env python3
"""
Example: CPU-Intensive Workload Optimization

This example demonstrates how to optimize a CPU-intensive Lambda function
for compute-heavy workloads like data processing, image manipulation, or
mathematical calculations.
"""

import asyncio
import json
from aws_lambda_tuner import TunerConfig, TunerOrchestrator


async def optimize_cpu_intensive_function():
    """Optimize a CPU-intensive Lambda function."""
    print("🔧 Optimizing CPU-intensive Lambda function...")
    
    # Configure for CPU-intensive workloads
    config = TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:cpu-heavy-processor",
        workload_type="cpu_intensive",
        optimization_strategy="speed_optimized",
        memory_range=(512, 10240),  # Test higher memory for CPU boost
        step_size=512,
        iterations_per_memory=20,
        concurrent_executions=10,
        payload_template={
            "operation": "matrix_multiplication",
            "size": 1000,
            "iterations": 5
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
        print(f"💰 Cost change: {result.recommendation.cost_change_percent:.1f}%")
        print(f"🤔 Recommendation: {result.recommendation.reasoning}")
        
        # Display workload-specific insights
        if hasattr(result.analysis, 'workload_analysis'):
            workload = result.analysis.workload_analysis
            print(f"\n🔍 CPU Workload Analysis:")
            print(f"   • CPU Utilization: {workload.resource_utilization.get('cpu', 0):.1f}%")
            print(f"   • Memory Utilization: {workload.resource_utilization.get('memory', 0):.1f}%")
            
            for opportunity in workload.optimization_opportunities:
                print(f"   • {opportunity['description']}: {opportunity['impact']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None


async def main():
    """Main execution function."""
    print("🚀 CPU-Intensive Lambda Optimization Example")
    print("=" * 50)
    
    result = await optimize_cpu_intensive_function()
    
    if result:
        print(f"\n💡 CPU-Intensive Optimization Tips:")
        print(f"   • Higher memory = more vCPU allocation for compute tasks")
        print(f"   • Consider provisioned concurrency for consistent performance")
        print(f"   • Monitor CPU utilization vs cost for optimal balance")
        print(f"   • Use ARM64 Graviton2 processors for cost efficiency")


if __name__ == "__main__":
    asyncio.run(main())