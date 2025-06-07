#!/usr/bin/env python3
"""
Example: Memory-Intensive Workload Optimization

This example demonstrates how to optimize memory-intensive Lambda functions
that process large datasets, perform in-memory analytics, or cache data.
"""

import asyncio
import json
from aws_lambda_tuner import TunerConfig, TunerOrchestrator


async def optimize_memory_intensive_function():
    """Optimize a memory-intensive Lambda function."""
    print("🔧 Optimizing memory-intensive Lambda function...")
    
    # Configure for memory-intensive workloads
    config = TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:data-analytics",
        workload_type="memory_intensive",
        optimization_strategy="balanced",
        memory_range=(1024, 10240),  # Higher memory range
        step_size=1024,
        iterations_per_memory=15,
        concurrent_executions=5,  # Lower concurrency due to memory usage
        payload_template={
            "operation": "data_processing",
            "dataset_size": "large",
            "processing_type": "in_memory_analytics",
            "data_points": 1000000
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
        
        # Display memory-specific insights
        if hasattr(result.analysis, 'workload_analysis'):
            workload = result.analysis.workload_analysis
            print(f"\n🧠 Memory Workload Analysis:")
            print(f"   • Memory Utilization: {workload.resource_utilization.get('memory', 0):.1f}%")
            print(f"   • Heap Usage: {workload.resource_utilization.get('heap_usage', 0):.1f}%")
            print(f"   • GC Overhead: {workload.resource_utilization.get('gc_overhead', 0):.1f}%")
            
            for opportunity in workload.optimization_opportunities:
                print(f"   • {opportunity['description']}: {opportunity['impact']}")
        
        # Display efficiency scores
        if hasattr(result.analysis, 'efficiency_scores'):
            print(f"\n📈 Memory Efficiency Scores:")
            for memory_size, score in result.analysis.efficiency_scores.items():
                print(f"   • {memory_size}MB: {score:.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None


async def analyze_memory_patterns():
    """Analyze memory usage patterns for different workloads."""
    print("\n🔍 Analyzing memory usage patterns...")
    
    # Example of analyzing different memory patterns
    patterns = [
        {"size": 512, "usage": "Small dataset processing"},
        {"size": 1024, "usage": "Medium dataset with caching"},
        {"size": 3008, "usage": "Large in-memory analytics"},
        {"size": 10240, "usage": "Maximum memory for huge datasets"}
    ]
    
    for pattern in patterns:
        efficiency = (pattern["size"] / 10240) * 100  # Example calculation
        print(f"   • {pattern['size']}MB - {pattern['usage']}: {efficiency:.1f}% efficiency")


async def main():
    """Main execution function."""
    print("🚀 Memory-Intensive Lambda Optimization Example")
    print("=" * 50)
    
    result = await optimize_memory_intensive_function()
    
    if result:
        await analyze_memory_patterns()
        
        print(f"\n💡 Memory-Intensive Optimization Tips:")
        print(f"   • Monitor memory utilization to avoid over-provisioning")
        print(f"   • Consider streaming data processing for large datasets")
        print(f"   • Use memory-efficient data structures and algorithms")
        print(f"   • Implement proper garbage collection strategies")
        print(f"   • Consider breaking large tasks into smaller chunks")
        print(f"   • Use external storage (S3, DynamoDB) for large data")


if __name__ == "__main__":
    asyncio.run(main())