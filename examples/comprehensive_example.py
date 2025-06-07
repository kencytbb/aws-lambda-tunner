#!/usr/bin/env python3
"""
Example: Comprehensive Lambda Optimization Workflow

This example demonstrates the complete workflow for optimizing Lambda functions
using all features of the AWS Lambda Tuner, including workload analysis,
intelligent recommendations, monitoring, and reporting.
"""

import asyncio
import json
from datetime import datetime, timedelta
from aws_lambda_tuner import (
    TunerConfig, TunerOrchestrator, ReportGenerator,
    PerformanceAnalyzer, ReportingService
)


async def comprehensive_optimization_workflow():
    """Complete optimization workflow for a Lambda function."""
    print("🚀 Starting comprehensive Lambda optimization workflow...")
    
    # Step 1: Initial Configuration
    print("\n📋 Step 1: Configuring optimization parameters")
    config = TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:production-api",
        workload_type="balanced",  # Auto-detect workload type
        optimization_strategy="intelligent",  # Use AI-powered recommendations
        memory_range=(128, 3008),
        step_size=128,
        iterations_per_memory=20,
        concurrent_executions=15,
        enable_cold_start_analysis=True,
        enable_concurrency_analysis=True,
        enable_cost_analysis=True,
        enable_trend_analysis=True,
        payload_template={
            "event_type": "api_request",
            "path": "/api/v1/users",
            "method": "GET",
            "query_params": {"limit": 100, "offset": 0}
        }
    )
    
    # Step 2: Initialize orchestrator with advanced features
    print("🎯 Step 2: Initializing advanced orchestrator")
    orchestrator = TunerOrchestrator(config)
    
    try:
        # Step 3: Run comprehensive analysis
        print("🔬 Step 3: Running comprehensive analysis...")
        result = await orchestrator.run_comprehensive_analysis()
        
        # Step 4: Display optimization results
        print("\n✅ Step 4: Optimization Results")
        print("=" * 40)
        display_optimization_results(result)
        
        # Step 5: Advanced analysis insights
        print("\n🔍 Step 5: Advanced Analysis Insights")
        print("=" * 40)
        await display_advanced_insights(result)
        
        # Step 6: Generate comprehensive reports
        print("\n📊 Step 6: Generating comprehensive reports")
        print("=" * 40)
        await generate_comprehensive_reports(result, config)
        
        # Step 7: Monitoring recommendations
        print("\n📈 Step 7: Monitoring and Continuous Optimization")
        print("=" * 50)
        display_monitoring_recommendations(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Optimization workflow failed: {e}")
        return None


def display_optimization_results(result):
    """Display comprehensive optimization results."""
    print(f"🏷️  Function: {result.function_arn.split(':')[-1]}")
    print(f"⏱️  Analysis Duration: {result.duration:.1f}s")
    print(f"🎯 Strategy: {result.strategy}")
    
    rec = result.recommendation
    print(f"\n💡 Recommendation Summary:")
    print(f"   Current Memory: {rec.current_memory_size}MB")
    print(f"   Optimal Memory: {rec.optimal_memory_size}MB")
    print(f"   Should Optimize: {'✅ Yes' if rec.should_optimize else '❌ No'}")
    print(f"   Performance Change: {rec.duration_change_percent:+.1f}%")
    print(f"   Cost Change: {rec.cost_change_percent:+.1f}%")
    print(f"   Confidence Score: {rec.confidence_score:.1f}/100")
    print(f"   Reasoning: {rec.reasoning}")


async def display_advanced_insights(result):
    """Display advanced analysis insights."""
    analysis = result.analysis
    
    # Cold start analysis
    if hasattr(analysis, 'cold_start_analysis'):
        cold_start = analysis.cold_start_analysis
        print(f"🥶 Cold Start Analysis:")
        print(f"   • Cold Start Ratio: {cold_start.cold_start_ratio:.1%}")
        print(f"   • Avg Cold Start: {cold_start.avg_cold_start_duration:.0f}ms")
        print(f"   • Avg Warm Start: {cold_start.avg_warm_start_duration:.0f}ms")
        print(f"   • Impact Score: {cold_start.cold_start_impact_score:.1f}/10")
        print(f"   • Optimal Memory for Cold Starts: {cold_start.optimal_memory_for_cold_starts}MB")
    
    # Concurrency analysis
    if hasattr(analysis, 'concurrency_analysis'):
        concurrency = analysis.concurrency_analysis
        print(f"\n🔀 Concurrency Analysis:")
        print(f"   • Avg Concurrent Executions: {concurrency.avg_concurrent_executions:.1f}")
        print(f"   • Peak Concurrency: {concurrency.peak_concurrent_executions}")
        print(f"   • Utilization: {concurrency.concurrency_utilization:.1%}")
        print(f"   • Scaling Efficiency: {concurrency.scaling_efficiency:.1%}")
        print(f"   • Throttling Events: {concurrency.throttling_events}")
        
        if concurrency.recommended_concurrency_limit:
            print(f"   • Recommended Limit: {concurrency.recommended_concurrency_limit}")
    
    # Performance trends
    if hasattr(analysis, 'trends'):
        print(f"\n📈 Performance Trends:")
        for trend_name, trend_data in analysis.trends.items():
            print(f"   • {trend_name}: {trend_data}")
    
    # Key insights
    if hasattr(analysis, 'insights'):
        print(f"\n💡 Key Insights:")
        for insight in analysis.insights:
            print(f"   • {insight.get('title', 'Insight')}: {insight.get('description', '')}")


async def generate_comprehensive_reports(result, config):
    """Generate comprehensive reports in multiple formats."""
    report_generator = ReportGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Generate detailed HTML report
        html_report = await report_generator.generate_comprehensive_report(
            result, 
            format="html",
            include_charts=True,
            include_recommendations=True
        )
        
        # Save HTML report
        html_filename = f"lambda_optimization_report_{timestamp}.html"
        with open(html_filename, 'w') as f:
            f.write(html_report)
        print(f"📄 HTML Report: {html_filename}")
        
        # Generate JSON report for programmatic access
        json_report = await report_generator.generate_comprehensive_report(
            result,
            format="json",
            include_raw_data=True
        )
        
        json_filename = f"lambda_optimization_data_{timestamp}.json"
        with open(json_filename, 'w') as f:
            f.write(json_report)
        print(f"📋 JSON Report: {json_filename}")
        
        # Generate executive summary
        summary = await report_generator.generate_executive_summary(result)
        print(f"📝 Executive Summary: {len(summary)} characters")
        
    except Exception as e:
        print(f"⚠️  Report generation error: {e}")


def display_monitoring_recommendations(result):
    """Display monitoring and continuous optimization recommendations."""
    print("🔧 Monitoring Setup Recommendations:")
    print("   • Set up CloudWatch alarms for duration and cost metrics")
    print("   • Monitor cold start ratios and implement warming strategies")
    print("   • Track memory utilization patterns over time")
    print("   • Set up automated re-optimization triggers")
    
    print(f"\n🔄 Continuous Optimization:")
    print("   • Schedule monthly optimization reviews")
    print("   • Monitor for workload pattern changes")
    print("   • Implement A/B testing for configuration changes")
    print("   • Set up cost anomaly detection")
    
    if result.recommendation.should_optimize:
        print(f"\n⚡ Next Steps:")
        print(f"   1. Update Lambda memory to {result.recommendation.optimal_memory_size}MB")
        print(f"   2. Monitor performance for 1-2 weeks")
        print(f"   3. Validate cost savings match projections")
        print(f"   4. Consider provisioned concurrency if cold starts are high")


async def batch_optimization_example():
    """Example of optimizing multiple functions in batch."""
    print("\n🔄 Batch Optimization Example")
    print("=" * 35)
    
    functions = [
        "arn:aws:lambda:us-east-1:123456789012:function:api-handler",
        "arn:aws:lambda:us-east-1:123456789012:function:data-processor",
        "arn:aws:lambda:us-east-1:123456789012:function:notification-service"
    ]
    
    total_savings = 0
    
    for function_arn in functions:
        print(f"\n🎯 Optimizing {function_arn.split(':')[-1]}...")
        
        config = TunerConfig(
            function_arn=function_arn,
            optimization_strategy="cost_optimized",
            memory_range=(128, 1024),
            iterations_per_memory=10  # Faster for batch processing
        )
        
        orchestrator = TunerOrchestrator(config)
        
        try:
            result = await orchestrator.run_optimization()
            savings = result.recommendation.estimated_monthly_savings
            total_savings += savings.get('total', {}).get('amount', 0)
            
            print(f"   ✅ Optimized: {result.recommendation.optimal_memory_size}MB")
            print(f"   💰 Monthly Savings: ${savings.get('total', {}).get('amount', 0):.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n💰 Total Monthly Savings: ${total_savings:.2f}")


async def main():
    """Main execution function."""
    print("🚀 AWS Lambda Tuner - Comprehensive Optimization Example")
    print("=" * 60)
    
    # Run comprehensive workflow
    result = await comprehensive_optimization_workflow()
    
    if result:
        # Run batch optimization example
        await batch_optimization_example()
        
        print(f"\n🎉 Optimization workflow completed successfully!")
        print(f"📖 Check the generated reports for detailed analysis")
        print(f"🔔 Set up monitoring for continuous optimization")


if __name__ == "__main__":
    asyncio.run(main())