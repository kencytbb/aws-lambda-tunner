# Migration Guide: AWS Lambda Tuner v1.0.0 → v2.0.0

This guide helps you migrate from AWS Lambda Tuner v1.0.0 to v2.0.0, which introduces significant new features including intelligent workload analysis, advanced monitoring, and enhanced optimization strategies.

## 🚀 What's New in v2.0.0

### Major New Features
- **Intelligent Workload Analysis**: Automatic detection and optimization for CPU-intensive, I/O-bound, and memory-intensive workloads
- **AI-Powered Recommendations**: Machine learning-based optimization suggestions
- **Advanced Monitoring**: Real-time performance monitoring and alert management
- **Continuous Optimization**: Automated monitoring and re-optimization capabilities
- **Enhanced Reporting**: Comprehensive reports with interactive dashboards
- **Cold Start Analysis**: Detailed cold start pattern analysis and optimization
- **Concurrency Analysis**: Advanced concurrency pattern detection and optimization

### New Classes and Modules
- `TunerOrchestrator` - Enhanced orchestration with intelligent workflows
- `PerformanceAnalyzer` - Advanced performance analysis capabilities
- `ReportingService` - Comprehensive reporting and export functionality
- `IntelligentRecommendationEngine` - ML-powered optimization recommendations
- `PerformanceMonitor` - Real-time monitoring and alerting
- `ContinuousOptimizer` - Automated continuous optimization

## 📋 Migration Steps

### 1. Update Dependencies

**Before (v1.0.0):**
```python
# requirements.txt
boto3>=1.26.0
click>=8.0.0
tabulate>=0.9.0
```

**After (v2.0.0):**
```python
# requirements.txt
boto3>=1.28.0
click>=8.1.0
tabulate>=0.9.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
aiohttp>=3.8.0
```

### 2. Configuration Changes

**Before (v1.0.0):**
```python
from aws_lambda_tuner import TunerConfig

config = TunerConfig(
    function_arn="arn:aws:lambda:us-east-1:123456789012:function:my-function",
    memory_range=(128, 1024),
    iterations=10
)
```

**After (v2.0.0):**
```python
from aws_lambda_tuner import TunerConfig

config = TunerConfig(
    function_arn="arn:aws:lambda:us-east-1:123456789012:function:my-function",
    workload_type="balanced",  # NEW: auto-detect or specify workload type
    optimization_strategy="intelligent",  # NEW: AI-powered optimization
    memory_range=(128, 1024),
    iterations_per_memory=10,  # RENAMED: more specific parameter name
    enable_cold_start_analysis=True,  # NEW: cold start analysis
    enable_concurrency_analysis=True,  # NEW: concurrency analysis
    enable_cost_analysis=True,  # NEW: enhanced cost analysis
)
```

### 3. Orchestration Changes

**Before (v1.0.0):**
```python
from aws_lambda_tuner.orchestrator import run_tuning_session

# Simple function call
result = run_tuning_session(config)
```

**After (v2.0.0):**
```python
from aws_lambda_tuner import TunerOrchestrator
import asyncio

# Enhanced async orchestrator
orchestrator = TunerOrchestrator(config)

async def optimize():
    # Run comprehensive analysis with all new features
    result = await orchestrator.run_comprehensive_analysis()
    return result

# Execute async function
result = asyncio.run(optimize())
```

### 4. Results Structure Changes

**Before (v1.0.0):**
```python
# Simple optimization result
print(f"Optimal memory: {result.optimal_memory}")
print(f"Cost savings: {result.cost_savings}")
```

**After (v2.0.0):**
```python
# Rich result structure with detailed analysis
print(f"Optimal memory: {result.recommendation.optimal_memory_size}")
print(f"Cost change: {result.recommendation.cost_change_percent}%")
print(f"Performance change: {result.recommendation.duration_change_percent}%")
print(f"Confidence: {result.recommendation.confidence_score}/100")
print(f"Reasoning: {result.recommendation.reasoning}")

# Access detailed analysis
if result.analysis.cold_start_analysis:
    cold_start = result.analysis.cold_start_analysis
    print(f"Cold start ratio: {cold_start.cold_start_ratio:.1%}")
    print(f"Cold start impact: {cold_start.cold_start_impact_score}/10")

if result.analysis.concurrency_analysis:
    concurrency = result.analysis.concurrency_analysis
    print(f"Avg concurrency: {concurrency.avg_concurrent_executions}")
    print(f"Scaling efficiency: {concurrency.scaling_efficiency:.1%}")
```

### 5. Reporting Changes

**Before (v1.0.0):**
```python
from aws_lambda_tuner.reports import generate_summary_report

# Basic text report
report = generate_summary_report(result)
print(report)
```

**After (v2.0.0):**
```python
from aws_lambda_tuner import ReportGenerator

# Enhanced reporting with multiple formats
report_generator = ReportGenerator()

# Generate comprehensive HTML report with charts
html_report = await report_generator.generate_comprehensive_report(
    result, 
    format="html",
    include_charts=True,
    include_recommendations=True
)

# Generate JSON report for programmatic access
json_report = await report_generator.generate_comprehensive_report(
    result,
    format="json",
    include_raw_data=True
)

# Generate executive summary
summary = await report_generator.generate_executive_summary(result)
```

## 🔄 Backward Compatibility

### Supported Legacy APIs
The following v1.0.0 APIs are still supported but deprecated:

```python
# DEPRECATED: Legacy function-based API
from aws_lambda_tuner.orchestrator import run_tuning_session
from aws_lambda_tuner.reports import generate_summary_report

# These still work but emit deprecation warnings
result = run_tuning_session(config)
report = generate_summary_report(result)
```

### Migration Helpers
We provide migration helpers to ease the transition:

```python
from aws_lambda_tuner.migration import LegacyAdapter

# Automatically convert v1.0.0 config to v2.0.0 format
legacy_config = {...}  # Your v1.0.0 configuration
new_config = LegacyAdapter.upgrade_config(legacy_config)

# Convert v1.0.0 results to v2.0.0 format
legacy_result = {...}  # Your v1.0.0 result
new_result = LegacyAdapter.upgrade_result(legacy_result)
```

## 🛠️ Common Migration Patterns

### Pattern 1: Simple Cost Optimization
**Before:**
```python
config = TunerConfig(function_arn=arn, strategy="cost")
result = run_tuning_session(config)
```

**After:**
```python
config = TunerConfig(function_arn=arn, optimization_strategy="cost_optimized")
orchestrator = TunerOrchestrator(config)
result = await orchestrator.run_optimization()
```

### Pattern 2: Performance Optimization
**Before:**
```python
config = TunerConfig(function_arn=arn, strategy="performance")
result = run_tuning_session(config)
```

**After:**
```python
config = TunerConfig(function_arn=arn, optimization_strategy="speed_optimized")
orchestrator = TunerOrchestrator(config)
result = await orchestrator.run_optimization()
```

### Pattern 3: Batch Optimization
**Before:**
```python
for function_arn in functions:
    config = TunerConfig(function_arn=function_arn)
    result = run_tuning_session(config)
    print(f"Optimized {function_arn}")
```

**After:**
```python
async def optimize_functions(functions):
    for function_arn in functions:
        config = TunerConfig(function_arn=function_arn)
        orchestrator = TunerOrchestrator(config)
        result = await orchestrator.run_optimization()
        print(f"Optimized {function_arn}: {result.recommendation.reasoning}")

await optimize_functions(functions)
```

## 🚨 Breaking Changes

### 1. Async/Await Required
All optimization operations are now async and require `await`:

```python
# OLD: Synchronous
result = run_tuning_session(config)

# NEW: Asynchronous
result = await orchestrator.run_optimization()
```

### 2. Configuration Parameter Changes
- `iterations` → `iterations_per_memory`
- `strategy` → `optimization_strategy`
- Added required `workload_type` parameter

### 3. Result Structure Changes
- `optimal_memory` → `recommendation.optimal_memory_size`
- `cost_savings` → `recommendation.cost_change_percent`
- Added comprehensive analysis data structures

### 4. Import Changes
```python
# OLD imports
from aws_lambda_tuner.orchestrator import run_tuning_session
from aws_lambda_tuner.reports import generate_summary_report

# NEW imports
from aws_lambda_tuner import TunerOrchestrator, ReportGenerator
```

## 📊 New Capabilities Examples

### Workload-Specific Optimization
```python
# Optimize for CPU-intensive workloads
config = TunerConfig(
    function_arn=arn,
    workload_type="cpu_intensive",
    optimization_strategy="speed_optimized"
)

# Optimize for I/O-bound workloads
config = TunerConfig(
    function_arn=arn,
    workload_type="io_bound",
    optimization_strategy="cost_optimized"
)
```

### Continuous Monitoring
```python
from aws_lambda_tuner.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(config)
await monitor.start_monitoring()
```

### Advanced Analysis
```python
# Get intelligent recommendations
result = await orchestrator.run_comprehensive_analysis()

# Access AI-powered insights
for insight in result.analysis.insights:
    print(f"Insight: {insight['title']}")
    print(f"Description: {insight['description']}")
    print(f"Impact: {insight['impact']}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all new dependencies are installed
2. **Async Errors**: Wrap optimization calls in `asyncio.run()` or use `await`
3. **Configuration Errors**: Update parameter names to v2.0.0 format
4. **Permission Errors**: Ensure IAM permissions include new CloudWatch and monitoring APIs

### Getting Help

- Check the [examples/](./examples/) directory for complete working examples
- Review the [API documentation](./docs/) for detailed parameter descriptions
- Use the migration helpers in `aws_lambda_tuner.migration` module

## 📝 Checklist

Use this checklist to ensure complete migration:

- [ ] Update dependencies in requirements.txt/pyproject.toml
- [ ] Update import statements to use new classes
- [ ] Convert synchronous calls to async/await pattern
- [ ] Update configuration parameters to v2.0.0 format
- [ ] Update result access patterns for new structure
- [ ] Test optimization workflows with new features
- [ ] Update reporting code to use new ReportGenerator
- [ ] Configure monitoring if desired
- [ ] Update error handling for new exception types
- [ ] Review and update any custom extensions

## 🎯 Recommended Upgrade Path

1. **Phase 1**: Install v2.0.0 alongside v1.0.0 (different virtual environment)
2. **Phase 2**: Test new features with non-production functions
3. **Phase 3**: Migrate configuration and basic optimization workflows
4. **Phase 4**: Adopt advanced features (monitoring, continuous optimization)
5. **Phase 5**: Full production migration and remove v1.0.0

This phased approach minimizes risk and allows you to gradually adopt new capabilities.