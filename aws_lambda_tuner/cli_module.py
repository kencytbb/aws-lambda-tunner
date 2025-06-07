"""
Command Line Interface for AWS Lambda Tuner.
Enhanced with workload-guided workflows and cost integration.
"""

import click
import asyncio
import json
import sys
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
from datetime import datetime

from .config_module import TunerConfig, ConfigManager
from .orchestrator_module import TunerOrchestrator
from .report_service import ReportGenerator, WorkloadType, TrafficPattern
# from .visualization_module import VisualizationEngine  # Commented out for architecture compatibility
from .reporting.workload_report_templates import WorkloadReportTemplates
from .reporting.interactive_dashboards import InteractiveDashboard
from .reporting.export_formats import MultiFormatExporter
from .exceptions import TunerException, ConfigurationError
from .utils import validate_arn, load_json_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0', prog_name='aws-lambda-tuner')
def cli():
    """AWS Lambda Performance Tuner - Optimize your Lambda functions for cost and performance."""
    pass


@cli.command()
@click.option('--output', '-o', default='tuner.config.json', help='Output configuration file path')
@click.option('--template', '-t', type=click.Choice(['speed', 'cost', 'balanced', 'comprehensive']), 
              default='balanced', help='Configuration template to use')
def init(output: str, template: str):
    """Generate a sample configuration file."""
    try:
        config_manager = ConfigManager()
        config = config_manager.create_from_template(template)
        
        # Convert to dict for saving
        config_dict = {
            'function_arn': 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
            'payload': config.payload,
            'memory_sizes': config.memory_sizes,
            'iterations': config.iterations,
            'strategy': config.strategy,
            'concurrent_executions': config.concurrent_executions,
            'timeout': config.timeout,
            'dry_run': config.dry_run,
            'output_dir': config.output_dir
        }
        
        with open(output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        click.echo(f"✅ Configuration file created: {output}")
        click.echo(f"   Template used: {template}")
        click.echo("\nNext steps:")
        click.echo("1. Edit the configuration file with your Lambda function ARN")
        click.echo("2. Adjust the payload if needed")
        click.echo("3. Run: aws-lambda-tuner tune --config tuner.config.json")
        
    except Exception as e:
        click.echo(f"❌ Error creating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--function-arn', '-f', help='Lambda function ARN')
@click.option('--payload', '-p', help='JSON payload for Lambda invocation')
@click.option('--payload-file', type=click.Path(exists=True), help='File containing JSON payload')
@click.option('--memory-sizes', '-m', help='Comma-separated memory sizes (e.g., 256,512,1024)')
@click.option('--iterations', '-i', type=int, default=10, help='Number of iterations per memory size')
@click.option('--strategy', '-s', type=click.Choice(['speed', 'cost', 'balanced']), 
              default='balanced', help='Optimization strategy')
@click.option('--concurrent', type=int, default=5, help='Number of concurrent executions')
@click.option('--timeout', type=int, default=300, help='Lambda execution timeout in seconds')
@click.option('--output-dir', '-o', default='./tuning-results', help='Output directory for results')
@click.option('--dry-run', is_flag=True, help='Perform a dry run without invoking Lambda')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'html']), 
              default='json', help='Output format for results')
@click.option('--visualize', is_flag=True, help='Generate visualization charts')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def tune(config: Optional[str], function_arn: Optional[str], payload: Optional[str],
         payload_file: Optional[str], memory_sizes: Optional[str], iterations: int,
         strategy: str, concurrent: int, timeout: int, output_dir: str,
         dry_run: bool, output_format: str, visualize: bool, verbose: bool):
    """Run Lambda performance tuning."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if config:
            config_data = load_json_file(config)
            tuner_config = TunerConfig(**config_data)
        else:
            # Build config from CLI arguments
            if not function_arn:
                raise ConfigurationError("Function ARN is required. Use --function-arn or provide a config file.")
            
            # Handle payload
            if payload_file:
                with open(payload_file, 'r') as f:
                    payload_data = f.read()
            else:
                payload_data = payload or '{}'
            
            # Parse memory sizes
            if memory_sizes:
                memory_list = [int(m.strip()) for m in memory_sizes.split(',')]
            else:
                memory_list = None
            
            tuner_config = TunerConfig(
                function_arn=function_arn,
                payload=payload_data,
                memory_sizes=memory_list,
                iterations=iterations,
                strategy=strategy,
                concurrent_executions=concurrent,
                timeout=timeout,
                dry_run=dry_run,
                output_dir=output_dir
            )
        
        # Validate configuration
        if not validate_arn(tuner_config.function_arn):
            raise ConfigurationError(f"Invalid Lambda ARN: {tuner_config.function_arn}")
        
        click.echo("🚀 Starting Lambda performance tuning...")
        click.echo(f"   Function: {tuner_config.function_arn}")
        click.echo(f"   Memory sizes: {tuner_config.memory_sizes}")
        click.echo(f"   Iterations: {tuner_config.iterations}")
        click.echo(f"   Strategy: {tuner_config.strategy}")
        
        if dry_run:
            click.echo("\n⚠️  DRY RUN MODE - No actual Lambda invocations will be made")
        
        # Run tuning
        orchestrator = TunerOrchestrator(tuner_config)
        results = asyncio.run(orchestrator.run_tuning())
        
        # Generate report
        click.echo("\n📊 Generating reports...")
        report_gen = ReportGenerator(results, tuner_config)
        
        # Save results in requested format
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            report_path = output_path / 'tuning-results.json'
            report_gen.save_json(str(report_path))
        elif output_format == 'csv':
            report_path = output_path / 'tuning-results.csv'
            report_gen.save_csv(str(report_path))
        elif output_format == 'html':
            report_path = output_path / 'tuning-report.html'
            report_gen.save_html(str(report_path))
        
        click.echo(f"✅ Results saved to: {report_path}")
        
        # Generate visualizations if requested
        if visualize:
            click.echo("\n📈 Generating visualizations...")
            viz_engine = VisualizationEngine(results)
            
            # Generate various charts
            viz_engine.plot_performance_comparison(str(output_path / 'performance-comparison.png'))
            viz_engine.plot_cost_analysis(str(output_path / 'cost-analysis.png'))
            viz_engine.plot_duration_distribution(str(output_path / 'duration-distribution.png'))
            viz_engine.plot_optimization_curve(str(output_path / 'optimization-curve.png'))
            
            click.echo("✅ Visualizations saved to output directory")
        
        # Display summary
        summary = report_gen.get_summary()
        click.echo("\n📋 TUNING SUMMARY")
        click.echo("="*50)
        click.echo(f"Optimal Configuration: {summary['optimal_memory']}MB")
        click.echo(f"Average Duration: {summary['optimal_duration']:.2f}ms")
        click.echo(f"Estimated Cost: ${summary['optimal_cost']:.6f} per invocation")
        click.echo(f"Performance Gain: {summary['performance_gain']:.1f}%")
        click.echo(f"Cost Savings: {summary['cost_savings']:.1f}%")
        click.echo("="*50)
        
    except TunerException as e:
        click.echo(f"❌ Tuner error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('results-file', type=click.Path(exists=True))
@click.option('--format', 'output_format', type=click.Choice(['summary', 'detailed', 'json']), 
              default='summary', help='Report format')
def report(results_file: str, output_format: str):
    """Generate a report from existing results."""
    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create report generator
        report_gen = ReportGenerator(results, None)
        
        if output_format == 'summary':
            summary = report_gen.get_summary()
            click.echo("\n📋 PERFORMANCE SUMMARY")
            click.echo("="*50)
            for key, value in summary.items():
                click.echo(f"{key.replace('_', ' ').title()}: {value}")
            click.echo("="*50)
            
        elif output_format == 'detailed':
            detailed = report_gen.get_detailed_report()
            click.echo(json.dumps(detailed, indent=2))
            
        elif output_format == 'json':
            click.echo(json.dumps(results, indent=2))
            
    except Exception as e:
        click.echo(f"❌ Error generating report: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results-file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./visualizations', help='Output directory for charts')
@click.option('--charts', '-c', multiple=True, 
              type=click.Choice(['performance', 'cost', 'distribution', 'optimization', 'all']),
              default=['all'], help='Charts to generate')
def visualize(results_file: str, output_dir: str, charts: List[str]):
    """Generate visualization charts from results."""
    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create visualization engine
        viz_engine = VisualizationEngine(results)
        
        # Generate requested charts
        if 'all' in charts or 'performance' in charts:
            viz_engine.plot_performance_comparison(str(output_path / 'performance-comparison.png'))
            click.echo("✅ Generated: performance-comparison.png")
            
        if 'all' in charts or 'cost' in charts:
            viz_engine.plot_cost_analysis(str(output_path / 'cost-analysis.png'))
            click.echo("✅ Generated: cost-analysis.png")
            
        if 'all' in charts or 'distribution' in charts:
            viz_engine.plot_duration_distribution(str(output_path / 'duration-distribution.png'))
            click.echo("✅ Generated: duration-distribution.png")
            
        if 'all' in charts or 'optimization' in charts:
            viz_engine.plot_optimization_curve(str(output_path / 'optimization-curve.png'))
            click.echo("✅ Generated: optimization-curve.png")
            
        click.echo(f"\n✅ All visualizations saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error generating visualizations: {e}", err=True)
        sys.exit(1)


@cli.command()
def templates():
    """List available configuration templates."""
    templates_info = {
        'speed': 'Optimized for fastest execution time',
        'cost': 'Optimized for lowest cost',
        'balanced': 'Balance between speed and cost',
        'comprehensive': 'Comprehensive testing across all memory sizes'
    }
    
    click.echo("\n📋 Available Configuration Templates")
    click.echo("="*50)
    for name, description in templates_info.items():
        click.echo(f"{name:15} - {description}")
    click.echo("="*50)
    click.echo("\nUse: aws-lambda-tuner init --template <name>")


@cli.command()
@click.option('--function-arn', '-f', required=True, help='Lambda function ARN')
@click.option('--workload-type', '-w', 
              type=click.Choice(['web_api', 'batch_processing', 'event_driven', 'scheduled', 'stream_processing']),
              help='Type of workload for specialized optimization')
@click.option('--output-dir', '-o', default='./workload-analysis', help='Output directory')
@click.option('--interactive', is_flag=True, help='Interactive workload configuration')
def workload_wizard(function_arn: str, workload_type: Optional[str], output_dir: str, interactive: bool):
    """Interactive workload-specific tuning wizard."""
    try:
        click.echo("🧙 Lambda Workload Optimization Wizard")
        click.echo("="*50)
        
        # Determine workload type
        if not workload_type and not interactive:
            click.echo("\nWorkload type not specified. Starting interactive mode...")
            interactive = True
        
        if interactive:
            workload_type = _interactive_workload_selection()
        
        # Get workload-specific configuration
        workload_config = _get_workload_config(workload_type, interactive)
        
        # Create optimized configuration
        config_manager = ConfigManager()
        base_config = config_manager.create_from_template('balanced')
        
        # Apply workload-specific settings
        tuner_config = _customize_config_for_workload(base_config, workload_type, workload_config, function_arn)
        
        # Run tuning
        click.echo(f"\n🚀 Starting {workload_type.replace('_', ' ').title()} optimization...")
        orchestrator = TunerOrchestrator(tuner_config)
        results = asyncio.run(orchestrator.run_tuning())
        
        # Generate workload-specific report
        click.echo("\n📊 Generating workload-specific analysis...")
        workload_enum = WorkloadType(workload_type)
        report_gen = ReportGenerator(results, tuner_config, workload_enum)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive reports
        workload_analysis = report_gen.workload_specific_reports()
        
        # Save reports in multiple formats
        report_templates = WorkloadReportTemplates()
        
        # HTML report
        html_report = report_templates.format_workload_report(workload_analysis, "html")
        with open(output_path / f"{workload_type}_report.html", 'w') as f:
            f.write(html_report)
        
        # JSON report
        json_report = report_templates.format_workload_report(workload_analysis, "json")
        with open(output_path / f"{workload_type}_analysis.json", 'w') as f:
            f.write(json_report)
        
        # Interactive dashboard
        dashboard = InteractiveDashboard(results)
        dashboard.create_performance_dashboard(
            str(output_path / f"{workload_type}_dashboard.html"), 
            workload_type
        )
        
        # Display results
        _display_workload_results(workload_analysis, workload_type)
        
        click.echo(f"\n✅ Workload analysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error in workload wizard: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results-file', type=click.Path(exists=True))
@click.option('--scenarios-file', type=click.Path(exists=True), 
              help='JSON file with cost projection scenarios')
@click.option('--monthly-invocations', type=int, default=1000000, 
              help='Monthly invocations for cost projection')
@click.option('--pattern', type=click.Choice(['steady', 'bursty', 'seasonal', 'growth']),
              default='steady', help='Traffic pattern for cost calculation')
@click.option('--output-dir', '-o', default='./cost-analysis', help='Output directory')
@click.option('--format', 'export_format', type=click.Choice(['html', 'pdf', 'excel', 'json']),
              default='html', help='Export format')
def cost_projection(results_file: str, scenarios_file: Optional[str], monthly_invocations: int,
                   pattern: str, output_dir: str, export_format: str):
    """Generate detailed cost projections and analysis."""
    try:
        click.echo("💰 Lambda Cost Projection Analysis")
        click.echo("="*50)
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Load or create scenarios
        if scenarios_file:
            with open(scenarios_file, 'r') as f:
                scenarios_data = json.load(f)
            scenarios = scenarios_data.get('scenarios', [])
        else:
            # Create default scenarios
            scenarios = [
                {
                    'name': 'Current Usage',
                    'daily_invocations': monthly_invocations // 30,
                    'pattern': pattern,
                    'duration_days': 30
                },
                {
                    'name': 'Low Volume',
                    'daily_invocations': 1000,
                    'pattern': 'steady',
                    'duration_days': 30
                },
                {
                    'name': 'High Volume',
                    'daily_invocations': monthly_invocations // 10,
                    'pattern': 'bursty',
                    'duration_days': 30
                }
            ]
        
        # Generate cost projections
        report_gen = ReportGenerator(results)
        cost_analysis = report_gen.cost_projection_reports(scenarios)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export in requested format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == 'html':
            # Generate HTML cost report
            html_content = _generate_cost_html_report(cost_analysis, scenarios)
            output_file = output_path / f"cost_projection_{timestamp}.html"
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        elif export_format in ['pdf', 'excel']:
            # Use multi-format exporter
            exporter = MultiFormatExporter(results)
            output_file = output_path / f"cost_projection_{timestamp}.{export_format}"
            exporter.export_report(str(output_file), export_format, "cost_analysis")
        
        else:  # JSON
            output_file = output_path / f"cost_projection_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(cost_analysis, f, indent=2, default=str)
        
        # Display summary
        _display_cost_summary(cost_analysis)
        
        click.echo(f"\n✅ Cost projection analysis saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Error generating cost projection: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results-files', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--workload-types', help='Comma-separated workload types for comparison')
@click.option('--output-dir', '-o', default='./comparison', help='Output directory')
@click.option('--interactive-dashboard', is_flag=True, help='Generate interactive comparison dashboard')
def compare_workloads(results_files: tuple, workload_types: Optional[str], output_dir: str, 
                     interactive_dashboard: bool):
    """Compare performance across different workload types."""
    try:
        click.echo("🔍 Multi-Workload Performance Comparison")
        click.echo("="*50)
        
        if len(results_files) < 2:
            click.echo("❌ At least 2 results files are required for comparison")
            sys.exit(1)
        
        # Parse workload types
        if workload_types:
            workload_list = [w.strip() for w in workload_types.split(',')]
        else:
            workload_list = ['generic'] * len(results_files)
        
        if len(workload_list) != len(results_files):
            click.echo(f"❌ Number of workload types ({len(workload_list)}) must match number of result files ({len(results_files)})")
            sys.exit(1)
        
        # Load all results
        comparison_data = []
        for i, (results_file, workload_type) in enumerate(zip(results_files, workload_list)):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Generate workload analysis
            if workload_type != 'generic':
                workload_enum = WorkloadType(workload_type)
                report_gen = ReportGenerator(results, None, workload_enum)
                analysis = report_gen.workload_specific_reports()
            else:
                report_gen = ReportGenerator(results)
                analysis = {
                    'workload_type': 'generic',
                    'key_metrics': {
                        'optimal_memory': report_gen._find_optimal_configuration()['memory_mb'],
                        'performance_gain': 0,
                        'cost_savings': 0
                    }
                }
            
            comparison_data.append({
                'workload_type': workload_type,
                'name': f"Workload {i+1}",
                'analysis': analysis,
                'results': results
            })
        
        # Generate comparative analysis
        primary_report_gen = ReportGenerator(comparison_data[0]['results'])
        comparative_analysis = primary_report_gen.comparative_analysis_reports(comparison_data[1:])
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison report
        comparison_html = _generate_comparison_html_report(comparative_analysis, comparison_data)
        with open(output_path / "workload_comparison.html", 'w') as f:
            f.write(comparison_html)
        
        # Generate interactive dashboard if requested
        if interactive_dashboard:
            from .reporting.interactive_dashboards import create_workload_comparison_dashboard
            create_workload_comparison_dashboard(
                comparison_data, 
                str(output_path / "interactive_comparison.html")
            )
        
        # Save detailed analysis
        with open(output_path / "comparison_analysis.json", 'w') as f:
            json.dump(comparative_analysis, f, indent=2, default=str)
        
        # Display summary
        _display_comparison_summary(comparative_analysis)
        
        click.echo(f"\n✅ Workload comparison saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error comparing workloads: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results-file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./dashboard', help='Output directory')
@click.option('--workload-type', '-w', 
              type=click.Choice(['web_api', 'batch_processing', 'event_driven', 'scheduled', 'stream_processing', 'generic']),
              default='generic', help='Workload type for specialized dashboard')
@click.option('--include-cost-scenarios', is_flag=True, help='Include cost projection scenarios')
@click.option('--real-time-template', is_flag=True, help='Generate real-time monitoring template')
def dashboard(results_file: str, output_dir: str, workload_type: str, 
              include_cost_scenarios: bool, real_time_template: bool):
    """Generate comprehensive interactive dashboards."""
    try:
        click.echo("📊 Generating Interactive Dashboard")
        click.echo("="*50)
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dashboard
        dashboard_gen = InteractiveDashboard(results)
        
        # Main performance dashboard
        dashboard_gen.create_performance_dashboard(
            str(output_path / "performance_dashboard.html"), 
            workload_type
        )
        click.echo("✅ Generated: performance_dashboard.html")
        
        # Cost analysis dashboard
        scenarios = None
        if include_cost_scenarios:
            scenarios = [
                {'name': 'Low Volume', 'daily_invocations': 1000, 'pattern': 'steady', 'duration_days': 30},
                {'name': 'Medium Volume', 'daily_invocations': 10000, 'pattern': 'bursty', 'duration_days': 30},
                {'name': 'High Volume', 'daily_invocations': 100000, 'pattern': 'seasonal', 'duration_days': 30}
            ]
        
        dashboard_gen.create_cost_analysis_dashboard(
            str(output_path / "cost_dashboard.html"), 
            scenarios
        )
        click.echo("✅ Generated: cost_dashboard.html")
        
        # Cold start analysis dashboard
        dashboard_gen.create_cold_start_dashboard(
            str(output_path / "cold_start_dashboard.html")
        )
        click.echo("✅ Generated: cold_start_dashboard.html")
        
        # Comprehensive dashboard
        dashboard_gen.create_comprehensive_dashboard(
            str(output_path / "comprehensive_dashboard.html"),
            workload_type,
            scenarios
        )
        click.echo("✅ Generated: comprehensive_dashboard.html")
        
        # Real-time monitoring template
        if real_time_template:
            dashboard_gen.create_real_time_monitoring_dashboard(
                str(output_path / "real_time_monitoring.html")
            )
            click.echo("✅ Generated: real_time_monitoring.html")
        
        # Generate index page
        index_html = _generate_dashboard_index(workload_type, include_cost_scenarios, real_time_template)
        with open(output_path / "index.html", 'w') as f:
            f.write(index_html)
        click.echo("✅ Generated: index.html")
        
        click.echo(f"\n🎉 Dashboard suite ready! Open {output_dir}/index.html in your browser")
        
    except Exception as e:
        click.echo(f"❌ Error generating dashboard: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results-file', type=click.Path(exists=True))
@click.option('--format', 'export_format', type=click.Choice(['pdf', 'excel', 'json', 'csv', 'html']),
              default='pdf', help='Export format')
@click.option('--workload-type', '-w', 
              type=click.Choice(['web_api', 'batch_processing', 'event_driven', 'scheduled', 'stream_processing', 'generic']),
              default='generic', help='Workload type for specialized export')
@click.option('--output', '-o', help='Output file path (optional)')
def export(results_file: str, export_format: str, workload_type: str, output: Optional[str]):
    """Export results in various formats (PDF, Excel, etc.)."""
    try:
        click.echo(f"📤 Exporting to {export_format.upper()} format")
        click.echo("="*50)
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Determine output path
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"lambda_analysis_{workload_type}_{timestamp}.{export_format}"
        
        # Export using multi-format exporter
        exporter = MultiFormatExporter(results)
        exporter.export_report(output, export_format, workload_type)
        
        click.echo(f"✅ Report exported to: {output}")
        
        # Display file info
        file_path = Path(output)
        if file_path.exists():
            file_size = file_path.stat().st_size
            click.echo(f"   File size: {file_size:,} bytes")
        
    except Exception as e:
        click.echo(f"❌ Error exporting report: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('function-arn')
@click.option('--days', default=7, help='Number of days to analyze (AWS Cost Explorer)')
@click.option('--output-dir', '-o', default='./cost-explorer', help='Output directory')
def cost_explorer(function_arn: str, days: int, output_dir: str):
    """Integrate with AWS Cost Explorer for historical cost analysis."""
    try:
        click.echo("💳 AWS Cost Explorer Integration")
        click.echo("="*50)
        
        # Note: This would require AWS Cost Explorer API integration
        # For now, we'll create a template implementation
        
        click.echo(f"Analyzing costs for: {function_arn}")
        click.echo(f"Period: Last {days} days")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate cost explorer report template
        cost_report = _generate_cost_explorer_template(function_arn, days)
        
        with open(output_path / "cost_explorer_analysis.html", 'w') as f:
            f.write(cost_report)
        
        click.echo("⚠️  Note: Full Cost Explorer integration requires additional AWS permissions")
        click.echo("   This generates a template for cost analysis integration")
        click.echo(f"\n✅ Cost Explorer template saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error accessing Cost Explorer: {e}", err=True)
        sys.exit(1)


# Helper functions for the new CLI commands
def _interactive_workload_selection() -> str:
    """Interactive workload type selection."""
    workload_options = {
        '1': ('web_api', 'Web API - Low latency REST/GraphQL APIs'),
        '2': ('batch_processing', 'Batch Processing - ETL jobs, data processing'),
        '3': ('event_driven', 'Event-Driven - S3, DynamoDB, SQS triggers'),
        '4': ('scheduled', 'Scheduled - Cron jobs, periodic tasks'),
        '5': ('stream_processing', 'Stream Processing - Kinesis, real-time data')
    }
    
    click.echo("\n🎯 Select your Lambda workload type:")
    for key, (_, description) in workload_options.items():
        click.echo(f"  {key}. {description}")
    
    while True:
        choice = click.prompt("\nEnter your choice (1-5)", type=str)
        if choice in workload_options:
            return workload_options[choice][0]
        click.echo("❌ Invalid choice. Please select 1-5.")


def _get_workload_config(workload_type: str, interactive: bool) -> Dict[str, Any]:
    """Get workload-specific configuration."""
    config = {}
    
    if not interactive:
        return config
    
    click.echo(f"\n⚙️  Configuring {workload_type.replace('_', ' ').title()} optimization:")
    
    if workload_type == 'web_api':
        config['target_latency'] = click.prompt("Target P95 latency (ms)", default=100, type=int)
        config['expected_concurrency'] = click.prompt("Expected peak concurrency", default=10, type=int)
        config['cold_start_tolerance'] = click.prompt("Cold start tolerance (%)", default=5, type=float)
        
    elif workload_type == 'batch_processing':
        config['batch_size'] = click.prompt("Typical batch size", default=1000, type=int)
        config['processing_time'] = click.prompt("Expected processing time (minutes)", default=10, type=int)
        config['cost_priority'] = click.confirm("Prioritize cost over speed?", default=True)
        
    elif workload_type == 'event_driven':
        config['event_rate'] = click.prompt("Events per minute", default=100, type=int)
        config['reliability_target'] = click.prompt("Target success rate (%)", default=99.9, type=float)
        config['dlq_enabled'] = click.confirm("Dead letter queue enabled?", default=True)
        
    elif workload_type == 'scheduled':
        config['frequency'] = click.prompt("Execution frequency (minutes)", default=60, type=int)
        config['time_window'] = click.prompt("Acceptable execution window (minutes)", default=5, type=int)
        config['consistency_priority'] = click.confirm("Prioritize execution consistency?", default=True)
        
    elif workload_type == 'stream_processing':
        config['records_per_second'] = click.prompt("Records per second", default=1000, type=int)
        config['latency_requirement'] = click.prompt("Max processing latency (ms)", default=500, type=int)
        config['parallel_shards'] = click.prompt("Number of parallel shards", default=1, type=int)
    
    return config


def _customize_config_for_workload(base_config: TunerConfig, workload_type: str, 
                                 workload_config: Dict[str, Any], function_arn: str) -> TunerConfig:
    """Customize configuration based on workload type."""
    # Apply workload-specific optimizations
    if workload_type == 'web_api':
        # Focus on lower memory sizes for cost efficiency, but ensure good performance
        memory_sizes = [128, 256, 512, 1024, 1536]
        iterations = 15  # More iterations for statistical significance
        concurrent = 10  # Higher concurrency to test under load
        
    elif workload_type == 'batch_processing':
        # Test higher memory sizes for potential CPU scaling benefits
        memory_sizes = [512, 1024, 2048, 3008]
        iterations = 10
        concurrent = 5  # Lower concurrency as batch jobs are typically sequential
        
    elif workload_type == 'event_driven':
        # Balanced approach with focus on reliability
        memory_sizes = [256, 512, 1024, 1536, 2048]
        iterations = 12
        concurrent = 8
        
    elif workload_type == 'scheduled':
        # Focus on consistency and cost efficiency
        memory_sizes = [128, 256, 512, 1024]
        iterations = 20  # More iterations for consistency analysis
        concurrent = 3  # Lower concurrency as scheduled jobs run independently
        
    elif workload_type == 'stream_processing':
        # Higher memory for stream processing performance
        memory_sizes = [512, 1024, 1536, 2048, 3008]
        iterations = 15
        concurrent = 6
    else:
        # Default balanced configuration
        memory_sizes = base_config.memory_sizes
        iterations = base_config.iterations
        concurrent = base_config.concurrent_executions
    
    return TunerConfig(
        function_arn=function_arn,
        payload=base_config.payload,
        memory_sizes=memory_sizes,
        iterations=iterations,
        strategy='balanced',
        concurrent_executions=concurrent,
        timeout=base_config.timeout,
        dry_run=base_config.dry_run,
        output_dir=base_config.output_dir
    )


def _display_workload_results(analysis: Dict[str, Any], workload_type: str):
    """Display workload-specific results."""
    click.echo(f"\n📋 {workload_type.replace('_', ' ').title()} Analysis Results")
    click.echo("="*60)
    
    key_metrics = analysis.get('key_metrics', {})
    recommendations = analysis.get('recommendations', [])
    
    # Display key metrics
    for metric, value in key_metrics.items():
        formatted_metric = metric.replace('_', ' ').title()
        if isinstance(value, float):
            click.echo(f"{formatted_metric:30} {value:.2f}")
        else:
            click.echo(f"{formatted_metric:30} {value}")
    
    # Display top recommendations
    click.echo("\n💡 Top Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        priority = rec.get('priority', 'medium')
        category = rec.get('category', 'general')
        description = rec.get('description', '')
        click.echo(f"{i}. [{priority.upper()}] {category.title()}: {description}")


def _display_cost_summary(cost_analysis: Dict[str, Any]):
    """Display cost projection summary."""
    click.echo("\n💰 Cost Projection Summary")
    click.echo("="*50)
    
    optimal = cost_analysis.get('optimal_configuration', {})
    projections = cost_analysis.get('projections', {})
    savings = cost_analysis.get('savings_analysis', {})
    
    click.echo(f"Optimal Memory: {optimal.get('memory_mb', 'N/A')}MB")
    click.echo(f"Cost per Invocation: ${optimal.get('avg_cost', 0):.6f}")
    
    if projections:
        click.echo("\nProjected Costs:")
        for scenario, data in projections.items():
            monthly_cost = data.get('monthly_cost', 0)
            click.echo(f"  {scenario:20} ${monthly_cost:.2f}/month")
    
    total_savings = savings.get('total_savings', 0)
    if total_savings > 0:
        click.echo(f"\nTotal Potential Savings: ${total_savings:.2f}")


def _display_comparison_summary(comparison: Dict[str, Any]):
    """Display workload comparison summary."""
    click.echo("\n🔍 Workload Comparison Summary")
    click.echo("="*50)
    
    current = comparison.get('current_workload', {})
    comparisons = comparison.get('comparisons', [])
    insights = comparison.get('cross_workload_insights', [])
    
    click.echo(f"Primary Workload: {current.get('type', 'Unknown')}")
    
    click.echo("\nComparison Results:")
    for comp in comparisons:
        workload_type = comp.get('workload_type', 'Unknown')
        analysis = comp.get('analysis', {})
        key_metrics = analysis.get('key_metrics', {})
        optimal_memory = key_metrics.get('optimal_memory', 'N/A')
        click.echo(f"  {workload_type:20} Optimal: {optimal_memory}MB")
    
    if insights:
        click.echo("\nKey Insights:")
        for insight in insights:
            click.echo(f"  • {insight.get('insight', '')}")


def _generate_cost_html_report(cost_analysis: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> str:
    """Generate HTML cost projection report."""
    template = WorkloadReportTemplates.get_cost_projection_template()
    
    optimal = cost_analysis.get('optimal_configuration', {})
    projections = cost_analysis.get('projections', {})
    savings = cost_analysis.get('savings_analysis', {})
    
    # Generate scenario cards
    scenario_cards = ""
    for scenario_name, projection in projections.items():
        monthly_cost = projection.get('monthly_cost', 0)
        yearly_cost = projection.get('yearly_cost', 0)
        invocations = projection.get('invocations_per_day', 0)
        
        scenario_cards += f"""
        <div class="scenario-card">
            <h3>{scenario_name.replace('_', ' ').title()}</h3>
            <p><strong>Daily Invocations:</strong> {invocations:,}</p>
            <p><strong>Monthly Cost:</strong> ${monthly_cost:.2f}</p>
            <p><strong>Yearly Cost:</strong> ${yearly_cost:.2f}</p>
        </div>
        """
    
    # Generate cost breakdown rows
    cost_rows = ""
    scenario_savings = savings.get('scenario_savings', {})
    for scenario_name, projection in projections.items():
        if scenario_name in scenario_savings:
            savings_data = scenario_savings[scenario_name]
            cost_rows += f"""
            <tr>
                <td>{scenario_name.replace('_', ' ').title()}</td>
                <td>{projection.get('invocations_per_day', 0):,}</td>
                <td>${projection.get('monthly_cost', 0):.2f}</td>
                <td>${savings_data.get('baseline_cost', 0) / 30:.2f}</td>
                <td>${savings_data.get('absolute_savings', 0) / 30:.2f}</td>
                <td>${savings_data.get('absolute_savings', 0) * 12:.2f}</td>
            </tr>
            """
    
    # Generate recommendations
    recommendations = cost_analysis.get('recommendations', [])
    rec_html = ""
    for rec in recommendations:
        rec_html += f"""
        <div class="recommendations">
            <h4>{rec.get('category', 'General').title()}</h4>
            <p>{rec.get('description', '')}</p>
        </div>
        """
    
    return template.format(
        optimal_memory=optimal.get('memory_mb', 'N/A'),
        total_savings=savings.get('total_savings', 0),
        avg_savings_percentage=savings.get('average_savings_percentage', 0),
        scenario_cards=scenario_cards,
        cost_breakdown_rows=cost_rows,
        cost_recommendations=rec_html,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def _generate_comparison_html_report(comparison: Dict[str, Any], comparison_data: List[Dict[str, Any]]) -> str:
    """Generate HTML workload comparison report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workload Performance Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .workload-card {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
            .metric {{ margin: 10px 0; }}
            .metric-label {{ font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>🔍 Workload Performance Comparison</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Comparison Overview</h2>
        <div class="comparison-grid">
    """
    
    # Add workload cards
    for data in comparison_data:
        workload_type = data.get('workload_type', 'Unknown')
        analysis = data.get('analysis', {})
        key_metrics = analysis.get('key_metrics', {})
        
        html_content += f"""
            <div class="workload-card">
                <h3>{workload_type.replace('_', ' ').title()}</h3>
        """
        
        for metric, value in key_metrics.items():
            formatted_metric = metric.replace('_', ' ').title()
            html_content += f"""
                <div class="metric">
                    <span class="metric-label">{formatted_metric}:</span> {value}
                </div>
            """
        
        html_content += "</div>"
    
    html_content += """
        </div>
        
        <h2>Cross-Workload Insights</h2>
    """
    
    # Add insights
    insights = comparison.get('cross_workload_insights', [])
    for insight in insights:
        html_content += f"<p>• {insight.get('insight', '')}</p>"
    
    # Add best practices
    best_practices = comparison.get('best_practices', [])
    if best_practices:
        html_content += "<h2>Best Practices</h2>"
        for practice in best_practices:
            html_content += f"<p><strong>{practice.get('category', 'General').title()}:</strong> {practice.get('practice', '')}</p>"
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content


def _generate_dashboard_index(workload_type: str, include_cost: bool, real_time: bool) -> str:
    """Generate dashboard index page."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lambda Performance Dashboard Suite</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
            .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .dashboard-card {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center; }}
            .dashboard-card:hover {{ background: #f8f9fa; }}
            .dashboard-card a {{ text-decoration: none; color: #333; }}
            .icon {{ font-size: 2em; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Lambda Performance Dashboard Suite</h1>
            <p>Workload Type: {workload_type.replace('_', ' ').title()}</p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="dashboard-grid">
                <div class="dashboard-card">
                    <a href="performance_dashboard.html">
                        <div class="icon">🚀</div>
                        <h3>Performance Dashboard</h3>
                        <p>Comprehensive performance analysis</p>
                    </a>
                </div>
                
                <div class="dashboard-card">
                    <a href="cost_dashboard.html">
                        <div class="icon">💰</div>
                        <h3>Cost Analysis</h3>
                        <p>Cost optimization insights</p>
                    </a>
                </div>
                
                <div class="dashboard-card">
                    <a href="cold_start_dashboard.html">
                        <div class="icon">❄️</div>
                        <h3>Cold Start Analysis</h3>
                        <p>Cold start performance metrics</p>
                    </a>
                </div>
                
                <div class="dashboard-card">
                    <a href="comprehensive_dashboard.html">
                        <div class="icon">📈</div>
                        <h3>Comprehensive View</h3>
                        <p>All-in-one performance dashboard</p>
                    </a>
                </div>
                
                {f'''
                <div class="dashboard-card">
                    <a href="real_time_monitoring.html">
                        <div class="icon">🔄</div>
                        <h3>Real-Time Monitoring</h3>
                        <p>Live performance monitoring template</p>
                    </a>
                </div>
                ''' if real_time else ''}
            </div>
            
            <footer style="text-align: center; margin-top: 40px; color: #666;">
                <p>🛠️ AWS Lambda Performance Tuner - Dashboard Suite</p>
            </footer>
        </div>
    </body>
    </html>
    """


def _generate_cost_explorer_template(function_arn: str, days: int) -> str:
    """Generate Cost Explorer integration template."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AWS Cost Explorer Integration</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .integration-info {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .code-block {{ background: #f5f5f5; padding: 15px; border-radius: 5px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>💳 AWS Cost Explorer Integration</h1>
        
        <div class="integration-info">
            <h3>Function Analysis</h3>
            <p><strong>Function ARN:</strong> {function_arn}</p>
            <p><strong>Analysis Period:</strong> Last {days} days</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <h2>Integration Setup</h2>
        <p>To integrate with AWS Cost Explorer, you'll need:</p>
        <ul>
            <li>AWS Cost Explorer API permissions</li>
            <li>IAM role with ce:GetCostAndUsage permission</li>
            <li>Function resource tags for accurate cost attribution</li>
        </ul>
        
        <h3>Required IAM Policy</h3>
        <div class="code-block">
{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Action": [
                "ce:GetCostAndUsage",
                "ce:GetUsageReport",
                "ce:GetDimensionValues"
            ],
            "Resource": "*"
        }}
    ]
}}
        </div>
        
        <h3>AWS CLI Example</h3>
        <div class="code-block">
aws ce get-cost-and-usage \\
    --time-period Start=2024-01-01,End=2024-01-31 \\
    --granularity DAILY \\
    --metrics BlendedCost \\
    --group-by Type=DIMENSION,Key=SERVICE
        </div>
        
        <h2>Cost Analysis Recommendations</h2>
        <ul>
            <li>Tag your Lambda functions with cost center information</li>
            <li>Monitor costs daily to identify trends</li>
            <li>Set up billing alerts for unexpected cost increases</li>
            <li>Use this tuner's recommendations to optimize memory allocation</li>
        </ul>
        
        <p><em>Note: This is a template for Cost Explorer integration. Actual implementation requires additional AWS SDK integration.</em></p>
    </body>
    </html>
    """


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
