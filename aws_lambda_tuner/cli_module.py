"""
Command Line Interface for AWS Lambda Tuner.
"""

import click
import asyncio
import json
import sys
from typing import Optional, List
import logging
from pathlib import Path

from .config_module import TunerConfig, ConfigManager
from .orchestrator_module import TunerOrchestrator
from .report_service import ReportGenerator
from .visualization_module import VisualizationEngine
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


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
