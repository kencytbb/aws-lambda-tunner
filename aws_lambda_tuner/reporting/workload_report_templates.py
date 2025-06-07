"""
Workload-specific report templates for AWS Lambda Tuner.
Provides HTML and JSON templates for different workload types.
"""

from typing import Dict, Any, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class WorkloadReportTemplates:
    """Templates for generating workload-specific reports."""
    
    @staticmethod
    def get_web_api_html_template() -> str:
        """HTML template for web API workload reports."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Web API Lambda Performance Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ 
            text-align: center; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
            margin-bottom: 30px; 
        }}
        .summary {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0; 
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center; 
            border-left: 4px solid #3498db; 
        }}
        .metric-value {{ 
            font-size: 2em; 
            font-weight: bold; 
            color: #2c3e50; 
            margin: 10px 0; 
        }}
        .metric-label {{ 
            font-size: 0.9em; 
            color: #7f8c8d; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
        }}
        .recommendations {{ 
            background: #d5f4e6; 
            border-left: 4px solid #27ae60; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        .warning {{ 
            background: #fdf2e9; 
            border-left: 4px solid #e67e22; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        .alert {{ 
            background: #fdebee; 
            border-left: 4px solid #e74c3c; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        .latency-chart {{ 
            background: white; 
            border: 1px solid #bdc3c7; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 20px 0; 
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
            background: white; 
            border-radius: 8px; 
            overflow: hidden; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }}
        th, td {{ 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid #ecf0f1; 
        }}
        th {{ 
            background: #34495e; 
            color: white; 
            font-weight: bold; 
            text-transform: uppercase; 
            font-size: 0.9em; 
        }}
        tr:hover {{ background-color: #f8f9fa; }}
        .optimal-row {{ background-color: #d5f4e6 !important; }}
        .footer {{ 
            text-align: center; 
            margin-top: 40px; 
            padding-top: 20px; 
            border-top: 1px solid #ecf0f1; 
            color: #7f8c8d; 
        }}
        .priority-high {{ color: #e74c3c; font-weight: bold; }}
        .priority-medium {{ color: #f39c12; font-weight: bold; }}
        .priority-low {{ color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Web API Lambda Performance Analysis</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Optimal Memory</div>
                    <div class="metric-value">{optimal_memory}MB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">P95 Latency</div>
                    <div class="metric-value">{p95_latency:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">P99 Latency</div>
                    <div class="metric-value">{p99_latency:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cold Start Rate</div>
                    <div class="metric-value">{cold_start_percentage:.1f}%</div>
                </div>
            </div>
        </div>

        <h2>🎯 Web API Specific Metrics</h2>
        <div class="latency-chart">
            <h3>Latency Performance</h3>
            <p><strong>Cold Start Impact:</strong> {cold_start_avg_penalty:.1f}ms average penalty</p>
            <p><strong>Recommendation:</strong> {latency_recommendation}</p>
        </div>

        {cold_start_section}

        <h2>📊 Configuration Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Memory (MB)</th>
                    <th>Avg Latency (ms)</th>
                    <th>P95 Latency (ms)</th>
                    <th>P99 Latency (ms)</th>
                    <th>Cold Start Rate (%)</th>
                    <th>Cost per Request ($)</th>
                    <th>API Gateway Score</th>
                </tr>
            </thead>
            <tbody>
                {configuration_rows}
            </tbody>
        </table>

        <h2>💡 Recommendations</h2>
        {recommendations_section}

        <h2>🔧 Scaling Recommendations</h2>
        {scaling_recommendations}

        <div class="footer">
            <p>Report generated on {timestamp}</p>
            <p>🛠️ AWS Lambda Performance Tuner - Web API Optimization</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def get_batch_processing_html_template() -> str:
        """HTML template for batch processing workload reports."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Batch Processing Lambda Performance Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ 
            text-align: center; 
            border-bottom: 3px solid #9b59b6; 
            padding-bottom: 10px; 
            margin-bottom: 30px; 
        }}
        .summary {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0; 
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center; 
            border-left: 4px solid #9b59b6; 
        }}
        .metric-value {{ 
            font-size: 2em; 
            font-weight: bold; 
            color: #2c3e50; 
            margin: 10px 0; 
        }}
        .metric-label {{ 
            font-size: 0.9em; 
            color: #7f8c8d; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
        }}
        .throughput-section {{ 
            background: #e8f5e8; 
            border-left: 4px solid #27ae60; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        .cost-efficiency {{ 
            background: #fff3cd; 
            border-left: 4px solid #ffc107; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
            background: white; 
            border-radius: 8px; 
            overflow: hidden; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }}
        th, td {{ 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid #ecf0f1; 
        }}
        th {{ 
            background: #6c5ce7; 
            color: white; 
            font-weight: bold; 
            text-transform: uppercase; 
            font-size: 0.9em; 
        }}
        .optimal-row {{ background-color: #d5f4e6 !important; }}
        .footer {{ 
            text-align: center; 
            margin-top: 40px; 
            padding-top: 20px; 
            border-top: 1px solid #ecf0f1; 
            color: #7f8c8d; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚙️ Batch Processing Lambda Performance Analysis</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Optimal Memory</div>
                    <div class="metric-value">{optimal_memory}MB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Duration</div>
                    <div class="metric-value">{avg_duration:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cost per Execution</div>
                    <div class="metric-value">${cost_per_execution:.6f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Throughput Improvement</div>
                    <div class="metric-value">{throughput_improvement:.1f}%</div>
                </div>
            </div>
        </div>

        <div class="throughput-section">
            <h3>🔄 Throughput Analysis</h3>
            <p><strong>Performance Optimization:</strong> {throughput_recommendation}</p>
            <p><strong>Cost Efficiency Ratio:</strong> {cost_efficiency_ratio:.2f}x better than baseline</p>
        </div>

        <div class="cost-efficiency">
            <h3>💰 Cost Efficiency</h3>
            <p><strong>Recommendation:</strong> {cost_efficiency_recommendation}</p>
            <p><strong>Batch Size Optimization:</strong> Consider processing larger batches for better cost efficiency</p>
        </div>

        <h2>📊 Configuration Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Memory (MB)</th>
                    <th>Avg Duration (ms)</th>
                    <th>Min Duration (ms)</th>
                    <th>Max Duration (ms)</th>
                    <th>Cost per Execution ($)</th>
                    <th>Throughput Score</th>
                    <th>Efficiency Ratio</th>
                </tr>
            </thead>
            <tbody>
                {configuration_rows}
            </tbody>
        </table>

        <h2>💡 Batch Processing Recommendations</h2>
        {recommendations_section}

        <h2>🔧 Scaling Recommendations</h2>
        {scaling_recommendations}

        <div class="footer">
            <p>Report generated on {timestamp}</p>
            <p>⚙️ AWS Lambda Performance Tuner - Batch Processing Optimization</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def get_event_driven_html_template() -> str:
        """HTML template for event-driven workload reports."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Event-Driven Lambda Performance Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            text-align: center; 
            border-bottom: 3px solid #e74c3c; 
            padding-bottom: 10px; 
            margin-bottom: 30px; 
            color: #2c3e50; 
        }}
        .summary {{ 
            background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0; 
        }}
        .reliability-section {{ 
            background: #e3f2fd; 
            border-left: 4px solid #2196f3; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        .event-processing {{ 
            background: #f3e5f5; 
            border-left: 4px solid #9c27b0; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 8px 8px 0; 
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center; 
            border-left: 4px solid #e74c3c; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Event-Driven Lambda Performance Analysis</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Optimal Memory</div>
                    <div class="metric-value">{optimal_memory}MB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Processing Time</div>
                    <div class="metric-value">{avg_processing_time:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Event Success Rate</div>
                    <div class="metric-value">{event_success_rate:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Memory Utilization</div>
                    <div class="metric-value">{memory_utilization:.1f}%</div>
                </div>
            </div>
        </div>

        <div class="reliability-section">
            <h3>🛡️ Reliability Analysis</h3>
            <p><strong>Event Processing:</strong> {reliability_recommendation}</p>
            <p><strong>Error Handling:</strong> {error_handling_recommendation}</p>
        </div>

        <div class="event-processing">
            <h3>🔄 Event Processing Efficiency</h3>
            <p><strong>Processing Rate:</strong> Optimized for {optimal_memory}MB configuration</p>
            <p><strong>Backlog Management:</strong> Monitor queue depths and processing rates</p>
        </div>

        {configuration_table}

        {recommendations_section}

        {scaling_recommendations}

        <div class="footer">
            <p>Report generated on {timestamp}</p>
            <p>⚡ AWS Lambda Performance Tuner - Event-Driven Optimization</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def get_generic_html_template() -> str:
        """Generic HTML template for any workload type."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Lambda Performance Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            text-align: center; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
            margin-bottom: 30px; 
            color: #2c3e50; 
        }}
        .summary {{ 
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0; 
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center; 
            border-left: 4px solid #3498db; 
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
            background: white; 
            border-radius: 8px; 
            overflow: hidden; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }}
        th {{ 
            background: #3498db; 
            color: white; 
            padding: 12px 15px; 
            font-weight: bold; 
        }}
        td {{ 
            padding: 12px 15px; 
            border-bottom: 1px solid #ecf0f1; 
        }}
        .optimal-row {{ background-color: #d5f4e6 !important; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Lambda Performance Analysis</h1>
        
        <div class="summary">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Optimal Memory</div>
                    <div class="metric-value">{optimal_memory}MB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Performance Gain</div>
                    <div class="metric-value">{performance_gain:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cost Savings</div>
                    <div class="metric-value">{cost_savings:.1f}%</div>
                </div>
            </div>
        </div>

        {configuration_table}
        {recommendations_section}

        <div class="footer">
            <p>Report generated on {timestamp}</p>
            <p>📊 AWS Lambda Performance Tuner</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def get_workload_json_template(workload_type: str) -> Dict[str, Any]:
        """Get JSON template structure for workload-specific reports."""
        base_template = {
            "metadata": {
                "workload_type": workload_type,
                "report_version": "2.0",
                "generated_at": None,
                "tuner_version": "1.0.0"
            },
            "summary": {
                "optimal_configuration": {},
                "key_metrics": {},
                "performance_summary": {}
            },
            "detailed_analysis": {
                "configurations": [],
                "workload_specific_metrics": {},
                "comparative_analysis": {}
            },
            "recommendations": {
                "immediate_actions": [],
                "optimization_opportunities": [],
                "scaling_recommendations": [],
                "cost_optimization": []
            },
            "visualizations": {
                "charts_generated": [],
                "dashboard_urls": []
            }
        }
        
        # Add workload-specific sections
        if workload_type == "web_api":
            base_template["detailed_analysis"]["latency_analysis"] = {}
            base_template["detailed_analysis"]["cold_start_impact"] = {}
            base_template["api_gateway_integration"] = {}
            
        elif workload_type == "batch_processing":
            base_template["detailed_analysis"]["throughput_analysis"] = {}
            base_template["detailed_analysis"]["cost_efficiency"] = {}
            base_template["batch_optimization"] = {}
            
        elif workload_type == "event_driven":
            base_template["detailed_analysis"]["event_processing_stats"] = {}
            base_template["detailed_analysis"]["reliability_metrics"] = {}
            base_template["event_source_configuration"] = {}
            
        elif workload_type == "scheduled":
            base_template["detailed_analysis"]["execution_consistency"] = {}
            base_template["detailed_analysis"]["schedule_optimization"] = {}
            base_template["cron_recommendations"] = {}
            
        elif workload_type == "stream_processing":
            base_template["detailed_analysis"]["stream_metrics"] = {}
            base_template["detailed_analysis"]["parallelization_analysis"] = {}
            base_template["stream_configuration"] = {}
        
        return base_template

    @staticmethod
    def format_workload_report(workload_analysis: Dict[str, Any], template_type: str = "html") -> str:
        """
        Format a workload analysis into a report using appropriate template.
        
        Args:
            workload_analysis: Analysis results from report service
            template_type: "html" or "json"
            
        Returns:
            Formatted report string
        """
        try:
            workload_type = workload_analysis.get('workload_type', 'generic')
            key_metrics = workload_analysis.get('key_metrics', {})
            recommendations = workload_analysis.get('recommendations', [])
            scaling_recs = workload_analysis.get('scaling_recommendations', [])
            
            if template_type == "json":
                template = WorkloadReportTemplates.get_workload_json_template(workload_type)
                template["metadata"]["generated_at"] = datetime.now().isoformat()
                template["summary"]["key_metrics"] = key_metrics
                template["recommendations"]["immediate_actions"] = recommendations
                template["recommendations"]["scaling_recommendations"] = scaling_recs
                template["detailed_analysis"]["workload_specific_metrics"] = workload_analysis
                return json.dumps(template, indent=2)
            
            # HTML formatting
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format recommendations
            rec_html = ""
            for rec in recommendations:
                priority_class = f"priority-{rec.get('priority', 'medium')}"
                rec_html += f"""
                <div class="recommendations">
                    <h4 class="{priority_class}">{rec.get('category', 'General').title()}</h4>
                    <p>{rec.get('description', '')}</p>
                </div>
                """
            
            # Format scaling recommendations
            scaling_html = ""
            for scale_rec in scaling_recs:
                scaling_html += f"""
                <div class="recommendations">
                    <h4>{scale_rec.get('metric', 'Scaling')}</h4>
                    <p>{scale_rec.get('recommendation', '')}</p>
                </div>
                """
            
            if workload_type == "web_api":
                template = WorkloadReportTemplates.get_web_api_html_template()
                
                # Format cold start section
                cold_start_section = ""
                if key_metrics.get('cold_start_percentage', 0) > 10:
                    cold_start_section = """
                    <div class="alert">
                        <h3>⚠️ Cold Start Alert</h3>
                        <p>High cold start rate detected. Consider implementing provisioned concurrency.</p>
                    </div>
                    """
                
                return template.format(
                    optimal_memory=key_metrics.get('optimal_memory', 'N/A'),
                    p95_latency=key_metrics.get('p95_latency', 0),
                    p99_latency=key_metrics.get('p99_latency', 0),
                    cold_start_percentage=key_metrics.get('cold_start_percentage', 0),
                    cold_start_avg_penalty=key_metrics.get('cold_start_avg_penalty', 0),
                    latency_recommendation="Optimize for sub-100ms P95 latency",
                    cold_start_section=cold_start_section,
                    configuration_rows="<!-- Configuration data would be inserted here -->",
                    recommendations_section=rec_html,
                    scaling_recommendations=scaling_html,
                    timestamp=timestamp
                )
                
            elif workload_type == "batch_processing":
                template = WorkloadReportTemplates.get_batch_processing_html_template()
                return template.format(
                    optimal_memory=key_metrics.get('optimal_memory', 'N/A'),
                    avg_duration=key_metrics.get('avg_duration', 0),
                    cost_per_execution=key_metrics.get('cost_per_execution', 0),
                    throughput_improvement=key_metrics.get('throughput_improvement', 0),
                    cost_efficiency_ratio=key_metrics.get('cost_efficiency_ratio', 1.0),
                    throughput_recommendation="Optimize batch size and concurrency",
                    cost_efficiency_recommendation="Balance memory allocation with processing time",
                    configuration_rows="<!-- Configuration data would be inserted here -->",
                    recommendations_section=rec_html,
                    scaling_recommendations=scaling_html,
                    timestamp=timestamp
                )
                
            elif workload_type == "event_driven":
                template = WorkloadReportTemplates.get_event_driven_html_template()
                return template.format(
                    optimal_memory=key_metrics.get('optimal_memory', 'N/A'),
                    avg_processing_time=key_metrics.get('avg_processing_time', 0),
                    event_success_rate=key_metrics.get('event_success_rate', 0),
                    memory_utilization=key_metrics.get('memory_utilization', 0),
                    reliability_recommendation="Ensure consistent event processing",
                    error_handling_recommendation="Implement proper error handling and DLQ",
                    configuration_table="<!-- Configuration table would be inserted here -->",
                    recommendations_section=rec_html,
                    scaling_recommendations=scaling_html,
                    timestamp=timestamp
                )
            else:
                # Generic template
                template = WorkloadReportTemplates.get_generic_html_template()
                return template.format(
                    optimal_memory=key_metrics.get('optimal_memory', 'N/A'),
                    performance_gain=key_metrics.get('performance_gain', 0),
                    cost_savings=key_metrics.get('cost_savings', 0),
                    configuration_table="<!-- Configuration table would be inserted here -->",
                    recommendations_section=rec_html,
                    timestamp=timestamp
                )
                
        except Exception as e:
            logger.error(f"Error formatting workload report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    @staticmethod
    def get_cost_projection_template() -> str:
        """Template for cost projection reports."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Lambda Cost Projection Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .cost-summary {{ background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .projection-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .scenario-card {{ background: white; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
        .savings {{ color: #27ae60; font-weight: bold; }}
        .cost-increase {{ color: #e74c3c; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>💰 Lambda Cost Projection Analysis</h1>
    
    <div class="cost-summary">
        <h2>Cost Optimization Summary</h2>
        <p><strong>Optimal Configuration:</strong> {optimal_memory}MB</p>
        <p><strong>Total Projected Savings:</strong> <span class="savings">${total_savings:.2f}</span></p>
        <p><strong>Average Savings Percentage:</strong> {avg_savings_percentage:.1f}%</p>
    </div>

    <h2>Scenario Projections</h2>
    <div class="projection-grid">
        {scenario_cards}
    </div>

    <h2>Detailed Cost Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Daily Invocations</th>
                <th>Monthly Cost (Optimized)</th>
                <th>Monthly Cost (Baseline)</th>
                <th>Monthly Savings</th>
                <th>Yearly Savings</th>
            </tr>
        </thead>
        <tbody>
            {cost_breakdown_rows}
        </tbody>
    </table>

    <h2>Cost Optimization Recommendations</h2>
    {cost_recommendations}

    <div style="margin-top: 40px; text-align: center; color: #666;">
        <p>Report generated on {timestamp}</p>
    </div>
</body>
</html>
"""


class ReportFormatter:
    """Utility class for formatting report data."""
    
    @staticmethod
    def format_configuration_table_rows(configurations: List[Dict[str, Any]], 
                                      optimal_memory: int) -> str:
        """Format configuration data into HTML table rows."""
        rows_html = ""
        
        for config in configurations:
            memory_mb = config.get('memory_mb', 0)
            is_optimal = memory_mb == optimal_memory
            row_class = 'class="optimal-row"' if is_optimal else ''
            
            # Extract metrics with defaults
            stats = config.get('statistics', {})
            duration_stats = stats.get('duration', {})
            cost_stats = stats.get('cost', {})
            
            avg_duration = duration_stats.get('mean', 0)
            min_duration = duration_stats.get('min', 0)
            max_duration = duration_stats.get('max', 0)
            p95_duration = duration_stats.get('p95', 0)
            avg_cost = cost_stats.get('mean', 0)
            success_rate = config.get('success_rate', 0)
            
            rows_html += f"""
            <tr {row_class}>
                <td>{memory_mb}MB</td>
                <td>{avg_duration:.2f}</td>
                <td>{min_duration:.2f}</td>
                <td>{max_duration:.2f}</td>
                <td>{p95_duration:.2f}</td>
                <td>${avg_cost:.6f}</td>
                <td>{success_rate:.1f}%</td>
            </tr>
            """
        
        return rows_html

    @staticmethod
    def format_cost_projection_scenarios(projections: Dict[str, Any]) -> str:
        """Format cost projection scenarios into HTML cards."""
        cards_html = ""
        
        for scenario_name, projection in projections.items():
            monthly_cost = projection.get('monthly_cost', 0)
            yearly_cost = projection.get('yearly_cost', 0)
            invocations_per_day = projection.get('invocations_per_day', 0)
            pattern_impact = projection.get('pattern_impact', 1.0)
            
            cards_html += f"""
            <div class="scenario-card">
                <h3>{scenario_name.replace('_', ' ').title()}</h3>
                <p><strong>Daily Invocations:</strong> {invocations_per_day:,}</p>
                <p><strong>Monthly Cost:</strong> ${monthly_cost:.2f}</p>
                <p><strong>Yearly Cost:</strong> ${yearly_cost:.2f}</p>
                <p><strong>Pattern Impact:</strong> {pattern_impact:.1%} adjustment</p>
            </div>
            """
        
        return cards_html