# AWS Lambda Performance Tuner (Python)

A comprehensive Python tool for optimizing AWS Lambda functions for cost and performance. This tool automatically tests different memory configurations and provides detailed performance reports to help you make data-driven decisions.

## 🚀 Features

- **AWS Lambda Optimization**: Works with any Lambda function regardless of runtime or language
- **Cost & Performance Analysis**: Find the optimal balance between execution time and cost
- **Automated Testing**: Run multiple iterations with different memory configurations
- **Detailed Reports**: Get comprehensive performance analytics after each tuning session
- **Visual Reports**: Generate charts and graphs to visualize performance trends
- **Export Results**: Save tuning results in JSON, CSV, or HTML formats
- **CLI & Programmatic API**: Use via command line or integrate into your applications

## 🛠 Installation

### Prerequisites

- Python 3.8+
- AWS CLI configured with appropriate permissions
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Tuning

```bash
# Tune a single Lambda function
python -m aws_lambda_tuner tune \
  --function-arn arn:aws:lambda:us-east-1:123456789012:function:my-function
```

### 2. Generate Configuration File

```bash
# Generate a sample configuration
python -m aws_lambda_tuner init
```

### 3. Run with Configuration

```bash
python -m aws_lambda_tuner tune --config tuner.config.json
```

## 📋 Usage Examples

### Example 1: Basic Tuning

```bash
python -m aws_lambda_tuner tune \
  --function-arn arn:aws:lambda:us-east-1:123456789012:function:api-handler \
  --payload '{"httpMethod": "GET", "path": "/users"}' \
  --memory-sizes 256,512,1024 \
  --iterations 20 \
  --strategy speed
```

### Example 2: Programmatic Usage

```python
from aws_lambda_tuner import TunerOrchestrator
from aws_lambda_tuner.config import TunerConfig

config = TunerConfig(
    function_arn='arn:aws:lambda:us-east-1:123456789012:function:my-function',
    memory_sizes=[256, 512, 1024, 2048],
    iterations=10,
    strategy='balanced'
)

orchestrator = TunerOrchestrator(config)
results = await orchestrator.run_with_reporting()
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=aws_lambda_tuner

# Run specific test file
python -m pytest tests/test_tuner.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [AWS Lambda Power Tuning](https://github.com/alexcasalboni/aws-lambda-power-tuning) by Alex Casalboni
- Built with ❤️ for the serverless community

## 📞 Support

- 🐛 [Report bugs](https://github.com/kencytbb/aws-lambda-tunner/issues)
- 💡 [Request features](https://github.com/kencytbb/aws-lambda-tunner/issues)
- 📖 [Documentation](https://github.com/kencytbb/aws-lambda-tunner/wiki)
- 💬 [Discussions](https://github.com/kencytbb/aws-lambda-tunner/discussions)