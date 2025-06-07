from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "boto3>=1.28.0",
    "click>=8.1.0",
    "tabulate>=0.9.0",
    "jinja2>=3.1.0",
    "pyyaml>=6.0",
    "colorama>=0.4.0",
    "rich>=13.5.0",
    "tqdm>=4.65.0",
    "numpy>=1.24.0",
    "pandas>=1.5.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.0.0",
    "aiohttp>=3.8.0",
    "asyncio-throttle>=1.0.0",
    "scikit-learn>=1.3.0"
]

setup(
    name="aws-lambda-tuner",
    version="2.0.0",
    author="kencytbb",
    author_email="",
    description="AWS Lambda performance tuner for cost and speed optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kencytbb/aws-lambda-tuner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aws-lambda-tuner=aws_lambda_tuner.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)