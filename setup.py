from setuptools import setup, find_packages

setup(
    name="ai-risk-manager",
    version="0.1.0",
    author="DAX",
    packages=find_packages(),
    install_requires=[
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "river>=0.21.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "faker>=19.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
)
