"""Setup configuration for House Prices ML Project."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="house-prices-ml",
    version="1.0.0",
    author="ML Team",
    description="House Prices - Advanced Regression Techniques ML Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/house-prices-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.0",
        "catboost>=1.2.0",
        "xgboost>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
            "pre-commit>=3.6.0",
        ],
    },
)
