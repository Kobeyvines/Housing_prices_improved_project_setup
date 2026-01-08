# ML Project Template

<!-- CI badge: update <OWNER> and <REPO> to use your repository -->
![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)

A reference implementation of a **well-structured Machine Learning project**.

This repository shows how to organize ML projects **like a professional software system**: modular, reproducible, and production-ready.

---

## Quick Start: Using as a Template

```bash
# Clone the template
git clone https://github.com/Kobeyvines/kobeyvines_machinelearning_template.git my-new-project
cd my-new-project

# Remove the template's remote and add your own
git remote remove origin
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git

# Install development environment
make dev-install

# Set up DVC and Git LFS
git lfs install
dvc init

# Start developing!
```

### Common Commands

```bash
make help          # Show all available commands
make dev-install   # Install dependencies + pre-commit hooks
make lint          # Check code style
make format        # Auto-format code
make test          # Run tests
make train         # Run training pipeline
make predict       # Run inference
make clean         # Remove cache files
```

---

## Setup Guide

### 1. Install Dependencies
```bash
make dev-install
```

### 2. Data Management with DVC
```bash
git lfs install
dvc add data/01-raw/
git add data/01-raw/.gitignore data/01-raw.dvc
git commit -m "Track raw data with DVC"
```

### 3. Configure GitHub Actions
Set these secrets in your GitHub repository:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub token
- `DOCKER_REPO`: Your Docker registry path

### 4. Pre-commit Hooks
Automatically runs on every commit:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **nbstripout**: Clean notebook outputs

---

## Project Structure

```
.
├── data/                      # Data folder (tracked with DVC)
│   ├── 01-raw/               # Raw data from source
│   ├── 02-preprocessed/      # Cleaned/processed data
│   ├── 03-features/          # Feature-engineered data
│   └── 04-predictions/       # Model predictions
│
├── notebooks/                 # Jupyter notebooks for EDA & experiments
│   ├── EDA.ipynb
│   └── Baseline.ipynb
│
├── src/                       # Production code
│   ├── pipelines/            # ML pipelines
│   │   ├── training_pipeline.py
│   │   ├── inference_pipeline.py
│   │   └── feature_eng_pipeline.py
│   └── utils.py              # Utility functions
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_training.py
│
├── entrypoint/               # Entry points for training/inference
│   ├── train.py
│   └── inference.py
│
├── config/                   # Configuration files
│   ├── local.yaml           # Local development config
│   └── prod.yaml            # Production config
│
├── .github/workflows/        # GitHub Actions CI/CD
│   └── ci.yml
│
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .gitignore               # Git ignore rules
├── Dockerfile               # Docker container
├── docker-compose.yml       # Docker compose for local dev
├── Makefile                 # Common commands
├── requirements-prod.txt    # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

---

## Key Features

- ✅ **GitHub Actions CI/CD** - Automated testing, linting, and Docker builds
- ✅ **DVC** - Version control for datasets and ML pipelines
- ✅ **Git LFS** - Store large files efficiently
- ✅ **Pre-commit hooks** - Catch issues before committing
- ✅ **Docker** - Containerize your ML application
- ✅ **Makefile** - Simple commands for common tasks
- ✅ **Config management** - Local vs production configs

---

## Requirements

- Python 3.11+
- Git & Git LFS
- Docker (optional, for containerization)

---

## How to Contribute

This is a template repository. Feel free to modify and customize for your needs!
