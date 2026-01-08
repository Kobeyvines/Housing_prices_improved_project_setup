.PHONY: help install dev-install lint format test train predict clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make dev-install   - Install development dependencies (lint, test, pre-commit)"
	@echo "  make lint          - Run flake8 and black checks"
	@echo "  make format        - Format code with black and sort imports"
	@echo "  make test          - Run pytest tests"
	@echo "  make train         - Run training pipeline"
	@echo "  make predict       - Run inference pipeline"
	@echo "  make clean         - Remove cache, .pyc, build artifacts"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install -r requirements-prod.txt

dev-install:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	flake8 src/ tests/ entrypoint/
	black --check src/ tests/ entrypoint/

format:
	black src/ tests/ entrypoint/
	isort src/ tests/ entrypoint/

test:
	pytest tests/ -v --tb=short

train:
	python entrypoint/train.py

predict:
	python entrypoint/inference.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/
	rm -rf .dvc/cache

docker-build:
	docker build -t california-housing:latest .

docker-run:
	docker run -it california-housing:latest
