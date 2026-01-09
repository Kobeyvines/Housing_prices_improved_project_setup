FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ /app/src/
COPY entrypoint/ /app/entrypoint/
COPY config/ /app/config/
COPY data/ /app/data/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create models directory
RUN mkdir -p /app/models

# Default command for training
CMD ["python", "entrypoint/train.py", "--config", "config/prod.yaml"]
