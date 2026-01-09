"""
QUICK START GUIDE
================

This project implements a complete House Prices regression ML system.
Follow these steps to get started.

## 1. INSTALL DEPENDENCIES

### Option A: Local Setup (Recommended for development)

```bash
# Install production dependencies
pip install -r requirements-prod.txt

# OR install development dependencies (includes testing, linting, etc.)
pip install -r requirements-dev.txt
```

### Option B: Docker Setup

```bash
# Build Docker image
docker build -t house-prices-ml .

# Run training in Docker
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/data:/app/data \
           house-prices-ml python entrypoint/train.py

# Run inference in Docker
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/data:/app/data \
           house-prices-ml python entrypoint/inference.py
```

## 2. GENERATE SAMPLE DATA (Optional - for testing)

If you don't have real data, generate synthetic data for testing:

```bash
python generate_sample_data.py
```

This creates:
- data/01-raw/train.csv (training data with target)
- data/01-raw/test.csv (test data without target)

## 3. RUN THE PIPELINE

### Training

Train the ML model:

```bash
# Using local config
python entrypoint/train.py --config config/local.yaml

# Using production config
python entrypoint/train.py --config config/prod.yaml
```

Output:
- Trained model: `models/model.pkl`
- Scaler: `models/scaler.pkl`
- Feature names: `models/feature_names.pkl`
- Metrics: `models/metrics.pkl`

### Inference

Generate predictions on test data:

```bash
# Make predictions
python entrypoint/inference.py --config config/local.yaml

# Predictions saved to: data/04-predictions/submission.csv
```

## 4. USING MAKE COMMANDS (If make is installed)

```bash
# Show all available commands
make help

# Install dependencies
make dev-install

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Train model
make train

# Run inference
make predict

# Clean cache and artifacts
make clean
```

## 5. RUN TESTS

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_training.py::TestTrainingPipeline -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 6. PROJECT STRUCTURE

```
.
├── src/
│   ├── utils.py                    # Utility functions
│   └── pipelines/
│       ├── feature_eng_pipeline.py # Feature engineering
│       ├── training_pipeline.py    # Model training
│       └── inference_pipeline.py   # Predictions
├── entrypoint/
│   ├── train.py                    # Training entry point
│   └── inference.py                # Inference entry point
├── config/
│   ├── local.yaml                  # Development config
│   └── prod.yaml                   # Production config
├── tests/
│   └── test_training.py            # Unit tests
├── data/
│   ├── 01-raw/                     # Raw data
│   ├── 02-preprocessed/            # Preprocessed data
│   ├── 03-features/                # Feature-engineered data
│   └── 04-predictions/             # Model predictions
├── models/                         # Trained model artifacts
├── requirements-prod.txt           # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose setup
├── Makefile                        # Common commands
├── pytest.ini                      # Pytest configuration
└── README.md                       # Project documentation
```

## 7. CONFIGURATION

### Local Development (config/local.yaml)
- Uses RandomForest with 100 estimators
- 20% validation split
- Suitable for quick experimentation

### Production (config/prod.yaml)
- Uses RandomForest with 200 estimators
- 15% validation split
- More conservative settings for stability

Edit config files to adjust:
- Data paths
- Model type and hyperparameters
- Preprocessing settings
- Evaluation metrics

## 8. KEY FEATURES

✓ **Feature Engineering Pipeline**
  - Handles missing values (median/mode imputation)
  - One-hot encodes categorical variables
  - Creates polynomial features
  - Removes outliers using Z-score

✓ **Training Pipeline**
  - Loads and preprocesses data
  - Supports multiple model types (RandomForest, Ridge)
  - Automatic train/validation split
  - Feature scaling with StandardScaler
  - Comprehensive evaluation metrics (MAE, RMSE, R²)
  - Saves model artifacts for later use

✓ **Inference Pipeline**
  - Loads trained model and artifacts
  - Applies same preprocessing as training
  - Generates predictions on test data
  - Exports results to CSV

✓ **Testing**
  - Unit tests for utilities and pipelines
  - Integration tests for data pipeline
  - Pytest with coverage reporting

## 9. TYPICAL WORKFLOW

### First Time Setup

```bash
# 1. Clone/download the project
cd house-prices-ml

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Generate sample data (optional)
python generate_sample_data.py

# 4. Run tests to verify setup
pytest tests/ -v

# 5. Train model
python entrypoint/train.py --config config/local.yaml

# 6. Run inference
python entrypoint/inference.py --config config/local.yaml

# 7. Check results
head data/04-predictions/submission.csv
```

### Iterative Development

```bash
# Make changes to code...

# Format code
make format

# Run linting
make lint

# Run tests
make test

# Train with updated code
make train

# Check predictions
make predict
```

## 10. TROUBLESHOOTING

### Issue: "No module named 'utils'"
- Ensure you're running from project root directory
- Check PYTHONPATH includes src/ directory
- The entry point scripts automatically handle this

### Issue: "Data not found"
- Verify data files exist at paths specified in config
- Use generate_sample_data.py to create sample data
- Check config YAML for correct data paths

### Issue: "Model training very slow"
- Reduce n_estimators in config/local.yaml
- Use Ridge instead of RandomForest for faster training
- Reduce dataset size for testing

### Issue: "Poor model performance"
- Check that training/test data is appropriate
- Verify feature engineering is working correctly
- Try different model hyperparameters
- Check for data leakage or encoding issues

## 11. NEXT STEPS

1. **Replace with Real Data**: Substitute generate_sample_data.py output with real House Prices dataset
2. **Tune Hyperparameters**: Use Optuna for automated hyperparameter tuning
3. **Add More Models**: Implement XGBoost, CatBoost, or ensemble methods
4. **Deploy to Production**: Use Docker/Kubernetes for deployment
5. **Set Up CI/CD**: Configure GitHub Actions for automated testing
6. **Monitor Performance**: Add logging and monitoring for production models

## SUPPORT

For issues or questions:
1. Check the README.md for full documentation
2. Review config files for examples
3. Run tests to verify setup
4. Check log output for error messages

Happy modeling!
"""
