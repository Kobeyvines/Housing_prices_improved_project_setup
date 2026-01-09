"""Unit tests for the training pipeline and utilities."""

# Add src to path
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.feature_eng_pipeline import FeatureEngineeringPipeline  # noqa: E402
from pipelines.training_pipeline import TrainingPipeline  # noqa: E402
from utils import (  # noqa: E402
    encode_categorical,
    handle_missing_values,
    load_data,
    load_model,
    remove_outliers,
    save_data,
    save_model,
)


class TestUtilsFunctions:
    """Test utility functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "A", "B", "A"],
            }
        )

    def test_handle_missing_values_median(self, sample_df):
        """Test handling missing values with median strategy."""
        df = sample_df.copy()
        df.loc[1, "feature1"] = np.nan

        result = handle_missing_values(df, strategy="median")

        assert not result.isnull().any().any()
        assert result.loc[1, "feature1"] == 3.0  # median of [1, 3, 4, 5]

    def test_handle_missing_values_drop(self, sample_df):
        """Test handling missing values with drop strategy."""
        df = sample_df.copy()
        df.loc[1, "feature1"] = np.nan

        result = handle_missing_values(df, strategy="drop")

        assert len(result) == 4
        assert not result.isnull().any().any()

    def test_remove_outliers(self, sample_df):
        """Test outlier removal."""
        df = sample_df.copy()
        df.loc[5] = [1000.0, 2000.0, "C"]  # Add outlier

        result = remove_outliers(df, threshold=3.0)

        assert len(result) == 5  # Original 5 rows, outlier removed

    def test_encode_categorical_onehot(self, sample_df):
        """Test one-hot encoding."""
        result, metadata = encode_categorical(sample_df, method="onehot")

        assert "category_A" in result.columns or "category_B" in result.columns
        assert metadata["method"] == "onehot"

    def test_save_and_load_data(self, sample_df):
        """Test saving and loading data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = str(Path(tmpdir) / "test.csv")

            save_data(sample_df, file_path)
            loaded_df = load_data(file_path)

            pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_save_and_load_model(self):
        """Test saving and loading models."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit([[1, 2], [3, 4]], [1, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model.pkl")

            save_model(model, model_path)
            loaded_model = load_model(model_path)

            # Check that loaded model works
            pred1 = model.predict([[5, 6]])
            pred2 = loaded_model.predict([[5, 6]])

            assert np.allclose(pred1, pred2)


class TestFeatureEngineeringPipeline:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "income": [30000, 40000, 50000, 60000, 70000],
                "city": ["NYC", "LA", "NYC", "LA", "NYC"],
            }
        )

    @pytest.fixture
    def config(self):
        """Create a sample config."""
        return {
            "preprocessing": {
                "handle_missing": "median",
                "categorical_encoding": "onehot",
            }
        }

    def test_pipeline_fit_transform(self, sample_df, config):
        """Test pipeline fit and transform."""
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.fit_transform(sample_df)

        # Check that output is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Check that we have more features due to one-hot encoding and new features
        assert result.shape[0] == sample_df.shape[0]

    def test_pipeline_detects_features(self, sample_df, config):
        """Test that pipeline detects numerical and categorical features."""
        pipeline = FeatureEngineeringPipeline(config)
        pipeline.fit(sample_df)

        assert len(pipeline.numerical_features) > 0
        assert len(pipeline.categorical_features) > 0

    def test_pipeline_handles_missing_values(self, config):
        """Test pipeline handles missing values."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0],
                "feature2": ["A", "B", "A", "B"],
            }
        )

        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.fit_transform(df)

        assert not result.isnull().any().any()


class TestTrainingPipeline:
    """Test training pipeline."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.choice(["A", "B", "C"], n_samples),
                "SalePrice": np.random.rand(n_samples) * 100 + 50,
            }
        )

        return data

    @pytest.fixture
    def config(self, tmp_path):
        """Create a sample config for training."""
        # Save sample data
        train_file = str(tmp_path / "train.csv")
        test_file = str(tmp_path / "test.csv")

        # Create and save dummy data
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.choice(["A", "B", "C"], n_samples),
                "SalePrice": np.random.rand(n_samples) * 100 + 50,
            }
        )
        df.to_csv(train_file, index=False)

        # Create test file without target
        test_df = df.drop(columns=["SalePrice"])
        test_df.to_csv(test_file, index=False)

        config = {
            "data": {
                "train_path": train_file,
                "test_path": test_file,
                "target_column": "SalePrice",
                "validation_split": 0.2,
            },
            "preprocessing": {
                "handle_missing": "median",
                "categorical_encoding": "onehot",
            },
            "model": {
                "type": "Ridge",
                "parameters": {
                    "alpha": 1.0,
                    "random_state": 42,
                },
            },
        }

        return config

    def test_pipeline_initializes(self, config):
        """Test pipeline initialization."""
        pipeline = TrainingPipeline(config)
        assert pipeline.config == config
        assert pipeline.model is None

    def test_pipeline_runs_successfully(self, config, tmp_path):
        """Test full pipeline run."""
        # Use Ridge model instead of RandomForest for faster testing
        config["model"]["type"] = "Ridge"

        pipeline = TrainingPipeline(config)
        metrics = pipeline.run()

        # Check that we got metrics
        assert "train_mae" in metrics
        assert "val_mae" in metrics
        assert "train_r2" in metrics
        assert "val_r2" in metrics

        # Check that metrics are reasonable
        assert 0 <= metrics["train_r2"] <= 1
        assert 0 <= metrics["val_r2"] <= 1
        assert metrics["train_mae"] >= 0
        assert metrics["val_mae"] >= 0

    def test_model_artifacts_saved(self, config, tmp_path):
        """Test that model artifacts are saved."""
        config["model"]["type"] = "Ridge"

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)

            pipeline = TrainingPipeline(config)
            pipeline.run()

            # Check that model files are created
            assert Path("models/model.pkl").exists()
            assert Path("models/scaler.pkl").exists()
            assert Path("models/feature_names.pkl").exists()
            assert Path("models/metrics.pkl").exists()

        finally:
            os.chdir(original_cwd)


class TestDataIntegration:
    """Integration tests for data loading and saving."""

    def test_data_roundtrip(self):
        """Test that data survives save/load cycle."""
        original_df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
                "col3": [1.1, 2.2, 3.3],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = str(Path(tmpdir) / "test.csv")

            save_data(original_df, file_path)
            loaded_df = load_data(file_path)

            pd.testing.assert_frame_equal(original_df, loaded_df)
