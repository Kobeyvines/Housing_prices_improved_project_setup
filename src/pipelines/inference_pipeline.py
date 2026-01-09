"""Inference pipeline for making predictions on new data."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipelines.feature_eng_pipeline import FeatureEngineeringPipeline
from utils import load_data, load_model, load_scaler, save_data

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Pipeline for making predictions on new data."""

    def __init__(self, config: Dict[str, Any], model_dir: str = "models"):
        """Initialize inference pipeline.

        Args:
            config: Configuration dictionary
            model_dir: Directory containing trained model artifacts
        """
        self.config = config
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_pipeline = None
        self.feature_names = None

    def load_artifacts(self) -> None:
        """Load trained model and preprocessing artifacts."""
        logger.info("Loading model artifacts...")

        # Load model
        model_path = Path(self.model_dir) / "model.pkl"
        self.model = load_model(str(model_path))

        # Load scaler
        scaler_path = Path(self.model_dir) / "scaler.pkl"
        self.scaler = load_scaler(str(scaler_path))

        # Load feature names
        feature_names_path = Path(self.model_dir) / "feature_names.pkl"
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)

        logger.info(f"Loaded artifacts from {self.model_dir}")

    def run(self, test_file: str = None, return_df: bool = True) -> pd.DataFrame:
        """Run inference pipeline on test data.

        Args:
            test_file: Path to test CSV file. If None, uses config path.
            return_df: Whether to return predictions as DataFrame

        Returns:
            DataFrame with predictions
        """
        logger.info("=" * 50)
        logger.info("Starting Inference Pipeline")
        logger.info("=" * 50)

        # Load artifacts if not already loaded
        if self.model is None:
            self.load_artifacts()

        # Load test data
        if test_file is None:
            test_file = self.config["data"]["test_path"]

        logger.info(f"Loading test data from {test_file}")
        test_df = load_data(test_file)
        logger.info(f"Test data shape: {test_df.shape}")

        # Apply feature engineering
        logger.info("Applying feature engineering...")
        self.feature_pipeline = FeatureEngineeringPipeline(self.config)
        test_engineered = self._prepare_test_data(test_df)

        # Ensure features match training features
        test_engineered = self._align_features(test_engineered)

        # Scale features
        logger.info("Scaling features...")
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_engineered),
            columns=test_engineered.columns,
            index=test_engineered.index,
        )

        # Make predictions
        logger.info("Making predictions...")
        predictions = self.model.predict(test_scaled)

        # Create results DataFrame
        results_df = pd.DataFrame({"prediction": predictions})

        # Add ID column if exists
        if "Id" in test_df.columns:
            results_df.insert(0, "Id", test_df["Id"].values)

        logger.info(f"Predictions shape: {results_df.shape}")
        logger.info(
            f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]"
        )

        # Save predictions
        output_file = self.config.get("data", {}).get(
            "predictions_path", "data/04-predictions/submission.csv"
        )
        save_data(results_df, output_file)

        logger.info("=" * 50)
        logger.info("Inference Pipeline Completed Successfully")
        logger.info("=" * 50)

        return results_df if return_df else None

    def _prepare_test_data(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare test data with feature engineering.

        Args:
            test_df: Raw test DataFrame

        Returns:
            Feature-engineered test DataFrame
        """
        # Keep ID if present
        id_column = None
        if "Id" in test_df.columns:
            id_column = test_df[["Id"]].copy()
            test_df = test_df.drop(columns=["Id"])

        # Apply feature engineering (fit on test data only, no fitting)
        test_engineered = self.feature_pipeline.fit_transform(test_df)

        # Add back ID if present
        if id_column is not None:
            test_engineered = pd.concat([id_column, test_engineered], axis=1)

        return test_engineered

    def _align_features(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Align test features with training features.

        Args:
            test_df: Test DataFrame with features

        Returns:
            DataFrame with aligned features
        """
        # Add missing features with zero values
        for col in self.feature_names:
            if col not in test_df.columns:
                test_df[col] = 0
                logger.warning(f"Added missing feature {col} with zeros")

        # Select only features used in training
        test_df = test_df[self.feature_names]

        return test_df
