"""Training pipeline for House Prices regression model."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pipelines.feature_eng_pipeline import FeatureEngineeringPipeline
from utils import load_data, save_model, save_scaler

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Pipeline for training ML models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize training pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_pipeline = None
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_names = None

    def run(self) -> Dict[str, Any]:
        """Run the complete training pipeline.

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 50)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 50)

        # Load data
        train_df = load_data(self.config["data"]["train_path"])

        # Separate features and target
        target_column = self.config["data"]["target_column"]
        if target_column not in train_df.columns:
            # Try 'SalePrice' as default for House Prices dataset
            target_column = (
                "SalePrice" if "SalePrice" in train_df.columns else train_df.columns[-1]
            )
            logger.warning(
                f"Target column {self.config['data']['target_column']} "
                f"not found. Using {target_column}"
            )

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        logger.info(f"Data shape: X={X.shape}, y={y.shape}")  # noqa: E501

        # Feature engineering
        logger.info("Applying feature engineering...")
        self.feature_pipeline = FeatureEngineeringPipeline(self.config)
        X_engineered = self.feature_pipeline.fit_transform(X)
        self.feature_names = self.feature_pipeline.get_feature_names(X_engineered)

        # Align X and y after feature engineering (in case rows were removed)
        valid_indices = (
            X_engineered.index
            if hasattr(X_engineered, "index")
            else range(len(X_engineered))
        )
        X_engineered = X_engineered.reset_index(drop=True)
        y = (
            y.loc[valid_indices].reset_index(drop=True)
            if hasattr(y, "loc")
            else pd.Series(y.values[valid_indices], index=range(len(X_engineered)))
        )

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_engineered,
            y,
            test_size=self.config["data"]["validation_split"],
            random_state=self.config["model"]["parameters"].get("random_state", 42),
        )

        logger.info(f"Train set: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Validation set: X={X_val.shape}, y={y_val.shape}")

        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )

        # Train model
        logger.info("Training model...")
        self.model = self._build_model()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        logger.info("Evaluating model...")
        self.metrics = self._evaluate_model(
            self.model, X_train_scaled, X_val_scaled, y_train, y_val
        )

        # Save artifacts
        logger.info("Saving artifacts...")
        self._save_artifacts()

        logger.info("=" * 50)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 50)

        return self.metrics

    def _build_model(self) -> Any:
        """Build the ML model based on config.

        Returns:
            Untrained model object
        """
        model_type = self.config["model"]["type"]
        params = self.config["model"]["parameters"]

        if model_type == "RandomForest":
            model = RandomForestRegressor(**params)
        elif model_type == "Ridge":
            model = Ridge(**{k: v for k, v in params.items() if k != "n_estimators"})
        else:
            logger.warning(f"Unknown model type {model_type}, using Ridge")
            model = Ridge()

        logger.info(f"Built {model_type} model")
        return model

    def _evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model on train and validation sets.

        Args:
            model: Trained model
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target

        Returns:
            Dictionary with evaluation metrics
        """
        # Training metrics
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        # Validation metrics
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        metrics = {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }

        logger.info("Training Metrics:")
        logger.info(f"  MAE: {train_mae:.4f}")
        logger.info(f"  RMSE: {train_rmse:.4f}")
        logger.info(f"  R²: {train_r2:.4f}")

        logger.info("Validation Metrics:")
        logger.info(f"  MAE: {val_mae:.4f}")
        logger.info(f"  RMSE: {val_rmse:.4f}")
        logger.info(f"  R²: {val_r2:.4f}")

        return metrics

    def _save_artifacts(self) -> None:
        """Save trained model and scaler to disk."""
        Path("models").mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = "models/model.pkl"
        save_model(self.model, model_path)

        # Save scaler
        scaler_path = "models/scaler.pkl"
        save_scaler(self.scaler, scaler_path)

        # Save feature names
        feature_names_path = "models/feature_names.pkl"
        with open(feature_names_path, "wb") as f:
            pickle.dump(self.feature_names, f)
        logger.info(f"Saved feature names to {feature_names_path}")

        # Save metrics
        metrics_path = "models/metrics.pkl"
        with open(metrics_path, "wb") as f:
            pickle.dump(self.metrics, f)
        logger.info(f"Saved metrics to {metrics_path}")
