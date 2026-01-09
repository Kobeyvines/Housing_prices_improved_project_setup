"""Feature engineering pipeline for House Prices dataset."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Pipeline for feature engineering and transformation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineering pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.numerical_features = []
        self.categorical_features = []
        self.features_created = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineeringPipeline":
        """Fit feature engineering pipeline.

        Args:
            df: Input DataFrame

        Returns:
            Self for method chaining
        """
        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        logger.info(f"Detected {len(self.numerical_features)} numerical features")
        logger.info(f"Detected {len(self.categorical_features)} categorical features")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with feature engineering.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Create new features
        df = self._create_new_features(df)

        # Encode categorical variables
        df = self._encode_categorical(df)

        # Remove outliers
        df = self._remove_outliers(df)

        logger.info(f"Feature engineering completed. Output shape: {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            logger.info(f"Found {missing_count.sum()} missing values")

            # Fill numerical columns with median
            for col in self.numerical_features:
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(
                        f"Filled missing values in {col} with median: {median_val:.2f}"
                    )

            # Fill categorical columns with mode
            for col in self.categorical_features:
                if col in df.columns and df[col].isnull().any():
                    mode_val = (
                        df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    )
                    df[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")

        return df

    def _create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        # Example: Create polynomial features for numerical columns
        for col in self.numerical_features[:3]:  # Limit to first 3 columns
            if col in df.columns:
                # Squared features
                df[f"{col}_squared"] = df[col] ** 2
                self.features_created[f"{col}_squared"] = "squared"

                # Log features (if no negative values)
                if (df[col] > 0).all():
                    df[f"{col}_log"] = np.log(df[col] + 1)
                    self.features_created[f"{col}_log"] = "log"

        logger.info(f"Created {len(self.features_created)} new features")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical features
        """
        if not self.categorical_features:
            return df

        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=self.categorical_features, drop_first=True)
        logger.info(
            f"One-hot encoded {len(self.categorical_features)} categorical features"
        )

        return df

    def _remove_outliers(
        self, df: pd.DataFrame, threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers using Z-score method.

        Args:
            df: Input DataFrame
            threshold: Z-score threshold

        Returns:
            DataFrame with outliers removed
        """
        initial_rows = len(df)

        numerical_cols = [col for col in self.numerical_features if col in df.columns]

        for col in numerical_cols:
            if df[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outlier rows")

        return df

    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get feature names after transformation.

        Args:
            df: Transformed DataFrame

        Returns:
            List of feature names
        """
        return df.columns.tolist()
