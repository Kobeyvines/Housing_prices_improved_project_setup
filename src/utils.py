"""Utility functions for data processing and model operations."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {config_path}")


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data file.

    Args:
        file_path: Path to CSV file

    Returns:
        Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data from {file_path}: shape {df.shape}")
    return df


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        file_path: Path to save CSV file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved data to {file_path}: shape {df.shape}")


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values
            - 'median': Fill with median (numerical)
            - 'mean': Fill with mean (numerical)
            - 'mode': Fill with mode (categorical)
            - 'drop': Drop rows with missing values

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    missing_count = df.isnull().sum()

    if missing_count.sum() == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Found {missing_count.sum()} missing values")

    if strategy == "drop":
        df = df.dropna()
    elif strategy in ["median", "mean"]:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                fill_value = (
                    df[col].median() if strategy == "median" else df[col].mean()
                )
                df[col].fillna(fill_value, inplace=True)
    elif strategy == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col].fillna(mode_val, inplace=True)

    logger.info(f"Handled missing values with strategy: {strategy}")
    return df


def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using Z-score method.

    Args:
        df: Input DataFrame
        threshold: Z-score threshold (default 3.0 for 99.7% confidence)

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]

    logger.info(f"Removed outliers with threshold: {threshold}")
    return df


def encode_categorical(
    df: pd.DataFrame, categorical_cols: list = None, method: str = "onehot"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode categorical variables.

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns. If None, auto-detect.
        method: Encoding method ('onehot' or 'label')

    Returns:
        Tuple of (encoded DataFrame, encoding metadata)
    """
    df = df.copy()
    metadata = {"method": method, "categorical_cols": categorical_cols}

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        metadata["categorical_cols"] = categorical_cols

    if not categorical_cols:
        logger.info("No categorical columns to encode")
        return df, metadata

    if method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.info(f"One-hot encoded {len(categorical_cols)} columns")
    elif method == "label":
        from sklearn.preprocessing import LabelEncoder

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        metadata["label_encoders"] = label_encoders
        logger.info(f"Label encoded {len(categorical_cols)} columns")

    return df, metadata


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None,
    method: str = "standard",
) -> Tuple[pd.DataFrame, Any]:
    """Scale numerical features.

    Args:
        X_train: Training features
        X_test: Test features (optional)
        method: Scaling method ('standard', 'minmax', or 'robust')

    Returns:
        Tuple of (scaled X_train, X_test if provided, scaler object)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        logger.info(f"Scaled features using {method} method")
        return X_train_scaled, X_test_scaled, scaler

    logger.info(f"Scaled features using {method} method")
    return X_train_scaled, scaler


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split data: train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_model(model: Any, model_path: str) -> None:
    """Save trained model to pickle file.

    Args:
        model: Trained model object
        model_path: Path to save model
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")


def load_model(model_path: str) -> Any:
    """Load trained model from pickle file.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model object
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")
    return model


def save_scaler(scaler: Any, scaler_path: str) -> None:
    """Save scaler to pickle file.

    Args:
        scaler: Fitted scaler object
        scaler_path: Path to save scaler
    """
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")


def load_scaler(scaler_path: str) -> Any:
    """Load scaler from pickle file.

    Args:
        scaler_path: Path to scaler file

    Returns:
        Loaded scaler object
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded scaler from {scaler_path}")
    return scaler


def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from model if available.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance scores
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance_df


def ensure_dir_exists(dir_path: str) -> None:
    """Ensure directory exists, create if not.

    Args:
        dir_path: Path to directory
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename: str, images_dir: str = "images") -> str:
    """Save matplotlib figure to images directory.

    Args:
        fig: Matplotlib figure object
        filename: Name of the file (with or without extension)
        images_dir: Directory to save images (default: 'images')

    Returns:
        Path to saved image file
    """
    ensure_dir_exists(images_dir)

    # Ensure filename has extension
    if not filename.endswith((".png", ".jpg", ".pdf")):
        filename = f"{filename}.png"

    filepath = Path(images_dir) / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {filepath}")

    return str(filepath)


def save_image(image_array, filename: str, images_dir: str = "images") -> str:
    """Save image array (numpy or PIL) to images directory.

    Args:
        image_array: Image array (numpy array or PIL Image)
        filename: Name of the file (with or without extension)
        images_dir: Directory to save images (default: 'images')

    Returns:
        Path to saved image file
    """
    ensure_dir_exists(images_dir)

    # Ensure filename has extension
    if not filename.endswith((".png", ".jpg", ".pdf")):
        filename = f"{filename}.png"

    filepath = Path(images_dir) / filename

    # Handle different image types
    try:
        from PIL import Image

        if isinstance(image_array, Image.Image):
            image_array.save(filepath, quality=95)
        else:
            # Assume numpy array
            Image.fromarray(image_array).save(filepath, quality=95)
    except ImportError:
        logger.warning("PIL not available, attempting numpy save")
        np.save(filepath, image_array)

    logger.info(f"Saved image to {filepath}")
    return str(filepath)
