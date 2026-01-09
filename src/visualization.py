"""Visualization utilities for creating and saving plots."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ImageSaver:
    """Helper class for saving visualization images to images/ folder."""

    def __init__(self, images_dir: str = "images", dpi: int = 300):
        """Initialize ImageSaver.

        Args:
            images_dir: Directory to save images (default: 'images')
            dpi: DPI for saving plots (default: 300)
        """
        self.images_dir = images_dir
        self.dpi = dpi
        Path(images_dir).mkdir(parents=True, exist_ok=True)

    def save_figure(
        self, fig: plt.Figure, filename: str, close_fig: bool = True
    ) -> str:
        """Save matplotlib figure.

        Args:
            fig: Matplotlib figure object
            filename: Name of the file (with or without extension)
            close_fig: Whether to close figure after saving (default: True)

        Returns:
            Path to saved image
        """
        # Ensure filename has extension
        if not filename.endswith((".png", ".jpg", ".pdf", ".svg")):
            filename = f"{filename}.png"

        filepath = Path(self.images_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(str(filepath), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved figure to {filepath}")

        if close_fig:
            plt.close(fig)

        return str(filepath)

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        filename: str = "feature_importance.png",
    ) -> str:
        """Plot and save feature importance.

        Args:
            importance: Feature importance values
            feature_names: Feature names
            top_n: Number of top features to display (default: 20)
            filename: Name of the output file (default: 'feature_importance.png')

        Returns:
            Path to saved image
        """
        # Get top features
        indices = np.argsort(importance)[-top_n:]
        top_importance = importance[indices]
        top_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_names, top_importance, color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")
        ax.invert_yaxis()

        return self.save_figure(fig, filename)

    def plot_distribution(
        self, data: pd.Series, filename: str = "distribution.png"
    ) -> str:
        """Plot and save data distribution.

        Args:
            data: Data series
            filename: Name of the output file (default: 'distribution.png')

        Returns:
            Path to saved image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1.hist(data, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.set_xlabel(data.name if data.name else "Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution")

        # Box plot
        ax2.boxplot(data)
        ax2.set_ylabel(data.name if data.name else "Value")
        ax2.set_title("Box Plot")

        return self.save_figure(fig, filename)

    def plot_correlation_heatmap(
        self, df: pd.DataFrame, filename: str = "correlation_heatmap.png"
    ) -> str:
        """Plot and save correlation heatmap.

        Args:
            df: DataFrame with numerical columns
            filename: Name of the output file (default: 'correlation_heatmap.png')

        Returns:
            Path to saved image
        """
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn not installed, using basic correlation plot")
            return self._plot_basic_correlation(df, filename)

        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=False, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Heatmap")

        return self.save_figure(fig, filename)

    def _plot_basic_correlation(self, df: pd.DataFrame, filename: str) -> str:
        """Plot basic correlation without seaborn."""
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df.corr()
        im = ax.imshow(correlation, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.columns)))
        ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation.columns)
        ax.set_title("Feature Correlation")
        fig.colorbar(im, ax=ax)

        return self.save_figure(fig, filename)

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "predictions_vs_actual.png",
    ) -> str:
        """Plot and save predictions vs actual values.

        Args:
            y_true: True target values
            y_pred: Predicted values
            filename: Name of the output file (default: 'predictions_vs_actual.png')

        Returns:
            Path to saved image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Predictions vs Actual")

        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax2.axhline(y=0, color="r", linestyle="--")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals Plot")

        return self.save_figure(fig, filename)

    def plot_training_history(
        self,
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        filename: str = "training_history.png",
    ) -> str:
        """Plot and save training history.

        Args:
            train_loss: Training loss values
            val_loss: Validation loss values (optional)
            filename: Name of the output file (default: 'training_history.png')

        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(train_loss, label="Train Loss", marker="o")
        if val_loss is not None:
            ax.plot(val_loss, label="Validation Loss", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self.save_figure(fig, filename)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        filename: str = "confusion_matrix.png",
    ) -> str:
        """Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names (optional)
            filename: Name of the output file (default: 'confusion_matrix.png')

        Returns:
            Path to saved image
        """
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn not installed, using basic confusion matrix")
            return self._plot_basic_confusion_matrix(y_true, y_pred, labels, filename)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")

        if labels is not None:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

        return self.save_figure(fig, filename)

    def _plot_basic_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]],
        filename: str,
    ) -> str:
        """Plot basic confusion matrix without seaborn."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="w")

        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

        fig.colorbar(im, ax=ax)
        return self.save_figure(fig, filename)

    def plot_metrics_summary(
        self, metrics: dict, filename: str = "metrics_summary.png"
    ) -> str:
        """Plot and save metrics summary.

        Args:
            metrics: Dictionary of metric names and values
            filename: Name of the output file (default: 'metrics_summary.png')

        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(metrics.keys())
        values = list(metrics.values())
        colors = ["green" if v > 0.5 else "orange" for v in values]

        ax.bar(names, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Metrics")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha="right")

        return self.save_figure(fig, filename)


def get_image_saver(images_dir: str = "images") -> ImageSaver:
    """Get ImageSaver instance.

    Args:
        images_dir: Directory to save images (default: 'images')

    Returns:
        ImageSaver instance
    """
    return ImageSaver(images_dir=images_dir)
