"""Training script - entry point for training the ML model."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.training_pipeline import TrainingPipeline  # noqa: E402
from utils import load_config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train House Prices regression model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Run training pipeline
        pipeline = TrainingPipeline(config)
        metrics = pipeline.run()

        # Print results
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Validation R² Score: {metrics['val_r2']:.4f}")
        logger.info(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"Validation MAE: {metrics['val_mae']:.4f}")
        logger.info("=" * 50)

        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
