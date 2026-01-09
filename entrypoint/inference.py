"""Inference script - entry point for making predictions on test data."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.inference_pipeline import InferencePipeline  # noqa: E402
from utils import load_config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/04-predictions/submission.csv",
        help="Path to save predictions",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Run inference pipeline
        logger.info(f"Loading model from {args.model_dir}")
        pipeline = InferencePipeline(config, model_dir=args.model_dir)

        logger.info("Starting inference...")
        predictions = pipeline.run(test_file=args.test_file, return_df=True)

        # Print results
        logger.info("\n" + "=" * 50)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Generated predictions for {len(predictions)} samples")
        pred_min = predictions["prediction"].min()
        pred_max = predictions["prediction"].max()
        logger.info(f"Prediction range: [{pred_min:.2f}, {pred_max:.2f}]")
        logger.info(f"Mean prediction: {predictions['prediction'].mean():.2f}")
        logger.info(f"Saved predictions to {args.output_file}")
        logger.info("=" * 50)

        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
