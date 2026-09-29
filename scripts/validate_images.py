#!/usr/bin/env python3
"""Validate a dataset of input images before running COLMAP."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.image_utils import discover_images, validate_image_dataset
from src.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate input images before starting the Structure-from-Motion pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the source photographs. Defaults to the configured image folder.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML configuration file. Defaults to configs/config.yaml.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=None,
        help="Minimum number of usable images required to proceed.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(
        name="validate_images",
        log_dir=config.logs_dir,
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
    )

    image_dir = Path(args.input_dir) if args.input_dir is not None else config.image_dir
    min_images = args.min_images if args.min_images is not None else config.min_images

    logger.info("Starting image validation")
    logger.info("Input directory: %s", image_dir)

    try:
        images = discover_images(image_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(str(exc))
        return 1

    if not images:
        logger.error("No images were found in %s.", image_dir)
        logger.error("Add photos to the data/images folder before running the pipeline.")
        return 1

    logger.info("Found %s image(s).", len(images))

    dataset_summary = validate_image_dataset(images, min_images=min_images)

    for record in dataset_summary["records"]:
        if record["is_valid"]:
            logger.info(
                "%s | %sx%s | channels=%s",
                record["filename"],
                record["width"],
                record["height"],
                record["channels"],
            )
        else:
            logger.warning("Invalid image: %s (%s)", record["filename"], record["warning"])

    for warning in dataset_summary["warnings"]:
        logger.warning(warning)

    if not dataset_summary["usable"]:
        logger.error("Image validation failed. Dataset is unusable for SfM.")
        logger.error("Minimum required image count: %s", min_images)
        logger.error("Usable images discovered: %s", dataset_summary["valid_images"])
        if dataset_summary["invalid_images"]:
            invalid_names = [item["filename"] for item in dataset_summary["invalid_images"]]
            logger.error("Invalid or unreadable images: %s", ", ".join(invalid_names))
        return 1

    logger.info("Image validation successful")
    logger.info("Summary: %s valid images found out of %s total.", dataset_summary["valid_images"], dataset_summary["total_images"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
