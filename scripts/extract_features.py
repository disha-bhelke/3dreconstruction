#!/usr/bin/env python3
"""Run COLMAP feature extraction for Stage 1B."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colmap_utils import (
    build_feature_extractor_command,
    ensure_colmap_executable,
    inspect_colmap_database,
    run_colmap_command,
)
from src.config import load_config
from src.image_utils import discover_images
from src.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run COLMAP feature extraction to build the Stage 1B database and SIFT descriptors.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML configuration file. Defaults to configs/config.yaml.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the input photographs. Defaults to the configured image folder.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete only the generated COLMAP database file before feature extraction.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(
        name="extract_features",
        log_dir=config.logs_dir,
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
    )

    image_dir = Path(args.input_dir) if args.input_dir is not None else config.image_dir
    database_path = Path(config.database_path)

    logger.info("Stage 1B started")
    logger.info("Input image directory: %s", image_dir)
    logger.info("Database path: %s", database_path)
    logger.info("COLMAP executable: %s", config.colmap_path)
    logger.info("Camera model: %s", config.camera_model)
    logger.info("Single camera setting: %s", config.single_camera)
    logger.info("SIFT max features: %s", config.sift_settings.get("max_num_features", 8192))
    logger.info("SIFT max image size: %s", config.sift_settings.get("max_image_size", 3200))
    logger.info("COLMAP threads: %s", config.sift_settings.get("num_threads", 1))
    logger.info("CPU/GPU mode: %s", "GPU" if config.use_gpu else "CPU")

    try:
        images = discover_images(image_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(str(exc))
        return 1

    if not images:
        logger.error("No images were found in %s.", image_dir)
        logger.error("Add photos to the data/images folder before running the pipeline.")
        return 1

    logger.info("Found %s image(s) in the dataset.", len(images))

    database_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset and database_path.exists():
        logger.warning("Reset requested. Deleting only the generated database file: %s", database_path)
        database_path.unlink()

    try:
        executable = ensure_colmap_executable(config.colmap_path)
    except Exception as exc:  # pragma: no cover - runtime validation path
        logger.error(str(exc))
        return 1

    logger.info("Verified COLMAP executable: %s", executable)

    image_list_path = database_path.parent / "image_list.txt"
    with image_list_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            handle.write(str(image_path.resolve()) + "\n")

    logger.info("Valid image list written to %s", image_list_path)

    try:
        command = build_feature_extractor_command(
            config,
            database_path,
            image_dir,
            image_list_path=image_list_path,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    start_time = time.time()
    logger.info("Starting COLMAP feature extraction")
    result = run_colmap_command(command, logger)
    elapsed = time.time() - start_time
    logger.info("Feature extraction finished in %.2f seconds.", elapsed)

    if result.returncode != 0:
        logger.error("COLMAP feature extraction failed. Check outputs/logs/ for details.")
        logger.error("Exit code: %s", result.returncode)
        if result.stderr:
            logger.error("Relevant stderr:\n%s", result.stderr.strip())
        return 1

    if not database_path.exists():
        logger.error("The COLMAP database was not created at %s.", database_path)
        return 1

    try:
        stats = inspect_colmap_database(database_path)
    except Exception as exc:  # pragma: no cover - runtime validation path
        logger.error("Database validation failed: %s", exc)
        return 1

    logger.info("Database inspected successfully: %s", database_path)
    logger.info("Cameras: %s", stats["cameras"])
    logger.info("Images: %s", stats["images"])
    logger.info("Images with keypoints: %s", stats["images_with_keypoints"])
    logger.info("Total keypoints: %s", stats["total_keypoints"])
    logger.info("Descriptors present: %s", "yes" if stats["descriptors_present"] else "no")

    if stats["images"] <= 0:
        logger.error("No image records were found in the database. The feature extraction did not populate the COLMAP database correctly.")
        return 1

    if stats["images_with_keypoints"] <= 0:
        logger.error("No keypoints were extracted. This usually means the input images were not processed correctly or COLMAP failed silently.")
        return 1

    logger.info("Stage 1B completed successfully")
    print("Stage 1B complete")
    print(f"Database: {database_path}")
    print(f"Images: {stats['images']}")
    print(f"Images with keypoints: {stats['images_with_keypoints']}")
    print(f"Total keypoints: {stats['total_keypoints']}")
    print(f"Descriptors present: {'yes' if stats['descriptors_present'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
