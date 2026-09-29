#!/usr/bin/env python3
"""Run COLMAP Mapper for Stage 1D sparse Structure-from-Motion."""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colmap_utils import (
    build_mapper_command,
    ensure_colmap_executable,
    inspect_colmap_database,
    inspect_matching_statistics,
    inspect_sparse_model,
    run_colmap_command,
)
from src.config import load_config
from src.image_utils import discover_images
from src.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run COLMAP Mapper to create a sparse Stage 1D reconstruction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML configuration file. Defaults to configs/config.yaml.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete only the generated sparse reconstruction directory before running Mapper.",
    )
    return parser


def _model_directories(sparse_dir: Path) -> list[Path]:
    return sorted(path for path in sparse_dir.iterdir() if path.is_dir())


def _unregistered_images(input_images: list[Path], registered_names: list[str]) -> list[str]:
    registered = {Path(name).name for name in registered_names}
    return [image.name for image in input_images if image.name not in registered]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(
        name="run_mapper",
        log_dir=config.logs_dir,
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
    )
    database_path = Path(config.database_path)
    image_dir = Path(config.image_dir)
    sparse_dir = Path(config.sparse_dir)

    logger.info("Stage 1D started")
    logger.info("Database path: %s", database_path)
    logger.info("Image path: %s", image_dir)
    logger.info("Sparse output path: %s", sparse_dir)
    logger.info("COLMAP executable: %s", config.colmap_path)
    logger.info("Mapper threads: %s", config.sift_settings.get("num_threads", 1))
    logger.info("Mapper bundle-adjustment GPU: disabled; CPU mode")

    if not database_path.is_file():
        logger.error("COLMAP database does not exist: %s", database_path)
        return 1
    if not image_dir.is_dir():
        logger.error("Image directory does not exist: %s", image_dir)
        return 1

    try:
        input_images = discover_images(image_dir)
        database_stats = inspect_colmap_database(database_path)
        matching_stats = inspect_matching_statistics(database_path)
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        logger.error("Stage 1D input validation failed: %s", exc)
        return 1

    required_tables = {"images", "keypoints", "descriptors", "matches", "two_view_geometries"}
    missing_tables = required_tables.difference(database_stats["tables"])
    if missing_tables:
        logger.error("Database is missing required Stage 1B/1C tables: %s", ", ".join(sorted(missing_tables)))
        return 1
    if database_stats["images"] <= 0:
        logger.error("No images exist in the COLMAP database.")
        return 1
    if database_stats["images_with_keypoints"] <= 0 or not database_stats["descriptors_present"]:
        logger.error("Stage 1B keypoints or descriptors are missing from the database.")
        return 1
    if matching_stats["matched_pairs"] <= 0:
        logger.error("No raw matches exist in the COLMAP database. Run Stage 1C first.")
        return 1
    if matching_stats["geometrically_verified_pairs"] <= 0:
        logger.error("No geometrically verified matches exist in the COLMAP database. Run Stage 1C first.")
        return 1

    logger.info("Input images: %s", len(input_images))
    logger.info("Database images: %s", database_stats["images"])
    logger.info("Raw matched pairs: %s", matching_stats["matched_pairs"])
    logger.info("Geometrically verified pairs: %s", matching_stats["geometrically_verified_pairs"])

    sparse_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.reset:
        if sparse_dir.exists():
            logger.warning("Reset requested. Deleting only sparse reconstruction output: %s", sparse_dir)
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
    elif sparse_dir.exists() and any(sparse_dir.iterdir()):
        logger.error("Sparse output already exists: %s", sparse_dir)
        logger.error("Use --reset to remove only this generated sparse output and rerun Mapper.")
        return 1
    else:
        sparse_dir.mkdir(parents=True, exist_ok=True)

    try:
        executable = ensure_colmap_executable(config.colmap_path)
        command = build_mapper_command(config, database_path, image_dir, sparse_dir)
    except (OSError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1

    logger.info("Verified COLMAP executable: %s", executable)
    start_time = time.time()
    result = run_colmap_command(command, logger)
    elapsed = time.time() - start_time
    logger.info("Mapper finished in %.2f seconds.", elapsed)

    if result.returncode != 0:
        logger.error("COLMAP Mapper failed.")
        logger.error("Exit code: %s", result.returncode)
        if result.stderr:
            logger.error("Relevant stderr:\n%s", result.stderr.strip())
        return 1

    models = []
    invalid_model_dirs = []
    for model_dir in _model_directories(sparse_dir):
        try:
            models.append(inspect_sparse_model(model_dir))
        except (OSError, ValueError, struct.error) as exc:
            invalid_model_dirs.append(f"{model_dir.name}: {exc}")

    if invalid_model_dirs:
        for message in invalid_model_dirs:
            logger.warning("Ignoring invalid sparse model: %s", message)
    if not models:
        logger.error("Mapper completed but produced no valid sparse reconstruction model in %s.", sparse_dir)
        return 1

    models.sort(key=lambda model: (model["registered_images"], model["points3d"]), reverse=True)
    selected = models[0]
    unregistered = _unregistered_images(input_images, selected["image_names"])
    registration_percentage = 100.0 * selected["registered_images"] / len(input_images) if input_images else 0.0

    logger.info("Reconstruction models found: %s", len(models))
    for index, model in enumerate(models):
        logger.info(
            "Model %s: %s registered images, %s cameras, %s 3D points",
            index,
            model["registered_images"],
            model["cameras"],
            model["points3d"],
        )
    logger.info("Selected reconstruction: %s", selected["directory"])
    logger.info("Registered images: %s / %s (%.2f%%)", selected["registered_images"], len(input_images), registration_percentage)
    logger.info("Cameras: %s", selected["cameras"])
    logger.info("Sparse 3D points: %s", selected["points3d"])
    logger.info("Observations: %s", selected["observations"])
    if selected["mean_reprojection_error"] is None:
        logger.info("Mean reprojection error: unavailable")
    else:
        logger.info("Mean reprojection error: %.6f pixels", selected["mean_reprojection_error"])
    if unregistered:
        logger.warning("Unregistered input images: %s", ", ".join(unregistered))

    print("Stage 1D complete")
    print(f"Sparse output: {sparse_dir}")
    print(f"Reconstruction models: {len(models)}")
    print(f"Selected reconstruction: {selected['directory']}")
    print(f"Registered images: {selected['registered_images']} / {len(input_images)} ({registration_percentage:.2f}%)")
    print(f"Cameras: {selected['cameras']}")
    print(f"Sparse 3D points: {selected['points3d']}")
    print(f"Observations: {selected['observations']}")
    print(
        "Mean reprojection error: "
        + (f"{selected['mean_reprojection_error']:.6f} pixels" if selected["mean_reprojection_error"] is not None else "unavailable")
    )
    if unregistered:
        print(f"Unregistered images: {', '.join(unregistered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
