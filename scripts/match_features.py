#!/usr/bin/env python3
"""Run COLMAP exhaustive feature matching for Stage 1C."""

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
    build_exhaustive_matcher_command,
    ensure_colmap_executable,
    inspect_colmap_database,
    inspect_matching_statistics,
    run_colmap_command,
)
from src.config import load_config
from src.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run COLMAP exhaustive matching using the Stage 1B SIFT database."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML configuration file. Defaults to configs/config.yaml.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(
        name="match_features",
        log_dir=config.logs_dir,
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
    )
    database_path = Path(config.database_path)

    logger.info("Stage 1C started")
    logger.info("Database path: %s", database_path)
    logger.info("COLMAP executable: %s", config.colmap_path)
    logger.info("Matcher type: exhaustive")
    logger.info("CPU/GPU mode: %s", "GPU" if config.use_gpu else "CPU")
    logger.info("Thread configuration: %s", config.sift_settings.get("num_threads", 1))

    if not database_path.exists():
        logger.error("COLMAP database does not exist: %s", database_path)
        logger.error("Run Stage 1B first with: python scripts/extract_features.py")
        return 1

    try:
        database_stats = inspect_colmap_database(database_path)
    except (OSError, ValueError) as exc:
        logger.error("COLMAP database could not be opened: %s", exc)
        return 1

    if database_stats["images"] <= 0:
        logger.error("No image records exist in the COLMAP database.")
        return 1
    if database_stats["images_with_keypoints"] <= 0:
        logger.error("No keypoints exist in the COLMAP database. Run Stage 1B feature extraction first.")
        return 1
    if not database_stats["descriptors_present"]:
        logger.error("No descriptors exist in the COLMAP database. Run Stage 1B feature extraction first.")
        return 1

    possible_pairs = database_stats["images"] * (database_stats["images"] - 1) // 2
    logger.info("Images in database: %s", database_stats["images"])
    logger.info("Possible unique image pairs: %s", possible_pairs)
    logger.info("Stage 1B keypoints: %s images, %s total keypoints", database_stats["images_with_keypoints"], database_stats["total_keypoints"])

    try:
        executable = ensure_colmap_executable(config.colmap_path)
        command = build_exhaustive_matcher_command(config, database_path)
    except (OSError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1

    logger.info("Verified COLMAP executable: %s", executable)
    logger.info("Existing matching rows will be recomputed by COLMAP; Stage 1B keypoints and descriptors are preserved.")

    start_time = time.time()
    result = run_colmap_command(command, logger)
    elapsed = time.time() - start_time
    logger.info("Matching finished in %.2f seconds.", elapsed)

    if result.returncode != 0:
        logger.error("COLMAP exhaustive matching failed.")
        logger.error("Exit code: %s", result.returncode)
        if result.stderr:
            logger.error("Relevant stderr:\n%s", result.stderr.strip())
        return 1

    try:
        matching_stats = inspect_matching_statistics(database_path)
    except (OSError, ValueError) as exc:
        logger.error("Matching database inspection failed: %s", exc)
        return 1

    logger.info("Matched image pairs: %s", matching_stats["matched_pairs"])
    logger.info("Total raw descriptor matches: %s", matching_stats["total_raw_matches"])
    if matching_stats["geometric_table_present"]:
        logger.info("Geometrically verified pairs: %s", matching_stats["geometrically_verified_pairs"])
        logger.info("Total geometrically verified matches: %s", matching_stats["total_geometric_matches"])
    else:
        logger.warning("The database does not expose a two_view_geometries table; geometric verification is unavailable.")

    if matching_stats["matched_pairs"] <= 0 and matching_stats["geometrically_verified_pairs"] <= 0:
        logger.error("Matching produced no raw or geometrically verified matches.")
        logger.error("Check image overlap, feature quality, and the COLMAP output log.")
        return 1

    print("Stage 1C complete")
    print(f"Database: {database_path}")
    print(f"Images: {matching_stats['images']}")
    print(f"Possible image pairs: {matching_stats['possible_pairs']}")
    print(f"Matched pairs: {matching_stats['matched_pairs']}")
    print(f"Total raw matches: {matching_stats['total_raw_matches']}")
    print(f"Geometrically verified pairs: {matching_stats['geometrically_verified_pairs']}")
    print(f"Total geometrically verified matches: {matching_stats['total_geometric_matches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
