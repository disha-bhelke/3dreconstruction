#!/usr/bin/env python3
"""Prepare Stage 2A dense reconstruction and run image undistortion only."""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colmap_utils import (
    build_image_undistorter_command,
    ensure_colmap_executable,
    inspect_colmap_command_help,
    inspect_sparse_model,
    probe_patch_match_stereo,
    run_colmap_command,
)
from src.config import load_config
from src.image_utils import discover_images
from src.logger import setup_logger


class DensePreparationError(RuntimeError):
    """Raised when Stage 2A prerequisites are not satisfied."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate sparse/0, check COLMAP dense support, and run image undistortion only."
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
        help="Delete only the configured dense workspace before preparation.",
    )
    return parser


def validate_dense_inputs(config):
    sparse_model = config.sparse_dir / config.dense_model_id
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [name for name in required_files if not (sparse_model / name).is_file()]
    if missing:
        raise DensePreparationError(
            f"Sparse model is incomplete at {sparse_model}; missing: {', '.join(missing)}"
        )
    if not config.database_path.is_file():
        raise DensePreparationError(f"COLMAP database does not exist: {config.database_path}")
    if not config.image_dir.is_dir():
        raise DensePreparationError(f"Input image directory does not exist: {config.image_dir}")

    try:
        images = discover_images(config.image_dir)
        with sqlite3.connect(config.database_path) as connection:
            connection.execute("PRAGMA integrity_check").fetchone()
        sparse_stats = inspect_sparse_model(sparse_model)
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        raise DensePreparationError(f"Dense input validation failed: {exc}") from exc

    if not images:
        raise DensePreparationError(f"No input images were found in {config.image_dir}")
    if sparse_stats["registered_images"] <= 0:
        raise DensePreparationError(f"Sparse model contains no registered images: {sparse_model}")

    available_names = {image.name for image in images}
    registered_names = {Path(name).name for name in sparse_stats["image_names"]}
    missing_registered = sorted(registered_names - available_names)
    if missing_registered:
        raise DensePreparationError(
            "Sparse model references images that are not available in data/images: "
            + ", ".join(missing_registered)
        )

    return {
        "sparse_model": sparse_model,
        "images": images,
        "sparse_stats": sparse_stats,
    }


def prepare_dense_workspace(config, reset: bool, logger: logging.Logger) -> None:
    dense_dir = config.dense_dir
    if dense_dir.exists() and any(dense_dir.iterdir()):
        if not reset:
            raise DensePreparationError(
                f"Dense workspace already exists: {dense_dir}. Use --reset to remove only this dense workspace."
            )
        logger.warning("Reset requested; removing only dense workspace: %s", dense_dir)
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)


def verify_undistorted_workspace(dense_dir: Path) -> dict[str, Path]:
    images_dir = dense_dir / "images"
    sparse_dir = dense_dir / "sparse"
    if not images_dir.is_dir() or not any(images_dir.iterdir()):
        raise DensePreparationError(f"Undistorted image directory is missing or empty: {images_dir}")
    if not sparse_dir.is_dir() or not any(sparse_dir.iterdir()):
        raise DensePreparationError(f"Undistorted sparse directory is missing or empty: {sparse_dir}")
    return {"dense_dir": dense_dir, "images_dir": images_dir, "sparse_dir": sparse_dir}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        name="prepare_dense",
        log_dir=config.logs_dir,
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
    )

    logger.info("Stage 2A dense preparation started")
    logger.info("Selected sparse model: %s", config.sparse_dir / config.dense_model_id)
    logger.info("Dense workspace: %s", config.dense_dir)

    try:
        inputs = validate_dense_inputs(config)
    except DensePreparationError as exc:
        logger.error(str(exc))
        print("Status: FAILED")
        return 1

    sparse_stats = inputs["sparse_stats"]
    print("Dense reconstruction preparation")
    print("-------------------------------")
    print(f"Sparse model: {inputs['sparse_model']}")
    print(f"Registered images: {sparse_stats['registered_images']}")
    print(f"Input images: {len(inputs['images'])}")
    print(f"Cameras: {sparse_stats['cameras']}")
    print(f"Sparse points: {sparse_stats['points3d']}")

    try:
        executable = ensure_colmap_executable(config.colmap_path)
        help_results = {
            name: inspect_colmap_command_help(executable, name)
            for name in ("image_undistorter", "patch_match_stereo", "stereo_fusion")
        }
        for name, result in help_results.items():
            logger.info("COLMAP command %s available: %s", name, result["available"])
        if not all(result["available"] for result in help_results.values()):
            missing = [name for name, result in help_results.items() if not result["available"]]
            raise DensePreparationError("COLMAP dense commands unavailable: " + ", ".join(missing))

        patch_probe = probe_patch_match_stereo(executable)
        logger.info("PatchMatch capability probe return code: %s", patch_probe["returncode"])
        if patch_probe["output"]:
            logger.info("PatchMatch capability probe output:\n%s", patch_probe["output"].strip())

        prepare_dense_workspace(config, args.reset, logger)
        command = build_image_undistorter_command(
            config,
            config.image_dir,
            inputs["sparse_model"],
            config.dense_dir,
        )
        logger.info("Running image undistortion")
        result = run_colmap_command(command, logger)
        if result.returncode != 0:
            if config.dense_dir.exists():
                shutil.rmtree(config.dense_dir)
                logger.warning("Removed incomplete dense workspace after undistortion failure: %s", config.dense_dir)
            raise DensePreparationError(
                f"COLMAP image undistortion failed with exit code {result.returncode}"
            )
        workspace = verify_undistorted_workspace(config.dense_dir)
    except DensePreparationError as exc:
        logger.error(str(exc))
        print(f"Status: FAILED ({exc})")
        return 1
    except OSError as exc:
        logger.error("Stage 2A failed: %s", exc)
        print(f"Status: FAILED ({exc})")
        return 1

    print(f"Undistorted workspace: {workspace['dense_dir']}")
    print(f"Undistorted images: {workspace['images_dir']}")
    print(f"Undistorted sparse data: {workspace['sparse_dir']}")
    if patch_probe["requires_cuda"]:
        message = (
            "Stage 2 dense reconstruction cannot currently run with the installed "
            "COLMAP build because PatchMatch Stereo requires CUDA/GPU support."
        )
        logger.error(message)
        print(message)
        print("PatchMatch Stereo: unsupported on this COLMAP build")
        print("Stereo Fusion: not run")
        print("Status: READY FOR CAPABILITY UPGRADE, NOT READY FOR DENSE PROCESSING")
        return 2

    print("PatchMatch Stereo capability probe did not report a CUDA requirement.")
    print("PatchMatch Stereo: not run")
    print("Stereo Fusion: not run")
    print("Status: READY FOR STAGE 2B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
