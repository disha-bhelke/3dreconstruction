#!/usr/bin/env python3
"""Run the complete Stage 1 sparse reconstruction workflow.

This orchestrator archives previous generated outputs before invoking the existing
stage scripts. It intentionally does not implement or run dense reconstruction.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colmap_utils import inspect_sparse_model
from src.config import ProjectConfig, load_config
from src.image_utils import discover_images
from src.logger import setup_logger


class Stage1Failure(RuntimeError):
    """Raised when a Stage 1 step fails."""


@dataclass(frozen=True)
class ArchiveResult:
    archive_path: Path | None
    archived_database: Path | None
    archived_sparse: Path | None


def _timestamped_archive_path(archive_root: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    candidate = archive_root / f"stage1_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = archive_root / f"stage1_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def archive_previous_outputs(
    database_path: str | Path,
    sparse_dir: str | Path,
    archive_root: str | Path,
    now: datetime | None = None,
) -> ArchiveResult:
    """Move only the previous generated database and sparse output to an archive."""
    database = Path(database_path)
    sparse = Path(sparse_dir)
    archive_base = Path(archive_root)
    existing_database = database.is_file()
    existing_sparse = sparse.exists()

    if not existing_database and not existing_sparse:
        return ArchiveResult(None, None, None)

    archive_path = _timestamped_archive_path(archive_base, now)
    archive_path.mkdir(parents=True, exist_ok=False)
    archived_database = None
    archived_sparse = None

    if existing_database:
        archived_database = archive_path / database.name
        shutil.move(str(database), str(archived_database))
    if existing_sparse:
        archived_sparse = archive_path / sparse.name
        shutil.move(str(sparse), str(archived_sparse))

    return ArchiveResult(archive_path, archived_database, archived_sparse)


def prepare_fresh_workspace(config: ProjectConfig, logger: logging.Logger) -> ArchiveResult:
    """Archive old Stage 1 outputs, then create empty output locations."""
    result = archive_previous_outputs(
        config.database_path,
        config.sparse_dir,
        config.archive_dir,
    )
    if result.archive_path:
        logger.info("Archived previous Stage 1 outputs to %s", result.archive_path)
    else:
        logger.info("No previous Stage 1 database or sparse output was found to archive")

    config.colmap_dir.mkdir(parents=True, exist_ok=True)
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    config.sparse_dir.mkdir(parents=True, exist_ok=True)
    if config.database_path.exists():
        raise Stage1Failure(f"Fresh database path is unexpectedly occupied: {config.database_path}")
    return result


def build_stage_command(
    script_name: str,
    config_path: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)]
    if config_path is not None:
        command.extend(["--config", str(Path(config_path))])
    command.extend(extra_args)
    return command


def run_stage_command(
    stage_name: str,
    command: Sequence[str],
    logger: logging.Logger,
) -> None:
    logger.info("Starting %s", stage_name)
    logger.info("Command: %s", " ".join(str(part) for part in command))
    result = subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        logger.info("%s stdout:\n%s", stage_name, result.stdout.strip())
    if result.stderr:
        logger.info("%s stderr:\n%s", stage_name, result.stderr.strip())
    if result.returncode != 0:
        raise Stage1Failure(f"{stage_name} failed with exit code {result.returncode}.")
    logger.info("%s completed successfully", stage_name)


def select_primary_model(models: Sequence[dict]) -> dict:
    """Select the model with the most registered images, then most points."""
    if not models:
        raise Stage1Failure("No valid sparse reconstruction models were found.")
    return max(models, key=lambda model: (model["registered_images"], model["points3d"]))


def validate_primary_reconstruction(config: ProjectConfig, input_count: int) -> dict:
    """Inspect all sparse models and return the highest-registration model summary."""
    if not config.sparse_dir.is_dir():
        raise Stage1Failure(f"Sparse reconstruction directory does not exist: {config.sparse_dir}")

    models = []
    for model_dir in sorted(path for path in config.sparse_dir.iterdir() if path.is_dir()):
        try:
            models.append(inspect_sparse_model(model_dir))
        except (OSError, ValueError, struct.error) as exc:
            raise Stage1Failure(f"Invalid sparse model at {model_dir}: {exc}") from exc

    primary = select_primary_model(models)
    summary = dict(primary)
    summary["model_count"] = len(models)
    summary["input_images"] = input_count
    summary["registration_percentage"] = (
        100.0 * primary["registered_images"] / input_count if input_count else 0.0
    )
    summary["models"] = models
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive previous outputs and run the complete Stage 1 sparse reconstruction workflow."
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = setup_logger(
        name=f"stage1_{timestamp}",
        log_dir=config.logs_dir,
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
        log_filename=f"stage1_{timestamp}.log",
    )

    logger.info("Stage 1 orchestrator started")
    archive_result = None
    try:
        archive_result = prepare_fresh_workspace(config, logger)
        config_arg = args.config
        run_stage_command("Image validation", build_stage_command("validate_images.py", config_arg), logger)
        run_stage_command(
            "SIFT feature extraction",
            build_stage_command("extract_features.py", config_arg, ("--reset",)),
            logger,
        )
        run_stage_command("Exhaustive matching and geometric verification", build_stage_command("match_features.py", config_arg), logger)
        run_stage_command("COLMAP Mapper / SfM", build_stage_command("run_mapper.py", config_arg), logger)

        input_images = discover_images(config.image_dir)
        summary = validate_primary_reconstruction(config, len(input_images))
        logger.info("Sparse reconstruction validation completed")
        logger.info("Input images: %s", summary["input_images"])
        logger.info("Registered images: %s", summary["registered_images"])
        logger.info("Registration percentage: %.2f%%", summary["registration_percentage"])
        logger.info("Cameras: %s", summary["cameras"])
        logger.info("Sparse 3D points: %s", summary["points3d"])
        logger.info("Observations: %s", summary["observations"])
        logger.info("Mean reprojection error: %s", summary["mean_reprojection_error"])
        logger.info("Primary model: %s", summary["directory"])
        logger.info("Archive path: %s", archive_result.archive_path if archive_result else "none")

        print("Stage 1 complete")
        print(f"Input images: {summary['input_images']}")
        print(f"Registered images: {summary['registered_images']}")
        print(f"Registration percentage: {summary['registration_percentage']:.2f}%")
        print(f"Cameras: {summary['cameras']}")
        print(f"Sparse 3D points: {summary['points3d']}")
        print(f"Observations: {summary['observations']}")
        print(f"Mean reprojection error: {summary['mean_reprojection_error']}")
        print(f"Primary model: {summary['directory']}")
        print(f"Archive path: {archive_result.archive_path if archive_result else 'none'}")
        return 0
    except Stage1Failure as exc:
        logger.error("Stage 1 stopped: %s", exc)
        if archive_result and archive_result.archive_path:
            logger.info("Previous outputs remain archived at %s", archive_result.archive_path)
        return 1
    except (OSError, ValueError) as exc:
        logger.error("Stage 1 stopped unexpectedly: %s", exc)
        if archive_result and archive_result.archive_path:
            logger.info("Previous outputs remain archived at %s", archive_result.archive_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
