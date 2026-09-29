from __future__ import annotations

import logging
import sys
from datetime import datetime

from scripts.run_stage1 import (
    archive_previous_outputs,
    build_stage_command,
    prepare_fresh_workspace,
    run_stage_command,
    select_primary_model,
)
from src.config import ProjectConfig


def make_config(tmp_path):
    return ProjectConfig(
        root_dir=tmp_path,
        image_dir=tmp_path / "data" / "images",
        output_dir=tmp_path / "outputs",
        colmap_dir=tmp_path / "outputs" / "colmap",
        sparse_dir=tmp_path / "outputs" / "colmap" / "sparse",
        logs_dir=tmp_path / "outputs" / "logs",
        archive_dir=tmp_path / "outputs" / "archive",
        database_path=tmp_path / "outputs" / "colmap" / "database.db",
    )


def test_archive_previous_outputs_preserves_old_database_and_sparse(tmp_path):
    config = make_config(tmp_path)
    config.database_path.parent.mkdir(parents=True)
    config.database_path.write_bytes(b"old database")
    config.sparse_dir.mkdir(parents=True)
    (config.sparse_dir / "points3D.bin").write_bytes(b"old sparse")
    (config.image_dir).mkdir(parents=True)
    image = config.image_dir / "input.jpg"
    image.write_bytes(b"input")

    result = archive_previous_outputs(
        config.database_path,
        config.sparse_dir,
        config.archive_dir,
        now=datetime(2026, 9, 29, 11, 30, 0),
    )

    assert result.archive_path == config.archive_dir / "stage1_2026-09-29_113000"
    assert (result.archive_path / "database.db").read_bytes() == b"old database"
    assert (result.archive_path / "sparse" / "points3D.bin").read_bytes() == b"old sparse"
    assert not config.database_path.exists()
    assert not config.sparse_dir.exists()
    assert image.read_bytes() == b"input"


def test_prepare_fresh_workspace_does_not_reuse_old_database(tmp_path):
    config = make_config(tmp_path)
    config.database_path.parent.mkdir(parents=True)
    config.database_path.write_bytes(b"old")
    config.sparse_dir.mkdir(parents=True)
    (config.sparse_dir / "old.bin").write_bytes(b"old")

    result = prepare_fresh_workspace(config, logging.getLogger("test_stage1"))

    assert result.archive_path is not None
    assert config.sparse_dir.is_dir()
    assert not config.database_path.exists()
    assert list(config.sparse_dir.iterdir()) == []
    assert (result.archive_path / "database.db").read_bytes() == b"old"


def test_archive_without_existing_outputs_returns_no_archive(tmp_path):
    result = archive_previous_outputs(
        tmp_path / "database.db",
        tmp_path / "sparse",
        tmp_path / "archive",
    )

    assert result.archive_path is None


def test_build_stage_command_uses_current_python_and_script_path():
    command = build_stage_command("match_features.py", extra_args=("--example",))

    assert command[0] == sys.executable
    assert command[1].endswith("scripts\\match_features.py") or command[1].endswith("scripts/match_features.py")
    assert command[-1] == "--example"


def test_select_primary_model_uses_registered_images_then_points():
    models = [
        {"directory": "0", "registered_images": 7, "points3d": 5000},
        {"directory": "1", "registered_images": 7, "points3d": 6000},
        {"directory": "2", "registered_images": 6, "points3d": 9000},
    ]

    assert select_primary_model(models)["directory"] == "1"


def test_run_stage_command_raises_on_nonzero_exit(tmp_path):
    logger = logging.getLogger("test_stage_failure")

    try:
        run_stage_command(
            "Synthetic failure",
            [sys.executable, "-c", "raise SystemExit(3)"],
            logger,
        )
    except RuntimeError as exc:
        assert "Synthetic failure failed with exit code 3" in str(exc)
    else:
        raise AssertionError("Expected a Stage1Failure for a non-zero subprocess exit")
