from __future__ import annotations

import logging

import pytest

from scripts.prepare_dense import (
    DensePreparationError,
    prepare_dense_workspace,
    validate_dense_inputs,
    verify_undistorted_workspace,
)
from src.config import ProjectConfig
from src.colmap_utils import ensure_colmap_executable


def make_config(tmp_path):
    return ProjectConfig(
        image_dir=tmp_path / "images",
        database_path=tmp_path / "colmap" / "database.db",
        sparse_dir=tmp_path / "colmap" / "sparse",
        dense_dir=tmp_path / "colmap" / "dense" / "0",
        colmap_dir=tmp_path / "colmap",
        logs_dir=tmp_path / "logs",
        dense_model_id="0",
    )


def test_missing_sparse_model_is_reported(tmp_path):
    config = make_config(tmp_path)
    config.image_dir.mkdir(parents=True)
    config.database_path.parent.mkdir(parents=True)
    config.database_path.write_bytes(b"not used")

    with pytest.raises(DensePreparationError, match="Sparse model is incomplete"):
        validate_dense_inputs(config)


def test_missing_colmap_executable_is_reported():
    with pytest.raises(RuntimeError, match="COLMAP executable was not found"):
        ensure_colmap_executable("definitely-not-a-real-colmap-executable")


def test_dense_workspace_creation_and_reset_are_scoped(tmp_path):
    config = make_config(tmp_path)
    config.dense_dir.mkdir(parents=True)
    (config.dense_dir / "old.txt").write_text("old", encoding="utf-8")
    source = config.image_dir
    source.mkdir(parents=True)
    (source / "input.jpg").write_bytes(b"input")

    prepare_dense_workspace(config, reset=True, logger=logging.getLogger("test_dense"))

    assert config.dense_dir.is_dir()
    assert list(config.dense_dir.iterdir()) == []
    assert (source / "input.jpg").read_bytes() == b"input"


def test_existing_dense_workspace_requires_reset(tmp_path):
    config = make_config(tmp_path)
    config.dense_dir.mkdir(parents=True)
    (config.dense_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(DensePreparationError, match="already exists"):
        prepare_dense_workspace(config, reset=False, logger=logging.getLogger("test_dense"))


def test_undistorted_workspace_requires_images_and_sparse_data(tmp_path):
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()

    with pytest.raises(DensePreparationError, match="Undistorted image directory"):
        verify_undistorted_workspace(dense_dir)
