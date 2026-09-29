from __future__ import annotations

import sqlite3
from textwrap import dedent

from src.colmap_utils import (
  build_exhaustive_matcher_command,
  build_feature_extractor_command,
    build_image_undistorter_command,
  build_mapper_command,
  inspect_colmap_database,
  inspect_matching_statistics,
  inspect_sparse_model,
)
from src.config import ProjectConfig


def test_load_config_reads_relative_paths(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent(
            """
            data:
              images_dir: "data/images"
              masks_dir: "data/masks"
            outputs:
              base_dir: "outputs"
              colmap_dir: "outputs/colmap"
              sparse_dir: "outputs/colmap/sparse"
              logs_dir: "outputs/logs"
              database_path: "outputs/colmap/database.db"
            colmap:
              executable_path: "colmap"
              single_camera: true
              matcher_type: "exhaustive"
              use_gpu: false
            logging:
              level: "INFO"
            validation:
              min_images: 2
            """
        ).strip(),
        encoding="utf-8",
    )

    config = ProjectConfig.from_yaml(config_file)

    assert config.image_dir == (tmp_path / "data" / "images").resolve()
    assert config.masks_dir == (tmp_path / "data" / "masks").resolve()
    assert config.logs_dir == (tmp_path / "outputs" / "logs").resolve()
    assert config.database_path == (tmp_path / "outputs" / "colmap" / "database.db").resolve()
    assert config.single_camera is True
    assert config.matcher_type == "exhaustive"
    assert config.min_images == 2
    assert config.use_gpu is False


def test_build_feature_extractor_command_uses_cpu_defaults(tmp_path):
    config = ProjectConfig(
        image_dir=tmp_path / "images",
        database_path=tmp_path / "colmap" / "database.db",
        colmap_path="C:/Program Files/COLMAP/bin/colmap.exe",
        camera_model="SIMPLE_RADIAL",
        single_camera=True,
        use_gpu=False,
    )

    command = build_feature_extractor_command(
        config=config,
        database_path=config.database_path,
        image_dir=config.image_dir,
    )

    assert command[0].lower().endswith("colmap.exe")
    assert "feature_extractor" in command
    assert "--database_path" in command
    assert "--image_path" in command
    assert "--ImageReader.camera_model" in command
    assert "SIMPLE_RADIAL" in command
    assert "--FeatureExtraction.use_gpu" in command
    assert "0" in command[command.index("--FeatureExtraction.use_gpu") + 1]


def test_inspect_colmap_database_reads_keypoint_statistics(tmp_path):
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE cameras (camera_id INTEGER)")
        cursor.execute("CREATE TABLE images (image_id INTEGER)")
        cursor.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER)")
        cursor.execute("CREATE TABLE descriptors (image_id INTEGER, rows INTEGER)")
        cursor.executemany("INSERT INTO cameras (camera_id) VALUES (?)", [(1,), (2,)])
        cursor.executemany("INSERT INTO images (image_id) VALUES (?)", [(1,), (2,), (3,)])
        cursor.executemany("INSERT INTO keypoints (image_id, rows) VALUES (?, ?)", [(1, 120), (2, 180)])
        cursor.executemany("INSERT INTO descriptors (image_id, rows) VALUES (?, ?)", [(1, 3), (2, 5)])
        connection.commit()

    stats = inspect_colmap_database(database_path)

    assert stats["cameras"] == 2
    assert stats["images"] == 3
    assert stats["images_with_keypoints"] == 2
    assert stats["total_keypoints"] == 300
    assert stats["descriptors_present"] is True


def test_build_exhaustive_matcher_command_uses_cpu_and_geometric_verification(tmp_path):
    config = ProjectConfig(
        colmap_path="colmap.exe",
        use_gpu=False,
        sift_settings={"num_threads": 1},
    )

    command = build_exhaustive_matcher_command(config, tmp_path / "database.db")

    assert command[1] == "exhaustive_matcher"
    assert "--database_path" in command
    assert "--FeatureMatching.use_gpu" in command
    assert command[command.index("--FeatureMatching.use_gpu") + 1] == "0"
    assert command[command.index("--FeatureMatching.skip_geometric_verification") + 1] == "0"


def test_inspect_matching_statistics_reads_raw_and_geometric_pairs(tmp_path):
    database_path = tmp_path / "database.db"
    max_num_images = 2147483647
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY)")
        cursor.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        cursor.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        cursor.executemany("INSERT INTO images (image_id) VALUES (?)", [(1,), (2,), (3,)])
        cursor.execute("INSERT INTO matches VALUES (?, ?)", (1 * max_num_images + 2, 250))
        cursor.execute("INSERT INTO matches VALUES (?, ?)", (1 * max_num_images + 3, 0))
        cursor.execute("INSERT INTO two_view_geometries VALUES (?, ?)", (1 * max_num_images + 2, 180))
        connection.commit()

    stats = inspect_matching_statistics(database_path)

    assert stats["images"] == 3
    assert stats["possible_pairs"] == 3
    assert stats["matched_pairs"] == 1
    assert stats["total_raw_matches"] == 250
    assert stats["geometrically_verified_pairs"] == 1
    assert stats["total_geometric_matches"] == 180


def test_build_mapper_command_uses_database_images_and_cpu_ba(tmp_path):
    config = ProjectConfig(colmap_path="colmap.exe", use_gpu=False, sift_settings={"num_threads": 1})

    command = build_mapper_command(
        config,
        tmp_path / "database.db",
        tmp_path / "images",
        tmp_path / "sparse",
    )

    assert command[1] == "mapper"
    assert "--database_path" in command
    assert "--image_path" in command
    assert "--output_path" in command
    assert command[command.index("--Mapper.ba_use_gpu") + 1] == "0"


def test_build_image_undistorter_command_does_not_use_gpu(tmp_path):
    config = ProjectConfig(colmap_path="colmap.exe", use_gpu=False)

    command = build_image_undistorter_command(
        config,
        tmp_path / "images",
        tmp_path / "sparse" / "0",
        tmp_path / "dense" / "0",
    )

    assert command[1] == "image_undistorter"
    assert "--copy_policy" in command
    assert command[command.index("--copy_policy") + 1] == "COPY"


def test_inspect_sparse_model_reads_binary_statistics(tmp_path):
    model_dir = tmp_path / "0"
    model_dir.mkdir()
    import struct

    with (model_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<QiiQQdddd", 1, 1, 2, 640, 480, 500.0, 320.0, 240.0, 0.0))
    with (model_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Qi4d3di", 1, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1))
        handle.write(b"image.jpg\x00")
        handle.write(struct.pack("<Qddq", 1, 10.0, 20.0, 7))
    with (model_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q Qddd BBB d Q ii", 1, 7, 1.0, 2.0, 3.0, 255, 0, 0, 0.5, 1, 1, 0))

    stats = inspect_sparse_model(model_dir)

    assert stats["cameras"] == 1
    assert stats["registered_images"] == 1
    assert stats["points3d"] == 1
    assert stats["observations"] == 1
    assert stats["mean_reprojection_error"] == 0.5
