from __future__ import annotations

import logging
import shutil
import sqlite3
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .config import ProjectConfig


class ColmapExecutableNotFoundError(RuntimeError):
    """Raised when the COLMAP executable cannot be found."""


def ensure_colmap_executable(colmap_path: str | Path) -> Path:
    candidate = Path(str(colmap_path))

    if candidate.exists():
        return candidate.resolve()

    resolved = shutil.which(str(colmap_path))
    if resolved:
        return Path(resolved).resolve()

    if candidate.name:
        resolved_name = shutil.which(candidate.name)
        if resolved_name:
            return Path(resolved_name).resolve()

    raise ColmapExecutableNotFoundError(
        "COLMAP executable was not found. Check the `colmap.executable` or `colmap.executable_path` value in configs/config.yaml."
    )


def check_gpu_support(executable: str | Path) -> bool:
    executable_path = ensure_colmap_executable(executable)
    try:
        result = subprocess.run(
            [str(executable_path), "feature_extractor", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to execute COLMAP at {executable_path}: {exc}") from exc

    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    lowered = output.lower()
    if "without gpu support" in lowered or "gpu support" in lowered and "not" in lowered:
        return False
    return True


def build_feature_extractor_command(
    config: ProjectConfig,
    database_path: str | Path,
    image_dir: str | Path,
    image_list_path: str | Path | None = None,
) -> list[str]:
    executable = Path(str(config.colmap_path))
    if config.use_gpu:
        supported = check_gpu_support(executable)
        if not supported:
            raise RuntimeError(
                "GPU mode was requested, but the installed COLMAP build does not support CUDA. Set `use_gpu: false` in configs/config.yaml or install a CUDA-enabled COLMAP build."
            )

    max_num_features = int(config.sift_settings.get("max_num_features", 8192))
    max_image_size = int(config.sift_settings.get("max_image_size", 3200))
    num_threads = int(config.sift_settings.get("num_threads", 1))

    command = [
        str(executable),
        "feature_extractor",
        "--database_path",
        str(Path(database_path)),
        "--image_path",
        str(Path(image_dir)),
        "--ImageReader.camera_model",
        str(config.camera_model),
        "--ImageReader.single_camera",
        "1" if config.single_camera else "0",
        "--FeatureExtraction.type",
        "SIFT",
        "--FeatureExtraction.use_gpu",
        "1" if config.use_gpu else "0",
        "--FeatureExtraction.num_threads",
        str(num_threads),
        "--FeatureExtraction.max_image_size",
        str(max_image_size),
        "--SiftExtraction.max_num_features",
        str(max_num_features),
    ]

    if image_list_path is not None:
        command.extend(["--image_list_path", str(Path(image_list_path))])

    return command


def build_exhaustive_matcher_command(
    config: ProjectConfig,
    database_path: str | Path,
) -> list[str]:
    """Build the COLMAP 4.x exhaustive SIFT matcher command."""
    executable = Path(str(config.colmap_path))
    if config.use_gpu and not check_gpu_support(executable):
        raise RuntimeError(
            "GPU mode was requested, but the installed COLMAP build does not support CUDA. "
            "Set `use_gpu: false` in configs/config.yaml or install a CUDA-enabled COLMAP build."
        )

    return [
        str(executable),
        "exhaustive_matcher",
        "--database_path",
        str(Path(database_path)),
        "--FeatureMatching.type",
        "SIFT_BRUTEFORCE",
        "--FeatureMatching.use_gpu",
        "1" if config.use_gpu else "0",
        "--FeatureMatching.num_threads",
        str(config.sift_settings.get("num_threads", 1)),
        "--FeatureMatching.skip_geometric_verification",
        "0",
    ]


def build_mapper_command(
    config: ProjectConfig,
    database_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Build the COLMAP 4.x CPU mapper command."""
    return [
        str(Path(str(config.colmap_path))),
        "mapper",
        "--database_path",
        str(Path(database_path)),
        "--image_path",
        str(Path(image_dir)),
        "--output_path",
        str(Path(output_dir)),
        "--Mapper.num_threads",
        str(config.sift_settings.get("num_threads", 1)),
        "--Mapper.ba_use_gpu",
        "0",
    ]


def build_image_undistorter_command(
    config: ProjectConfig,
    image_dir: str | Path,
    sparse_model_dir: str | Path,
    dense_dir: str | Path,
) -> list[str]:
    """Build the COLMAP image-undistorter command for Stage 2A."""
    return [
        str(Path(str(config.colmap_path))),
        "image_undistorter",
        "--image_path",
        str(Path(image_dir)),
        "--input_path",
        str(Path(sparse_model_dir)),
        "--output_path",
        str(Path(dense_dir)),
        "--output_type",
        "COLMAP",
        "--copy_policy",
        "COPY",
        "--num_threads",
        str(config.sift_settings.get("num_threads", 1)),
        "--max_image_size",
        str(config.dense_max_image_size),
    ]


def inspect_colmap_command_help(executable: str | Path, command_name: str) -> dict[str, Any]:
    """Check whether a COLMAP subcommand exposes help without running its work."""
    executable_path = ensure_colmap_executable(executable)
    try:
        result = subprocess.run(
            [str(executable_path), command_name, "-h"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command_name, "available": False, "output": str(exc), "returncode": None}
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return {
        "command": command_name,
        "available": result.returncode == 0,
        "output": output,
        "returncode": result.returncode,
    }


def probe_patch_match_stereo(executable: str | Path) -> dict[str, Any]:
    """Run a bounded empty-workspace probe to detect CUDA-only PatchMatch builds."""
    executable_path = ensure_colmap_executable(executable)
    with tempfile.TemporaryDirectory(prefix="colmap_stage2a_probe_") as workspace:
        command = [
            str(executable_path),
            "patch_match_stereo",
            "--workspace_path",
            workspace,
            "--PatchMatchStereo.num_threads",
            "1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "initialized": False,
                "requires_cuda": False,
                "output": str(exc),
                "returncode": None,
            }
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lowered = output.lower()
    requires_cuda = "requires cuda" in lowered or "cuda" in lowered and "not available" in lowered
    return {
        "command": command,
        "initialized": result.returncode == 0,
        "requires_cuda": requires_cuda,
        "output": output,
        "returncode": result.returncode,
    }


_CAMERA_MODEL_PARAM_COUNTS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 12,  # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 8,  # THIN_PRISM_FISHEYE
    11: 12, # RAD_TAN_THIN_PRISM_FISHEYE
    12: 4,  # EQUIRECTANGULAR
    13: 5,  # FISHEYE
}


def _read_struct(handle: Any, format_string: str) -> tuple[Any, ...]:
    size = struct.calcsize(format_string)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected end of COLMAP binary model file.")
    return struct.unpack(format_string, data)


def read_cameras_bin(path: str | Path) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            camera_id, model_id = _read_struct(handle, "<ii")
            width, height = _read_struct(handle, "<QQ")
            param_count = _CAMERA_MODEL_PARAM_COUNTS.get(model_id)
            if param_count is None:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            params = _read_struct(handle, "<" + "d" * param_count)
            cameras.append({
                "camera_id": camera_id,
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            })
    return cameras


def read_images_bin(path: str | Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            (image_id,) = _read_struct(handle, "<i")
            qvec = _read_struct(handle, "<dddd")
            tvec = _read_struct(handle, "<ddd")
            (camera_id,) = _read_struct(handle, "<i")
            name_bytes = bytearray()
            while True:
                byte = handle.read(1)
                if not byte:
                    raise ValueError("Unexpected end of COLMAP images.bin while reading image name.")
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
            (point_count,) = _read_struct(handle, "<Q")
            points = []
            for _ in range(point_count):
                x, y, point3d_id = _read_struct(handle, "<ddq")
                points.append((x, y, point3d_id))
            images.append({
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name_bytes.decode("utf-8"),
                "points2d": points,
            })
    return images


def read_points3d_bin(path: str | Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            point_id = _read_struct(handle, "<Q")[0]
            xyz = _read_struct(handle, "<ddd")
            rgb = _read_struct(handle, "<BBB")
            (error,) = _read_struct(handle, "<d")
            (track_length,) = _read_struct(handle, "<Q")
            track = [_read_struct(handle, "<ii") for _ in range(track_length)]
            points.append({
                "point3d_id": point_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            })
    return points


def inspect_sparse_model(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir)
    required_files = [model_path / "cameras.bin", model_path / "images.bin", model_path / "points3D.bin"]
    missing_files = [str(path.name) for path in required_files if not path.is_file()]
    empty_files = [str(path.name) for path in required_files if path.is_file() and path.stat().st_size == 0]
    if missing_files or empty_files:
        raise FileNotFoundError(
            f"Sparse model {model_path} is incomplete. Missing: {missing_files}; empty: {empty_files}."
        )

    cameras = read_cameras_bin(required_files[0])
    images = read_images_bin(required_files[1])
    points = read_points3d_bin(required_files[2])
    total_observations = sum(len(point["track"]) for point in points)
    reprojection_errors = [point["error"] for point in points]

    return {
        "directory": str(model_path),
        "cameras": len(cameras),
        "registered_images": len(images),
        "image_names": [image["name"] for image in images],
        "points3d": len(points),
        "observations": total_observations,
        "mean_reprojection_error": (
            sum(reprojection_errors) / len(reprojection_errors)
            if reprojection_errors else None
        ),
        "files": [path.name for path in model_path.iterdir() if path.is_file()],
    }


def _pair_id_to_image_ids(pair_id: int, image_ids: list[int]) -> tuple[int, int] | None:
    """Decode a COLMAP pair id using the current database image ids."""
    max_num_images = 2147483647
    first_id = pair_id // max_num_images
    second_id = pair_id % max_num_images
    if first_id in image_ids and second_id in image_ids and first_id != second_id:
        return first_id, second_id
    return None


def inspect_matching_statistics(database_path: str | Path) -> dict[str, Any]:
    """Inspect raw and geometrically verified match tables without changing the DB."""
    database_file = Path(database_path)
    if not database_file.exists():
        raise FileNotFoundError(f"COLMAP database was not found at {database_file}.")

    with sqlite3.connect(str(database_file)) as connection:
        cursor = connection.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }
        image_ids = [row[0] for row in cursor.execute("SELECT image_id FROM images ORDER BY image_id")]
        image_count = len(image_ids)
        possible_pairs = image_count * (image_count - 1) // 2

        def read_pair_table(table_name: str) -> tuple[int, int, int]:
            if table_name not in tables:
                return 0, 0, 0
            pair_rows = cursor.execute(
                f"SELECT pair_id, rows FROM {table_name} WHERE rows > 0"
            ).fetchall()
            valid_pair_count = sum(
                _pair_id_to_image_ids(int(pair_id), image_ids) is not None
                for pair_id, _ in pair_rows
            )
            total_rows = sum(int(rows) for _, rows in pair_rows)
            return len(pair_rows), valid_pair_count, total_rows

        raw_rows, raw_pairs, raw_matches = read_pair_table("matches")
        geometry_rows, geometry_pairs, geometry_matches = read_pair_table("two_view_geometries")

        return {
            "database_path": str(database_file),
            "tables": sorted(tables),
            "images": image_count,
            "possible_pairs": possible_pairs,
            "raw_match_rows": raw_rows,
            "matched_pairs": raw_pairs,
            "total_raw_matches": raw_matches,
            "geometric_table_present": "two_view_geometries" in tables,
            "geometric_match_rows": geometry_rows,
            "geometrically_verified_pairs": geometry_pairs,
            "total_geometric_matches": geometry_matches,
        }


def inspect_colmap_database(database_path: str | Path) -> dict[str, Any]:
    database_file = Path(database_path)
    if not database_file.exists():
        raise FileNotFoundError(f"COLMAP database was not found at {database_file}.")

    with sqlite3.connect(str(database_file)) as connection:
        cursor = connection.cursor()
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
        ]

        cameras = 0
        images = 0
        images_with_keypoints = 0
        total_keypoints = 0
        descriptors_present = False

        if "cameras" in tables:
            cameras = int(cursor.execute("SELECT COUNT(*) FROM cameras").fetchone()[0])
        if "images" in tables:
            images = int(cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        if "keypoints" in tables:
            try:
                images_with_keypoints = int(
                    cursor.execute("SELECT COUNT(DISTINCT image_id) FROM keypoints").fetchone()[0]
                )
            except sqlite3.DatabaseError:
                images_with_keypoints = int(cursor.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0])
            try:
                total_keypoints = int(cursor.execute("SELECT COALESCE(SUM(rows), 0) FROM keypoints").fetchone()[0])
            except sqlite3.DatabaseError:
                total_keypoints = 0
        if "descriptors" in tables:
            descriptors_present = int(cursor.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0]) > 0

        return {
            "database_path": str(database_file),
            "tables": tables,
            "cameras": cameras,
            "images": images,
            "images_with_keypoints": images_with_keypoints,
            "total_keypoints": total_keypoints,
            "descriptors_present": descriptors_present,
        }


def run_colmap_command(command: Sequence[str], logger: logging.Logger | None = None) -> subprocess.CompletedProcess[str]:
    command_text = " ".join(str(part) for part in command)
    if logger is not None:
        logger.info("Running COLMAP command")
        logger.info("Command: %s", command_text)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if logger is not None:
        logger.info("Return code: %s", completed.returncode)
        if completed.stdout:
            logger.info("STDOUT:\n%s", completed.stdout.strip())
        if completed.stderr:
            logger.error("STDERR:\n%s", completed.stderr.strip())
    return completed
