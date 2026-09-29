from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve_path(raw_value: str | Path | None, base_dir: Path) -> Path:
    if raw_value is None:
        return base_dir

    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        return candidate

    return (base_dir / candidate).resolve()


@dataclass
class ProjectConfig:
    root_dir: Path = PROJECT_ROOT
    image_dir: Path = PROJECT_ROOT / "data" / "images"
    masks_dir: Path = PROJECT_ROOT / "data" / "masks"
    output_dir: Path = PROJECT_ROOT / "outputs"
    colmap_dir: Path = PROJECT_ROOT / "outputs" / "colmap"
    sparse_dir: Path = PROJECT_ROOT / "outputs" / "colmap" / "sparse"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"
    archive_dir: Path = PROJECT_ROOT / "outputs" / "archive"
    database_path: Path = PROJECT_ROOT / "outputs" / "colmap" / "database.db"
    dense_model_id: str = "0"
    dense_dir: Path = PROJECT_ROOT / "outputs" / "colmap" / "dense" / "0"
    dense_max_image_size: int = -1
    dense_patch_match_threads: int = 1
    dense_stereo_fusion_threads: int = 1
    colmap_path: str = "colmap"
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    use_gpu: bool = False
    matcher_type: str = "exhaustive"
    sift_settings: dict[str, Any] = field(default_factory=lambda: {
        "max_num_features": 8192,
        "max_image_size": 3200,
        "first_octave": -1,
        "num_octaves": 4,
        "edge_threshold": 10,
        "peak_threshold": 0.0067,
        "magnification": 3,
    })
    logging_level: str = "INFO"
    min_images: int = 3

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "ProjectConfig":
        target_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        resolved_path = target_path.resolve()

        base_dir = PROJECT_ROOT if resolved_path == DEFAULT_CONFIG_PATH else resolved_path.parent

        raw_data: dict[str, Any] = {}
        if resolved_path.exists():
            with resolved_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                raw_data = loaded

        data_cfg = raw_data.get("data", {}) if isinstance(raw_data.get("data", {}), dict) else {}
        output_cfg = raw_data.get("outputs", {}) if isinstance(raw_data.get("outputs", {}), dict) else {}
        colmap_cfg = raw_data.get("colmap", {}) if isinstance(raw_data.get("colmap", {}), dict) else {}
        logging_cfg = raw_data.get("logging", {}) if isinstance(raw_data.get("logging", {}), dict) else {}
        validation_cfg = raw_data.get("validation", {}) if isinstance(raw_data.get("validation", {}), dict) else {}
        dense_cfg = raw_data.get("dense", {}) if isinstance(raw_data.get("dense", {}), dict) else {}
        patch_cfg = dense_cfg.get("patch_match", {}) if isinstance(dense_cfg.get("patch_match", {}), dict) else {}
        fusion_cfg = dense_cfg.get("stereo_fusion", {}) if isinstance(dense_cfg.get("stereo_fusion", {}), dict) else {}
        sift_cfg = colmap_cfg.get("sift", {}) if isinstance(colmap_cfg.get("sift", {}), dict) else {}

        executable_value = colmap_cfg.get("executable") or colmap_cfg.get("executable_path") or "colmap"

        return cls(
            root_dir=_resolve_path(raw_data.get("root_dir", "."), base_dir),
            image_dir=_resolve_path(data_cfg.get("images_dir", "data/images"), base_dir),
            masks_dir=_resolve_path(data_cfg.get("masks_dir", "data/masks"), base_dir),
            output_dir=_resolve_path(output_cfg.get("base_dir", "outputs"), base_dir),
            colmap_dir=_resolve_path(output_cfg.get("colmap_dir", "outputs/colmap"), base_dir),
            sparse_dir=_resolve_path(output_cfg.get("sparse_dir", "outputs/colmap/sparse"), base_dir),
            logs_dir=_resolve_path(output_cfg.get("logs_dir", "outputs/logs"), base_dir),
            archive_dir=_resolve_path(output_cfg.get("archive_dir", "outputs/archive"), base_dir),
            database_path=_resolve_path(output_cfg.get("database_path", "outputs/colmap/database.db"), base_dir),
            dense_model_id=str(dense_cfg.get("sparse_model", "0")),
            dense_dir=_resolve_path(dense_cfg.get("output_dir", "outputs/colmap/dense/0"), base_dir),
            dense_max_image_size=int(dense_cfg.get("max_image_size", -1)),
            dense_patch_match_threads=int(patch_cfg.get("num_threads", 1)),
            dense_stereo_fusion_threads=int(fusion_cfg.get("num_threads", 1)),
            colmap_path=str(executable_value),
            camera_model=str(colmap_cfg.get("camera_model", "SIMPLE_RADIAL")),
            single_camera=bool(colmap_cfg.get("single_camera", True)),
            use_gpu=bool(colmap_cfg.get("use_gpu", False)),
            matcher_type=str(colmap_cfg.get("matcher_type", "exhaustive")),
            sift_settings={
                "max_num_features": int(sift_cfg.get("max_num_features", 8192)),
                "max_image_size": int(sift_cfg.get("max_image_size", 3200)),
                "num_threads": int(colmap_cfg.get("num_threads", 1)),
                "first_octave": int(sift_cfg.get("first_octave", -1)),
                "num_octaves": int(sift_cfg.get("num_octaves", 4)),
                "edge_threshold": int(sift_cfg.get("edge_threshold", 10)),
                "peak_threshold": float(sift_cfg.get("peak_threshold", 0.0067)),
                "magnification": int(sift_cfg.get("magnification", 3)),
            },
            logging_level=str(logging_cfg.get("level", "INFO")).upper(),
            min_images=int(validation_cfg.get("min_images", 3)),
        )


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    return ProjectConfig.from_yaml(config_path)
