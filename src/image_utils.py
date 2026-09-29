from __future__ import annotations

import cv2
from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    image_dir = Path(directory)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {image_dir}")

    if recursive:
        iterable = image_dir.rglob("*")
    else:
        iterable = image_dir.iterdir()

    images = [
        path
        for path in iterable
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    return sorted(images)


def validate_image_file(path: str | Path) -> dict:
    image_path = Path(path)
    result = {
        "path": str(image_path),
        "filename": image_path.name,
        "width": None,
        "height": None,
        "channels": None,
        "is_valid": False,
        "warning": None,
    }

    if not image_path.exists():
        result["warning"] = "File does not exist"
        return result

    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        result["warning"] = f"Unsupported file type: {image_path.suffix}"
        return result

    image_array = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_array is None:
        result["warning"] = "OpenCV could not read the image"
        return result

    height, width = image_array.shape[:2]
    channels = 1
    if len(image_array.shape) == 3:
        channels = image_array.shape[2]

    result["width"] = int(width)
    result["height"] = int(height)
    result["channels"] = int(channels)
    result["is_valid"] = True
    return result


def validate_image_dataset(images: list[str | Path], min_images: int = 3) -> dict:
    records = [validate_image_file(image) for image in images]
    invalid_records = [record for record in records if not record["is_valid"]]

    valid_records = [record for record in records if record["is_valid"]]
    resolutions = {(
        record["width"],
        record["height"],
    ) for record in valid_records}

    warnings: list[str] = []
    if len(valid_records) > 1 and len(resolutions) > 1:
        warnings.append("Image resolutions are inconsistent across the dataset.")

    usable = len(valid_records) >= min_images and not invalid_records
    return {
        "total_images": len(records),
        "valid_images": len(valid_records),
        "invalid_images": invalid_records,
        "records": records,
        "warnings": warnings,
        "min_images": min_images,
        "usable": usable,
    }
