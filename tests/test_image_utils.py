from __future__ import annotations

import cv2
import numpy as np

from src.image_utils import discover_images, validate_image_dataset, validate_image_file


def test_discover_images_filters_supported_extensions(tmp_path):
    valid_png = tmp_path / "sample.png"
    valid_jpg = tmp_path / "sample.jpg"
    invalid_txt = tmp_path / "notes.txt"

    cv2.imwrite(str(valid_png), np.zeros((32, 48, 3), dtype=np.uint8))
    cv2.imwrite(str(valid_jpg), np.zeros((16, 24, 3), dtype=np.uint8))
    invalid_txt.write_text("this is not an image", encoding="utf-8")

    found = discover_images(tmp_path)
    names = [path.name for path in found]

    assert valid_png.name in names
    assert valid_jpg.name in names
    assert invalid_txt.name not in names


def test_validate_image_file_reads_valid_image(tmp_path):
    image_path = tmp_path / "valid.png"
    cv2.imwrite(str(image_path), np.zeros((90, 120, 3), dtype=np.uint8))

    result = validate_image_file(image_path)

    assert result["is_valid"] is True
    assert result["width"] == 120
    assert result["height"] == 90
    assert result["channels"] == 3


def test_validate_image_dataset_marks_corrupt_file(tmp_path):
    valid_image = tmp_path / "good.png"
    invalid_image = tmp_path / "bad.jpg"

    cv2.imwrite(str(valid_image), np.zeros((50, 50, 3), dtype=np.uint8))
    invalid_image.write_bytes(b"not a real jpeg")

    summary = validate_image_dataset([valid_image, invalid_image], min_images=2)

    assert summary["total_images"] == 2
    assert summary["valid_images"] == 1
    assert summary["usable"] is False
    assert summary["invalid_images"]
