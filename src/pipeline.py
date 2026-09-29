from __future__ import annotations

import argparse
from pathlib import Path


def validate_images_step() -> None:
    """Placeholder for the Step 1A image-validation stage."""
    raise NotImplementedError("This step is implemented in scripts/validate_images.py.")


def initialize_colmap_database() -> None:
    """Future Stage 1B: create the COLMAP database."""
    raise NotImplementedError("This stage is intentionally not implemented in the current checkpoint.")


def extract_features() -> None:
    """Future Stage 1C: feature extraction with COLMAP."""
    raise NotImplementedError("This stage is intentionally not implemented in the current checkpoint.")


def match_features() -> None:
    """Future Stage 1D: feature matching with COLMAP."""
    raise NotImplementedError("This stage is intentionally not implemented in the current checkpoint.")


def run_mapper() -> None:
    """Future Stage 1E: sparse reconstruction with COLMAP mapper."""
    raise NotImplementedError("This stage is intentionally not implemented in the current checkpoint.")


def validate_reconstruction() -> None:
    """Future Stage 1F: reconstruction validation."""
    raise NotImplementedError("This stage is intentionally not implemented in the current checkpoint.")


def visualize_sparse() -> None:
    """Optional future visualization stage."""
    raise NotImplementedError("This stage is intentionally not implemented in the current checkpoint.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 sparse-reconstruction pipeline scaffold for the 3D Reconstruction project.",
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Run the image-dataset validation checkpoint.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.validate_images:
        print("Use the script 'python scripts/validate_images.py' for the current checkpoint.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
