#!/usr/bin/env python3
"""Placeholder for sparse-point-cloud visualization.

This stage could use Open3D to inspect the sparse reconstruction and camera poses.
It is intentionally not implemented in the current checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a sparse COLMAP reconstruction.")
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    print("Visualization is not implemented in the current checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
