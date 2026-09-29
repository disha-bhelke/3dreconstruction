# 3D Reconstruction from 2D Images

This project implements the first stage of a 3D reconstruction workflow using COLMAP for the Structure-from-Motion (SfM) backbone. The current focus is on a reliable sparse reconstruction pipeline and not on dense reconstruction or web viewing.

## Project goal

The goal is to reconstruct a 3D scene or object from overlapping 2D images by estimating camera poses, triangulating 3D points, and producing a sparse point cloud.

## Pipeline overview

Stage 1: Sparse reconstruction (current focus)
1. Validate input images
2. Initialize a COLMAP SQLite database
3. Extract SIFT features with COLMAP
4. Match features between images
5. Run the mapper to estimate camera poses and sparse geometry
6. Validate the reconstruction
7. Optionally inspect the sparse point cloud

This repository intentionally stops before dense reconstruction, MVS, mesh generation, and texturing.

## What Structure-from-Motion means

Structure-from-Motion is the process of recovering the 3D geometry of a scene and the poses of the cameras that captured it using only overlapping images. It typically begins with feature detection and matching, then camera calibration, pose estimation, triangulation, and bundle adjustment.

## What SIFT features are

SIFT (Scale-Invariant Feature Transform) detects distinctive keypoints that are stable across changes in scale, rotation, and partial viewpoint variation. In a COLMAP pipeline, SIFT features are used to identify matching points between multiple images.

## What feature matching does

Feature matching compares extracted keypoints between images to find correspondences. These correspondences are the basis for estimating camera motion and reconstructing 3D points.

## What COLMAP does

COLMAP is an open-source computer vision pipeline for SfM and MVS. In this project, COLMAP performs the actual feature extraction, feature matching, geometric verification, and sparse reconstruction steps.

## What the Mapper does

The mapper uses the image pairs, correspondences, and geometric constraints to estimate camera poses and triangulate 3D points. The result is a sparse point cloud and camera pose graph.

## What a sparse point cloud is

A sparse point cloud is a set of 3D points reconstructed from matched image features. It is not a dense volumetric representation and is usually a lightweight representation useful for reconstruction validation and inspection.

## Installation requirements

- Python 3.11+
- COLMAP installed and available on your PATH
- OpenCV
- NumPy
- PyYAML
- tqdm
- pytest
- Open3D for optional visualization

## COLMAP installation

1. Download COLMAP from the official project website or release page.
2. Install it on your Windows machine.
3. Ensure the `colmap` executable is available in your shell.
4. If needed, add the COLMAP installation folder to your PATH.

On Windows, a common setup is:

- Install COLMAP to a folder such as `C:\Program Files\COLMAP\bin`
- Add that `bin` folder to your `PATH`
- Verify with:

  `colmap --help`

If `colmap` is not found, update the `colmap.executable_path` value in [configs/config.yaml](configs/config.yaml).

## Python environment setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to add images

Place input images in the `data/images` folder. Use overlapping photos of the same object or scene from multiple viewpoints.

Recommended:
- keep the folder focused on the target object or scene
- avoid near-duplicate frames
- prefer well-lit, in-focus images
- keep file formats to `.jpg`, `.jpeg`, `.png`, or `.webp`

## How to run each Stage 1 step

The project is organized so that individual scripts can be run independently during debugging.

```powershell
python scripts/validate_images.py
python scripts/extract_features.py
python scripts/match_features.py
python scripts/run_mapper.py
python scripts/validate_reconstruction.py
python scripts/visualize_sparse.py
```

You can also run the package entry point:

```powershell
python -m src.pipeline --help
```

## Stage 1B: COLMAP database initialization and SIFT extraction

Stage 1B creates the COLMAP database and extracts SIFT features from the validated image set. It does not perform matching or reconstruction.

What this stage does:
- validates the configured input image directory
- ensures the output folders exist
- validates the configured COLMAP executable
- initializes or reuses the SQLite database at `outputs/colmap/database.db`
- calls `colmap feature_extractor`
- stores the generated descriptors and keypoints in the database

Why features are needed:
- feature extraction identifies stable keypoints such as corners and texture-rich regions
- those points are later matched across different images to estimate geometry
- without extracted features, matching and reconstruction cannot proceed

What the COLMAP database contains:
- image records for the input set
- camera metadata
- feature keypoints per image
- descriptor data for the extracted SIFT features

How to run Stage 1B:

```powershell
python scripts/extract_features.py
```

To recreate the database from scratch before extraction:

```powershell
python scripts/extract_features.py --reset
```

The reset option deletes only the file `outputs/colmap/database.db` and leaves the source images, masks, source code, and logs untouched.

CPU and GPU behavior:
- the project is portable and uses configuration to decide how COLMAP runs
- on the current laptop, the installed COLMAP build is CPU-only and the config uses `use_gpu: false`
- on a CUDA-enabled machine, the same project can switch to `use_gpu: true`
- if GPU mode is requested but the installed build does not support CUDA, the script raises a clear error instead of silently falling back

Expected successful output:
- the database file exists at `outputs/colmap/database.db`
- the database can be opened with Python's `sqlite3` module
- the number of image records corresponds to the number of valid input images
- keypoints and descriptors are stored for the images

## Expected outputs

After a successful Stage 1 run, you should see:

- a validated set of input images
- a COLMAP SQLite database at `outputs/colmap/database.db`
- extracted feature data via COLMAP
- matched feature data
- a sparse model stored under `outputs/colmap/sparse`
- logs in `outputs/logs`

## Stage 1C: exhaustive feature matching

Stage 1C creates feature correspondences between pairs of images using the SIFT descriptors produced in Stage 1B. It does not create a 3D reconstruction.

For the current 34-image dataset, exhaustive matching considers:

```text
34 * 33 / 2 = 561 possible image pairs
```

COLMAP compares descriptor data for those pairs, then performs its normal geometric verification. Raw matches are stored in the `matches` table, while geometrically verified results are stored in the `two_view_geometries` table when exposed by the installed COLMAP version.

Run Stage 1C with:

```powershell
python scripts/match_features.py
```

The existing database at `outputs/colmap/database.db` is reused. The script validates that Stage 1B image records, keypoints, and descriptors exist before invoking:

```text
colmap exhaustive_matcher --database_path outputs/colmap/database.db
```

The actual executable path and CPU/GPU settings come from `configs/config.yaml`. The current no-CUDA build uses `use_gpu: false` and `num_threads: 1`.

Rerunning `python scripts/match_features.py` is the safe rerun procedure. It recomputes matching data through COLMAP while preserving the Stage 1B keypoints and descriptors. There is intentionally no `--reset` option because deleting the complete database would also delete the Stage 1B data required for matching.

The script reports the database path, image count, possible pairs, matched pairs, total raw matches, and geometrically verified pairs/matches when those values are available from the database schema. The number of valid matches is data-dependent; it is not assumed to equal 561 pairs.

Stage 1C does not run the Mapper, estimate camera poses, triangulate 3D points, or produce a sparse reconstruction. Mapper/SfM is the next stage.

## Stage 1D: Mapper and sparse Structure-from-Motion

Stage 1D produces the first actual 3D reconstruction. It is a sparse reconstruction created by COLMAP Mapper from the Stage 1B keypoints/descriptors and the Stage 1C raw and geometrically verified matches.

Structure-from-Motion estimates camera poses and orientations from image correspondences. COLMAP then triangulates matched observations into 3D points and uses bundle adjustment to refine camera and point parameters together.

Run Stage 1D with:

```powershell
python scripts/run_mapper.py
```

The Mapper uses:

```text
--database_path outputs/colmap/database.db
--image_path data/images
--output_path outputs/colmap/sparse
```

The sparse output may contain one or more model directories such as `0`, `1`, and so on. Each valid model is inspected through COLMAP's binary `cameras.bin`, `images.bin`, and `points3D.bin` files. The script reports total input images separately from registered images, because a successful reconstruction may not register every input image.

To safely rerun Mapper:

```powershell
python scripts/run_mapper.py --reset
```

The reset removes only `outputs/colmap/sparse`. It never deletes the database, keypoints, descriptors, matches, geometries, or source images. A normal run refuses to overwrite an existing sparse output and explains how to use `--reset`.

The current no-CUDA installation runs Mapper and bundle adjustment in CPU mode. Mapper does not require a CUDA GPU for this project.

Stage 1D does not run dense reconstruction, PatchMatch Stereo, stereo fusion, meshing, texturing, Open3D processing, or a viewer. Those are later stages.

## Complete Stage 1 orchestrator

After replacing the photographs in `data/images`, run the complete sparse workflow with one command:

```powershell
python scripts/run_stage1.py
```

The orchestrator performs these steps in order and stops at the first failure:

1. Image validation
2. Fresh COLMAP database creation through feature extraction
3. SIFT feature extraction
4. Exhaustive matching and geometric verification
5. Mapper / SfM
6. Sparse reconstruction validation
7. Final summary

Before starting, it moves any existing `outputs/colmap/database.db` and `outputs/colmap/sparse` into a timestamped folder under `outputs/archive`, for example `outputs/archive/stage1_2026-09-29_113000`. It then creates a fresh database and sparse directory, so old images, features, matches, and reconstructions cannot be mixed with the replacement dataset. `data/images` is never archived or modified.

The final summary selects the model with the highest registered-image count, then uses 3D-point count as a tie breaker. It reports input images, registered images, registration percentage, cameras, sparse points, observations, mean reprojection error, primary model path, and archive path.

The orchestrator writes a run log as `outputs/logs/stage1_YYYY-MM-DD_HHMMSS.log`. It uses the configured CPU-only COLMAP installation and does not install CUDA or download models.

This command runs Stage 1 only. It does not perform dense reconstruction, PatchMatch Stereo, stereo fusion, meshing, texturing, Open3D processing, GLB export, React, Three.js, or a web viewer.

## Stage 2A: dense reconstruction preparation

Stage 2 will eventually follow this flow:

```text
Sparse model
-> image undistortion
-> PatchMatch Stereo
-> Stereo Fusion
-> dense point cloud
```

The current selected sparse model is `outputs/colmap/sparse/0`. Stage 2A validates that model, checks its registered images against `data/images`, checks the COLMAP dense commands, and can run image undistortion into `outputs/colmap/dense/0` without modifying the sparse model or source images.

Run Stage 2A with:

```powershell
python scripts/prepare_dense.py
```

Use `--reset` only when explicitly rebuilding the dense workspace:

```powershell
python scripts/prepare_dense.py --reset
```

The reset removes only `outputs/colmap/dense/0`. It never removes `data/images`, `outputs/colmap/database.db`, or `outputs/colmap/sparse/0`.

The installed COLMAP 4.2.0 commands `image_undistorter`, `patch_match_stereo`, and `stereo_fusion` are present. However, a bounded PatchMatch probe reports:

```text
Dense stereo reconstruction requires CUDA, which is not available on your system.
```

Therefore PatchMatch Stereo and Stereo Fusion are not currently supported by this no-CUDA build and are not run by Stage 2A. The current sparse model also calibrates a `960x1280` camera while the source JPEGs decode as `967x1299`, so image undistortion reports that source/model dimension mismatch instead of producing invalid dense images.

There is no Stage 2B command implemented yet. Dense processing must wait for a CUDA-capable COLMAP build and compatible source images/model calibration. Stage 3 surface reconstruction is not implemented.

## Common errors

- `COLMAP executable was not found. Check colmap_path in configs/config.yaml.`
- `No images were found in data/images/.`
- `COLMAP feature extraction failed. Check outputs/logs/ for details.`
- `No registered images were produced by the mapper. This usually means the images do not contain enough overlap or reliable features.`

## How to reset the reconstruction

Use the `--reset` option on the COLMAP-related scripts when they are available. This safely removes previously generated reconstructions and recreates the database and sparse outputs.

## How to interpret the reconstruction statistics

The validation script reports:
- total input images
- successfully registered images
- registration percentage
- number of cameras
- number of sparse 3D points
- reconstructed model directories

If a value cannot be measured reliably, the script reports it as unavailable rather than guessing.

## Current project status

This repository is currently set up for the first checkpoint only: image validation.

The larger COLMAP pipeline steps are scaffolded but not yet executed here until the validation step is confirmed.
