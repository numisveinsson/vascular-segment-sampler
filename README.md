# Vascular Segment Sampler

This repository contains code to process data used to train machine learning methods for geometric modeling of blood vessels using medical image data. The data used is:
    1. Medical image scans
    2. Ground truth segmentations
    3. Centerlines
    4. Surface meshes (if applicable)

The fundamental idea is to 'piece up' vasculature into hundreds/thousands of vascular segments. These segments can be:
    1. Image subvolumes/patches (3D/2D)
    2. Local surface representations
    3. Local centerline segments
    4. Local outlet/bifurcation/size/orientation information
    etc.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Main Scripts](#main-scripts)
- [Preprocessing Scripts](#preprocessing-scripts)
- [Global Processing Scripts](#global-processing-scripts)
- [Cardiac Processing Scripts](#cardiac-processing-scripts)
- [Modules](#modules)
- [Testing](#testing)

## Installation

Install the required dependencies using conda:

```bash
conda env create -f environment.yml
conda activate vascular-segment-sampler
```

Or create the environment with a custom name:

```bash
conda env create -f environment.yml -n vascular-segment-sampler
conda activate vascular-segment-sampler
```

## Project Structure

```
vascular-segment-sampler/
├── config/              # Configuration YAML files
├── modules/             # Shared utility modules
│   ├── vtk_functions.py # VTK-related functions
│   ├── sitk_functions.py # SimpleITK-related functions
│   └── ...
├── preprocessing/       # Image format conversion and preprocessing
├── global/             # Global processing scripts
├── cardiac/            # Cardiac-specific processing
├── tests/              # Unit tests
└── ...
```

## Configuration

The main configuration file is `config/global.yaml`. This file contains settings for:
- `DATASET_NAME`: Name of the dataset
- `IMG_EXT`: Image file extension
- Various processing flags (see below)

**Note:** The following parameters are now provided as command-line arguments instead of config file entries:
- `DATA_DIR`: Use `--data_dir` argument (required)
- `TESTING`: Use `--testing` flag (optional, defaults to config value or False)
- `MODALITY`: Use `--modality` argument (optional, defaults to config value)
- `VALIDATION_PROP`: Use `--validation_prop` argument (optional, defaults to config value)
- `MAX_SAMPLES`: Use `--max_samples` argument (optional, defaults to config value)

### Key Configuration Parameters

- `DATASET_NAME`: Name of the dataset ('vmr' or 'other')
- `IMG_EXT`: File extension for images (e.g., '.mha', '.vti', '.nrrd')
- `ANATOMY`: Type of anatomy ('ALL' or specific list)
- `EXTRACT_VOLUMES`: Extract volumes from images
- `ROTATE_VOLUMES`: Rotate volumes during processing
- `RESAMPLE_VOLUMES`: Resample volumes
- `RESAMPLE_SIZE`: Target size for resampling
- `FIXED_EXTRACT_SIZE`: Optional `[nx, ny, nz]` voxel dimensions. When set, extracts always the same voxel size (and same physical size for same spacing). Omit for radius-based variable extraction.
- `WRITE_SAMPLES`: Write sample data
- `WRITE_IMG`: Write image files
- `WRITE_SURFACE`: Write surface meshes
- `WRITE_CENTERLINE`: Write centerline data
- `WRITE_CROSS_SECTIONAL`: Write cross-sectional data
- `NUM_CROSS_SECTIONS`: Number of cross-sections
- `RESAMPLE_CROSS_IMG`: Cross-sectional image resampling size
- `WRITE_TRAJECTORIES`: Write trajectory data
- `N_SLICES`: Number of slices for processing

## Main Scripts

### main.py

Main script for parallel processing of data sampling from multiple cases. Uses multiprocessing to speed up processing.

**Usage:**
```bash
python3 main.py \
    --config_name global \
    --data_dir /path/to/data \
    --outdir ./extracted_data/ \
    --num_cores 4 \
    --perc_dataset 1.0
```

**Required Arguments:**
- `--config_name` / `-config_name`: Name of configuration file (without .yaml extension)
  - Example: `--config_name global` uses `config/global.yaml`
- `--data_dir` / `-data_dir`: Directory where input data is stored

**Optional Arguments:**
- `--outdir` / `-outdir`: Output directory for extracted data (default: `./extracted_data/`)
- `--perc_dataset` / `-perc_dataset`: Percentage of dataset to use, 0.0 to 1.0 (default: `1.0`)
- `--num_cores` / `-num_cores`: Number of CPU cores to use for parallel processing (default: `1`)
- `--start_from` / `-start_from`: Start processing from case number (default: `0`)
- `--end_at` / `-end_at`: End processing at case number, -1 for all cases (default: `-1`)
- `--testing` / `-testing`: Enable testing mode (uses TEST_CASES from config instead of training cases). If not provided, uses config value or defaults to False
- `--validation_prop` / `-validation_prop`: Validation set proportion (0.0-1.0). If not provided, uses config value
- `--max_samples` / `-max_samples`: Maximum number of samples to extract. Useful for quick testing. If not provided, uses config value
- `--modality` / `-modality`: Imaging modality: `CT`, `MR`, or comma-separated list (e.g., `CT,MR`). If not provided, uses config value

**Examples:**
```bash
# Process all cases with 4 cores
python3 main.py --config_name global --data_dir /path/to/data --num_cores 4

# Process 50% of dataset with 8 cores
python3 main.py --config_name global --data_dir /path/to/data --perc_dataset 0.5 --num_cores 8

# Process cases 10 to 20
python3 main.py --config_name global --data_dir /path/to/data --start_from 10 --end_at 20

# Custom output directory
python3 main.py --config_name global --data_dir /path/to/data --outdir /path/to/output

# Enable testing mode and limit samples for quick test
python3 main.py --config_name global --data_dir /path/to/data --testing --max_samples 100

# Process only CT modality with custom validation split
python3 main.py --config_name global --data_dir /path/to/data --modality CT --validation_prop 0.15

# Process both CT and MR modalities
python3 main.py --config_name global --data_dir /path/to/data --modality CT,MR
```

The script uses configuration from the specified YAML file in `config/` and processes cases in the dataset according to the provided arguments.

## Preprocessing Scripts

All preprocessing scripts use command-line arguments. Default values are provided where appropriate.

### change_img_format.py

Convert images between different formats (.mha, .vti, .nrrd, .nii.gz, etc.).

**Usage:**
```bash
python preprocessing/change_img_format.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --input_format .nrrd \
    --output_format .mha \
    --label_if_string seg
```

**Arguments:**
- `--input_dir` / `--input-dir`: Input directory (default: `./data/images/`)
- `--output_dir` / `--output-dir`: Output directory (default: inferred from input_dir)
- `--input_format` / `--input-format`: Input file extension (default: `.nrrd`)
- `--output_format` / `--output-format`: Output file extension (default: `.mha`)
- `--label`: Treat all files as label segmentations (flag)
- `--label_if_string` / `--label-if-string`: Auto-detect labels by filename string
- `--rem_str` / `--rem-str`: String to remove from filenames
- `--surface`: Also create surface meshes for label images (flag)

### change_img_resample.py

Resample images to target size or spacing.

**Usage:**
```bash
# Resample to target spacing
python preprocessing/change_img_resample.py \
    --input_dir /path/to/images \
    --target_spacing 0.03 0.03 0.03 \
    --order 1

# Resample to target size
python preprocessing/change_img_resample.py \
    --input_dir /path/to/images \
    --target_size 512 512 512
```

**Arguments:**
- `--input_dir`: Input directory (default: `./data/images/`)
- `--output_dir`: Output directory (default: inferred from input_dir)
- `--target_size`: Target size [x, y, z] (mutually exclusive with `--target_spacing`, either this or `--target_spacing` is required)
- `--target_spacing`: Target spacing [x, y, z] in mm (mutually exclusive with `--target_size`, either this or `--target_size` is required)
- `--order`: Interpolation order (0=nearest, 1=linear, 2=bspline) (default: `1`)
- `--input_format`: Input file extension (default: `.mha`)
- `--testing_samples`: Filter to specific sample names (optional)
- `--no_skip_existing`: Re-process existing files (by default, skips existing files)

### change_img_scale_coords.py

Scale image spacing/origin and transform coordinate systems.

**Usage:**
```bash
python preprocessing/change_img_scale_coords.py \
    --input_dir /path/to/images \
    --scale 0.1 \
    --change_lps_to_ras \
    --verbose
```

**Arguments:**
- `--input_dir` / `--input-dir`: Input directory (default: `./data/`)
- `--output_dir` / `--output-dir`: Output directory (default: inferred from input_dir)
- `--input_format` / `--input-format`: Input file extension (default: `.mha`)
- `--output_format` / `--output-format`: Output file extension (default: `.mha`)
- `--scale`: Scale factor for spacing (default: 1.0)
- `--scale_origin` / `--scale-origin`: Scale factor for origin (optional)
- `--spacing_file` / `--spacing-file`: CSV file with spacing values (optional)
- `--direction_matrix` / `--direction-matrix`: 3x3 direction matrix (9 values, optional)
- `--flip`: Flip image (flag)
- `--flip_axis` / `--flip-axis`: Axis to flip [x, y, z] (default: `[False, False, False]`)
- `--permute`: Permute axes (flag)
- `--change_lps_to_ras` / `--change-lps-to-ras`: Convert LPS to RAS (flag)
- `--filter_names` / `--filter-names`: Filter files by name (optional)
- `--verbose`: Detailed output (flag)

### change_vtk_scale_coords.py

Scale VTK surface files (.vtp, .stl).

**Usage:**
```bash
python preprocessing/change_vtk_scale_coords.py \
    --input_dir /path/to/surfaces \
    --output_dir /path/to/output \
    --scale_factor 0.1
```

**Arguments:**
- `--input_dir` / `--input-dir`: Input directory (default: `./data/surfaces/`)
- `--output_dir` / `--output-dir`: Output directory (default: inferred from input_dir)
- `--scale_factor` / `--scale-factor`: Scale factor (required)

### change_lps_ras.py

Convert files from LPS to RAS coordinate system.

**Usage:**
```bash
# Single file
python preprocessing/change_lps_ras.py \
    --input_file /path/to/file.vtp \
    --output_file /path/to/output.vtp \
    --file_type vtp

# Batch processing
python preprocessing/change_lps_ras.py \
    --input_dir /path/to/surfaces \
    --output_dir /path/to/output \
    --file_type vtp
```

**Arguments:**
- `--input_file` / `--input-file`: Input file path (for single file processing)
- `--output_file` / `--output-file`: Output file path (required when using `--input_file`)
- `--input_dir` / `--input-dir`: Input directory (for batch processing)
- `--output_dir` / `--output-dir`: Output directory (for batch processing)
- `--file_type` / `--file-type`: Type of file (`vtp` or `image`) (required)
- `--transform_type` / `--transform-type`: `lps_to_ras` or `permute_xyz` (default: `lps_to_ras`)

### compare_imgs.py

Compare image properties and optionally transform images.

**Usage:**
```bash
python preprocessing/compare_imgs.py \
    --image1 /path/to/img1.mha \
    --image2 /path/to/img2.mha \
    --image3 /path/to/img3.mha
```

**Arguments:**
- `--image1`: Path to first image file (for comparison, requires at least `--image2`)
- `--image2`: Path to second image file (for comparison)
- `--image3`: Path to third image file (optional)
- `--vti_file` / `--vti-file`: VTI file to transform (requires `--output_file`)
- `--output_file` / `--output-file`: Output path for transformed VTI file
- `--transform`: Apply 90-degree rotation transform to VTI image (flag)

## Global Processing Scripts

### create_seg_from_surf.py

Create segmentation images from surface meshes.

**Usage:**
```bash
python global/create_seg_from_surf.py \
    --surfaces_dir /path/to/surfaces \
    --images_dir /path/to/images \
    --output_dir /path/to/output
```

**Arguments:**
- `--surfaces_dir` / `--surfaces-dir`: Directory with surface mesh files (.vtp or .stl) (default: `./data/surfaces/`)
- `--images_dir` / `--images-dir`: Directory with image files (default: `./data/images/`)
- `--output_dir` / `--output-dir`: Directory to write output segmentations (default: `./data/truths/`)
- `--img_ext` / `--img-ext`: Input image file extension (default: `.mha`)
- `--output_ext` / `--output-ext`: Output file extension (default: `.mha`)

### create_surf_from_seg.py

Create surface meshes from segmentation images.

**Usage:**
```bash
python global/create_surf_from_seg.py \
    --segmentations_dir /path/to/segmentations \
    --output_dir /path/to/output \
    --smooth \
    --keep_largest
```

**Arguments:**
- `--segmentations_dir` / `--segmentations-dir`: Directory with segmentation images (default: `./data/truths/`)
- `--output_dir` / `--output-dir`: Directory to write output surfaces (default: inferred from segmentations_dir)
- `--spacing_file` / `--spacing-file`: CSV file with spacing values (optional)
- `--filter_string` / `--filter-string`: Filter images by string (optional)
- `--smooth`: Apply smoothing to surfaces (flag)
- `--keep_largest` / `--keep-largest`: Keep only largest connected component (flag)
- `--img_ext` / `--img-ext`: Input image extension (default: `.mha`)
- `--output_ext` / `--output-ext`: Output surface extension (default: `.vtp`)

## Cardiac Processing Scripts

### combine_segs.py

Combine cardiac and vascular segmentations.

**Usage:**
```bash
python cardiac/combine_segs.py \
    --meshes_dir /path/to/meshes \
    --images_dir /path/to/images \
    --vascular_dir /path/to/vascular_segs
```

**Arguments:**
- `--meshes_dir` / `--meshes-dir`: Directory with cardiac mesh files (.vtp) (default: `./data/meshes/`)
- `--images_dir` / `--images-dir`: Directory with image files (.vti) (default: inferred from meshes_dir)
- `--vascular_dir` / `--vascular-dir`: Directory with vascular segmentation files (.vti) (default: inferred from meshes_dir)
- `--write_all` / `--write-all`: Write all intermediate files (default: True)
- `--no_write_all` / `--no-write-all`: Skip writing intermediate files
- `--no_valve` / `--no-valve`: Process without valve (default: True)
- `--with_valve` / `--with-valve`: Process with valve

## Modules

The repository uses a modular structure with shared functions:

### modules/vtk_functions.py

Central repository for VTK-related utility functions including:
- `vtk_marching_cube`, `vtk_marching_cube_multi`
- `smooth_polydata`, `decimation`
- `exportSitk2VTK`, `exportPython2VTK`
- `convertPolyDataToImageData`
- `vtkImageResample`
- And many more...

### modules/sitk_functions.py

SimpleITK-related utility functions including:
- `eraseBoundary`
- `convert_seg_to_surfs`
- `read_image`, `write_image`
- `sitk_to_numpy`, `numpy_to_sitk`
- And more...

## Command-Line Arguments

All scripts use command-line arguments exclusively. Default values are provided for optional arguments, making scripts easier to use while maintaining flexibility.

**Common Patterns:**
- Directory arguments typically default to `./data/` with appropriate subdirectories
- Output directories are often inferred from input directories if not explicitly provided
- Required arguments are clearly indicated in the help text

**Example:**
```bash
# Using explicit paths
python preprocessing/change_img_format.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --input_format .nrrd \
    --output_format .mha

# Using defaults (will use ./data/images/ and infer output directory)
python preprocessing/change_img_format.py \
    --input_format .nrrd \
    --output_format .mha
```

For detailed argument information, use the `--help` flag on any script:
```bash
python <script_name>.py --help
```

## Data Structure

The code expects data to be stored in a particular folder structure. The base directory can be specified via command-line arguments (defaults to `./data/`):

```
<base_directory>/
├── images/
│   └── case0.mha
├── centerlines/
│   └── case0.vtp
├── truths/
│   └── case0.mha
└── surfaces/  (if applicable)
    └── case0.vtp
```

Note: The exact structure may vary depending on the script and dataset. Check individual script documentation for specific requirements.

## Testing

Run tests with:

```bash
python -m pytest tests/
```

Or run specific tests:

```bash
python -m pytest tests/test_create_seg_from_surf.py
python -m pytest tests/test_create_surf_from_seg.py
```

## Additional Information

### Code Refactoring

The codebase has been refactored to:
- Consolidate duplicate functions into shared modules
- Remove hardcoded paths (now use command-line arguments exclusively)
- Replace print statements with proper logging
- Remove debug code (`pdb.set_trace()`)
- Remove environment variable dependencies (use command-line arguments only)
- Ensure consistent configuration files

### Help Documentation

All scripts include comprehensive help documentation:

```bash
python <script_name>.py --help
```

This will display:
- Available arguments
- Default values
- Usage examples
- Argument descriptions

## License

See LICENSE file for details.

## Contributing

When contributing:
1. Use the centralized modules for shared functionality
2. Use command-line arguments exclusively (no environment variables or hardcoded paths)
3. Use the logger module instead of print statements
4. Update tests when adding new functionality
5. Follow the existing code style
6. Provide sensible defaults for optional arguments
