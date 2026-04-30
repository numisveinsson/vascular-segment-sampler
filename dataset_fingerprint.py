"""
Dataset Fingerprinting Tool

Analyzes a medical imaging dataset with images, surfaces, centerlines, and truth segmentations.
Extracts comprehensive statistics and checks for consistency across modalities.
"""

import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
from tqdm import tqdm


def _normalize_dataset_name(val):
    """YAML/config DATASET_NAME may include stray spaces or wrong case."""
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s else None


def _resolve_cases_from_global_config(global_config, split_mode='full'):
    """
    Resolve case IDs (Legacy Names for VMR) that match a pipeline YAML config.

    Args:
        global_config: Dict from load_yaml with DATA_DIR, DATASET_NAME, etc.
        split_mode:
            - 'full': All cohort cases present under DATA_DIR (spreadsheet filters +
              BAD_CASES exclusion); ignores TEST_CASES train/test split.
            - 'train': Same as pipeline with TESTING=false — excludes TEST_CASES.
            - 'test': Same as pipeline with TESTING=true — only TEST_CASES in cohort.

    Returns:
        Set of allowed base names (str), or None if filtering should not apply.
    """
    dataset_name = _normalize_dataset_name(global_config.get('DATASET_NAME'))
    data_dir = global_config.get('DATA_DIR')
    if not data_dir or not dataset_name:
        return None

    if dataset_name == 'vmr':
        from dataset_dirs.datasets import VMR_dataset, create_dataset

        if split_mode == 'full':
            img_mods = global_config.get('VMR_IMAGE_MODALITIES')
            modality_arg = global_config.get('MODALITY') or ['CT']
            anatomy = global_config.get('ANATOMY')
            Dataset = VMR_dataset(
                data_dir,
                modality_arg,
                anatomy,
                image_modalities=img_mods,
            )
            if Dataset.df is None:
                raise RuntimeError(
                    'VMR cohort metadata unavailable (spreadsheet/VMR_dataset_names.csv). '
                    'Cannot apply --config with split-mode full.'
                )
            cases = Dataset.df['Legacy Name'].tolist()
            cases = Dataset.check_which_cases_in_image_dir(cases)
            bad = global_config.get('BAD_CASES') or []
            return set(c for c in cases if c not in bad)

        allowed = set()
        modalities = global_config.get('MODALITY') or ['CT']
        gc = dict(global_config)
        gc['TESTING'] = split_mode == 'test'
        for modality in modalities:
            allowed.update(create_dataset(gc, modality))
        return allowed

    if dataset_name == 'other':
        if split_mode == 'full':
            # Dataset layout varies; fingerprint whole folder unless split is specified.
            return None

        from dataset_dirs.datasets import get_dataset_cases

        img_ext = global_config.get('IMG_EXT', '.mha')
        return set(get_dataset_cases(
            data_dir,
            img_ext,
            global_config.get('TEST_CASES') or [],
            testing=(split_mode == 'test'),
        ))

    return None


def load_global_config_with_data_dir(config_path, data_dir_override):
    """Load YAML config and set DATA_DIR from the fingerprint dataset root."""
    from modules import io

    config_path = os.path.abspath(os.path.expanduser(config_path))
    cfg = io.load_yaml(config_path)
    if cfg is None:
        raise ValueError(f'Could not load config: {config_path}')
    cfg = dict(cfg)
    cfg['DATA_DIR'] = os.path.abspath(data_dir_override)
    if 'BAD_CASES' not in cfg:
        cfg['BAD_CASES'] = []
    if 'TEST_CASES' not in cfg:
        cfg['TEST_CASES'] = []
    return cfg


# Extensions used when mapping images/truths filenames to case IDs (must stay consistent).
_VOLUME_CASE_EXTENSIONS = ('.mha', '.nii.gz', '.nii', '.nrrd', '.vti', '.vtk')


def _stem_case_filename(filename):
    """Basename without extension, same convention as get_base_names for volume files."""
    name = os.path.basename(filename)
    for ext in _VOLUME_CASE_EXTENSIONS:
        if name.endswith(ext):
            return name.replace(ext, '')
    return None


def get_base_names(folder, extensions=['.mha', '.nii.gz', '.nii', '.nrrd', '.vti', '.vtp', '.stl']):
    """Get base filenames without extensions from a folder."""
    if not os.path.exists(folder):
        return set()
    
    files = os.listdir(folder)
    base_names = set()
    
    for f in files:
        for ext in extensions:
            if f.endswith(ext):
                base_names.add(f.replace(ext, ''))
                break
    
    return base_names


def analyze_image(img_path):
    """Extract comprehensive statistics from a medical image."""
    try:
        img = sitk.ReadImage(img_path)
        arr = sitk.GetArrayFromImage(img)
        
        stats = {
            'size': img.GetSize(),
            'spacing': img.GetSpacing(),
            'origin': img.GetOrigin(),
            'direction': img.GetDirection(),
            'pixel_type': img.GetPixelIDTypeAsString(),
            'dimensions': img.GetDimension(),
            'intensity_min': float(np.min(arr)),
            'intensity_max': float(np.max(arr)),
            'intensity_mean': float(np.mean(arr)),
            'intensity_std': float(np.std(arr)),
            'intensity_median': float(np.median(arr)),
            'non_zero_voxels': int(np.count_nonzero(arr)),
            'total_voxels': int(arr.size),
        }
        return stats, None
    except Exception as e:
        return None, str(e)


def analyze_segmentation(seg_path):
    """Extract statistics from a segmentation mask."""
    try:
        seg = sitk.ReadImage(seg_path)
        arr = sitk.GetArrayFromImage(seg)
        
        unique_labels = np.unique(arr)
        label_counts = {int(label): int(np.sum(arr == label)) for label in unique_labels}
        
        stats = {
            'size': seg.GetSize(),
            'spacing': seg.GetSpacing(),
            'unique_labels': [int(x) for x in unique_labels],
            'num_labels': len(unique_labels),
            'label_counts': label_counts,
            'pixel_type': seg.GetPixelIDTypeAsString(),
        }
        
        # Calculate volume percentages
        total_voxels = arr.size
        label_percentages = {int(label): (count / total_voxels * 100) 
                           for label, count in label_counts.items()}
        stats['label_percentages'] = label_percentages
        
        return stats, None
    except Exception as e:
        return None, str(e)


def get_image_physical_bounds(img_path):
    """Get the physical bounds of an image in world coordinates."""
    try:
        img = sitk.ReadImage(img_path)
        size = img.GetSize()
        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = np.array(img.GetDirection()).reshape(3, 3)
        
        # Calculate the physical extent in each direction
        # The max corner is at origin + direction * (size - 1) * spacing
        max_corner = origin + direction @ np.array([(size[i] - 1) * spacing[i] for i in range(3)])
        
        # Get min and max for each axis
        bounds = [
            min(origin[0], max_corner[0]), max(origin[0], max_corner[0]),
            min(origin[1], max_corner[1]), max(origin[1], max_corner[1]),
            min(origin[2], max_corner[2]), max(origin[2], max_corner[2])
        ]
        return bounds, None
    except Exception as e:
        return None, str(e)


def check_points_in_bounds(points_array, image_bounds):
    """Check if points are within image bounds. Returns dict with stats."""
    if image_bounds is None:
        return None
    
    x_min, x_max, y_min, y_max, z_min, z_max = image_bounds
    
    out_of_bounds = []
    for i, point in enumerate(points_array):
        x, y, z = point
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            out_of_bounds.append(i)
    
    return {
        'total_points': len(points_array),
        'out_of_bounds_count': len(out_of_bounds),
        'out_of_bounds_indices': out_of_bounds[:100] if len(out_of_bounds) > 100 else out_of_bounds,  # Limit to first 100
        'all_points_contained': len(out_of_bounds) == 0,
    }


def analyze_centerline(cent_path, image_bounds=None):
    """Extract statistics from a centerline VTP file."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(cent_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        num_points = polydata.GetNumberOfPoints()
        num_lines = polydata.GetNumberOfLines()
        
        stats = {
            'num_points': num_points,
            'num_lines': num_lines,
            'num_cells': polydata.GetNumberOfCells(),
        }
        
        # Get bounds
        bounds = polydata.GetBounds()
        stats['bounds'] = bounds
        stats['bounding_box_size'] = [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ]
        
        # Check if points are within image bounds
        if image_bounds is not None:
            points = polydata.GetPoints()
            points_array = np.array([points.GetPoint(i) for i in range(num_points)])
            containment = check_points_in_bounds(points_array, image_bounds)
            stats['containment_check'] = containment
        
        # Extract radius if available
        point_data = polydata.GetPointData()
        if point_data.HasArray('MaximumInscribedSphereRadius'):
            radii = vtk_to_numpy(point_data.GetArray('MaximumInscribedSphereRadius'))
            stats['radius_min'] = float(np.min(radii))
            stats['radius_max'] = float(np.max(radii))
            stats['radius_mean'] = float(np.mean(radii))
            stats['radius_std'] = float(np.std(radii))
        
        # Count branches (bifurcations)
        if point_data.HasArray('BifurcationId'):
            bifurc_ids = vtk_to_numpy(point_data.GetArray('BifurcationId'))
            unique_bifurcations = len(np.unique(bifurc_ids[bifurc_ids > 0]))
            stats['num_bifurcations'] = unique_bifurcations
        
        # Estimate total length
        points = polydata.GetPoints()
        total_length = 0.0
        lines = polydata.GetLines()
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        
        while lines.GetNextCell(id_list):
            for i in range(id_list.GetNumberOfIds() - 1):
                p1 = np.array(points.GetPoint(id_list.GetId(i)))
                p2 = np.array(points.GetPoint(id_list.GetId(i + 1)))
                total_length += np.linalg.norm(p2 - p1)
        
        stats['total_length'] = float(total_length)
        
        return stats, None
    except Exception as e:
        return None, str(e)


def analyze_surface(surf_path, image_bounds=None):
    """Extract statistics from a surface VTP or STL file."""
    try:
        import vtk
        
        # Choose reader based on file extension
        if surf_path.endswith('.stl'):
            reader = vtk.vtkSTLReader()
        else:  # .vtp file
            reader = vtk.vtkXMLPolyDataReader()
        
        reader.SetFileName(surf_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        num_points = polydata.GetNumberOfPoints()
        
        stats = {
            'num_points': num_points,
            'num_cells': polydata.GetNumberOfCells(),
            'num_polys': polydata.GetNumberOfPolys(),
        }
        
        # Get bounds
        bounds = polydata.GetBounds()
        stats['bounds'] = bounds
        stats['bounding_box_size'] = [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ]
        
        # Check if points are within image bounds
        if image_bounds is not None:
            points = polydata.GetPoints()
            points_array = np.array([points.GetPoint(i) for i in range(num_points)])
            containment = check_points_in_bounds(points_array, image_bounds)
            stats['containment_check'] = containment
        
        # Calculate surface area
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(polydata)
        mass_properties.Update()
        
        stats['surface_area'] = mass_properties.GetSurfaceArea()
        stats['volume'] = mass_properties.GetVolume()
        
        return stats, None
    except Exception as e:
        return None, str(e)


def detect_file_type(file_path):
    """
    Detect if a file is likely a mask/segmentation or an image.
    
    Returns:
        'mask' if file appears to be a segmentation mask
        'image' if file appears to be an image
        None if detection fails
    """
    try:
        img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(img)
        
        # Get unique values
        unique_values = np.unique(arr)
        num_unique = len(unique_values)
        
        # Get pixel type
        pixel_type = img.GetPixelIDTypeAsString()
        
        # Calculate statistics
        intensity_range = float(np.max(arr) - np.min(arr))
        intensity_std = float(np.std(arr))
        total_voxels = arr.size
        
        # Masks typically have:
        # - Few unique values (labels, usually < 20 for most segmentations)
        # - Integer pixel types
        # - Discrete values (low standard deviation relative to range)
        # - Values that are typically small integers (0, 1, 2, etc.)
        # - Most voxels concentrated in a few values (background + labels)
        
        # Images typically have:
        # - Many unique values (continuous or semi-continuous)
        # - Floating point or wider integer types
        # - Higher standard deviation
        # - More uniform distribution of values
        
        # Check if values are small integers (common for masks)
        is_integer_type = 'Int' in pixel_type or 'UInt' in pixel_type
        # Check first 20 unique values to see if they're small integers
        sample_values = unique_values[:min(20, len(unique_values))]
        values_are_small_integers = all(v >= 0 and v <= 255 and abs(v - int(v)) < 1e-6 for v in sample_values)
        
        # Check concentration: if most voxels are in top few values, likely a mask
        # For masks, typically background (0) dominates, then a few label values
        top_n = min(5, len(unique_values))
        top_values = unique_values[:top_n]
        top_values_count = sum(np.sum(arr == v) for v in top_values)
        top_values_percentage = top_values_count / total_voxels if total_voxels > 0 else 0
        
        # Decision logic
        has_few_labels = num_unique <= 20
        
        # Strong indicators of mask:
        # 1. Very few unique values (<= 10) and they're integers
        # 2. Few unique values (<= 20) and > 95% of voxels in top 5 values
        if has_few_labels:
            if num_unique <= 10 and (is_integer_type or values_are_small_integers):
                return 'mask'
            if top_values_percentage > 0.95 and (is_integer_type or values_are_small_integers):
                return 'mask'
        
        # Strong indicators of image:
        # 1. Many unique values (> 50)
        # 2. Floating point type
        # 3. Moderate unique values (> 20) with high standard deviation
        if num_unique > 50:
            return 'image'
        if 'Float' in pixel_type:
            return 'image'
        if num_unique > 20 and intensity_std > 10:
            return 'image'
        
        # Default: if uncertain, return None
        return None
        
    except Exception as e:
        return None


def check_file_type_mismatches(base_folder, restrict_case_names=None):
    """
    Check for files that are in the wrong folder (masks in images, images in truths).

    Args:
        restrict_case_names: If set, only inspect volume files whose case stem is in this set.

    Returns:
        Dictionary with lists of mismatched files
    """
    mismatches = {
        'masks_in_images': [],
        'images_in_truths': []
    }

    # Check images folder for masks
    images_folder = os.path.join(base_folder, 'images')
    if os.path.exists(images_folder):
        image_files = glob.glob(os.path.join(images_folder, '*'))
        image_extensions = list(_VOLUME_CASE_EXTENSIONS)
        image_files = [f for f in image_files if any(f.endswith(ext) for ext in image_extensions)]
        if restrict_case_names is not None:
            image_files = [
                f for f in image_files
                if (_stem_case_filename(f) in restrict_case_names)
            ]

        for img_file in tqdm(image_files, desc="Checking images folder for masks", leave=False):
            file_type = detect_file_type(img_file)
            if file_type == 'mask':
                filename = os.path.basename(img_file)
                mismatches['masks_in_images'].append(filename)
    
    # Check truths folder for images
    truths_folder = os.path.join(base_folder, 'truths')
    if os.path.exists(truths_folder):
        truth_files = glob.glob(os.path.join(truths_folder, '*'))
        truth_extensions = list(_VOLUME_CASE_EXTENSIONS)
        truth_files = [f for f in truth_files if any(f.endswith(ext) for ext in truth_extensions)]
        if restrict_case_names is not None:
            truth_files = [
                f for f in truth_files
                if (_stem_case_filename(f) in restrict_case_names)
            ]

        for truth_file in tqdm(truth_files, desc="Checking truths folder for images", leave=False):
            file_type = detect_file_type(truth_file)
            if file_type == 'image':
                filename = os.path.basename(truth_file)
                mismatches['images_in_truths'].append(filename)
    
    return mismatches


def check_name_consistency(base_folder, restrict_case_names=None):
    """Check if filenames match across all subfolders."""
    subfolders = ['images', 'surfaces', 'centerlines', 'truths']
    name_sets = {}

    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        name_sets[subfolder] = get_base_names(folder_path)

    if restrict_case_names is not None:
        name_sets = {k: (v & restrict_case_names) for k, v in name_sets.items()}

    # Find common and unique names
    all_names = set()
    for names in name_sets.values():
        all_names.update(names)
    
    consistency_report = {
        'total_unique_names': len(all_names),
        'names_per_folder': {k: len(v) for k, v in name_sets.items()},
        'common_to_all': set.intersection(*name_sets.values()) if all(name_sets.values()) else set(),
    }
    
    # Find missing files per folder
    for subfolder in subfolders:
        if name_sets[subfolder]:
            missing = all_names - name_sets[subfolder]
            if missing:
                consistency_report[f'missing_in_{subfolder}'] = list(missing)
    
    return consistency_report, name_sets


def fingerprint_dataset(
    base_folder,
    output_file=None,
    max_samples=None,
    config_path=None,
    config_split_mode='full',
):
    """
    Extract comprehensive fingerprint of a medical imaging dataset.

    Args:
        base_folder: Path to folder containing images/, surfaces/, centerlines/, truths/ subfolders
        output_file: Optional path to save JSON report
        max_samples: Optional limit on number of files to analyze (for large datasets)
        config_path: Optional YAML path (same shape as pipeline configs). DATA_DIR is taken from
            base_folder; cohort filters (e.g. MODALITY, ANATOMY, VMR spreadsheet) restrict cases.
        config_split_mode: If config_path is set: 'full' (all cohort cases on disk), 'train'
            (exclude TEST_CASES), or 'test' (TEST_CASES only). Ignored without config_path.

    Returns:
        Dictionary with comprehensive dataset statistics
    """
    print(f"Analyzing dataset at: {base_folder}")
    print("=" * 80)
    
    # Check folder structure (images required, others optional)
    required_folders = ['images', 'surfaces', 'centerlines', 'truths']
    existing_folders = [f for f in required_folders 
                       if os.path.exists(os.path.join(base_folder, f))]

    if 'images' not in existing_folders:
        raise FileNotFoundError(f"Required folder 'images' not found in {base_folder}")

    print(f"Found folders: {existing_folders}")

    loaded_cfg = None
    allowed_cases = None
    if config_path:
        loaded_cfg = load_global_config_with_data_dir(config_path, base_folder)
        allowed_cases = _resolve_cases_from_global_config(
            loaded_cfg, split_mode=config_split_mode,
        )
        raw_dn = loaded_cfg.get('DATASET_NAME')
        norm_dn = _normalize_dataset_name(raw_dn)
        if allowed_cases is None:
            print(
                "\n  WARNING: Config did not yield a cohort case list — scanning and "
                f"processing all cases in the folder. DATASET_NAME={raw_dn!r} "
                f"(normalized={norm_dn!r}); expected normalized name 'vmr' or 'other' "
                "with DATA_DIR set."
            )
        else:
            print(
                f"\n  Cohort filter from config: {len(allowed_cases)} case IDs "
                "(restricting mask/type checks and per-case statistics)."
            )

    restrict_scan = allowed_cases if allowed_cases is not None else None

    # Check for file type mismatches (masks in images, images in truths)
    print("\nChecking for file type mismatches...")
    type_mismatches = check_file_type_mismatches(
        base_folder, restrict_case_names=restrict_scan,
    )
    
    if type_mismatches['masks_in_images']:
        print(f"  WARNING: Found {len(type_mismatches['masks_in_images'])} files in 'images' folder that appear to be masks:")
        for filename in type_mismatches['masks_in_images'][:10]:  # Show first 10
            print(f"    - {filename}")
        if len(type_mismatches['masks_in_images']) > 10:
            print(f"    ... and {len(type_mismatches['masks_in_images']) - 10} more")
    else:
        print("  ✓ No masks detected in 'images' folder")
    
    if type_mismatches['images_in_truths']:
        print(f"  WARNING: Found {len(type_mismatches['images_in_truths'])} files in 'truths' folder that appear to be images:")
        for filename in type_mismatches['images_in_truths'][:10]:  # Show first 10
            print(f"    - {filename}")
        if len(type_mismatches['images_in_truths']) > 10:
            print(f"    ... and {len(type_mismatches['images_in_truths']) - 10} more")
    else:
        print("  ✓ No images detected in 'truths' folder")
    
    # Check name consistency
    print("\nChecking filename consistency...")
    consistency_report, name_sets = check_name_consistency(
        base_folder, restrict_case_names=restrict_scan,
    )
    
    # Add type mismatches to consistency report
    consistency_report['file_type_mismatches'] = type_mismatches

    print(f"  Total unique names: {consistency_report['total_unique_names']}")
    for folder, count in consistency_report['names_per_folder'].items():
        print(f"  {folder}: {count} files")

    # Only process cases in both images and truths
    image_names = set(name_sets.get('images', set()))
    truth_names = set(name_sets.get('truths', set()))
    folder_common_names = image_names & truth_names
    common_names = sorted(folder_common_names)

    config_meta = None
    if config_path and loaded_cfg is not None:
        allowed = allowed_cases
        config_meta = {
            'config_path': os.path.abspath(os.path.expanduser(config_path)),
            'DATASET_NAME': loaded_cfg.get('DATASET_NAME'),
            'dataset_name_normalized': _normalize_dataset_name(
                loaded_cfg.get('DATASET_NAME'),
            ),
            'split_mode': config_split_mode,
            'allowed_count_before_filter': len(allowed) if allowed is not None else None,
        }
        if allowed is not None:
            before = len(common_names)
            common_names = sorted(n for n in common_names if n in allowed)
            config_meta['cases_after_filter'] = len(common_names)
            config_meta['cases_removed_by_config'] = before - len(common_names)
            print(
                f"  Config filter ({config_path}, split={config_split_mode}): "
                f"{before} → {len(common_names)} cases"
            )
        else:
            config_meta['note'] = (
                'No case-name filter applied (unsupported DATASET_NAME or split mode).'
            )
            print(
                f"  Config loaded ({config_path}) but cohort case filter skipped: "
                f"{config_meta['note']}"
            )

    print(f"  Cases in both images and truths (folder): {len(folder_common_names)}")
    if config_meta and config_meta.get('cases_after_filter') is not None:
        print(f"  Cases after config cohort filter: {config_meta['cases_after_filter']}")
    if len(image_names) > len(folder_common_names):
        missing_truths = len(image_names - truth_names)
        print(f"    ({missing_truths} images without corresponding truths)")
    if len(truth_names) > len(folder_common_names):
        missing_images = len(truth_names - image_names)
        print(f"    ({missing_images} truths without corresponding images)")

    if max_samples and len(common_names) > max_samples:
        print(f"\nLimiting analysis to {max_samples} samples (out of {len(common_names)})")
        common_names = common_names[:max_samples]
    
    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'base_folder': base_folder,
        'config_filter': config_meta,
        'consistency': consistency_report,
        'images': {},
        'truths': {},
        'centerlines': {},
        'surfaces': {},
        'errors': defaultdict(list),
    }
    # Analyze each common case
    print(f"\nAnalyzing {len(common_names)} cases...")
    for name in tqdm(common_names, desc="Processing cases"):
        image_bounds = None
        if 'centerlines' in existing_folders or 'surfaces' in existing_folders:
            image_folder = os.path.join(base_folder, 'images')
            image_files = glob.glob(os.path.join(image_folder, name + '*'))
            if image_files:
                image_bounds, bounds_error = get_image_physical_bounds(image_files[0])
                if bounds_error:
                    results['errors']['image_bounds'].append({'name': name, 'error': bounds_error})
        
        # Find actual file paths with extensions
        for subfolder in existing_folders:
            folder_path = os.path.join(base_folder, subfolder)
            files = glob.glob(os.path.join(folder_path, name + '*'))
            
            if not files:
                continue
                
            file_path = files[0]
            
            if subfolder == 'images':
                stats, error = analyze_image(file_path)
                if error:
                    results['errors']['images'].append({'name': name, 'error': error})
                else:
                    results['images'][name] = stats
                    
            elif subfolder == 'truths':
                stats, error = analyze_segmentation(file_path)
                if error:
                    results['errors']['truths'].append({'name': name, 'error': error})
                else:
                    results['truths'][name] = stats
                    
            elif subfolder == 'centerlines':
                stats, error = analyze_centerline(file_path, image_bounds=image_bounds)
                if error:
                    results['errors']['centerlines'].append({'name': name, 'error': error})
                else:
                    results['centerlines'][name] = stats
                    
            elif subfolder == 'surfaces':
                stats, error = analyze_surface(file_path, image_bounds=image_bounds)
                if error:
                    results['errors']['surfaces'].append({'name': name, 'error': error})
                else:
                    results['surfaces'][name] = stats
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    results['summary'] = generate_summary(results)
    
    # Print summary
    print_summary(results['summary'])
    
    # Save to file if requested
    if output_file:
        # Convert numpy types to native Python for JSON serialization
        results_serializable = convert_to_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def generate_summary(results):
    """Generate summary statistics from detailed results."""
    summary = {}
    
    # Include file type mismatches in summary if present
    if 'file_type_mismatches' in results.get('consistency', {}):
        summary['file_type_mismatches'] = results['consistency']['file_type_mismatches']
    
    # Image statistics
    if results['images']:
        image_data = results['images']
        
        # Collect all spacings, sizes, etc.
        spacings = [v['spacing'] for v in image_data.values()]
        sizes = [v['size'] for v in image_data.values()]
        intensities_min = [v['intensity_min'] for v in image_data.values()]
        intensities_max = [v['intensity_max'] for v in image_data.values()]
        pixel_types = [v['pixel_type'] for v in image_data.values()]
        
        summary['images'] = {
            'num_files': len(image_data),
            'spacing_unique': len(set(map(tuple, spacings))),
            'spacing_range': {
                'x': (min(s[0] for s in spacings), max(s[0] for s in spacings)),
                'y': (min(s[1] for s in spacings), max(s[1] for s in spacings)),
                'z': (min(s[2] for s in spacings), max(s[2] for s in spacings)),
            },
            'size_unique': len(set(map(tuple, sizes))),
            'size_range': {
                'x': (min(s[0] for s in sizes), max(s[0] for s in sizes)),
                'y': (min(s[1] for s in sizes), max(s[1] for s in sizes)),
                'z': (min(s[2] for s in sizes), max(s[2] for s in sizes)),
            },
            'intensity_range': (min(intensities_min), max(intensities_max)),
            'pixel_types': list(set(pixel_types)),
        }
    
    # Truth segmentation statistics
    if results['truths']:
        truth_data = results['truths']
        
        all_labels = set()
        for v in truth_data.values():
            all_labels.update(v['unique_labels'])
        
        label_counts = defaultdict(int)
        for v in truth_data.values():
            for label in v['unique_labels']:
                label_counts[label] += 1
        
        summary['truths'] = {
            'num_files': len(truth_data),
            'all_unique_labels': sorted(list(all_labels)),
            'label_frequency': dict(label_counts),
            'binary_segmentations': sum(1 for v in truth_data.values() if v['num_labels'] == 2),
            'multi_class_segmentations': sum(1 for v in truth_data.values() if v['num_labels'] > 2),
        }
    
    # Centerline statistics
    if results['centerlines']:
        cent_data = results['centerlines']
        
        num_branches = [v.get('num_bifurcations', 0) for v in cent_data.values() 
                       if 'num_bifurcations' in v]
        lengths = [v['total_length'] for v in cent_data.values() if 'total_length' in v]
        
        summary['centerlines'] = {
            'num_files': len(cent_data),
            'avg_points': np.mean([v['num_points'] for v in cent_data.values()]),
            'avg_lines': np.mean([v['num_lines'] for v in cent_data.values()]),
        }
        
        if num_branches:
            summary['centerlines']['branches_range'] = (min(num_branches), max(num_branches))
            summary['centerlines']['avg_branches'] = np.mean(num_branches)
        
        if lengths:
            summary['centerlines']['length_range'] = (min(lengths), max(lengths))
            summary['centerlines']['avg_length'] = np.mean(lengths)
        
        # Containment check summary
        containment_checks = [v.get('containment_check') for v in cent_data.values() 
                             if 'containment_check' in v]
        if containment_checks:
            files_with_oob = sum(1 for c in containment_checks if not c['all_points_contained'])
            total_oob_points = sum(c['out_of_bounds_count'] for c in containment_checks)
            summary['centerlines']['containment'] = {
                'files_checked': len(containment_checks),
                'files_with_out_of_bounds_points': files_with_oob,
                'total_out_of_bounds_points': total_oob_points,
            }
    
    # Surface statistics
    if results['surfaces']:
        surf_data = results['surfaces']
        
        summary['surfaces'] = {
            'num_files': len(surf_data),
            'avg_points': np.mean([v['num_points'] for v in surf_data.values()]),
            'avg_cells': np.mean([v['num_cells'] for v in surf_data.values()]),
            'avg_surface_area': np.mean([v['surface_area'] for v in surf_data.values()]),
            'avg_volume': np.mean([v['volume'] for v in surf_data.values()]),
        }
        
        # Containment check summary
        containment_checks = [v.get('containment_check') for v in surf_data.values() 
                             if 'containment_check' in v]
        if containment_checks:
            files_with_oob = sum(1 for c in containment_checks if not c['all_points_contained'])
            total_oob_points = sum(c['out_of_bounds_count'] for c in containment_checks)
            summary['surfaces']['containment'] = {
                'files_checked': len(containment_checks),
                'files_with_out_of_bounds_points': files_with_oob,
                'total_out_of_bounds_points': total_oob_points,
            }
    
    return summary


def print_summary(summary):
    """Print formatted summary statistics."""
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    # Print file type mismatch warnings if present
    if 'file_type_mismatches' in summary:
        mismatches = summary['file_type_mismatches']
        if mismatches.get('masks_in_images') or mismatches.get('images_in_truths'):
            print("\n⚠️  FILE TYPE MISMATCHES DETECTED:")
            if mismatches.get('masks_in_images'):
                print(f"  Masks in 'images' folder: {len(mismatches['masks_in_images'])} files")
                print(f"    Files: {mismatches['masks_in_images'][:5]}")
                if len(mismatches['masks_in_images']) > 5:
                    print(f"    ... and {len(mismatches['masks_in_images']) - 5} more")
            if mismatches.get('images_in_truths'):
                print(f"  Images in 'truths' folder: {len(mismatches['images_in_truths'])} files")
                print(f"    Files: {mismatches['images_in_truths'][:5]}")
                if len(mismatches['images_in_truths']) > 5:
                    print(f"    ... and {len(mismatches['images_in_truths']) - 5} more")
    
    if 'images' in summary:
        print("\nIMAGES:")
        print(f"  Files: {summary['images']['num_files']}")
        print(f"  Unique spacings: {summary['images']['spacing_unique']}")
        print(f"  Spacing range:")
        for axis, (min_val, max_val) in summary['images']['spacing_range'].items():
            print(f"    {axis}: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  Unique sizes: {summary['images']['size_unique']}")
        print(f"  Size range:")
        for axis, (min_val, max_val) in summary['images']['size_range'].items():
            print(f"    {axis}: [{min_val}, {max_val}]")
        print(f"  Intensity range: [{summary['images']['intensity_range'][0]:.2f}, {summary['images']['intensity_range'][1]:.2f}]")
        print(f"  Pixel types: {', '.join(summary['images']['pixel_types'])}")
    
    if 'truths' in summary:
        print("\nSEGMENTATIONS (TRUTHS):")
        print(f"  Files: {summary['truths']['num_files']}")
        print(f"  Unique labels across dataset: {summary['truths']['all_unique_labels']}")
        print(f"  Label frequency: {summary['truths']['label_frequency']}")
        print(f"  Binary segmentations: {summary['truths']['binary_segmentations']}")
        print(f"  Multi-class segmentations: {summary['truths']['multi_class_segmentations']}")
    
    if 'centerlines' in summary:
        print("\nCENTERLINES:")
        print(f"  Files: {summary['centerlines']['num_files']}")
        print(f"  Avg points: {summary['centerlines']['avg_points']:.1f}")
        print(f"  Avg lines: {summary['centerlines']['avg_lines']:.1f}")
        if 'branches_range' in summary['centerlines']:
            print(f"  Branches range: {summary['centerlines']['branches_range']}")
            print(f"  Avg branches: {summary['centerlines']['avg_branches']:.1f}")
        if 'length_range' in summary['centerlines']:
            print(f"  Length range: [{summary['centerlines']['length_range'][0]:.2f}, {summary['centerlines']['length_range'][1]:.2f}]")
            print(f"  Avg length: {summary['centerlines']['avg_length']:.2f}")
        if 'length_range' in summary['centerlines']:
            print(f"  Length range: [{summary['centerlines']['length_range'][0]:.2f}, {summary['centerlines']['length_range'][1]:.2f}]")
            print(f"  Avg length: {summary['centerlines']['avg_length']:.2f}")
        if 'containment' in summary['centerlines']:
            cont = summary['centerlines']['containment']
            print(f"  Containment check: {cont['files_checked']} files checked")
            if cont['files_with_out_of_bounds_points'] > 0:
                print(f"    WARNING: {cont['files_with_out_of_bounds_points']} files have points outside image bounds!")
                print(f"    Total out-of-bounds points: {cont['total_out_of_bounds_points']}")
            else:
                print(f"    ✓ All centerline points contained within image bounds")
    
    if 'surfaces' in summary:
        print("\nSURFACES:")
        print(f"  Files: {summary['surfaces']['num_files']}")
        print(f"  Avg points: {summary['surfaces']['avg_points']:.1f}")
        print(f"  Avg cells: {summary['surfaces']['avg_cells']:.1f}")
        print(f"  Avg surface area: {summary['surfaces']['avg_surface_area']:.2f}")
        print(f"  Avg volume: {summary['surfaces']['avg_volume']:.2f}")
        if 'containment' in summary['surfaces']:
            cont = summary['surfaces']['containment']
            print(f"  Containment check: {cont['files_checked']} files checked")
            if cont['files_with_out_of_bounds_points'] > 0:
                print(f"    WARNING: {cont['files_with_out_of_bounds_points']} files have points outside image bounds!")
                print(f"    Total out-of-bounds points: {cont['total_out_of_bounds_points']}")
            else:
                print(f"    ✓ All surface points contained within image bounds")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract dataset fingerprint')
    parser.add_argument('folder', type=str, help='Path to dataset folder')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path (default: saves in dataset folder)')
    parser.add_argument('--max-samples', '-n', type=int, help='Maximum number of samples to analyze')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML config (e.g. config/vmr_splits/...). DATA_DIR in the file is ignored; '
             'use the positional folder as the dataset root. Restricts cases to the cohort '
             'defined by the config (MODALITY, ANATOMY, VMR_IMAGE_MODALITIES, spreadsheet, …).',
    )
    parser.add_argument(
        '--config-split',
        type=str,
        choices=('full', 'train', 'test'),
        default='full',
        help="With --config: 'full' = all cohort cases under the folder (default); "
             "'train' = exclude TEST_CASES; 'test' = only TEST_CASES (same as pipeline TESTING flag).",
    )

    args = parser.parse_args()

    output_file = args.output
    if not output_file:
        # Save in the dataset folder by default
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.folder, f'dataset_fingerprint_{timestamp}.json')

    fingerprint_dataset(
        args.folder,
        output_file=output_file,
        max_samples=args.max_samples,
        config_path=args.config,
        config_split_mode=args.config_split,
    )
