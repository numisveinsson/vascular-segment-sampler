import os
import sys
from pathlib import Path

import SimpleITK as sitk

# Add project root to path so "from modules import ..." works
_project_root = Path(__file__).resolve().parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from modules import vtk_functions as vf


def _compute_bounds_sitk(img):
    """Compute axis-aligned physical bounds of a SITK image using origin, spacing, direction."""
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    # Build 3x3 direction matrix
    dir_mat = (
        (direction[0], direction[1], direction[2]),
        (direction[3], direction[4], direction[5]),
        (direction[6], direction[7], direction[8]),
    )

    # Generate corner indices
    ix = [0, max(size[0] - 1, 0)]
    iy = [0, max(size[1] - 1, 0)]
    iz = [0, max(size[2] - 1, 0)]

    corners = []
    for i in ix:
        for j in iy:
            for k in iz:
                # scaled voxel index
                sx = i * spacing[0]
                sy = j * spacing[1]
                sz = k * spacing[2]
                # apply direction matrix
                px = origin[0] + dir_mat[0][0] * sx + dir_mat[0][1] * sy + dir_mat[0][2] * sz
                py = origin[1] + dir_mat[1][0] * sx + dir_mat[1][1] * sy + dir_mat[1][2] * sz
                pz = origin[2] + dir_mat[2][0] * sx + dir_mat[2][1] * sy + dir_mat[2][2] * sz
                corners.append((px, py, pz))

    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    min_z = min(c[2] for c in corners)
    max_z = max(c[2] for c in corners)

    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


def _compute_bounds_vtk(vtk_img):
    """Compute axis-aligned physical bounds of a VTK image using origin, spacing, extent."""
    img = vtk_img
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    ex = img.GetExtent()  # (xmin, xmax, ymin, ymax, zmin, zmax)

    ix = [ex[0], ex[1]]
    iy = [ex[2], ex[3]]
    iz = [ex[4], ex[5]]

    corners = []
    for i in ix:
        for j in iy:
            for k in iz:
                px = origin[0] + i * spacing[0]
                py = origin[1] + j * spacing[1]
                pz = origin[2] + k * spacing[2]
                corners.append((px, py, pz))

    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    min_z = min(c[2] for c in corners)
    max_z = max(c[2] for c in corners)

    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


def _compare_bounds(bounds_a, bounds_b, tol=1e-4):
    """Compare two sets of bounds per axis with tolerance. Returns (ok, diffs)."""
    diffs = []
    for axis in range(3):
        dmin = abs(bounds_a[axis][0] - bounds_b[axis][0])
        dmax = abs(bounds_a[axis][1] - bounds_b[axis][1])
        diffs.append((dmin, dmax))
    ok = all(dmin <= tol and dmax <= tol for dmin, dmax in diffs)
    return ok, diffs


def change_mha_vti(file_dir, label=False):
    """
    Change the format of a file from .mha to .vti
    SITK does not support .vti format, so we need to use the vtk functions
    Args:
        file_dir: str, path to the file
    Returns:
        None
    """
    img = sitk.ReadImage(file_dir)
    if label:
        img = sitk.Cast(img, sitk.sitkUInt8)
    img = vf.exportSitk2VTK(img)[0]

    return img


def change_vti_sitk(file_dir, label=False):
    """
    Change the format of a file from .vti to .mha
    SITK does not support .vti format, so we need to use the vtk functions
    Args:
        file_dir: str, path to the file
        label: bool, whether this is a label segmentation (will cast to UInt8)
    Returns:
        None
    """
    img = vf.read_img(file_dir)
    img = vf.exportVTK2Sitk(img)
    
    if label:
        img = sitk.Cast(img, sitk.sitkUInt8)

    return img


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert image files between different formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python change_img_format.py --input_dir /path/to/images --output_dir /path/to/output --input_format .nrrd --output_format .mha
  
  # Convert with label detection:
  python change_img_format.py --input_dir /path/to/images --output_format .mha --label_if_string seg
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input image files. '
                            'Trailing slash optional. Defaults to ./data/images')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write output files. '
                            'Defaults to inferred from input_dir')
    parser.add_argument('--input_format', '--input-format',
                       type=str,
                       default='.nrrd',
                       help='Input file extension (default: .nrrd). Options: .dcm, .nii.gz, .vti, .mha, .nrrd')
    parser.add_argument('--output_format', '--output-format',
                       type=str,
                       default='.mha',
                       help='Output file extension (default: .mha). Options: .mha, .nii.gz, .vti')
    parser.add_argument('--label',
                       action='store_true',
                       default=False,
                       help='Treat all files as label segmentations (cast to UInt8)')
    parser.add_argument('--label_if_string', '--label-if-string',
                       type=str,
                       default='',
                       help='Only treat files as labels if filename contains this string (e.g., "seg", "mask", "label")')
    parser.add_argument('--rem_str', '--rem-str',
                       type=str,
                       default='',
                       help='String to remove from filenames before saving')
    parser.add_argument('--surface',
                       action='store_true',
                       default=False,
                       help='Also create surface mesh (.vtp) files for label images')
    
    args = parser.parse_args()
    
    input_format = args.input_format
    output_format = args.output_format
    label = args.label
    surface = args.surface
    label_if_string = args.label_if_string
    rem_str = args.rem_str

    # Use command-line arguments (required or default)
    data_folder = os.path.normpath(
        os.path.expanduser(args.input_dir or os.path.join('.', 'data', 'images'))
    )
    if args.output_dir:
        out_folder = os.path.normpath(os.path.expanduser(args.output_dir))
    else:
        out_folder = data_folder.replace(
            'images', 'images_' + output_format.replace('.', '')
        )
    
    # Validate directories
    if not os.path.exists(data_folder):
        raise ValueError(f"Input directory not found: {data_folder}. "
                        f"Provide --input_dir argument.")
    
    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]

    # sort the files
    imgs = sorted(imgs)

    # Create output directory
    try:
        os.makedirs(out_folder, exist_ok=True)
        imgs_old = []
    except Exception as e:
        logger.error(f"Failed to create output directory {out_folder}: {e}")
        raise
    else:
        imgs_old = os.listdir(out_folder) if os.path.exists(out_folder) else []

    if input_format == '.dcm':
        # it's a dicom folder with multiple files
        # create a 3D image from the dicom files
        logger.info(f"Reading dicom files from {data_folder}")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(data_folder)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        out_3d = os.path.join(out_folder, '3D_image' + output_format)
        logger.info(f"Saving 3D image to {out_3d}")
        sitk.WriteImage(img, out_3d)
    else:
        # it's a single file
        for fn in imgs:
            if fn.replace(input_format, output_format) in imgs_old:
                logger.info(f"File {fn} already exists in folder, skipping")
                continue
            else:
                logger.info(f"Converting file {fn} to new format {output_format}")
            
            # Determine if this file should be treated as a label
            is_label = label  # Default to global setting
            if label_if_string:
                is_label = label_if_string in fn
                logger.debug(f"  File contains '{label_if_string}': {is_label} -> treating as {'label' if is_label else 'image'}")
            
            # Compute pre-conversion bounds
            before_bounds = None
            in_path = os.path.join(data_folder, fn)
            if input_format == '.vti':
                vtk_before = vf.read_img(in_path).GetOutput()
                before_bounds = _compute_bounds_vtk(vtk_before)
            else:
                sitk_before = sitk.ReadImage(in_path)
                before_bounds = _compute_bounds_sitk(sitk_before)
            
            if input_format != '.vti' and output_format != '.vti':
                img = sitk.ReadImage(in_path)
                if is_label:
                    img = sitk.Cast(img, sitk.sitkUInt8)
            elif input_format == '.vti':
                if output_format == '.mha' or output_format == '.nii.gz':
                    img = change_vti_sitk(in_path, is_label)
                else:
                    img = vf.read_img(in_path).GetOutput()
            elif output_format == '.vti':
                if input_format in ['.mha', '.nii.gz', '.nii', '.mhd']:
                    img = change_mha_vti(in_path, is_label)
            else:
                logger.error('Invalid input/output format combination')
                break

            if rem_str:
                fn = fn.replace(rem_str, '')

            img_name = fn.replace(input_format, '')

            # Compute post-conversion bounds and compare
            if output_format != '.vti':
                after_bounds = _compute_bounds_sitk(img)
            else:
                after_bounds = _compute_bounds_vtk(img)
            ok, diffs = _compare_bounds(before_bounds, after_bounds)
            logger.debug(f"  Bounds before:  X{before_bounds[0]} Y{before_bounds[1]} Z{before_bounds[2]}")
            logger.debug(f"  Bounds after:   X{after_bounds[0]} Y{after_bounds[1]} Z{after_bounds[2]}")
            if ok:
                logger.debug("  Bounds check: OK (no accidental transform)")
            else:
                logger.warning(f"  WARNING: Bounds differ (tol=1e-4). Diffs per axis (min,max): {diffs}")

            logger.info(f"Saving {img_name}")
            out_path = os.path.join(out_folder, img_name + output_format)
            if output_format != '.vti':
                sitk.WriteImage(img, out_path)
            else:
                vf.write_img(out_path, img)

            if surface and is_label:  # Only create surfaces for label files
                img_vtk = vf.exportSitk2VTK(img)[0]
                poly = vf.vtk_marching_cube(img_vtk, 0, 1)
                vf.write_geo(os.path.join(out_folder, img_name + '.vtp'), poly)
