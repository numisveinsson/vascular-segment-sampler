"""
Scale all data in a dataset folder structure by a factor.

Expects a root directory with subfolders: images, surfaces, centerlines, truths.
- images, truths: scaled via image spacing (SimpleITK)
- surfaces, centerlines: scaled via point coordinates (VTK polydata)

Uses functionality from change_img_scale_coords.py and change_vtk_scale_coords.py.
"""
import os
import sys

import SimpleITK as sitk

# Ensure repo root is on path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.change_img_scale_coords import change_img_scale
from preprocessing.change_vtk_scale_coords import scale_polydata

# Subfolders expected in the dataset root
IMAGE_FOLDERS = ('images', 'truths')
VTK_FOLDERS = ('surfaces', 'centerlines')

# Supported extensions
IMAGE_EXTENSIONS = ('.mha', '.mhd', '.nii', '.nii.gz', '.nrrd', '.vti')
VTK_EXTENSIONS = ('.vtp', '.stl')


def process_image_folder(input_folder, output_folder, scale_factor, input_format='.mha',
                         output_format=None, scale_origin=None, verbose=False):
    """Scale all images in a folder by modifying spacing and optionally origin."""
    output_format = output_format or input_format
    if not os.path.exists(input_folder):
        return 0

    files = [f for f in os.listdir(input_folder) if f.endswith(input_format)]
    if not files:
        return 0

    os.makedirs(output_folder, exist_ok=True)
    for file_name in sorted(files):
        input_path = os.path.join(input_folder, file_name)
        out_name = file_name
        if input_format != output_format:
            out_name = file_name.replace(input_format, output_format)
        output_path = os.path.join(output_folder, out_name)

        img = change_img_scale(
            input_path, scale_factor,
            scale_origin=scale_origin,
            verbose=verbose
        )
        sitk.WriteImage(img, output_path)

    return len(files)


def process_vtk_folder_wrapper(input_folder, output_folder, scale_factor):
    """Scale all VTK polydata files in a folder."""
    if not os.path.exists(input_folder):
        return 0

    files = [f for f in os.listdir(input_folder) if f.endswith(VTK_EXTENSIONS)]
    if not files:
        return 0

    os.makedirs(output_folder, exist_ok=True)
    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        scale_polydata(input_path, output_path, scale_factor)

    return len(files)


def scale_dataset(root_dir, output_dir, scale_factor, input_format='.mha',
                  output_format=None, scale_origin=None, folders=None, verbose=False):
    """
    Scale all data in a dataset folder structure.

    :param root_dir: Root directory containing images/, surfaces/, centerlines/, truths/
    :param output_dir: Output root directory (subfolders will be created)
    :param scale_factor: Scale factor (e.g., 0.1 to convert mm to cm)
    :param input_format: Image file extension for images/truths (default: .mha)
    :param output_format: Output image extension (default: same as input_format)
    :param scale_origin: Optional scale factor for image origin (default: same as scale_factor)
    :param folders: Optional list of folder names to process (default: all)
    :param verbose: Print detailed per-file information
    :return: Dict with counts per folder
    """
    output_format = output_format or input_format
    scale_origin = scale_origin if scale_origin is not None else scale_factor

    folders_to_process = folders or (IMAGE_FOLDERS + VTK_FOLDERS)
    counts = {}

    try:
        from modules.logger import get_logger
        logger = get_logger(__name__)
    except Exception:
        logger = None

    def log(msg):
        if logger:
            logger.info(msg)
        if verbose:
            print(msg)

    for folder_name in folders_to_process:
        input_folder = os.path.join(root_dir, folder_name)
        output_folder = os.path.join(output_dir, folder_name)

        if not os.path.exists(input_folder):
            log(f"Skipping {folder_name}: directory not found")
            continue

        if folder_name in IMAGE_FOLDERS:
            n = process_image_folder(
                input_folder, output_folder, scale_factor,
                input_format=input_format,
                output_format=output_format,
                scale_origin=scale_origin,
                verbose=verbose
            )
            counts[folder_name] = n
            if n > 0:
                log(f"Scaled {n} image(s) in {folder_name}/")
        elif folder_name in VTK_FOLDERS:
            n = process_vtk_folder_wrapper(input_folder, output_folder, scale_factor)
            counts[folder_name] = n
            if n > 0:
                log(f"Scaled {n} VTK file(s) in {folder_name}/")
        else:
            log(f"Unknown folder type: {folder_name}, skipping")

    return counts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Scale all data in a dataset folder (images, surfaces, centerlines, truths) by a factor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scale_dataset_coords.py --input_dir /path/to/dataset --output_dir /path/to/output --scale_factor 0.1

  # Scale only specific folders:
  python scale_dataset_coords.py --input_dir ./data --scale_factor 0.1 --folders images surfaces

  # With custom image format:
  python scale_dataset_coords.py --input_dir ./data --scale_factor 0.1 --input_format .nii.gz
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                        type=str,
                        required=True,
                        help='Root directory containing images/, surfaces/, centerlines/, truths/')
    parser.add_argument('--output_dir', '--output-dir',
                        type=str,
                        default=None,
                        help='Output root directory. Defaults to input_dir + "_scaled"')
    parser.add_argument('--scale_factor', '--scale-factor',
                        type=float,
                        required=True,
                        help='Scale factor (e.g., 0.1 to convert mm to cm)')
    parser.add_argument('--scale_origin', '--scale-origin',
                        type=float,
                        default=None,
                        help='Scale factor for image origin (default: same as scale_factor)')
    parser.add_argument('--input_format', '--input-format',
                        type=str,
                        default='.mha',
                        help='Image file extension for images/truths (default: .mha)')
    parser.add_argument('--output_format', '--output-format',
                        type=str,
                        default=None,
                        help='Output image extension (default: same as input_format)')
    parser.add_argument('--folders',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Only process these folders (default: images surfaces centerlines truths)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print detailed per-file information')

    args = parser.parse_args()

    root_dir = args.input_dir
    output_dir = args.output_dir or (root_dir.rstrip('/') + '_scaled')
    scale_factor = args.scale_factor

    if not os.path.exists(root_dir):
        raise ValueError(f"Input directory not found: {root_dir}")

    # Initialize logger
    try:
        from modules.logger import get_logger
        logger = get_logger(__name__)
        logger.info(f"Scaling dataset from {root_dir} to {output_dir} with factor {scale_factor}")
    except Exception:
        pass

    counts = scale_dataset(
        root_dir, output_dir, scale_factor,
        input_format=args.input_format,
        output_format=args.output_format,
        scale_origin=args.scale_origin,
        folders=args.folders,
        verbose=args.verbose
    )

    total = sum(counts.values())
    print(f"Done. Scaled {total} file(s) across {len(counts)} folder(s): {counts}")
