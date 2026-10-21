"""
Scale a dataset between millimeters and centimeters.

Expects a root directory with subfolders: images, surfaces, centerlines, truths.
- images, truths: spacing and origin are scaled (SimpleITK). Voxel values stay the same.
- surfaces, centerlines: point coordinates and radius arrays are scaled (VTK).

0.1 converts mm -> cm. 10 converts cm -> mm.

Examples:
  python preprocessing/scale_dataset_coords.py -i /path/to/dataset --to cm
  python preprocessing/scale_dataset_coords.py -i /path/to/dataset --to mm
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

# Supported extensions. Longer suffixes are matched first so .nii.gz
# is not treated as .nii.
IMAGE_EXTENSIONS = ('.nii.gz', '.mha', '.mhd', '.nii', '.nrrd', '.vti')
VTK_EXTENSIONS = ('.vtp', '.stl')

# Target unit -> scale factor. Assumes the other unit is the source.
UNIT_SCALE = {
    'cm': 0.1,   # mm -> cm
    'mm': 10.0,  # cm -> mm
}


def _matches_extension(filename, extensions):
    """Return the matched extension, or None.

    ``.nii`` does not match ``.nii.gz``.
    """
    lower = filename.lower()
    for ext in extensions:
        ext_l = ext.lower()
        if not lower.endswith(ext_l):
            continue
        if ext_l == '.nii' and lower.endswith('.nii.gz'):
            continue
        return ext
    return None


def list_files(folder, extensions):
    """Sorted filenames in ``folder`` whose suffix is in ``extensions``."""
    if isinstance(extensions, str):
        extensions = (extensions,)
    found = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if _matches_extension(name, extensions):
            found.append(name)
    return found


def _output_name(filename, input_format, output_format):
    if not output_format or output_format == input_format:
        return filename
    matched = _matches_extension(filename, (input_format,))
    if matched is None:
        return filename
    return filename[: -len(matched)] + output_format


def process_image_folder(input_folder, output_folder, scale_factor, input_format=None,
                         output_format=None, scale_origin=None, verbose=False):
    """Scale all images in a folder by modifying spacing and origin."""
    if not os.path.exists(input_folder):
        return 0

    extensions = (input_format,) if input_format else IMAGE_EXTENSIONS
    files = list_files(input_folder, extensions)
    if not files:
        return 0

    os.makedirs(output_folder, exist_ok=True)
    for file_name in files:
        matched = _matches_extension(file_name, extensions)
        input_path = os.path.join(input_folder, file_name)
        out_name = _output_name(file_name, matched, output_format)
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

    files = list_files(input_folder, VTK_EXTENSIONS)
    if not files:
        return 0

    os.makedirs(output_folder, exist_ok=True)
    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        scale_polydata(input_path, output_path, scale_factor)

    return len(files)


def scale_dataset(root_dir, output_dir, scale_factor, input_format=None,
                  output_format=None, scale_origin=None, folders=None, verbose=False):
    """
    Scale all data in a dataset folder structure.

    :param root_dir: Root directory containing images/, surfaces/, centerlines/, truths/
    :param output_dir: Output root directory (subfolders will be created)
    :param scale_factor: Scale factor (0.1 for mm->cm, 10 for cm->mm)
    :param input_format: Image extension to process, or None for every supported format
    :param output_format: Output image extension (default: same as each input file)
    :param scale_origin: Scale factor for image origin (default: same as scale_factor)
    :param folders: Folder names to process (default: images, surfaces, centerlines, truths)
    :param verbose: Print detailed per-file information
    :return: Dict with counts per folder
    """
    if os.path.abspath(root_dir) == os.path.abspath(output_dir):
        raise ValueError(
            f"Output directory must differ from the input directory: {root_dir}"
        )

    scale_origin = scale_origin if scale_origin is not None else scale_factor

    folders_to_process = folders or (IMAGE_FOLDERS + VTK_FOLDERS)
    present = [
        name for name in folders_to_process
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    if not present:
        expected = ', '.join(folders_to_process)
        raise ValueError(
            f"No dataset subfolders found in {root_dir}. Expected: {expected}"
        )

    counts = {}

    try:
        from modules.logger import get_logger
        logger = get_logger(__name__)
    except Exception:
        logger = None

    def log(msg):
        if logger:
            logger.info(msg)
        else:
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
            log(f"Scaled {n} image(s) in {folder_name}/")
        elif folder_name in VTK_FOLDERS:
            n = process_vtk_folder_wrapper(input_folder, output_folder, scale_factor)
            counts[folder_name] = n
            log(f"Scaled {n} surface/centerline file(s) in {folder_name}/")
        else:
            log(f"Unknown folder type: {folder_name}, skipping")

    return counts


def scale_factor_for_target(target_unit):
    """Return the multiply factor that converts into ``target_unit`` (cm or mm)."""
    try:
        return UNIT_SCALE[target_unit]
    except KeyError as exc:
        raise ValueError(
            f"Target unit must be 'cm' (scale 0.1) or 'mm' (scale 10), got {target_unit!r}"
        ) from exc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Scale images, surfaces, centerlines, and truths between cm and mm. '
            'Writes a new folder; the input is left unchanged.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Millimeters to centimeters (multiply by 0.1):
  python preprocessing/scale_dataset_coords.py -i /path/to/dataset --to cm

  # Centimeters to millimeters (multiply by 10):
  python preprocessing/scale_dataset_coords.py -i /path/to/dataset --to mm

  # Same conversion with an explicit factor:
  python preprocessing/scale_dataset_coords.py -i /path/to/dataset --scale_factor 0.1
        """
    )
    parser.add_argument('-i', '--input_dir', '--input-dir',
                        type=str,
                        required=True,
                        help='Root directory containing images/, surfaces/, centerlines/, truths/')
    parser.add_argument('-o', '--output_dir', '--output-dir',
                        type=str,
                        default=None,
                        help='Output root directory. Defaults to input_dir + "_cm" or "_mm"')
    unit_group = parser.add_mutually_exclusive_group(required=True)
    unit_group.add_argument('--to',
                        choices=('cm', 'mm'),
                        help='Target unit. cm multiplies by 0.1 (from mm). mm multiplies by 10 (from cm).')
    unit_group.add_argument('--scale_factor', '--scale-factor',
                        type=float,
                        help='Explicit scale factor. Use 0.1 for mm->cm or 10 for cm->mm.')
    parser.add_argument('--scale_origin', '--scale-origin',
                        type=float,
                        default=None,
                        help='Scale factor for image origin (default: same as the coordinate scale)')
    parser.add_argument('--input_format', '--input-format',
                        type=str,
                        default=None,
                        help='Only scale images/truths with this extension '
                             '(default: .mha .mhd .nii .nii.gz .nrrd .vti)')
    parser.add_argument('--output_format', '--output-format',
                        type=str,
                        default=None,
                        help='Output image extension (default: same as each input file)')
    parser.add_argument('--folders',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Only process these folders (default: images surfaces centerlines truths)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print spacing and origin for each image')

    args = parser.parse_args()

    root_dir = args.input_dir
    if args.to:
        scale_factor = scale_factor_for_target(args.to)
        unit_note = f"{'mm -> cm' if args.to == 'cm' else 'cm -> mm'}"
        default_suffix = f"_{args.to}"
    else:
        scale_factor = args.scale_factor
        unit_note = 'mm -> cm' if scale_factor == 0.1 else (
            'cm -> mm' if scale_factor == 10 else 'custom scale'
        )
        default_suffix = '_scaled'
    output_dir = args.output_dir or (root_dir.rstrip('/') + default_suffix)

    if not os.path.isdir(root_dir):
        raise ValueError(f"Input directory not found: {root_dir}")

    print(f"Scaling {root_dir}")
    print(f"  factor {scale_factor} ({unit_note})")
    print(f"  writing {output_dir}")

    counts = scale_dataset(
        root_dir, output_dir, scale_factor,
        input_format=args.input_format,
        output_format=args.output_format,
        scale_origin=args.scale_origin,
        folders=args.folders,
        verbose=args.verbose
    )

    total = sum(counts.values())
    print(f"Done. Scaled {total} file(s): {counts}")
