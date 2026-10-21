"""
Scale a dataset between millimeters and centimeters.

Expects a root directory with subfolders: images, surfaces, centerlines, truths.
- images, truths: spacing and origin are scaled (SimpleITK). Voxel values stay the same.
- surfaces, centerlines: point coordinates and radius arrays are scaled (VTK).

0.1 converts mm -> cm. 10 converts cm -> mm.

The standalone copy is ``preprocessing/scale_dataset_coords.py``.
"""
import os

import SimpleITK as sitk

from vascular_segment_sampler.preprocessing.change_img_scale_coords import change_img_scale
from vascular_segment_sampler.preprocessing.change_vtk_scale_coords import scale_polydata

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
        from vascular_segment_sampler.logger import get_logger
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
