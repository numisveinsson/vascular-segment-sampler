import os
import sys
from pathlib import Path

import SimpleITK as sitk
import math
from datetime import datetime

# Add project root to path so "from modules import ..." works
_project_root = Path(__file__).resolve().parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))


def change_img_scale(img_path, scale, scale_origin=None, direction_matrix=None, change_lps_to_ras=False, verbose=False, if_spacing_file=False, spacing_value=None):
    """
    Change the scale of the image in the path

    Do this by scaling the image spacing
    but keeping everything else the same

    :param img_path: path to the image
    :param scale: new scale
    :param change_lps_to_ras: whether to change image from LPS to RAS coordinate system
    :return: sitk image
    """
    img = sitk.ReadImage(img_path)

    if if_spacing_file:
        img.SetSpacing(spacing_value)

    if verbose:
        try:
            print(f"[VERBOSE] Reading image: {img_path}")
            print(f"[VERBOSE] Original spacing: {img.GetSpacing()}")
            print(f"[VERBOSE] Original origin: {img.GetOrigin()}")
            print(f"[VERBOSE] Original direction: {img.GetDirection()}")
        except Exception:
            # Some SITK images may not expose these; ignore in that case
            pass

    if scale != 1:
        if verbose:
            print(f"[VERBOSE] Changing image scale of {img_path} by factor {scale}")
        img.SetSpacing((img.GetSpacing()[0]*scale,
                    img.GetSpacing()[1]*scale, img.GetSpacing()[2]*scale))

    if scale_origin:
        if verbose:
            print(f"[VERBOSE] Changing origin by factor {scale_origin}")
        img.SetOrigin((img.GetOrigin()[0]*scale_origin,
                       img.GetOrigin()[1]*scale_origin, img.GetOrigin()[2]*scale_origin))

    # If a direction matrix (3x3) is provided, set it on the image.
    # Expect a flat list/tuple of length 9 (row-major), e.g. [1,0,0,0,1,0,0,0,1]
    if direction_matrix is not None:
        # Basic validation
        if not (hasattr(direction_matrix, '__len__') and len(direction_matrix) == 9):
            raise ValueError("direction_matrix must be an iterable of 9 numbers (3x3 matrix flattened)")
        if verbose:
            print(f"[VERBOSE] Setting direction matrix to: {direction_matrix}")
        img.SetDirection(tuple(direction_matrix))

    if change_lps_to_ras:
        if verbose:
            print(f"[VERBOSE] Changing image from LPS to RAS coordinate system")
        # Flip the first two axes (X and Y)
        direction = list(img.GetDirection())
        direction = [-direction[0], -direction[1], direction[2],
                     -direction[3], -direction[4], direction[5],
                     -direction[6], -direction[7], direction[8]]
        img.SetDirection(tuple(direction))

        origin = list(img.GetOrigin())
        origin = [-origin[0], -origin[1], origin[2]]
        img.SetOrigin(tuple(origin))

    if verbose:
        try:
            print(f"[VERBOSE] Final spacing: {img.GetSpacing()}")
            print(f"[VERBOSE] Final origin: {img.GetOrigin()}")
            print(f"[VERBOSE] Final direction: {img.GetDirection()}")
        except Exception:
            pass

    return img


def flip_img(img, flip_ax):
    """
    Flip the image in the axis

    :param img: sitk image
    :param flip_ax: list of axis to flip the image
        [True, False, False] flips the image in the x axis
    :return: sitk image
    """
    return sitk.Flip(img, flip_ax)


if __name__ == '__main__':
    import argparse
    import ast
    
    parser = argparse.ArgumentParser(
        description='Change the scale and coordinate system of images in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scale images by factor 0.1:
  python change_img_scale_coords.py --input_dir /path/to/images --scale 0.1
  
  # Scale with custom direction matrix:
  python change_img_scale_coords.py --input_dir /path/to/images --scale 0.1 --direction_matrix 1 0 0 0 1 0 0 0 1
  
  # Flip and convert LPS to RAS:
  python change_img_scale_coords.py --input_dir /path/to/images --flip --flip_axis True True False --change_lps_to_ras
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input images. '
                            'Defaults to ./data/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write output images. '
                            'Defaults to inferred from input_dir')
    parser.add_argument('--input_format', '--input-format',
                       type=str,
                       default='.mha',
                       help='Input file extension (default: .mha)')
    parser.add_argument('--output_format', '--output-format',
                       type=str,
                       default='.mha',
                       help='Output file extension (default: .mha)')
    parser.add_argument('--scale',
                       type=float,
                       default=1.0,
                       help='Scale factor for image spacing (default: 1.0, no scaling)')
    parser.add_argument('--scale_origin', '--scale-origin',
                       type=float,
                       default=None,
                       help='Scale factor for image origin (default: None, no scaling)')
    parser.add_argument('--spacing_file', '--spacing-file',
                       type=str,
                       default=None,
                       help='CSV file containing spacing values')
    parser.add_argument('--direction_matrix', '--direction-matrix',
                       type=float,
                       nargs=9,
                       metavar=('M00', 'M01', 'M02', 'M10', 'M11', 'M12', 'M20', 'M21', 'M22'),
                       default=None,
                       help='3x3 direction matrix as 9 values (row-major). Default: keep original')
    parser.add_argument('--flip',
                       action='store_true',
                       default=False,
                       help='Flip the image')
    parser.add_argument('--flip_axis', '--flip-axis',
                       type=ast.literal_eval,
                       default=[False, False, False],
                       help='Axis to flip as list [x, y, z] (default: [False, False, False])')
    parser.add_argument('--permute',
                       action='store_true',
                       default=False,
                       help='Permute image axes')
    parser.add_argument('--change_lps_to_ras', '--change-lps-to-ras',
                       action='store_true',
                       default=False,
                       help='Convert from LPS to RAS coordinate system')
    parser.add_argument('--filter_names', '--filter-names',
                       type=str,
                       nargs='+',
                       default=None,
                       help='Only process files containing these strings in filename')
    parser.add_argument('--verbose',
                       action='store_true',
                       default=False,
                       help='Print detailed per-file information')
    
    args = parser.parse_args()
    
    input_format = args.input_format
    output_format = args.output_format
    scale = args.scale
    scale_origin = args.scale_origin
    if_spacing_file = args.spacing_file is not None
    spacing_file = args.spacing_file
    list_names = args.filter_names or []
    flip = args.flip
    permute = args.permute
    change_lps_to_ras = args.change_lps_to_ras
    direction_matrix = args.direction_matrix
    verbose = args.verbose
    flip_axis = args.flip_axis

    # Use command-line arguments (required or default)
    data_folder = args.input_dir or './data/'
    out_folder = args.output_dir or data_folder.rstrip('/') + '_scaled/'
    
    # Validate directories
    if not os.path.exists(data_folder):
        raise ValueError(f"Input directory not found: {data_folder}. "
                        f"Provide --input_dir argument.")
    
    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]
    imgs = sorted(imgs)

    if if_spacing_file:
        import pandas as pd
        spacing_df = pd.read_csv(spacing_file)
        # only keep 'spacing', they are sorted
        spacing_values = spacing_df['spacing'].values
        # read as tuples
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    if list_names:
        imgs = [f for f in imgs if any(name in f for name in list_names)]

    # Create output directory
    try:
        os.makedirs(out_folder, exist_ok=True)
        imgs_old = []
    except Exception as e:
        logger.error(f"Failed to create output directory {out_folder}: {e}")
        raise
    else:
        imgs_old = os.listdir(out_folder) if os.path.exists(out_folder) else []

    # Prepare changelog file in output folder
    log_path = os.path.join(out_folder, 'change_log.txt')
    def write_log(msg: str):
        with open(log_path, 'a') as lf:
            lf.write(msg + '\n')

    # Write header with parameters and timestamp
    header = (
        f"===== Change Log - {datetime.now().isoformat()} =====\n"
        f"input_format: {input_format}\n"
        f"output_format: {output_format}\n"
        f"scale: {scale}\n"
        f"scale_origin: {scale_origin}\n"
        f"direction_matrix: {direction_matrix}\n"
        f"flip: {flip}\n"
        f"permute: {permute}\n"
        f"verbose: {verbose}\n"
        "--------------------------------------------"
    )
    write_log(header)

    for ind, img in enumerate(imgs):
        img_path = os.path.join(data_folder, img)
        out_path = os.path.join(out_folder, img.replace(input_format,
                                                        output_format))
        if if_spacing_file:
            spacing_value = spacing_values[imgs.index(img)]
        else:
            spacing_value = None

        img = change_img_scale(img_path, scale, scale_origin, direction_matrix, change_lps_to_ras=change_lps_to_ras, verbose=verbose, if_spacing_file=if_spacing_file, spacing_value=spacing_value)

        if flip:
            if verbose:
                print(f'[VERBOSE] Flipping image {img_path} in axis {flip_axis}')
            img = flip_img(img, flip_axis)

        if permute:
            if verbose:
                print(f'[VERBOSE] Permuting axes for image {img_path}')
            img = sitk.PermuteAxes(img, [0, 1, 2])
                
        sitk.WriteImage(img, out_path)
        msg = (
            f"{datetime.now().isoformat()} | Saved {img} -> {out_path} | "
            f"scale={scale if scale!=1 else 'none'} | scale_origin={scale_origin} | "
            f"direction_set={'yes' if direction_matrix is not None else 'no'} | "
            f"flipped={'yes' if flip else 'no'} | permuted={'yes' if permute else 'no'}"
        )
        write_log(msg)
        logger.info(f'Image {ind+1}/{len(imgs)} saved to {out_path}')

    # Write footer summary
    write_log(f"Finished processing {len(imgs)} images at {datetime.now().isoformat()}")
