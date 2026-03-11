"""
Extract a subset (ROI) of a 3D image given voxel index bounds.

ROI is specified as inclusive [min, max] indices per axis (x, y, z).
Example: x 50–450, y 50–450, z 40–250 extracts a 401×401×211 voxel region.
"""
import os
import sys

import SimpleITK as sitk

# Ensure repo root is on path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def extract_roi(input_path, output_path, x_min, x_max, y_min, y_max, z_min, z_max,
                clamp=True, verbose=False):
    """
    Extract a rectangular ROI from a 3D image.

    :param input_path: Path to input image (.mha, .mhd, .nii, .nii.gz, .nrrd)
    :param output_path: Path for output image
    :param x_min: Start index in x (inclusive)
    :param x_max: End index in x (inclusive)
    :param y_min: Start index in y (inclusive)
    :param y_max: End index in y (inclusive)
    :param z_min: Start index in z (inclusive)
    :param z_max: End index in z (inclusive)
    :param clamp: If True, clamp ROI to image bounds (default True)
    :param verbose: Print size and bounds info
    :return: The extracted SimpleITK image
    """
    img = sitk.ReadImage(input_path)
    size = img.GetSize()

    index = [x_min, y_min, z_min]
    roi_size = [
        x_max - x_min + 1,
        y_max - y_min + 1,
        z_max - z_min + 1,
    ]

    if clamp:
        for i in range(3):
            index[i] = max(0, min(index[i], size[i] - 1))
            roi_size[i] = min(
                roi_size[i],
                size[i] - index[i],
            )
            roi_size[i] = max(1, roi_size[i])

    # Validate
    for i in range(3):
        if index[i] < 0 or index[i] >= size[i]:
            raise ValueError(
                f"ROI index {index[i]} out of bounds for axis {i} (size {size[i]})"
            )
        if index[i] + roi_size[i] > size[i]:
            raise ValueError(
                f"ROI extends beyond image: index {index[i]} + size {roi_size[i]} > {size[i]}"
            )

    if verbose:
        print(f"Input size: {size}")
        print(f"ROI index: {index}")
        print(f"ROI size:  {roi_size}")

    roi = sitk.RegionOfInterest(img, size=roi_size, index=index)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sitk.WriteImage(roi, output_path)
    return roi


def parse_roi_arg(s):
    """Parse 'min,max' string into (min, max) int tuple."""
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError(f"Expected 'min,max', got '{s}'")
    return int(parts[0].strip()), int(parts[1].strip())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract a rectangular ROI from a 3D image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_roi.py -i image.mha -o roi.mha --x 50,450 --y 50,450 --z 40,250

  # Same with explicit output:
  python extract_roi.py -i /path/to/image.nii.gz -o /path/to/roi.nii.gz \\
      --x 50,450 --y 50,450 --z 40,250
        """
    )
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='Input image path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='Output image path (default: input_roi.<ext>)')
    parser.add_argument('--x',
                        type=str,
                        required=True,
                        help='x range as min,max (e.g. 50,450)')
    parser.add_argument('--y',
                        type=str,
                        required=True,
                        help='y range as min,max (e.g. 50,450)')
    parser.add_argument('--z',
                        type=str,
                        required=True,
                        help='z range as min,max (e.g. 40,250)')
    parser.add_argument('--no-clamp',
                        action='store_true',
                        help='Do not clamp ROI to image bounds (fail if out of bounds)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print ROI size and bounds')

    args = parser.parse_args()

    x_min, x_max = parse_roi_arg(args.x)
    y_min, y_max = parse_roi_arg(args.y)
    z_min, z_max = parse_roi_arg(args.z)

    if x_min > x_max or y_min > y_max or z_min > z_max:
        raise ValueError("min must be <= max for each axis")

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = args.output
    if output_path is None:
        if input_path.endswith('.nii.gz'):
            base = input_path[:-7]  # strip .nii.gz
            ext = '.nii.gz'
        else:
            base, ext = os.path.splitext(input_path)
        output_path = base + '_roi' + ext

    extract_roi(
        input_path, output_path,
        x_min, x_max, y_min, y_max, z_min, z_max,
        clamp=not args.no_clamp,
        verbose=args.verbose,
    )
    print(f"Saved ROI to {output_path}")
