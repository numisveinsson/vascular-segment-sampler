"""
Cut surfaces (.vtp) so they are bounded by the extent of their matching image
volume, then save the clipped surfaces to a new folder.

Each surface is paired with an image by file stem (the filename without its
extension). Images may be in any SimpleITK-readable format (.mha, .mhd, .nii,
.nii.gz, .nrrd, ...) or .vti. The image is converted to a VTK image and the
surface is clipped to its axis-aligned bounds using
``modules.vtk_functions.bound_polydata_by_image``.
"""

import os
import sys

# Add project root to path for module imports (when run as script)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import SimpleITK as sitk

from modules import vtk_functions as vf
from modules.logger import get_logger

# Image extensions we know how to read. Multi-part extensions (e.g. .nii.gz)
# must come before their single-part prefixes so stems are computed correctly.
IMAGE_EXTENSIONS = ('.nii.gz', '.mha', '.mhd', '.vti', '.nii', '.nrrd', '.vtk')


def _strip_extension(filename, extensions):
    """Return (stem, matched_ext) for the first matching extension, else (None, None)."""
    for ext in extensions:
        if filename.endswith(ext):
            return filename[: -len(ext)], ext
    return None, None


def _read_image_as_vtk(image_path):
    """Read an image from disk (SITK or .vti) and return a vtkImageData."""
    if image_path.endswith('.vti'):
        return vf.read_img(image_path).GetOutput()
    img_sitk = sitk.ReadImage(image_path)
    return vf.exportSitk2VTK(img_sitk)[0]


def cut_surfaces_by_image(surfaces_folder, images_folder, out_folder,
                          threshold=0.0, image_extensions=IMAGE_EXTENSIONS,
                          surface_format='.vtp', skip_existing=True):
    """
    Clip every surface in ``surfaces_folder`` to the bounds of its matching
    image volume and write the result to ``out_folder``.

    Args:
        surfaces_folder: Folder containing input surfaces (.vtp).
        images_folder: Folder containing image volumes used to define bounds.
        out_folder: Folder where clipped surfaces are written.
        threshold: Padding applied to the image bounds. Positive values shrink
            the clipping box (see ``bound_polydata_by_image``).
        image_extensions: Iterable of image extensions to consider when pairing.
        surface_format: Surface file extension to process (default: .vtp).
        skip_existing: Skip a surface if its output already exists.

    Returns:
        List of processed surface filenames.
    """
    logger = get_logger(__name__)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Build a lookup from image stem -> image path.
    image_map = {}
    for f in sorted(os.listdir(images_folder)):
        stem, ext = _strip_extension(f, image_extensions)
        if stem is None:
            continue
        # Keep the first match for a stem to avoid ambiguous duplicates.
        image_map.setdefault(stem, os.path.join(images_folder, f))

    surfaces = sorted(f for f in os.listdir(surfaces_folder)
                      if f.endswith(surface_format))

    logger.info(f'Found {len(surfaces)} surfaces and {len(image_map)} images')

    processed = []
    for surf in surfaces:
        stem = surf[: -len(surface_format)]
        out_path = os.path.join(out_folder, surf)

        if skip_existing and os.path.exists(out_path):
            logger.info(f'Surface {surf} already processed, skipping...')
            continue

        if stem not in image_map:
            logger.warning(f'No matching image found for surface {surf}, skipping...')
            continue

        surf_path = os.path.join(surfaces_folder, surf)
        image_path = image_map[stem]

        logger.debug(f'Cutting surface {surf} using image {os.path.basename(image_path)}')

        poly = vf.read_geo(surf_path).GetOutput()
        img_vtk = _read_image_as_vtk(image_path)
        clipped = vf.bound_polydata_by_image(img_vtk, poly, threshold)

        vf.write_geo(out_path, clipped)
        logger.info(f'Surface {surf} cut and saved to {out_path}')
        processed.append(surf)

    logger.info(f'Done. Processed {len(processed)} surfaces.')
    return processed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Cut surfaces (.vtp) to the bounds of matching image volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cut surfaces using images, write to a new folder:
  python cut_surfaces_by_image.py --surfaces_dir /path/to/surfaces \\
      --images_dir /path/to/images --output_dir /path/to/cut_surfaces

  # Add padding (shrink the clip box by 1 mm on each side):
  python cut_surfaces_by_image.py -s ./data/surfaces -m ./data/images \\
      -o ./data/surfaces_cut --threshold 1.0
        """
    )
    parser.add_argument('-s', '--surfaces_dir', '--surfaces-dir',
                        type=str, required=True,
                        help='Directory containing input surfaces (.vtp)')
    parser.add_argument('-m', '--images_dir', '--images-dir',
                        type=str, required=True,
                        help='Directory containing image volumes (.mha, .vti, .nii.gz, ...)')
    parser.add_argument('-o', '--output_dir', '--output-dir',
                        type=str, default=None,
                        help='Directory for cut surfaces. '
                             'Default: surfaces_dir/cut/')
    parser.add_argument('--threshold',
                        type=float, default=0.0,
                        help='Padding applied to image bounds. Positive values '
                             'shrink the clip box (default: 0.0)')
    parser.add_argument('--surface_format', '--surface-format',
                        type=str, default='.vtp',
                        help='Surface file extension to process (default: .vtp)')
    parser.add_argument('--no_skip_existing', '--no-skip-existing',
                        dest='skip_existing', action='store_false',
                        help='Re-process surfaces even if output already exists')

    args = parser.parse_args()

    if not os.path.isdir(args.surfaces_dir):
        parser.error(f'Surfaces directory not found: {args.surfaces_dir}')
    if not os.path.isdir(args.images_dir):
        parser.error(f'Images directory not found: {args.images_dir}')

    out_folder = args.output_dir or os.path.join(args.surfaces_dir, 'cut')

    cut_surfaces_by_image(
        surfaces_folder=args.surfaces_dir,
        images_folder=args.images_dir,
        out_folder=out_folder,
        threshold=args.threshold,
        surface_format=args.surface_format,
        skip_existing=args.skip_existing,
    )
