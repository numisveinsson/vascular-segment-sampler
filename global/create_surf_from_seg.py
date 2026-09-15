import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys
import os

# Add project root to path so "from modules import ..." works
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import vtk_functions as vf
from modules import sitk_functions as sf


def rotate_mesh(mesh, vtkLabel, center=None):
    """
    Rotate the mesh by 90 degrees
    around the origin of the image y-axis

    The steps are as follows:
    1. Specify the axis of rotation
    2. Create a transform
    3. Apply the transform to the mesh

    Args:
        mesh: vtk PolyData
        vtkLabel: vtk ImageData
    Returns:
        mesh: rotated vtk PolyData
    """

    # Get the center of the image
    if center is None:
        center = vtkLabel.GetCenter()

    # Specify the axis of rotation
    axis = [0, 1, 0]
    # Create a transform
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.RotateWXYZ(90, axis[0], axis[1], axis[2])
    transform.Translate(center[0], center[1], center[2])

    # Apply the transform to the mesh
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    return transformFilter.GetOutput()


def parse_label_names(spec):
    """Parse '1=airways,2=label2' into {1: 'airways', 2: 'label2'}."""
    if not spec:
        return {}
    names = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(
                f"Invalid --label-names entry '{part}'. Expected id=name, e.g. 1=airways"
            )
        raw_id, name = part.split('=', 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty label name in --label-names entry '{part}'")
        names[int(raw_id.strip())] = name
    return names


def parse_mask_labels(spec):
    """Parse comma-separated label ids. None/empty -> None (all non-zero)."""
    if spec is None:
        return None
    labels = [int(x.strip()) for x in spec.split(',') if x.strip()]
    if not labels:
        raise ValueError("--masks must specify at least one label value")
    return labels


def label_folder_name(label_id, name_map=None):
    """Directory/file suffix for a label id."""
    if name_map and label_id in name_map:
        return name_map[label_id]
    return f'label{label_id}'


def label_output_path(output_dir, stem, label_id, name_map=None,
                      dir_prefix=None, output_ext='.vtp'):
    """Path for one per-label surface.

    With dir_prefix: {output_dir}/{prefix}_{name}/{stem}{ext}
    Without:         {output_dir}/{stem}_{name}{ext}
    """
    name = label_folder_name(label_id, name_map)
    filename = stem + output_ext
    if dir_prefix:
        return os.path.join(output_dir, f'{dir_prefix}_{name}', filename)
    return os.path.join(output_dir, f'{stem}_{name}{output_ext}')


def unique_nonzero_labels(seg, mask_labels=None, bg_id=0):
    """Sorted unique label ids in a SimpleITK image, excluding background."""
    arr = sitk.GetArrayViewFromImage(seg)
    ids = np.unique(arr)
    ids = ids[ids != bg_id]
    if mask_labels is not None:
        allowed = np.array(mask_labels, dtype=ids.dtype)
        ids = ids[np.isin(ids, allowed)]
    return [int(x) for x in ids]


def apply_smoothing(poly, smooth_method, iterations=None, smoothing_factor=None):
    """Smooth a surface with the method used by this script's --smooth_method.

    vtk_taubin uses vtkWindowedSincPolyDataFilter. Pass band is
    10**(-4 * smoothing_factor): 0.0 -> 1.0 (almost none), 0.5 -> 0.01
    (same default as cardiac/combine_segs.py).
    """
    if smooth_method == 'cotangent_taubin':
        it = 400 if iterations is None else iterations
        return vf.taubin_smooth_polydata(poly, it=it, mu1=0.5, mu2=0.51)
    if smooth_method == 'vtk_taubin':
        it = 25 if iterations is None else iterations
        factor = 0.5 if smoothing_factor is None else smoothing_factor
        smoothed = vf.smooth_polydata(poly, iteration=it, smoothingFactor=factor)
        return vf.compute_normals(smoothed)
    if smooth_method == 'vtk_laplacian':
        it = 25 if iterations is None else iterations
        return vf.laplacian_smooth_polydata(poly, iteration=it, relaxation=0.1)
    raise ValueError(f"Unknown smooth method: {smooth_method}")


def extract_per_label_surfaces(seg, bg_id=0, mask_labels=None):
    """Discrete marching cubes, one vtkPolyData per non-empty label.

    Returns dict mapping original label id -> polydata. Labels with 0 points
    are omitted.
    """
    vtk_img, _ = vf.exportSitk2VTK(seg)
    result = {}
    for label_id in unique_nonzero_labels(seg, mask_labels=mask_labels, bg_id=bg_id):
        poly = vf.vtk_discrete_marching_cube(vtk_img, bg_id, label_id)
        if poly is None or poly.GetNumberOfPoints() == 0:
            continue
        result[label_id] = poly
    return result


# All other functions moved to modules - import from there
# vtk_marching_cube_multi -> vf.vtk_marching_cube_multi
# eraseBoundary -> sf.eraseBoundary
# surface_to_image -> vf.surface_to_image
# convert_seg_to_surfs -> sf.convert_seg_to_surfs
# build_transform_matrix -> vf.build_transform_matrix
# exportSitk2VTK -> vf.exportSitk2VTK
# vtkImageResample -> vf.vtkImageResample
# vtk_marching_cube -> vf.vtk_discrete_marching_cube
# exportPython2VTK -> vf.exportPython2VTK
# smooth_polydata -> vf.smooth_polydata
# decimation -> vf.decimation
# appendPolyData -> vf.appendPolyData
# bound_polydata_by_image -> vf.bound_polydata_by_image
# convertPolyDataToImageData -> vf.convertPolyDataToImageData


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create surface meshes from segmentation images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_surf_from_seg.py --segmentations_dir /path/to/segmentations --output_dir /path/to/output
  
  # Extract only specific mask labels (e.g. aorta=1, vena cava=2):
  python create_surf_from_seg.py --masks 1,2

  # One surface file per label, plus Taubin copies:
  python create_surf_from_seg.py --per-label --label-names 1=airways,3=arteries \\
      --label-dir-prefix surfaces_mc --also-smooth --smooth-dir-prefix surfaces_taubin
  
  # Using default directory:
  python create_surf_from_seg.py
        """
    )
    parser.add_argument('--segmentations_dir', '--segmentations-dir',
                       type=str,
                       default=None,
                       help='Directory containing segmentation image files. '
                            'Defaults to ./data/truths/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write output surface files. '
                            'Defaults to segmentations_dir with "truths" replaced by "surfaces_mc"')
    parser.add_argument('--spacing_file', '--spacing-file',
                       type=str,
                       default=None,
                       help='CSV file containing spacing values')
    parser.add_argument('--filter_string', '--filter-string',
                       type=str,
                       default='',
                       help='Only process images containing this string (default: process all)')
    parser.add_argument('--smooth',
                       action='store_true',
                       default=False,
                       help='Apply smoothing to surfaces')
    parser.add_argument('--smooth_method', '--smooth-method',
                       type=str,
                       default='cotangent_taubin',
                       choices=['cotangent_taubin', 'vtk_taubin', 'vtk_laplacian'],
                       help='Which smoothing method to apply when --smooth/--also-smooth is set '
                            '(default: cotangent_taubin). '
                            'cotangent_taubin: numpy cotangent-Laplacian Taubin smoothing; '
                            'vtk_taubin: VTK windowed-sinc (Taubin pass-band) smoothing; '
                            'vtk_laplacian: VTK Laplacian smoothing.')
    parser.add_argument('--smooth-iterations', '--smooth_iterations',
                       type=int,
                       default=None,
                       help='Smoothing iterations. Defaults: vtk_taubin/vtk_laplacian=25, '
                            'cotangent_taubin=400.')
    parser.add_argument('--smoothing-factor', '--smoothing_factor',
                       type=float,
                       default=0.5,
                       help='vtk_taubin windowed-sinc factor (pass band = 10^(-4 * factor)). '
                            'Default 0.5 -> pass band 0.01. 0.0 -> pass band 1.0 (almost none).')
    parser.add_argument('--keep_largest', '--keep-largest',
                       type=int,
                       nargs='?',
                       const=1,
                       default=None,
                       metavar='N',
                       help='Keep the N largest connected components (default N=1). '
                            'Example: --keep-largest 2 keeps aorta + cava. Omit flag to keep all components.')
    parser.add_argument('--img_ext', '--img-ext',
                       type=str,
                       default='.mha',
                       help='Input image file extension (default: .mha)')
    parser.add_argument('--output_ext', '--output-ext',
                       type=str,
                       default='.vtp',
                       help='Output surface file extension (default: .vtp)')
    parser.add_argument('--masks', '--mask',
                       type=str,
                       default=None,
                       help='Comma-separated list of label values to extract (e.g. "1" or "1,2,3"). '
                            'If not specified, extracts surfaces for all non-zero labels.')
    parser.add_argument('--per-label', '--per_label',
                       action='store_true',
                       default=False,
                       help='Write one surface file per label instead of a combined mesh.')
    parser.add_argument('--label-names', '--label_names',
                       type=str,
                       default=None,
                       help='Comma-separated id=name map, e.g. "1=airways,2=label2,3=arteries". '
                            'Unnamed labels use "label{id}".')
    parser.add_argument('--label-dir-prefix', '--label_dir_prefix',
                       type=str,
                       default=None,
                       help='With --per-label, write to {output_dir}/{prefix}_{name}/{stem}.vtp. '
                            'Example: surfaces_mc -> surfaces_mc_airways/<id>.vtp')
    parser.add_argument('--also-smooth', '--also_smooth',
                       action='store_true',
                       default=False,
                       help='With --per-label, also write a smoothed copy of each surface. '
                            'Requires --smooth-dir-prefix when --label-dir-prefix is set.')
    parser.add_argument('--smooth-dir-prefix', '--smooth_dir_prefix',
                       type=str,
                       default='surfaces_taubin',
                       help='Directory prefix for --also-smooth outputs '
                            '(default: surfaces_taubin).')
    parser.add_argument('--skip-existing', '--skip_existing',
                       action='store_true',
                       default=False,
                       help='Skip writing files that already exist (resume a long run).')
    
    args = parser.parse_args()
    
    if_smooth = args.smooth
    smooth_method = args.smooth_method
    smooth_iterations = args.smooth_iterations
    smoothing_factor = args.smoothing_factor
    keep_largest_n = args.keep_largest
    per_label = args.per_label
    also_smooth = args.also_smooth
    skip_existing = args.skip_existing
    label_dir_prefix = args.label_dir_prefix
    smooth_dir_prefix = args.smooth_dir_prefix
    label_names = parse_label_names(args.label_names)

    if also_smooth and not per_label:
        raise ValueError("--also-smooth requires --per-label")
    if label_dir_prefix and not per_label:
        raise ValueError("--label-dir-prefix requires --per-label")
    if args.label_names and not per_label:
        raise ValueError("--label-names requires --per-label")
    if also_smooth and label_dir_prefix and not smooth_dir_prefix:
        raise ValueError("--also-smooth with --label-dir-prefix requires --smooth-dir-prefix")

    if_spacing_file = args.spacing_file is not None
    spacing_file = args.spacing_file
    if if_spacing_file and not spacing_file:
        raise ValueError("--spacing_file argument must be set when using spacing file")
    
    mask_labels = parse_mask_labels(args.masks)
    
    # Filter option: only process images containing this string
    filter_string = args.filter_string or ''

    # Use command-line arguments (required or default)
    dir_segmentations = args.segmentations_dir or './data/truths/'
    
    if not os.path.exists(dir_segmentations):
        raise ValueError(f"Segmentations directory not found: {dir_segmentations}. "
                        f"Provide --segmentations_dir argument.")

    img_ext = args.img_ext
    img_ext_out = args.output_ext
    # Which folder to write surfaces to
    out_dir = (args.output_dir or 
              dir_segmentations.replace('truths', 'surfaces_mc/'))
    
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        raise

    imgs = os.listdir(dir_segmentations)
    imgs = [img for img in imgs if img.endswith(img_ext)]
    
    if filter_string:
        original_count = len(imgs)
        imgs = [img for img in imgs if filter_string in img]
        logger.info(f"Filtered from {original_count} to {len(imgs)} images containing '{filter_string}'")
    else:
        logger.info(f"Processing all {len(imgs)} images")
    
    imgs.sort()
    
    if len(imgs) == 0:
        logger.error("No images to process!")
        exit(1)

    if if_spacing_file:
        import pandas as pd
        spacing_df = pd.read_csv(spacing_file)
        spacing_values = spacing_df['spacing'].values
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    def maybe_write_geo(path, poly):
        if skip_existing and os.path.exists(path):
            logger.debug(f"Skipping existing: {path}")
            return False
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        vf.write_geo(path, poly)
        return True

    for img in imgs:
        logger.info(f"Starting case: {img}")
        seg = sitk.ReadImage(os.path.join(dir_segmentations, img))
        origin = seg.GetOrigin()
        stem = img[:-len(img_ext)] if img.endswith(img_ext) else os.path.splitext(img)[0]

        if if_spacing_file:
            seg.SetSpacing(spacing_values[imgs.index(img)])
            if not per_label:
                sitk.WriteImage(seg, os.path.join(out_dir, img.replace(img_ext, img_ext_out)))

        logger.debug(f"Image size: {seg.GetSize()}")
        logger.debug(f"Image spacing: {seg.GetSpacing()}")

        if per_label:
            vtk_img, _ = vf.exportSitk2VTK(seg)
            labels = unique_nonzero_labels(seg, mask_labels=mask_labels, bg_id=0)
            if not labels:
                logger.info(f"No labels to extract for {img}")
                continue
            logger.info(f"Labels for {img}: {labels}")
            for label_id in labels:
                raw_path = label_output_path(
                    out_dir, stem, label_id, label_names,
                    dir_prefix=label_dir_prefix, output_ext=img_ext_out,
                )
                smooth_path = None
                if also_smooth:
                    smooth_path = label_output_path(
                        out_dir, stem, label_id, label_names,
                        dir_prefix=smooth_dir_prefix, output_ext=img_ext_out,
                    )
                raw_exists = os.path.exists(raw_path)
                smooth_exists = smooth_path is not None and os.path.exists(smooth_path)
                if skip_existing and raw_exists and (smooth_path is None or smooth_exists):
                    logger.debug(f"Skipping existing label {label_id} for {img}")
                    continue

                poly = None
                if not (skip_existing and raw_exists):
                    poly = vf.vtk_discrete_marching_cube(vtk_img, 0, int(label_id))
                    if keep_largest_n is not None:
                        poly = vf.get_k_largest_connected_polydata(poly, keep_largest_n)
                    if poly is None or poly.GetNumberOfPoints() == 0:
                        logger.info(f"Empty surface for {img} label {label_id}, skipping")
                        continue
                    maybe_write_geo(raw_path, poly)
                    logger.debug(f"Wrote {raw_path}")
                elif also_smooth and not smooth_exists:
                    reader = vf.read_geo(raw_path)
                    poly = reader.GetOutput()

                if also_smooth and not (skip_existing and smooth_exists):
                    if poly is None or poly.GetNumberOfPoints() == 0:
                        logger.info(f"Empty surface for {img} label {label_id}, skipping smooth")
                        continue
                    smoothed = apply_smoothing(
                        poly, smooth_method,
                        iterations=smooth_iterations,
                        smoothing_factor=smoothing_factor,
                    )
                    maybe_write_geo(smooth_path, smoothed)
                    logger.debug(f"Wrote {smooth_path}")
            logger.info(f"Finished case: {img}")
            continue

        if mask_labels is not None:
            seg_arr = sitk.GetArrayFromImage(seg)
            mask = np.isin(seg_arr, mask_labels)
            seg_arr_filtered = np.where(mask, seg_arr, 0)
            seg_filtered = sitk.GetImageFromArray(seg_arr_filtered)
            seg_filtered.CopyInformation(seg)
            seg = seg_filtered
            logger.debug(f"Extracting masks: {mask_labels}")

        combined_path = os.path.join(out_dir, img.replace(img_ext, img_ext_out))
        if skip_existing and os.path.exists(combined_path):
            logger.info(f"Skipping existing: {combined_path}")
            continue

        poly = vf.vtk_marching_cube_multi(vf.exportSitk2VTK(seg)[0], 0, rotate=False, center=origin)

        if keep_largest_n is not None:
            poly = vf.get_k_largest_connected_polydata(poly, keep_largest_n)

        if if_smooth:
            poly = apply_smoothing(
                poly, smooth_method,
                iterations=smooth_iterations,
                smoothing_factor=smoothing_factor,
            )
            logger.debug(f"Smoothed surface using method: {smooth_method}")
        vf.write_geo(combined_path, poly)
        logger.info(f"Finished case: {img}")
    logger.info("All done.")
