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
    parser.add_argument('--keep_largest', '--keep-largest',
                       action='store_true',
                       default=False,
                       help='Keep only the largest connected component')
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
    
    args = parser.parse_args()
    
    if_smooth = args.smooth
    if_keep_largest = args.keep_largest

    if_spacing_file = args.spacing_file is not None
    spacing_file = args.spacing_file
    if if_spacing_file and not spacing_file:
        raise ValueError("--spacing_file argument must be set when using spacing file")
    
    # Mask option: which labels to extract (None = all non-zero)
    mask_labels = None
    if args.masks is not None:
        mask_labels = [int(x.strip()) for x in args.masks.split(',') if x.strip()]
        if not mask_labels:
            raise ValueError("--masks must specify at least one label value")
    
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
    
    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        raise

    # all segmentations we have, create surfaces for each
    imgs = os.listdir(dir_segmentations)
    imgs = [img for img in imgs if img.endswith(img_ext)]
    
    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    # Filter images by string if specified
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
        # only keep 'spacing', they are sorted
        spacing_values = spacing_df['spacing'].values
        # read as tuples
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    for img in imgs:
        logger.info(f"Starting case: {img}")
        # Load segmentation
        seg = sitk.ReadImage(os.path.join(dir_segmentations, img))
        origin = seg.GetOrigin()

        if if_spacing_file:
            # set the spacing
            seg.SetSpacing(spacing_values[imgs.index(img)])
            sitk.WriteImage(seg, os.path.join(out_dir, img.replace(img_ext, img_ext_out)))

        logger.debug(f"Image size: {seg.GetSize()}")
        logger.debug(f"Image spacing: {seg.GetSpacing()}")
        
        # Optionally filter to specific mask labels
        if mask_labels is not None:
            seg_arr = sitk.GetArrayFromImage(seg)
            mask = np.isin(seg_arr, mask_labels)
            seg_arr_filtered = np.where(mask, seg_arr, 0)
            seg_filtered = sitk.GetImageFromArray(seg_arr_filtered)
            seg_filtered.CopyInformation(seg)
            seg = seg_filtered
            logger.debug(f"Extracting masks: {mask_labels}")
        
        # Create surfaces
        # poly = sf.convert_seg_to_surfs(seg, new_spacing=[.5,.5,.5], target_node_num=1e5, bound=False)
        poly = vf.vtk_marching_cube_multi(vf.exportSitk2VTK(seg)[0], 0, rotate=False, center=origin)

        if if_keep_largest:
            # keep only the largest connected component
            poly = vf.get_largest_connected_polydata(poly)

        if if_smooth:
            # smooth the surface
            poly = vf.smooth_polydata(poly, iteration=50)
        # Write surfaces
        vf.write_geo(os.path.join(out_dir, img.replace(img_ext, '.vtp')), poly)
        logger.info(f"Finished case: {img}")
    logger.info("All done.")
