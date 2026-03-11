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


def create_seg_from_surface(surface, image):
    """
    Check all voxels:
    if voxel inside surface: voxel = 1
    if outside: voxel = 0
    Args:
        surface: VTK PolyData
        image: Sitk Image
    """

    # Assemble all points in image
    img_size = image.GetSize()
    points = vtk.vtkPoints()
    count = 0
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            for k in range(img_size[2]):
                point = image.TransformIndexToPhysicalPoint((i,j,k))
                points.InsertNextPoint(point)
                count += 1

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)

    # Create filter to check inside/outside
    enclosed_filter = vtk.vtkSelectEnclosedPoints()
    enclosed_filter.SetTolerance(0.001)
    # enclosed_filter.SetSurfaceClosed(True)
    # enclosed_filter.SetCheckSurface(True)

    enclosed_filter.SetInputData(pointsPolydata)
    enclosed_filter.SetSurfaceData(surface)
    enclosed_filter.Update()

    # Create new image to assemble
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            for k in range(img_size[2]):
                point = image.TransformIndexToPhysicalPoint((i,j,k))
                is_inside = enclosed_filter.IsInsideSurface(point[0], point[1], point[2])
                if is_inside:
                    # Voxel is inside surface
                    image[i, j, k] = 1
                else:
                    image[i, j, k] = 0

    return image


# All functions moved to modules - import from there
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
        description='Create segmentation images from surface meshes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_seg_from_surf.py --surfaces_dir /path/to/surfaces --images_dir /path/to/images --output_dir /path/to/output
  
  # Using default directories:
  python create_seg_from_surf.py
        """
    )
    parser.add_argument('--surfaces_dir', '--surfaces-dir',
                       type=str,
                       default=None,
                       help='Directory containing surface mesh files (.vtp or .stl). '
                            'Defaults to ./data/surfaces/')
    parser.add_argument('--images_dir', '--images-dir',
                       type=str,
                       default=None,
                       help='Directory containing image files. '
                            'Defaults to ./data/images/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write output segmentation files. '
                            'Defaults to ./data/truths/')
    parser.add_argument('--img_ext', '--img-ext',
                       type=str,
                       default='.mha',
                       help='Image file extension (default: .mha)')
    parser.add_argument('--output_ext', '--output-ext',
                       type=str,
                       default='.mha',
                       help='Output file extension (default: .mha)')
    
    args = parser.parse_args()
    
    # Let's create GT segmentations from surfaces
    img_ext = args.img_ext
    output_ext = args.output_ext
    
    # Use command-line arguments (required or default)
    dir_surfaces = args.surfaces_dir or './data/surfaces/'
    dir_imgs = args.images_dir or './data/images/'
    out_dir = args.output_dir or './data/truths/'
    
    # Validate directories exist
    if not os.path.exists(dir_surfaces):
        raise ValueError(f"Surfaces directory not found: {dir_surfaces}. "
                        f"Provide --surfaces_dir argument.")
    if not os.path.exists(dir_imgs):
        raise ValueError(f"Images directory not found: {dir_imgs}. "
                        f"Provide --images_dir argument.")

    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    # create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # all imgs we have, create segs for them
    imgs = os.listdir(dir_imgs)
    imgs = [img for img in imgs if img.endswith(img_ext)]
    
    for img in imgs:
        # Check for both .vtp and .stl surface files
        surf_path_vtp = os.path.join(dir_surfaces, img.replace(img_ext, '.vtp'))
        surf_path_stl = os.path.join(dir_surfaces, img.replace(img_ext, '.stl'))
        
        # Determine which surface file exists
        if os.path.exists(surf_path_vtp):
            surf_path = surf_path_vtp
        elif os.path.exists(surf_path_stl):
            surf_path = surf_path_stl
        else:
            logger.warning(f"Skipping case {img}: No surface file (.vtp or .stl) found")
            continue
        
        output_path = os.path.join(out_dir, img.replace(img_ext, output_ext))

        # Check if output file already exists
        if os.path.exists(output_path):
            logger.info(f"Skipping case {img}: Output file {output_path} already exists")
            continue
        
        # Read surface based on file type
        if surf_path.endswith('.stl'):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(surf_path)
            reader.Update()
            surf_vtp = reader.GetOutput()
        else:  # .vtp file
            surf_vtp = vf.read_geo(surf_path).GetOutput()
        
        img_sitk = sitk.ReadImage(os.path.join(dir_imgs, img))
        img_vtk = vf.exportSitk2VTK(img_sitk)[0]
        # img_vtk = vf.read_img(dir_imgs+img).GetOutput()
        # seg = vf.convertPolyDataToImageData(surf_vtp, img_vtk)
        seg = vf.convertPolyDataToImageData(surf_vtp, img_vtk)
        
        # Write output in the specified format
        if output_ext == '.vti':
            vf.write_img(output_path, seg)
        else:
            # Convert VTK image to SITK and save
            # Get numpy array from VTK image
            vtk_array = vtk_to_numpy(seg.GetPointData().GetScalars())
            dims = seg.GetDimensions()
            vtk_array = vtk_array.reshape(dims, order='F')
            
            # Create SITK image from numpy array
            seg_sitk = sitk.GetImageFromArray(vtk_array.transpose(2, 1, 0))
            
            # Copy metadata from original image
            seg_sitk.SetOrigin(img_sitk.GetOrigin())
            seg_sitk.SetSpacing(img_sitk.GetSpacing())
            seg_sitk.SetDirection(img_sitk.GetDirection())
            
            seg_sitk = sitk.Cast(seg_sitk, sitk.sitkUInt8)
            sitk.WriteImage(seg_sitk, output_path)
        
        # vf.change_vti_vtk(output_path)
        logger.info(f"Done case: {img}")
