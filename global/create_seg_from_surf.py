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
from preprocessing.change_img_resample import resample_image


def load_surface_polydata(surface_path):
    """VTK PolyData from a triangular mesh file (.vtp via project reader, or .stl)."""
    if surface_path.endswith('.stl'):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(surface_path)
        reader.Update()
        return reader.GetOutput()
    return vf.read_geo(surface_path).GetOutput()


def reference_sitk_for_seg(img_sitk, target_spacing=None, order=1):
    """
    Grid used for rasterizing the surface: same as img_sitk, or resampled to target_spacing.

    Origin, direction, and physical extent follow the input image (see modules.pre_process.resample).
    """
    if target_spacing is None:
        return img_sitk
    return resample_image(img_sitk, target_spacing=list(target_spacing), order=order)


def _rasterize_surface_to_vtk_seg(
    surface_polydata,
    reference_image_sitk,
    target_spacing=None,
    resample_order=1,
):
    ref_sitk = reference_sitk_for_seg(
        reference_image_sitk, target_spacing=target_spacing, order=resample_order
    )
    img_vtk = vf.exportSitk2VTK(ref_sitk)[0]
    seg_vtk = vf.convertPolyDataToImageData(surface_polydata, img_vtk)
    return seg_vtk, ref_sitk


def _vtk_seg_to_sitk_u8(seg_vtk, ref_sitk):
    vtk_array = vtk_to_numpy(seg_vtk.GetPointData().GetScalars())
    dims = seg_vtk.GetDimensions()
    vtk_array = vtk_array.reshape(dims, order='F')
    seg_sitk = sitk.GetImageFromArray(vtk_array.transpose(2, 1, 0))
    seg_sitk.SetOrigin(ref_sitk.GetOrigin())
    seg_sitk.SetSpacing(ref_sitk.GetSpacing())
    seg_sitk.SetDirection(ref_sitk.GetDirection())
    return sitk.Cast(seg_sitk, sitk.sitkUInt8)


def seg_sitk_from_surface_polydata(
    surface_polydata,
    reference_image_sitk,
    target_spacing=None,
    resample_order=1,
):
    """
    Rasterize a closed surface into a binary SimpleITK label image on the same grid as
    ``reference_image_sitk`` (optionally resampled to ``target_spacing`` first), using
    ``vtkPolyDataToImageStencil`` / ``vtkImageStencil`` (same path as the CLI batch tool).

    Args:
        surface_polydata: vtkPolyData mesh.
        reference_image_sitk: SimpleITK image defining geometry (and optionally voxel size before resample).
        target_spacing: Optional (sx, sy, sz) in mm for the seg grid.
        resample_order: Interpolation order when resampling the reference image (SimpleITK).

    Returns:
        sitk.Image, UInt8, foreground 1 / background 0.
    """
    seg_vtk, ref_sitk = _rasterize_surface_to_vtk_seg(
        surface_polydata,
        reference_image_sitk,
        target_spacing=target_spacing,
        resample_order=resample_order,
    )
    return _vtk_seg_to_sitk_u8(seg_vtk, ref_sitk)


def create_segs_from_surface_dirs(
    surfaces_dir,
    images_dir,
    output_dir,
    img_ext='.mha',
    output_ext='.mha',
    target_spacing=None,
    resample_order=1,
    logger=None,
):
    """
    Rasterize each surface (.vtp or .stl) onto the grid of its matching image and write
    segmentations to output_dir.

    Args:
        surfaces_dir: Directory of meshes named like the image basename with .vtp/.stl.
        images_dir: Directory of images (filtered by img_ext).
        output_dir: Write segmentations here (created if missing).
        img_ext: Input image filename suffix (e.g. '.mha').
        output_ext: Output suffix (e.g. '.mha' or '.vti').
        target_spacing: Optional (sx, sy, sz) mm for reference grid (see reference_sitk_for_seg).
        resample_order: Interpolation order when resampling the reference image.
        logger: Optional logger; if None, uses modules.logger.get_logger(__name__).
    """
    if logger is None:
        from modules.logger import get_logger
        logger = get_logger(__name__)

    if not os.path.exists(surfaces_dir):
        raise ValueError(
            f"Surfaces directory not found: {surfaces_dir}."
        )
    if not os.path.exists(images_dir):
        raise ValueError(
            f"Images directory not found: {images_dir}."
        )

    os.makedirs(output_dir, exist_ok=True)

    imgs = [f for f in os.listdir(images_dir) if f.endswith(img_ext)]

    for img in imgs:
        surf_path_vtp = os.path.join(surfaces_dir, img.replace(img_ext, '.vtp'))
        surf_path_stl = os.path.join(surfaces_dir, img.replace(img_ext, '.stl'))

        if os.path.exists(surf_path_vtp):
            surf_path = surf_path_vtp
        elif os.path.exists(surf_path_stl):
            surf_path = surf_path_stl
        else:
            logger.warning(f"Skipping case {img}: No surface file (.vtp or .stl) found")
            continue

        output_path = os.path.join(output_dir, img.replace(img_ext, output_ext))

        if os.path.exists(output_path):
            logger.info(f"Skipping case {img}: Output file {output_path} already exists")
            continue

        surf_vtp = load_surface_polydata(surf_path)

        img_sitk = sitk.ReadImage(os.path.join(images_dir, img))
        seg_vtk, ref_sitk = _rasterize_surface_to_vtk_seg(
            surf_vtp,
            img_sitk,
            target_spacing=target_spacing,
            resample_order=resample_order,
        )

        if output_ext == '.vti':
            vf.write_img(output_path, seg_vtk)
        else:
            seg_sitk = _vtk_seg_to_sitk_u8(seg_vtk, ref_sitk)
            sitk.WriteImage(seg_sitk, output_path)

        logger.info(f"Done case: {img}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create segmentation images from surface meshes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_seg_from_surf.py --surfaces_dir /path/to/surfaces --images_dir /path/to/images --output_dir /path/to/output
  
  # Finer seg grid than the image (same origin/direction/extent as preprocessing/change_img_resample.py):
  python create_seg_from_surf.py --images_dir /path/to/images --surfaces_dir /path/to/surfaces --target_spacing 0.4 0.4 0.4
  
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
    parser.add_argument('--target_spacing', '--target-spacing',
                       type=float,
                       nargs=3,
                       metavar=('SX', 'SY', 'SZ'),
                       default=None,
                       help='Optional output voxel spacing in mm [x, y, z]. '
                            'Seg is rasterized on the same grid as resampling the image to this '
                            'spacing (origin, direction, and extent from the image; see '
                            'preprocessing/change_img_resample.py). Omit to match image spacing.')

    args = parser.parse_args()

    create_segs_from_surface_dirs(
        surfaces_dir=args.surfaces_dir or './data/surfaces/',
        images_dir=args.images_dir or './data/images/',
        output_dir=args.output_dir or './data/truths/',
        img_ext=args.img_ext,
        output_ext=args.output_ext,
        target_spacing=args.target_spacing,
    )
