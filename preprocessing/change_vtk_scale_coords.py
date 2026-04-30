import os
import sys
from pathlib import Path

import vtk

# Add project root to path so "from modules import ..." works
_project_root = Path(__file__).resolve().parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))


def scale_polydata(
    input_file,
    output_file,
    scale_factor,
    direction_matrix=None,
    rotation_center=None,
    scale_about_center=False,
):
    # Determine file type and use appropriate reader
    if input_file.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif input_file.endswith('.stl'):
        reader = vtk.vtkSTLReader()
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    reader.SetFileName(input_file)
    reader.Update()

    # Get the polydata from the reader
    polydata = reader.GetOutput()

    # Get the points of the polydata
    points = polydata.GetPoints()

    # Scale and optionally transform the points
    # rotation_center: center of rotation in input mesh coords (e.g. image origin).
    # Applied after scaling so mesh and image use same pivot.
    m = direction_matrix  # row-major 3x3: [m00,m01,m02, m10,m11,m12, m20,m21,m22]
    rcx, rcy, rcz = (0.0, 0.0, 0.0) if rotation_center is None else rotation_center
    # Anchor scaling at center if requested; otherwise keep legacy scale-around-origin behavior.
    scx, scy, scz = (rcx, rcy, rcz) if scale_about_center else (0.0, 0.0, 0.0)
    # Rotation happens after scaling; when scaling is about origin, pivot must be scaled too.
    if scale_about_center:
        cx, cy, cz = rcx, rcy, rcz
    else:
        cx, cy, cz = rcx * scale_factor, rcy * scale_factor, rcz * scale_factor
    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)
        if scale_about_center:
            x = (x - scx) * scale_factor + scx
            y = (y - scy) * scale_factor + scy
            z = (z - scz) * scale_factor + scz
        else:
            x, y, z = x * scale_factor, y * scale_factor, z * scale_factor
        if m is not None:
            # Rotate around center: p' = R*(p - c) + c
            x, y, z = x - cx, y - cy, z - cz
            x, y, z = (
                m[0] * x + m[1] * y + m[2] * z,
                m[3] * x + m[4] * y + m[5] * z,
                m[6] * x + m[7] * y + m[8] * z,
            )
            x, y, z = x + cx, y + cy, z + cz
        points.SetPoint(i, x, y, z)

    # Also scale any data arrays named 'MaximumInscribedSphereRadius' in point or cell data
    try:
        from modules.logger import get_logger
        logger = get_logger(__name__)
    except Exception:
        logger = None

    for data_name, data in (('point', polydata.GetPointData()), ('cell', polydata.GetCellData())):
        if not data:
            continue
        arr = data.GetArray('MaximumInscribedSphereRadius')
        if arr is None:
            continue
        nc = arr.GetNumberOfComponents()
        n_tuples = arr.GetNumberOfTuples()
        for tidx in range(n_tuples):
            if nc == 1:
                try:
                    val = arr.GetTuple1(tidx)
                    arr.SetTuple1(tidx, val * scale_factor)
                except AttributeError:
                    tup = arr.GetTuple(tidx)
                    arr.SetTuple(tidx, (tup[0] * scale_factor,))
            else:
                tup = list(arr.GetTuple(tidx))
                for c in range(nc):
                    tup[c] = tup[c] * scale_factor
                arr.SetTuple(tidx, tup)
        if logger:
            logger.info(f"Scaled 'MaximumInscribedSphereRadius' in {data_name} data by {scale_factor}")

    polydata.Modified()

    # Determine file type and use appropriate writer
    if output_file.endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
    elif output_file.endswith('.stl'):
        writer = vtk.vtkSTLWriter()
    else:
        raise ValueError(f"Unsupported file format: {output_file}")
    
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()


def _find_image_for_surface(surface_name, image_folder, image_extensions=('.mha', '.mhd', '.nii', '.nii.gz', '.nrrd')):
    """Find image in folder matching surface base name. Returns path or None."""
    stem = Path(surface_name).stem
    for ext in image_extensions:
        candidate = os.path.join(image_folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def process_folder(
    input_folder,
    output_folder,
    scale_factor,
    direction_matrix=None,
    rotation_center=None,
    image_folder_for_origin=None,
    scale_about_center=False,
):
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    # Loop over all .vtp and .stl files in the folder
    files = [f for f in os.listdir(input_folder) if f.endswith(('.vtp', '.stl'))]
    logger.info(f"Found {len(files)} surface files to process")
    
    for file_name in files:
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)

        # Resolve rotation center: fixed value, or from matching image in folder
        rc = rotation_center
        if rc is None and image_folder_for_origin is not None:
            img_path = _find_image_for_surface(file_name, image_folder_for_origin)
            if img_path:
                import SimpleITK as sitk
                img = sitk.ReadImage(img_path)
                origin = img.GetOrigin()
                rc = [float(origin[i]) for i in range(min(3, len(origin)))] if origin else None

        # Scale and optionally transform the polydata, then save
        scale_polydata(
            input_file,
            output_file,
            scale_factor,
            direction_matrix,
            rc,
            scale_about_center=scale_about_center,
        )
        logger.info(f"Scaled {file_name} and saved to {output_file}")
    
    logger.info(f"Completed scaling {len(files)} files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scale VTK polydata files (surfaces) by a factor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python change_vtk_scale_coords.py --input_dir /path/to/surfaces --output_dir /path/to/output --scale_factor 0.1
  
  # Rotate 180 deg around Z (e.g. to match image transform):
  python change_vtk_scale_coords.py --input_dir /path/to/surfaces --output_dir /path/to/rotated --scale_factor 1 --direction_matrix -1 0 0 0 -1 0 0 0 1
  
  # Same rotation around image origin (so mesh aligns with rotated image):
  python change_vtk_scale_coords.py --input_dir /path/to/surfaces --output_dir /path/to/rotated --direction_matrix -1 0 0 0 -1 0 0 0 1 --image_folder_for_origin /path/to/images
  
  # Using default directory:
  python change_vtk_scale_coords.py --scale_factor 0.1
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input surface files (.vtp or .stl). '
                            'Defaults to ./data/surfaces/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write scaled surface files. '
                            'Defaults to inferred from input_dir')
    parser.add_argument('--scale_factor', '--scale-factor',
                       type=float,
                       default=1.0,
                       help='Scale factor to apply to coordinates (default: 1.0, e.g. 0.1 to convert mm to cm)')
    parser.add_argument('--direction_matrix', '--direction-matrix',
                       type=float,
                       nargs=9,
                       metavar=('M00', 'M01', 'M02', 'M10', 'M11', 'M12', 'M20', 'M21', 'M22'),
                       default=None,
                       help='3x3 transform matrix (row-major) to apply to coordinates after scaling. '
                            'E.g. 180 deg around Z: -1 0 0 0 -1 0 0 0 1')
    parser.add_argument('--rotation_center', '--rotation-center',
                       type=float,
                       nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=None,
                       help='Center of rotation in mesh coordinates (e.g. image origin). '
                            'Use with direction_matrix to rotate around same point as image.')
    parser.add_argument('--image_folder_for_origin', '--image-folder-for-origin',
                       type=str,
                       default=None,
                       help='Folder of images; for each surface, use origin of matching image (same base name) '
                            'as rotation_center. Convenient when rotating mesh to match rotated images.')
    parser.add_argument('--scale_about_center', '--scale-about-center',
                       action='store_true',
                       default=False,
                       help='Scale around rotation_center instead of (0,0,0). '
                            'Useful with --image_folder_for_origin to keep mesh aligned to image origin.')
    
    args = parser.parse_args()
    
    # Use command-line arguments (required or default)
    input_folder = args.input_dir or './data/surfaces/'
    output_folder = args.output_dir or input_folder.rstrip('/') + '_scaled/'
    scale_factor = args.scale_factor
    direction_matrix = args.direction_matrix
    if args.rotation_center is not None:
        rotation_center = args.rotation_center
    else:
        rotation_center = None
    image_folder_for_origin = args.image_folder_for_origin
    scale_about_center = args.scale_about_center

    # Validate directories
    if not os.path.exists(input_folder):
        raise ValueError(f"Input directory not found: {input_folder}. "
                        f"Provide --input_dir argument.")

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)
    logger.info(
        f"Processing surfaces from {input_folder} to {output_folder} "
        f"(scale={scale_factor}, direction_matrix={'yes' if direction_matrix else 'no'}, "
        f"rotation_center={rotation_center}, image_folder_for_origin={image_folder_for_origin}, "
        f"scale_about_center={scale_about_center})"
    )
    
    process_folder(
        input_folder,
        output_folder,
        scale_factor,
        direction_matrix,
        rotation_center,
        image_folder_for_origin,
        scale_about_center=scale_about_center,
    )
