# import time
# start_time = time.time()

import os
import sys
import glob

# Add project root to path for module imports (when run as script)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from modules.pre_process import resample
import SimpleITK as sitk


def resample_image(img_sitk, target_size=None, target_spacing=None, order=1):
    """
    Resample a SimpleITK image to a target size or spacing.
    
    Args:
        img_sitk: SimpleITK image to resample
        target_size: Target image size [x, y, z]. If provided, target_spacing is ignored.
        target_spacing: Target voxel spacing [x, y, z] in mm
        order: Interpolation order (0=nearest, 1=linear, 2=bspline)
    
    Returns:
        Resampled SimpleITK image
    
    Raises:
        ValueError: If neither target_size nor target_spacing is provided
    """
    if target_size is None and target_spacing is None:
        raise ValueError("Either target_size or target_spacing must be provided")
    
    if target_size is not None:
        # Calculate new spacing to achieve target size
        new_res = [img_sitk.GetSize()[0] / target_size[0],
                   img_sitk.GetSize()[1] / target_size[1],
                   img_sitk.GetSize()[2] / target_size[2]]
        new_res = [img_sitk.GetSpacing()[0] * new_res[0],
                   img_sitk.GetSpacing()[1] * new_res[1],
                   img_sitk.GetSpacing()[2] * new_res[2]]
    else:
        # Use target spacing directly
        new_res = target_spacing
    
    # Resample the image
    resampled_img = resample(img_sitk, resolution=new_res, order=order, dim=3)
    
    return resampled_img


def resample_images_batch(data_folder, out_folder, input_format='.mha', 
                          target_size=None, target_spacing=None, order=1,
                          testing_samples=None, skip_existing=True,
                          output_pixel_type=None, folder_label=''):
    """
    Batch resample all images in a folder.
    
    Args:
        data_folder: Input folder containing images
        out_folder: Output folder for resampled images
        input_format: Image file extension (e.g., '.mha', '.vti')
        target_size: Target image size [x, y, z]
        target_spacing: Target voxel spacing [x, y, z] in mm
        order: Interpolation order (0=nearest, 1=linear, 2=bspline)
        testing_samples: Optional list of sample names to filter
        skip_existing: Skip processing if output file already exists
        output_pixel_type: Optional sitk pixel type to cast before writing
    
    Returns:
        List of (filename, spacing_record) for processed images. spacing_record: {orig_spacing, new_spacing, orig_size, new_size}
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]
    imgs = sorted(imgs)
    
    # Filter images if testing_samples is provided
    if testing_samples:
        imgs = [img for img in imgs if any(ts in img for ts in testing_samples)]
    
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info(f'Found {len(imgs)} images to resample')
    logger.debug(f'Images to resample: {imgs}')
    
    processed = []
    spacing_records = []
    
    for img in imgs:
        img_path = os.path.join(data_folder, img)
        img_out_path = os.path.join(out_folder, img)
        
        # Skip if output already exists
        if skip_existing and os.path.exists(img_out_path):
            logger.info(f'Image {img} already processed, skipping...')
            continue
        
        img_sitk = sitk.ReadImage(img_path)
        orig_spacing = list(img_sitk.GetSpacing())
        orig_size = list(img_sitk.GetSize())
        
        logger.debug(f'Image {img} read')
        logger.debug(f"Image {img} shape: {img_sitk.GetSize()}")
        logger.debug(f"Image {img} spacing: {img_sitk.GetSpacing()}")
        
        # Resample the image
        if target_size is not None:
            logger.debug(f"Image {img} target size: {target_size}")
            img_sitk = resample_image(img_sitk, target_size=target_size, order=order)
        else:
            logger.debug(f"Image {img} target spacing: {target_spacing}")
            img_sitk = resample_image(img_sitk, target_spacing=target_spacing, order=order)
        
        logger.debug(f"Image {img} resampled shape: {img_sitk.GetSize()}")
        logger.debug(f"Image {img} resampled spacing: {img_sitk.GetSpacing()}")
        
        if output_pixel_type:
            img_sitk = sitk.Cast(img_sitk, output_pixel_type)
        sitk.WriteImage(img_sitk, img_out_path)
        logger.info(f'Image {img} resampled and saved to {img_out_path}')
        
        spacing_records.append({
            'file': img, 'folder': folder_label,
            'orig_spacing': orig_spacing, 'new_spacing': list(img_sitk.GetSpacing()),
            'orig_size': orig_size, 'new_size': list(img_sitk.GetSize())
        })
        processed.append(img)
    
    return processed, spacing_records


if __name__=='__main__':
    import argparse
    import ast
    
    parser = argparse.ArgumentParser(
        description='Resample images to target size or spacing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resample to target spacing:
  python change_img_resample.py --input_dir /path/to/images --output_dir /path/to/output --target_spacing 0.03 0.03 0.03
  
  # Resample to target size:
  python change_img_resample.py --input_dir /path/to/images --target_size 512 512 512
  
  # Using default directory:
  python change_img_resample.py --target_spacing 1.0 1.0 1.0
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input images. '
                            'Defaults to ./data/images/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Base directory for resampled output. '
                            'Default: input_dir/resampled/ (writes images_resampled/ and truths_resampled/)')
    parser.add_argument('--input_format', '--input-format',
                       type=str,
                       default='.mha',
                       help='Input file extension (default: .mha)')
    parser.add_argument('--target_size', '--target-size',
                       type=int,
                       nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=None,
                       help='Target image size [x, y, z]. Mutually exclusive with --target_spacing')
    parser.add_argument('--target_spacing', '--target-spacing',
                       type=float,
                       nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=None,
                       help='Target voxel spacing [x, y, z] in mm. Mutually exclusive with --target_size')
    parser.add_argument('--order',
                       type=int,
                       default=1,
                       choices=[0, 1, 2],
                       help='Interpolation order: 0=nearest, 1=linear, 2=bspline (default: 1)')
    parser.add_argument('--testing_samples', '--testing-samples',
                       type=str,
                       nargs='+',
                       default=None,
                       help='Optional list of sample names to filter (process only these)')
    parser.add_argument('--no_skip_existing', '--no-skip-existing',
                       dest='skip_existing',
                       action='store_false',
                       help='Re-process files even if output already exists')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target_size and args.target_spacing:
        parser.error("--target_size and --target_spacing are mutually exclusive")
    if not args.target_size and not args.target_spacing:
        parser.error("Either --target_size or --target_spacing must be provided")
    
    # Use command-line arguments (required or default)
    data_folder = args.input_dir or './data/images/'
    
    # Validate root directory exists
    if not os.path.exists(data_folder):
        raise ValueError(f"Input directory not found: {data_folder}. "
                        f"Provide --input_dir argument.")
    
    # Auto-detect images subfolder: if no matching files in data_folder, try data_folder/images
    images_folder = data_folder
    images_subdir = os.path.join(data_folder, 'images')
    if os.path.isdir(images_subdir):
        files_in_root = [f for f in os.listdir(data_folder) if f.endswith(args.input_format)]
        files_in_images = [f for f in os.listdir(images_subdir) if f.endswith(args.input_format)]
        if not files_in_root and files_in_images:
            images_folder = images_subdir
    
    # Default: data_folder/resampled/images_resampled and data_folder/resampled/truths_resampled
    resampled_base = args.output_dir or os.path.join(data_folder, 'resampled')
    images_out_folder = os.path.join(resampled_base, 'images_resampled')
    truths_out_folder = os.path.join(resampled_base, 'truths_resampled')
    
    all_spacing_records = []
    
    # Process main batch
    _, records = resample_images_batch(
        data_folder=images_folder, out_folder=images_out_folder,
        input_format=args.input_format, target_size=args.target_size,
        target_spacing=args.target_spacing, order=args.order,
        testing_samples=args.testing_samples, skip_existing=args.skip_existing,
        folder_label='images'
    )
    all_spacing_records.extend(records)
    
    # Resample truths folder if it exists (use nearest-neighbor for label images)
    truths_folder = os.path.join(data_folder, 'truths')
    if os.path.isdir(truths_folder):
        _, records = resample_images_batch(
            data_folder=truths_folder, out_folder=truths_out_folder,
            input_format=args.input_format, target_size=args.target_size,
            target_spacing=args.target_spacing, order=0,
            testing_samples=args.testing_samples, skip_existing=args.skip_existing,
            output_pixel_type=sitk.sitkInt8, folder_label='truths'
        )
        all_spacing_records.extend(records)
    
    # Write spacing changes log
    log_path = os.path.join(resampled_base, 'spacing_changes.txt')
    with open(log_path, 'w') as f:
        f.write("Resampling spacing changes\n")
        f.write("========================\n")
        if args.target_spacing:
            f.write(f"Target spacing: {list(args.target_spacing)} mm\n")
        else:
            f.write(f"Target size: {list(args.target_size)}\n")
        f.write("\n")
        for r in all_spacing_records:
            f.write(f"{r['folder']}/{r['file']}\n")
            f.write(f"  original: spacing={r['orig_spacing']} size={r['orig_size']}\n")
            f.write(f"  resampled: spacing={r['new_spacing']} size={r['new_size']}\n\n")
