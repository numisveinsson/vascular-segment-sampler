import vtk
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Convert files from LPS to RAS coordinate system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert VTP surface file:
  python change_lps_ras.py --input_file /path/to/file.vtp --output_file /path/to/output.vtp --file_type vtp
  
  # Convert image file:
  python change_lps_ras.py --input_file /path/to/file.nrrd --output_file /path/to/output.nrrd --file_type image
  
  # Process all files in a directory (VTP):
  python change_lps_ras.py --input_dir /path/to/surfaces --output_dir /path/to/output --file_type vtp
        """
    )
    parser.add_argument('--input_file', '--input-file',
                       type=str,
                       default=None,
                       help='Input file path (for single file processing)')
    parser.add_argument('--output_file', '--output-file',
                       type=str,
                       default=None,
                       help='Output file path (for single file processing)')
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Input directory (for batch processing)')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Output directory (for batch processing)')
    parser.add_argument('--file_type', '--file-type',
                       type=str,
                       choices=['vtp', 'image'],
                       required=True,
                       help='Type of file to process: "vtp" for surface files, "image" for image files')
    parser.add_argument('--transform_type', '--transform-type',
                       type=str,
                       choices=['lps_to_ras', 'permute_xyz'],
                       default='lps_to_ras',
                       help='Type of transformation: lps_to_ras (flip X,Y) or permute_xyz (default: lps_to_ras)')
    
    args = parser.parse_args()
    
    from vascular_segment_sampler.logger import get_logger
    logger = get_logger(__name__)
    
    if args.input_file:
        # Single file processing
        if not args.output_file:
            parser.error("--output_file is required when using --input_file")
        
        input_path = args.input_file
        output_path = args.output_file
        
        if not os.path.exists(input_path):
            raise ValueError(f"Input file not found: {input_path}")
        
        if args.file_type == 'vtp':
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(input_path)
            reader.Update()
            poly_data = reader.GetOutput()
            points = poly_data.GetPoints()
            
            if args.transform_type == 'lps_to_ras':
                # Flip X and Y coordinates (LPS to RAS)
                for i in range(points.GetNumberOfPoints()):
                    x, y, z = points.GetPoint(i)
                    points.SetPoint(i, -x, -y, z)
            else:  # permute_xyz
                # Permute coordinates: z, y, x
                for i in range(points.GetNumberOfPoints()):
                    x, y, z = points.GetPoint(i)
                    points.SetPoint(i, z, y, x)
            
            poly_data.Modified()
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(output_path)
            writer.SetInputData(poly_data)
            writer.Write()
            logger.info(f"Converted VTP file: {input_path} -> {output_path}")
        
        elif args.file_type == 'image':
            import SimpleITK as sitk
            image = sitk.ReadImage(input_path)
            
            if args.transform_type == 'lps_to_ras':
                # Create direction matrix for RAS coordinate system
                lps_direction_matrix = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
                new_origin = (0, 0, image.GetOrigin()[2])
                image.SetDirection(lps_direction_matrix)
                image.SetOrigin(new_origin)
            # Note: permute_xyz not typically used for images
            
            sitk.WriteImage(image, output_path)
            logger.info(f"Converted image file: {input_path} -> {output_path}")
    
    elif args.input_dir:
        # Batch processing
        if not args.output_dir:
            parser.error("--output_dir is required when using --input_dir")
        
        input_dir = args.input_dir
        output_dir = args.output_dir
        
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if args.file_type == 'vtp':
            files = [f for f in os.listdir(input_dir) if f.endswith('.vtp')]
            logger.info(f"Found {len(files)} VTP files to process")
            
            for file_name in files:
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name.replace('.vtp', '_ras.vtp'))
                
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(input_path)
                reader.Update()
                poly_data = reader.GetOutput()
                points = poly_data.GetPoints()
                
                if args.transform_type == 'lps_to_ras':
                    for i in range(points.GetNumberOfPoints()):
                        x, y, z = points.GetPoint(i)
                        points.SetPoint(i, -x, -y, z)
                else:
                    for i in range(points.GetNumberOfPoints()):
                        x, y, z = points.GetPoint(i)
                        points.SetPoint(i, z, y, x)
                
                poly_data.Modified()
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName(output_path)
                writer.SetInputData(poly_data)
                writer.Write()
                logger.info(f"Converted: {file_name}")
        
        else:
            import SimpleITK as sitk
            files = [f for f in os.listdir(input_dir) 
                    if f.endswith(('.mha', '.nrrd', '.nii.gz', '.nii'))]
            logger.info(f"Found {len(files)} image files to process")
            
            for file_name in files:
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name.replace('.nrrd', '_ras.nrrd').replace('.mha', '_ras.mha'))
                
                image = sitk.ReadImage(input_path)
                
                if args.transform_type == 'lps_to_ras':
                    lps_direction_matrix = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
                    new_origin = (0, 0, image.GetOrigin()[2])
                    image.SetDirection(lps_direction_matrix)
                    image.SetOrigin(new_origin)
                
                sitk.WriteImage(image, output_path)
                logger.info(f"Converted: {file_name}")
    
    else:
        parser.error("Either --input_file or --input_dir must be provided")