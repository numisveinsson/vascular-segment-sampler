import os
import SimpleITK as sitk
import vtk


def transform_image(vtk_image_data):
    # Create a transform that rotates 90 degrees around the x-axis
    transform = vtk.vtkTransform()
    transform.RotateX(90)

    # Apply the transform to the vtkImageData
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(vtk_image_data)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Get the transformed output as vtkStructuredGrid
    structured_grid = transform_filter.GetOutput()

    # Convert vtkStructuredGrid back to vtkImageData
    bounds = structured_grid.GetBounds()
    spacing = vtk_image_data.GetSpacing()
    origin = vtk_image_data.GetOrigin()  # Use the original origin
    extent = [0] * 6

    # Calculate new extents based on the transformed bounds and spacing
    for i in range(3):
        extent[2 * i] = 0
        extent[2 * i + 1] = int((bounds[2 * i + 1] - bounds[2 * i]) / spacing[i])

    # Create vtkImageData object for the output
    rotated_image_data = vtk.vtkImageData()
    rotated_image_data.SetSpacing(spacing)
    rotated_image_data.SetExtent(extent)
    rotated_image_data.SetOrigin(origin)  # Use original origin

    # Copy the point data (scalar values, etc.) from the structured grid to the vtkImageData
    rotated_image_data.GetPointData().ShallowCopy(structured_grid.GetPointData())

    return rotated_image_data


# Function to read .vti file using VTK
def load_vti_image(file_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()  # Read the file
    return reader.GetOutput()


# Function to get image properties
def get_image_info_vtk(image_data):
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    dimensions = image_data.GetDimensions()

    # For orientation, we use the direction cosines (identity matrix in most cases)
    mat = image_data.GetDirectionMatrix()
    # print(f"Direction matrix:\n{mat}")

    # Compute bounds
    bounds = [origin[i] + spacing[i] * (dimensions[i] - 1) for i in range(3)]
    for i in range(3):
        if bounds[i] < origin[i]:
            origin[i], bounds[i] = bounds[i], origin

    info = {
        "Origin": origin,
        "Spacing": spacing,
        "Dimensions": dimensions,
        "Bounds": bounds}
    return info


# Load images using SimpleITK
def load_image(image_path):
    return sitk.ReadImage(image_path)


# Get relevant properties
def get_image_info(image):
    # Cpmpute bounds
    bounds = [image.TransformIndexToPhysicalPoint((image.GetSize()[i] - 1, 0, 0))[i] for i in range(3)]

    info = {
        "Origin": image.GetOrigin(),
        "Spacing": image.GetSpacing(),
        "Size": image.GetSize(),
        "Direction": image.GetDirection(),
        "Bounds": bounds
    }
    return info


# Compare image properties
def compare_images(image1, image2):
    images_info = [get_image_info(image) for image in [image1, image2]]

    # Function to compare each property
    def compare_property(prop_name):
        prop_values = [info[prop_name] for info in images_info]
        if all(val == prop_values[0] for val in prop_values):
            return f"{prop_name} is the same across all images: {prop_values[0]}"
        else:
            return f"{prop_name} differs: {prop_values}"

    # Compare all properties
    for prop in ["Origin", "Spacing", "Size", "Direction"]:
        print(compare_property(prop))


# Check RAS coordinate system
def check_coordinate_system(image):
    direction = image.GetDirection()
    if direction == (1, 0, 0, 0, 1, 0, 0, 0, 1):
        return "Image is in RAS coordinate system"
    else:
        return "Image is not in RAS coordinate system"


# Main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare image properties and optionally transform images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two images:
  python compare_imgs.py --image1 /path/to/image1.mha --image2 /path/to/image2.mha
  
  # Transform and save VTI image:
  python compare_imgs.py --vti_file /path/to/image.vti --output_file /path/to/output.vti --transform
  
  # Compare multiple images:
  python compare_imgs.py --image1 /path/to/img1.mha --image2 /path/to/img2.mha --image3 /path/to/img3.mha
        """
    )
    parser.add_argument('--image1',
                       type=str,
                       default=None,
                       help='Path to first image file (for comparison)')
    parser.add_argument('--image2',
                       type=str,
                       default=None,
                       help='Path to second image file (for comparison)')
    parser.add_argument('--image3',
                       type=str,
                       default=None,
                       help='Path to third image file (optional, for comparison)')
    parser.add_argument('--vti_file', '--vti-file',
                       type=str,
                       default=None,
                       help='Path to VTI file to transform')
    parser.add_argument('-o', '--output_file', '--output-file',
                       type=str,
                       default=None,
                       help='Output path for transformed VTI file')
    parser.add_argument('--transform',
                       action='store_true',
                       default=False,
                       help='Apply 90-degree rotation transform to VTI image')
    
    args = parser.parse_args()
    
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    # Transform VTI if requested
    if args.vti_file:
        if not args.output_file:
            parser.error("--output_file is required when using --vti_file")
        
        if not os.path.exists(args.vti_file):
            raise ValueError(f"VTI file not found: {args.vti_file}")
        
        image_data = load_vti_image(args.vti_file)
        
        if args.transform:
            image_data = transform_image(image_data)
            logger.info("Applied rotation transform to image")
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(args.output_file)
        writer.SetInputData(image_data)
        writer.Write()
        logger.info(f"Saved transformed image to {args.output_file}")
        
        # Get and print image properties
        image_info = get_image_info_vtk(image_data)
        for key, value in image_info.items():
            logger.info(f"{key}: {value}")
    
    # Compare images if provided
    images_to_compare = []
    if args.image1:
        images_to_compare.append(load_image(args.image1))
        logger.info(f"Loaded image 1: {args.image1}")
    if args.image2:
        images_to_compare.append(load_image(args.image2))
        logger.info(f"Loaded image 2: {args.image2}")
    if args.image3:
        images_to_compare.append(load_image(args.image3))
        logger.info(f"Loaded image 3: {args.image3}")
    
    if len(images_to_compare) >= 2:
        logger.info("Comparing images...")
        compare_images(images_to_compare[0], images_to_compare[1])
        
        # Check bounds
        for i, img in enumerate(images_to_compare, 1):
            info = get_image_info(img)
            logger.info(f"Image {i} - Origin: {info['Origin']}, Bounds: {info['Bounds']}")
    
    if not args.vti_file and len(images_to_compare) < 2:
        parser.error("Either provide --vti_file with --output_file, or at least --image1 and --image2 for comparison")