# Functions to bind SITK functionality

import SimpleITK as sitk
import numpy as np


def read_image(file_dir_image):
    """
    Read image from file
    Args:
        file_dir_image: image directory
    Returns:
        SITK image reader
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_dir_image)
    file_reader.ReadImageInformation()
    return file_reader


def read_image_numpy(file_dir_image):
    """
    Read image from file as numpy array
    Args:
        file_dir_image: image directory
    Returns:
        SITK image reader
        numpy array
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_dir_image)
    file_reader.ReadImageInformation()

    file_img = sitk.ReadImage(file_dir_image)
    file_np_array = sitk.GetArrayFromImage(file_img)

    return file_reader, file_np_array


def create_new(file_reader):
    """
    Create new SITK image with same formating as another
    Args:
        file_reader: reader from another image
    Returns:
        SITK image
    """
    result_img = sitk.Image(file_reader.GetSize(), file_reader.GetPixelID(),
                            file_reader.GetNumberOfComponents())
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())
    return result_img


def create_new_from_numpy(file_reader, np_array):
    """
    Create new SITK image with same formating as another
    And values from an input numpy array
    Args:
        file_reader: reader from another image
        np_array: np array with image values
    Returns:
        SITK image
    """
    result_img = sitk.GetImageFromArray(np_array)
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())

    return result_img


def write_image(image, outputImageFileName):
    """
    Write image to file
    Args:
        SITK image, filename
    Returns:
        image file
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)

    return None


def remove_other_vessels(image, seed):
    """
    Remove all labelled vessels except the one of interest
    Args:
        SITK image, seed point pointing to point in vessel of interest
    Returns:
        binary image file (either 0 or 1)
    """
    ccimage = sitk.ConnectedComponent(image)
    # check number of components in image
    num_components = sitk.GetArrayFromImage(ccimage).max()

    if num_components == 1:
        return image

    # print("Before num comp: " + str(num_components))
    # get label of component containing seed point
    label = ccimage[seed]
    # print("The label is " + str(label))
    if label == 0:
        label = 1
    # now only keep the component with the label
    labelImage = sitk.BinaryThreshold(ccimage, label, label)
    # check number of components in image
    num_components = sitk.GetArrayFromImage(labelImage).max()
    # print("After num comp: " + str(num_components))

    return labelImage


def connected_comp_info(original_seg, print_condition):
    """
    Print info on the component being kept
    """
    removed_seg = sitk.ConnectedComponent(original_seg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(removed_seg, original_seg)
    means = []

    for l in stats.GetLabels():
        if print_condition:
            print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), stats.GetPhysicalSize(l)))
        means.append(stats.GetMean(l))

    return stats.GetLabels(), means, removed_seg


def extract_volume(reader_im, index_extract, size_extract):
    """
    Function to extract a smaller volume from a larger one using sitk
    args:
        reader_im: sitk image reader
        index_extract: the index of the lower corner for extraction
        size_extract: number of voxels to extract in each direction
    return:
        new_img: sitk image volume
    """
    # if it is a reader object, do the following
    if type(reader_im) == sitk.ImageFileReader:
        reader_im.SetExtractIndex(index_extract)
        reader_im.SetExtractSize(size_extract)
        new_img = reader_im.Execute()
    # if it is a sitk image, do the following
    elif type(reader_im) == sitk.Image:
        new_img = sitk.RegionOfInterest(reader_im, size_extract, index_extract)
    else:
        print('Error: reader_im must be a sitk image or reader object')
        return None

    return new_img


def rotate_volume_tangent(sitk_img, tangent, point, return_vecs=False,
                          verbose=False):
    """
    Function to rotate a volume so that the tangent is aligned with the x-axis
    args:
        sitk_img: sitk image volume
        tangent: tangent vector
        point: point to rotate around, np array
    """
    # sitk needs point to be a tuple of floats
    point = tuple([float(i) for i in point])

    # Get the direction of the image
    direction = sitk_img.GetDirection()

    # Get the angle between the tangent and the x-axis
    angle = np.arccos(np.dot(direction[0:3], tangent)/np.linalg.norm(direction[0:3])/np.linalg.norm(tangent))
    # print(f"Angle: {angle*360/2/np.pi} between {direction[0:3]} and {tangent}")

    # If the angle is less than 1 degree, return the image
    if angle < np.pi/180:
        if return_vecs:
            return sitk_img, direction[3:6], direction[6:9], np.eye(3)
        else:
            return sitk_img
    # or if the angle is between 179 and 181 degrees, return the image
    elif angle > np.pi*179/180 and angle < np.pi*181/180:
        if return_vecs:
            return sitk_img, direction[3:6], direction[6:9], np.eye(3)
        else:
            return sitk_img

    # Get the axis of rotation
    axis = np.cross(direction[0:3], tangent)

    # Create the rotation matrix
    rotation = sitk.VersorTransform(axis, angle)

    # Create the affine transformation
    affine = sitk.AffineTransform(3)
    affine.SetCenter(point)
    affine.SetMatrix(rotation.GetMatrix())

    # Apply the transformation
    # If segmentation, use nearest neighbor interpolation
    # check how many unique values there are in the image
    # if there are only 2, then it is a segmentation
    if len(np.unique(sitk.GetArrayFromImage(sitk_img))) == 2:
        sitk_img = sitk.Resample(sitk_img, sitk_img, affine,
                                 sitk.sitkNearestNeighbor, 0.0,
                                 sitk_img.GetPixelID())
    else:
        sitk_img = sitk.Resample(sitk_img, sitk_img, affine, sitk.sitkLinear,
                                 0.0, sitk_img.GetPixelID())

    # return the transformed y and z vectors
    rot_matrix_np = np.array(rotation.GetMatrix()).reshape(3,3)
    x_og_np = np.array(direction[0:3])
    y_og_np = np.array(direction[3:6])
    z_og_np = np.array(direction[6:9])
    x = np.dot(rot_matrix_np, x_og_np)
    y = np.dot(rot_matrix_np, y_og_np)
    z = np.dot(rot_matrix_np, z_og_np)

    # Check rotation matrix
    # R = rot_matrix_np
    # is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3))
    # is_right_handed = np.isclose(np.linalg.det(R), 1)
    # print(f"Orthogonal: {is_orthogonal}, Right-Handed: {is_right_handed}")

    # original_basis = np.eye(3)  # [1,0,0], [0,1,0], [0,0,1]
    # new_basis = np.dot(R, original_basis.T).T  # Transform basis vectors
    # print("New Basis Vectors:", new_basis)

    # # Project a simple point, e.g., [1, 1, 1]
    # point = np.array([1, 1, 1])
    # projected_point = np.dot(point, new_basis.T)
    # print("Projected Point in New Basis:", projected_point)

    # compare the rotated vectors to the original vectors
    if verbose:
        print(f"Original x: {x_og_np}, Rotated x: {x}")
        print(f"Original y: {y_og_np}, Rotated y: {y}")
        print(f"Original z: {z_og_np}, Rotated z: {z}")
        # compare the rotated vectors to the tangent
        print(f"Tangent: {tangent}, Rotated x: {x}")

    if return_vecs:
        return sitk_img, y, z, rot_matrix_np

    return sitk_img


def rotate_volume_x_plane(sitk_img, point, angle, return_vecs=False, verbose=False):
    """
    Function to rotate a volume around the x-axis
    args:
        sitk_img: sitk image volume
        point: point to rotate around, np array
        angle: angle to rotate around x-axis
    """
    # sitk needs point to be a tuple of floats
    point = tuple([float(i) for i in point])

    # Get the direction of the image
    direction = sitk_img.GetDirection()

    # Create the rotation matrix
    rotation = sitk.VersorTransform([1, 0, 0], angle)

    # Create the affine transformation
    affine = sitk.AffineTransform(3)
    affine.SetCenter(point)
    affine.SetMatrix(rotation.GetMatrix())

    # Apply the transformation
    # If segmentation, use nearest neighbor interpolation
    # check how many unique values there are in the image
    # if there are only 2, then it is a segmentation
    if len(np.unique(sitk.GetArrayFromImage(sitk_img))) == 2:
        sitk_img = sitk.Resample(sitk_img, sitk_img, affine,
                                 sitk.sitkNearestNeighbor, 0.0,
                                 sitk_img.GetPixelID())
    else:
        sitk_img = sitk.Resample(sitk_img, sitk_img, affine, sitk.sitkLinear,
                                 0.0, sitk_img.GetPixelID())

    # return the transformed y and z vectors
    rot_matrix_np = np.array(rotation.GetMatrix()).reshape(3,3)
    x_og_np = np.array(direction[0:3])
    y_og_np = np.array(direction[3:6])
    z_og_np = np.array(direction[6:9])
    x = np.dot(rot_matrix_np, x_og_np)
    y = np.dot(rot_matrix_np, y_og_np)
    z = np.dot(rot_matrix_np, z_og_np)

    # Check rotation matrix
    # R = rot_matrix_np
    # is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3))
    # is_right_handed = np.isclose(np.linalg.det(R), 1)
    # print(f"Orthogonal: {is_orthogonal}, Right-Handed: {is_right_handed}")

    # original_basis = np.eye(3)  # [1,0,0], [0,1,0], [0,0,1]
    # new_basis = np.dot(R, original_basis.T).T  # Transform basis vectors
    # print("New Basis Vectors:", new_basis)

    # # Project a simple point, e.g., [1, 1, 1]
    # point = np.array([1, 1, 1])
    # projected_point = np.dot(point, new_basis.T)
    # print("Projected Point in New Basis:", projected_point)

    # compare the rotated vectors to the original vectors
    if verbose:
        print(f"Original x: {x_og_np}, Rotated x: {x}")
        print(f"Original y: {y_og_np}, Rotated y: {y}")
        print(f"Original z: {z_og_np}, Rotated z: {z}")

    if return_vecs:
        return sitk_img, y, z, rot_matrix_np

    return sitk_img


def map_to_image(point, radius, size_volume, origin_im, spacing_im, size_im,
                 prop=1, min_dim=5, fixed_size=None):
    """
    Function to map a point and radius to volume metrics
    args:
        point: point of volume center
        radius: radius at that point
        size_volume: multiple of radius equal the intended
            volume size (ignored when fixed_size is set)
        origin_im: image origin
        spacing_im: image spacing
        prop: proportion of image to be counted for caps contraint
        min_dim: minimum number of voxels in each dimension
        fixed_size: optional [nx, ny, nz] voxel dimensions. If set, extracts
            always the same voxel size (and same physical size for same spacing)
    return:
        size_extract: number of voxels to extract in each dim
        index_extract: index for sitk volume extraction
        voi_min/max: boundaries of volume for caps constraint
        Returns (None, None, None, None) when fixed_size is used and extraction
        would go out of image bounds (caller should skip the sample)
    """
    ratio = 1/2  # how much can be outside volume

    if fixed_size is not None:
        fixed_size = np.array(fixed_size, dtype=np.float64)
        size_extract = np.ceil(fixed_size).astype(np.int64)
        # Center extraction on point (in physical coords -> voxel index)
        center_voxel = (point - origin_im) / spacing_im
        index_extract = np.rint(center_voxel - size_extract / 2.0).astype(np.int64)
        physical_half = (size_extract * spacing_im) / 2.0
        voi_min = point - physical_half * prop
        voi_max = point + physical_half * prop

        # With fixed size, skip if extraction would go out of bounds
        end_bounds = index_extract + size_extract
        if np.any(index_extract < 0) or np.any(end_bounds > size_im):
            return None, None, None, None

        return size_extract.astype(np.float64), index_extract.astype(np.float64), voi_min, voi_max

    size_extract = np.ceil(size_volume*radius/spacing_im)
    # if size_extract is smaller than min_dim, set to min_dim
    size_extract = np.maximum(size_extract, min_dim)

    index_extract = np.rint((point-origin_im - (size_volume/2)*radius)/spacing_im)
    end_bounds = index_extract+size_extract

    voi_min = point - (size_volume/2)*radius*prop
    voi_max = point + (size_volume/2)*radius*prop

    for i, ind in enumerate(np.logical_and(end_bounds > size_im,
                                           (end_bounds - size_im) < ratio * size_extract)):
        if ind:
            # print('\nsub-volume outside global volume, correcting\n')
            size_extract[i] = size_im[i] - index_extract[i]

    for i, ind in enumerate(np.logical_and(index_extract < np.zeros(3),(np.zeros(3)-index_extract) < ratio*size_extract )):
        if ind:
            # print('\nsub-volume outside global volume, correcting\n')
            index_extract[i] = 0

    return size_extract, index_extract, voi_min, voi_max


def import_image(image_dir):
    """
    Function to import image via sitk
    args:
        file_dir_image: image directory
    return:
        reader_img: sitk image volume reader
        origin_im: image origin coordinates
        size_im: image size
        spacing_im: image spacing
    """
    reader_im = read_image(image_dir)
    origin_im = np.array(list(reader_im.GetOrigin()))
    size_im = np.array(list(reader_im.GetSize()))
    spacing_im = np.array(list(reader_im.GetSpacing()))

    return reader_im, origin_im, size_im, spacing_im


def sitk_to_numpy(Image):

    np_array = sitk.GetArrayFromImage(Image)
    return np_array


def numpy_to_sitk(numpy, file_reader=None):

    Image = sitk.GetImageFromArray(numpy)

    if file_reader:

        Image.SetSpacing(file_reader.GetSpacing())
        Image.SetOrigin(file_reader.GetOrigin())
        Image.SetDirection(file_reader.GetDirection())

    return Image


def eraseBoundary(labels, pixels, bg_id):
    """
    Erase anything on the boundary by a specified number of pixels
    Args:
        labels: python nd array
        pixels: number of pixel width to erase
        bg_id: id number of background class
    Returns:
        labels: edited label maps
    """
    x, y, z = labels.shape
    labels[:pixels, :, :] = bg_id
    labels[-pixels:, :, :] = bg_id
    labels[:, :pixels, :] = bg_id
    labels[:, -pixels:, :] = bg_id
    labels[:, :, :pixels] = bg_id
    labels[:, :, -pixels:] = bg_id

    return labels


def convert_seg_to_surfs(seg, new_spacing=[1., 1., 1.], target_node_num=2048, bound=False):
    """
    Convert segmentation to surfaces using marching cubes
    Args:
        seg: SimpleITK Image segmentation
        new_spacing: target spacing for resampling
        target_node_num: target number of nodes for decimation
        bound: whether to bound the polydata by image
    Returns:
        poly: vtk PolyData with surfaces
    """
    import sys
    import os
    from os.path import dirname, join, abspath
    
    # Add modules directory to path
    modules_path = join(dirname(dirname(abspath(__file__))), 'modules')
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)
    
    import vtk_functions as vf
    from vtk.util.numpy_support import numpy_to_vtk
    
    py_seg = sitk.GetArrayFromImage(seg).astype(np.int32)
    py_seg = eraseBoundary(py_seg, 1, 0)
    labels = np.unique(py_seg)
    for i, l in enumerate(labels):
        py_seg[py_seg == l] = i
    seg2 = sitk.GetImageFromArray(py_seg.astype(np.int32))
    seg2.CopyInformation(seg)

    seg_vtk, _ = vf.exportSitk2VTK(seg2)
    seg_vtk = vf.vtkImageResample(seg_vtk, new_spacing, 'NN')
    poly_l = []
    for i, _ in enumerate(labels):
        if i == 0:
            continue
        p = vf.vtk_discrete_marching_cube(seg_vtk, 0, i)
        p = vf.smooth_polydata(p, iteration=50)
        rate = max(0., 1. - float(target_node_num)/float(p.GetNumberOfPoints()))
        p = vf.decimation(p, rate)
        arr = np.ones(p.GetNumberOfPoints())*i
        arr_vtk = numpy_to_vtk(arr)
        arr_vtk.SetName('RegionId')
        p.GetPointData().AddArray(arr_vtk)
        poly_l.append(p)
    poly = vf.appendPolyData(poly_l)
    if bound:
        poly = vf.bound_polydata_by_image(seg_vtk, poly, 1.5)
    return poly
