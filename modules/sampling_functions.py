import SimpleITK as sitk
import numpy as np
import os
import random
import hashlib
from .vtk_functions import (
    collect_arrays, calc_normal_vectors, get_location_cells, clean_boundaries,
    exportSitk2VTK, bound_polydata_by_image, bound_polydata_by_sphere,
    get_largest_connected_polydata, points2polydata, write_geo, calc_caps,
    subdivide_lines, connectivity_points,
    )
from .sitk_functions import (
    extract_volume, rotate_volume_tangent, remove_other_vessels,
    connected_comp_info, rotate_volume_x_plane)
from vtk.util.numpy_support import vtk_to_numpy as v2n

import time
np.random.seed(0)
random.seed(0)


def print_info_file(global_config, cases, testing_samples, info_file_name):

    if not global_config['TESTING']:
        f = open(global_config['OUT_DIR'] + info_file_name, "w+")
        for key in global_config.keys():
            f.write("\n " + key + " :")
            f.write("\n     " + str(global_config[key]))
        f.write("\n Testing models not included: ")
        for i in range(len(testing_samples)):
            f.write(" \n     Model " + str(i) + " : " + testing_samples[i])
        f.write(f"\n Number of training models: {len(cases)}")
        f.write("\n Training models included: ")
        for i in range(len(cases)):
            f.write(" \n     Model " + str(i) + " : " + cases[i])
        f.close()
    else:
        f = open(global_config['OUT_DIR'] + info_file_name, "w+")
        for key in global_config.keys():
            f.write("\n " + key + " :")
            f.write("\n     " + str(global_config[key]))
        f.write("\n Testing models included: ")
        for i in range(len(testing_samples)):
            f.write(" \n     Model " + str(i) + " : " + testing_samples[i])
        f.close()


def create_directories(output_folder, modality, global_config):

    # create out folder if it doesnt exist
    try:
        os.mkdir(output_folder)
    except Exception as e:
        print(e)

    suffix = global_config.get('OUTPUT_SUFFIX', '')

    if global_config['TESTING']:
        fns = ['_test']
    elif global_config['VALIDATION_PROP'] == 0:
        fns = ['_train']
    else:
        fns = ['_train', '_val']

    if global_config['WRITE_IMG']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+suffix)
            except Exception as e: print(e)
            try:
                os.mkdir(output_folder+modality+fn+suffix+'_masks')
            except Exception as e: print(e)

    if global_config['WRITE_VTK']:
        try:
            os.mkdir(output_folder+'vtk_data'+suffix)
        except Exception as e: print(e)

    if global_config['WRITE_SURFACE']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+suffix+'_masks_surfaces')
            except Exception as e: print(e)
            # try:
            #     os.mkdir(output_folder+modality+fn+'_masks_surfaces_box')
            # except Exception as e: print(e)
    if global_config['WRITE_CENTERLINE']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+suffix+'_masks_centerlines')
            except Exception as e: print(e)
    if global_config['WRITE_OUTLET_STATS']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+suffix+'_img_outlet_detection')
                os.mkdir(output_folder+modality+fn+suffix+'_masks_img_outlet_detection')
            except Exception as e: print(e)


def create_vtk_dir(output_folder, case_name, throwout=0.0, suffix=''):

    vtk_base = output_folder + 'vtk_data' + suffix + '/'
    os.mkdir(vtk_base + 'vtk_' + case_name)
    os.mkdir(vtk_base + 'vtk_mask_' + case_name)
    if throwout:
        os.mkdir(vtk_base + 'vtk_throwout_' + case_name)


def get_cent_ids(num_points, cent_id, ip):

    try:
        # ids of points belonging to centerline ip
        ids = [i for i in range(num_points) if cent_id[i, ip] == 1]
    except Exception as e:
        print(e)
        ids = [i for i in range(num_points)]
    return ids


def get_surf_caps(surface):
    """
    Function to get the locations of the cells
    on the caps of the surface
    """
    surf_data = collect_arrays(surface.GetCellData())       # collect arrays of cell data
    surf_locs = get_location_cells(surface)                 # get locations of cell centroids
    try:
        ids_caps = np.where(surf_data['CapID']!= 0)[0]      # get ids of cells on caps
    except Exception as e:
        print(e)
        ids_caps = np.where(surf_data['BC_FaceID']!= 0)[0]
    cap_locs = surf_locs[ids_caps]                          # get locations of cap cells
    return cap_locs


def get_longest_centerline(cent_ids, c_loc):
    """ Get the longest centerline by computing the total
        accumulated length of the centerline between each point
    Args:
        cent_ids: list of centerline ids
        c_loc: centerline locations
    """
    lengths = []
    for ids in cent_ids:
        locs = c_loc[ids]
        length = 0
        for i in range(len(locs)-1):
            length += np.linalg.norm(locs[i+1] - locs[i])
        lengths.append(length)
    return np.argmax(lengths)


def sort_centerline_by_length(cent_ids, c_loc):
    """ Sort centerline ids by length
        The longest come first
    Args:
        cent_ids: list of centerline ids
        c_loc: centerline locations
    """
    lengths = []
    for ids in cent_ids:
        locs = c_loc[ids]
        length = 0
        for i in range(len(locs)-1):
            length += np.linalg.norm(locs[i+1] - locs[i])
        lengths.append(length)
    return np.argsort(lengths)[::-1]


def flip_radius(cent_ids, radii):
    """ Flip the centerline ids so that the radius is always larger
    Args:
        cent_ids: list of centerline ids
        radii: list of radii
    """
    for i in range(len(cent_ids)):
        if radii[cent_ids[i][0]] < radii[cent_ids[i][-1]]:
            cent_ids[i] = cent_ids[i][::-1]
    return cent_ids


def sort_centerline(centerline, sub_divide=1, debug=False):
    """
    Function to sort the centerline data
    """
    check_duplicates = False
    # if sub_divide > 1:
    #     if debug:
    #         print(f"Subdividing centerline by {sub_divide}")
    #     # subsample the centerline
    #     for i in range(int(np.sqrt(sub_divide))):
    #         centerline = subdivide_lines(centerline)
    #     # write the centerline to file
    #     write_geo('./output_debug/centerline_subdivided.vtp', centerline)

    # point_to_cells, non_connected = connectivity_points(centerline)

    # if non_connected:
    #     # Some points are not connected in any cells
    #     if debug:
    #         print(f"Centerline has {len(non_connected)} points not connected to any cells")
    #     # remove all except the connected points
    #     centerline0 = get_largest_connected_polydata(centerline)
    #     centerline = centerline0
    #     if debug:
    #         print("Centerline has been filtered to only include largest connected component")
    #     point_to_cells, non_connected = connectivity_points(centerline)
    #     write_geo('./output_debug/centerline_connected.vtp', centerline)

    num_points = centerline.GetNumberOfPoints()               # number of points in centerline
    cent_data = collect_arrays(centerline.GetPointData())
    c_loc = v2n(centerline.GetPoints().GetData())             # point locations as numpy array
    try:
        radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    except Exception as e:
        print(e)
        radii = cent_data['f']
    # get cent_ids, a list of lists
    # each list is the ids of the points belonging to a centerline
    try:
        cent_ids = get_point_ids_post_proc(centerline)
        bifurc_id = cent_data['BifurcationIdTmp']
    except Exception as e:
        # print(e)
        # centerline hasnt been processed
        cent_ids = get_point_ids_no_post_proc(centerline)
        bifurc_id = np.zeros(num_points)
        print("Warning: Centerline has not been processed, no known bifurcations")

    # print(f"Number of branches: {len(cent_ids)}")
    # # remove identical centerlines
    # cent_ids_new = []
    # for i in range(len(cent_ids)):
    #     if not cent_ids[i]:
    #         # empty list
    #         continue
    #     if cent_ids[i] not in cent_ids_new:
    #         cent_ids_new.append(cent_ids[i])
    # cent_ids = cent_ids_new
    # print(f"Number of branches after removing duplicates: {len(cent_ids)}")    

    # cent_ids = sort_centerline_ids(cent_ids, point_to_cells, centerline)
    # print("\nCenterline has been sorted\n")

    # check if there are duplicate points
    if check_duplicates:
        if np.unique(c_loc, axis=0).shape[0] != c_loc.shape[0]:
            # remove duplicate points
            print("\nCenterline has duplicate points, removing them\n")
            _, unique_ids = np.unique(c_loc, axis=0, return_index=True)
            # same for cent_ids, but keep same order
            cent_ids_new = []
            for i in range(len(cent_ids)):
                cent_ids_new.append([])
                for j in range(len(cent_ids[i])):
                    if cent_ids[i][j] in unique_ids:
                        cent_ids_new[i].append(cent_ids[i][j])
            cent_ids = cent_ids_new
    else:
        print("Warning: Centerline has not been checked for duplicate points")

    num_cent = len(cent_ids)
    # print(f"Num branches {num_cent}, Num points: {num_points}")

    return num_points, c_loc, radii, cent_ids, bifurc_id, num_cent


def sort_centerline_ids(cent_ids, point_to_cells, centerline_poly, debug=False):
    """
    Function to sort the centerline ids
    by connectivity of the points

    Args:
        cent_ids: list of lists of point ids
        point_to_cells: dictionary with points as keys and cells
            as values
        centerline_poly: vtk polydata of centerline
    Returns:
        cent_ids: list of lists of point ids
    """
    # sort the centerline ids
    # by connectivity of the points
    cent_ids_sorted = []
    for ids in cent_ids:
        if not ids:
            # empty list
            cent_ids_sorted.append([])
            continue
        ids_sorted = []
        if len(ids) == 1:
            ids_sorted.append(ids[0])
        else:
            # find ids at the ends
            end_ids = find_end_ids(ids, point_to_cells)
            if len(end_ids) == 0:
                print("No end points found")
                ids_sorted.append(ids[0])
            else:
                print(f"End points found: {end_ids}")
                ids_sorted.append(end_ids[0])
                # NOTE: we can start from either end

            for i in range(len(ids)-1):
                # find the point that share a cell with the previous point
                # and is not in the list
                cell_ids = point_to_cells[ids_sorted[-1]]
                # loop through the cells of the previous point
                if debug:
                    print(f"Point {ids_sorted[-1]} is connected to cells {cell_ids}")
                for cell_id in cell_ids:
                    # get the cell
                    cell = centerline_poly.GetCell(cell_id)
                    # loop through the points of the cell
                    for j in range(cell.GetNumberOfPoints()):
                        point_id = cell.GetPointId(j)
                        if debug:
                            print(f"Checking point {point_id}")
                        # check if the point is in the list
                        # and if it is not in the sorted list
                        if point_id in ids:
                            if point_id not in ids_sorted:
                                # add the point to the sorted list
                                ids_sorted.append(point_id)
                                break
                            # else:
                            #     print(f"Point {point_id} is already in the list")
            # assert the last ids is the other end id
            if len(end_ids) == 2:
                assert ids_sorted[-1] == end_ids[-1], "Last point should be the other end point"

        if debug:
            print(f"Sorted ids: {ids_sorted}")

        cent_ids_sorted.append(ids_sorted)

    return cent_ids_sorted


def find_end_ids(ids, point_to_cells):
    """
    Function to find the end ids
    based on which points are connected to only one cell
    """
    end_ids = []
    for i in range(len(ids)):
        if ids[i] < len(point_to_cells):
            cell_ids = point_to_cells[ids[i]]
        if len(cell_ids) == 1:
            end_ids.append(ids[i])
    return end_ids


def get_point_ids_post_proc(centerline_poly):

    cent = centerline_poly
    # number of points in centerline
    num_points = cent.GetNumberOfPoints()
    # number of points in centerline
    cent_data = collect_arrays(cent.GetPointData())

    # cell_data = collect_arrays(cent.GetCellData())

    cent_id = cent_data['CenterlineId']
    # number of centerlines (one is assembled of multiple)
    try:
        num_cent = len(cent_id[0])
    except:
        num_cent = 1  # in the case of only one centerline

    point_ids_list = []
    for ip in range(num_cent):
        try:
            # ids of points belonging to centerline ip
            ids = [i for i in range(num_points) if cent_id[i, ip] == 1]
        except:
            ids = [i for i in range(num_points)]
        point_ids_list.append(ids)

    return point_ids_list


def get_point_ids_no_post_proc(centerline_poly):
    """
    For this case, the polydata does not have CenterlineIds,
    so we need to find the centerline ids manually based on the
    connectivity of the points
    Args:
        centerline_poly: vtk polydata of centerline
    Returns:
        point_ids: point ids of centerline (list of lists)
    """
    # the centerline is composed of vtk lines
    # Get the lines from the polydata
    point_ids_list = []
    # Iterate through cells and extract lines
    for i in range(centerline_poly.GetNumberOfCells()):
        cell = centerline_poly.GetCell(i)
        if cell.GetCellType() == 4:
            point_ids = []
            for j in range(cell.GetNumberOfPoints()):
                point_id = cell.GetPointId(j)
                # point = centerline_poly.GetPoint(point_id)
                point_ids.append(point_id)
            point_ids_list.append(point_ids)
        else:
            print(f"Cell {i} is not a line")

    return point_ids_list


def choose_destination(trace_testing, val_prop, img_test, seg_test, img_val,
                       seg_val, img_train, seg_train, ip=None, case_name=None):
    # If tracing test, save to test
    if trace_testing:
        image_out_dir = img_test
        seg_out_dir = seg_test
        val_port = False
    # Else, have a probability to save to validation
    else:
        # Use deterministic hash-based split for reproducibility
        # Combine case_name and ip to create a unique, reproducible hash
        if case_name is not None and ip is not None:
            # Create a deterministic value based on case name and centerline id using MD5
            hash_obj = hashlib.md5(f"{case_name}_{ip}".encode('utf-8'))
            # Convert first 8 bytes to integer and normalize to [0, 1) range
            hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
            rand = (hash_int % 10000) / 10000.0
        else:
            # Fallback to random if case_name not provided (backwards compatibility)
            rand = random.uniform(0, 1)
        # print(" random is " + str(rand))
        if rand < val_prop and ip != 0:
            image_out_dir = img_val
            seg_out_dir = seg_val
            val_port = True  # label to say this sample is validation
        else:
            image_out_dir = img_train
            seg_out_dir = seg_train
            val_port = False
    return image_out_dir, seg_out_dir, val_port


def get_tangent(locs, count):
    """
    Function to calculate the tangent
    """
    if count == 0:
        tangent = locs[count+1] - locs[count]
    elif count == len(locs)-1:
        tangent = locs[count] - locs[count-1]
    else:
        tangent = locs[count+1] - locs[count-1]

    # normalize
    tangent = tangent/np.linalg.norm(tangent)

    return tangent


def calc_samples(count, bifurc, locs, rads, global_config):
    """
    Function to calculate the number of samples
    and their locations and sizes
    """
    if bifurc[count] == 2:
        save_bif = 1
        n_samples = global_config['NUMBER_SAMPLES_BIFURC']
    else:
        save_bif = 0
        n_samples = global_config['NUMBER_SAMPLES']
    # if in the beginning of centerline, have more 
    if count < len(locs)/20:
        n_samples = global_config['NUMBER_SAMPLES_START']

    # Sample size(s) and shift(s)
    sizes = np.random.normal(global_config['MU_SIZE'],
                             global_config['SIGMA_SIZE'], n_samples)
    shifts = np.random.normal(global_config['MU_SHIFT'],
                              global_config['SIGMA_SHIFT'], n_samples)
    # Make first be correct size, Make first be on centerline
    sizes[0], shifts[0] = global_config['MU_SIZE'], global_config['MU_SHIFT']
    # print("sizes are: " + str(sizes))
    # print("shifts are: " + str(shifts))

    # Calculate vectors
    if not global_config['ROTATE_VOLUMES']:
        if count < len(locs)/2:
            vec0 = locs[count+1] - locs[count]
        else:
            vec0 = locs[count] - locs[count-1]
    else:
        # vec0 is x-axis
        vec0 = np.array([1, 0, 0])

    vec1, vec2 = calc_normal_vectors(vec0)

    # Shift centers
    centers = []
    for sample in range(n_samples):
        value = random.uniform(0, 1)
        vector = vec1*value + vec2*(1-value)
        centers.append(locs[count]+shifts[sample]*vector*rads[count])
    # print("Number of centers are " + str(len(centers)))

    return centers, sizes, save_bif, n_samples, vec0


def rotate_volumes(reader_im, reader_seg, tangent, point, visualize=False,
                   outdir=None):
    """
    Function to rotate the volumes
    Inputs are:
        reader_im: sitk image reader
        reader_seg: sitk image reader
        tangent: tangent vector
        point: point to rotate around
    """
    # read in the volumes if reader
    if isinstance(reader_im, sitk.ImageFileReader):
        reader_im = reader_im.Execute()
        reader_seg = reader_seg.Execute()

    # rotate the volumes
    new_img, y, z, rot_matrix = rotate_volume_tangent(reader_im, tangent, point,
                                          return_vecs=True)
    new_seg, y, z, rot_matrix = rotate_volume_tangent(reader_seg, tangent, point,
                                          return_vecs=True)
    origin_im = np.array(list(new_img.GetOrigin()))
    if visualize and outdir:
        # write the rotated volumes and non-rotated volumes
        # sitk.WriteImage(new_img, outdir + '/rotated_image.mha')
        # sitk.WriteImage(new_seg, outdir + '/rotated_seg.mha')
        # sitk.WriteImage(reader_im, outdir + '/original_image.mha')
        # sitk.WriteImage(reader_seg, outdir + '/original_seg.mha')
        # write the tangent and normal vectors as polydata
        tangent_pd = points2polydata([point, point + tangent])
        write_geo(outdir + '/tangent.vtp', tangent_pd)
        normal_pd = points2polydata([point, point + z])
        write_geo(outdir + '/normal.vtp', normal_pd)
        binormal_pd = points2polydata([point, point + y])
        write_geo(outdir + '/binormal.vtp', binormal_pd)
        # create a plane from the tangent and normal vectors
        create_plane_from_vectors(point, tangent, z, outdir=outdir, name='planey')
        create_plane_from_vectors(point, tangent, y, outdir=outdir, name='planez')
        create_plane_from_vectors(point, z, y, outdir=outdir, name='planex')

    return new_img, new_seg, origin_im, y, z, rot_matrix


def create_plane_from_vectors(origin, vec1, vec2, resolution=(10, 10),
                              size=(10, 10),
                              outdir=None, name='plane'):
    """
    Creates a VTK polydata plane from two vectors and a point, and writes it to a VTP file.

    Parameters:
        origin (list or np.ndarray): A point [x, y, z] defining the center of the plane.
        vec1 (list or np.ndarray): A vector defining one axis of the plane.
        vec2 (list or np.ndarray): A vector defining the other axis of the plane.
        resolution (tuple): The resolution (number of subdivisions) of the plane along each vector.
        size (tuple): The size of the plane along each vector.
        output_file (str): The name of the output VTP file.
        name (str): The name of the plane.

    Returns:
        None
    """
    import vtk
    import numpy as np

    # Normalize input vectors
    vec1 = np.array(vec1, dtype=np.float64)
    vec2 = np.array(vec2, dtype=np.float64)
    origin = np.array(origin, dtype=np.float64)

    # Create a vtkPoints object to hold the plane points
    points = vtk.vtkPoints()

    # Create a vtkCellArray to hold the plane's cells
    cells = vtk.vtkCellArray()

    # Define the grid resolution
    res_x, res_y = resolution

    # Generate points for the plane with origin at the center
    for i in range(res_x):
        for j in range(res_y):
            # Calculate the point coordinates
            x = origin + (i - res_x / 2) / res_x * size[0] * vec1 + (j - res_y / 2) / res_y * size[1] * vec2
            # Add the point to the vtkPoints object
            points.InsertNextPoint(x)

    # Generate cells for the plane (quads)
    for i in range(res_x - 1):
        for j in range(res_y - 1):
            # Define the quad points
            p0 = i * res_y + j
            p1 = i * res_y + j + 1
            p2 = (i + 1) * res_y + j + 1
            p3 = (i + 1) * res_y + j
            # Create a vtkQuad object
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, p0)
            quad.GetPointIds().SetId(1, p1)
            quad.GetPointIds().SetId(2, p2)
            quad.GetPointIds().SetId(3, p3)
            # Add the quad to the vtkCellArray
            cells.InsertNextCell(quad)

    # Create the polydata object
    plane_polydata = vtk.vtkPolyData()
    plane_polydata.SetPoints(points)
    plane_polydata.SetPolys(cells)

    # Write to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    output_file = outdir + '/' + name + '.vtp'
    writer.SetFileName(output_file)
    writer.SetInputData(plane_polydata)
    writer.Write()

    print(f"Plane polydata saved to {output_file}")


def extract_subvolumes(reader_im, reader_seg, index_extract, size_extract,
                       origin_im, spacing_im, location, radius, size_r, number,
                       name, O=None, global_img=False,
                       remove_others=True,
                       binarize=True, rotate=False,
                       orig_im=None, orig_seg=None,
                       outdir=None):
    """"
    Function to extract subvolumes
    Both image data and GT segmentation
    Also calculates some statistics on
        the subvolumes of interest
    """

    index_extract = index_extract.astype(int).tolist()
    size_extract = size_extract.astype(int).tolist()

    new_img = extract_volume(reader_im, index_extract, size_extract)
    new_seg = extract_volume(reader_seg, index_extract, size_extract)
    im_np = sitk.GetArrayFromImage(new_img)
    seg_np = sitk.GetArrayFromImage(new_seg)

    # if rotate:
    #     # also extract the original image and segmentation
    #     orig_img = extract_volume(orig_im, index_extract, size_extract)
    #     orig_seg = extract_volume(orig_seg, index_extract, size_extract)
    #     sitk.WriteImage(orig_img, outdir + '/orig_image.mha')
    #     sitk.WriteImage(orig_seg, outdir + '/orig_seg.mha')

    if seg_np.max() > 1:
        # binarize the segmentation
        new_seg_bin = new_seg / seg_np.max()
        new_seg_bin = sitk.Cast(new_seg_bin, sitk.sitkUInt8)
        seg_np = sitk.GetArrayFromImage(new_seg_bin)
    else:
        new_seg_bin = new_seg

    # print("Original Seg")
    # labels, means, _ = connected_comp_info(new_seg, False)
    # print("Seg w removed bodies")
    # labels1, means1 = connected_comp_info(removed_seg)

    seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
    removed_seg_bin = remove_other_vessels(new_seg_bin, seed)
    # labels, means, _ = connected_comp_info(removed_seg, True)
    # mask seg with removed seg (cast mask to UInt8 to avoid MaskImageFilter deprecation warning)
    if binarize:
        removed_seg = removed_seg_bin
    else:
        removed_seg = sitk.Mask(new_seg, sitk.Cast(removed_seg_bin, sitk.sitkUInt8))

    rem_np = sitk.GetArrayFromImage(removed_seg)
    blood_np = im_np[seg_np > 0.1]

    if remove_others:
        ground_truth = rem_np
    else:
        ground_truth = seg_np

    center_volume = (seed)*spacing_im + origin_im
    stats = create_base_stats(number, name, size_r, radius, size_extract,
                              origin_im.tolist(), spacing_im.tolist(),
                              index_extract, center_volume)
    stats = add_image_stats(stats, im_np)

    if not global_img:
        diff_cent = np.linalg.norm(location - center_volume)
        labels, means, _ = connected_comp_info(new_seg, False)
        stats, O = add_local_stats(stats, location, diff_cent, blood_np,
                                   ground_truth, means, removed_seg, im_np, O)
        # mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
        # print("The sample has more than one label: " + case_dict['NAME'] +'_'+ str(N-n_old))

    if remove_others:
        new_seg = removed_seg

    return stats, new_img, new_seg, O


def resample_vol(removed_seg, resample_size): #TODO
    """
    Function to resample the volume

    Still in development
    """
    from modules.pre_process import resample_spacing
    removed_seg1 = resample_spacing(removed_seg, template_size=resample_size, order=1)[0]
    # if min(size_extract)<resample_size[0]:
    removed_seg = clean_boundaries(sitk.GetArrayFromImage(removed_seg))
    return removed_seg


def define_cases(global_config, modality):

    cases_dir = global_config['CASES_DIR']+'_'+modality
    cases_raw = os.listdir(cases_dir)
    cases = [cases_dir+'/'+f for f in cases_raw if 'case.' in f]

    testing_cases_raw = global_config['TEST_CASES']
    testing_samples = ['./cases'+'_'+modality+'/case.' + i + '.yml' for i in testing_cases_raw]

    bad_cases_raw = global_config['BAD_CASES']
    bad_cases = ['./cases'+'_'+modality+'/case.' + i + '.yml' for i in bad_cases_raw]

    for case in bad_cases:
        if case in cases:
            cases.remove(case)

    if global_config['TESTING']:
        cases = [i for i in cases if i in testing_samples]
    else:
        cases = [i for i in cases if i not in testing_samples]

    return cases, testing_samples, bad_cases


def extract_surface(img, surface, center, size):
    """
    Function to cut global surface
    into a local part
    size: radius of sphere to cut
    """
    stats = {}
    vtkimage = exportSitk2VTK(img)
    surface_local_box = bound_polydata_by_image(vtkimage[0], surface, 0)    
    surface_local_sphere = bound_polydata_by_sphere(surface, center, size)
    # surface_local_box = get_largest_connected_polydata(surface_local_box)
    surface_local_sphere = get_largest_connected_polydata(surface_local_sphere)

    outlets, outlet_areas = calc_caps(surface_local_sphere)
    stats['OUTLETS'] = outlets
    stats['NUM_OUTLETS'] = len(outlets)
    stats['OUTLET_AREAS'] = outlet_areas

    return stats, surface_local_box, surface_local_sphere


def extract_centerline(img, centerline):
    """
    Function to cut global centerline
    into a local part
    """
    stats = {}
    vtkimage = exportSitk2VTK(img)
    cent_local = bound_polydata_by_image(vtkimage[0], centerline, 0)
    cent_local = get_largest_connected_polydata(cent_local)
    print("  Keeping largest connected centerline")
    point_to_cells, non_connected = connectivity_points(cent_local)
    return stats, cent_local


def clean_cent_ids(cent_id, num_cent):
    """
    Function to clean centerline ids
    """
    columns_to_delete = []
    for i in range(num_cent):
        if np.sum(cent_id[:, i]) == 0:
            columns_to_delete.append(i)
            num_cent -= 1
    cent_id = np.delete(cent_id, columns_to_delete, axis=1) # delete columns with no centerline

    return cent_id, num_cent


def get_bounds(img):
    """
    Function to get the bounds of the image
    """
    bounds = np.zeros((2,3))
    for i in range(3):
        bounds[0,i] = img.TransformIndexToPhysicalPoint((0,0,0))[i]
        bounds[1,i] = img.TransformIndexToPhysicalPoint((img.GetSize()[0]-1,img.GetSize()[1]-1,img.GetSize()[2]-1))[i]
    return bounds


def transform_to_ref(locs, bounds):
    """
    Function to transform the locations
    to the reference frame
    """
    delta = bounds[1,:] - bounds[0,:] # size of image
    locs = (locs - bounds[0,:])/delta # transform to [0,1]

    return locs


def transform_from_ref(locs, bounds):
    """
    Function to transform the locations
    from the reference frame
    """
    delta = bounds[1,:] - bounds[0,:] # size of image
    locs = locs*delta + bounds[0,:]

    return locs


def discretize_centerline(centerline, img, N = None, sub = None, name = None, outdir = None, num_discr_points=10, suffix=''):
    """
    Function to discretize centerline mesh into points
    with labels
    Input: centerline .vtp
    Output: stats dictionary

    Args:
        centerline: vtk polydata of centerline
        img: sitk image
        N: number of centerline
        sub: number of subcenterline
        name: name of the centerline
        outdir: output directory
        num_discr_points: number of discretized points
    Returns:
        stats: dictionary with statistics
    """
    bounds = get_bounds(img)

    num_points, c_loc, radii, cent_id, bifurc_id, num_cent = sort_centerline(centerline)

    c_loc = transform_to_ref(c_loc, bounds)
    # cent_id, num_cent = clean_cent_ids(cent_id, num_cent)
    total_ids = []
    stats = {}
    if name:
        stats['NAME'] = name
    if N:
        stats['No'] = N
    steps = np.empty((0, 6))
    for ip in range(num_cent):

        ids = cent_id[ip]
        # Remove ids that are already in the total_ids
        ids = [i for i in ids if i not in total_ids]
        total_ids.extend(ids)
        # print(f'Number of points in centerline {ip}: {len(ids)}')

        locs, rads, bifurc = c_loc[ids], radii[ids], bifurc_id[ids]
        num_ids = len(ids)
        if num_ids > num_discr_points:
            ids = np.linspace(0, num_ids-1, num_discr_points).astype(int)
            locs, rads, bifurc = locs[ids], rads[ids], bifurc[ids]
        else:
            num_cent -= 1
            continue
        # create polydata from locs
        if outdir:
            locs_pd = points2polydata(locs)
            vtk_base = outdir + '/vtk_data' + suffix + '/'
            write_geo(vtk_base + 'vtk_' + name[:9] +'/' +str(N)+'_'+str(sub)+'_'+str(ip)+ '.vtp', locs_pd)

        steps = create_steps(steps, locs, rads, bifurc, ip)
    # print(steps[:,-2:])
    # print(f'Number of centerlines: {num_cent}')
    stats['NUM_CENT'] = num_cent
    stats['STEPS'] = steps
    return stats


def get_outlet_stats(stats, img, seg, upsample=False):
    """
    Function to get outlet statistics
    """

    planes_img = get_outside_volume(img)
    planes_seg = get_outside_volume(seg)

    if upsample:
        planes_img = upsample_planes(planes_img, size=200, seg=False)
        planes_seg = upsample_planes(planes_seg, size=200, seg=True)

    centers, widths, pos_example, planes_seg = get_boxes(planes_seg)

    names_add = ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']
    stats_all = []
    for i in range(6):

        name = stats['NAME'] + '_' + names_add[i]
        stats_new = {}
        stats_new['NAME'] = name
        stats_new['CENTER'] = centers[i]
        stats_new['WIDTH'] = widths[i]
        stats_new['SIZE'] = planes_seg[i].shape

        stats_all.append(stats_new)

    return stats_all, planes_img, planes_seg, pos_example


def get_cross_sectional_planes(stats, img, seg, upsample=None):
    """
    Function to get cross sectional planes

    :param stats: dictionary with stats
    :param img: sitk image
    :param seg: sitk segmentation
    :param upsample: bool to upsample the planes
    :param traj: trajectory of centerline
    """
    img = sitk.GetArrayFromImage(img)
    seg = sitk.GetArrayFromImage(seg)

    planes_img = []
    planes_seg = []
    for i in range(3):
        # Get the middle plane
        if i == 0:
            plane_img = img[int(img.shape[0]/2), :, :]
            plane_seg = seg[int(seg.shape[0]/2), :, :]
        elif i == 1:
            plane_img = img[:, int(img.shape[1]/2), :]
            plane_seg = seg[:, int(seg.shape[1]/2), :]
        else:
            plane_img = img[:, :, int(img.shape[2]/2)]
            plane_seg = seg[:, :, int(seg.shape[2]/2)]

        planes_img.append(plane_img)
        planes_seg.append(plane_seg)

    if upsample is not None:
        planes_img = upsample_planes(planes_img, size=upsample, seg=False)
        planes_seg = upsample_planes(planes_seg, size=upsample, seg=True)

    names = ['z', 'y', 'x']
    stats_all = []
    for i in range(3):
        name = stats['NAME'] + '_' + names[i]
        stats_new = {}
        stats_new['NAME'] = name
        stats_new['SIZE'] = planes_seg[i].shape

        stats_all.append(stats_new)

    return stats_all, planes_img, planes_seg


def upsample_planes(planes, size=480, seg=False):
    """
    Function to resample images to size

    Args:
        planes: list of 2d images
        size: size to resample to

    Returns:
        planes_up: list of upsampled images
    """
    import cv2

    if seg:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_CUBIC

    planes_up = []
    for i in range(len(planes)):
        plane = planes[i]
        input_size = plane.shape
        # get the minimum size
        min_size = min(input_size)
        # get the size to upsample to
        size_hor = size; size_vert = size
        # size_hor = int(input_size[0]/min_size*size)
        # size_vert = int(input_size[1]/min_size*size)
        # upsample
        # change to int16
        plane = plane.astype(np.int16)
        plane = cv2.resize(plane, (size_hor, size_vert), interpolation=interp)
        planes_up.append(plane)

    return planes_up


def write_2d_planes(planes, stats_out, image_out_dir,
                    add='_img_outlet_detection'):
    """
    Function to write 2d planes
    Values are normalized to 0-255
    Written out as png
    """
    import cv2
    # add to dir name end
    image_out_dir = image_out_dir[:-1] + '/' + add + '/'

    # make dir if it doesnt exist
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)

    for i in range(len(planes)):
        plane = planes[i]
        plane = cv2.convertScaleAbs(plane, alpha=(255.0/np.amax(plane)))
        # print(f"Shape of plane: {plane.shape}")
        fn_name = image_out_dir + stats_out[i]['NAME']+'.jpg'
        cv2.imwrite(fn_name, plane)


def get_proj_traj(stats,
                  img,
                  seg,
                  global_centerline,
                  trajs,
                  num_trajs,
                  tangent,
                  rot_point,
                  outdir=None,
                  visualize=False,
                  suffix='',
                  write_rotated_centerline=False,
                  img_size=400,
                  n_slices=10,
                  return_rot_matrices=False):
    """
    Function to get the projected trajectory
    of the centerline

    Note: The tangent is the x-axis
          the y_vec is the y-axis
          the z_vec is the z-axis
    params:
        stats: dictionary with stats
        img: sitk image
        global_centerline: vtk polydata of centerline
        trajs: list of lists of trajectories
        outdir: output directory

    returns:
        trajs: list of lists of trajectories
    """
    one_img_per_centerline = False
    one_img_all_centerlines = True

    keep_only_if_intersect = False
    keep_only_half = True

    split_dirs = True

    if return_rot_matrices:
        rot_matrices = {}

    # rotate so x-axis aligns with tangent
    (img_x, seg_x, origin_im, y_vec, z_vec, rot_matrix_x
     ) = rotate_volumes(img, seg, tangent, rot_point,
                        outdir=outdir)
    # import pdb; pdb.set_trace()
    # get angles, evenly distributed from 0-90 degrees
    angles = get_angles(n_slices)

    for angle_number, angle in enumerate(angles):
        # print(f"Angle: {angle}")

        if angle != 0:
            # rotate along x-axis by angle
            img, y, z, rot_matrix_np_angle = rotate_volume_x_plane(img_x, rot_point,
                                                                   angle, return_vecs=True)
            seg, y, z, rot_matrix_np_angle = rotate_volume_x_plane(seg_x, rot_point,
                                                                   angle, return_vecs=True)
            # multiply the rotation matrices so rot_matrix_x happens first
            rot_matrix = np.matmul(rot_matrix_x, rot_matrix_np_angle)
        else:
            img = img_x
            seg = seg_x
            rot_matrix = rot_matrix_x

        (stats_out, planes_img, planes_seg
         ) = get_cross_sectional_planes(
            stats, img, seg,
            upsample=img_size)
        # write cross sectional planes
        write_2d_planes(planes_img[:-1], stats_out,
                        outdir, add='_cross_rot'+suffix)
        write_2d_planes(planes_seg[:-1], stats_out,
                        outdir, add='_cross_rot_seg'+suffix)

        planes_loop = ['z', 'y']  # , 'x']

        tangent = np.array([1, 0, 0])
        y_vec = np.array([0, 1, 0])
        z_vec = np.array([0, 0, 1])

        # print(f"tangent: {tangent}")

        _, c_loc, _, cent_id, _, num_cent = sort_centerline(global_centerline)
        # perform affine transformation (if necessary)
        c_loc_aff = np.dot(c_loc - rot_point, rot_matrix) + rot_point

        # get bounds of image
        bounds = get_bounds(img)
        # define bounds of smaller image, half the size
        bounds_half = np.array([[0.35, 0.35, 0.35], [0.65, 0.65, 0.65]])

        # keep only the points that are in the image
        c_loc_indes = np.all(c_loc >= bounds[0], axis=1) & np.all(c_loc <= bounds[1], axis=1)
        # update cent_id
        cent_id = keep_indices(cent_id, c_loc_indes)

        # write the centerline to file
        if outdir and write_rotated_centerline:
            vtk_base = outdir + '/vtk_data' + suffix + '/'
            # centerline points
            locs_pd = points2polydata(c_loc_aff)
            write_geo(vtk_base + 'vtk_' + stats['NAME'] + '_'
                      + str(angle_number) + '_centerline.vtp', locs_pd)
            # images
            sitk.WriteImage(img, vtk_base + 'vtk_' + stats['NAME'] + '_' + str(angle_number)+ '_image.mha')
            sitk.WriteImage(seg, vtk_base + 'vtk_' + stats['NAME'] + '_' + str(angle_number)+ '_seg.mha')

        # transform to reference frame
        c_loc = transform_to_ref(c_loc_aff, bounds)

        if one_img_per_centerline:
            for ip in range(num_cent):
                if not cent_id[ip]:
                    continue
                ids = cent_id[ip]
                locs = c_loc[ids]
                trackId = ip
                for i, plane in enumerate(planes_loop):
                    sceneId = stats['NAME'] + '_' + str(angle_number) + '_' + plane

                    if return_rot_matrices:
                        rot_matrices[sceneId] = rot_matrix

                    metaId = num_trajs
                    if keep_only_half:
                        if plane == 'z':
                            # keep only points with z in the middle
                            locs = locs[np.logical_and(locs[:,2] > bounds_half[0,2], locs[:,2] < bounds_half[1,2])]
                        elif plane == 'y':
                            locs = locs[np.logical_and(locs[:,1] > bounds_half[0,1], locs[:,1] < bounds_half[1,1])]
                        elif plane == 'x':
                            locs = locs[np.logical_and(locs[:,0] > bounds_half[0,0], locs[:,0] < bounds_half[1,0])]

                    locs_proj = project_points(locs, plane, tangent, y_vec, z_vec)
                    if locs_proj.size == 0:
                        continue
                    # set to length of 20
                    locs_proj = downsample(locs_proj, number_points=20)
                    if keep_only_if_intersect:
                        # Check if the line intersects the plane
                        if plane == 'z':
                            plane_normal = z_vec
                        elif plane == 'y':
                            plane_normal = y_vec
                        elif plane == 'x':
                            plane_normal = tangent
                        if not line_intersects_plane(locs, [0.5, 0.5], plane_normal):
                            continue
                    if visualize:
                        # visualize the projected points
                        visualize_points(locs_proj, plane, planes_img[i], stats['NAME']+'_'+str(angle_number),
                                         ip, outdir, split_dirs=split_dirs)
                        visualize_points(locs_proj, plane, planes_seg[i], stats['NAME']+'_'+str(angle_number),
                                         ip, outdir, split_dirs=split_dirs, seg=True)
                    locs_proj = shift_invert(locs_proj, img_size)
                    assert len(locs_proj) == 20, f"Length of locs_proj is {len(locs_proj)}"
                    for j in range(len(locs_proj)):
                        time = int(j/len(locs_proj)*228)
                        traj = [time, trackId, locs_proj[j][0], locs_proj[j][1], sceneId, metaId]
                        trajs.append(traj)
                    num_trajs += 1

        # One image for all centerlines 
        elif one_img_all_centerlines:
            for i, plane in enumerate(planes_loop):
                locs_proj_accumulated = []
                sceneId = stats['NAME'] + '_' + str(angle_number) + '_' + plane
                num_cent_plotted = 0

                if return_rot_matrices:
                    rot_matrices[sceneId] = rot_matrix

                for ip in range(num_cent):
                    if not cent_id[ip]:
                        continue
                    ids = cent_id[ip]
                    locs = c_loc[ids]

                    if keep_only_half:
                        if plane == 'z':
                            # keep only points with z in the middle
                            locs = locs[np.logical_and(locs[:,2] > bounds_half[0,2], locs[:,2] < bounds_half[1,2])]
                        elif plane == 'y':
                            locs = locs[np.logical_and(locs[:,1] > bounds_half[0,1], locs[:,1] < bounds_half[1,1])]
                        elif plane == 'x':
                            locs = locs[np.logical_and(locs[:,0] > bounds_half[0,0], locs[:,0] < bounds_half[1,0])]
                        # if empty, continue
                        if locs.size == 0:
                            continue
                    if keep_only_if_intersect:
                        # Check if the line intersects the plane
                        if plane == 'z':
                            plane_normal = z_vec
                        elif plane == 'y':
                            plane_normal = y_vec
                        elif plane == 'x':
                            plane_normal = tangent

                        if not (line_intersects_plane(locs, [0.5, 0.5, 0.5], plane_normal)
                                # or line_intersects_plane(locs, [0.45, 0.45, 0.45], plane_normal)
                                # or line_intersects_plane(locs, [0.55, 0.55, 0.55], plane_normal)
                                ):
                            continue
                    num_cent_plotted += 1
                    locs_proj = project_points(locs, plane, tangent, y_vec, z_vec)
                    # if locs_proj is empty, continue
                    if locs_proj.shape[0] < 4:
                        continue
                    # set to length of 20
                    locs_proj = downsample(locs_proj, number_points=20)
                    locs_proj_accumulated.append(locs_proj)

                    locs_proj = shift_invert(locs_proj, img_size)

                    assert len(locs_proj) == 20, f"Length of locs_proj is {len(locs_proj)}"
                    for j in range(len(locs_proj)):
                        # time is % of centerline, max 228, integers
                        time = int(j/len(locs_proj)*228)
                        traj = [time, ip, locs_proj[j][0], locs_proj[j][1], sceneId, num_trajs]
                        trajs.append(traj)
                    num_trajs += 1
                if visualize:
                    # visualize the projected points
                    if locs_proj_accumulated:
                        locs_proj_accumulated = np.concatenate(locs_proj_accumulated, axis=0)
                    visualize_points(locs_proj_accumulated, plane, planes_img[i], stats['NAME']+'_'+str(angle_number),
                                    num_cent_plotted, outdir, split_dirs=split_dirs)
                    visualize_points(locs_proj_accumulated, plane, planes_seg[i], stats['NAME']+'_'+str(angle_number),
                                    num_cent_plotted, outdir, split_dirs=split_dirs, seg=True)

    if return_rot_matrices:
        return trajs, num_trajs, rot_matrices
    else:
        return trajs, num_trajs


def get_deproj_traj(data, rot_matrices, img, rot_point, outdir=None, img_size=400):
    """
    Function to deproject the trajectory

    :param pred_data: dict of dicts; first key is the sceneId,
        second keys are 'last_observed', 'gt_future', 'pred_future', 'gt_goal', 'pred_goal'
    :param rot_matrices: dict of rotation matrices
    :return: past_3d, gt_future_3d, pred_future_3d
    """
    
    past_3d, gt_future_3d, pred_future_3d = [], [], []

    bounds = get_bounds(img)

    for sceneId, traj in data.items():
        rot_matrix = rot_matrices[sceneId]
        # invert the rotation matrix
        rot_matrix = rot_matrix.T
        if 'y' in sceneId:
            plane = 'y'
        elif 'z' in sceneId:
            plane = 'z'
        # import ipdb; ipdb.set_trace()
        locs_proj = np.array(traj['pred_future'])[:,0,:]  # N_trajx12x2 array
        locs_proj = locs_proj.reshape(-1, 2)
        locs_proj = shift_invert(locs_proj, img_size=(1/img_size))
        locs = deproject_points(locs_proj, plane)
        locs = transform_from_ref(locs, bounds)
        locs = np.dot(locs - rot_point, rot_matrix) + rot_point
        pred_future_3d.append(locs)

        locs_proj = np.array(traj['gt_future'])[0,:]  # N_trajx12x2 array
        locs_proj = shift_invert(locs_proj, img_size=(1/img_size))
        locs = deproject_points(locs_proj, plane)
        locs = transform_from_ref(locs, bounds)
        locs = np.dot(locs - rot_point, rot_matrix) + rot_point
        gt_future_3d.append(locs)

        locs_proj = np.array(traj['last_observed'])  # 8x2 array
        locs_proj = shift_invert(locs_proj, img_size=(1/img_size))
        locs = deproject_points(locs_proj, plane)
        locs = transform_from_ref(locs, bounds)
        locs = np.dot(locs - rot_point, rot_matrix) + rot_point
        past_3d.append(locs)

    # convert to numpy arrays
    past_3d = np.array(past_3d)
    gt_future_3d = np.array(gt_future_3d)
    pred_future_3d = np.array(pred_future_3d)

    return past_3d, gt_future_3d, pred_future_3d


def get_angles(n_slices):
    """
    Function to get the angles

    Evenly split 90 degrees into n_slices/2

    return [0] if n_slices=2
    return [0,pi/2] if n_slices=4
    etc
    """
    angles = np.linspace(0, np.pi/2, int(n_slices/2))
    return angles


def downsample(locs, number_points=20):
    """
    Function to make the trajectory a set length

    We use interpolation to get the correct number of points
    """
    locs = interpolate(locs, number_points)

    assert len(locs) == number_points, f"Length of locs is {len(locs)}"

    return locs


def interpolate(locs, number_points):
    """
    Function to interpolate the trajectory
    to number_points
    """
    from scipy.interpolate import interp1d
    x = np.arange(len(locs))
    f = interp1d(x, locs, axis=0)
    xnew = np.linspace(0, len(locs)-1, number_points)
    locs = f(xnew)
    return locs


def shift_invert(locs_proj, img_size=None):
    """
    Function to shift the points in y-dir by 0.5
    and invert the y-dir values
    """
    locs_proj_shifted = locs_proj.copy()
    # locs_proj_shifted[:, 1] = 1 - locs_proj_shifted[:, 1]
    if img_size:
        locs_proj_shifted = locs_proj_shifted * img_size
    return locs_proj_shifted


def line_intersects_plane(line_points, plane_point, plane_normal):
    """
    Checks if a line intersects a plane.
    Parameters:
        line_points (np.ndarray): Nx3 array of ordered points forming the line.
        plane_point (np.ndarray): A point on the plane (1x3).
        plane_normal (np.ndarray): The normal vector to the plane (1x3).

    Returns:
        bool: True if the line intersects the plane, False otherwise.
    """
    # Ensure numpy arrays
    line_points = np.array(line_points)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    # Normalize the plane normal for safety
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Loop through consecutive points in the line
    for i in range(len(line_points) - 1):
        # Points defining the line segment
        p1 = line_points[i]
        p2 = line_points[i + 1]

        # Direction vector of the line segment
        line_dir = p2 - p1

        # Check if the line segment is parallel to the plane
        denom = np.dot(plane_normal, line_dir)
        if np.isclose(denom, 0):
            continue  # Skip parallel segments

        # Compute the parameter t for intersection
        t = np.dot(plane_normal, plane_point - p1) / denom

        # Check if the intersection point lies within the segment
        if 0 <= t <= 1:
            return True  # Intersection found

    # No intersection
    return False


def visualize_points(locs_proj, plane, planes, name, nr, outdir,
                     split_dirs=False, seg=False):
    """
    Function to visualize the projected points on top of the image.
    All points are shown on one image.
    Image is saved as a PNG.

    Parameters:
        locs_proj (np.ndarray): Projected points (Nx2).
        plane (str): Plane name ('x', 'y', or 'z').
        planes (np.ndarray): Image of the plane.
        name (str): Name to use for the saved image.
        outdir (str): Output directory for saving the image.
    """
    import cv2
    # Create the output directory if it doesn't exist
    image_out_dir = os.path.join(outdir, 'img_traj')
    if seg:
        image_out_dir = os.path.join(outdir, 'seg_traj')
    os.makedirs(image_out_dir, exist_ok=True)

    # Create the output directory for grayscale images
    image_out_dir_gray = os.path.join(outdir, 'train')
    if seg:
        image_out_dir_gray = os.path.join(outdir, 'train_seg')
    os.makedirs(image_out_dir_gray, exist_ok=True)

    # Normalize and convert planes to grayscale if necessary
    if len(planes.shape) == 3 and planes.shape[-1] != 1:
        # Assume RGB, convert to grayscale
        planes = cv2.cvtColor(planes, cv2.COLOR_BGR2GRAY)
    planes = planes - np.amin(planes)  # Shift values to be positive
    planes = (planes / max(np.amax(planes), 1)) * 255  # Normalize to [0, 255]
    planes = planes.astype(np.uint8)

    # Create dir for grayscale image
    new_dir_name = name + '_' + plane
    new_image_out_dir_gray = os.path.join(image_out_dir_gray, new_dir_name)
    os.makedirs(new_image_out_dir_gray, exist_ok=True)
    # Save grayscale image as 'reference.jpg' image
    filename = os.path.join(new_image_out_dir_gray, 'reference.jpg')
    cv2.imwrite(filename, planes)

    # Convert grayscale to BGR for visualization
    planes_color = cv2.cvtColor(planes, cv2.COLOR_GRAY2BGR)

    # Get image dimensions
    img_height, img_width = planes_color.shape[:2]

    n_past = 8
    n_total = 20
    counter = 0
    for num, loc in enumerate(locs_proj):
        # Scale to image size
        x = int(loc[0] * img_width)
        y = int(loc[1] * img_height)
        # Clamp the values to be within the image bounds
        x = min(max(x, 0), img_width - 1)
        y = min(max(y, 0), img_height - 1)

        # For every n_total points, make first 8 blue
        if num % n_total == 0:
            counter = 0
        if counter < n_past:
            cv2.circle(planes_color, (x, y), 5, (255, 0, 0), -1)
            counter += 1
        # if last point (n_total-1), make it pink star
        elif num % n_total == n_total-1:
            cv2.drawMarker(planes_color, (x, y), (255, 0, 255), markerType=cv2.MARKER_STAR, markerSize=20)
        else:
            # Draw a red dot at the location
            cv2.circle(planes_color, (x, y), 5, (0, 0, 255), -1)

    # Save the output image
    if not split_dirs:
        output_filename = os.path.join(image_out_dir, f"{name}_{nr}_{plane}.jpg")
        cv2.imwrite(output_filename, planes_color)
        print(f"Image saved to {output_filename}")
    else:
        # make dir for nr
        image_out_dir = os.path.join(image_out_dir, f"{nr}")
        os.makedirs(image_out_dir, exist_ok=True)
        output_filename = os.path.join(image_out_dir, f"{name}_{plane}.jpg")
        cv2.imwrite(output_filename, planes_color)


def project_points(locs, plane, x_vec, y_vec, z_vec):
    """
    Function to project points to a plane.
    If the plane is x, project to yz defined by y_vec and z_vec.
    If the plane is y, project to xz defined by x_vec and z_vec.
    If the plane is z, project to xy defined by x_vec and y_vec.

    Parameters:
        locs (np.ndarray): Array of locations (Nx3).
        plane (str): Plane to project to ('x', 'y', or 'z').
        x_vec (np.ndarray): x-axis vector (1x3).
        y_vec (np.ndarray): y-axis vector (1x3).
        z_vec (np.ndarray): z-axis vector (1x3).

    Returns:
        np.ndarray: List of projected locations.
    """
    # Normalize the input vectors
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)

    # Initialize the projection basis
    if plane == 'x':
        basis = np.stack([y_vec, z_vec], axis=0)
    elif plane == 'y':
        basis = np.stack([x_vec, z_vec], axis=0)
    elif plane == 'z':
        basis = np.stack([x_vec, y_vec], axis=0)
    else:
        raise ValueError("Plane must be one of 'x', 'y', or 'z'.")

    # Project each point onto the specified plane
    locs_proj = np.dot(locs, basis.T)
    # Remove coordinates that have either negative or greater than 1 values
    locs_proj = locs_proj[(locs_proj >= 0).all(axis=1) & (locs_proj <= 1).all(axis=1)]

    return locs_proj


def deproject_points(locs_proj, plane):
    """
    Function to deproject points from a plane.

    Parameters:
        locs_proj (np.ndarray): Array of projected locations (Nx2).
        plane (str): Plane to project from ('x', 'y', or 'z').

    Returns:
        np.ndarray: List of deprojected locations (Nx3).
    """
    # Initialize the deprojection basis
    if plane == 'x':
        basis = np.array([[0, 1, 0], [0, 0, 1]])
    elif plane == 'y':
        basis = np.array([[1, 0, 0], [0, 0, 1]])
    elif plane == 'z':
        basis = np.array([[1, 0, 0], [0, 1, 0]])
    else:
        raise ValueError("Plane must be one of 'x', 'y', or 'z'.")

    # Deproject each point from the specified plane
    locs = np.dot(locs_proj, basis)

    # Add 0.5 to either x, y, or z depending on the plane
    if plane == 'x':
        locs[:, 0] += 0.5
    elif plane == 'y':
        locs[:, 1] += 0.5
    elif plane == 'z':
        locs[:, 2] += 0.5

    return locs



def keep_indices(cent_id, c_loc_indes):
    """
    Function to keep only the indices
    that are in the image
    """
    cent_id_new = []
    for ids in cent_id:
        ids_new = [ids[i] for i in range(len(ids)) if c_loc_indes[ids[i]]]
        cent_id_new.append(ids_new)

    return cent_id_new


def get_boxes(planes):

    centers, widths = [], []
    pos_example = 0
    for i in range(len(planes)):
        centers.append([])
        widths.append([])
        # check if plane is empty
        if np.sum(planes[i]) == 0:
            continue
        else:
            pos_example += 1
            # check how many connected components
            img_plane = sitk.GetImageFromArray(planes[i])
            labels, means, connected_seg = connected_comp_info(img_plane,
                                                               False)
            connected_seg = sitk.GetArrayFromImage(connected_seg)
            # print(f"Max of Connected Seg: {np.amax(connected_seg)}, Min of Connected Seg: {np.amin(connected_seg)}")
            for j, label in enumerate(labels):
                # if the connected component is has N or less pixels, skip
                if np.sum(connected_seg == label) <= 50:
                    connected_seg[connected_seg == label] = 0
                    continue

                # get the pixel in the center of the connected component
                center = np.array(np.where(connected_seg == label)).mean(axis=1).astype(int).tolist()
                # get the width and length of the connected component
                width = np.array(np.where(connected_seg == label)).max(axis=1) - np.array(np.where(connected_seg==label)).min(axis=1)
                # have at least 3 pixels in width and length
                width = np.where(width < 3, 3, width).tolist()
                # append to lists
                centers[i].append(center)
                widths[i].append(width)
            planes[i] = connected_seg

            # print(f"Shape: {connected_seg.shape}, Center: {center}, Width/Length: {width}")

    return centers, widths, pos_example, planes


def get_outside_volume(seg):
    """
    Function to get outside sides of volume
    """
    seg_np = sitk.GetArrayFromImage(seg)
    planes = []
    planes.append(seg_np[0, :, :])
    planes.append(seg_np[-1, :, :])
    planes.append(seg_np[:, 0, :])
    planes.append(seg_np[:, -1, :])
    planes.append(seg_np[:, :, 0])
    planes.append(seg_np[:, :, -1])

    return planes


def get_outside_volume_3(seg):
    """
    Function to get outside sides of volume
    The last three channels
    Make channels last
    """
    seg_np = sitk.GetArrayFromImage(seg)
    planes = []
    planes.append(seg_np[:3,:,:].transpose(1,2,0))
    planes.append(seg_np[-3:,:,:].transpose(1,2,0))
    planes.append(seg_np[:,:3,:].transpose(0,2,1))
    planes.append(seg_np[:,-3:,:].transpose(0,2,1))
    planes.append(seg_np[:,:,:3].transpose(0,1,2))
    planes.append(seg_np[:,:,-3:].transpose(0,1,2))

    return planes


def create_steps(steps, locs, rads, bifurc, ip):
    """
    Function to create a step
    Input:
        steps: np array to add to
        locs: locations of points
        rads: rads of points
        bifurc: bifurcations label of points
        ip: index of centerline
    """
    bifurc = binarize_bifurc(bifurc)
    step_add = np.zeros((len(locs), 6))
    step_add[:, 0:3] = locs
    step_add[:, 3] = rads
    step_add[:, 4] = bifurc
    step_add[-1, 5] = 1
    steps = np.append(steps, step_add, axis=0)

    steps_list = steps.tolist()
    for i in range(len(steps_list)):
        steps_list[i][0] = round(steps_list[i][0], 3)
        steps_list[i][1] = round(steps_list[i][1], 3)
        steps_list[i][2] = round(steps_list[i][2], 3)
        steps_list[i][3] = round(steps_list[i][3], 3)
        steps_list[i][4] = int(steps_list[i][4])
        steps_list[i][5] = int(steps_list[i][5])

    return steps_list


def binarize_bifurc(bifurc):
    "Function to binarize bifurcation labels"
    bifurc[bifurc >= 0] = 1
    bifurc[bifurc < 0] = 0

    return bifurc


def add_local_stats(stats, location, diff_cent, blood_np, ground_truth, means, removed_seg, im_np, O):
    """
    Function to add local stats to the stats dictionary
    """
    stats.update({"DIFF_CENT": diff_cent, "POINT_CENT": location.tolist(),
                  "BLOOD_MEAN": np.mean(blood_np),     "BLOOD_MIN": np.amin(blood_np),
                  "BLOOD_STD": np.std(blood_np),       "BLOOD_MAX": np.amax(blood_np),  
                  "GT_MEAN": np.mean(ground_truth),   "GT_STD": np.std(ground_truth),     
                  "GT_MAX": np.amax(ground_truth),    "GT_MIN": np.amin(ground_truth) 
                  })
    if len(means) != 1:
        larg_np = sitk.GetArrayFromImage(removed_seg)
        rem_np = im_np[larg_np > 0.1]
        stats_rem = {
            "LARGEST_MEAN": np.mean(rem_np),"LARGEST_STD": np.std(rem_np),
            "LARGEST_MAX": np.amax(rem_np), "LARGEST_MIN": np.amin(rem_np)}
        stats.update(stats_rem)
        O += 1
    return stats, O


def add_tangent_stats(stats, vec0, save_bif):
    stats.update({"TANGENTX": (vec0/np.linalg.norm(vec0))[0], "TANGENTY": (vec0/np.linalg.norm(vec0))[1], 
                  "TANGENTZ": (vec0/np.linalg.norm(vec0))[2], "BIFURCATION": save_bif})
    return stats


def add_image_stats(stats, im_np):
    stats.update({
        "IM_MEAN": np.mean(im_np), "IM_MIN": np.amin(im_np), 
        "IM_STD": np.std(im_np),   "IM_MAX": np.amax(im_np),
    })
    return stats


def create_base_stats(N, name, size_r, radius, size_extract, origin_im, spacing_im, index_extract, center_volume):
    stats = {"No": N,
             "NAME": name,
             "SIZE": size_r*radius,
             "RADIUS": radius,
             "RESOLUTION": size_extract,
             "ORIGIN": origin_im,
             "SPACING": spacing_im,
             "INDEX": index_extract,   
             "VOL_CENT": center_volume.tolist(),
             "NUM_VOX": size_extract[0]*size_extract[1]*size_extract[2]}
    return stats


def append_stats(stats, csv_list, csv_list_val, val_port):
    if val_port:
        csv_list_val.append(stats)
    else:
        csv_list.append(stats)
    return csv_list, csv_list_val


def find_next_point(count, locs, rads, bifurc, global_config, on_cent):
    """
    Function to find the next point to move to
    """
    lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[count:], axis=0),
                                                 axis=1), 0, 0))
    move = 1
    count = count+1
    if count == len(locs):
        on_cent = False
        return count, on_cent
    move_distance = global_config['MOVE_DIST']*rads[count]
    if rads[count] >= 0.4:  # Move slower for larger vessels
        move_distance = move_distance * global_config['MOVE_SLOWER_LARGE']
        # print("slowerrrrr")
    if bifurc[count] == 2:  # Move slower for bifurcating vessels
        move_distance = move_distance * global_config['MOVE_SLOWER_BIFURC']
    while lengths[move] < move_distance:
        count = count+1
        move = move+1
        if count == len(locs):
            on_cent = False
            break
    return count, on_cent


def print_model_info(case_name, N, n_old, M, m_old):
    print(case_name)
    print("\n****************** All done for this model! ******************")
    print("****************** " + str(N-n_old) + " extractions! ******************")
    print("****************** " + str(M-m_old) + " throwouts! ****************** \n")


def print_into_info(info_file_name, case_name, N, n_old, M, m_old, K, k_old,
                    out_dir):
    f = open(out_dir + info_file_name, 'a')
    f.write("\n " + case_name)
    f.write("\n " + str([N-n_old, M-m_old, K-k_old]))
    f.write("\n ")
    f.close()


def print_into_info_all_done(info_file_name, N, M, K, O, out_dir,
                             start_time=None):
    f = open(out_dir + info_file_name, 'a')
    f.write("\n *** " + str(N) + " extractions! ***")
    f.write("\n *** " + str(M) + " throwouts! ***")
    f.write("\n *** " + str(K) + " errors in saving! ***")
    f.write("\n *** " + str(O) + " have more than one label! ***")
    if start_time:
        f.write(f"\n *** Time: {(time.time()-start_time)/60} minutes ***")
    f.close()


def print_all_done(info, N, M, K, O, mul_l=None):
    for i in info:
        print(i)
        print(info[i])
        print(" ")

    print("\n**** All done for all models! ****")
    print("**** " + str(N) + " extractions! ****")
    print("**** " + str(M) + " throwouts! ****")
    print("**** " + str(K) + " errors in saving! **** \n")
    print("**** O: " + str(O) + " have more than one label! They are: **** \n")

    if mul_l:
        for i in mul_l:
            print(i)


def write_vtk(new_img, removed_seg, out_dir, case_name, N, n_old, sub, suffix=''):
    # write vtk, if N is a multiple of 10
    # if N-n_old%10 == 0:
    vtk_base = out_dir + 'vtk_data' + suffix + '/'
    sitk.WriteImage(new_img, vtk_base + 'vtk_' + case_name + '/' + str(N-n_old)+'_'+str(sub)+ '.mha')
    if sitk.GetArrayFromImage(removed_seg).max() == 1:
        removed_seg *= 255
    sitk.WriteImage(removed_seg, vtk_base + 'vtk_mask_' + case_name + '/' + str(N-n_old)+'_'+str(sub)+ '.mha')


def write_vtk_throwout(reader_seg, index_extract, size_extract, out_dir,
                       case_name, N, n_old, sub, suffix=''):
    vtk_base = out_dir + 'vtk_data' + suffix + '/'
    new_seg = extract_volume(reader_seg, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())
    sitk.WriteImage(new_seg, vtk_base + 'vtk_throwout_' + case_name +'/'+str(N-n_old)+ '_'+str(sub)+'.mha')


def write_subvolume_img(new_img, removed_seg, image_out_dir, seg_out_dir, case_name, N,
              n_old, sub, binarize=True):
    print(f"Max seg value: {sitk.GetArrayFromImage(removed_seg).max()}")
    sitk.WriteImage(new_img, image_out_dir + case_name + '_' + str(N-n_old) + '_' + str(sub)+'.nii.gz')
    max_seg_value = sitk.GetArrayFromImage(removed_seg).max()
    if max_seg_value != 1 and binarize:
        removed_seg = removed_seg / float(max_seg_value*1.0)
        print(f"Max seg value after scaling: {sitk.GetArrayFromImage(removed_seg).max()}")

    # make image unsigned int, removed_seg is sitk image
    removed_seg = sitk.Cast(removed_seg, sitk.sitkUInt8)
    # assert max_seg_value is 1
    if binarize:
        assert sitk.GetArrayFromImage(removed_seg).max() == 1
    sitk.WriteImage(removed_seg, seg_out_dir + case_name + '_' + str(N-n_old)
                    + '_' + str(sub)+'.nii.gz')


def write_surface(new_surf_box, new_surf_sphere, seg_out_dir, case_name, N,
                  n_old, sub):
    # write_geo(seg_out_dir.replace('masks','masks_surfaces_box') + case_dict['NAME']+'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_box)
    write_geo(seg_out_dir.replace('masks','masks_surfaces') + case_name + '_' + str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_sphere)


def write_centerline(new_cent, seg_out_dir, case_name, N, n_old, sub):
    write_geo(seg_out_dir.replace('masks','masks_centerlines') + case_name + '_' + str(N-n_old)+'_'+str(sub)+ '.vtp', new_cent)
    # pts_pd = points2polydata(stats_surf['OUTLETS'])
    # write_geo(out_dir+'vtk_data/vtk_' + case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '_caps.vtp', pts_pd)


def write_csv(csv_list, csv_list_val, modality, global_config):
    import csv
    suffix = global_config.get('OUTPUT_SUFFIX', '')
    csv_file = "_Sample_stats.csv"
    if global_config['TESTING']:
        csv_file = '_test'+suffix+csv_file
    else:
        csv_file = '_train'+suffix+csv_file

    csv_columns = ["No",            "NAME",         "SIZE",     "RESOLUTION",   "ORIGIN",
                   "SPACING",       "POINT_CENT",   "INDEX",    "SIZE_EXTRACT", "VOL_CENT", 
                   "DIFF_CENT",     "IM_MEAN",      "IM_STD",   "IM_MAX",       "IM_MIN",
                   "BLOOD_MEAN",    "BLOOD_STD",    "BLOOD_MAX", "BLOOD_MIN",    "GT_MEAN", 
                   "GT_STD",        "GT_MAX",       "GT_MIN",   "LARGEST_MEAN", "LARGEST_STD",
                   "LARGEST_MAX",   "LARGEST_MIN",  "RADIUS",   "TANGENTX",     "TANGENTY", 
                   "TANGENTZ",      "BIFURCATION",  "NUM_VOX",  "OUTLETS",      "NUM_OUTLETS",
                   "OUTLET_AREAS"]
    with open(global_config['OUT_DIR']+modality+csv_file, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        # write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        for data in csv_list:
            writer.writerow(data)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+csv_file.replace('train', 'val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if csvfile.tell() == 0:
                writer.writeheader()
            for data in csv_list_val:
                writer.writerow(data)


def write_csv_discrete_cent(csv_discrete_centerline,
                            csv_discrete_centerline_val, modality,
                            global_config):
    import csv
    suffix = global_config.get('OUTPUT_SUFFIX', '')
    csv_file = "_Discrete_Centerline.csv"
    if global_config['TESTING']:
        csv_file = '_test'+suffix+csv_file
    else:
        csv_file = '_train'+suffix+csv_file

    csv_columns = ["No", "NAME", "NUM_CENT", "STEPS"]
    with open(global_config['OUT_DIR']+modality+csv_file, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_discrete_centerline:
            writer.writerow(data)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+csv_file.replace('train', 'val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_discrete_centerline_val:
                writer.writerow(data)


def write_csv_outlet_stats(csv_outlet_stats, csv_outlet_stats_val, modality,
                           global_config):

    import csv
    suffix = global_config.get('OUTPUT_SUFFIX', '')
    csv_file = "_Outlet_Stats.csv"
    if global_config['TESTING']:
        csv_file = '_test'+suffix+csv_file
    else:
        csv_file = '_train'+suffix+csv_file

    csv_columns = ["NAME", "CENTER", "WIDTH", "SIZE"]
    with open(global_config['OUT_DIR']+modality+csv_file, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_outlet_stats:
            writer.writerow(data)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+csv_file.replace('train','val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_outlet_stats_val:
                writer.writerow(data)


def write_pkl_outlet_stats(pkl_outlet_stats, pkl_outlet_stats_val, modality,
                           global_config):
    import pickle
    suffix = global_config.get('OUTPUT_SUFFIX', '')
    pkl_file = "_Outlet_Stats.pkl"
    if global_config['TESTING']:
        pkl_file = '_test'+suffix+pkl_file
    else:
        pkl_file = '_train'+suffix+pkl_file

    with open(global_config['OUT_DIR']+modality+pkl_file, 'wb') as f:
        pickle.dump(pkl_outlet_stats, f)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+pkl_file.replace('train', 'val'), 'wb') as f:
            pickle.dump(pkl_outlet_stats_val, f)


def print_csv_stats(out_dir, global_config, modality):
    import csv
    suffix = global_config.get('OUTPUT_SUFFIX', '')
    csv_file = "_Sample_stats.csv"
    if global_config['TESTING']:
        csv_file = modality + '_test'+suffix+csv_file
    else:
        csv_file = modality + '_train'+suffix+csv_file

    print("Here come some AVG stats for the samples")
    # Calculate avg GT mean and std
    with open(out_dir+csv_file, 'r') as f:
        reader = csv.DictReader(f)
        GT_MEAN = []
        GT_STD = []
        for row in reader:
            try:
                GT_MEAN.append(float(row['GT_MEAN']))
                GT_STD.append(float(row['GT_STD']))
            except:
                print(row)
        print("GT_MEAN: " + str(np.mean(GT_MEAN)))
        print("GT_STD: " + str(np.mean(GT_STD)))
    # Calculate avg IM mean and std
    with open(out_dir+csv_file, 'r') as f:
        reader = csv.DictReader(f)
        IM_MEAN = []
        IM_STD = []
        for row in reader:
            try:
                IM_MEAN.append(float(row['IM_MEAN']))
                IM_STD.append(float(row['IM_STD']))
            except:
                print(row)
        print("IM_MEAN: " + str(np.mean(IM_MEAN)))
        print("IM_STD: " + str(np.mean(IM_STD)))
    # Calculate avg radius and number of voxels
    with open(out_dir+csv_file, 'r') as f:
        reader = csv.DictReader(f)
        RADIUS = []
        NUM_VOX = []
        for row in reader:
            try:
                RADIUS.append(float(row['RADIUS']))
                NUM_VOX.append(float(row['NUM_VOX']))
            except:
                print(row)
        print("RADIUS: " + str(np.mean(RADIUS)))
        print("NUM_VOX: " + str(np.mean(NUM_VOX)))
        print("DIM: " + str(np.mean(NUM_VOX)**(1/3)))