# Built on top of code from Martin Pfaller

# !/usr/bin/env python

import os
import vtk

import numpy as np
from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import get_vtk_array_type


class Integration:
    """
    Class to perform integration on slices
    """

    def __init__(self, inp):
        try:
            self.integrator = vtk.vtkIntegrateAttributes()
        except AttributeError:
            raise Exception('vtkIntegrateAttributes is currently only supported by pvpython')

        if not inp.GetOutput().GetNumberOfPoints():
            raise Exception('Empty slice')

        self.integrator.SetInputData(inp.GetOutput())
        self.integrator.Update()

    def evaluate(self, res_name):
        """
        Evaluate integral.
        Distinguishes between scalar integration (e.g. pressure) and normal projection (velocity)
        Optionally divides integral by integrated area
        Args:
            field: pressure, velocity, ...
            res_name: name of array

        Returns:
            Scalar integral
        """
        # type of result
        field = res_name.split('_')[0]

        if field == 'velocity':
            int_name = 'normal_' + res_name
        else:
            int_name = res_name

        # evaluate integral
        integral = v2n(self.integrator.GetOutput().GetPointData().GetArray(int_name))[0]

        # choose if integral should be divided by area
        if field == 'velocity':
            return integral
        else:
            return integral / self.area()

    def area(self):
        """
        Evaluate integrated surface area
        Returns:
        Area
        """
        return v2n(self.integrator.GetOutput().GetCellData().GetArray('Area'))[0]


class ClosestPoints:
    """
    Find closest points within a geometry
    """
    def __init__(self, inp):
        if isinstance(inp, str):
            geo = read_geo(inp)
            inp = geo.GetOutput()
        dataset = vtk.vtkPolyData()
        dataset.SetPoints(inp.GetPoints())

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        self.locator = locator

    def search(self, points, radius=None):
        """
        Get ids of points in geometry closest to input points
        Args:
            points: list of points to be searched
            radius: optional, search radius
        Returns:
            Id list
        """
        ids = []
        for p in points:
            if radius is not None:
                result = vtk.vtkIdList()
                self.locator.FindPointsWithinRadius(radius, p, result)
                ids += [result.GetId(k) for k in range(result.GetNumberOfIds())]
            else:
                ids += [self.locator.FindClosestPoint(p)]
        return ids


def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res


def get_all_arrays(geo):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())

    return point_data, cell_data


def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def write_geo(fname, input):
    """
    Write geometry to file
    Args:
        fname: file name
        input: vtk object
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def read_img(fname):
    """
    Read image from file, chose corresponding vtk reader
    Args:
        fname: vti image

    Returns:
        vtk reader
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vti':
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def write_img(fname, input):
    """
    Write image to file
    Args:
        fname: file name
        input: vtk object
    """
    _, ext = os.path.splitext(fname)
    if ext == '.mha':
        writer = vtk.vtkXMLPolyDataWriter()
        # if input is vtkImageData, convert to vtkPolyData
        if isinstance(input, vtk.vtkImageData):
            input = geo(input)
    elif ext == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    elif ext == '.vtk':
        writer = vtk.vtkDataSetWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def change_vti_vtk(fname):
    """
    Change image file from vti to vtk
    Args:
        fname: file name
    """
    # Read in the VTI file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()

    # Write out the VTK file
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(fname.replace('.vti','.vtk'))
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()


def threshold(inp, t, name):
    """
    Threshold according to cell array
    Args:
        inp: InputConnection
        t: BC_FaceID
        name: name in cell data used for thresholding
    Returns:
        reader, point data
    """
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(inp)
    thresh.SetInputArrayToProcess(0, 0, 0, 1, name)
    thresh.ThresholdBetween(t, t)
    thresh.Update()
    return thresh


def calculator(inp, function, inp_arrays, out_array):
    """
    Function to add vtk calculator
    Args:
        inp: InputConnection
        function: string with function expression
        inp_arrays: list of input point data arrays
        out_array: name of output array
    Returns:
        calc: calculator object
    """
    calc = vtk.vtkArrayCalculator()
    for a in inp_arrays:
        calc.AddVectorArrayName(a)
    calc.SetInputData(inp.GetOutput())
    if hasattr(calc, 'SetAttributeModeToUsePointData'):
        calc.SetAttributeModeToUsePointData()
    else:
        calc.SetAttributeTypeToPointData()
    calc.SetFunction(function)
    calc.SetResultArrayName(out_array)
    calc.Update()
    return calc


def cut_plane(inp, origin, normal):
    """
    Cuts geometry at a plane
    Args:
        inp: InputConnection
        origin: cutting plane origin
        normal: cutting plane normal
    Returns:
        cut: cutter object
    """
    # define cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    # define cutter
    cut = vtk.vtkCutter()
    cut.SetInputData(inp)
    cut.SetCutFunction(plane)
    cut.Update()
    return cut


def get_points_cells_pd(polydata):
    cells = []
    for i in range(polydata.GetNumberOfCells()):
        cell_points = []
        for j in range(polydata.GetCell(i).GetNumberOfPoints()):
            cell_points += [polydata.GetCell(i).GetPointId(j)]
        cells += [cell_points]
    return v2n(polydata.GetPoints().GetData()), np.array(cells)


def get_points_cells(inp):
    cells = []
    for i in range(inp.GetOutput().GetNumberOfCells()):
        cell_points = []
        for j in range(inp.GetOutput().GetCell(i).GetNumberOfPoints()):
            cell_points += [inp.GetOutput().GetCell(i).GetPointId(j)]
        cells += [cell_points]
    return v2n(inp.GetOutput().GetPoints().GetData()), np.array(cells)


def connectivity(inp, origin):
    """
    If there are more than one unconnected geometries, extract the closest one
    Args:
        inp: InputConnection
        origin: region closest to this point will be extracted
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp) #.GetOutput())
    con.SetExtractionModeToClosestPointRegion()
    con.SetClosestPoint(origin[0], origin[1], origin[2])
    con.Update()
    return con


def connectivity_all(inp):
    """
    Color regions according to connectivity
    Args:
        inp: InputConnection
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp)
    con.SetExtractionModeToAllRegions()
    con.ColorRegionsOn()
    con.Update()
    assert con.GetNumberOfExtractedRegions() > 0, 'empty geometry'
    return con


def extract_surface(inp):
    """
    Extract surface from 3D geometry
    Args:
        inp: InputConnection
    Returns:
        extr: vtkExtractSurface object
    """
    extr = vtk.vtkDataSetSurfaceFilter()
    extr.SetInputData(inp)
    extr.Update()
    return extr.GetOutput()


def clean(inp):
    """
    Merge duplicate Points
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(inp)
    # cleaner.SetTolerance(1.0e-3)
    cleaner.PointMergingOn()
    cleaner.Update()
    return cleaner.GetOutput()


def scalar_array(length, name, fill):
    """
    Create vtkIdTypeArray array with given name and constant value
    """
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfValues(length)
    ids.SetName(name)
    ids.Fill(fill)
    return ids


def add_scalars(inp, name, fill):
    """
    Add constant value array to point and cell data
    """
    inp.GetOutput().GetCellData().AddArray(scalar_array(inp.GetOutput().GetNumberOfCells(), name, fill))
    inp.GetOutput().GetPointData().AddArray(scalar_array(inp.GetOutput().GetNumberOfPoints(), name, fill))


def rename(inp, old, new):
    if inp.GetOutput().GetCellData().HasArray(new):
        inp.GetOutput().GetCellData().RemoveArray(new)
    if inp.GetOutput().GetPointData().HasArray(new):
        inp.GetOutput().GetPointData().RemoveArray(new)
    inp.GetOutput().GetCellData().GetArray(old).SetName(new)
    inp.GetOutput().GetPointData().GetArray(old).SetName(new)


def replace(inp, name, array):
    arr = n2v(array)
    arr.SetName(name)
    inp.GetOutput().GetCellData().RemoveArray(name)
    inp.GetOutput().GetCellData().AddArray(arr)


def geo(inp):
    poly = vtk.vtkGeometryFilter()
    poly.SetInputData(inp)
    poly.Update()
    return poly.GetOutput()


def region_grow(geo, seed_points, seed_ids, n_max=99):
    # initialize output arrays
    array_dist = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
    array_ids = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
    array_ids[seed_points] = seed_ids

    # initialize ids
    cids_all = set()
    pids_all = set(seed_points.tolist())
    pids_new = set(seed_points.tolist())

    # surf = extract_surface(geo)
    # pids_surf = set(v2n(surf.GetPointData().GetArray('GlobalNodeID')).tolist())

    # loop until region stops growing or reaches maximum number of iterations
    i = 0
    while len(pids_new) > 0 and i < n_max:
        # count grow iterations
        i += 1

        # update
        pids_old = pids_new

        # print progress
        print_str = 'Iteration ' + str(i)
        print_str += '\tNew points ' + str(len(pids_old)) + '     '
        print_str += '\tTotal points ' + str(len(pids_all))
        print(print_str)

        # grow region one step
        pids_new = grow(geo, array_ids, pids_old, pids_all, cids_all)

        # convert to array
        pids_old_arr = list(pids_old)

        # create point locator with old wave front
        points = vtk.vtkPoints()
        points.Initialize()
        for i_old in pids_old:
            points.InsertNextPoint(geo.GetPoint(i_old))

        dataset = vtk.vtkPolyData()
        dataset.SetPoints(points)

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        # find closest point in new wave front
        for i_new in pids_new:
            array_ids[i_new] = array_ids[pids_old_arr[locator.FindClosestPoint(geo.GetPoint(i_new))]]
            array_dist[i_new] = i

    return array_ids, array_dist + 1


def grow(geo, array, pids_in, pids_all, cids_all):
    # ids of propagating wave-front
    pids_out = set()

    # loop all points in wave-front
    for pi_old in pids_in:
        cids = vtk.vtkIdList()
        geo.GetPointCells(pi_old, cids)

        # get all connected cells in wave-front
        for j in range(cids.GetNumberOfIds()):
            # get cell id
            ci = cids.GetId(j)

            # skip cells that are already in region
            if ci in cids_all:
                continue
            else:
                cids_all.add(ci)

            pids = vtk.vtkIdList()
            geo.GetCellPoints(ci, pids)

            # loop all points in cell
            for k in range(pids.GetNumberOfIds()):
                # get point id
                pi_new = pids.GetId(k)

                # add point only if it's new and doesn't fullfill stopping criterion
                if array[pi_new] == -1 and pi_new not in pids_in:
                    pids_out.add(pi_new)
                    pids_all.add(pi_new)

    return pids_out


def cell_connectivity(geo):
    """
    Extract the point connectivity from vtk and return a dictionary that can be used in meshio
    """
    vtk_to_meshio = {3: 'line', 5: 'triangle', 10: 'tetra'}

    cells = defaultdict(list)
    for i in range(geo.GetNumberOfCells()):
        cell_type_vtk = geo.GetCellType(i)
        if cell_type_vtk in vtk_to_meshio:
            cell_type = vtk_to_meshio[cell_type_vtk]
        else:
            raise ValueError('vtkCellType ' + str(cell_type_vtk) + ' not supported')

        points = geo.GetCell(i).GetPointIds()
        point_ids = []
        for j in range(points.GetNumberOfIds()):
            point_ids += [points.GetId(j)]
        cells[cell_type] += [point_ids]

    for t, c in cells.items():
        cells[t] = np.array(c)

    return cells


def get_location_cells(surface):
    """
    Compute centers of cells and return their surface_locations
    Args:
        vtk polydata, e.g. surface
    Returns:
        np.array with centroid surface_locations
    """
    ecCentroidFilter = vtk.vtkCellCenters()
    ecCentroidFilter.VertexCellsOn()
    ecCentroidFilter.SetInputData(surface)
    ecCentroidFilter.Update()
    ecCentroids = ecCentroidFilter.GetOutput()

    surface_locations = v2n(ecCentroids.GetPoints().GetData())
    return surface_locations


def voi_contain_caps(voi_min, voi_max, caps_locations):
    """
    See if model caps are enclosed in volume
    Args:
        voi_min: min bounding values of volume
        voi_max: max bounding values of volume
    Returns:
        contain: boolean if a cap point was found within volume
    """
    larger = caps_locations > voi_min
    smaller = caps_locations < voi_max

    contain = np.any(np.logical_and(smaller.all(axis=1), larger.all(axis=1)))
    return contain


def calc_caps(polyData):

    # Now extract feature edges
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(polyData)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.Update()
    output = boundaryEdges.GetOutput()

    # get info on points and cells along the cap boundary
    conn = connectivity_all(output)
    data = get_points_cells(conn)#.GetOutput())

    # Get the RegionId array (created by connectivity_all with ColorRegionsOn)
    point_data = conn.GetOutput().GetPointData()
    connects = None
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        if 'Region' in array_name or array_name == 'RegionId':
            connects = v2n(point_data.GetArray(i))
            break
    
    if connects is None:
        raise ValueError("Could not find RegionId array in connectivity output")

    caps_locs = []
    caps_areas = []
    for i in range(connects.max()+1):

        # get the points that belong to the same cap
        locs = data[0][connects == i]
        # calculate the center of the cap
        center = np.mean(locs, axis=0)
        caps_locs.append(center)

        # calculate the area of the cap
        cells = data[1][connects == i]
        area = 0
        for cell in cells:
            # Use original data[0] array since cell indices reference it
            p0 = data[0][cell[0]]
            p1 = data[0][cell[1]]
            p2 = center
            area += np.linalg.norm(np.cross(p1-p0, p2-p0))/2
        caps_areas.append(area)

    return caps_locs, caps_areas


def get_largest_connected_polydata(poly):

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()

    return poly


def get_k_largest_connected_polydata(poly, k):
    """
    Keep the k largest connected components (by point count) as one vtkPolyData.
    """
    if k <= 0:
        return poly
    if k == 1:
        return get_largest_connected_polydata(poly)

    labeler = vtk.vtkPolyDataConnectivityFilter()
    labeler.SetInputData(poly)
    labeler.SetExtractionModeToAllRegions()
    labeler.ColorRegionsOn()
    labeler.Update()
    if labeler.GetNumberOfExtractedRegions() == 0:
        return poly

    labeled = labeler.GetOutput()
    rid_array = labeled.GetPointData().GetArray('RegionId')
    if rid_array is None:
        rid_array = labeled.GetCellData().GetArray('RegionId')
    if rid_array is None:
        raise ValueError("Expected RegionId from vtkPolyDataConnectivityFilter (ColorRegionsOn)")

    ids = v2n(rid_array)
    unique, counts = np.unique(ids, return_counts=True)
    order = np.argsort(-counts)
    take = min(k, len(unique))
    top_regions = [int(unique[order[i]]) for i in range(take)]

    extract = vtk.vtkPolyDataConnectivityFilter()
    extract.SetInputData(poly)
    extract.SetExtractionModeToSpecifiedRegions()
    extract.InitializeSpecifiedRegionList()
    for rid in top_regions:
        extract.AddSpecifiedRegion(rid)
    extract.Update()
    return extract.GetOutput()


def get_seed(cent_fn, centerline_num, point_on_cent):
    """
    Get a location and radius at a point along centerline
    Args:
        cent_fn: file directory for centerline
        centerline_num: starting from 0, which sub centerline do you wish to sample from
        point_on_cent: starting from 0, how far along the sub centerline you wish to sample
    Returns:
        location coords, radius at the specific point
    """

    ## Centerline
    cent = read_geo(cent_fn).GetOutput()
    num_points = cent.GetNumberOfPoints()               # number of points in centerline
    cent_data = collect_arrays(cent.GetPointData())
    c_loc = v2n(cent.GetPoints().GetData())             # point locations as numpy array
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    cent_id = cent_data['CenterlineId']

    try:
        num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
    except:
        num_cent = 1 # in the case of only one centerline

    ip = centerline_num
    count = point_on_cent

    try:
        ids = [i for i in range(num_points) if cent_id[i,ip]==1] # ids of points belonging to centerline ip
    except:
        ids = [i for i in range(num_points)]
    locs = c_loc[ids]
    rads = radii[ids]

    return locs[count], rads[count]


def calc_normal_vectors(vec0):
    """
    Function to calculate two orthonormal vectors
    to a particular direction vec
    """
    vec0 = vec0/np.linalg.norm(vec0)
    vec1 = np.random.randn(3)       # take a random vector
    vec1 -= vec1.dot(vec0) * vec0   # make it orthogonal to k
    vec1 /= np.linalg.norm(vec1)    # normalize it
    vec2 = np.cross(vec0, vec1)     # calculate third vector

    return vec1, vec2


def clean_boundaries(resampled_image_array):
    """
    Function to see which pixels are inside mesh.
    If they are: set as 1, otherwise 0.
    Input: a binary seg array that has been resampled.
    
    TODO: This function is incomplete and needs implementation.
    """
    # TODO: Implement boundary cleaning logic
    # for pixel in resampled_image:
    raise NotImplementedError("clean_boundaries is not yet implemented")


def bound_polydata_by_image(image, poly, threshold):
    """
    Function to cut polydata to be bounded
    by image volume
    """
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()
    b_bound = [b+threshold if (i % 2) ==0 else b-threshold for i, b in enumerate(b_bound)]
    # print("Bounding box: ", b_bound)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def bound_polydata_by_sphere(poly, center, radius):

    sphereSource = vtk.vtkSphere()
    sphereSource.SetCenter(center[0], center[1], center[2])
    sphereSource.SetRadius(radius)

    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(sphereSource)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def exportSitk2VTK(sitkIm, spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    import SimpleITK as sitk
    img = sitk.GetArrayFromImage(sitkIm).transpose(2,1,0)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0.,0.,0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    # imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix


def exportVTK2Sitk(vtkIm):
    """
    This function creates a simple itk image from a vtk image
    Args:
        vtkIm: vtk image
    Returns:
        sitkIm: simple itk image
    """
    import SimpleITK as sitk
    vtkIm = vtkIm.GetOutput()
    vtkIm.GetPointData().GetScalars().SetName('Scalars_')
    vtkArray = v2n(vtkIm.GetPointData().GetScalars())
    vtkArray = np.reshape(vtkArray, vtkIm.GetDimensions(), order='F')
    vtkArray = np.transpose(vtkArray, (2, 1, 0))
    sitkIm = sitk.GetImageFromArray(vtkArray)
    sitkIm.SetSpacing(vtkIm.GetSpacing())
    sitkIm.SetOrigin(vtkIm.GetOrigin())
    return sitkIm


def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix


def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array
    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    vtkArray = n2v(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    # vtkArray = n2v(img.flatten())
    return vtkArray


def points2polydata(xyz):
    """
    Function to convert list of points to polydata
    """
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    # Add points
    for i in range(0, len(xyz)):
        try:
            p = xyz.loc[i].values.tolist()
        except:
            p = xyz[i]

        point_id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)
    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.Modified()

    return polydata


def remove_duplicate_points(centerline):
    """
    Function to remove duplicate points from centerline
    input: centerline as polydata
    output: centerline as polydata
    """
    # Create the tree
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()

    # Get the points
    points = centerline.GetPoints()
    num_points = points.GetNumberOfPoints()

    # Create new points
    new_points = vtk.vtkPoints()
    new_points.SetNumberOfPoints(num_points)

    # Create new polydata with all arr
    new_centerline = vtk.vtkPolyData()
    new_centerline.SetPoints(new_points)
    new_centerline.GetPointData().ShallowCopy(centerline.GetPointData())
    new_centerline.GetCellData().ShallowCopy(centerline.GetCellData())

    # Loop through points
    for i in range(num_points):
        # Get the point
        point = points.GetPoint(i)

        # Find the closest point
        closest_point_id = locator.FindClosestPoint(point)

        # Get the closest point
        closest_point = points.GetPoint(closest_point_id)

        # Check if the points are the same
        if np.array_equal(point, closest_point):
            # If they are the same, add the point to the new polydata
            new_points.InsertPoint(i, point)
        else:
            # If they are not the same, add the closest point to the new polydata
            new_points.InsertPoint(i, closest_point)

    # Return the new polydata
    return new_centerline


def vtk_marching_cube(vtkLabel, bg_id, seg_id):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh


def vtk_marching_cube_multi(vtkLabel, bg_id, smooth=None, rotate=False, center=None):
    """
    Use the VTK marching cube to create isosurfaces for all classes excluding the background
    Args:
        vtkLabel: vtk image containing the label map
        bg_id: id number of background class
        smooth: smoothing iteration (unused, kept for compatibility)
        rotate: whether to rotate mesh (unused, kept for compatibility)
        center: center for rotation (unused, kept for compatibility)
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    ids = np.unique(v2n(vtkLabel.GetPointData().GetScalars()))
    ids = np.delete(ids, np.where(ids == bg_id))

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    for index, i in enumerate(ids):
        contour.SetValue(index, i)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh


def vtk_discrete_marching_cube(vtkLabel, bg_id, seg_id, smooth=None):
    """
    Use the VTK discrete marching cube to create isosurface for a single class
    Args:
        vtkLabel: vtk image containing the label map
        bg_id: id number of background class
        seg_id: id number of segmentation class to extract
        smooth: smoothing iteration (unused, kept for compatibility)
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh


def _get_mesh_laplacian(pos, face):
    """Compute mesh Laplacian (cotangent weights) for a triangle mesh.

    Replicates torch_geometric.utils.get_mesh_laplacian using numpy.
    Returns (edge_index, edge_weight) where edge_index is (2, num_edges)
    with row=source, col=target, and edge_weight has cotangent weights
    plus self-loops with negative degree.
    """
    assert pos.shape[1] == 3 and face.shape[0] == 3
    num_nodes = pos.shape[0]

    def get_cots(left, centre, right):
        left_pos = pos[left]
        central_pos = pos[centre]
        right_pos = pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = np.einsum('ij,ij->i', left_vec, right_vec)
        cross = np.linalg.norm(np.cross(left_vec, right_vec, axis=1), axis=1)
        cot = np.where(np.abs(cross) > 1e-12, dot / cross, 0.0)
        return cot / 2.0

    # Cotangent at each vertex of each face
    cot_021 = get_cots(face[0], face[2], face[1])
    cot_102 = get_cots(face[1], face[0], face[2])
    cot_012 = get_cots(face[0], face[1], face[2])
    cot_weight = np.concatenate([cot_021, cot_102, cot_012])

    # Edges: (0,1), (1,2), (0,2) for each face
    cot_index = np.concatenate([
        face[:2],      # (0,1)
        face[1:],      # (1,2)
        face[::2],     # (0,2)
    ], axis=1)

    # Make undirected: for each (i,j) add (j,i) with same weight
    row_a, col_a = cot_index[0], cot_index[1]
    row_b, col_b = cot_index[1], cot_index[0]
    row = np.concatenate([row_a, row_b])
    col = np.concatenate([col_a, col_b])
    w = np.concatenate([cot_weight, cot_weight])
    # Coalesce duplicate (row,col) by summing weights
    edge_key = row.astype(np.int64) * num_nodes + col
    uniq_keys, inv = np.unique(edge_key, return_inverse=True)
    cot_weight_undir = np.zeros(len(uniq_keys))
    np.add.at(cot_weight_undir, inv, w)
    row = (uniq_keys // num_nodes).astype(np.int64)
    col = (uniq_keys % num_nodes).astype(np.int64)
    cot_weight = cot_weight_undir

    # Diagonal: -sum of cotangent weights per vertex
    cot_deg = np.zeros(num_nodes)
    np.add.at(cot_deg, row, cot_weight)

    # Add self-loops
    self_row = np.arange(num_nodes, dtype=np.int64)
    self_col = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.vstack([
        np.concatenate([row, self_row]),
        np.concatenate([col, self_col]),
    ])
    edge_weight = np.concatenate([cot_weight, -cot_deg])
    return edge_index, edge_weight


def _scatter_mean(src, index, num_nodes):
    """Scatter reduce='mean': for each index j, mean of src[i] where index[i]==j."""
    out = np.zeros((num_nodes, src.shape[1]), dtype=src.dtype)
    counts = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(out, index, src)
    np.add.at(counts, index, 1.0)
    counts = np.maximum(counts, 1e-12)  # avoid div by zero
    return out / counts[:, np.newaxis]


def taubin_smoothing(V, F, it, mu1, mu2):
    """Taubin λ-μ smoothing: alternates smoothing (mu1) and inflation (-mu2) to reduce shrinkage.

    Uses cotangent Laplacian. Equivalent to the torch_geometric/torch_scatter version
    but implemented with numpy only (no torch).

    Parameters
    ----------
    V : np.ndarray
        Vertex positions, shape (n_vertices, 3).
    F : np.ndarray
        Face indices, shape (3, n_faces) - 3 vertex IDs per face.
    it : int
        Number of Taubin iterations.
    mu1 : float
        Smoothing factor (positive).
    mu2 : float
        Inflation factor (positive, applied as -mu2 to counteract shrinkage).

    Returns
    -------
    np.ndarray
        Smoothed vertex positions, shape (n_vertices, 3).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if F.shape[1] == 3 and F.shape[0] != 3:
        F = F.T  # (n_faces, 3) -> (3, n_faces)
    edge_index, edge_weight = _get_mesh_laplacian(V, F)
    row, col = edge_index[0], edge_index[1]
    num_nodes = V.shape[0]

    Vtemp = V.copy()
    for _ in range(it):
        inputs_lap = Vtemp[row] * edge_weight[:, np.newaxis]
        lap = _scatter_mean(inputs_lap, col, num_nodes)
        Vtemp = Vtemp + mu1 * lap

        inputs_lap = Vtemp[row] * edge_weight[:, np.newaxis]
        lap = _scatter_mean(inputs_lap, col, num_nodes)
        Vtemp = Vtemp - mu2 * lap
    return Vtemp


def taubin_smooth_polydata(poly, it=50, mu1=0.5, mu2=0.51):
    """Apply Taubin smoothing to VTK polydata using numpy (no torch).

    Parameters
    ----------
    poly : vtk.vtkPolyData
        Input surface mesh (triangle cells).
    it : int
        Number of Taubin iterations (default 50).
    mu1 : float
        Smoothing factor (default 0.5).
    mu2 : float
        Inflation factor (default 0.51, slightly > mu1 to prevent shrinkage).

    Returns
    -------
    vtk.vtkPolyData
        Smoothed mesh (same connectivity, updated points).
    """
    if poly is None or poly.GetPoints() is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(
            "Input polydata has no points. The mesh may be empty (e.g. from marching cubes "
            "on an empty segmentation) or the file may be corrupted."
        )
    pts, cells = get_points_cells_pd(poly)
    V = np.asarray(pts, dtype=np.float64)
    faces = [c for c in cells if len(c) == 3]
    if not faces:
        return poly
    F = np.array(faces, dtype=np.int64).T  # (3, n_faces)
    V_smooth = taubin_smoothing(V, F, it, mu1, mu2)
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    vtk_arr = n2v(V_smooth.ravel(order='C'))
    vtk_arr.SetNumberOfComponents(3)
    pts = vtk.vtkPoints()
    pts.SetData(vtk_arr)
    out.SetPoints(pts)
    return out


def smooth_polydata(poly, iteration=25, boundary=False, feature=False, smoothingFactor=0.):
    """
    This function smooths a vtk polydata
    Args:
        poly: vtk polydata to smooth
        iteration: number of smoothing iterations
        boundary: boundary smooth bool
        feature: feature edge smoothing bool
        smoothingFactor: smoothing factor (affects pass band)
    Returns:
        smoothed: smoothed vtk polydata
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetPassBand(pow(10., -4. * smoothingFactor))
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed


def decimation(poly, rate):
    """
    Simplifies a VTK PolyData
    Args:
        poly: vtk PolyData
        rate: target rate reduction
    Returns:
        output: decimated vtk PolyData
    """
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.ScalarsAttributeOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOff()
    decimate.Update()
    output = decimate.GetOutput()
    return output


def appendPolyData(poly_list):
    """
    Combine multiple VTK PolyData objects together
    Args:
        poly_list: list of polydata
    Return:
        poly: combined PolyData
    """
    appendFilter = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appendFilter.AddInputData(poly)
    appendFilter.Update()
    out = appendFilter.GetOutput()
    return out


def convertPolyDataToImageData(poly, ref_im):
    """
    Convert the vtk polydata to imagedata
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """
    ref_im.GetPointData().SetScalars(n2v(np.zeros(
           v2n(ref_im.GetPointData().GetScalars()).shape, dtype=np.int32)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    # Convert output to integer type
    output_array = v2n(output.GetPointData().GetScalars()).astype(np.int32)
    output.GetPointData().SetScalars(n2v(output_array))

    return output


def thresholdPolyData(poly, attr, threshold, mode):
    """
    Get the polydata after thresholding based on the input attribute
    Args:
        poly: vtk PolyData to apply threshold
        attr: attribute of the cell/point array
        threshold: (min, max)
        mode: 'cell' or 'point'
    Returns:
        output: resulted vtk PolyData
    """
    surface_thresh = vtk.vtkThreshold()
    surface_thresh.SetInputData(poly)
    surface_thresh.ThresholdBetween(*threshold)
    if mode == 'cell':
        surface_thresh.SetInputArrayToProcess(0, 0, 0,
                                              vtk.vtkDataObject
                                              .FIELD_ASSOCIATION_CELLS, attr)
    else:
        surface_thresh.SetInputArrayToProcess(0, 0, 0,
                                              vtk.vtkDataObject
                                              .FIELD_ASSOCIATION_POINTS, attr)
    surface_thresh.Update()
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(surface_thresh.GetOutput())
    surf_filter.Update()
    return surf_filter.GetOutput()


def vtkImageResample(image, spacing, opt):
    """
    Resamples the vtk image to the given spacing
    Args:
        image: vtk Image data
        spacing: image new spacing
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """
    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(image)
    if opt == 'linear':
        reslicer.SetInterpolationModeToLinear()
    elif opt == 'NN':
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif opt == 'cubic':
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()


def surface_to_image(mesh, image):
    """
    Find the corresponding pixel of the mesh vertices,
    create a new image delineate the surface for testing

    Args:
        mesh: VTK PolyData
        image: VTK ImageData or Sitk Image
    Returns:
        new_image: VTK ImageData or Sitk Image with surface marked
    """
    import SimpleITK as sitk
    mesh_coords = v2n(mesh.GetPoints().GetData())
    if type(image) == vtk.vtkImageData:
        indices = ((mesh_coords - image.GetOrigin())/image.GetSpacing()).astype(int)

        py_im = np.zeros(image.GetDimensions(), dtype=np.int32)
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1

        new_image = vtk.vtkImageData()
        new_image.DeepCopy(image)
        new_image.GetPointData().SetScalars(n2v(py_im.flatten('F')))
    elif type(image) == sitk.Image:
        matrix = build_transform_matrix(image)
        mesh_coords = np.append(mesh_coords, np.ones((len(mesh_coords),1)),axis=1)
        matrix = np.linalg.inv(matrix)
        indices = np.matmul(matrix, mesh_coords.transpose()).transpose().astype(int)
        py_im = sitk.GetArrayFromImage(image).transpose(2,1,0).astype(np.int32)
        py_im.fill(0)  # Initialize with zeros
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1
        new_image = sitk.GetImageFromArray(py_im.transpose(2,1,0))
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(image.GetDirection())
    return new_image


def bound_polydata_by_image_extended(image, poly, threshold=10, name=""):
    """
    Function to cut polydata to be bounded by image volume (extended version with name parameter)
    Args:
        image: vtk ImageData
        poly: vtk PolyData
        threshold: threshold value or list of thresholds for bounding box
        name: name of the case (used for case-specific thresholds)
    Returns:
        output: clipped vtk PolyData
    """
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()

    b_bound = define_bounding_box(b_bound, threshold, name)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def define_bounding_box(bounds, threshold, name):
    """
    Define bounding box for the image
    Args:
        bounds: image bounds
        threshold: threshold value or list
        name: name of the case
    Returns:
        b_bound: adjusted bounding box
    """
    threshold = get_threshold(name) if name else threshold
    if isinstance(threshold, (int, float)):
        b_bound = [b+threshold if (i % 2) == 0 else b-threshold
                   for i, b in enumerate(bounds)]
    else:
        b_bound = [b+threshold[i] if (i % 2) == 0 else b-threshold[i]
                   for i, b in enumerate(bounds)]
    return b_bound


def get_threshold(name):
    """
    Get the threshold for the bounding box based on case name
    Args:
        name: case name
    Returns:
        threshold: list, threshold for the bounding box [x0, x1, y0, y1, z0, z1]
    """
    if '0174_0000' in name:
        threshold = [80, 30, 10, 10, 10, 5]
    elif '0176_0000' in name:
        threshold = [30, 30, 10, 10, 10, 10]
    elif '0188_0001' in name:
        threshold = [10, 10, 10, 10, 10, 10]
    elif 'O150323_2009_aorta' in name:
        threshold = [10, 10, 10, 10, 10, 5]
    elif 'O344211000_2006_aorta' in name:
        threshold = [10, 10, 10, 10, 10, 10]
    else:
        threshold = [10, 10, 10, 10, 10, 10]  # default
    return threshold


def vectors2polydata(vectors):
    """
    Function to convert list of vectors to polydata
    If the vectors don't have start points, they are assumed to start at origin
    The vectors are assumed to be in 3D
    The function uses vtkPolyData and GetPointData and SetVectors to store the vectors
    and cell type vtkVertex
    
    Args:
        vectors: list or np.array of vectors, shape (n, 3) or (n, 6)
    Returns:
        polydata: vtk polydata object
    """
    # Create the points
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    # Add points
    for i in range(0, len(vectors)):
        # Get the vector
        try:
            v = vectors.loc[i].values.tolist()
        except:
            v = vectors[i]

        # Check if the vector has start and end points
        if len(v) == 3:
            # If the vector doesn't have start and end points, assume it starts at origin
            start = [0, 0, 0]
            end = v
        elif len(v) == 6:
            # If the vector has start and end points, get the start and end points
            start = v[:3]
            end = v[3:]
        else:
            raise ValueError("The vectors should have either 3 or 6 elements")

        # Add the start point
        point_id = points.InsertNextPoint(start)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)

        # Add the end point
        # point_id = points.InsertNextPoint(end)
        # vertices.InsertNextCell(1)
        # vertices.InsertCellPoint(point_id)

    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.Modified()

    # Create the vectors
    vectors_vtk = vtk.vtkDoubleArray()
    vectors_vtk.SetNumberOfComponents(3)
    vectors_vtk.SetName("Vectors")
    for i in range(0, len(vectors)):
        # Get the vector
        try:
            v = vectors.loc[i].values.tolist()
        except:
            v = vectors[i]

        # Check if the vector has start and end points
        if len(v) == 3:
            # If the vector doesn't have start and end points, assume it starts at origin
            start = [0, 0, 0]
            end = v
        elif len(v) == 6:
            # If the vector has start and end points, get the start and end points
            start = v[:3]
            end = v[3:]
        else:
            raise ValueError("The vectors should have either 3 or 6 elements")

        # Calculate the vector
        vector = np.array(end) - np.array(start)

        # Change data to list
        vector = vector.tolist()

        # Add the vector to the polydata
        vectors_vtk.InsertNextTuple(vector)

    # Make sure we have same number of vectors as points
    assert vectors_vtk.GetNumberOfTuples() == polydata.GetNumberOfPoints(), "Number of vectors should be the same as the number of points"

    # Set the vectors to the polydata
    polydata.GetPointData().SetVectors(vectors_vtk)

    # Return the polydata
    return polydata


def subdivide_lines(polydata, num_subdivisions=2):
    """
    Create a function to subdivide the line cells
    in a polydata object
    Done by creating a new point in the middle of each line
    These new points are then used to create new lines
    and assemble the new polydata object
    so each cell is divided into num_subdivisions cells
    Args:
        polydata: vtkPolyData
        num_subdivisions: int, number of subdivisions
    Returns:
        polydata: vtkPolyData
    """
    # New polydata object
    new_polydata = vtk.vtkPolyData()

    # Get the number of cells
    num_cells = polydata.GetNumberOfCells()
    c_loc = v2n(polydata.GetPoints().GetData())

    # Initialize a new points array
    new_points = vtk.vtkPoints()
    new_points.DeepCopy(polydata.GetPoints())

    # Initialize a new cells array
    new_cells = vtk.vtkCellArray()

    # Collect arrays
    arrays = collect_arrays(polydata.GetPointData())

    # If contains 'MaximumInscribedSphereRadius' in point data, create new array
    if 'MaximumInscribedSphereRadius' in arrays.keys():
        new_point_data0 = vtk.vtkDoubleArray()
        new_point_data0.DeepCopy(polydata.GetPointData().GetArray('MaximumInscribedSphereRadius'))
        new_point_data0.SetName('MaximumInscribedSphereRadius')

    if 'CenterlineId' in arrays.keys():
        # numpy array Nx10
        new_point_data1 = arrays['CenterlineId']

    if 'BifurcationIdTmp' in arrays.keys():
        # numpy array Nx1
        new_point_data2 = arrays['BifurcationIdTmp']

    # Loop through each cell
    for cell_id in range(num_cells):
        # Get the current cell
        line = polydata.GetCell(cell_id)

        # Get the number of points in the line
        num_points = line.GetNumberOfPoints()

        # Loop through each point in the line
        for i in range(num_points - 1):

            # Get the current point
            point0 = c_loc[line.GetPointId(i)]
            point1 = c_loc[line.GetPointId(i + 1)]

            # Calculate the new point
            new_point = [(point0[0] + point1[0]) / 2,
                         (point0[1] + point1[1]) / 2,
                         (point0[2] + point1[2]) / 2]

            # Add the new point to the points array
            new_point_id = new_points.InsertNextPoint(new_point)

            # Add the new cell to the cells array
            new_line = vtk.vtkLine()
            new_line.GetPointIds().SetId(0, line.GetPointId(i))
            new_line.GetPointIds().SetId(1, new_point_id)
            new_cells.InsertNextCell(new_line)

            # If contains 'MaximumInscribedSphereRadius' in data, add to new point data
            if 'MaximumInscribedSphereRadius' in arrays.keys():
                # Get the index of the old points
                old_point_id0 = line.GetPointId(i)
                old_point_id1 = line.GetPointId(i + 1)
                old_radius0 = arrays['MaximumInscribedSphereRadius'][old_point_id0]
                old_radius1 = arrays['MaximumInscribedSphereRadius'][old_point_id1]

                # Calculate the new radius
                new_radius = (old_radius0 + old_radius1) / 2

                # Add the new radius to the new point data with the right index
                new_point_data0.InsertNextValue(new_radius)

            if 'CenterlineId' in arrays.keys():
                # Get the index of the old point
                old_point_id0 = line.GetPointId(i)
                old_point_id1 = line.GetPointId(i + 1)

                # Add the new centerline id
                new_centerline_id0 = arrays['CenterlineId'][old_point_id0]
                new_centerline_id1 = arrays['CenterlineId'][old_point_id1]

                # Append to numpy array so N+1x10
                # If the centerline id is the same, keep the first id
                if (new_centerline_id0 == new_centerline_id1).all():
                    new_point_data1 = np.append(new_point_data1, np.expand_dims(new_centerline_id0,0), axis=0)
                # If the centerline id is different, add the new centerline id
                else:
                    new_point_data1 = np.append(new_point_data1, np.expand_dims(new_centerline_id1,0), axis=0)

            if 'BifurcationIdTmp' in arrays.keys():
                # Get the index of the old point
                old_point_id0 = line.GetPointId(i)

                # Add the new bifurcation id
                new_bifurcation_id = arrays['BifurcationIdTmp'][old_point_id0]

                # Append to numpy array so N+1x1
                new_point_data2 = np.append(new_point_data2, new_bifurcation_id)

            # If we are at the last point, add the last line
            if i == num_points - 2:
                new_line = vtk.vtkLine()
                new_line.GetPointIds().SetId(0, new_point_id)
                new_line.GetPointIds().SetId(1, line.GetPointId(i + 1))
                new_cells.InsertNextCell(new_line)

    # Update the polydata
    new_polydata.SetPoints(new_points)
    new_polydata.SetLines(new_cells)

    # If contains 'MaximumInscribedSphereRadius' in data, add to new polydata
    if 'MaximumInscribedSphereRadius' in arrays.keys():
        # assert same number of points and radii
        assert (new_point_data0.GetNumberOfTuples() == new_polydata.GetNumberOfPoints()), "Number of radii should be the same as the number of points"
        new_polydata.GetPointData().AddArray(new_point_data0)

    if 'CenterlineId' in arrays.keys():
        new_array1 = n2v(new_point_data1)
        new_array1.SetName('CenterlineId')
        new_polydata.GetPointData().AddArray(new_array1)

    if 'BifurcationIdTmp' in arrays.keys():
        new_array2 = n2v(new_point_data2)
        new_array2.SetName('BifurcationIdTmp')
        new_polydata.GetPointData().AddArray(new_array2)

    return new_polydata


def connectivity_points(polydata, debug=False):
    """
    Function to find the connectivity of points in a polydata
    Args:
        polydata: vtkPolyData
    Returns:
        point_to_cells: dictionary with points as keys and cells
            as values
    """

    # Ensure your polydata contains vtkLine cells
    if polydata.GetNumberOfCells() > 0 and polydata.GetCellType(0) == vtk.VTK_LINE:
        print("PolyData contains vtkLine cells.")

    # Initialize a dictionary to track the connectivity (points to cells)
    point_to_cells = {}

    # Get the number of cells (lines)
    num_cells = polydata.GetNumberOfCells()
    print(f"PolyData contains {num_cells} cells.")
    print(f"PolyData contains {polydata.GetNumberOfPoints()} points.")

    # Loop through each cell (line) to find the points that it connects
    for cell_id in range(num_cells):
        line = polydata.GetCell(cell_id)  # Get the current cell
        cell_points = line.GetPoints()    # Get the points of the current cell

        # Loop through each point in the line
        for i in range(cell_points.GetNumberOfPoints()):
            point_id = line.GetPointId(i)  # Get the point ID

            # Add the cell to the connectivity mapping for this point
            if point_id not in point_to_cells:
                point_to_cells[point_id] = []
            point_to_cells[point_id].append(cell_id)

    # Display the connectivity (which cells share points)
    if debug:
        for point_id, connected_cells in point_to_cells.items():
            print(f"Point {point_id} is used by cells: {connected_cells}")

    # Check which points are not connected to any cells
    non_cell_points = []
    c_loc = v2n(polydata.GetPoints().GetData())
    for i in range(polydata.GetNumberOfPoints()):
        if i not in point_to_cells:
            non_cell_points.append(c_loc[i])
            if debug:
                print(f"Point {i} is not connected to any cells.")
    # Write the points to a file
    # write_geo("./output_debug/non_cell_points.vtp", points2polydata(non_cell_points))

    return point_to_cells, non_cell_points


def reorganize_cells(cent_local, point_to_cells):
    """
    Function to reorganize/combine the cells in a centerline

    One cell per centerline instead of multiple cells per centerline
    with cell type 4 = vtkLine
    """
    # Initialize a new cells array
    new_cells = vtk.vtkCellArray()

    # Loop through each cell in the centerline
    for point_id, connected_cells in point_to_cells.items():
        # Create a new cell
        new_line = vtk.vtkLine()

        # Get the points of the current cell
        cell = cent_local.GetCell(connected_cells[0])
        cell_points = cell.GetPoints()

        # Loop through each point in the line
        for i in range(cell_points.GetNumberOfPoints()):
            point_id = cell.GetPointId(i)  # Get the point ID
            new_line.GetPointIds().SetId(i, point_id)

        # Add the new line to the cells array
        new_cells.InsertNextCell(new_line)

    # Update the polydata
    cent_local.SetLines(new_cells)

    return cent_local

