"""
Unit tests for create_surf_from_seg.py

Tests the conversion of segmentation images to surface meshes.
"""
import unittest
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import sys
import os

# Import from modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modules'))
import vtk_functions as vf
import sitk_functions as sf

# Import functions from modules
vtk_marching_cube_multi = vf.vtk_marching_cube_multi
eraseBoundary = sf.eraseBoundary
build_transform_matrix = vf.build_transform_matrix
exportSitk2VTK = vf.exportSitk2VTK
vtkImageResample = vf.vtkImageResample
vtk_marching_cube = vf.vtk_discrete_marching_cube
exportPython2VTK = vf.exportPython2VTK
smooth_polydata = vf.smooth_polydata
decimation = vf.decimation
appendPolyData = vf.appendPolyData
convert_seg_to_surfs = sf.convert_seg_to_surfs

# Import rotate_mesh from global (it's specific to that script)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global'))
from create_surf_from_seg import (
    rotate_mesh,
    parse_label_names,
    parse_mask_labels,
    label_folder_name,
    label_output_path,
    unique_nonzero_labels,
    extract_per_label_surfaces,
)


class TestEraseBoundary(unittest.TestCase):
    """Test the eraseBoundary function."""
    
    def test_erase_boundary_basic(self):
        """Test basic boundary erasing."""
        labels = np.ones((10, 10, 10), dtype=np.int32)
        result = eraseBoundary(labels, 1, 0)
        
        # Check boundary is erased
        self.assertEqual(result[0, 5, 5], 0)
        self.assertEqual(result[-1, 5, 5], 0)
        
        # Check interior is preserved
        self.assertEqual(result[5, 5, 5], 1)
    
    def test_erase_boundary_with_labels(self):
        """Test boundary erasing with different label values."""
        labels = np.ones((15, 15, 15), dtype=np.int32) * 5
        result = eraseBoundary(labels, 2, 0)
        
        # Check that boundary pixels are set to bg_id
        self.assertEqual(result[1, 7, 7], 0)
        self.assertEqual(result[2, 7, 7], 5)


class TestBuildTransformMatrix(unittest.TestCase):
    """Test the build_transform_matrix function."""
    
    def test_basic_transform(self):
        """Test building a basic transform matrix."""
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img.SetOrigin([5, 10, 15])
        img.SetSpacing([2.0, 2.0, 2.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        matrix = build_transform_matrix(img)
        
        # Verify matrix properties
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix[0, 0], 2.0)  # spacing
        self.assertEqual(matrix[0, 3], 5.0)  # origin
        self.assertEqual(matrix[3, 3], 1.0)  # homogeneous coordinate


class TestExportSitk2VTK(unittest.TestCase):
    """Test the exportSitk2VTK function."""
    
    def test_export_segmentation(self):
        """Test exporting a segmentation image."""
        # Create a simple segmentation
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[3:7, 3:7, 3:7] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        
        vtk_img, matrix = exportSitk2VTK(img)
        
        # Verify VTK image
        self.assertIsNotNone(vtk_img)
        self.assertEqual(vtk_img.GetDimensions(), (10, 10, 10))
        
    def test_export_with_metadata(self):
        """Test export preserves metadata."""
        img = sitk.Image([5, 5, 5], sitk.sitkFloat32)
        img.SetOrigin([10, 20, 30])
        img.SetSpacing([2.0, 3.0, 4.0])
        
        vtk_img, _ = exportSitk2VTK(img)
        
        # Spacing should be preserved
        self.assertEqual(vtk_img.GetSpacing(), (2.0, 3.0, 4.0))


class TestVTKMarchingCube(unittest.TestCase):
    """Test the vtk_marching_cube function."""
    
    def test_single_label_extraction(self):
        """Test extracting a single label."""
        # Create segmentation with cube
        arr = np.zeros((15, 15, 15), dtype=np.int32)
        arr[5:10, 5:10, 5:10] = 2
        
        vtk_array = numpy_to_vtk(arr.flatten('F'))
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(15, 15, 15)
        vtk_img.GetPointData().SetScalars(vtk_array)
        vtk_img.SetSpacing(1, 1, 1)
        vtk_img.SetOrigin(0, 0, 0)
        
        mesh = vtk_marching_cube(vtk_img, 0, 2)
        
        # Verify mesh was created
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.GetNumberOfPoints(), 0)
        self.assertGreater(mesh.GetNumberOfCells(), 0)


class TestVTKMarchingCubeMulti(unittest.TestCase):
    """Test the vtk_marching_cube_multi function."""
    
    def test_multi_label_extraction(self):
        """Test extracting multiple labels."""
        # Create segmentation with multiple regions
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[3:8, 3:8, 3:8] = 1
        arr[12:17, 12:17, 12:17] = 2
        
        vtk_array = numpy_to_vtk(arr.flatten('F'))
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(20, 20, 20)
        vtk_img.GetPointData().SetScalars(vtk_array)
        vtk_img.SetSpacing(1, 1, 1)
        vtk_img.SetOrigin(0, 0, 0)
        
        mesh = vtk_marching_cube_multi(vtk_img, 0)
        
        # Verify mesh contains both regions
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.GetNumberOfPoints(), 0)
        self.assertGreater(mesh.GetNumberOfCells(), 0)


class TestRotateMesh(unittest.TestCase):
    """Test the rotate_mesh function."""
    
    def test_rotate_sphere(self):
        """Test rotating a sphere mesh."""
        # Create a sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(5, 0, 0)
        sphere.SetRadius(2.0)
        sphere.Update()
        
        # Create reference image for center
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(10, 10, 10)
        vtk_img.SetSpacing(1, 1, 1)
        vtk_img.SetOrigin(0, 0, 0)
        
        # Rotate mesh
        rotated = rotate_mesh(sphere.GetOutput(), vtk_img, center=[5, 5, 5])
        
        # Verify rotation occurred
        self.assertIsNotNone(rotated)
        self.assertEqual(rotated.GetNumberOfPoints(), sphere.GetOutput().GetNumberOfPoints())
        
        # Points should have different coordinates after rotation
        orig_point = sphere.GetOutput().GetPoint(0)
        rot_point = rotated.GetPoint(0)
        self.assertNotEqual(orig_point, rot_point)


class TestSmoothPolydata(unittest.TestCase):
    """Test the smooth_polydata function."""
    
    def test_smooth_basic(self):
        """Test basic smoothing operation."""
        # Create a cube with rough edges
        cube = vtk.vtkCubeSource()
        cube.Update()
        
        poly = cube.GetOutput()
        initial_points = poly.GetNumberOfPoints()
        
        # Smooth it
        smoothed = smooth_polydata(poly, iteration=20)
        
        # Points should remain the same count
        self.assertIsNotNone(smoothed)
        self.assertEqual(smoothed.GetNumberOfPoints(), initial_points)
    
    def test_smooth_with_options(self):
        """Test smoothing with different options."""
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(3.0)
        sphere.Update()
        
        smoothed = smooth_polydata(
            sphere.GetOutput(), 
            iteration=10, 
            boundary=True, 
            feature=True,
            smoothingFactor=0.5
        )
        
        self.assertIsNotNone(smoothed)
        self.assertGreater(smoothed.GetNumberOfPoints(), 0)


class TestDecimation(unittest.TestCase):
    """Test the decimation function."""
    
    def test_reduce_points(self):
        """Test that decimation reduces point count."""
        # Create dense sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(5.0)
        sphere.SetThetaResolution(64)
        sphere.SetPhiResolution(64)
        sphere.Update()
        
        poly = sphere.GetOutput()
        initial_points = poly.GetNumberOfPoints()
        
        # Decimate by 70%
        decimated = decimation(poly, 0.7)
        
        # Should have fewer points
        self.assertIsNotNone(decimated)
        self.assertLess(decimated.GetNumberOfPoints(), initial_points * 0.5)
    
    def test_decimation_zero_rate(self):
        """Test decimation with zero reduction rate."""
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(3.0)
        sphere.Update()
        
        poly = sphere.GetOutput()
        initial_points = poly.GetNumberOfPoints()
        
        # No decimation
        decimated = decimation(poly, 0.0)
        
        # Points should be roughly the same
        self.assertIsNotNone(decimated)
        self.assertAlmostEqual(decimated.GetNumberOfPoints(), initial_points, delta=5)


class TestAppendPolyData(unittest.TestCase):
    """Test the appendPolyData function."""
    
    def test_append_multiple(self):
        """Test appending multiple polydata objects."""
        # Create three spheres
        sphere1 = vtk.vtkSphereSource()
        sphere1.SetCenter(0, 0, 0)
        sphere1.Update()
        
        sphere2 = vtk.vtkSphereSource()
        sphere2.SetCenter(5, 0, 0)
        sphere2.Update()
        
        sphere3 = vtk.vtkSphereSource()
        sphere3.SetCenter(0, 5, 0)
        sphere3.Update()
        
        poly_list = [
            sphere1.GetOutput(),
            sphere2.GetOutput(),
            sphere3.GetOutput()
        ]
        
        # Append all
        combined = appendPolyData(poly_list)
        
        # Should have combined point count
        total_points = sum(p.GetNumberOfPoints() for p in poly_list)
        self.assertIsNotNone(combined)
        self.assertEqual(combined.GetNumberOfPoints(), total_points)


class TestConvertSegToSurfs(unittest.TestCase):
    """Test the convert_seg_to_surfs function."""
    
    def test_convert_simple_seg(self):
        """Test converting a simple segmentation."""
        # Create simple segmentation
        arr = np.zeros((30, 30, 30), dtype=np.int32)
        arr[10:20, 10:20, 10:20] = 1
        
        seg = sitk.GetImageFromArray(arr)
        seg.SetOrigin([0, 0, 0])
        seg.SetSpacing([1.0, 1.0, 1.0])
        
        # Convert to surface
        poly = convert_seg_to_surfs(seg, new_spacing=[1.0, 1.0, 1.0], target_node_num=500)
        
        # Verify surface was created
        self.assertIsNotNone(poly)
        self.assertGreater(poly.GetNumberOfPoints(), 0)
        self.assertGreater(poly.GetNumberOfCells(), 0)
        
        # Check that it has RegionId array
        self.assertIsNotNone(poly.GetPointData().GetArray('RegionId'))
    
    def test_convert_multi_label_seg(self):
        """Test converting multi-label segmentation."""
        # Create segmentation with two labels
        arr = np.zeros((40, 40, 40), dtype=np.int32)
        arr[8:18, 8:18, 8:18] = 1
        arr[22:32, 22:32, 22:32] = 2
        
        seg = sitk.GetImageFromArray(arr)
        seg.SetOrigin([0, 0, 0])
        seg.SetSpacing([1.0, 1.0, 1.0])
        
        # Convert to surface
        poly = convert_seg_to_surfs(seg, target_node_num=1000)
        
        # Verify both regions are represented
        self.assertIsNotNone(poly)
        self.assertGreater(poly.GetNumberOfPoints(), 100)
        
        # Check RegionId array has multiple values
        region_ids = vtk_to_numpy(poly.GetPointData().GetArray('RegionId'))
        unique_ids = np.unique(region_ids)
        self.assertEqual(len(unique_ids), 2)


class TestVTKImageResample(unittest.TestCase):
    """Test the vtkImageResample function."""
    
    def test_resample_upscale(self):
        """Test upscaling image."""
        img = vtk.vtkImageData()
        img.SetDimensions(10, 10, 10)
        img.SetSpacing(2.0, 2.0, 2.0)
        img.SetOrigin(0, 0, 0)
        img.AllocateScalars(vtk.VTK_INT, 1)
        
        # Resample to finer spacing
        resampled = vtkImageResample(img, [1.0, 1.0, 1.0], 'NN')
        
        self.assertIsNotNone(resampled)
        self.assertEqual(resampled.GetSpacing(), (1.0, 1.0, 1.0))
    
    def test_resample_interpolation_options(self):
        """Test different interpolation methods."""
        img = vtk.vtkImageData()
        img.SetDimensions(10, 10, 10)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0, 0, 0)
        img.AllocateScalars(vtk.VTK_FLOAT, 1)
        
        # Test each interpolation method
        for method in ['NN', 'linear', 'cubic']:
            resampled = vtkImageResample(img, [0.5, 0.5, 0.5], method)
            self.assertIsNotNone(resampled)
    
    def test_resample_invalid_option(self):
        """Test that invalid interpolation raises error."""
        img = vtk.vtkImageData()
        img.SetDimensions(5, 5, 5)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.AllocateScalars(vtk.VTK_FLOAT, 1)
        
        with self.assertRaises(ValueError):
            vtkImageResample(img, [1.0, 1.0, 1.0], 'invalid')


class TestPerLabelSurfaces(unittest.TestCase):
    """Test per-label marching cubes helpers and extraction."""

    def test_parse_label_names(self):
        names = parse_label_names('1=airways,2=label2,3=arteries,4=veins')
        self.assertEqual(names, {1: 'airways', 2: 'label2', 3: 'arteries', 4: 'veins'})
        self.assertEqual(parse_label_names(None), {})
        self.assertEqual(parse_label_names(''), {})
        with self.assertRaises(ValueError):
            parse_label_names('airways')

    def test_parse_mask_labels(self):
        self.assertIsNone(parse_mask_labels(None))
        self.assertEqual(parse_mask_labels('1,2,3'), [1, 2, 3])
        with self.assertRaises(ValueError):
            parse_mask_labels('')

    def test_label_output_path_with_prefix(self):
        names = {1: 'airways', 3: 'arteries'}
        path = label_output_path(
            '/data/airRC_dataset', 'case0', 1, names,
            dir_prefix='surfaces_mc', output_ext='.vtp',
        )
        self.assertEqual(path, '/data/airRC_dataset/surfaces_mc_airways/case0.vtp')
        path2 = label_output_path(
            '/data/airRC_dataset', 'case0', 3, names,
            dir_prefix='surfaces_taubin', output_ext='.vtp',
        )
        self.assertEqual(path2, '/data/airRC_dataset/surfaces_taubin_arteries/case0.vtp')
        self.assertEqual(label_folder_name(2, names), 'label2')

    def test_label_output_path_without_prefix(self):
        path = label_output_path('/tmp/out', 'case0', 1, {1: 'airways'})
        self.assertEqual(path, '/tmp/out/case0_airways.vtp')

    def test_unique_nonzero_labels(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        arr[5:7, 5:7, 5:7] = 3
        seg = sitk.GetImageFromArray(arr)
        self.assertEqual(unique_nonzero_labels(seg), [1, 3])
        self.assertEqual(unique_nonzero_labels(seg, mask_labels=[3]), [3])
        self.assertEqual(unique_nonzero_labels(seg, mask_labels=[2]), [])

    def test_extract_two_label_meshes(self):
        arr = np.zeros((24, 24, 24), dtype=np.int32)
        arr[4:10, 4:10, 4:10] = 1
        arr[14:20, 14:20, 14:20] = 2
        seg = sitk.GetImageFromArray(arr)
        seg.SetSpacing([1.0, 1.0, 1.0])
        seg.SetOrigin([0.0, 0.0, 0.0])

        surfaces = extract_per_label_surfaces(seg)
        self.assertEqual(sorted(surfaces.keys()), [1, 2])
        for label_id, poly in surfaces.items():
            self.assertGreater(poly.GetNumberOfPoints(), 0, msg=f'label {label_id}')
            self.assertGreater(poly.GetNumberOfCells(), 0, msg=f'label {label_id}')

        only_one = extract_per_label_surfaces(seg, mask_labels=[2])
        self.assertEqual(list(only_one.keys()), [2])


if __name__ == '__main__':
    unittest.main()
