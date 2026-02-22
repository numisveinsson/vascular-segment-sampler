"""
Unit tests for preprocessing/change_img_resample.py

Tests the resampling of images to target sizes or spacings.
"""
import unittest
import tempfile
import os
import numpy as np
import SimpleITK as sitk
import sys

# Add preprocessing directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessing'))
from change_img_resample import resample_image, resample_images_batch


class TestResampleImage(unittest.TestCase):
    """Test the resample_image function."""
    
    def setUp(self):
        """Create test images."""
        # Create a test image with known size and spacing
        arr = np.random.rand(50, 50, 50).astype(np.float32)
        self.test_img = sitk.GetImageFromArray(arr)
        self.test_img.SetOrigin([0, 0, 0])
        self.test_img.SetSpacing([1.0, 1.0, 1.0])
        self.test_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    
    def test_resample_to_target_spacing(self):
        """Test resampling to a target spacing."""
        target_spacing = [2.0, 2.0, 2.0]
        result = resample_image(self.test_img, target_spacing=target_spacing, order=1)
        
        # Check new spacing
        result_spacing = result.GetSpacing()
        self.assertAlmostEqual(result_spacing[0], 2.0, places=4)
        self.assertAlmostEqual(result_spacing[1], 2.0, places=4)
        self.assertAlmostEqual(result_spacing[2], 2.0, places=4)
        
        # Size should be approximately halved
        self.assertLess(result.GetSize()[0], self.test_img.GetSize()[0])
        self.assertLess(result.GetSize()[1], self.test_img.GetSize()[1])
        self.assertLess(result.GetSize()[2], self.test_img.GetSize()[2])
    
    def test_resample_to_target_size(self):
        """Test resampling to a target size."""
        target_size = [25, 25, 25]
        result = resample_image(self.test_img, target_size=target_size, order=1)
        
        # Check new size
        result_size = result.GetSize()
        self.assertEqual(result_size[0], 25)
        self.assertEqual(result_size[1], 25)
        self.assertEqual(result_size[2], 25)
    
    def test_resample_upscaling(self):
        """Test upscaling to finer spacing."""
        target_spacing = [0.5, 0.5, 0.5]
        result = resample_image(self.test_img, target_spacing=target_spacing, order=1)
        
        # Size should be approximately doubled
        self.assertGreater(result.GetSize()[0], self.test_img.GetSize()[0])
        self.assertGreater(result.GetSize()[1], self.test_img.GetSize()[1])
        self.assertGreater(result.GetSize()[2], self.test_img.GetSize()[2])
    
    def test_resample_nearest_neighbor(self):
        """Test resampling with nearest neighbor interpolation."""
        target_size = [30, 30, 30]
        result = resample_image(self.test_img, target_size=target_size, order=0)
        
        self.assertEqual(result.GetSize()[0], 30)
        self.assertIsNotNone(result)
    
    def test_resample_linear_interpolation(self):
        """Test resampling with linear interpolation."""
        target_spacing = [1.5, 1.5, 1.5]
        result = resample_image(self.test_img, target_spacing=target_spacing, order=1)
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.GetSpacing()[0], 1.5, places=4)
    
    def test_resample_bspline_interpolation(self):
        """Test resampling with bspline interpolation."""
        target_spacing = [1.2, 1.2, 1.2]
        result = resample_image(self.test_img, target_spacing=target_spacing, order=2)
        
        self.assertIsNotNone(result)
    
    def test_resample_anisotropic_spacing(self):
        """Test resampling to anisotropic spacing."""
        target_spacing = [0.5, 1.0, 2.0]
        result = resample_image(self.test_img, target_spacing=target_spacing, order=1)
        
        result_spacing = result.GetSpacing()
        self.assertAlmostEqual(result_spacing[0], 0.5, places=4)
        self.assertAlmostEqual(result_spacing[1], 1.0, places=4)
        self.assertAlmostEqual(result_spacing[2], 2.0, places=4)
    
    def test_resample_anisotropic_size(self):
        """Test resampling to anisotropic size."""
        target_size = [100, 50, 25]
        result = resample_image(self.test_img, target_size=target_size, order=1)
        
        self.assertEqual(result.GetSize()[0], 100)
        self.assertEqual(result.GetSize()[1], 50)
        self.assertEqual(result.GetSize()[2], 25)
    
    def test_no_target_raises_error(self):
        """Test that missing both targets raises error."""
        with self.assertRaises(ValueError):
            resample_image(self.test_img)
    
    def test_target_size_takes_precedence(self):
        """Test that target_size takes precedence over target_spacing."""
        target_size = [40, 40, 40]
        target_spacing = [2.0, 2.0, 2.0]
        
        result = resample_image(self.test_img, target_size=target_size, 
                               target_spacing=target_spacing, order=1)
        
        # Should match target_size, not target_spacing
        self.assertEqual(result.GetSize()[0], 40)
    
    def test_origin_preserved(self):
        """Test that image origin is preserved after resampling."""
        self.test_img.SetOrigin([10, 20, 30])
        target_spacing = [2.0, 2.0, 2.0]
        
        result = resample_image(self.test_img, target_spacing=target_spacing, order=1)
        
        # Origin should be preserved
        result_origin = result.GetOrigin()
        self.assertAlmostEqual(result_origin[0], 10.0, places=4)
        self.assertAlmostEqual(result_origin[1], 20.0, places=4)
        self.assertAlmostEqual(result_origin[2], 30.0, places=4)
    
    def test_direction_preserved(self):
        """Test that image direction is preserved after resampling."""
        # Set non-identity direction
        direction = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        self.test_img.SetDirection(direction)
        
        target_spacing = [2.0, 2.0, 2.0]
        result = resample_image(self.test_img, target_spacing=target_spacing, order=1)
        
        # Direction should be preserved
        result_direction = result.GetDirection()
        for i, val in enumerate(direction):
            self.assertAlmostEqual(result_direction[i], val, places=4)


class TestResampleImagesBatch(unittest.TestCase):
    """Test the resample_images_batch function."""
    
    def setUp(self):
        """Set up temporary directories and test images."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir)
        
        # Create test images
        for i in range(3):
            arr = np.random.rand(20, 20, 20).astype(np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetOrigin([0, 0, 0])
            img.SetSpacing([1.0, 1.0, 1.0])
            
            sitk.WriteImage(img, os.path.join(self.input_dir, f'test_{i}.mha'))
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_batch_resample_to_spacing(self):
        """Test batch resampling to target spacing."""
        target_spacing = [2.0, 2.0, 2.0]
        
        processed, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            input_format='.mha',
            target_spacing=target_spacing,
            order=1
        )
        
        # Check that all files were processed
        self.assertEqual(len(processed), 3)
        
        # Check output files exist
        for i in range(3):
            output_path = os.path.join(self.output_dir, f'test_{i}.mha')
            self.assertTrue(os.path.exists(output_path))
            
            # Verify spacing
            img = sitk.ReadImage(output_path)
            self.assertAlmostEqual(img.GetSpacing()[0], 2.0, places=4)
    
    def test_batch_resample_to_size(self):
        """Test batch resampling to target size."""
        target_size = [10, 10, 10]
        
        processed, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            input_format='.mha',
            target_size=target_size,
            order=1
        )
        
        self.assertEqual(len(processed), 3)
        
        # Verify size
        for i in range(3):
            output_path = os.path.join(self.output_dir, f'test_{i}.mha')
            img = sitk.ReadImage(output_path)
            self.assertEqual(img.GetSize()[0], 10)
    
    def test_skip_existing_files(self):
        """Test that existing files are skipped."""
        target_spacing = [2.0, 2.0, 2.0]
        
        # First run
        processed1, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            target_spacing=target_spacing,
            skip_existing=True
        )
        
        self.assertEqual(len(processed1), 3)
        
        # Second run should skip all files
        processed2, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            target_spacing=target_spacing,
            skip_existing=True
        )
        
        self.assertEqual(len(processed2), 0)
    
    def test_overwrite_existing_files(self):
        """Test overwriting existing files."""
        target_spacing = [2.0, 2.0, 2.0]
        
        # First run
        processed1, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            target_spacing=target_spacing
        )
        
        # Second run with skip_existing=False should process all
        processed2, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            target_spacing=target_spacing,
            skip_existing=False
        )
        
        self.assertEqual(len(processed2), 3)
    
    def test_filter_by_testing_samples(self):
        """Test filtering images by testing_samples list."""
        target_spacing = [2.0, 2.0, 2.0]
        testing_samples = ['test_0', 'test_2']
        
        processed, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            target_spacing=target_spacing,
            testing_samples=testing_samples
        )
        
        # Should only process 2 images
        self.assertEqual(len(processed), 2)
        self.assertIn('test_0.mha', processed)
        self.assertIn('test_2.mha', processed)
    
    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        new_output = os.path.join(self.temp_dir, 'new_output')
        self.assertFalse(os.path.exists(new_output))
        
        resample_images_batch(
            self.input_dir,
            new_output,
            target_spacing=[2.0, 2.0, 2.0]
        )
        
        self.assertTrue(os.path.exists(new_output))
    
    def test_empty_folder(self):
        """Test processing empty folder."""
        empty_dir = os.path.join(self.temp_dir, 'empty')
        os.makedirs(empty_dir)
        
        processed, _ = resample_images_batch(
            empty_dir,
            self.output_dir,
            target_spacing=[2.0, 2.0, 2.0]
        )
        
        self.assertEqual(len(processed), 0)
    
    def test_different_input_format(self):
        """Test with different input format filter."""
        # Create a NIFTI file
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        nii_path = os.path.join(self.input_dir, 'test_nii.nii.gz')
        sitk.WriteImage(img, nii_path)
        
        # Process only .nii.gz files
        processed, _ = resample_images_batch(
            self.input_dir,
            self.output_dir,
            input_format='.nii.gz',
            target_spacing=[2.0, 2.0, 2.0]
        )
        
        # Should only process 1 file
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0], 'test_nii.nii.gz')


class TestResampleEdgeCases(unittest.TestCase):
    """Test edge cases for resampling."""
    
    def test_resample_very_small_image(self):
        """Test resampling very small image."""
        arr = np.ones((2, 2, 2), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([1.0, 1.0, 1.0])
        
        result = resample_image(img, target_size=[4, 4, 4], order=1)
        
        self.assertEqual(result.GetSize()[0], 4)
    
    def test_resample_with_large_spacing_ratio(self):
        """Test resampling with large spacing change."""
        arr = np.random.rand(100, 100, 100).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([0.1, 0.1, 0.1])
        
        # Increase spacing by 100x
        result = resample_image(img, target_spacing=[10.0, 10.0, 10.0], order=1)
        
        # Should result in much smaller image
        self.assertLess(result.GetSize()[0], 10)
    
    def test_resample_preserves_pixel_type(self):
        """Test that resampling preserves appropriate pixel types."""
        # Integer image
        arr = np.arange(27).reshape(3, 3, 3).astype(np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([1.0, 1.0, 1.0])
        
        result = resample_image(img, target_spacing=[2.0, 2.0, 2.0], order=0)
        
        # Result should still be valid
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
