"""
Unit tests for map_to_image with FIXED_EXTRACT_SIZE option.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from modules.sitk_functions import map_to_image


class TestMapToImageFixedSize(unittest.TestCase):
    """Test map_to_image with fixed_size parameter."""

    def test_fixed_size_returns_exact_voxel_dimensions(self):
        """Fixed extraction returns exactly the requested voxel size."""
        point = np.array([50.0, 50.0, 50.0])
        origin_im = np.array([0.0, 0.0, 0.0])
        spacing_im = np.array([1.0, 1.0, 1.0])
        size_im = np.array([100, 100, 100])
        fixed_size = [64, 64, 64]

        size_extract, index_extract, voi_min, voi_max = map_to_image(
            point, radius=1.0, size_volume=5.0,
            origin_im=origin_im, spacing_im=spacing_im, size_im=size_im,
            fixed_size=fixed_size
        )
        self.assertIsNotNone(size_extract)
        np.testing.assert_array_equal(size_extract, [64, 64, 64])

    def test_fixed_size_same_physical_size_for_same_spacing(self):
        """With same spacing, physical size (voi extent) is consistent."""
        point = np.array([50.0, 50.0, 50.0])
        origin_im = np.array([0.0, 0.0, 0.0])
        spacing_im = np.array([1.0, 1.0, 1.0])
        size_im = np.array([100, 100, 100])
        fixed_size = [64, 64, 64]

        _, _, voi_min, voi_max = map_to_image(
            point, radius=1.0, size_volume=5.0,
            origin_im=origin_im, spacing_im=spacing_im, size_im=size_im,
            fixed_size=fixed_size
        )
        physical_size = voi_max - voi_min
        np.testing.assert_array_almost_equal(physical_size, [64.0, 64.0, 64.0])

    def test_fixed_size_out_of_bounds_returns_none(self):
        """When extraction would go out of bounds, returns None to signal skip."""
        point = np.array([5.0, 5.0, 5.0])  # Near corner
        origin_im = np.array([0.0, 0.0, 0.0])
        spacing_im = np.array([1.0, 1.0, 1.0])
        size_im = np.array([20, 20, 20])
        fixed_size = [64, 64, 64]  # Too large to fit

        result = map_to_image(
            point, radius=1.0, size_volume=5.0,
            origin_im=origin_im, spacing_im=spacing_im, size_im=size_im,
            fixed_size=fixed_size
        )
        self.assertEqual(result[0], None)

    def test_without_fixed_size_unchanged_behavior(self):
        """Without fixed_size, original radius-based behavior is preserved."""
        point = np.array([50.0, 50.0, 50.0])
        origin_im = np.array([0.0, 0.0, 0.0])
        spacing_im = np.array([1.0, 1.0, 1.0])
        size_im = np.array([100, 100, 100])
        radius = 2.0
        size_volume = 5.0

        size_extract, _, _, _ = map_to_image(
            point, radius=radius, size_volume=size_volume,
            origin_im=origin_im, spacing_im=spacing_im, size_im=size_im,
            fixed_size=None
        )
        # size_extract = ceil(size_volume * radius / spacing) = ceil(5*2/1) = 10
        np.testing.assert_array_equal(size_extract, [10, 10, 10])


if __name__ == '__main__':
    unittest.main()
