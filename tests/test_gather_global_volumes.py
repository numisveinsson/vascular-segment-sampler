"""Tests for gather_global_volumes (whole-image nnU-Net dataset creation)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vascular_segment_sampler.sampling import gather_global_volumes


_MINIMAL_VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="1" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0</DataArray>
      </Points>
    </Piece>
  </PolyData>
</VTKFile>
"""


def _write_volume(path, array, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


class TestGatherGlobalVolumes(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.data_dir = self.root / "data"
        self.outdir = self.root / "out"
        case = "toy_global"
        img = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        seg = np.zeros((8, 8, 8), dtype=np.uint8)
        seg[2:6, 2:6, 2:6] = 1
        _write_volume(self.data_dir / "images" / f"{case}.nii.gz", img)
        _write_volume(self.data_dir / "truths" / f"{case}.nii.gz", seg)
        cl_dir = self.data_dir / "centerlines"
        cl_dir.mkdir(parents=True, exist_ok=True)
        (cl_dir / f"{case}.vtp").write_text(_MINIMAL_VTP)
        (self.data_dir / "surfaces").mkdir(exist_ok=True)
        self.config = {
            "DATASET_NAME": "other",
            "BINARIZE": True,
            "WRITE_SAMPLES": True,
            "WRITE_IMG": True,
            "WRITE_VTK": False,
            "WRITE_SURFACE": False,
            "WRITE_CENTERLINE": False,
            "TESTING": False,
            "VALIDATION_PROP": 0.0,
            "TEST_CASES": [],
            "BAD_CASES": [],
            "MODALITY": ["CT"],
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writes_train_volumes(self):
        out = gather_global_volumes(
            data_dir=str(self.data_dir),
            outdir=str(self.outdir),
            config=self.config,
            yes=True,
        )
        img = Path(out) / "ct_train" / "toy_global.nii.gz"
        lab = Path(out) / "ct_train_masks" / "toy_global.nii.gz"
        self.assertTrue(img.is_file(), f"missing {img}")
        self.assertTrue(lab.is_file(), f"missing {lab}")
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(lab)))
        self.assertEqual(int(seg.max()), 1)
        self.assertEqual(int(seg.min()), 0)

    def test_convert_nnunet(self):
        nnunet_out = self.root / "nnunet_raw"
        gather_global_volumes(
            data_dir=str(self.data_dir),
            outdir=str(self.outdir),
            config=self.config,
            convert_nnunet=True,
            nnunet_name="TOY",
            nnunet_dataset_number=999,
            nnunet_outdir=str(nnunet_out),
            yes=True,
        )
        ds = nnunet_out / "Dataset0999_TOYCT"
        self.assertTrue((ds / "dataset.json").is_file())
        images = list((ds / "imagesTr").glob("*.nii.gz"))
        labels = list((ds / "labelsTr").glob("*.nii.gz"))
        self.assertEqual(len(images), 1)
        self.assertEqual(len(labels), 1)
        self.assertTrue(images[0].name.endswith("_0000.nii.gz"))
        self.assertFalse(labels[0].name.endswith("_0000.nii.gz"))

    def test_cli_help(self):
        from vascular_segment_sampler.cli import global_main

        with self.assertRaises(SystemExit) as ctx:
            global_main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
