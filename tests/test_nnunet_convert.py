"""Tests for nnU-Net dataset conversion."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from vascular_segment_sampler.nnunet import write_nnunet_dataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")


class TestNnUNetConvert(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.indir = self.root / "in"
        self.outdir = self.root / "out"
        train = self.indir / "ct_train"
        masks = self.indir / "ct_train_masks"
        for name in ("case_a.nii.gz", "case_b.nii.gz", "case_c.nii.gz"):
            _touch(train / name)
        _touch(masks / "case_a.nii.gz")
        _touch(masks / "case_c.nii.gz")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dataset_id_formatting(self):
        cases = (
            (1, "Dataset001_TOYCT"),
            (99, "Dataset099_TOYCT"),
            (100, "Dataset100_TOYCT"),
            (999, "Dataset999_TOYCT"),
        )
        for number, expected in cases:
            with self.subTest(dataset_number=number):
                path = write_nnunet_dataset(
                    indir=str(self.indir),
                    name="TOY",
                    dataset_number=number,
                    modality="ct",
                    outdir=str(self.outdir / str(number)),
                )
                self.assertEqual(Path(path).name, expected)
                self.assertTrue((Path(path) / "dataset.json").is_file())

    def test_skips_unmatched_masks_without_shifting_pairs(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            path = write_nnunet_dataset(
                indir=str(self.indir),
                name="TOY",
                dataset_number=1,
                modality="ct",
                outdir=str(self.outdir),
            )
        ds = Path(path)
        images = sorted((ds / "imagesTr").iterdir())
        labels = sorted((ds / "labelsTr").iterdir())
        self.assertEqual([p.name for p in images], ["toyct_001_0000.nii.gz", "toyct_002_0000.nii.gz"])
        self.assertEqual([p.name for p in labels], ["toyct_001.nii.gz", "toyct_002.nii.gz"])

        mapping = (ds / "name_mapping.txt").read_text()
        self.assertIn("imagesTr/toyct_001_0000.nii.gz -> ct_train/case_a.nii.gz", mapping)
        self.assertIn("labelsTr/toyct_001.nii.gz -> ct_train_masks/case_a.nii.gz", mapping)
        self.assertIn("imagesTr/toyct_002_0000.nii.gz -> ct_train/case_c.nii.gz", mapping)
        self.assertIn("labelsTr/toyct_002.nii.gz -> ct_train_masks/case_c.nii.gz", mapping)
        self.assertNotIn("case_b.nii.gz", mapping)

        with open(ds / "dataset.json") as f:
            dataset_json = json.load(f)
        self.assertEqual(dataset_json["numTraining"], 2)
        self.assertIn("Warning: no matching label for case_b.nii.gz, skipping", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
