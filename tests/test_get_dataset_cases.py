"""Case listing must ignore centerlines that have no image."""
import tempfile
import unittest
from pathlib import Path

from vascular_segment_sampler.datasets import get_dataset_cases


class TestGetDatasetCases(unittest.TestCase):
    def test_skips_centerlines_without_a_matching_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("images", "centerlines"):
                (root / sub).mkdir()
            (root / "images" / "caseA.nii.gz").write_bytes(b"x")
            (root / "centerlines" / "caseA.vtp").write_bytes(b"x")
            (root / "centerlines" / "orphan.vtp").write_bytes(b"x")
            (root / "centerlines" / "._caseA.vtp").write_bytes(b"x")

            cases = get_dataset_cases(str(root), img_ext=".nii.gz")

        self.assertEqual(cases, ["caseA"])

    def test_removes_test_cases_from_the_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("images", "centerlines"):
                (root / sub).mkdir()
            for name in ("keep", "holdout"):
                (root / "images" / f"{name}.nii.gz").write_bytes(b"x")
                (root / "centerlines" / f"{name}.vtp").write_bytes(b"x")

            cases = get_dataset_cases(
                str(root), img_ext=".nii.gz", test_cases=["holdout"]
            )

        self.assertEqual(cases, ["keep"])


if __name__ == "__main__":
    unittest.main()
