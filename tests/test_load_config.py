"""Unit tests for packaged default_mm overlay loading."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vascular_segment_sampler import io
from vascular_segment_sampler.io import load_config, load_default_config


class TestLoadConfig(unittest.TestCase):

    def test_package_ships_only_library_defaults(self):
        pkg = Path(io._PKG_DIR) / "config"
        names = sorted(p.name for p in pkg.iterdir() if p.suffix in {".yaml", ".yml"})
        self.assertEqual(
            names,
            ["default_cm.yaml", "default_mm.yaml", "test.yml"],
        )
        for name in ("default_mm", "default_cm", "test.yml"):
            resolved = Path(io.resolve_config_path(name)).resolve()
            self.assertEqual(resolved.parent, pkg.resolve())

    def test_default_has_radius_relative_sampling(self):
        cfg = load_config("default_mm")
        self.assertEqual(cfg["MOVE_DIST"], 1.5)
        self.assertEqual(cfg["MU_SIZE"], 5)
        self.assertEqual(cfg["RADIUS_ADD"], 0.2)
        self.assertEqual(cfg["RADIUS_SCALE"], 1)
        self.assertEqual(cfg["MIN_DIM"], 5)
        self.assertEqual(cfg["UNITS"], "mm")
        self.assertEqual(cfg["LARGE_RADIUS_THRESH"], 4.0)

    def test_default_mm_scales_absolute_lengths(self):
        mm = load_config("default_mm")
        cm = load_config("default_cm")
        self.assertEqual(mm["UNITS"], "mm")
        self.assertEqual(cm["UNITS"], "cm")
        self.assertEqual(mm["RADIUS_ADD"], 0.2)
        self.assertEqual(mm["LARGE_RADIUS_THRESH"], 4.0)
        self.assertEqual(mm["MOVE_DIST"], cm["MOVE_DIST"])
        self.assertEqual(mm["MU_SIZE"], cm["MU_SIZE"])
        self.assertAlmostEqual(mm["RADIUS_ADD"], cm["RADIUS_ADD"] * 10)
        self.assertAlmostEqual(mm["LARGE_RADIUS_THRESH"], cm["LARGE_RADIUS_THRESH"] * 10)

    def test_test_yml_caps_samples_and_keeps_mm_defaults(self):
        cfg = load_config("test.yml")
        packaged = (Path(io._PKG_DIR) / "config" / "test.yml").resolve()
        self.assertEqual(Path(io.resolve_config_path("test.yml")).resolve(), packaged)
        self.assertEqual(Path(io.resolve_config_path("test")).name, "test.yaml")
        self.assertEqual(cfg["MAX_SAMPLES"], 12)
        self.assertEqual(cfg["MOVE_DIST"], 8)
        self.assertEqual(cfg["NUMBER_SAMPLES"], 1)
        self.assertEqual(cfg["NUMBER_SAMPLES_BIFURC"], 1)
        self.assertEqual(cfg["NUMBER_SAMPLES_START"], 1)
        self.assertEqual(cfg["RADIUS_ADD"], 0.2)
        self.assertEqual(cfg["MU_SIZE"], 5)
        self.assertEqual(cfg["UNITS"], "mm")
        self.assertFalse(cfg["WRITE_TRAJECTORIES"])

    def test_named_config_overrides_defaults(self):
        cfg = load_config("global")
        self.assertEqual(cfg["ANATOMY"], ["Aorta", "Abdominal Aorta"])
        self.assertEqual(cfg["RADIUS_ADD"], 0.1)
        self.assertEqual(cfg["MOVE_DIST"], 1.5)

    def test_dict_overlay_inherits_missing_keys(self):
        cfg = load_config({"ANATOMY": ["Coronary"], "DATASET_NAME": "vmr"})
        self.assertEqual(cfg["ANATOMY"], ["Coronary"])
        self.assertEqual(cfg["DATASET_NAME"], "vmr")
        self.assertEqual(cfg["RADIUS_ADD"], 0.2)
        self.assertEqual(cfg["MOVE_DIST"], 1.5)

    def test_empty_yaml_list_keys_become_lists(self):
        cfg = load_config("default_mm")
        self.assertEqual(cfg["TEST_CASES"], [])
        self.assertEqual(cfg["BAD_CASES"], [])

    def test_null_overlay_list_keys_become_lists(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("TEST_CASES:\nBAD_CASES:\n")
            path = fh.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["TEST_CASES"], [])
            self.assertEqual(cfg["BAD_CASES"], [])
        finally:
            os.unlink(path)

    def test_print_info_file_accepts_null_test_cases(self):
        from vascular_segment_sampler.sampling.functions import print_info_file

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = tmp if tmp.endswith(os.sep) else tmp + os.sep
            cfg = {"TESTING": False, "OUT_DIR": out_dir}
            print_info_file(cfg, ["case_a"], None, "info.txt")
            with open(os.path.join(tmp, "info.txt")) as fh:
                text = fh.read()
            self.assertIn("Testing models not included:", text)
            self.assertIn("Number of training models: 1", text)

    def test_file_overlay_inherits_missing_keys(self):
        defaults = load_default_config()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("ANATOMY: ['Coronary']\nDATASET_NAME: 'vmr'\n")
            path = fh.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["ANATOMY"], ["Coronary"])
            self.assertEqual(cfg["RADIUS_ADD"], defaults["RADIUS_ADD"])
            self.assertEqual(cfg["NUMBER_SAMPLES"], defaults["NUMBER_SAMPLES"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
