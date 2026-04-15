#!/usr/bin/env python3
"""
Flatten a nested dataset: each case is a subfolder with VTK image under Images/
and surface under Models/. Reads .vti from Images/, converts to the requested
output image format (via preprocessing.change_img_format), writes under
output/images, and copies .vtp from Models/ to output/surfaces. If the image
basename (stem) differs from the surface stem, the surface is written using the
image stem with a .vtp extension. All such renames are logged in the output folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import SimpleITK as sitk

# Project root on path for preprocessing.* and modules.*
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from modules import vtk_functions as vf
from preprocessing.change_img_format import change_vti_sitk

# Supported when reading .vti (matches change_img_format.py main conversion paths)
_VTI_OUTPUT_FORMATS = frozenset({".vti", ".mha", ".nii.gz"})


def _normalize_ext(s: str) -> str:
    s = s.strip().lower()
    if not s.startswith("."):
        s = "." + s
    return s


def _convert_vti_to_format(
    vti_path: Path,
    dest_path: Path,
    output_format: str,
    *,
    label: bool,
) -> None:
    """Write ``dest_path`` using the same rules as preprocessing/change_img_format.py."""
    if output_format == ".vti":
        vtk_img = vf.read_img(str(vti_path)).GetOutput()
        vf.write_img(str(dest_path), vtk_img)
    elif output_format in (".mha", ".nii.gz"):
        img = change_vti_sitk(str(vti_path), label=label)
        sitk.WriteImage(img, str(dest_path))
    else:
        raise ValueError(f"Unsupported output image format for .vti input: {output_format}")


def _list_files(folder: Path, suffix: str) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read .vti from each subfolder's Images/, convert to --output-image-format "
            "(preprocessing.change_img_format rules), write under output/images/; copy .vtp "
            "from Models/ to output/surfaces/; align surface basename to image stem when needed."
        )
    )
    p.add_argument("input_folder", type=Path, help="Root folder containing one folder per case")
    p.add_argument("output_folder", type=Path, help="Destination root (images/ and surfaces/ created here)")
    p.add_argument(
        "--output-image-format",
        "--output_image_format",
        type=str,
        default=".vti",
        help=(
            "Image extension written under output/images/ after converting from .vti "
            f"(default: .vti). Supported: {', '.join(sorted(_VTI_OUTPUT_FORMATS))}."
        ),
    )
    p.add_argument(
        "--label",
        action="store_true",
        help="Pass through to change_vti_sitk: cast to UInt8 for label images (.mha / .nii.gz only)",
    )
    p.add_argument(
        "--log-name",
        default="surface_name_changes.txt",
        help="Log file name written under output_folder (default: surface_name_changes.txt)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root: Path = args.input_folder.resolve()
    output_root: Path = args.output_folder.resolve()
    log_path = output_root / args.log_name
    output_image_format = _normalize_ext(args.output_image_format)
    if output_image_format not in _VTI_OUTPUT_FORMATS:
        print(
            f"Error: --output-image-format must be one of {sorted(_VTI_OUTPUT_FORMATS)}, "
            f"got {output_image_format!r}",
            file=sys.stderr,
        )
        return 1

    if not input_root.is_dir():
        print(f"Error: input folder is not a directory: {input_root}", file=sys.stderr)
        return 1

    images_out = output_root / "images"
    surfaces_out = output_root / "surfaces"
    images_out.mkdir(parents=True, exist_ok=True)
    surfaces_out.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    rename_lines: list[str] = []

    for case_dir in sorted(input_root.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("."):
            continue

        images_sub = case_dir / "Images"
        models_sub = case_dir / "Models"

        vti_files = _list_files(images_sub, ".vti")
        vtp_files = _list_files(models_sub, ".vtp")

        if not vti_files:
            msg = f"[skip] {case_dir.name}: no .vti under Images/"
            print(msg, file=sys.stderr)
            log_lines.append(msg)
            continue
        if not vtp_files:
            msg = f"[skip] {case_dir.name}: no .vtp under Models/"
            print(msg, file=sys.stderr)
            log_lines.append(msg)
            continue

        if len(vti_files) > 1:
            msg = f"[warn] {case_dir.name}: multiple .vti in Images/, using {vti_files[0].name}"
            print(msg, file=sys.stderr)
            log_lines.append(msg)
        if len(vtp_files) > 1:
            msg = f"[warn] {case_dir.name}: multiple .vtp in Models/, using {vtp_files[0].name}"
            print(msg, file=sys.stderr)
            log_lines.append(msg)

        vti_path = vti_files[0]
        vtp_path = vtp_files[0]

        image_stem = vti_path.stem
        surface_stem = vtp_path.stem

        if image_stem == surface_stem:
            out_vtp_name = vtp_path.name
        else:
            out_vtp_name = f"{image_stem}.vtp"
            line = (
                f"case={case_dir.name}\timage={vti_path.name}\tsurface_src={vtp_path.name}\t"
                f"surface_dst={out_vtp_name}"
            )
            rename_lines.append(line)
            print(line)

        out_img_name = f"{image_stem}{output_image_format}"
        dest_img = images_out / out_img_name
        dest_surf = surfaces_out / out_vtp_name

        if dest_img.exists():
            print(f"[warn] overwriting existing image: {dest_img}", file=sys.stderr)
        if dest_surf.exists() and dest_surf.resolve() != vtp_path.resolve():
            print(f"[warn] overwriting existing surface: {dest_surf}", file=sys.stderr)

        _convert_vti_to_format(vti_path, dest_img, output_image_format, label=args.label)
        shutil.copy2(vtp_path, dest_surf)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Output image format: {output_image_format}\n")
        if args.label:
            f.write("Label mode: UInt8 cast for .mha / .nii.gz (change_vti_sitk)\n")
        f.write("\n")
        if rename_lines:
            f.write("Surface files renamed to match image stem (basename without .vti -> .vtp):\n")
            for line in rename_lines:
                f.write(line + "\n")
        else:
            f.write("No surface renames were required (image and surface stems already matched).\n")
        if log_lines:
            f.write("\nNotes / warnings:\n")
            for line in log_lines:
                f.write(line + "\n")

    print(f"Wrote log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
