"""Gather whole images/labels into modality train folders (and optionally nnU-Net)."""

from __future__ import annotations

from datetime import datetime
import os
import random

import SimpleITK as sitk
from tqdm import tqdm

from vascular_segment_sampler import create_seg_from_surf as _create_seg_from_surf
from vascular_segment_sampler import io
from vascular_segment_sampler.datasets import (
    create_dataset,
    get_case_dict_dir,
    resolve_case_surface_path,
)
from vascular_segment_sampler.nnunet import write_nnunet_dataset
from vascular_segment_sampler.preprocessing.change_img_resample import resample_image
from vascular_segment_sampler.sampling.extract import _load_config
from vascular_segment_sampler.sampling.functions import (
    choose_destination,
    create_directories,
    print_info_file,
)


def _trailing_sep(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    return os.path.join(path, "")


def _write_text_lines(filepath, lines):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def _load_seg_from_truths(case_dict):
    seg_path = case_dict.get("SEGMENTATION")
    if not seg_path or not os.path.isfile(seg_path):
        return None
    try:
        return sitk.Cast(sitk.ReadImage(seg_path), sitk.sitkUInt8)
    except Exception as e:
        print(e)
        return None


def _seg_from_surface(data_dir, case_name, img, target_spacing=None):
    surf_path = resolve_case_surface_path(data_dir, case_name)
    if surf_path is None:
        return None, None
    surface = _create_seg_from_surf.load_surface_polydata(surf_path)
    seg = _create_seg_from_surf.seg_sitk_from_surface_polydata(
        surface,
        img,
        target_spacing=target_spacing,
        resample_order=3,
    )
    return seg, surf_path


def _binarize_seg(seg, img, binarize):
    if seg is None:
        return None
    seg = sitk.Cast(seg, sitk.sitkUInt8)
    seg_np = sitk.GetArrayFromImage(seg)
    max_val = seg_np.max()
    if binarize and max_val > 1:
        seg_np = seg_np / max_val
        seg = sitk.GetImageFromArray(seg_np)
        seg.CopyInformation(img)
        print("Seg Max value: ", seg_np.max())
        print("Seg Min value: ", seg_np.min())
        seg = sitk.Cast(seg, sitk.sitkUInt8)
    return seg


def _load_image_and_seg(
    global_config,
    case_dict,
    *,
    target_spacing,
    truth_from_surface,
    truth_regenerate,
):
    """Load the whole image and a matching segmentation for one case."""
    data_dir = os.path.abspath(os.path.expanduser(global_config["DATA_DIR"]))
    img = sitk.ReadImage(case_dict["IMAGE"])
    name = case_dict["NAME"]

    if target_spacing is not None:
        surf_path = resolve_case_surface_path(data_dir, name)
        if surf_path is None:
            raise FileNotFoundError(
                f"No surface mesh found for case {name} under "
                f"{os.path.join(data_dir, 'surfaces')} (.vtp or .stl)"
            )
        print(
            f"{name}: converting surface to segmentation at "
            f"target spacing {list(target_spacing)} ({surf_path})"
        )
        img = resample_image(img, target_spacing=list(target_spacing), order=3)
        seg, _ = _seg_from_surface(data_dir, name, img, target_spacing=None)
        return img, seg

    seg = None
    if not truth_regenerate:
        truths_dir = os.path.join(data_dir, "truths")
        if os.path.isdir(truths_dir):
            seg = _load_seg_from_truths(case_dict)
            if seg is not None and truth_from_surface:
                print(f"Using existing segmentation: {case_dict['SEGMENTATION']}")

    if seg is None:
        seg, surf_path = _seg_from_surface(data_dir, name, img, target_spacing=None)
        if seg is None:
            raise FileNotFoundError(
                f"No usable truth for case {name}: "
                f"truths/ missing or empty, truth file missing or unreadable, and no surface under "
                f"{os.path.join(data_dir, 'surfaces')} (.vtp or .stl)"
            )
        print(
            f"{name}: segmentation from surface at native image spacing "
            f"(truths dir or volume unavailable) ({surf_path})"
        )
    return img, seg


def gather_global_volumes(
    data_dir,
    outdir="./extracted_data/",
    config="global",
    *,
    perc_dataset=1.0,
    num_cores=1,
    start_from=0,
    end_at=-1,
    testing=False,
    validation_prop=None,
    modality=None,
    output_suffix="",
    truth_from_surface=False,
    truth_target_spacing=None,
    truth_regenerate=False,
    target_spacing=None,
    convert_nnunet=False,
    nnunet_name="AORTAS",
    nnunet_dataset_number=1,
    nnunet_outdir=None,
    nnunet_start_from=0,
    also_test=False,
    yes=False,
    verbose=False,
):
    """
    Copy whole-case images and labels into ``{modality}_train`` / ``_masks`` folders.

    Same case listing as ``extract_patches`` (``images/``, ``truths/``, ``centerlines/``),
    but writes one volume per case instead of sampled patches. Optionally convert
    those folders to an nnU-Net ``DatasetXXX_*`` layout.

    Parameters
    ----------
    data_dir : str
        Directory with images/, truths/, centerlines/, surfaces/.
    outdir : str
        Output directory for whole-image copies (and nnU-Net conversion if requested).
    config : dict | str
        Config dict or path/name of a YAML config (e.g. ``"global"``).
    """
    global_config = _load_config(config)
    global_config["DATA_DIR"] = data_dir

    if testing:
        global_config["TESTING"] = True
    elif "TESTING" not in global_config:
        global_config["TESTING"] = False

    if validation_prop is not None:
        global_config["VALIDATION_PROP"] = validation_prop
    elif "VALIDATION_PROP" not in global_config:
        global_config["VALIDATION_PROP"] = 0.0

    if modality is not None:
        modalities_list = [m.strip().upper() for m in str(modality).split(",")]
        global_config["MODALITY"] = modalities_list
    elif "MODALITY" not in global_config:
        global_config["MODALITY"] = ["CT"]

    output_suffix = output_suffix or ""
    global_config["OUTPUT_SUFFIX"] = output_suffix
    global_config["SEG_FROM_SURFACE"] = bool(truth_from_surface)
    spacing = target_spacing if target_spacing is not None else truth_target_spacing
    global_config["TRUTH_TARGET_SPACING"] = spacing
    global_config["TRUTH_REGENERATE"] = bool(truth_regenerate)
    global_config["VERBOSE"] = bool(verbose)

    if "WRITE_SAMPLES" not in global_config:
        global_config["WRITE_SAMPLES"] = True
    if "WRITE_IMG" not in global_config:
        global_config["WRITE_IMG"] = True
    if "WRITE_VTK" not in global_config:
        global_config["WRITE_VTK"] = False
    if "BINARIZE" not in global_config:
        global_config["BINARIZE"] = True
    if "TEST_CASES" not in global_config:
        global_config["TEST_CASES"] = []

    if global_config["WRITE_SAMPLES"]:
        global_config["WRITE_IMG"] = True

    if num_cores > 1:
        print(
            "gather_global_volumes runs sequentially; "
            f"ignoring num_cores={num_cores}"
        )

    modalities = global_config["MODALITY"]
    out_dir = _trailing_sep(outdir)
    global_config["OUT_DIR"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    dt_string = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
    processed_modalities = []

    for modality_name in tqdm(modalities, desc="Modalities"):
        cases = create_dataset(global_config, modality_name)
        cases = sorted(cases)

        random.seed(42)
        random.shuffle(cases)
        n_keep = int(perc_dataset * len(cases))
        cases_after_perc = cases[:n_keep]
        cases_excluded_perc = cases[n_keep:]

        if end_at != -1:
            cases_after_perc = cases_after_perc[start_from:end_at]
        else:
            cases_after_perc = cases_after_perc[start_from:]

        done_file_path = os.path.join(out_dir, f"done{output_suffix}.txt")
        if os.path.exists(done_file_path):
            with open(done_file_path, "r", encoding="utf-8") as f:
                done = f.read().splitlines()
            done_set = set(done)
            skipped_done = [c for c in cases_after_perc if c in done_set]
            for case in skipped_done:
                print(f"Skipping {case}")
            cases = [c for c in cases_after_perc if c not in done_set]
        else:
            skipped_done = []
            cases = cases_after_perc

        modality_l = modality_name.lower()
        suffix = output_suffix
        info_file_name = "info" + "_" + modality_l + dt_string + suffix + ".txt"

        if cases_excluded_perc:
            _write_text_lines(
                os.path.join(
                    out_dir,
                    f"cases_excluded_perc_dataset_{modality_l}{dt_string}{suffix}.txt",
                ),
                cases_excluded_perc,
            )
        if skipped_done:
            _write_text_lines(
                os.path.join(
                    out_dir,
                    f"cases_skipped_done_{modality_l}{dt_string}{suffix}.txt",
                ),
                skipped_done,
            )
        _write_text_lines(
            os.path.join(
                out_dir, f"cases_in_run_{modality_l}{dt_string}{suffix}.txt"
            ),
            cases,
        )

        create_directories(out_dir, modality_l, global_config)

        image_out_dir_train = os.path.join(out_dir, modality_l + "_train" + suffix)
        seg_out_dir_train = os.path.join(out_dir, modality_l + "_train" + suffix + "_masks")
        image_out_dir_val = os.path.join(out_dir, modality_l + "_val" + suffix)
        seg_out_dir_val = os.path.join(out_dir, modality_l + "_val" + suffix + "_masks")
        image_out_dir_test = os.path.join(out_dir, modality_l + "_test" + suffix)
        seg_out_dir_test = os.path.join(out_dir, modality_l + "_test" + suffix + "_masks")

        print(f"\n--- {modality_l} ---")
        print(f"--- {len(cases)} cases ---")
        for case in cases:
            print(f"Case: {case}")

        if cases and not yes and not io.prompt_continue("Wish to continue? [y/n]: "):
            print("Aborting before starting first case.")
            continue

        print_info_file(
            global_config, cases, global_config["TEST_CASES"], info_file_name
        )

        for i, case_fn in enumerate(tqdm(cases, desc=f"{modality_l} cases", leave=False)):
            case_dict = get_case_dict_dir(
                global_config["DATA_DIR"], case_fn, global_config.get("IMG_EXT")
            )
            print(f"\n {i + 1}/{len(cases)}: {case_dict['NAME']}")

            image_out_dir, seg_out_dir, _val_port = choose_destination(
                global_config["TESTING"],
                global_config["VALIDATION_PROP"],
                image_out_dir_test,
                seg_out_dir_test,
                image_out_dir_val,
                seg_out_dir_val,
                image_out_dir_train,
                seg_out_dir_train,
                ip=1,
                case_name=case_dict["NAME"],
            )

            img, seg = _load_image_and_seg(
                global_config,
                case_dict,
                target_spacing=spacing,
                truth_from_surface=truth_from_surface,
                truth_regenerate=truth_regenerate,
            )
            size_im = img.GetSize()

            img_np = sitk.GetArrayFromImage(img)
            print("Img Max value: ", img_np.max())
            print("Img Min value: ", img_np.min())
            if seg is not None:
                seg_np = sitk.GetArrayFromImage(seg)
                print("Seg Max value: ", seg_np.max())
                print("Seg Min value: ", seg_np.min())

            seg = _binarize_seg(seg, img, global_config["BINARIZE"])

            if global_config["WRITE_SAMPLES"]:
                sitk.WriteImage(
                    img, os.path.join(image_out_dir, case_dict["NAME"] + ".nii.gz")
                )
                if seg is not None:
                    sitk.WriteImage(
                        seg, os.path.join(seg_out_dir, case_dict["NAME"] + ".nii.gz")
                    )
            if global_config["WRITE_VTK"]:
                vtk_dir = os.path.join(out_dir, "vtk_data" + suffix)
                sitk.WriteImage(
                    img,
                    os.path.join(vtk_dir, "vtk_" + case_dict["NAME"] + ".mha"),
                )
                if seg is not None:
                    sitk.WriteImage(
                        seg * 255,
                        os.path.join(
                            vtk_dir, "vtk_mask_" + case_dict["NAME"] + ".mha"
                        ),
                    )

            print(f"\n Finished: ' {case_dict['NAME']}, {size_im}")
            with open(done_file_path, "a", encoding="utf-8") as f_done:
                f_done.write(case_dict["NAME"] + "\n")

        processed_modalities.append(modality_l)

    if convert_nnunet:
        convert_out = nnunet_outdir if nnunet_outdir is not None else out_dir
        os.makedirs(convert_out, exist_ok=True)
        for i, modality_l in enumerate(processed_modalities):
            ds_num = nnunet_dataset_number + i
            path = write_nnunet_dataset(
                indir=out_dir,
                name=nnunet_name,
                dataset_number=ds_num,
                modality=modality_l,
                outdir=convert_out,
                start_from=nnunet_start_from,
                also_test=also_test,
            )
            print(f"Wrote nnU-Net dataset to {path}")

    print("All done")
    return out_dir
