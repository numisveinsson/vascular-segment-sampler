"""Convert extracted patch datasets into nnU-Net raw dataset layout."""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional


SUPPORTED_EXTENSIONS = (".nii.gz", ".nrrd")


def _get_extension(filename: str):
    """Return the file extension (.nii.gz or .nrrd) or None if not supported."""
    if filename.endswith(".nii.gz"):
        return ".nii.gz"
    if filename.endswith(".nrrd"):
        return ".nrrd"
    return None


def _get_base_name(filename: str) -> str:
    """Strip extension to get base name for matching image/label pairs."""
    ext = _get_extension(filename)
    if ext:
        return filename[: -len(ext)]
    return filename


def _filter_by_extensions(filelist):
    """Filter file list to supported extensions and return (filtered_list, detected_extension)."""
    filtered = [f for f in filelist if _get_extension(f) is not None]
    ext = _get_extension(filtered[0]) if filtered else ".nii.gz"
    return filtered, ext


def save_json(data, filename: str) -> None:
    """Save JSON file with indentation."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def write_nnunet_dataset(
    indir: str,
    name: str,
    dataset_number: int,
    modality: str,
    outdir: Optional[str] = None,
    start_from: int = 0,
    also_test: bool = False,
) -> str:
    """
    Create an nnU-Net DatasetXXX_* folder from extracted patches or images/labels.

    Supported input layouts under ``indir``:
      1. ``{modality}_train`` + ``{modality}_train_masks``
      2. ``images`` + ``labels``
      3. ``images`` + ``truths``

    Returns the path to the created Dataset directory.
    """
    directory = indir
    directory_out = outdir if outdir is not None else indir
    modality = modality.lower()

    if dataset_number < 10:
        dataset_number_str = "0" + str(dataset_number)
    else:
        dataset_number_str = str(dataset_number)

    new_dir_dataset_name = "Dataset0" + dataset_number_str + "_" + name + modality.upper()
    append = name.lower() + modality.lower()
    out_data_dir = os.path.join(directory_out, new_dir_dataset_name)

    os.makedirs(directory_out, exist_ok=True)
    try:
        os.mkdir(os.path.join(directory_out, new_dir_dataset_name))
    except FileExistsError:
        print(f"Directory {new_dir_dataset_name} already exists")

    if os.path.exists(os.path.join(directory, modality + "_train")) and os.path.exists(
        os.path.join(directory, modality + "_train_masks")
    ):
        fns_in = [modality + "_train", modality + "_train_masks"]
        if also_test:
            fns_in.extend([modality + "_test", modality + "_test_masks"])
        input_format = "modality"
    elif os.path.exists(os.path.join(directory, "images")) and os.path.exists(
        os.path.join(directory, "labels")
    ):
        fns_in = ["images", "labels"]
        input_format = "images_labels"
    elif os.path.exists(os.path.join(directory, "images")) and os.path.exists(
        os.path.join(directory, "truths")
    ):
        fns_in = ["images", "truths"]
        input_format = "images_labels"
    else:
        raise FileNotFoundError(
            f"Could not find expected directory structure in {directory}. "
            f"Expected either: (1) {modality}_train and {modality}_train_masks, "
            "(2) images and labels, or (3) images and truths."
        )

    fns_out = ["imagesTr", "labelsTr"]
    if also_test and input_format == "modality":
        fns_out.extend(["imagesTs", "labelsTs"])

    for fn in fns_out:
        try:
            os.mkdir(os.path.join(directory_out, new_dir_dataset_name, fn))
        except FileExistsError:
            print(f"Directory {fn} already exists")

    file_ending = ".nii.gz"
    name_mappings = []
    num_training = 0

    if input_format == "modality":
        for fn in fns_in:
            if not os.path.exists(os.path.join(directory, fn)):
                print(f"{fn} does not exist")
                continue
            all_files = os.listdir(os.path.join(directory, fn))
            imgs, file_ext = _filter_by_extensions(all_files)
            imgs.sort()

            out_subfolder = fns_out[fns_in.index(fn)]
            for i, img in enumerate(imgs):
                ext = _get_extension(img)
                new_name = f"{append}_{(i + 1 + start_from):03d}_0000{ext}"
                if out_subfolder in ("labelsTr", "labelsTs"):
                    new_name = new_name.replace("_0000", "")
                name_mappings.append((f"{out_subfolder}/{new_name}", f"{fn}/{img}"))
                print(f"Copying {img} to {new_name}")
                if img != new_name:
                    shutil.copy(
                        os.path.join(directory, fn, img),
                        os.path.join(out_data_dir, out_subfolder, new_name),
                    )
            if fn == fns_in[0]:
                num_training = len(imgs)
                file_ending = file_ext
    else:
        images_dir = os.path.join(directory, "images")
        labels_dir = os.path.join(directory, fns_in[1])
        all_files = os.listdir(images_dir)
        imgs, file_ext = _filter_by_extensions(all_files)
        imgs.sort()
        file_ending = file_ext

        copy_count = 0
        for img in imgs:
            ext = _get_extension(img)
            base = _get_base_name(img)
            label_name = base + ext
            label_path = os.path.join(labels_dir, label_name)
            if not os.path.exists(label_path):
                print(f"Warning: no matching label for {img}, skipping")
                continue
            copy_count += 1
            idx = copy_count + start_from
            img_new = f"{append}_{idx:03d}_0000{ext}"
            label_new = f"{append}_{idx:03d}{ext}"
            name_mappings.append((f"imagesTr/{img_new}", f"images/{img}"))
            name_mappings.append((f"labelsTr/{label_new}", f"{fns_in[1]}/{label_name}"))
            print(f"Copying {img} to {img_new}")
            shutil.copy(
                os.path.join(images_dir, img),
                os.path.join(out_data_dir, "imagesTr", img_new),
            )
            print(f"Copying {label_name} to {label_new}")
            shutil.copy(label_path, os.path.join(out_data_dir, "labelsTr", label_new))
        num_training = copy_count

    dataset_json = {
        "channel_names": {"0": modality.upper()},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": num_training,
        "file_ending": file_ending,
    }
    save_json(dataset_json, os.path.join(out_data_dir, "dataset.json"))

    mapping_path = os.path.join(out_data_dir, "name_mapping.txt")
    with open(mapping_path, "w") as f:
        f.write("# nnunet_name -> original_path\n")
        for nnunet_path, original_path in name_mappings:
            f.write(f"{nnunet_path} -> {original_path}\n")

    return out_data_dir
