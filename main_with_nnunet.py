#!/usr/bin/env python3
"""
Run extract_patches then write_nnunet_dataset for each modality.
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from vascular_segment_sampler import io
from vascular_segment_sampler.nnunet import write_nnunet_dataset
from vascular_segment_sampler.sampling import extract_patches


def run_extract(args):
    """Run extract_patches with the provided arguments."""
    print("=" * 80)
    print("Running extract_patches...")
    print("=" * 80)
    extract_patches(
        data_dir=args.data_dir,
        outdir=args.outdir,
        config=args.config_name,
        perc_dataset=args.perc_dataset,
        num_cores=args.num_cores,
        start_from=args.start_from,
        end_at=args.end_at,
        testing=args.testing,
        validation_prop=args.validation_prop,
        max_samples=args.max_samples,
        modality=args.modality,
        truth_from_surface=args.truth_from_surface,
        truth_target_spacing=args.truth_target_spacing,
        truth_regenerate=args.truth_regenerate,
        yes=True,  # already confirmed below
    )
    print()
    print("=" * 80)
    print("extract_patches completed successfully")
    print("=" * 80)
    print()


def run_create_nnunet(args, modality):
    """Run write_nnunet_dataset for a specific modality."""
    print("=" * 80)
    print(f"Running write_nnunet_dataset for modality: {modality}")
    print("=" * 80)

    outdir = args.outdir if args.outdir else "./extracted_data/"
    path = write_nnunet_dataset(
        indir=outdir,
        outdir=outdir,
        name=args.nnunet_name,
        dataset_number=args.nnunet_dataset_number,
        modality=modality.lower(),
        start_from=args.nnunet_start_from,
    )
    print(f"Wrote {path}")
    print()
    print("=" * 80)
    print(f"write_nnunet_dataset completed successfully for {modality}")
    print("=" * 80)
    print()
    return True


def get_processed_modalities(outdir):
    """Get list of modalities that were processed."""
    modalities = []
    outdir_path = Path(outdir) if outdir else Path("./extracted_data/")

    if not outdir_path.exists():
        return modalities

    for item in outdir_path.iterdir():
        if not item.is_dir() or item.name.endswith("_masks"):
            continue
        if "_train" in item.name:
            mod = item.name.split("_train")[0].upper()
            if mod and mod not in modalities:
                modalities.append(mod)

    return modalities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patches then convert to nnU-Net format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-outdir", "--outdir", default="./extracted_data/", type=str)
    parser.add_argument("-config_name", "--config_name", type=str, required=True)
    parser.add_argument("-perc_dataset", "--perc_dataset", default=1.0, type=float)
    parser.add_argument("-num_cores", "--num_cores", default=1, type=int)
    parser.add_argument("-start_from", "--start_from", default=0, type=int)
    parser.add_argument("-end_at", "--end_at", default=-1, type=int)
    parser.add_argument("-data_dir", "--data_dir", required=True, type=str)
    parser.add_argument("-testing", "--testing", action="store_true")
    parser.add_argument("-validation_prop", "--validation_prop", type=float, default=None)
    parser.add_argument("-max_samples", "--max_samples", type=float, default=None)
    parser.add_argument("-modality", "--modality", type=str, default=None)
    parser.add_argument(
        "--truth_from_surface",
        "--seg_from_surface",
        dest="truth_from_surface",
        action="store_true",
    )
    parser.add_argument(
        "--truth_target_spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=None,
    )
    parser.add_argument("--truth_regenerate", action="store_true")
    parser.add_argument("--yes", action="store_true")

    parser.add_argument("-nnunet_name", "--nnunet_name", type=str, default="AORTAS")
    parser.add_argument(
        "-nnunet_dataset_number", "--nnunet_dataset_number", type=int, default=1
    )
    parser.add_argument("-nnunet_start_from", "--nnunet_start_from", type=int, default=0)
    parser.add_argument("-skip_nnunet", "--skip_nnunet", action="store_true")

    args = parser.parse_args()

    print("=" * 80)
    print("Vascular Segment Sampler with nnUNet Conversion")
    print("=" * 80)
    print()

    if not args.yes and not io.prompt_continue("Wish to continue? [y/n]: "):
        print("Aborting before starting first case.")
        sys.exit(0)

    try:
        run_extract(args)
        if args.truth_regenerate and args.truth_target_spacing is not None:
            print(
                "Info: Regenerated-truth resampling spacing details are written "
                "to info txt by extract_patches."
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR running extract_patches: {e}")
        sys.exit(1)

    if not args.skip_nnunet:
        modalities = []
        if args.modality:
            modalities = [m.strip().upper() for m in args.modality.split(",")]
        else:
            try:
                global_config = io.load_yaml(f"./config/{args.config_name}.yaml")
                if "MODALITY" in global_config:
                    modalities = [
                        m.upper() if isinstance(m, str) else m
                        for m in global_config["MODALITY"]
                    ]
                else:
                    modalities = get_processed_modalities(args.outdir)
            except Exception as e:
                print(
                    f"Warning: Could not load config, trying to detect modalities "
                    f"from output: {e}"
                )
                modalities = get_processed_modalities(args.outdir)

        if not modalities:
            print("WARNING: No modalities detected. Skipping nnUNet conversion.")
            print("You may need to specify --modality explicitly.")
        else:
            print(f"Detected modalities: {modalities}")
            print()
            for modality in tqdm(modalities, desc="nnUNet modalities"):
                try:
                    success = run_create_nnunet(args, modality)
                    if not success:
                        print(f"WARNING: nnUNet conversion failed for {modality}")
                except KeyboardInterrupt:
                    print(f"\nInterrupted during nnUNet conversion for {modality}")
                    sys.exit(1)
                except Exception as e:
                    print(f"\nERROR running write_nnunet_dataset for {modality}: {e}")

    print("=" * 80)
    print("All processing completed!")
    print("=" * 80)
