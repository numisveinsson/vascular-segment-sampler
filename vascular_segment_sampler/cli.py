"""Console entry points: vss-sample, vss-to-nnunet."""

from __future__ import annotations

import argparse
import sys

from vascular_segment_sampler.nnunet import write_nnunet_dataset
from vascular_segment_sampler.sampling import extract_patches


def sample_main(argv=None) -> None:
    """CLI for extract_patches (console script: vss-sample)."""
    parser = argparse.ArgumentParser(
        description="Extract vascular segment patches from medical image cases."
    )
    parser.add_argument(
        "-outdir", "--outdir", default="./extracted_data/", type=str, help="Output directory"
    )
    parser.add_argument(
        "-config_name",
        "--config_name",
        type=str,
        default="global",
        help="Config name (under config/) or path to a YAML file",
    )
    parser.add_argument(
        "-perc_dataset", "--perc_dataset", default=1.0, type=float, help="Fraction of dataset"
    )
    parser.add_argument("-num_cores", "--num_cores", default=1, type=int, help="Worker processes")
    parser.add_argument("-start_from", "--start_from", default=0, type=int)
    parser.add_argument("-end_at", "--end_at", default=-1, type=int)
    parser.add_argument(
        "-data_dir", "--data_dir", required=True, type=str, help="Input data directory"
    )
    parser.add_argument("-testing", "--testing", action="store_true")
    parser.add_argument("-validation_prop", "--validation_prop", type=float, default=None)
    parser.add_argument("-max_samples", "--max_samples", type=float, default=None)
    parser.add_argument("-modality", "--modality", type=str, default=None)
    parser.add_argument("-output_suffix", "--output_suffix", type=str, default="")
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

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
        output_suffix=args.output_suffix,
        truth_from_surface=args.truth_from_surface,
        truth_target_spacing=args.truth_target_spacing,
        truth_regenerate=args.truth_regenerate,
        yes=args.yes,
        verbose=args.verbose,
    )


def nnunet_main(argv=None) -> None:
    """CLI for write_nnunet_dataset (console script: vss-to-nnunet)."""
    parser = argparse.ArgumentParser(
        description="Convert extracted patches (or images/labels) to nnU-Net Dataset format."
    )
    parser.add_argument("-outdir", "--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("-indir", "--indir", type=str, required=True, help="Input directory")
    parser.add_argument("-name", "--name", default="AORTAS", type=str, help="Dataset name")
    parser.add_argument(
        "-dataset_number", "--dataset_number", default=1, type=int, help="Dataset number"
    )
    parser.add_argument("-modality", "--modality", type=str, required=True, help="Modality (ct/mr)")
    parser.add_argument(
        "-start_from",
        "--start_from",
        type=int,
        default=0,
        help="Index offset when appending to an existing dataset",
    )
    args = parser.parse_args(argv)

    outdir = args.outdir if args.outdir is not None else args.indir
    path = write_nnunet_dataset(
        indir=args.indir,
        name=args.name,
        dataset_number=args.dataset_number,
        modality=args.modality,
        outdir=outdir,
        start_from=args.start_from,
    )
    print(f"Wrote nnU-Net dataset to {path}")


if __name__ == "__main__":
    # Default to sample CLI when invoked as a module
    sample_main(sys.argv[1:])
