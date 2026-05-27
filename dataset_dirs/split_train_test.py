"""
Split dataset into training and testing sets.

This script takes a folder containing subfolders (images, centerlines, surfaces, truths, labels)
and splits cases into training and testing sets either randomly (by fraction) or from a CSV
(e.g. dataset_dirs/VMR_train_test_split.csv with row_type, split, legacy_name columns).
"""

import csv
import os
import sys
import shutil
import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Add project root to path so modules can be imported when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.logger import get_logger


def get_base_names(folder, extensions=['.mha', '.nii.gz', '.nii', '.vti', '.vtp', '.stl', '.mhd']):
    """Get base filenames without extensions from a folder."""
    if not os.path.exists(folder):
        return set()
    
    files = os.listdir(folder)
    base_names = set()
    
    for f in files:
        if f.startswith('.'):
            continue
        for ext in extensions:
            if f.endswith(ext):
                base_names.add(f.replace(ext, ''))
                break
    
    return base_names


DEFAULT_SUBFOLDERS = ['images', 'centerlines', 'surfaces', 'truths', 'labels']


def _resolve_subfolder_paths(base_folder: str, subfolders: list, require_existing: bool) -> list:
    """Return subfolder names that exist under base_folder; error if required and missing."""
    logger = get_logger(__name__)
    found = []
    missing = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            found.append(subfolder)
        else:
            missing.append(subfolder)

    if missing:
        msg = f"Subfolder(s) not found under {base_folder}: {missing}"
        if require_existing:
            raise ValueError(msg)
        for subfolder in missing:
            logger.debug(f"Subfolder {subfolder}/ does not exist, skipping")

    return found


def get_case_names(base_folder, case_subfolders=None, require_existing=False):
    """
    Get case names from the dataset folder.

    Args:
        base_folder: Path to the base folder containing subfolders
        case_subfolders: Subfolders whose file basenames are intersected to define cases
        require_existing: If True, every listed subfolder must exist

    Returns:
        Tuple of (case_names, case_subfolders_used)
    """
    if case_subfolders is None:
        case_subfolders = DEFAULT_SUBFOLDERS

    logger = get_logger(__name__)
    existing = _resolve_subfolder_paths(base_folder, case_subfolders, require_existing)

    if not existing:
        raise ValueError(
            f"No valid case subfolders found in {base_folder}. Expected one of: {case_subfolders}"
        )

    name_sets = {}
    for subfolder in existing:
        names = get_base_names(os.path.join(base_folder, subfolder))
        name_sets[subfolder] = names
        logger.info(f"Found {len(names)} cases in {subfolder}/")

    case_names = list(set.intersection(*[name_sets[sf] for sf in existing]))
    logger.info(
        f"Using cases present in all case subfolders {existing}: {len(case_names)} cases"
    )

    if not case_names:
        raise ValueError(f"No cases found in {base_folder} for subfolders {existing}")

    case_names.sort()
    return case_names, existing


def resolve_split_subfolders(
    base_folder: str,
    data_subfolders: Optional[list] = None,
    case_subfolders: Optional[list] = None,
) -> Tuple[list, list]:
    """
    Resolve which subfolders define cases vs which are copied/moved.

    If neither is set, uses DEFAULT_SUBFOLDERS and keeps only those that exist.
    If either is set explicitly, every listed name must exist.
    """
    explicit = data_subfolders is not None or case_subfolders is not None

    if data_subfolders is None and case_subfolders is None:
        data_subfolders = DEFAULT_SUBFOLDERS
        case_subfolders = DEFAULT_SUBFOLDERS
        require_existing = False
    else:
        if data_subfolders is None:
            data_subfolders = case_subfolders
        if case_subfolders is None:
            case_subfolders = data_subfolders
        require_existing = True

    case_names, case_dirs = get_case_names(
        base_folder, case_subfolders=case_subfolders, require_existing=require_existing
    )
    data_dirs = _resolve_subfolder_paths(
        base_folder, data_subfolders, require_existing=require_existing
    )

    if not data_dirs:
        raise ValueError(
            f"No data subfolders to split under {base_folder}. "
            f"Requested: {data_subfolders}"
        )

    if explicit:
        logger = get_logger(__name__)
        if set(case_dirs) != set(data_dirs):
            logger.info(f"Case discovery subfolders: {case_dirs}")
            logger.info(f"Copy/move subfolders: {data_dirs}")

    return case_names, data_dirs


def load_train_test_from_csv(
    csv_path: str,
    id_column: str = "legacy_name",
    alias_columns: Optional[list] = None,
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """
    Like load_train_test_from_csv but also maps alias column values to the same split.

    Returns (train_ids, test_ids, alias_to_canonical) where alias_to_canonical maps
    any alias value to the primary id_column value for that row.
    """
    path = Path(csv_path)
    alias_columns = alias_columns or []
    columns_needed = {id_column, "split", "row_type"} | set(alias_columns)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Split CSV is empty or has no header: {csv_path}")
        fieldnames = {h.strip() for h in reader.fieldnames}
        missing = columns_needed - fieldnames
        if missing:
            raise ValueError(
                f"CSV missing columns {sorted(missing)}. Available: {sorted(fieldnames)}"
            )

        train_cases: Set[str] = set()
        test_cases: Set[str] = set()
        alias_to_id: Dict[str, str] = {}
        id_to_split: Dict[str, str] = {}

        for row in reader:
            row_type = (row.get("row_type") or "").strip().lower()
            if row_type and row_type != "case":
                continue

            primary_id = (row.get(id_column) or "").strip()
            if not primary_id:
                continue

            split_label = (row.get("split") or "").strip().lower()
            if split_label not in ("train", "test"):
                continue

            if primary_id in id_to_split and id_to_split[primary_id] != split_label:
                raise ValueError(
                    f"Case '{primary_id}' has conflicting splits in CSV"
                )
            id_to_split[primary_id] = split_label

            ids_for_row = {primary_id}
            for col in alias_columns:
                alias = (row.get(col) or "").strip()
                if alias:
                    ids_for_row.add(alias)

            target = train_cases if split_label == "train" else test_cases
            for cid in ids_for_row:
                if cid in alias_to_id and alias_to_id[cid] != primary_id:
                    prev = alias_to_id[cid]
                    if id_to_split.get(prev) != split_label:
                        raise ValueError(
                            f"Alias '{cid}' maps to conflicting splits "
                            f"({prev} vs {primary_id})"
                        )
                alias_to_id[cid] = primary_id
                target.add(cid)

    overlap = train_cases & test_cases
    if overlap:
        raise ValueError(f"IDs appear in both train and test: {sorted(overlap)[:10]}")

    if not train_cases and not test_cases:
        raise ValueError(f"No train/test cases found in CSV: {csv_path}")

    return train_cases, test_cases, alias_to_id


def assign_cases_from_csv_aliases(
    case_names: list,
    csv_train: Set[str],
    csv_test: Set[str],
    csv_path: str,
) -> Tuple[Set[str], Set[str]]:
    """Assign folder cases to train/test when CSV includes alias IDs (e.g. name + legacy_name)."""
    logger = get_logger(__name__)
    case_set = set(case_names)
    csv_all = csv_train | csv_test

    train_cases = case_set & csv_train
    test_cases = case_set & csv_test
    unmatched_folder = case_set - csv_all
    unmatched_csv = csv_all - case_set

    if unmatched_folder:
        logger.warning(
            f"{len(unmatched_folder)} folder case(s) not in CSV (skipped): "
            f"{sorted(unmatched_folder)[:5]}{'...' if len(unmatched_folder) > 5 else ''}"
        )
    if unmatched_csv:
        logger.info(
            f"{len(unmatched_csv)} CSV ID(s) not present as folder basenames ({csv_path})"
        )

    if not train_cases and not test_cases:
        raise ValueError(
            "No folder cases matched CSV. Try --csv-id-column legacy_name or --csv-aliases name"
        )

    return train_cases, test_cases


def _write_split_outputs(
    base_folder: str,
    existing_subfolders: list,
    train_cases: Set[str],
    test_cases: Set[str],
    output_folder: str,
    copy_files: bool,
) -> None:
    logger = get_logger(__name__)
    train_base = os.path.join(output_folder, "train")
    test_base = os.path.join(output_folder, "test")

    os.makedirs(train_base, exist_ok=True)
    os.makedirs(test_base, exist_ok=True)

    for subfolder in existing_subfolders:
        os.makedirs(os.path.join(train_base, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_base, subfolder), exist_ok=True)

    operation = "Copying" if copy_files else "Moving"
    logger.info(f"{operation} files to train/test folders...")

    for subfolder in existing_subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        files = os.listdir(subfolder_path)

        for file in files:
            if file.startswith("."):
                continue

            base_name = None
            for ext in [".mha", ".nii.gz", ".nii", ".vti", ".vtp", ".stl", ".mhd"]:
                if file.endswith(ext):
                    base_name = file.replace(ext, "")
                    break

            if base_name is None:
                logger.warning(f"Unknown file extension for {file}, skipping")
                continue

            if base_name in train_cases:
                dest_folder = os.path.join(train_base, subfolder)
            elif base_name in test_cases:
                dest_folder = os.path.join(test_base, subfolder)
            else:
                continue

            src_path = os.path.join(subfolder_path, file)
            dest_path = os.path.join(dest_folder, file)

            if copy_files:
                shutil.copy2(src_path, dest_path)
            else:
                shutil.move(src_path, dest_path)

        logger.info(f"  {operation.lower()} {subfolder}/ files...")

    train_list_path = os.path.join(output_folder, "train_cases.txt")
    test_list_path = os.path.join(output_folder, "test_cases.txt")

    with open(train_list_path, "w", encoding="utf-8") as f:
        for case in sorted(train_cases):
            f.write(f"{case}\n")

    with open(test_list_path, "w", encoding="utf-8") as f:
        for case in sorted(test_cases):
            f.write(f"{case}\n")

    logger.info(f"Saved case lists to {output_folder}/")
    logger.info(f"Split complete! Output folder: {output_folder}")


def split_dataset(
    base_folder,
    train_split=None,
    output_folder=None,
    seed=None,
    copy_files=True,
    data_subfolders=None,
    case_subfolders=None,
    split_csv=None,
    csv_id_column="legacy_name",
    csv_aliases=None,
):
    """
    Split dataset into training and testing sets.

    Use either train_split (random fraction) or split_csv (fixed assignments).
    """
    logger = get_logger(__name__)

    if split_csv is None and train_split is None:
        raise ValueError("Provide either train_split (random) or split_csv (CSV-based split)")

    case_names, data_subfolders = resolve_split_subfolders(
        base_folder,
        data_subfolders=data_subfolders,
        case_subfolders=case_subfolders,
    )
    logger.info(f"Total cases found: {len(case_names)}")

    if split_csv is not None:
        logger.info(f"Using split assignments from CSV: {split_csv}")
        csv_train, csv_test, _ = load_train_test_from_csv(
            split_csv,
            id_column=csv_id_column,
            alias_columns=csv_aliases or [],
        )
        train_cases, test_cases = assign_cases_from_csv_aliases(
            case_names, csv_train, csv_test, split_csv
        )
    else:
        if not 0 < train_split < 1:
            raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
        if seed is not None:
            random.seed(seed)
            logger.info(f"Using random seed: {seed}")

        shuffled_cases = case_names.copy()
        random.shuffle(shuffled_cases)
        n_train = int(len(shuffled_cases) * train_split)
        train_cases = set(shuffled_cases[:n_train])
        test_cases = set(shuffled_cases[n_train:])

    n_total = len(train_cases) + len(test_cases)
    if n_total:
        logger.info(
            f"Training cases: {len(train_cases)} ({len(train_cases) / n_total * 100:.1f}%)"
        )
        logger.info(
            f"Testing cases: {len(test_cases)} ({len(test_cases) / n_total * 100:.1f}%)"
        )

    if output_folder is None:
        output_folder = base_folder.rstrip("/").rstrip("\\") + "_split"

    _write_split_outputs(
        base_folder, data_subfolders, train_cases, test_cases, output_folder, copy_files
    )

    return train_cases, test_cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split dataset into training and testing sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random split: 80%% training, 20%% testing (copy files)
  python dataset_dirs/split_train_test.py --folder /path/to/data --train_split 0.8

  # Split from VMR CSV (match file basenames to legacy_name column)
  python dataset_dirs/split_train_test.py --folder /path/to/data \\
      --split-csv dataset_dirs/VMR_train_test_split.csv

  # CSV split when files use the 'name' column instead of legacy_name
  python dataset_dirs/split_train_test.py --folder /path/to/data \\
      --split-csv dataset_dirs/VMR_train_test_split.csv --csv-id-column name

  # Only split images and truths subfolders
  python dataset_dirs/split_train_test.py --folder /path/to/data --train_split 0.8 \\
      --subfolders images truths

  # Discover cases from images only, but also move centerlines and truths
  python dataset_dirs/split_train_test.py --folder /path/to/data --train_split 0.8 \\
      --case-subfolders images --subfolders images centerlines truths
        """
    )
    
    parser.add_argument('--folder', '--input_folder', '--input-folder',
                       type=str,
                       required=True,
                       help='Path to folder containing subfolders (images, centerlines, surfaces, truths, labels)')
    parser.add_argument(
        '--subfolders',
        '--data-subfolders',
        '--data_subfolders',
        type=str,
        nargs='+',
        default=None,
        metavar='NAME',
        help='Subfolder names to copy or move into train/test output (e.g. images truths). '
             'When set, each name must exist. Default: images, centerlines, surfaces, truths, labels '
             'that exist under --folder.',
    )
    parser.add_argument(
        '--case-subfolders',
        '--case_subfolders',
        type=str,
        nargs='+',
        default=None,
        metavar='NAME',
        help='Subfolders used to discover cases (intersection of basenames). '
             'Default: same as --subfolders.',
    )
    parser.add_argument('--train_split', '--train-split',
                       type=float,
                       default=None,
                       help='Fraction of data for training (0.0 to 1.0). Required unless --split-csv is set.')
    parser.add_argument('--split-csv', '--split_csv',
                       type=str,
                       default=None,
                       help='CSV with case splits (e.g. VMR_train_test_split.csv: row_type=case, split, legacy_name)')
    parser.add_argument('--csv-id-column', '--csv_id_column',
                       type=str,
                       default='legacy_name',
                       help='CSV column whose values match file basenames (default: legacy_name)')
    parser.add_argument('--csv-aliases', '--csv_aliases',
                       type=str,
                       nargs='*',
                       default=None,
                       help='Extra CSV columns to treat as alternate file basenames (e.g. name)')
    parser.add_argument('--output', '--output_folder', '--output-folder',
                       type=str,
                       default=None,
                       help='Output folder for train/test splits (default: <input_folder>_split)')
    parser.add_argument('--seed',
                       type=int,
                       default=None,
                       help='Random seed for reproducibility (optional)')
    parser.add_argument('--move',
                       action='store_true',
                       default=False,
                       help='Move files instead of copying (default: copy files)')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(__name__)
    
    # Validate input folder
    if not os.path.exists(args.folder):
        raise ValueError(f"Input folder not found: {args.folder}")
    
    if args.split_csv is None and args.train_split is None:
        raise ValueError("Provide --train_split for a random split or --split-csv for a CSV-based split")
    if args.split_csv is not None and args.train_split is not None:
        logger.warning("--train_split ignored because --split-csv was provided")
    if args.split_csv is None and not 0 < args.train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {args.train_split}")

    try:
        train_cases, test_cases = split_dataset(
            base_folder=args.folder,
            train_split=args.train_split if args.split_csv is None else None,
            output_folder=args.output,
            seed=args.seed,
            copy_files=not args.move,
            data_subfolders=args.subfolders,
            case_subfolders=args.case_subfolders,
            split_csv=args.split_csv,
            csv_id_column=args.csv_id_column,
            csv_aliases=args.csv_aliases,
        )
        
        logger.info("=" * 80)
        logger.info("Split summary:")
        logger.info(f"  Total cases: {len(train_cases) + len(test_cases)}")
        logger.info(f"  Training cases: {len(train_cases)}")
        logger.info(f"  Testing cases: {len(test_cases)}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during split: {e}")
        raise

