#!/usr/bin/env python3
"""
Compare case IDs in a dataset folder (subfolders) against a CSV.

Reports cases present in the CSV but missing on disk (per subfolder), and on-disk
cases not listed in the CSV (with which subfolders contain them).

Supports:
  - VMR_train_test_split.csv (row_type, split, legacy_name, name, ...)
  - VMR_dataset_names.csv (Legacy Name, Name, ... — no split column)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_dirs.split_train_test import DEFAULT_SUBFOLDERS, get_base_names, get_case_names

# User-facing names -> possible CSV header spellings
COLUMN_ALIASES: Dict[str, List[str]] = {
    "legacy_name": ["legacy_name", "Legacy Name"],
    "name": ["name", "Name"],
}


def _normalize_fieldnames(fieldnames: List[str]) -> Dict[str, str]:
    """Map lowercase stripped header -> actual header in file."""
    return {h.strip().lower(): h for h in fieldnames}


def resolve_csv_column(fieldnames: List[str], column: str) -> str:
    """Resolve a column argument to the exact header name in the CSV."""
    if column in fieldnames:
        return column
    key = column.strip().lower().replace(" ", "_")
    for candidate in COLUMN_ALIASES.get(key, [column]):
        if candidate in fieldnames:
            return candidate
    norm = _normalize_fieldnames(fieldnames)
    for candidate in COLUMN_ALIASES.get(key, [column]):
        hit = norm.get(candidate.strip().lower())
        if hit:
            return hit
    raise ValueError(
        f"Column '{column}' not in CSV. Available: {fieldnames}"
    )


def auto_id_column(fieldnames: List[str], preferred: Optional[str]) -> str:
    if preferred:
        return resolve_csv_column(fieldnames, preferred)
    for default in ("legacy_name", "Legacy Name", "name", "Name"):
        try:
            return resolve_csv_column(fieldnames, default)
        except ValueError:
            continue
    raise ValueError(
        f"Could not infer ID column. Specify --csv-id-column. Available: {fieldnames}"
    )


def auto_alias_columns(
    fieldnames: List[str], id_column: str, requested: Optional[List[str]]
) -> List[str]:
    if requested is not None:
        return [resolve_csv_column(fieldnames, c) for c in requested]
    aliases: List[str] = []
    for alias_key in ("name", "Name"):
        try:
            resolved = resolve_csv_column(fieldnames, alias_key)
        except ValueError:
            continue
        if resolved != id_column and resolved not in aliases:
            aliases.append(resolved)
    return aliases


def is_split_csv(fieldnames: List[str]) -> bool:
    norm = _normalize_fieldnames(fieldnames)
    return "row_type" in norm or "split" in norm


CsvCaseRow = Tuple[str, str, Set[str]]  # primary_id, split_label, match_ids


def load_csv_cases(
    csv_path: str,
    id_column: str,
    alias_columns: List[str],
) -> Tuple[List[CsvCaseRow], Set[str]]:
    """
    Load case rows from CSV.

    Returns:
        case_rows: (primary_id, split_label, match_ids) per case row
        match_ids: all strings that match folder basenames (primary + aliases)
    """
    path = Path(csv_path)
    if not path.is_file():
        raise ValueError(f"CSV not found: {csv_path}")

    case_rows: List[CsvCaseRow] = []
    match_ids: Set[str] = set()
    split_fmt = False
    row_type_col: Optional[str] = None
    split_col: Optional[str] = None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV is empty or has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        split_fmt = is_split_csv(fieldnames)
        norm = _normalize_fieldnames(fieldnames)
        if split_fmt:
            row_type_col = norm.get("row_type")
            split_col = norm.get("split")

        for row in reader:
            if split_fmt and row_type_col:
                row_type = (row.get(row_type_col) or "").strip().lower()
                if row_type and row_type != "case":
                    continue

            primary = (row.get(id_column) or "").strip()
            if not primary:
                continue

            ids = {primary}
            for col in alias_columns:
                alias = (row.get(col) or "").strip()
                if alias:
                    ids.add(alias)

            if split_fmt and split_col:
                split_label = (row.get(split_col) or "").strip().lower() or "(none)"
            else:
                split_label = "(catalog)"

            case_rows.append((primary, split_label, ids))
            match_ids.update(ids)

    if not case_rows:
        raise ValueError(f"No case rows found in CSV: {csv_path}")

    return case_rows, match_ids


def build_subfolder_cases(base_folder: str, subfolders: List[str]) -> Dict[str, Set[str]]:
    """Map subfolder name -> set of file basenames in that subfolder."""
    return {
        sf: get_base_names(os.path.join(base_folder, sf))
        for sf in subfolders
    }


def _on_disk_basename(ids: Set[str], subfolder_cases: Dict[str, Set[str]]) -> Optional[str]:
    """Return whichever CSV id appears on disk, if any."""
    on_disk = set().union(*subfolder_cases.values()) if subfolder_cases else set()
    for cid in sorted(ids):
        if cid in on_disk:
            return cid
    return None


def analyze_missing_in_folder(
    case_rows: List[CsvCaseRow],
    subfolder_cases: Dict[str, Set[str]],
    subfolders: List[str],
) -> List[Tuple[str, str, Optional[str], List[str]]]:
    """
    CSV cases missing files in one or more subfolders.

    Returns list of (primary_id, split, on_disk_basename_or_none, missing_subfolders).
    """
    results: List[Tuple[str, str, Optional[str], List[str]]] = []
    for primary, split_label, ids in case_rows:
        missing_sf = [
            sf
            for sf in subfolders
            if not (ids & subfolder_cases[sf])
        ]
        if missing_sf:
            results.append((primary, split_label, _on_disk_basename(ids, subfolder_cases), missing_sf))
    return sorted(results)


def analyze_missing_in_csv(
    subfolder_cases: Dict[str, Set[str]],
    csv_match_ids: Set[str],
) -> List[Tuple[str, List[str]]]:
    """On-disk basenames not in CSV, with subfolders where they appear."""
    by_basename: Dict[str, List[str]] = {}
    for sf, names in subfolder_cases.items():
        for basename in names:
            if basename not in csv_match_ids:
                by_basename.setdefault(basename, []).append(sf)
    return sorted((name, sorted(sfs)) for name, sfs in by_basename.items())


def format_missing_in_folder(
    primary: str,
    split_label: str,
    on_disk: Optional[str],
    missing_subfolders: List[str],
    all_subfolders: List[str],
) -> str:
    if on_disk and on_disk != primary:
        id_part = f"{primary} [{split_label}] (on disk as {on_disk})"
    else:
        id_part = f"{primary} [{split_label}]"
    if len(missing_subfolders) == len(all_subfolders):
        where = "all subfolders: " + ", ".join(missing_subfolders)
    else:
        where = "missing from: " + ", ".join(missing_subfolders)
    return f"{id_part}: {where}"


def format_missing_in_csv(basename: str, present_in: List[str]) -> str:
    return f"{basename}: present in {', '.join(present_in)}"


def print_list(title: str, items: List[str]) -> None:
    print(f"\n{title} ({len(items)}):")
    if not items:
        print("  (none)")
        return
    for item in items:
        print(f"  {item}")


def compare(
    folder: str,
    csv_path: str,
    id_column: Optional[str] = None,
    alias_columns: Optional[List[str]] = None,
    case_subfolders: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, List[str]]], List[Tuple[str, str, Optional[str], List[str]]]]:
    path = Path(csv_path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

    resolved_id = auto_id_column(fieldnames, id_column)
    resolved_aliases = auto_alias_columns(fieldnames, resolved_id, alias_columns)
    csv_fmt = "split" if is_split_csv(fieldnames) else "catalog"

    require_existing = case_subfolders is not None
    if case_subfolders is None:
        case_subfolders = DEFAULT_SUBFOLDERS

    folder_case_list, used_subfolders = get_case_names(
        folder,
        case_subfolders=case_subfolders,
        require_existing=require_existing,
    )
    folder_cases = set(folder_case_list)
    subfolder_cases = build_subfolder_cases(folder, used_subfolders)

    case_rows, csv_match_ids = load_csv_cases(
        csv_path, resolved_id, resolved_aliases
    )

    missing_in_folder = analyze_missing_in_folder(
        case_rows, subfolder_cases, used_subfolders
    )
    missing_in_csv = analyze_missing_in_csv(subfolder_cases, csv_match_ids)

    complete_csv_cases = len(case_rows) - len(missing_in_folder)
    print("=" * 60)
    print(f"Folder:     {folder}")
    print(f"Subfolders: {used_subfolders}")
    print(f"CSV:        {csv_path}")
    print(f"CSV format: {csv_fmt}")
    print(f"ID column:  {resolved_id}")
    if resolved_aliases:
        print(f"Aliases:    {resolved_aliases}")
    print("-" * 60)
    print(f"Cases on disk (intersection): {len(folder_cases)}")
    print(f"Cases in CSV:                 {len(case_rows)}")
    print(f"CSV cases complete on disk:   {complete_csv_cases}")
    print(f"CSV cases missing file(s):    {len(missing_in_folder)}")
    print(f"On-disk IDs not in CSV:       {len(missing_in_csv)}")

    print_list(
        "Missing in folder (CSV case absent from subfolder(s))",
        [
            format_missing_in_folder(p, s, on_disk, missing_sf, used_subfolders)
            for p, s, on_disk, missing_sf in missing_in_folder
        ],
    )
    print_list(
        "Missing in CSV (on disk, not listed in CSV)",
        [format_missing_in_csv(b, sfs) for b, sfs in missing_in_csv],
    )

    return missing_in_csv, missing_in_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dataset folder cases to a CSV and list mismatches.",
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Dataset root with subfolders (images, truths, ...)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Case list or split CSV (VMR_dataset_names.csv or VMR_train_test_split.csv)",
    )
    parser.add_argument(
        "--csv-id-column",
        default=None,
        help="CSV column matching file basenames (default: auto — Legacy Name or legacy_name)",
    )
    parser.add_argument(
        "--csv-aliases",
        nargs="*",
        default=None,
        help="Extra CSV columns as alternate basenames (default: auto-add Name if present)",
    )
    parser.add_argument(
        "--subfolders",
        "--case-subfolders",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Subfolders to intersect for on-disk cases (default: standard set that exists)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any case is missing on either side",
    )
    args = parser.parse_args()

    missing_in_csv, missing_in_folder = compare(
        folder=args.folder,
        csv_path=args.csv,
        id_column=args.csv_id_column,
        alias_columns=args.csv_aliases,
        case_subfolders=args.subfolders,
    )

    if args.strict and (missing_in_csv or missing_in_folder):
        sys.exit(1)


if __name__ == "__main__":
    main()
