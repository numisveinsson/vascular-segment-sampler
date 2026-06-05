#!/usr/bin/env python3
"""Scan vascular datasets into a single SQLite catalog.

Reads dataset descriptors from ``catalog_sources.yaml``, scans each dataset's
on-disk artifacts (images / truths / centerlines / surfaces), fills metadata
(joined from ``VMR_dataset_names.csv`` for VMR, or descriptor defaults for
``other`` datasets), and upserts everything into ``catalog.db``.

Usage:
    python -m dataset_dirs.catalog.build_catalog
    python -m dataset_dirs.catalog.build_catalog --splits
    python -m dataset_dirs.catalog.build_catalog --sources path/to/sources.yaml --db path/to/catalog.db
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from dataset_dirs.datasets import (
    directories,
    get_vmr_dataset_names_local,
    resolve_case_surface_path,
)
from dataset_dirs.generate_vmr_train_test_csv import load_test_cases, modality_bucket

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SOURCES = HERE / "catalog_sources.yaml"
DEFAULT_DB = HERE / "catalog.db"
SCHEMA = HERE / "schema.sql"
VMR_SPLITS_DIR = ROOT / "config" / "vmr_splits"


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open the DB and ensure the schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def _list_cases(folder: str, suffixes: List[str]) -> Dict[str, str]:
    """Map case_id -> filename for files in ``folder`` ending with any suffix.

    The longest matching suffix is stripped so multi-dot extensions like
    ``.nii.gz`` are handled before single-dot ones.
    """
    out: Dict[str, str] = {}
    if not os.path.isdir(folder):
        return out
    ordered = sorted(suffixes, key=len, reverse=True)
    for fn in os.listdir(folder):
        if fn.startswith("."):
            continue
        for suf in ordered:
            if fn.endswith(suf):
                case = fn[: -len(suf)]
                # surfaces may carry a trailing .seg before the extension
                if case.endswith(".seg"):
                    case = case[: -len(".seg")]
                out.setdefault(case, fn)
                break
    return out


def scan_dataset_cases(root: str, img_ext: str) -> Dict[str, dict]:
    """Discover cases under ``root`` and the artifacts each one has.

    Returns case_id -> dict with has_* flags and resolved paths.
    """
    root = _expand(root)
    images = _list_cases(os.path.join(root, "images"), [img_ext])
    truths = _list_cases(os.path.join(root, "truths"), [img_ext])
    centerlines = _list_cases(os.path.join(root, "centerlines"), [".vtp"])
    surfaces = _list_cases(os.path.join(root, "surfaces"), [".vtp", ".stl"])

    all_cases = set(images) | set(truths) | set(centerlines) | set(surfaces)

    result: Dict[str, dict] = {}
    for case in sorted(all_cases):
        dir_image, dir_seg, dir_cent, dir_surf = directories(root, case, img_ext)
        surf_path = resolve_case_surface_path(root, case)
        result[case] = {
            "has_image": int(case in images),
            "has_seg": int(case in truths),
            "has_centerline": int(case in centerlines),
            "has_surface": int(case in surfaces),
            "image_path": dir_image if case in images else None,
            "seg_path": dir_seg if case in truths else None,
            "centerline_path": dir_cent if case in centerlines else None,
            "surface_path": surf_path if case in surfaces else None,
        }
    return result


def _normalize_modality(image_modality: Optional[str]) -> Optional[str]:
    if not image_modality:
        return None
    bucket = modality_bucket(image_modality)
    return bucket if bucket != "Other" else (image_modality.strip() or None)


def _vmr_metadata_by_legacy() -> Dict[str, dict]:
    """legacy_name -> metadata dict from VMR_dataset_names.csv."""
    df = get_vmr_dataset_names_local()
    meta: Dict[str, dict] = {}
    for _, row in df.iterrows():
        legacy = str(row.get("Legacy Name", "")).strip()
        if not legacy or legacy.lower() == "nan":
            continue
        image_modality = str(row.get("Image Modality", "")).strip()
        meta[legacy] = {
            "anatomy": (str(row.get("Anatomy", "")).strip() or None),
            "modality": _normalize_modality(image_modality),
            "disease": (str(row.get("Disease", "")).strip() or None),
            "sex": (str(row.get("Sex", "")).strip() or None),
            "age": (str(row.get("Age", "")).strip() or None),
            "species": (str(row.get("Species", "")).strip() or None),
            "source_meta": {
                "name": (str(row.get("Name", "")).strip() or None),
                "image_modality_raw": image_modality or None,
                "procedure": (str(row.get("Procedure", "")).strip() or None),
            },
        }
    return meta


# Catalog case columns that a metadata CSV column_map may target directly.
KNOWN_META_FIELDS = ("anatomy", "modality", "disease", "sex", "age", "species")


def _resolve_metadata_csv(csv_path: str, root: str) -> str:
    """Resolve a metadata_csv path: absolute, ~-expanded, or relative to root."""
    expanded = os.path.expanduser(csv_path)
    if os.path.isabs(expanded):
        return expanded
    rooted = os.path.join(_expand(root), expanded)
    if os.path.exists(rooted):
        return rooted
    return os.path.abspath(expanded)


def _normalize_split_role(value: Optional[str], split_value_map: Optional[dict]) -> Optional[str]:
    """Map a raw CSV split value to 'train'/'test', or None if unrecognized."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if split_value_map:
        mapped = split_value_map.get(v, split_value_map.get(v.lower()))
        if mapped in ("train", "test"):
            return mapped
    low = v.lower()
    if low.startswith("train"):
        return "train"
    if low.startswith("test"):
        return "test"
    return None


def _load_metadata_csv(
    csv_path: str, column_map: dict, split_value_map: Optional[dict] = None
) -> Dict[str, dict]:
    """Load a per-dataset metadata CSV keyed by case id.

    ``column_map`` maps catalog fields to CSV column names. It must include
    ``case_id``; it may include any of KNOWN_META_FIELDS plus ``split`` (a column
    whose value indicates train/test). Any CSV column not referenced by the map
    is preserved in ``source_meta``.

    Returns case_id -> {"fields": {...}, "split_role": "train"/"test"/None,
                         "source_meta": {...}}.
    """
    import pandas as pd

    case_col = column_map.get("case_id")
    if not case_col:
        raise ValueError("column_map must include a 'case_id' entry")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if case_col not in df.columns:
        raise ValueError(
            f"case_id column '{case_col}' not in CSV; available columns: {list(df.columns)}"
        )

    mapped_cols = set(column_map.values())
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        case_id = str(row[case_col]).strip()
        if not case_id:
            continue
        entry = {"fields": {}, "split_role": None, "source_meta": {}}
        for field, col in column_map.items():
            if field == "case_id" or col not in df.columns:
                continue
            val = str(row[col]).strip()
            if field == "split":
                entry["split_role"] = _normalize_split_role(val, split_value_map)
            elif field in KNOWN_META_FIELDS:
                if val:
                    entry["fields"][field] = val
            elif val:
                entry["source_meta"][field] = val
        for col in df.columns:
            if col not in mapped_cols:
                val = str(row[col]).strip()
                if val:
                    entry["source_meta"][col] = val
        out[case_id] = entry
    return out


def upsert_split_rows(
    conn: sqlite3.Connection,
    split_name: str,
    dataset_id: int,
    assignments: List[tuple],
) -> None:
    """Replace this dataset's rows for split_name with the given (case_id, role)."""
    conn.execute(
        "DELETE FROM splits WHERE split_name = ? AND dataset_id = ?",
        (split_name, dataset_id),
    )
    for case_id, role in assignments:
        conn.execute(
            """
            INSERT INTO splits (split_name, dataset_id, case_id, role)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(split_name, dataset_id, case_id) DO UPDATE SET role=excluded.role
            """,
            (split_name, dataset_id, case_id, role),
        )


def upsert_dataset(conn: sqlite3.Connection, source: dict) -> int:
    """Insert/update a dataset row and return its id."""
    conn.execute(
        """
        INSERT INTO datasets (name, root_path, img_ext, source_type, notes)
        VALUES (:name, :root_path, :img_ext, :source_type, :notes)
        ON CONFLICT(name) DO UPDATE SET
            root_path=excluded.root_path,
            img_ext=excluded.img_ext,
            source_type=excluded.source_type,
            notes=excluded.notes
        """,
        {
            "name": source["name"],
            "root_path": _expand(source["root_path"]),
            "img_ext": source["img_ext"],
            "source_type": source["source_type"],
            "notes": source.get("notes"),
        },
    )
    row = conn.execute(
        "SELECT id FROM datasets WHERE name = ?", (source["name"],)
    ).fetchone()
    return int(row[0])


def upsert_case(conn: sqlite3.Connection, dataset_id: int, case_id: str, fields: dict) -> None:
    payload = {
        "dataset_id": dataset_id,
        "case_id": case_id,
        "anatomy": fields.get("anatomy"),
        "modality": fields.get("modality"),
        "disease": fields.get("disease"),
        "sex": fields.get("sex"),
        "age": fields.get("age"),
        "species": fields.get("species"),
        "has_image": fields.get("has_image", 0),
        "has_seg": fields.get("has_seg", 0),
        "has_centerline": fields.get("has_centerline", 0),
        "has_surface": fields.get("has_surface", 0),
        "image_path": fields.get("image_path"),
        "seg_path": fields.get("seg_path"),
        "centerline_path": fields.get("centerline_path"),
        "surface_path": fields.get("surface_path"),
        "source_meta": json.dumps(fields["source_meta"]) if fields.get("source_meta") else None,
    }
    conn.execute(
        """
        INSERT INTO cases (
            dataset_id, case_id, anatomy, modality, disease, sex, age, species,
            has_image, has_seg, has_centerline, has_surface,
            image_path, seg_path, centerline_path, surface_path, source_meta
        ) VALUES (
            :dataset_id, :case_id, :anatomy, :modality, :disease, :sex, :age, :species,
            :has_image, :has_seg, :has_centerline, :has_surface,
            :image_path, :seg_path, :centerline_path, :surface_path, :source_meta
        )
        ON CONFLICT(dataset_id, case_id) DO UPDATE SET
            anatomy=excluded.anatomy,
            modality=excluded.modality,
            disease=excluded.disease,
            sex=excluded.sex,
            age=excluded.age,
            species=excluded.species,
            has_image=excluded.has_image,
            has_seg=excluded.has_seg,
            has_centerline=excluded.has_centerline,
            has_surface=excluded.has_surface,
            image_path=excluded.image_path,
            seg_path=excluded.seg_path,
            centerline_path=excluded.centerline_path,
            surface_path=excluded.surface_path,
            source_meta=excluded.source_meta
        """,
        payload,
    )


def ingest_dataset(conn: sqlite3.Connection, source: dict) -> int:
    """Scan one dataset and upsert all of its cases. Returns case count."""
    name = source["name"]
    root = source["root_path"]
    img_ext = source["img_ext"]
    source_type = source["source_type"]

    if not os.path.isdir(_expand(root)):
        print(f"[skip] {name}: root not found ({_expand(root)})")
        return 0

    dataset_id = upsert_dataset(conn, source)
    scanned = scan_dataset_cases(root, img_ext)

    vmr_meta: Dict[str, dict] = {}
    if source_type == "vmr":
        try:
            vmr_meta = _vmr_metadata_by_legacy()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {name}: could not load VMR metadata CSV: {e}")

    # Optional per-dataset metadata CSV (applied on top of vmr/defaults).
    csv_meta: Dict[str, dict] = {}
    meta_csv = source.get("metadata_csv")
    if meta_csv:
        column_map = source.get("column_map") or {}
        try:
            csv_path = _resolve_metadata_csv(meta_csv, root)
            csv_meta = _load_metadata_csv(csv_path, column_map, source.get("split_value_map"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {name}: could not load metadata_csv: {e}")

    matched_meta = 0
    matched_csv = 0
    split_assignments: List[tuple] = []
    for case_id, artifacts in scanned.items():
        fields = dict(artifacts)
        if source_type == "vmr":
            meta = vmr_meta.get(case_id)
            if meta:
                matched_meta += 1
                fields.update(
                    {
                        "anatomy": meta["anatomy"],
                        "modality": meta["modality"],
                        "disease": meta["disease"],
                        "sex": meta["sex"],
                        "age": meta["age"],
                        "species": meta["species"],
                        "source_meta": meta["source_meta"],
                    }
                )
        else:
            fields["anatomy"] = source.get("anatomy")
            fields["modality"] = source.get("modality")

        # CSV metadata overrides any value it explicitly provides.
        cmeta = csv_meta.get(case_id)
        if cmeta:
            matched_csv += 1
            fields.update(cmeta["fields"])
            if cmeta["source_meta"]:
                merged = dict(fields.get("source_meta") or {})
                merged.update(cmeta["source_meta"])
                fields["source_meta"] = merged
            if cmeta["split_role"]:
                split_assignments.append((case_id, cmeta["split_role"]))

        upsert_case(conn, dataset_id, case_id, fields)

    if split_assignments:
        split_name = source.get("split_name") or f"{name}_csv"
        upsert_split_rows(conn, split_name, dataset_id, split_assignments)

    conn.commit()
    extras = []
    if source_type == "vmr":
        extras.append(f"{matched_meta} with VMR metadata")
    if meta_csv:
        extras.append(f"{matched_csv} matched metadata_csv")
    if split_assignments:
        split_name = source.get("split_name") or f"{name}_csv"
        n_test = sum(1 for _, r in split_assignments if r == "test")
        extras.append(f"split '{split_name}' ({n_test} test / {len(split_assignments) - n_test} train)")
    suffix = (", " + ", ".join(extras)) if extras else ""
    print(f"[ok]   {name}: {len(scanned)} cases{suffix}")
    return len(scanned)


def ingest_splits(conn: sqlite3.Connection) -> int:
    """Parse config/vmr_splits/*.yaml TEST_CASES into the splits table.

    Each split file becomes a split_name; its TEST_CASES are 'test', all other
    catalogued cases are 'train'. case_id is matched across all datasets, so
    dataset_id is resolved per matching case.
    """
    if not VMR_SPLITS_DIR.is_dir():
        print(f"[skip] splits: {VMR_SPLITS_DIR} not found")
        return 0

    # case_id -> dataset_id (first match wins; case ids are globally unique here)
    case_to_dataset: Dict[str, int] = {}
    for case_id, dataset_id in conn.execute("SELECT case_id, dataset_id FROM cases"):
        case_to_dataset.setdefault(case_id, dataset_id)

    total = 0
    for yaml_path in sorted(glob.glob(str(VMR_SPLITS_DIR / "*.yaml"))):
        split_name = Path(yaml_path).stem
        test_cases = set(load_test_cases(Path(yaml_path)))
        conn.execute("DELETE FROM splits WHERE split_name = ?", (split_name,))
        n = 0
        for case_id, dataset_id in case_to_dataset.items():
            role = "test" if case_id in test_cases else "train"
            conn.execute(
                """
                INSERT INTO splits (split_name, dataset_id, case_id, role)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(split_name, dataset_id, case_id) DO UPDATE SET role=excluded.role
                """,
                (split_name, dataset_id, case_id, role),
            )
            n += 1
        total += n
        print(f"[ok]   split '{split_name}': {len(test_cases)} test cases listed, {n} cases tagged")
    conn.commit()
    return total


def load_sources(sources_path: Path) -> List[dict]:
    data = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    datasets = data.get("datasets", []) if isinstance(data, dict) else data
    if not datasets:
        raise SystemExit(f"No datasets found in {sources_path}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SQLite dataset catalog.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="Path to catalog_sources.yaml")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Output SQLite DB path")
    parser.add_argument("--splits", action="store_true", help="Also ingest config/vmr_splits/*.yaml into the splits table")
    args = parser.parse_args()

    sources = load_sources(args.sources)
    conn = init_db(args.db)
    try:
        total = 0
        for source in sources:
            total += ingest_dataset(conn, source)
        print(f"\nCatalogued {total} cases across {len(sources)} dataset descriptor(s).")
        if args.splits:
            ingest_splits(conn)
        print(f"\nDatabase written to {args.db}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
