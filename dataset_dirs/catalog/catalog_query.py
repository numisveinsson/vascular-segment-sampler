#!/usr/bin/env python3
"""Query helpers for the SQLite dataset catalog.

Provides a uniform query surface across all catalogued datasets, filtering by
overlapping attributes (anatomy, modality, disease, ...), dataset, and split.

Examples:
    from dataset_dirs.catalog import catalog_query as cq

    # Case names with aorta or coronary anatomy across every dataset
    cq.query_cases(anatomy=["Aorta", "Coronary"])

    # CT cases that have a centerline, as pipeline-ready path dicts
    cq.query_cases_with_paths(modality="CT", has_centerline=True)

    # Test cases for a named split
    cq.query_cases(split="vmr_split_06_all_ct", role="test")

    # A DataFrame for interactive exploration
    cq.query_df(anatomy="Pulmonary")
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "catalog.db"

StrOrList = Union[str, Sequence[str]]


def connect(db_path: Union[str, Path] = DEFAULT_DB) -> sqlite3.Connection:
    """Open a connection to the catalog DB with row access by name."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _as_list(value: Optional[StrOrList]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _build_query(
    columns: str,
    anatomy: Optional[StrOrList],
    modality: Optional[StrOrList],
    disease: Optional[StrOrList],
    dataset: Optional[StrOrList],
    split: Optional[str],
    role: Optional[str],
    has_centerline: Optional[bool],
    has_surface: Optional[bool],
    has_image: Optional[bool],
    has_seg: Optional[bool],
) -> Tuple[str, list]:
    where: List[str] = []
    params: list = []

    def add_in(column: str, values: Optional[List[str]]) -> None:
        if values:
            placeholders = ", ".join("?" for _ in values)
            where.append(f"{column} IN ({placeholders})")
            params.extend(values)

    sql = f"SELECT {columns} FROM cases c JOIN datasets d ON c.dataset_id = d.id"

    if split is not None:
        sql += " JOIN splits s ON s.case_id = c.case_id AND s.dataset_id = c.dataset_id"
        where.append("s.split_name = ?")
        params.append(split)
        if role is not None:
            where.append("s.role = ?")
            params.append(role)

    add_in("c.anatomy", _as_list(anatomy))
    add_in("c.modality", _as_list(modality))
    add_in("c.disease", _as_list(disease))
    add_in("d.name", _as_list(dataset))

    for flag, col in (
        (has_centerline, "c.has_centerline"),
        (has_surface, "c.has_surface"),
        (has_image, "c.has_image"),
        (has_seg, "c.has_seg"),
    ):
        if flag is not None:
            where.append(f"{col} = ?")
            params.append(1 if flag else 0)

    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY d.name, c.case_id"
    return sql, params


def query_cases(
    anatomy: Optional[StrOrList] = None,
    modality: Optional[StrOrList] = None,
    disease: Optional[StrOrList] = None,
    dataset: Optional[StrOrList] = None,
    split: Optional[str] = None,
    role: Optional[str] = None,
    has_centerline: Optional[bool] = None,
    has_surface: Optional[bool] = None,
    has_image: Optional[bool] = None,
    has_seg: Optional[bool] = None,
    db_path: Union[str, Path] = DEFAULT_DB,
) -> List[str]:
    """Return matching case ids as a sorted list of strings."""
    sql, params = _build_query(
        "c.case_id", anatomy, modality, disease, dataset, split, role,
        has_centerline, has_surface, has_image, has_seg,
    )
    conn = connect(db_path)
    try:
        return [row["case_id"] for row in conn.execute(sql, params)]
    finally:
        conn.close()


def query_cases_with_paths(
    anatomy: Optional[StrOrList] = None,
    modality: Optional[StrOrList] = None,
    disease: Optional[StrOrList] = None,
    dataset: Optional[StrOrList] = None,
    split: Optional[str] = None,
    role: Optional[str] = None,
    has_centerline: Optional[bool] = None,
    has_surface: Optional[bool] = None,
    has_image: Optional[bool] = None,
    has_seg: Optional[bool] = None,
    db_path: Union[str, Path] = DEFAULT_DB,
) -> List[Dict[str, Optional[str]]]:
    """Return matching cases as dicts shaped like ``get_case_dict_dir`` output.

    Keys: NAME, IMAGE, SEGMENTATION, CENTERLINE, SURFACE, plus DATASET.
    """
    columns = (
        "c.case_id, c.image_path, c.seg_path, c.centerline_path, c.surface_path, d.name AS dataset_name"
    )
    sql, params = _build_query(
        columns, anatomy, modality, disease, dataset, split, role,
        has_centerline, has_surface, has_image, has_seg,
    )
    conn = connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [
        {
            "NAME": r["case_id"],
            "IMAGE": r["image_path"],
            "SEGMENTATION": r["seg_path"],
            "CENTERLINE": r["centerline_path"],
            "SURFACE": r["surface_path"],
            "DATASET": r["dataset_name"],
        }
        for r in rows
    ]


def query_df(
    anatomy: Optional[StrOrList] = None,
    modality: Optional[StrOrList] = None,
    disease: Optional[StrOrList] = None,
    dataset: Optional[StrOrList] = None,
    split: Optional[str] = None,
    role: Optional[str] = None,
    has_centerline: Optional[bool] = None,
    has_surface: Optional[bool] = None,
    has_image: Optional[bool] = None,
    has_seg: Optional[bool] = None,
    db_path: Union[str, Path] = DEFAULT_DB,
):
    """Return matching cases (full rows + dataset name) as a pandas DataFrame."""
    import pandas as pd

    columns = "c.*, d.name AS dataset_name"
    sql, params = _build_query(
        columns, anatomy, modality, disease, dataset, split, role,
        has_centerline, has_surface, has_image, has_seg,
    )
    conn = connect(db_path)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the SQLite dataset catalog.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--anatomy", nargs="*")
    parser.add_argument("--modality", nargs="*")
    parser.add_argument("--disease", nargs="*")
    parser.add_argument("--dataset", nargs="*")
    parser.add_argument("--split")
    parser.add_argument("--role", choices=["train", "test"])
    parser.add_argument("--has-centerline", dest="has_centerline", action="store_true", default=None)
    parser.add_argument("--paths", action="store_true", help="Print path dicts instead of names")
    args = parser.parse_args()

    kwargs = dict(
        anatomy=args.anatomy,
        modality=args.modality,
        disease=args.disease,
        dataset=args.dataset,
        split=args.split,
        role=args.role,
        has_centerline=args.has_centerline,
        db_path=args.db,
    )
    if args.paths:
        for d in query_cases_with_paths(**kwargs):
            print(d)
    else:
        cases = query_cases(**kwargs)
        for c in cases:
            print(c)
        print(f"\n{len(cases)} cases")
