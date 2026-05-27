#!/usr/bin/env python3
"""
Build VMR train/test CSV from VMR_dataset_names.csv and config/vmr_splits/*.yaml.

Output: dataset_dirs/VMR_train_test_split.csv
  - All cases in VMR_dataset_names.csv
  - Split = test if Legacy Name is in vmr_split_08_all.yaml TEST_CASES, else train
  - Grouped by anatomy (vmr_split order) then modality bucket (CT, MR) like vmr_splits
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "dataset_dirs" / "VMR_dataset_names.csv"
CSV_OUT = ROOT / "dataset_dirs" / "VMR_train_test_split.csv"
SPLIT_08 = ROOT / "config" / "vmr_splits" / "vmr_split_08_all.yaml"
SPLIT_06 = ROOT / "config" / "vmr_splits" / "vmr_split_06_all_ct.yaml"
SPLIT_07 = ROOT / "config" / "vmr_splits" / "vmr_split_07_all_mr_incl_ct_mr.yaml"

ANATOMY_ORDER = [
    "Aorta",
    "Abdominal Aorta",
    "Coronary",
    "Cerebral",
    "Pulmonary",
    "Pulmonary Fontan",
    "Pulmonary Glenn",
]


def load_test_cases(yaml_path: Path) -> List[str]:
    text = yaml_path.read_text(encoding="utf-8")
    in_block = False
    out: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("TEST_CASES:"):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\s*-\s*'", line):
                m = re.search(r"'([^']+)'", line)
                if m:
                    out.append(m.group(1))
            elif line.strip() and not line.strip().startswith("#"):
                break
    return out


def modality_bucket(image_modality: str) -> str:
    m = (image_modality or "").strip()
    if m in ("CT", "CTA"):
        return "CT"
    if m in ("MR", "CT_MR"):
        return "MR"
    return "Other"


def in_ct_cohort(row: dict) -> bool:
    return modality_bucket(row.get("Image Modality", "")) == "CT"


def in_mr_cohort(row: dict) -> bool:
    return modality_bucket(row.get("Image Modality", "")) == "MR"


def anatomy_sort_key(anatomy: str) -> Tuple[int, str]:
    a = (anatomy or "").strip()
    if a in ANATOMY_ORDER:
        return (ANATOMY_ORDER.index(a), a)
    return (len(ANATOMY_ORDER), a)


def main() -> None:
    test_08 = set(load_test_cases(SPLIT_08))
    test_06 = set(load_test_cases(SPLIT_06))
    test_07 = set(load_test_cases(SPLIT_07))

    with open(CSV_IN, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_legacy: Dict[str, dict] = {}
    for r in rows:
        leg = (r.get("Legacy Name") or "").strip()
        if leg:
            by_legacy[leg] = r

    missing_test = sorted(test_08 - set(by_legacy.keys()))
    if missing_test:
        raise SystemExit(f"TEST_CASES not in VMR_dataset_names.csv: {missing_test}")

    fieldnames = [
        "row_type",
        "anatomy",
        "modality_bucket",
        "image_modality",
        "split",
        "n_train_in_group",
        "n_test_in_group",
        "legacy_name",
        "name",
        "disease",
        "species",
        "sex",
        "age",
        "in_vmr_split_06_ct_cohort",
        "in_vmr_split_07_mr_cohort",
        "in_vmr_split_06_test",
        "in_vmr_split_07_test",
    ]

    out_rows: List[dict] = []

    # Group keys: (anatomy, modality_bucket) with CT before MR before Other
    bucket_order = {"CT": 0, "MR": 1, "Other": 2}
    groups: Dict[Tuple[str, str], List[str]] = {}
    for leg, r in by_legacy.items():
        ana = (r.get("Anatomy") or "").strip() or "(unknown)"
        bucket = modality_bucket(r.get("Image Modality", ""))
        groups.setdefault((ana, bucket), []).append(leg)

    sorted_groups = sorted(
        groups.keys(),
        key=lambda k: (anatomy_sort_key(k[0]), bucket_order.get(k[1], 99), k[1]),
    )

    for ana, bucket in sorted_groups:
        legs = groups[(ana, bucket)]
        train_legs = sorted(leg for leg in legs if leg not in test_08)
        test_legs = sorted(leg for leg in legs if leg in test_08)

        out_rows.append(
            {
                "row_type": "group_header",
                "anatomy": ana,
                "modality_bucket": bucket,
                "image_modality": "",
                "split": "",
                "n_train_in_group": str(len(train_legs)),
                "n_test_in_group": str(len(test_legs)),
                "legacy_name": "",
                "name": "",
                "disease": "",
                "species": "",
                "sex": "",
                "age": "",
                "in_vmr_split_06_ct_cohort": "",
                "in_vmr_split_07_mr_cohort": "",
                "in_vmr_split_06_test": "",
                "in_vmr_split_07_test": "",
            }
        )

        for split_label, leg_list in (("test", test_legs), ("train", train_legs)):
            for leg in leg_list:
                r = by_legacy[leg]
                out_rows.append(
                    {
                        "row_type": "case",
                        "anatomy": ana,
                        "modality_bucket": bucket,
                        "image_modality": (r.get("Image Modality") or "").strip(),
                        "split": split_label,
                        "n_train_in_group": "",
                        "n_test_in_group": "",
                        "legacy_name": leg,
                        "name": (r.get("Name") or "").strip(),
                        "disease": (r.get("Disease") or "").strip(),
                        "species": (r.get("Species") or "").strip(),
                        "sex": (r.get("Sex") or "").strip(),
                        "age": (r.get("Age") or "").strip(),
                        "in_vmr_split_06_ct_cohort": "yes" if in_ct_cohort(r) else "no",
                        "in_vmr_split_07_mr_cohort": "yes" if in_mr_cohort(r) else "no",
                        "in_vmr_split_06_test": "yes" if leg in test_06 else "no",
                        "in_vmr_split_07_test": "yes" if leg in test_07 else "no",
                    }
                )

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    n_test = len(test_08)
    n_train = len(by_legacy) - n_test
    print(f"Wrote {CSV_OUT}")
    print(f"  cases: {len(by_legacy)} (train={n_train}, test={n_test})")
    print(f"  groups: {len(sorted_groups)} anatomy×modality_bucket sections")


if __name__ == "__main__":
    main()
