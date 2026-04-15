#!/usr/bin/env python3
"""
Build VMR cohort YAML configs with ~20% random test holds (Legacy Name in TEST_CASES).

Requires: dataset_dirs/VMR_dataset_names.csv
Output:   config/vmr_splits/*.yaml

`vmr_split_06_all_ct` and `vmr_split_07_all_mr_incl_ct_mr` reuse the same test Legacy Names as the
matching individual splits (coronary / pulmonary / Fontan). Remaining cases (other anatomies) are
split with ~20% test per anatomy (stratified), each subgroup using a deterministic seed derived from
RNG_SEED + anatomy name, so no single anatomy dominates the leftover draw.

Re-run after CSV updates. Seed fixed at 42 for reproducibility.
"""
from __future__ import annotations

import csv
import os
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "dataset_dirs" / "VMR_dataset_names.csv"
GLOBAL_YAML = ROOT / "config" / "global.yaml"
OUT_DIR = ROOT / "config" / "vmr_splits"

RNG_SEED = 42
TEST_FRAC = 0.2
# VMR volumes on disk use this extension (not the default in config/global.yaml).
VMR_IMG_EXT = ".mha"
VMR_RADIUS_ADD = 0.01
VMR_RADIUS_SCALE = 1

ALL_ANATOMIES = [
    "Aorta",
    "Abdominal Aorta",
    "Coronary",
    "Cerebral",
    "Pulmonary",
    "Pulmonary Fontan",
    "Pulmonary Glenn",
]


def load_rows() -> List[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_ct_family(row: dict) -> bool:
    """Treat CTA like CT for cohort selection (matches VMR get_vmr_names_modality)."""
    m = (row.get("Image Modality") or "").strip()
    return m in ("CT", "CTA")


def legacy_names(rows: List[dict], pred: Callable[[dict], bool]) -> List[str]:
    out = []
    for r in rows:
        leg = (r.get("Legacy Name") or "").strip()
        if not leg:
            continue
        if pred(r):
            out.append(leg)
    return sorted(set(out))


def split_test(train_pool: List[str], seed: int = RNG_SEED, frac: float = TEST_FRAC) -> Tuple[List[str], List[str]]:
    """Return (train_ids, test_ids). Test size ~frac of pool; at least 1 test when n>=2; all train when n<=1."""
    pool = list(train_pool)
    n = len(pool)
    if n == 0:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(pool)
    if n == 1:
        return pool, []
    n_test = round(n * frac)
    n_test = max(1, n_test)
    n_test = min(n_test, n - 1)
    test_ids = sorted(pool[:n_test])
    train_ids = sorted(pool[n_test:])
    return train_ids, test_ids


def derived_seed(anatomy_label: str, bucket_tag: str) -> int:
    """Reproducible per-anatomy seed for stratified remainder splits."""
    payload = f"{RNG_SEED}|{bucket_tag}|{anatomy_label}".encode()
    h = zlib.adler32(payload) & 0x7FFFFFFF
    return RNG_SEED + (h % 1_000_003)


def split_test_stratified_by_anatomy(
    legacies: List[str],
    legacy_anatomy: Dict[str, str],
    bucket_tag: str,
    frac: float = TEST_FRAC,
) -> List[str]:
    """~frac test per anatomy subgroup (same rules as split_test within each group)."""
    by_ana: Dict[str, List[str]] = defaultdict(list)
    for leg in legacies:
        ana = legacy_anatomy.get(leg, "").strip() or "(unknown)"
        by_ana[ana].append(leg)
    test_all: List[str] = []
    for ana in sorted(by_ana.keys()):
        pool_g = sorted(by_ana[ana])
        _, test_g = split_test(pool_g, seed=derived_seed(ana, bucket_tag), frac=frac)
        test_all.extend(test_g)
    return sorted(test_all)


def legacy_anatomy_map(rows: List[dict], pred: Callable[[dict], bool]) -> Dict[str, str]:
    """First row wins for Legacy Name among rows matching pred."""
    m: Dict[str, str] = {}
    for r in rows:
        leg = (r.get("Legacy Name") or "").strip()
        if not leg or not pred(r):
            continue
        if leg not in m:
            m[leg] = (r.get("Anatomy") or "").strip() or "(unknown)"
    return m


def print_anatomy_test_summary(
    label: str,
    pool: List[str],
    test_ids: List[str],
    legacy_anatomy: Dict[str, str],
) -> None:
    """Stdout: per-anatomy pool vs test counts (sanity check for balanced coverage)."""
    pool_s = set(pool)
    test_s = set(test_ids)
    by_pool: Dict[str, int] = defaultdict(int)
    by_test: Dict[str, int] = defaultdict(int)
    for leg in pool_s:
        by_pool[legacy_anatomy.get(leg, "(unknown)")] += 1
    for leg in test_s:
        if leg in pool_s:
            by_test[legacy_anatomy.get(leg, "(unknown)")] += 1
    print(f"\n--- {label}: test counts by anatomy (pool → test) ---")
    for ana in sorted(set(by_pool.keys()) | set(by_test.keys())):
        p, t = by_pool[ana], by_test[ana]
        frac = (t / p) if p else 0.0
        print(f"  {ana!r}: pool={p} test={t} ({frac:.0%})")


def verify_individual_tests_in_aggregate(
    individual_tests: Dict[str, Set[str]],
    aggregate_tests: Set[str],
    aggregate_label: str,
) -> None:
    for name, ids in individual_tests.items():
        missing = ids - aggregate_tests
        if missing:
            raise SystemExit(f"{aggregate_label}: individual {name} has tests not in aggregate: {sorted(missing)}")


def yaml_quote_list(items: List[str], indent: int = 2) -> str:
    pad = " " * indent
    lines = [f"{pad}- '{x}'" for x in items]
    return "\n".join(lines)


def global_yaml_middle() -> str:
    """Settings from config/global.yaml (IMG_EXT … MOVE_SLOWER_BIFURC); omits ANATOMY (set per cohort header)."""
    text = GLOBAL_YAML.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    capture = False
    for line in lines:
        if line.startswith("IMG_EXT:"):
            capture = True
            continue
        if not capture:
            continue
        if line.startswith("ANATOMY:"):
            continue
        out.append(line)
        if line.startswith("MOVE_SLOWER_BIFURC:"):
            break
    return "\n".join(out) + "\n"


def apply_vmr_radius_overrides(block: str) -> str:
    """Override RADIUS_* copied from global.yaml for all VMR split configs."""
    lines_out: List[str] = []
    for line in block.splitlines():
        if line.startswith("RADIUS_ADD:"):
            lines_out.append(f"RADIUS_ADD: {VMR_RADIUS_ADD}")
        elif line.startswith("RADIUS_SCALE:"):
            lines_out.append(f"RADIUS_SCALE: {VMR_RADIUS_SCALE}")
        else:
            lines_out.append(line)
    return "\n".join(lines_out) + "\n"


def write_config(
    stem: str,
    title: str,
    anatomy: List[str],
    modality_loop: List[str],
    vmr_image_modalities: Optional[List[str]],
    test_cases: List[str],
    n_total: int,
    n_test: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}.yaml"

    vmr_line = ""
    if vmr_image_modalities is not None:
        inner = ", ".join(f"'{x}'" for x in vmr_image_modalities)
        vmr_line = f"VMR_IMAGE_MODALITIES: [{inner}]\n"

    anatomy_inner = ", ".join(f"'{a}'" for a in anatomy)
    mod_inner = ", ".join(f"'{m}'" for m in modality_loop)

    header = f"""# Auto-generated — VMR cohort split (test held out in TEST_CASES).
# {title}
# Cases in cohort: {n_total}; test: {n_test} (~{TEST_FRAC:.0%}); RNG seed {RNG_SEED}.
# Set DATA_DIR when running. Use DATASET_NAME: 'vmr'.

DATASET_NAME: 'vmr'

# DATA_DIR: '/path/to/vascular_data_3d/'

MODALITY: [{mod_inner}]
{vmr_line}ANATOMY: [{anatomy_inner}]

IMG_EXT: '{VMR_IMG_EXT}'

"""
    middle = apply_vmr_radius_overrides(global_yaml_middle())
    test_block = (
        "# Held-out test cases (Legacy Name); excluded from training when TESTING is false.\n"
        "TEST_CASES:\n"
        + yaml_quote_list(sorted(test_cases))
        + "\n\n"
        "# No default bad-case exclusions for this cohort (add if needed).\n"
        "BAD_CASES: []\n"
    )

    path.write_text(header + middle + test_block, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    rows = load_rows()
    random.seed(RNG_SEED)

    pred_coronary_ct = (
        lambda r: (r.get("Anatomy") or "").strip() == "Coronary" and is_ct_family(r)
    )
    pred_coronary_mr = (
        lambda r: (r.get("Anatomy") or "").strip() == "Coronary"
        and (r.get("Image Modality") or "").strip() == "MR"
    )
    pred_pulm_mr_ctmr = (
        lambda r: (r.get("Anatomy") or "").strip() == "Pulmonary"
        and (r.get("Image Modality") or "").strip() in ("MR", "CT_MR")
    )
    pred_pulm_ct = lambda r: (r.get("Anatomy") or "").strip() == "Pulmonary" and is_ct_family(r)
    pred_fontan_ct_mr = (
        lambda r: (r.get("Anatomy") or "").strip() == "Pulmonary Fontan"
        and (is_ct_family(r) or (r.get("Image Modality") or "").strip() == "MR")
    )
    pred_fontan_ct = (
        lambda r: (r.get("Anatomy") or "").strip() == "Pulmonary Fontan" and is_ct_family(r)
    )
    pred_fontan_mr = (
        lambda r: (r.get("Anatomy") or "").strip() == "Pulmonary Fontan"
        and (r.get("Image Modality") or "").strip() in ("MR", "CT_MR")
    )
    pred_all_ct = is_ct_family
    pred_all_mr = lambda r: (r.get("Image Modality") or "").strip() in ("MR", "CT_MR")

    pool_coronary_ct = legacy_names(rows, pred_coronary_ct)
    _, test_coronary_ct = split_test(pool_coronary_ct)

    pool_coronary_mr = legacy_names(rows, pred_coronary_mr)
    _, test_coronary_mr = split_test(pool_coronary_mr)

    pool_pulm_mr_ctmr = legacy_names(rows, pred_pulm_mr_ctmr)
    _, test_pulm_mr_ctmr = split_test(pool_pulm_mr_ctmr)

    pool_pulm_ct = legacy_names(rows, pred_pulm_ct)
    _, test_pulm_ct = split_test(pool_pulm_ct)

    pool_fontan = legacy_names(rows, pred_fontan_ct_mr)
    _, test_fontan = split_test(pool_fontan)

    pool_fontan_ct = legacy_names(rows, pred_fontan_ct)
    pool_fontan_mr = legacy_names(rows, pred_fontan_mr)
    test_fontan_ct = [x for x in test_fontan if x in set(pool_fontan_ct)]
    test_fontan_mr = [x for x in test_fontan if x in set(pool_fontan_mr)]

    pool_all_ct = legacy_names(rows, pred_all_ct)
    pool_all_mr = legacy_names(rows, pred_all_mr)

    ct_legacy_anatomy = legacy_anatomy_map(rows, pred_all_ct)
    mr_legacy_anatomy = legacy_anatomy_map(rows, pred_all_mr)

    set_cor_ct = set(pool_coronary_ct)
    set_pulm_ct = set(pool_pulm_ct)
    set_fontan_ct = set(pool_fontan_ct)
    other_ct = sorted(set(pool_all_ct) - set_cor_ct - set_pulm_ct - set_fontan_ct)
    test_other_ct = split_test_stratified_by_anatomy(other_ct, ct_legacy_anatomy, "ct_remainder")

    set_cor_mr = set(pool_coronary_mr)
    set_pulm_mr = set(pool_pulm_mr_ctmr)
    set_fontan_mr = set(pool_fontan_mr)
    other_mr = sorted(set(pool_all_mr) - set_cor_mr - set_pulm_mr - set_fontan_mr)
    test_other_mr = split_test_stratified_by_anatomy(other_mr, mr_legacy_anatomy, "mr_remainder")

    test_all_ct: List[str] = sorted(
        set(test_coronary_ct) | set(test_pulm_ct) | set(test_fontan_ct) | set(test_other_ct)
    )
    test_all_mr: List[str] = sorted(
        set(test_coronary_mr) | set(test_pulm_mr_ctmr) | set(test_fontan_mr) | set(test_other_mr)
    )

    verify_individual_tests_in_aggregate(
        {
            "coronary_ct": set(test_coronary_ct),
            "pulmonary_ct": set(test_pulm_ct),
            "fontan_ct": set(test_fontan_ct),
        },
        set(test_all_ct),
        "vmr_split_06_all_ct",
    )
    verify_individual_tests_in_aggregate(
        {
            "coronary_mr": set(test_coronary_mr),
            "pulmonary_mr_ctmr": set(test_pulm_mr_ctmr),
            "fontan_mr": set(test_fontan_mr),
        },
        set(test_all_mr),
        "vmr_split_07_all_mr_incl_ct_mr",
    )
    print_anatomy_test_summary("All CT", pool_all_ct, test_all_ct, ct_legacy_anatomy)
    print_anatomy_test_summary("All MR (+CT_MR)", pool_all_mr, test_all_mr, mr_legacy_anatomy)

    cohorts: Dict[str, Tuple[str, List[str], List[str], Optional[List[str]], List[str]]] = {
        "vmr_split_01_coronary_ct": (
            "Coronary — CT and CTA (excludes CT_MR)",
            ["Coronary"],
            ["CT"],
            None,
            test_coronary_ct,
        ),
        "vmr_split_02_coronary_mr": (
            "Coronary — Image Modality MR only",
            ["Coronary"],
            ["MR"],
            None,
            test_coronary_mr,
        ),
        "vmr_split_03_pulmonary_mr_incl_ct_mr": (
            "Pulmonary — MR and CT_MR (CT_MR counted as MR)",
            ["Pulmonary"],
            ["MR"],
            ["MR", "CT_MR"],
            test_pulm_mr_ctmr,
        ),
        "vmr_split_04_pulmonary_ct": (
            "Pulmonary — CT and CTA (excludes CT_MR)",
            ["Pulmonary"],
            ["CT"],
            None,
            test_pulm_ct,
        ),
        "vmr_split_05_pulmonary_fontan": (
            "Pulmonary Fontan — CT, CTA, and MR",
            ["Pulmonary Fontan"],
            ["MR"],
            ["CT", "MR"],
            test_fontan,
        ),
        "vmr_split_06_all_ct": (
            "All anatomies — CT and CTA; TEST = union(individual CT tests) + ~20% stratified remainder per anatomy",
            ALL_ANATOMIES,
            ["CT"],
            None,
            test_all_ct,
        ),
        "vmr_split_07_all_mr_incl_ct_mr": (
            "All anatomies — MR+CT_MR; TEST = union(individual MR tests) + ~20% stratified remainder per anatomy",
            ALL_ANATOMIES,
            ["MR"],
            ["MR", "CT_MR"],
            test_all_mr,
        ),
    }

    for stem, (title, anatomy, mod_loop, img_mods, test_ids) in cohorts.items():
        pred_map = {
            "vmr_split_01_coronary_ct": pred_coronary_ct,
            "vmr_split_02_coronary_mr": pred_coronary_mr,
            "vmr_split_03_pulmonary_mr_incl_ct_mr": pred_pulm_mr_ctmr,
            "vmr_split_04_pulmonary_ct": pred_pulm_ct,
            "vmr_split_05_pulmonary_fontan": pred_fontan_ct_mr,
            "vmr_split_06_all_ct": pred_all_ct,
            "vmr_split_07_all_mr_incl_ct_mr": pred_all_mr,
        }
        pool = legacy_names(rows, pred_map[stem])
        write_config(
            stem,
            title,
            anatomy,
            mod_loop,
            img_mods,
            test_ids,
            n_total=len(pool),
            n_test=len(test_ids),
        )

if __name__ == "__main__":
    os.chdir(ROOT)
    main()
