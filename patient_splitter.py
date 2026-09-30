"""
patient_splitter.py

Splits eligible patients into train/val/test by subject_id (never by
recording), per the team plan's non-negotiable rule (Team Plan §8).

Includes verification checks V1, V3-V6 from the plan's chain of
verification (§8.3). V7-V11 belong to pair generation / label-permutation
probing, which happen in a later script.

Usage:
    python3 patient_splitter.py --eligible eligible_records_downloaded.csv \
        --out split_manifest.json

Input file requirements:
    A CSV with at least a `subject_id` column and, ideally, a `study_id`
    column (used for V6). One row per eligible record (recording), not
    per patient -- the script does the patient-level dedup itself.
"""

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

SEED = 42  # fixed forever -- "freeze it day one" per the team plan


def split_patients(df: pd.DataFrame, seed: int = SEED):
    patient_ids = df["subject_id"].astype(str).unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # test gets whatever's left, so rounding never drops/duplicates a patient

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train:n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def sha256_of_sorted_ids(ids: set) -> str:
    """V2-style hash of a sorted ID list, for reproducibility checks on
    later re-runs. Not required to pass on a first run, but write it down
    now so a future re-run can be checked against it."""
    joined = ",".join(sorted(ids))
    return hashlib.sha256(joined.encode()).hexdigest()


def run_checks(df: pd.DataFrame, train_ids: set, val_ids: set, test_ids: set):
    results = {}

    # V1 -- Patient inventory
    all_patients = set(df["subject_id"].astype(str).unique())
    n_elig = len(all_patients)
    results["V1_patient_inventory"] = {
        "pass": n_elig > 0,
        "Nelig": n_elig,
    }
    print(f"[V1] Nelig = {n_elig}  -> {'PASS' if n_elig > 0 else 'FAIL'}")

    # V3 -- Disjointness
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    v3_pass = len(tv) == 0 and len(tt) == 0 and len(vt) == 0
    results["V3_disjointness"] = {
        "pass": v3_pass,
        "train_val_overlap": len(tv),
        "train_test_overlap": len(tt),
        "val_test_overlap": len(vt),
    }
    print(f"[V3] Disjointness -> {'PASS' if v3_pass else 'FAIL'}")
    if not v3_pass:
        print(f"      train∩val={len(tv)} train∩test={len(tt)} val∩test={len(vt)}")

    # V4 -- Completeness
    union = train_ids | val_ids | test_ids
    missing = all_patients - union
    v4_pass = len(missing) == 0
    results["V4_completeness"] = {
        "pass": v4_pass,
        "missing_count": len(missing),
    }
    print(f"[V4] Completeness -> {'PASS' if v4_pass else 'FAIL'} "
          f"({len(missing)} patients missing from all partitions)")

    # V5 -- Proportions (within ±1 percentage point of 70/15/15)
    n = len(all_patients)
    pct_train = 100 * len(train_ids) / n
    pct_val = 100 * len(val_ids) / n
    pct_test = 100 * len(test_ids) / n
    v5_pass = (abs(pct_train - 70) <= 1 and abs(pct_val - 15) <= 1
               and abs(pct_test - 15) <= 1)
    results["V5_proportions"] = {
        "pass": v5_pass,
        "pct_train": round(pct_train, 2),
        "pct_val": round(pct_val, 2),
        "pct_test": round(pct_test, 2),
    }
    print(f"[V5] Proportions -> train={pct_train:.2f}% val={pct_val:.2f}% "
          f"test={pct_test:.2f}% -> {'PASS' if v5_pass else 'FAIL'}")

    # V6 -- Recording containment (only if study_id column exists)
    if "study_id" in df.columns:
        df = df.copy()
        df["subject_id"] = df["subject_id"].astype(str)
        df["partition"] = np.select(
            [df["subject_id"].isin(train_ids),
             df["subject_id"].isin(val_ids),
             df["subject_id"].isin(test_ids)],
            ["train", "val", "test"],
            default="MISSING",
        )
        bad_rows = df[df["partition"] == "MISSING"]
        # every study_id should appear in exactly one partition, consistent
        # with its subject_id's assigned partition (guaranteed by
        # construction here, but we check for stray/duplicate study_ids)
        dup_study_ids = int(df["study_id"].duplicated().sum())
        v6_pass = bool(len(bad_rows) == 0 and dup_study_ids == 0)
        results["V6_recording_containment"] = {
            "pass": v6_pass,
            "orphaned_recordings": len(bad_rows),
            "duplicate_study_ids": int(dup_study_ids),
        }
        print(f"[V6] Recording containment -> {'PASS' if v6_pass else 'FAIL'} "
              f"(orphaned={len(bad_rows)}, dup study_ids={dup_study_ids})")
    else:
        print("[V6] Skipped -- no 'study_id' column in input file")
        results["V6_recording_containment"] = {"pass": None, "note": "skipped, no study_id column"}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligible", required=True,
                     help="eligible_records_downloaded.csv (or theoretical eligible_records.csv)")
    ap.add_argument("--out", default="split_manifest.json",
                     help="output path for the split manifest")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df = pd.read_csv(args.eligible, dtype=str)
    if "subject_id" not in df.columns:
        print(f"!! No 'subject_id' column found. Columns present: {list(df.columns)}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {args.eligible}")
    train_ids, val_ids, test_ids = split_patients(df, seed=args.seed)

    print(f"\nSplit sizes -> train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}\n")

    print("Running verification checks V1, V3-V6...")
    results = run_checks(df, train_ids, val_ids, test_ids)

    all_pass = all(v.get("pass") in (True, None) for v in results.values())
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'ONE OR MORE CHECKS FAILED -- do not ship this split'}")

    manifest = {
        "seed": args.seed,
        "source_file": args.eligible,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "hashes": {
            "train_sha256": sha256_of_sorted_ids(train_ids),
            "val_sha256": sha256_of_sorted_ids(val_ids),
            "test_sha256": sha256_of_sorted_ids(test_ids),
        },
        "checks": results,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote manifest to {args.out}")
    if not all_pass:
        print("WARNING: manifest was written despite failed checks. Fix before sharing with the team.")
        sys.exit(1)


if __name__ == "__main__":
    main()
