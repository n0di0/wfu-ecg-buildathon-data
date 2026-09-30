#!/usr/bin/env python3
"""cross_vendor_pairs.py -- clinical <-> Apple Watch pair lists for the SJLIFE (Akbilgic) set.

WHAT IT BUILDS, per partition (train / val / test) from the FROZEN participant split
    positive  (label 1): a participant's clinical ECG paired with their OWN Apple Watch ECG
    negative  (label 0): a participant's clinical ECG paired with a DIFFERENT participant's watch ECG
    Columns are identical to the MIMIC pair files: ECG_ID_A, ECG_ID_B, Participant_A, Participant_B, label.
    The partition is in the file name, and the vendor pairing (clinical-watch) is implied for every row.
    Column A is always the clinical ECG and column B is always the watch ECG, so (a,b) and (b,a)
    can never both appear. Both members of every pair come from the same partition.

NEGATIVE COUNTS (defaults, both adjustable)
    train : --train-neg-per-pos 1   -> 1:1 balance, sampled, seeded (plan 8.2)
    val   : --eval-neg-per-pos 0    -> 0 means EXHAUSTIVE: every clinical x every other participant's watch
    test  : same as val
    Why exhaustive for val/test: nothing is sampled, so there is no sampling randomness in the frozen
    evaluation sets. AUROC does not depend on class balance (plan 8.2). The noise is still governed mostly by
    the number of positives (one per participant).

USAGE
    python3 cross_vendor_pairs.py --metadata ecg_akbilgic_metadata.csv \\
        --split-manifest split_manifest_sjlife.json --out-prefix ecg_akbilgic

FROZEN FILES
    val and test pair files are FROZEN once written: re-running keeps them, verifies their SHA-256 against the
    pairs manifest (V9), and only regenerates train. --force regenerates val/test too (a team decision, it
    voids every comparison made against them).

OUTPUT
    <prefix>_train_pairs.csv, <prefix>_val_pairs.csv, <prefix>_test_pairs.csv
    <prefix>_pairs_manifest.json   seed, counts, SHA-256 per file, check results, split-manifest hash

CHECKS (any FAIL stops the run before anything is written)
    S    split manifest untouched   SHA-256 of each participant list matches what the manifest recorded
    V7   pair validity              labels match participant relationships, IDs match participants and vendors,
                                    no duplicates, no reversed pairs, all members inside the partition
    V8   cap enforcement            exactly one positive per participant (cap = 1)
    V9   frozen files               val/test SHA-256 equals the recorded hash
    V10  isolation                  no participant appears in pairs of two different partitions
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
PARTITIONS = ["train", "val", "test"]
FROZEN = ["val", "test"]
COLUMNS = ["ECG_ID_A", "ECG_ID_B", "Participant_A", "Participant_B", "label"]  # identical to the MIMIC pair files
POS_CAP = 1  # one clinical + one watch recording per participant -> one positive each


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _atomic(path, writer):
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile("w", dir=path.resolve().parent, prefix=f".{path.name}.", suffix=".tmp",
                                      delete=False, newline="")
    try:
        writer(tmp)
        tmp.close()
        Path(tmp.name).replace(path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def load_inputs(metadata_path, split_manifest_path):
    meta = pd.read_csv(metadata_path, dtype=str)
    missing = {"ECG_ID", "Participant_ID", "Source"} - set(meta.columns)
    if missing:
        sys.exit(f"STOPPED: {metadata_path} is missing columns {sorted(missing)}")
    clin_df = meta[meta["Source"] == "akbilgic_clinical"]
    watch_df = meta[meta["Source"] == "akbilgic_watch"]
    if clin_df["Participant_ID"].duplicated().any() or watch_df["Participant_ID"].duplicated().any():
        sys.exit("STOPPED: a participant has more than one clinical or more than one watch recording")
    clin = dict(zip(clin_df["Participant_ID"], clin_df["ECG_ID"]))
    watch = dict(zip(watch_df["Participant_ID"], watch_df["ECG_ID"]))

    split = json.load(open(split_manifest_path))
    parts = split["participants"]
    checks = []
    for p in PARTITIONS:
        ok = sha256_ids(parts[p]) == split["sha256_sorted_ids"][p]
        checks.append({"check": f"S split manifest ({p})", "passed": ok,
                       "detail": "participant list matches its recorded SHA-256" if ok else "MISMATCH: manifest was altered"})
    in_split = set().union(*[set(parts[p]) for p in PARTITIONS])
    both = set(clin) & set(watch)
    ok = in_split == both
    checks.append({"check": "S participants match metadata", "passed": ok,
                   "detail": f"{len(in_split)} in split, {len(both)} with both recordings" + ("" if ok else " -- DIFFER")})
    return clin, watch, parts, split, checks


def make_pairs(partition, pids, clin, watch, neg_per_pos, rng):
    pids = sorted(pids)
    rows = [(clin[a], watch[a], a, a, 1) for a in pids]
    n = len(pids)
    note = ""
    if n > 1:
        k = n - 1 if neg_per_pos == 0 else min(neg_per_pos, n - 1)
        if neg_per_pos and neg_per_pos > n - 1:
            note = f"requested {neg_per_pos} negatives per positive but only {n - 1} exist; using all {n - 1}"
        for a in pids:
            others = [b for b in pids if b != a]
            chosen = others if k == n - 1 else sorted(others[i] for i in rng.choice(len(others), size=k, replace=False))
            rows.extend((clin[a], watch[b], a, b, 0) for b in chosen)
    df = pd.DataFrame(rows, columns=["ECG_ID_A", "ECG_ID_B", "Participant_A", "Participant_B", "label"])
    return df[COLUMNS], note


def check_pairs(df, partition, pids, clin, watch):
    out = []
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    same_ok = bool((pos["Participant_A"] == pos["Participant_B"]).all())
    diff_ok = bool((neg["Participant_A"] != neg["Participant_B"]).all())
    ids_ok = all(clin.get(r.Participant_A) == r.ECG_ID_A and watch.get(r.Participant_B) == r.ECG_ID_B
                 for r in df.itertuples())
    dupes = int(df.duplicated(["ECG_ID_A", "ECG_ID_B"]).sum())
    reversed_ok = set(df["ECG_ID_A"]).isdisjoint(set(df["ECG_ID_B"]))
    inside = (set(df["Participant_A"]) | set(df["Participant_B"])) <= set(pids)
    ok = same_ok and diff_ok and ids_ok and dupes == 0 and reversed_ok and inside
    out.append({"check": f"V7 pair validity ({partition})", "passed": ok,
                "detail": f"positives same-participant: {same_ok}; negatives different-participant: {diff_ok}; "
                          f"IDs match participant+vendor: {ids_ok}; duplicates: {dupes}; no reversed pairs: {reversed_ok}; "
                          f"all inside partition: {inside}"})
    per = pos.groupby("Participant_A").size()
    ok8 = bool(len(per) and per.max() <= POS_CAP and set(per.index) == set(pids))
    out.append({"check": f"V8 cap enforcement ({partition})", "passed": ok8,
                "detail": f"max positives per participant = {int(per.max()) if len(per) else 0} (cap {POS_CAP}); "
                          f"{len(per)} of {len(pids)} participants have a positive"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--out-prefix", default="ecg_akbilgic")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-neg-per-pos", type=int, default=1)
    ap.add_argument("--eval-neg-per-pos", type=int, default=0, help="0 = exhaustive")
    ap.add_argument("--force", action="store_true", help="regenerate frozen val/test files (a team decision)")
    a = ap.parse_args()

    clin, watch, parts, split, checks = load_inputs(a.metadata, a.split_manifest)
    manifest_path = Path(f"{a.out_prefix}_pairs_manifest.json")
    old = json.load(open(manifest_path)) if manifest_path.exists() else {}
    old_files = old.get("files", {})

    frames, info, notes = {}, {}, []
    for i, p in enumerate(PARTITIONS):
        pids = parts[p]
        if not pids:
            continue
        path = Path(f"{a.out_prefix}_{p}_pairs.csv")
        if p in FROZEN and path.exists() and not a.force:
            df = pd.read_csv(path, dtype=str)
            df["label"] = df["label"].astype(int)
            actual, recorded = sha256_file(path), old_files.get(p, {}).get("sha256")
            checks.append({"check": f"V9 frozen file ({p})", "passed": recorded is not None and recorded == actual,
                           "detail": "SHA-256 equals the recorded hash" if recorded == actual else
                                     ("no recorded hash in the pairs manifest" if recorded is None else "HASH CHANGED since it was frozen")})
            info[p] = {"regenerated": False, "neg_per_pos": old_files.get(p, {}).get("neg_per_pos")}
        else:
            npp = a.train_neg_per_pos if p == "train" else a.eval_neg_per_pos
            df, note = make_pairs(p, pids, clin, watch, npp, np.random.default_rng([a.seed, i]))
            if note:
                notes.append(f"{p}: {note}")
            info[p] = {"regenerated": True, "neg_per_pos": int(npp)}
        frames[p] = df
        checks.extend(check_pairs(df, p, pids, clin, watch))

    used = {p: set(frames[p]["Participant_A"]) | set(frames[p]["Participant_B"]) for p in frames}
    names = list(frames)
    bad = {f"{x}&{y}": sorted(used[x] & used[y])[:3] for j, x in enumerate(names) for y in names[j + 1:] if used[x] & used[y]}
    checks.append({"check": "V10 cross-vendor isolation", "passed": not bad,
                   "detail": "no participant appears in pairs of two partitions" if not bad else f"OVERLAP {bad}"})

    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    for n in notes:
        print(f"[NOTE] {n}")
    if not all(c["passed"] for c in checks):
        sys.exit("STOPPED: at least one check FAILED, nothing was written.")

    files = {}
    for p, df in frames.items():
        path = Path(f"{a.out_prefix}_{p}_pairs.csv")
        if info[p]["regenerated"]:
            _atomic(path, lambda f, d=df: d.to_csv(f, index=False))
        files[p] = {"file": path.name, "sha256": sha256_file(path), "n_pairs": int(len(df)),
                    "n_pos": int((df["label"] == 1).sum()), "n_neg": int((df["label"] == 0).sum()),
                    "neg_per_pos": info[p]["neg_per_pos"], "frozen": p in FROZEN}
    manifest = {"seed": int(a.seed), "split_manifest_file": Path(a.split_manifest).name,
                "split_manifest_sha256": sha256_file(a.split_manifest), "files": files,
                "checks": checks, "notes": notes}
    _atomic(manifest_path, lambda f: json.dump(manifest, f, indent=2))

    print()
    for p, m in files.items():
        print(f"{p:5s} {m['n_pairs']:6d} pairs  ({m['n_pos']} pos / {m['n_neg']} neg)  "
              f"{'regenerated' if info[p]['regenerated'] else 'kept (frozen)'}  sha256 {m['sha256'][:16]}...")
    print(f"\nwrote {manifest_path}   manifest sha256: {sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
