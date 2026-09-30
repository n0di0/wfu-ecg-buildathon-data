#!/usr/bin/env python3
"""sjlife_splitter.py -- deterministic PARTICIPANT-level split for the SJLIFE (Akbilgic) paired set.

WHY
    Same rule as MIMIC: split by participant, never by recording. Every SJLIFE participant has
    exactly one clinical and one Apple Watch ECG. Both must land in the same partition, otherwise
    a cross-vendor positive pair straddles train and eval (team plan V10).

USAGE
    python3 sjlife_splitter.py --metadata ecg_akbilgic_metadata.csv \\
        --train-frac 0.5 --val-frac 0.25 --test-frac 0.25

    The fractions have NO defaults on purpose. They are a decision, and the manifest records them.
    They must sum to 1. Use --test-frac 0 for a two-way split (train / val).

OUTPUT
    split_manifest_sjlife.json   FROZEN: the script refuses to overwrite it unless you pass --force.
                                 Its SHA-256 is printed at the end -- put that in your log.

CHECKS (each prints PASS/FAIL; any FAIL stops the run before anything is written)
    V1 inventory     every participant has exactly one clinical + one watch record
    V2 determinism   re-running the split gives identical partitions (SHA-256 of sorted IDs)
    V3 disjointness  no participant in two partitions
    V4 completeness  every participant lands in a partition
    V5 proportions   each partition within +/-1 percentage point of its target
    V6 containment   every ECG_ID resolves to its participant's partition; each partition holds
                     as many clinical as watch recordings

NOTES
    - Participant IDs are SORTED before shuffling, so the result does not depend on row order in the CSV.
    - Seed 42, the same frozen seed as the MIMIC split.
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
DEFAULT_EXCLUSION_NOTE = (
    "Excluded pending confirmation from the dataset authors: CSV sudoIDs 131, 210, 232 have no files on disk; "
    "files 209, 231, 376 exist on disk with no CSV row. No mapping was guessed."
)


def sha256_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def split_participants(participant_ids, fractions, seed=SEED):
    ids = np.array(sorted(set(participant_ids)))
    shuffled = ids[np.random.default_rng(seed).permutation(len(ids))]
    n = len(shuffled)
    active = [p for p in PARTITIONS if fractions[p] > 0]
    parts, start = {p: [] for p in PARTITIONS}, 0
    for i, p in enumerate(active):
        # the last active partition takes the remainder, so rounding never drops or invents a participant
        size = n - start if i == len(active) - 1 else int(round(n * fractions[p]))
        parts[p] = sorted(str(x) for x in shuffled[start:start + size])
        start += size
    return parts


def run_checks(meta, parts, fractions, seed):
    results = []

    def add(name, ok, detail):
        results.append({"check": name, "passed": bool(ok), "detail": detail})

    n = meta["Participant_ID"].nunique()
    per = meta.groupby("Participant_ID")["Source"].apply(lambda s: sorted(s))
    bad = [pid for pid, srcs in per.items() if srcs != ["akbilgic_clinical", "akbilgic_watch"]]
    add("V1 inventory", n > 0 and not bad, f"{n} participants; {len(bad)} without exactly one clinical + one watch")

    again = split_participants(meta["Participant_ID"], fractions, seed)
    same = all(sha256_ids(again[p]) == sha256_ids(parts[p]) for p in PARTITIONS)
    add("V2 determinism", same, "re-run produced identical partitions" if same else "re-run DIFFERED")

    overlaps = {f"{a}&{b}": sorted(set(parts[a]) & set(parts[b]))
                for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:]}
    overlaps = {k: v for k, v in overlaps.items() if v}
    add("V3 disjointness", not overlaps, "no participant in two partitions" if not overlaps else f"OVERLAP {overlaps}")

    union = set().union(*[set(parts[p]) for p in PARTITIONS])
    everyone = set(meta["Participant_ID"])
    add("V4 completeness", union == everyone,
        f"{len(union)} assigned of {len(everyone)}" + ("" if union == everyone else f"; missing {sorted(everyone - union)[:5]}"))

    diffs = {p: abs(len(parts[p]) / n - fractions[p]) for p in PARTITIONS}
    add("V5 proportions", all(d <= 0.01 for d in diffs.values()),
        ", ".join(f"{p} {len(parts[p])} ({len(parts[p]) / n:.1%} vs {fractions[p]:.1%})" for p in PARTITIONS))

    where = {pid: p for p in PARTITIONS for pid in parts[p]}
    m = meta.assign(_part=meta["Participant_ID"].map(where))
    unresolved = int(m["_part"].isna().sum())
    balanced = all(
        (g["Source"] == "akbilgic_clinical").sum() == (g["Source"] == "akbilgic_watch").sum()
        for _, g in m.groupby("_part"))
    add("V6 containment", unresolved == 0 and balanced,
        f"{unresolved} ECG_IDs unresolved; clinical/watch counts equal in every partition: {balanced}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="ecg_akbilgic_metadata.csv from ingest.py")
    ap.add_argument("--train-frac", type=float, required=True)
    ap.add_argument("--val-frac", type=float, required=True)
    ap.add_argument("--test-frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="split_manifest_sjlife.json")
    ap.add_argument("--exclusion-note", default=DEFAULT_EXCLUSION_NOTE)
    ap.add_argument("--force", action="store_true", help="overwrite a frozen manifest (a team decision)")
    a = ap.parse_args()

    fractions = {"train": a.train_frac, "val": a.val_frac, "test": a.test_frac}
    if min(fractions.values()) < 0 or abs(sum(fractions.values()) - 1.0) > 1e-9:
        sys.exit(f"STOPPED: fractions must be >= 0 and sum to 1 (got {fractions}, sum {sum(fractions.values())})")
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"STOPPED: {out} already exists and is FROZEN. Use --force only after a team decision.")

    meta = pd.read_csv(a.metadata, dtype=str)
    missing = {"Participant_ID", "Source", "ECG_ID"} - set(meta.columns)
    if missing:
        sys.exit(f"STOPPED: {a.metadata} is missing columns {sorted(missing)}")
    if meta["ECG_ID"].duplicated().any():
        sys.exit("STOPPED: duplicate ECG_ID values in the metadata")

    parts = split_participants(meta["Participant_ID"], fractions, a.seed)
    checks = run_checks(meta, parts, fractions, a.seed)
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if not all(c["passed"] for c in checks):
        sys.exit("STOPPED: at least one check FAILED, nothing was written.")

    manifest = {
        "dataset": "SJLIFE / Akbilgic paired clinical + Apple Watch",
        "seed": int(a.seed),
        "fractions": {p: float(fractions[p]) for p in PARTITIONS},
        "n_participants": int(meta["Participant_ID"].nunique()),
        "n_ecgs": int(len(meta)),
        "counts": {p: int(len(parts[p])) for p in PARTITIONS},
        "sha256_sorted_ids": {p: sha256_ids(parts[p]) for p in PARTITIONS},
        "source_metadata_file": Path(a.metadata).name,
        "source_metadata_sha256": sha256_file(a.metadata),
        "checks": checks,
        "exclusion_note": a.exclusion_note,
        "participants": parts,
    }
    tmp = tempfile.NamedTemporaryFile("w", dir=out.resolve().parent, prefix=f".{out.name}.", suffix=".tmp", delete=False)
    try:
        json.dump(manifest, tmp, indent=2)
        tmp.close()
        Path(tmp.name).replace(out)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    print(f"\nwrote {out}  counts: {manifest['counts']}")
    print(f"manifest sha256: {sha256_file(out)}   <- put this in your log; it is the freeze hash")


if __name__ == "__main__":
    main()
