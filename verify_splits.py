#!/usr/bin/env python3
"""verify_splits.py -- the plan's chain of verification (section 8.3), MIMIC and SJLIFE, in one script.

Run it before any training job. A run that has not passed the chain does not get a row on the leaderboard.
It reads the frozen manifests and pair files and re-checks them INDEPENDENTLY: the checks below are written from the
plan's definitions, not by calling the code that made the files. (A checker that reuses the generator's own
verification can only confirm the generator agrees with itself.)

USAGE (from the folder holding the files; every path has this default)
    python3 verify_splits.py
    python3 verify_splits.py --record-list /path/to/record_list.csv --write-ledger
    python3 verify_splits.py --skip-sjlife            # MIMIC only
    python3 verify_splits.py --v11-result v11.json    # Modeling's label-permutation result: {"val_auroc": 0.51}

CHECKS
    V1  patient inventory       counts add up, Nelig > 0
    V2  determinism             (a) each ID list still hashes to the recorded SHA-256
                                (b) re-running the splitter reproduces the recorded partitions
    V3  disjointness            no participant in two partitions
    V4  completeness            every eligible participant lands in a partition
    V5  proportions             within +/-1 percentage point of target
    V6  recording containment   every ECG resolves to a participant in one partition; no duplicate IDs
    V7  pair validity           labels match participant relationships, IDs match participants (and vendors), no
                                duplicated or reversed pairs, all members inside the partition
    V8  cap enforcement         MIMIC: per-participant positive/negative caps; SJLIFE: exactly one positive each
    V9  frozen validation       val/test pair files still match the hash recorded when they were frozen
    V10 cross-vendor isolation  SJLIFE participants appear in one partition's pairs only; no SJLIFE ECG in MIMIC pairs
    V11 label-permutation probe owned by Modeling; read from --v11-result, otherwise PENDING

STATUS   PASS / FAIL / SKIP.  SKIP means the check could not run (a file was not found); it is never counted as a pass.
EXIT     1 if anything FAILED (or, with --strict, if anything was skipped or V11 is pending), else 0.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIMIC_TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
POS_CAP = {"train": 5, "val": 2, "test": 2}    # must match pair_generator.py (cross-checked below)
NEG_CAP = {"train": 10, "val": 5, "test": 5}
PARTS = ["train", "val", "test"]
RESULTS = []


def rec(section, vid, name, status, detail):
    RESULTS.append({"section": section, "id": vid, "name": name, "status": status, "detail": detail})
    print(f"[{status}] {section} {vid} {name}: {detail}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_comma(ids):      # patient_splitter.py's scheme
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def hash_newline(ids):    # sjlife_splitter.py's scheme
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def content_hash(df):     # pair_generator.py's order-insensitive scheme
    rows = sorted(f"{a}|{b}|{l}" for a, b, l in zip(df["ECG_ID_A"], df["ECG_ID_B"], df["label"]))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def load_module(path, name):
    path = Path(path)
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_pairs(path):
    df = pd.read_csv(path, dtype=str)
    return df


def read_table_col(path, col):
    try:
        return pd.read_csv(path, usecols=[col], dtype=str)[col]
    except ValueError:
        return pd.read_csv(path, usecols=[col], dtype=str, sep="\t")[col]


# ------------------------------------------------------------ shared pair checks
def pair_validity(df, members, ecg2part, ecg2src, sjlife):
    """Returns (ok, detail). Vectorised, independent of the generators' own verification."""
    need = {"ECG_ID_A", "ECG_ID_B", "Participant_A", "Participant_B", "label"}
    if need - set(df.columns):
        return False, f"missing columns {sorted(need - set(df.columns))}"
    if not df["label"].isin(["0", "1"]).all():
        return False, "label column has values other than 0 and 1"
    lab = df["label"].astype(int)
    same = df["Participant_A"] == df["Participant_B"]
    issues = []
    if not same[lab == 1].all():
        issues.append("a positive pair spans two participants")
    if same[lab == 0].any():
        issues.append("a negative pair is within one participant")
    if not sjlife and (df["ECG_ID_A"][lab == 1] == df["ECG_ID_B"][lab == 1]).any():
        issues.append("a positive pair uses the same recording twice")
    fwd = df["ECG_ID_A"] + "|" + df["ECG_ID_B"]
    rev = df["ECG_ID_B"] + "|" + df["ECG_ID_A"]
    n_dup = int(fwd.duplicated().sum())
    clash = int((rev.isin(set(fwd)) & (df["ECG_ID_A"] != df["ECG_ID_B"])).sum())
    if n_dup:
        issues.append(f"{n_dup} duplicated pair(s)")
    if clash:
        issues.append(f"{clash} pair(s) also present reversed")
    used = set(df["Participant_A"]) | set(df["Participant_B"])
    outside = used - members
    if outside:
        issues.append(f"{len(outside)} participant(s) outside this partition (e.g. {sorted(outside)[:2]})")
    pa, pb = df["ECG_ID_A"].map(ecg2part), df["ECG_ID_B"].map(ecg2part)
    bad_id = int((pa != df["Participant_A"]).sum() + (pb != df["Participant_B"]).sum())
    if bad_id:
        issues.append(f"{bad_id} ECG_ID(s) unknown or attached to the wrong participant per the metadata")
    if sjlife:
        wrong_vendor = int((df["ECG_ID_A"].map(ecg2src) != "akbilgic_clinical").sum() + (df["ECG_ID_B"].map(ecg2src) != "akbilgic_watch").sum())
        if wrong_vendor:
            issues.append(f"{wrong_vendor} pair member(s) on the wrong side (A must be clinical, B must be watch)")
    n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
    return (not issues), (f"{len(df)} pairs ({n_pos} pos / {n_neg} neg); all rules hold" if not issues else "; ".join(issues))


def frozen_check(section, part, df, path, manifest_entry, scheme):
    if manifest_entry is None or not manifest_entry.get("sha256"):
        rec(section, "V9", f"frozen file ({part})", "FAIL", "no recorded hash in the pairs manifest")
        return
    actual = content_hash(df.assign(label=df["label"].astype(int))) if scheme == "content" else sha256_file(path)
    ok = actual == manifest_entry["sha256"]
    rec(section, "V9", f"frozen file ({part})", "PASS" if ok else "FAIL",
        "hash equals the one recorded when it was frozen" if ok else "HASH CHANGED since it was frozen: every prior comparison is void")


# ------------------------------------------------------------------------- MIMIC
def verify_mimic(a, ledger):
    S = "MIMIC"
    for p in (a.mimic_split, a.mimic_eligible, a.mimic_metadata):
        if not Path(p).exists():
            rec(S, "--", "inputs", "SKIP", f"{p} not found"); return
    split = json.load(open(a.mimic_split))
    ids = {p: set(map(str, split[f"{p}_ids"])) for p in PARTS}
    elig = pd.read_csv(a.mimic_eligible, dtype=str)
    all_pat = set(elig["subject_id"].astype(str))
    n_ids = sum(len(v) for v in ids.values())

    counts_ok = all(len(ids[p]) == split["counts"][p] for p in PARTS)
    rec(S, "V1", "patient inventory", "PASS" if (len(all_pat) > 0 and counts_ok and n_ids == len(all_pat)) else "FAIL",
        f"Nelig = {len(all_pat)} unique patients in {Path(a.mimic_eligible).name}; manifest lists {n_ids}; counts match lists: {counts_ok}")
    ledger["mimic_nelig"] = len(all_pat)
    ledger["mimic_patients"] = {p: len(ids[p]) for p in PARTS}

    rec_ok = all(hash_comma(ids[p]) == split["hashes"][f"{p}_sha256"] for p in PARTS)
    rec(S, "V2a", "recorded hashes", "PASS" if rec_ok else "FAIL",
        "every ID list still hashes to the value recorded in the manifest" if rec_ok else "an ID list no longer matches its recorded hash (manifest edited?)")
    ps = load_module(a.patient_splitter, "patient_splitter_under_test")
    if ps is None:
        rec(S, "V2b", "re-run reproduces split", "SKIP", f"{a.patient_splitter} not found")
    else:
        tr, va, te = ps.split_patients(elig, seed=split["seed"])
        again = {"train": tr, "val": va, "test": te}
        same = all(hash_comma(again[p]) == split["hashes"][f"{p}_sha256"] for p in PARTS)
        rec(S, "V2b", "re-run reproduces split", "PASS" if same else "FAIL",
            "re-running the splitter with the recorded seed gives identical partitions" if same else
            "re-run DIFFERS. The MIMIC splitter shuffles patients in the order they appear in the eligible file, so a regenerated or reordered eligible file gives a different split.")

    ov = {f"{x}&{y}": len(ids[x] & ids[y]) for i, x in enumerate(PARTS) for y in PARTS[i + 1:]}
    rec(S, "V3", "disjointness", "PASS" if not any(ov.values()) else "FAIL", f"overlaps {ov}")
    union = set().union(*ids.values())
    rec(S, "V4", "completeness", "PASS" if union == all_pat else "FAIL",
        f"{len(union)} assigned of {len(all_pat)} eligible; missing {len(all_pat - union)}, extra {len(union - all_pat)}")
    pct = {p: len(ids[p]) / max(len(all_pat), 1) for p in PARTS}
    ok5 = all(abs(pct[p] - MIMIC_TARGET[p]) <= 0.01 for p in PARTS)
    rec(S, "V5", "proportions", "PASS" if ok5 else "FAIL", ", ".join(f"{p} {pct[p]:.2%} (target {MIMIC_TARGET[p]:.0%})" for p in PARTS))

    where = {pid: p for p in PARTS for pid in ids[p]}
    meta = pd.read_csv(a.mimic_metadata, dtype=str, usecols=["ECG_ID", "Participant_ID", "Source"])
    unassigned_rec = int((~elig["subject_id"].astype(str).isin(where)).sum())
    unassigned_ecg = int((~meta["Participant_ID"].isin(where)).sum())
    dup_study, dup_ecg = int(elig["study_id"].duplicated().sum()), int(meta["ECG_ID"].duplicated().sum())
    ok6 = unassigned_rec == 0 and unassigned_ecg == 0 and dup_study == 0 and dup_ecg == 0
    rec(S, "V6", "recording containment", "PASS" if ok6 else "FAIL",
        f"{len(elig)} eligible records and {len(meta)} ingested ECGs; unassigned {unassigned_rec}/{unassigned_ecg}; duplicate study_ids {dup_study}, ECG_IDs {dup_ecg}")

    pg = load_module(a.pair_generator, "pair_generator_under_test")
    if pg is not None:
        same_caps = getattr(pg, "POS_CAP", None) == POS_CAP and getattr(pg, "NEG_CAP", None) == NEG_CAP
        rec(S, "V8", "caps match the generator", "PASS" if same_caps else "FAIL",
            "verify_splits.py uses the same caps as pair_generator.py" if same_caps else
            f"caps differ: generator has POS {getattr(pg, 'POS_CAP', None)} NEG {getattr(pg, 'NEG_CAP', None)}")
    ecg2part = dict(zip(meta["ECG_ID"], meta["Participant_ID"]))
    pm_path = Path(f"{a.mimic_pairs_prefix}_pairs_manifest.json")
    pm = json.load(open(pm_path)) if pm_path.exists() else None
    if pm is None:
        rec(S, "V9", "frozen files", "SKIP", f"{pm_path} not found")
    pair_counts, mimic_pair_ecgs = {}, set()
    for p in PARTS:
        path = Path(f"{a.mimic_pairs_prefix}_{p}_pairs.csv")
        if not path.exists():
            rec(S, "V7", f"pair validity ({p})", "SKIP", f"{path} not found"); continue
        df = read_pairs(path)
        ok, detail = pair_validity(df, ids[p], ecg2part, None, sjlife=False)
        rec(S, "V7", f"pair validity ({p})", "PASS" if ok else "FAIL", detail)
        lab = df["label"].astype(int)
        pos_c = df["Participant_A"][lab == 1].value_counts()
        neg_c = pd.concat([df["Participant_A"][lab == 0], df["Participant_B"][lab == 0]]).value_counts()
        mp, mn = int(pos_c.max()) if len(pos_c) else 0, int(neg_c.max()) if len(neg_c) else 0
        rec(S, "V8", f"cap enforcement ({p})", "PASS" if (mp <= POS_CAP[p] and mn <= NEG_CAP[p]) else "FAIL",
            f"max positives per participant {mp} (cap {POS_CAP[p]}); max negative appearances {mn} (cap {NEG_CAP[p]})")
        pair_counts[p] = {"pos": int((lab == 1).sum()), "neg": int((lab == 0).sum())}
        mimic_pair_ecgs |= set(df["ECG_ID_A"]) | set(df["ECG_ID_B"])
        if p in ("val", "test") and pm is not None:
            frozen_check(S, p, df, path, pm.get("partitions", {}).get(p), "content")
    ledger["mimic_pairs"] = pair_counts
    ledger["_mimic_pair_ecgs"] = mimic_pair_ecgs


# ------------------------------------------------------------------------ SJLIFE
def verify_sjlife(a, ledger):
    S = "SJLIFE"
    for p in (a.sjlife_split, a.sjlife_metadata):
        if not Path(p).exists():
            rec(S, "--", "inputs", "SKIP", f"{p} not found"); return
    split = json.load(open(a.sjlife_split))
    parts = {p: set(split["participants"][p]) for p in PARTS}
    fr = split["fractions"]
    meta = pd.read_csv(a.sjlife_metadata, dtype=str, usecols=["ECG_ID", "Participant_ID", "Source"])
    everyone = set(meta["Participant_ID"])
    per = meta.groupby("Participant_ID")["Source"].apply(sorted)
    bad = [pid for pid, s in per.items() if s != ["akbilgic_clinical", "akbilgic_watch"]]
    n_ids = sum(len(v) for v in parts.values())
    rec(S, "V1", "participant inventory", "PASS" if (len(everyone) > 0 and not bad and n_ids == len(everyone)) else "FAIL",
        f"{len(everyone)} participants in the metadata, {n_ids} in the split; {len(bad)} without exactly one clinical + one watch recording")
    ledger["sjlife_participants"] = {p: len(parts[p]) for p in PARTS}

    ok_rec = all(hash_newline(parts[p]) == split["sha256_sorted_ids"][p] for p in PARTS)
    rec(S, "V2a", "recorded hashes", "PASS" if ok_rec else "FAIL",
        "every participant list still hashes to the recorded value" if ok_rec else "a list no longer matches its recorded hash (manifest edited?)")
    sp = load_module(a.sjlife_splitter, "sjlife_splitter_under_test")
    if sp is None:
        rec(S, "V2b", "re-run reproduces split", "SKIP", f"{a.sjlife_splitter} not found")
    else:
        again = sp.split_participants(list(meta["Participant_ID"]), fr, split["seed"])
        same = all(hash_newline(again[p]) == split["sha256_sorted_ids"][p] for p in PARTS)
        rec(S, "V2b", "re-run reproduces split", "PASS" if same else "FAIL",
            "re-running the splitter with the recorded seed and fractions gives identical partitions" if same else "re-run DIFFERS from the recorded split")

    ov = {f"{x}&{y}": len(parts[x] & parts[y]) for i, x in enumerate(PARTS) for y in PARTS[i + 1:]}
    rec(S, "V3", "disjointness", "PASS" if not any(ov.values()) else "FAIL", f"overlaps {ov}")
    union = set().union(*parts.values())
    rec(S, "V4", "completeness", "PASS" if union == everyone else "FAIL", f"{len(union)} assigned of {len(everyone)}; missing {len(everyone - union)}, extra {len(union - everyone)}")
    ok5 = all(abs(len(parts[p]) / max(len(everyone), 1) - fr[p]) <= 0.01 for p in PARTS)
    rec(S, "V5", "proportions", "PASS" if ok5 else "FAIL", ", ".join(f"{p} {len(parts[p])} ({len(parts[p]) / max(len(everyone), 1):.1%} vs {fr[p]:.1%})" for p in PARTS))
    where = {pid: p for p in PARTS for pid in parts[p]}
    m = meta.assign(part=meta["Participant_ID"].map(where))
    unresolved = int(m["part"].isna().sum())
    balanced = all((g["Source"] == "akbilgic_clinical").sum() == (g["Source"] == "akbilgic_watch").sum() for _, g in m.groupby("part"))
    dup = int(meta["ECG_ID"].duplicated().sum())
    rec(S, "V6", "recording containment", "PASS" if (unresolved == 0 and balanced and dup == 0) else "FAIL",
        f"{len(meta)} ECGs; {unresolved} unassigned; duplicate ECG_IDs {dup}; clinical/watch counts equal in every partition: {balanced}")

    ecg2part, ecg2src = dict(zip(meta["ECG_ID"], meta["Participant_ID"])), dict(zip(meta["ECG_ID"], meta["Source"]))
    pm_path = Path(f"{a.sjlife_pairs_prefix}_pairs_manifest.json")
    pm = json.load(open(pm_path)) if pm_path.exists() else None
    if pm is None:
        rec(S, "V9", "frozen files", "SKIP", f"{pm_path} not found")
    else:
        ok_link = pm.get("split_manifest_sha256") == sha256_file(a.sjlife_split)
        rec(S, "S", "pairs were built from this split", "PASS" if ok_link else "FAIL",
            "the pairs manifest records this exact split manifest" if ok_link else "the split manifest changed since the pairs were built")
    used, pair_counts, sj_pair_ecgs = {}, {}, set()
    for p in PARTS:
        path = Path(f"{a.sjlife_pairs_prefix}_{p}_pairs.csv")
        if not path.exists():
            rec(S, "V7", f"pair validity ({p})", "SKIP", f"{path} not found"); continue
        df = read_pairs(path)
        ok, detail = pair_validity(df, parts[p], ecg2part, ecg2src, sjlife=True)
        rec(S, "V7", f"pair validity ({p})", "PASS" if ok else "FAIL", detail)
        lab = df["label"].astype(int)
        pos_c = df["Participant_A"][lab == 1].value_counts()
        ok8 = len(pos_c) > 0 and pos_c.max() == 1 and set(pos_c.index) == parts[p]
        rec(S, "V8", f"cap enforcement ({p})", "PASS" if ok8 else "FAIL",
            f"{len(pos_c)} of {len(parts[p])} participants have a positive; max per participant {int(pos_c.max()) if len(pos_c) else 0} (cap 1)")
        used[p] = set(df["Participant_A"]) | set(df["Participant_B"])
        pair_counts[p] = {"pos": int((lab == 1).sum()), "neg": int((lab == 0).sum())}
        sj_pair_ecgs |= set(df["ECG_ID_A"]) | set(df["ECG_ID_B"])
        if p in ("val", "test") and pm is not None:
            frozen_check(S, p, df, path, pm.get("files", {}).get(p), "file")
    ledger["sjlife_pairs"] = pair_counts
    bad_iso = {f"{x}&{y}": len(used[x] & used[y]) for i, x in enumerate(list(used)) for y in list(used)[i + 1:] if used[x] & used[y]}
    rec(S, "V10", "cross-vendor isolation", "PASS" if not bad_iso else "FAIL",
        "no participant appears in the pairs of two partitions" if not bad_iso else f"OVERLAP {bad_iso}")
    if "_mimic_pair_ecgs" in ledger:
        leak = len(sj_pair_ecgs & ledger["_mimic_pair_ecgs"])
        rec(S, "V10", "no SJLIFE ECG in MIMIC pairs", "PASS" if leak == 0 else "FAIL", f"{leak} shared ECG_IDs between the two pair sets")


# --------------------------------------------------------------------- V11, ledger
def verify_v11(a):
    if a.v11_result and Path(a.v11_result).exists():
        v = json.load(open(a.v11_result)).get("val_auroc")
        if v is None:
            rec("ALL", "V11", "label-permutation probe", "FAIL", f"{a.v11_result} has no 'val_auroc'")
        else:
            rec("ALL", "V11", "label-permutation probe", "PASS" if v <= 0.55 else "FAIL",
                f"shuffled-label validation AUROC = {v:.3f} (must be about 0.50; above 0.55 means the answer is reaching the model through its inputs)")
    else:
        rec("ALL", "V11", "label-permutation probe", "SKIP", "PENDING: Modeling runs this after training briefly on shuffled labels (pass --v11-result)")


def write_ledger(a, ledger, n_unique):
    L = ledger
    rows = [("Unique patients N (MIMIC)", "about 160,000", f"{n_unique:,}" if n_unique else "not measured (pass --record-list)"),
            ("Eligible patients Nelig (MIMIC, downloaded)", "measure it", f"{L.get('mimic_nelig', 'n/a'):,}" if isinstance(L.get("mimic_nelig"), int) else "n/a")]
    for p, tgt in zip(PARTS, ("70% of Nelig", "15%", "15% (touched once, at the end)")):
        rows.append((f"MIMIC {p} patients", tgt, f"{L.get('mimic_patients', {}).get(p, 'n/a'):,}" if L.get("mimic_patients") else "n/a"))
    for p in PARTS:
        c = L.get("mimic_pairs", {}).get(p)
        rows.append((f"MIMIC {p} pairs (pos / neg)", "about 1:1", f"{c['pos']:,} / {c['neg']:,}" if c else "n/a"))
    for p in PARTS:
        c = L.get("sjlife_participants", {}).get(p)
        rows.append((f"SJLIFE {p} participants", "decided at freeze", str(c) if c is not None else "n/a"))
    for p in PARTS:
        c = L.get("sjlife_pairs", {}).get(p)
        rows.append((f"SJLIFE {p} pairs (pos / neg)", "cross-vendor", f"{c['pos']:,} / {c['neg']:,}" if c else "n/a"))
    pos_total = sum(v["pos"] for v in L.get("sjlife_pairs", {}).values()) if L.get("sjlife_pairs") else None
    rows.append(("Cross-vendor positives (SJLIFE, all partitions)", "at most 243", str(pos_total) if pos_total is not None else "n/a"))
    md = [f"# Size ledger (measured {datetime.date.today().isoformat()} by verify_splits.py)", "",
          "| Quantity | Target | Measured |", "|---|---|---|"] + [f"| {q} | {t} | {m} |" for q, t, m in rows]
    Path(a.ledger_out).write_text("\n".join(md) + "\n")
    print(f"\nwrote {a.ledger_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-split", default="split_manifest.json")
    ap.add_argument("--mimic-eligible", default="eligible_records_downloaded.csv")
    ap.add_argument("--mimic-metadata", default="ecg_mimic_metadata.csv")
    ap.add_argument("--mimic-pairs-prefix", default="ecg_mimic")
    ap.add_argument("--sjlife-split", default="split_manifest_sjlife.json")
    ap.add_argument("--sjlife-metadata", default="ecg_akbilgic_metadata.csv")
    ap.add_argument("--sjlife-pairs-prefix", default="ecg_akbilgic")
    ap.add_argument("--patient-splitter", default="patient_splitter.py")
    ap.add_argument("--pair-generator", default="pair_generator.py")
    ap.add_argument("--sjlife-splitter", default="sjlife_splitter.py")
    ap.add_argument("--record-list", help="MIMIC record_list.csv, only to measure N (all patients) for the ledger")
    ap.add_argument("--v11-result")
    ap.add_argument("--skip-mimic", action="store_true")
    ap.add_argument("--skip-sjlife", action="store_true")
    ap.add_argument("--strict", action="store_true", help="treat SKIP and a pending V11 as failures")
    ap.add_argument("--write-ledger", action="store_true")
    ap.add_argument("--ledger-out", default="size_ledger.md")
    ap.add_argument("--report", default="verify_splits_report.json")
    a = ap.parse_args()

    ledger = {}
    if not a.skip_mimic:
        verify_mimic(a, ledger)
    if not a.skip_sjlife:
        verify_sjlife(a, ledger)
    verify_v11(a)

    n_unique = None
    if a.record_list and Path(a.record_list).exists():
        n_unique = int(read_table_col(a.record_list, "subject_id").nunique())
        print(f"\nN (all patients in {Path(a.record_list).name}) = {n_unique:,}")
    if a.write_ledger:
        write_ledger(a, ledger, n_unique)

    n = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "FAIL", "SKIP")}
    inputs = {str(p): sha256_file(p) for p in (a.mimic_split, a.sjlife_split, f"{a.mimic_pairs_prefix}_pairs_manifest.json",
                                               f"{a.sjlife_pairs_prefix}_pairs_manifest.json") if Path(p).exists()}
    Path(a.report).write_text(json.dumps({"when": datetime.datetime.now().isoformat(timespec="seconds"), "counts": n,
                                          "manifest_sha256": inputs, "results": RESULTS}, indent=2))
    print(f"\n{n['PASS']} passed, {n['FAIL']} FAILED, {n['SKIP']} skipped   (report: {a.report})")
    skipped = [f"{r['section']} {r['id']}" for r in RESULTS if r["status"] == "SKIP"]
    if skipped:
        print("skipped:", ", ".join(skipped))
    if n["FAIL"]:
        print("CHAIN FAILED: do not start a training job or record a leaderboard row until this is fixed.")
        sys.exit(1)
    if a.strict and n["SKIP"]:
        print("STRICT: some checks were skipped or are pending.")
        sys.exit(1)
    print("chain passed for everything that ran." + (" (V11 is still pending)" if any(r["id"] == "V11" and r["status"] == "SKIP" for r in RESULTS) else ""))


if __name__ == "__main__":
    main()
