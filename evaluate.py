#!/usr/bin/env python3
"""evaluate.py -- overall AUROC plus every stratification, for one frozen pair file.

WHAT IT DOES
    Scores come from the model, pairs come from the frozen pair files. This script joins them and reports
    AUROC overall and by stratum. It only needs numpy, pandas and scipy (no scikit-learn).

USAGE
    python3 evaluate.py --pairs ecg_akbilgic_val_pairs.csv --scores my_scores.csv \\
        --pairs-manifest ecg_akbilgic_pairs_manifest.json \\
        --metadata ecg_akbilgic_metadata.csv \\
        --participant-metadata ecg_akbilgic_participant_metadata.csv \\
        --bootstrap 500 --out my_eval.json

    MIMIC validation, with recording times for the time-gap strata:
    python3 evaluate.py --pairs ecg_mimic_val_pairs.csv --scores my_scores.csv \\
        --pairs-manifest ecg_mimic_pairs_manifest.json --metadata ecg_mimic_metadata.csv \\
        --record-times eligible_records_downloaded.csv

SCORES FILE (the contract with Modeling)
    A CSV with columns  ECG_ID_A, ECG_ID_B, score   -- one row per pair in the pair file.
    Higher score = stronger evidence the two recordings are the same person. Every pair must have exactly one finite
    score. Extra rows are ignored with a warning.

STRATA
    overall              all pairs
    vendor pairing       from each ECG's Source (metadata), e.g. "akbilgic_clinical + akbilgic_watch"
    time gap             POSITIVE pairs binned by the gap between their two recordings, each bin scored against ALL
                         negatives (negatives have no gap). MIMIC gaps come from --record-times. SJLIFE positives were
                         recorded about 42 minutes apart (team plan 6.1), so they all go in the shortest bin (ASSUMED,
                         flagged in the output).
    age band             by participant A's age, from --participant-metadata. Missing age -> "unknown".
    pathology burden     not available with the current data; reported as such.

UNCERTAINTY
    se_hm         Hanley-McNeil standard error (the plan's noise-floor method). It assumes independent pairs.
                  When negatives reuse the same participants (SJLIFE val/test use every negative), it is optimistic.
    --bootstrap N adds a 95% interval from resampling PARTICIPANTS (a participant's pairs move together), which
                  respects that dependence. Recommended for SJLIFE.

GUARDS
    --pairs-manifest    verifies a frozen val/test pair file still matches the hash recorded when it was frozen
                        (works with both the MIMIC and SJLIFE manifest formats). A mismatch stops the run.
    test set            refuses to evaluate a *_test_pairs file unless --allow-test is passed. The plan opens the local
                        test set exactly once, at the very end, by the Clinical Lead.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SJLIFE_ASSUMED_GAP_DAYS = 42 / (60 * 24)  # plan 6.1: paired recordings ~42 minutes apart
FEW_POSITIVES = 30
PARTITIONS = ("train", "val", "test")


# ---------------------------------------------------------------- statistics
def auroc(y, s, w=None):
    """AUROC = P(random positive scores above random negative), ties count half. Optional pair weights."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    keep = w > 0
    y, s, w = y[keep], s[keep], w[keep]
    pos_w, neg_w_total = w[y == 1].sum(), w[y == 0].sum()
    if pos_w == 0 or neg_w_total == 0:
        return float("nan")
    uniq, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=w * (y == 1), minlength=len(uniq))
    neg = np.bincount(inv, weights=w * (y == 0), minlength=len(uniq))
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos_w * neg_w_total))


def hanley_mcneil_se(a, n_pos, n_neg):
    if not (n_pos > 1 and n_neg > 1) or not np.isfinite(a):
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def participant_bootstrap(y, s, a_idx, b_idx, counts):
    """95% interval from participant-level resampling. counts: (n_boot, n_participants) resample multiplicities."""
    same = a_idx == b_idx
    vals = []
    for c in counts:
        w = np.where(same, c[a_idx], c[a_idx] * c[b_idx])  # a positive is one participant's pair; a negative needs both
        v = auroc(y, s, w)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, len(counts) // 2):
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def stats_for(df, mask, boot):
    sub = df[mask]
    y, s = sub["label"].to_numpy(), sub["score"].to_numpy()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    a = auroc(y, s)
    out = {"n_pos": n_pos, "n_neg": n_neg, "auroc": None if not np.isfinite(a) else round(a, 5)}
    if np.isfinite(a):
        out["se_hm"] = round(hanley_mcneil_se(a, n_pos, n_neg), 5)
        if boot is not None:
            ci = participant_bootstrap(y, s, boot["a"][mask.to_numpy()], boot["b"][mask.to_numpy()], boot["counts"])
            out["ci95_participant_bootstrap"] = None if ci is None else [round(ci[0], 5), round(ci[1], 5)]
    out["few_positives"] = bool(n_pos < FEW_POSITIVES)
    return out


# ------------------------------------------------------------------- inputs
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_pairs(path):
    df = pd.read_csv(path, dtype=str)
    need = {"ECG_ID_A", "ECG_ID_B", "Participant_A", "Participant_B", "label"}
    if need - set(df.columns):
        sys.exit(f"STOPPED: {path} is missing columns {sorted(need - set(df.columns))}")
    if not df["label"].isin(["0", "1"]).all():
        sys.exit("STOPPED: label column must contain only 0 and 1")
    df["label"] = df["label"].astype(int)
    return df


def content_hash(df):
    """Same order-insensitive pair hash that pair_generator.py records (a|b|label, sorted)."""
    rows = sorted(f"{a}|{b}|{l}" for a, b, l in zip(df["ECG_ID_A"], df["ECG_ID_B"], df["label"]))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def infer_partition(path):
    for p in PARTITIONS:
        if f"_{p}_pairs" in Path(path).name:
            return p
    return None


def check_frozen(df, pairs_path, manifest_path, partition):
    m = json.load(open(manifest_path))
    if "partitions" in m:            # MIMIC format (pair_generator.py): order-insensitive content hash
        recorded, actual = m["partitions"].get(partition, {}).get("sha256"), content_hash(df)
    elif "files" in m:               # SJLIFE format (cross_vendor_pairs.py): SHA-256 of the file bytes
        recorded, actual = m["files"].get(partition, {}).get("sha256"), sha256_file(pairs_path)
    else:
        sys.exit(f"STOPPED: {manifest_path} is not a recognised pairs manifest")
    if recorded is None:
        sys.exit(f"STOPPED: no recorded hash for '{partition}' in {manifest_path}")
    if recorded != actual:
        sys.exit(f"STOPPED: {pairs_path} does NOT match the hash recorded when it was frozen. "
                 f"Every comparison against it would be invalid.")
    return True


def load_scores(path, pairs):
    sc = pd.read_csv(path)
    need = {"ECG_ID_A", "ECG_ID_B", "score"}
    if need - set(sc.columns):
        sys.exit(f"STOPPED: {path} is missing columns {sorted(need - set(sc.columns))} (need ECG_ID_A, ECG_ID_B, score)")
    sc["ECG_ID_A"], sc["ECG_ID_B"] = sc["ECG_ID_A"].astype(str), sc["ECG_ID_B"].astype(str)
    sc["score"] = pd.to_numeric(sc["score"], errors="coerce")
    if not np.isfinite(sc["score"].to_numpy(dtype=float)).all():
        sys.exit(f"STOPPED: {int((~np.isfinite(sc['score'].to_numpy(dtype=float))).sum())} score(s) are missing, NaN or infinite")
    dup = int(sc.duplicated(["ECG_ID_A", "ECG_ID_B"]).sum())
    if dup:
        sys.exit(f"STOPPED: {dup} pair(s) appear more than once in the scores file")
    merged = pairs.merge(sc[["ECG_ID_A", "ECG_ID_B", "score"]], on=["ECG_ID_A", "ECG_ID_B"], how="left")
    missing = int(merged["score"].isna().sum())
    if missing:
        sys.exit(f"STOPPED: {missing} of {len(pairs)} pairs have no score in {path}. Every pair needs exactly one score.")
    extra = len(sc) - len(pairs)
    return merged, extra


def source_lookup(meta_paths, ids):
    if meta_paths:
        m = pd.concat([pd.read_csv(p, dtype=str, usecols=["ECG_ID", "Source"]) for p in meta_paths])
        lookup = dict(zip(m["ECG_ID"], m["Source"]))
        unknown = [i for i in ids if i not in lookup]
        if unknown:
            print(f"[WARN] {len(unknown)} ECG_IDs not found in --metadata; using the ID pattern for those")
    else:
        lookup = {}
        print("[NOTE] no --metadata given: vendor is read from the ECG_ID pattern (mimic_..., akbilgic_..._clinical/_watch)")

    def src(i):
        if i in lookup:
            return lookup[i]
        if i.startswith("akbilgic_") and i.endswith("_clinical"):
            return "akbilgic_clinical"
        if i.startswith("akbilgic_") and i.endswith("_watch"):
            return "akbilgic_watch"
        return "mimic" if i.startswith("mimic_") else "unknown"
    return src


def parse_times(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # older pandas without format="mixed"
        return pd.to_datetime(series, errors="coerce")


def binned(values, edges, labels):
    return pd.Series(pd.cut(values, bins=[-np.inf] + list(edges) + [np.inf], labels=labels, right=False), index=values.index)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--partition", choices=PARTITIONS)
    ap.add_argument("--pairs-manifest")
    ap.add_argument("--metadata", nargs="+")
    ap.add_argument("--participant-metadata", nargs="+")
    ap.add_argument("--record-times", help="CSV with subject_id, study_id, ecg_time (e.g. eligible_records_downloaded.csv)")
    ap.add_argument("--age-edges", default="30,40,50,60,70")
    ap.add_argument("--gap-edges", default="1,7,30,365", help="day boundaries between time-gap bins")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    partition = a.partition or infer_partition(a.pairs)
    if partition == "test" and not a.allow_test:
        sys.exit("STOPPED: this is the LOCAL TEST set. The plan opens it exactly once, at the very end, by the Clinical Lead. "
                 "If that is you, and the model is final, re-run with --allow-test.")
    pairs = load_pairs(a.pairs)
    print(f"pairs: {a.pairs}  ({len(pairs)} pairs, partition: {partition or 'unknown'})")
    if a.pairs_manifest:
        if partition is None:
            sys.exit("STOPPED: cannot tell which partition this is; pass --partition to use --pairs-manifest")
        if partition == "train":
            print("[NOTE] train pair files are not frozen; skipping the frozen-hash check")
        else:
            check_frozen(pairs, a.pairs, a.pairs_manifest, partition)
            print(f"[PASS] frozen-hash check: {Path(a.pairs).name} matches the recorded hash")

    df, extra = load_scores(a.scores, pairs)
    if extra > 0:
        print(f"[WARN] {extra} extra row(s) in the scores file were ignored")
    if df["score"].nunique() == 1:
        print("[WARN] every score is identical -- AUROC will be 0.5 and means nothing")

    boot = None
    if a.bootstrap > 0:
        parts = sorted(set(df["Participant_A"]) | set(df["Participant_B"]))
        index = {p: i for i, p in enumerate(parts)}
        rng = np.random.default_rng(a.seed)
        draws = rng.integers(0, len(parts), size=(a.bootstrap, len(parts)))
        counts = np.stack([np.bincount(d, minlength=len(parts)) for d in draws])
        boot = {"a": df["Participant_A"].map(index).to_numpy(), "b": df["Participant_B"].map(index).to_numpy(), "counts": counts}

    src = source_lookup(a.metadata, set(df["ECG_ID_A"]) | set(df["ECG_ID_B"]))
    df["source_A"], df["source_B"] = df["ECG_ID_A"].map(src), df["ECG_ID_B"].map(src)
    df["vendor_pair"] = [" + ".join(sorted((x, y))) for x, y in zip(df["source_A"], df["source_B"])]

    report = {"pairs_file": Path(a.pairs).name, "pairs_sha256": sha256_file(a.pairs), "scores_file": Path(a.scores).name,
              "scores_sha256": sha256_file(a.scores), "partition": partition, "n_pairs": int(len(df)),
              "bootstrap_resamples": int(a.bootstrap), "strata": {}, "notes": []}
    everything = pd.Series(True, index=df.index)
    report["overall"] = stats_for(df, everything, boot)
    if report["overall"]["auroc"] is not None and report["overall"]["auroc"] < 0.4:
        report["notes"].append("overall AUROC is well below 0.5: the scores may be inverted (higher must mean same person)")

    # vendor pairing
    report["strata"]["vendor_pairing"] = {k: stats_for(df, df["vendor_pair"] == k, boot) for k in sorted(df["vendor_pair"].unique())}

    # time gap: positives by gap bin against all negatives
    gap_edges = [float(x) for x in a.gap_edges.split(",")]
    gap_labels = ([f"under {gap_edges[0]:g} day"]
                  + [f"{gap_edges[i]:g} to <{gap_edges[i + 1]:g} days" for i in range(len(gap_edges) - 1)]
                  + [f"{gap_edges[-1]:g}+ days"])
    gaps = pd.Series(np.nan, index=df.index)
    times = None
    if a.record_times:
        rt = pd.read_csv(a.record_times, dtype=str, usecols=["subject_id", "study_id", "ecg_time"])
        rt["t"] = parse_times(rt["ecg_time"])
        bad = int(rt["t"].isna().sum())
        if bad:
            report["notes"].append(f"{bad} of {len(rt)} record times could not be parsed and are treated as unknown gap")
        times = dict(zip("mimic_" + rt["subject_id"] + "_" + rt["study_id"], rt["t"]))
    pos = df["label"] == 1
    ta, tb = df["ECG_ID_A"].map(times) if times else None, df["ECG_ID_B"].map(times) if times else None
    if times:
        gaps[pos] = ((pd.to_datetime(tb[pos]) - pd.to_datetime(ta[pos])).abs().dt.total_seconds() / 86400.0)
    sj = pos & df["source_A"].str.startswith("akbilgic") & df["source_B"].str.startswith("akbilgic")
    if sj.any():
        gaps[sj] = SJLIFE_ASSUMED_GAP_DAYS
        report["notes"].append("SJLIFE positive pairs were assigned a gap of ~42 minutes from team plan 6.1 (assumed, not measured per pair)")
    gap_bin = binned(gaps, gap_edges, gap_labels)
    neg = df["label"] == 0
    tg = {}
    for lab in gap_labels:
        m = neg | (pos & (gap_bin == lab))
        if (pos & (gap_bin == lab)).sum() > 0:
            tg[lab] = stats_for(df, m, boot)
    unknown_gap = int((pos & gaps.isna()).sum())
    if unknown_gap:
        tg["unknown gap (positives only)"] = {"n_pos": unknown_gap, "note": "no recording time available"}
    report["strata"]["time_gap"] = tg if tg else "unavailable: pass --record-times (MIMIC) or use SJLIFE pairs"

    # age band by participant A
    age_edges = [float(x) for x in a.age_edges.split(",")]
    age_labels = ([f"under {age_edges[0]:g}"]
                  + [f"{age_edges[i]:g} to <{age_edges[i + 1]:g}" for i in range(len(age_edges) - 1)]
                  + [f"{age_edges[-1]:g}+"])
    if a.participant_metadata:
        pm = pd.concat([pd.read_csv(p, dtype=str, usecols=["Participant_ID", "Age"]) for p in a.participant_metadata])
        age = df["Participant_A"].map(dict(zip(pm["Participant_ID"], pd.to_numeric(pm["Age"], errors="coerce"))))
    else:
        age = pd.Series(np.nan, index=df.index)
    band = binned(age, age_edges, age_labels).astype(object).where(age.notna(), "unknown")
    ab = {}
    for lab in age_labels + ["unknown"]:
        m = band == lab
        if m.sum() and (df["label"][m] == 1).any() and (df["label"][m] == 0).any():
            ab[lab] = stats_for(df, m, boot)
    report["strata"]["age_band"] = ab if ab else "unavailable: no ages for these participants"
    if not a.participant_metadata:
        report["notes"].append("no --participant-metadata given, so age bands are unavailable")
    report["strata"]["pathology_burden"] = "unavailable: no diagnosis data in the current metadata"

    # ---- print
    def row(name, s):
        if s.get("auroc") is None:
            return f"  {name:<44} {s.get('n_pos', 0):>6} {s.get('n_neg', 0):>7}   {s.get('note', 'AUROC undefined (needs both classes)')}"
        ci = s.get("ci95_participant_bootstrap")
        extra_txt = f"  95% [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        flag = "  (few positives: treat as a direction, not a number)" if s.get("few_positives") else ""
        return f"  {name:<44} {s['n_pos']:>6} {s['n_neg']:>7}   AUROC {s['auroc']:.4f}  se {s['se_hm']:.4f}{extra_txt}{flag}"

    print(f"\n{'':<46}{'n_pos':>6} {'n_neg':>7}")
    print(row("OVERALL", report["overall"]))
    for title, block in report["strata"].items():
        print(f"\n{title.replace('_', ' ')}:")
        if isinstance(block, str):
            print(f"  {block}")
        else:
            for k, v in block.items():
                print(row(k, v))
    for n in report["notes"]:
        print(f"[NOTE] {n}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
