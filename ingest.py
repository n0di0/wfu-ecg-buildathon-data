"""
ingest.py — ECG Similarity & Patient Matching, Data & Infrastructure lane

PHILOSOPHY
    "Preserve the ECG as faithfully as possible, then make a standardized
    ML representation without destroying the original."

    This module produces TWO separate outputs, never conflated:

    1. THE MASTER (preservation) LAYER
       - Native sample rate. Native duration. NEVER resampled if already
         at a source's own rate, never cropped, never padded.
       - float64 throughout (matches what wfdb/numpy already hand us —
         no precision is thrown away by this pipeline).
       - Stored as an .npz archive, one array per ECG_ID, so recordings of
         different native lengths can coexist without forcing a shared
         shape (which would itself require cropping or padding).
       - This is the dataset you could reconstruct MIMIC/Akbilgic-shaped
         analysis from years from now, independent of any one model's
         input-shape requirements.

    2. THE ML-READY LAYER
       - Resampled to 250 Hz ONLY if the native rate isn't already 250 Hz.
       - Forced to exactly 2500 samples via crop (documented, only when
         longer) or zero-pad (documented, only when shorter — should be
         rare to never in practice for these two sources).
       - Derived FROM the master, never the other way around. If the
         challenge's required input shape ever changes, you regenerate
         this layer from the untouched master rather than re-downloading
         or re-reading raw files.

    Both layers share one identity: the ECG_ID column in ecg_metadata.csv
    points into both the master .npz (key ecg_<ECG_ID>) and the ML matrix
    (row index == the row's position in ecg_metadata.csv).

INPUT
    - MIMIC: nested WFDB records under <mimic_root>/files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.{hea,dat}
    - Akbilgic clinical: ClinicalECGs_full_243/clinical_ecg_NNN.npy, shape (1,12,5000) @ 500Hz
    - Akbilgic watch:    AppleECGs_full_243/apple_ecg_NNN.npy,       shape (15360,)   @ 512Hz

OUTPUT
    - ecg_master.npz        : one array per ECG_ID, key "ecg_<ECG_ID>",
                               native rate/length, float64, UNALTERED beyond
                               selecting Lead I from a multi-lead source
    - ecg_data_ml.npy       : (N, 2500) array, ML-ready, row i <-> ecg_metadata.csv row i
    - ecg_metadata.csv      : ECG_ID, Participant_ID, Pair_ID, Source,
                               Original_Hz, Final_Hz, Original_samples,
                               Final_samples, Cropped, Padded
    - participant_metadata.csv : Participant_ID + demographic columns,
                               left as a skeleton (NaN) where we don't have
                               the data yet — Clinical lead fills this in.
                               Joined to ecg_metadata.csv by Participant_ID.
    - ecg_failures.csv      : manifest rows that failed to ingest, with reason
    - ecg_checksums.json    : SHA-256 of every output file above, so a
                               teammate can confirm byte-identical results
                               without re-running ingestion or diffing 68GB
                               of files by hand

USAGE
    python3 ingest.py --selftest

    python3 ingest.py --manifest eligible_records_downloaded.csv \\
        --mimic-root MimicDataAWS --out-prefix ecg

DESIGN NOTES
    - Demographics NEVER touch the waveform arrays (master or ML). They
      live only in participant_metadata.csv, joined by Participant_ID —
      this holds even for sources (like MIMIC) where we don't have
      demographics on hand yet, so the schema stays uniform once we do.
    - IDs are validated as plain digits before being used to build
      filesystem paths (path-traversal guard).
    - All output files are written atomically (temp file -> rename), so a
      crash or Ctrl+C mid-write never leaves a corrupt file at the real path.
    - crop_mode "middle" (used for the 30s->10s Apple Watch ML crop) is a
      documented, deliberate choice — the team's own research found no
      substantial difference between crop windows, so this is a settled
      decision, not an open question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest")

ML_TARGET_RATE_HZ = 250
ML_TARGET_LENGTH = 2500  # 10s @ 250Hz — the CHALLENGE's required input shape, not a preservation rule

_SAFE_ID_RE = re.compile(r"^[0-9]+$")


# ---------------------------------------------------------------------------
# Data model: raw (master) vs. derived (ML) are explicitly different types
# ---------------------------------------------------------------------------

@dataclass
class RawSignal:
    """One ECG, at its native rate and native length. float64. Never
    resampled, cropped, or padded. This is the preservation-layer object."""
    signal: np.ndarray          # native length, float64
    ecg_id: str
    participant_id: str
    pair_id: Optional[str]      # links a clinical+watch pair (Akbilgic); None for MIMIC
    source: str                 # "mimic" | "akbilgic_clinical" | "akbilgic_watch"
    native_rate_hz: float
    native_length: int


@dataclass
class MLSignal:
    """The same ECG, forced into the model's required shape. Derived from a
    RawSignal, never read independently from disk."""
    signal: np.ndarray          # ML_TARGET_LENGTH, float32 (or configurable)
    ecg_id: str
    cropped: bool
    padded: bool
    padded_samples: int = 0


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def _validate_id(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"{field_name}={value!r} is not a plain numeric ID — refusing to "
            f"use it in a filesystem path (possible path traversal / malformed manifest row)"
        )
    return value


def _atomic_write_bytes(path: Path, write_fn) -> None:
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Writes a .npy file atomically. IMPORTANT: np.save() auto-appends
    '.npy' to a filename that doesn't already end that way -- so calling it
    directly on a '.tmp'-suffixed atomic temp path would silently create a
    *different* file (path.tmp.npy) while leaving the original temp file
    empty, which then gets renamed into place, producing a 0-byte final
    file. Passing an explicit open file OBJECT (not a path string) avoids
    numpy's auto-extension behavior entirely."""
    def _write(tmp_path: Path):
        with open(tmp_path, "wb") as f:
            np.save(f, array)
    _atomic_write_bytes(path, _write)


def _save_npz_atomic(path: Path, arrays: dict) -> None:
    """Same fix as _save_npy_atomic, for np.savez()."""
    def _write(tmp_path: Path):
        with open(tmp_path, "wb") as f:
            np.savez(f, **arrays)
    _atomic_write_bytes(path, _write)


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core signal ops
# ---------------------------------------------------------------------------

def resample_signal(signal: np.ndarray, orig_rate: float, target_rate: float, dtype=np.float64) -> np.ndarray:
    """Resample using an exact rational ratio (via fractions.Fraction), not
    a rounded approximation. If orig_rate == target_rate, this is a true
    no-op: the signal is returned as-is (cast to `dtype`), skipping
    resample_poly entirely, since even a 1:1-ratio call isn't guaranteed to
    be a perfect identity once filtering is involved."""
    if orig_rate == target_rate:
        return signal.astype(dtype, copy=True)
    ratio = Fraction(int(round(target_rate * 1000)), int(round(orig_rate * 1000))).limit_denominator(10_000)
    up, down = ratio.numerator, ratio.denominator
    resampled = resample_poly(signal, up, down)
    return resampled.astype(dtype)


def crop_or_pad(signal: np.ndarray, target_length: int, crop_mode: str = "middle") -> tuple[np.ndarray, bool, bool, int]:
    """Force a signal to exactly target_length samples. ML-layer use only —
    never call this on data destined for the master archive.

    Returns (signal, cropped, padded, padded_samples) — cropped and padded
    are tracked as SEPARATE booleans (never both True for one record, since
    a signal is either longer, shorter, or already exact) so they can each
    get their own column in ecg_metadata.csv.
    """
    n = len(signal)
    if n == target_length:
        return signal, False, False, 0
    if n > target_length:
        if crop_mode == "middle":
            start = (n - target_length) // 2
        elif crop_mode == "start":
            start = 0
        else:
            raise ValueError(f"unknown crop_mode: {crop_mode}")
        return signal[start:start + target_length], True, False, 0
    pad_width = target_length - n
    padded_signal = np.pad(signal, (0, pad_width), mode="constant", constant_values=0.0)
    return padded_signal, False, True, pad_width


def derive_ml_signal(raw: RawSignal, crop_mode: str = "start", ml_dtype=np.float32) -> MLSignal:
    """The ONLY place native data becomes ML-shaped data. Resamples (or
    skips resampling, if already at target rate) then crops/pads to exactly
    ML_TARGET_LENGTH. The master RawSignal this was derived from is
    untouched by this call."""
    resampled = resample_signal(raw.signal, raw.native_rate_hz, ML_TARGET_RATE_HZ, dtype=np.float64)
    fixed, cropped, padded, pad_n = crop_or_pad(resampled, ML_TARGET_LENGTH, crop_mode=crop_mode)
    if padded:
        log.warning("ecg_id=%s was zero-padded by %d samples (short recording)", raw.ecg_id, pad_n)
    return MLSignal(
        signal=fixed.astype(ml_dtype),
        ecg_id=raw.ecg_id,
        cropped=cropped,
        padded=padded,
        padded_samples=pad_n,
    )


def verify_master(raw_signals: list[RawSignal]) -> None:
    """Strict version: raises on the first problem found. Used by the
    selftest, where every input is synthetic and should be clean by
    construction. NOT used directly on a real batch — see
    quarantine_non_finite() below, which handles the real-world case where
    one bad recording (lead-off, sensor glitch) shouldn't invalidate the
    other 232,000+ good ones."""
    for r in raw_signals:
        assert r.signal.dtype == np.float64, f"{r.ecg_id}: master signal is not float64 ({r.signal.dtype})"
        assert len(r.signal) == r.native_length, f"{r.ecg_id}: length mismatch vs. recorded native_length"
        n_bad = int(np.sum(~np.isfinite(r.signal)))
        assert n_bad == 0, f"{r.ecg_id}: {n_bad} non-finite values in master signal"
    log.info("verify_master passed: %d records, all float64, all finite", len(raw_signals))


def quarantine_non_finite(raw_signals: list[RawSignal]) -> tuple[list[RawSignal], list[dict]]:
    """Splits raw_signals into (clean, quarantined) based on non-finite
    (NaN/Inf) values. A record with bad values in the source recording
    itself (not something OUR pipeline introduced -- we don't filter,
    resample, or alter master data) is excluded from both the master
    archive and the ML matrix, and logged with a reason, rather than
    silently dropped or allowed to crash the whole batch.

    This is the "handle imperfect input explicitly" requirement from the
    team plan (§7.2) applied to the non-finite-values case specifically.
    """
    clean: list[RawSignal] = []
    quarantined: list[dict] = []
    for r in raw_signals:
        n_bad = int(np.sum(~np.isfinite(r.signal)))
        if n_bad > 0:
            quarantined.append({
                "ecg_id": r.ecg_id,
                "participant_id": r.participant_id,
                "source": r.source,
                "n_non_finite": n_bad,
                "signal_length": len(r.signal),
                "reason": "non-finite (NaN/Inf) values present in source recording",
            })
            log.warning("quarantined %s: %d non-finite value(s) in source signal", r.ecg_id, n_bad)
        else:
            clean.append(r)
    return clean, quarantined


def verify_ml_matrix(matrix: np.ndarray) -> None:
    assert matrix.ndim == 2, f"expected 2D matrix, got shape {matrix.shape}"
    assert matrix.shape[1] == ML_TARGET_LENGTH, f"expected length {ML_TARGET_LENGTH}, got {matrix.shape[1]}"
    n_bad = int(np.sum(~np.isfinite(matrix)))
    assert n_bad == 0, f"{n_bad} non-finite values (NaN/Inf) found in ML matrix"
    log.info("verify_ml_matrix passed: shape=%s dtype=%s, all finite", matrix.shape, matrix.dtype)


# ---------------------------------------------------------------------------
# Per-source RAW ingestion (master layer — no resample/crop/pad here)
# ---------------------------------------------------------------------------

def build_mimic_record_path(mimic_root: str, subject_id: str, study_id: str) -> Path:
    subject_id = _validate_id(subject_id, "subject_id")
    study_id = _validate_id(study_id, "study_id")
    shard = f"p{subject_id[:4]}"
    return Path(mimic_root) / "files" / shard / f"p{subject_id}" / f"s{study_id}" / study_id


def ingest_mimic_raw(mimic_root: str, subject_id: str, study_id: str) -> RawSignal:
    """Reads Lead I at whatever rate/length MIMIC actually recorded it at.
    No resampling, no cropping, no padding. float64 throughout (wfdb's
    p_signal is already float64; we simply don't discard that)."""
    import wfdb

    record_path = build_mimic_record_path(mimic_root, subject_id, study_id)
    record = wfdb.rdrecord(str(record_path))
    if "I" not in record.sig_name:
        raise ValueError(f"{record_path}: no lead named 'I' found in sig_name={record.sig_name}")
    lead_idx = record.sig_name.index("I")
    lead_I = record.p_signal[:, lead_idx].astype(np.float64)

    ecg_id = f"mimic_{subject_id}_{study_id}"
    return RawSignal(
        signal=lead_I,
        ecg_id=ecg_id,
        participant_id=subject_id,
        pair_id=None,
        source="mimic",
        native_rate_hz=float(record.fs),
        native_length=len(lead_I),
    )


def ingest_akbilgic_clinical_raw(npy_path: str, participant_id: str) -> RawSignal:
    """Akbilgic clinical .npy, shape (1, 12, 5000) @ 500Hz. Lead I is index 0
    (verified physiologically — see the team plan §6.1). No resample/crop/pad."""
    arr = np.load(npy_path)
    lead_I = arr[0, 0, :].astype(np.float64)

    return RawSignal(
        signal=lead_I,
        ecg_id=f"akbilgic_{participant_id}_clinical",
        participant_id=f"akbilgic_{participant_id}",
        pair_id=f"akbilgic_{participant_id}_pair",
        source="akbilgic_clinical",
        native_rate_hz=500.0,
        native_length=len(lead_I),
    )


def ingest_akbilgic_watch_raw(npy_path: str, participant_id: str) -> RawSignal:
    """Akbilgic Apple Watch .npy, shape (15360,) @ 512Hz, 30s.
    KEPT AT FULL 30s / 512Hz in the master layer — no crop here. The 10s
    ML crop happens only in derive_ml_signal(), on a separate copy."""
    arr = np.load(npy_path).astype(np.float64)

    return RawSignal(
        signal=arr,
        ecg_id=f"akbilgic_{participant_id}_watch",
        participant_id=f"akbilgic_{participant_id}",
        pair_id=f"akbilgic_{participant_id}_pair",
        source="akbilgic_watch",
        native_rate_hz=512.0,
        native_length=len(arr),
    )


# ---------------------------------------------------------------------------
# Assembly: master archive, ML matrix, and the two metadata tables
# ---------------------------------------------------------------------------

def build_master_archive(raw_signals: list[RawSignal], out_path: Path) -> None:
    """Writes every native-format signal into one .npz archive, keyed by
    ecg_id. npz supports arrays of different lengths under different keys,
    so MIMIC's 5000-sample records and the watch's 15360-sample records can
    coexist in the same file without anyone being cropped or padded to match."""
    arrays = {f"ecg_{r.ecg_id}": r.signal for r in raw_signals}
    _save_npz_atomic(out_path, arrays)  # uncompressed -- np.savez, not savez_compressed: lossless AND fast


def build_ml_matrix(ml_signals: list[MLSignal]) -> np.ndarray:
    return np.stack([m.signal for m in ml_signals])


def build_ecg_metadata(raw_signals: list[RawSignal], ml_signals: list[MLSignal]) -> pd.DataFrame:
    ml_by_id = {m.ecg_id: m for m in ml_signals}
    rows = []
    for r in raw_signals:
        m = ml_by_id[r.ecg_id]
        rows.append({
            "ECG_ID": r.ecg_id,
            "Participant_ID": r.participant_id,
            "Pair_ID": r.pair_id,
            "Source": r.source,
            "Original_Hz": r.native_rate_hz,
            "Final_Hz": ML_TARGET_RATE_HZ,
            "Original_samples": r.native_length,
            "Final_samples": ML_TARGET_LENGTH,
            "Cropped": m.cropped,
            "Padded": m.padded,
            "Padded_samples": m.padded_samples,
        })
    return pd.DataFrame(rows)


def build_participant_metadata_skeleton(raw_signals: list[RawSignal]) -> pd.DataFrame:
    """A skeleton table: one row per unique Participant_ID, demographic
    columns present but empty (NaN) until Clinical fills them in from
    record_list.csv / shared_paired_data_243.csv. Keeping the schema here
    now means ecg_metadata.csv can always be joined the same way, whether
    or not demographics have landed yet for a given source."""
    participants = sorted({r.participant_id for r in raw_signals})
    return pd.DataFrame({
        "Participant_ID": participants,
        "Age": pd.NA,
        "Sex": pd.NA,
        "Race": pd.NA,
        "Ethnicity": pd.NA,
        "Resting_ECG_Info": pd.NA,
    })


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

def run_batch(manifest_csv: str, mimic_root: str, out_prefix: str, ml_dtype=np.float32,
              progress_every: int = 500, chunk_size: int = 10_000) -> None:
    """Memory-safe version: never holds more than `chunk_size` full-length
    float64 RawSignal objects in memory at once. Each chunk is converted to
    its ML-ready form and appended to disk immediately; only small
    metadata rows accumulate across the whole run. On a machine with ~7.6GB
    RAM, holding 230,000+ native-length float64 signals simultaneously
    (multiple GB) was pushing the system into swap and risking an OOM kill
    -- this chunked approach caps memory use to roughly chunk_size records
    at a time, regardless of total manifest size.

    Master data for each chunk is appended into the same .npz-style master
    archive incrementally (via a growing dict written out at the end of
    each chunk, replacing the prior partial file atomically) so a crash
    partway through only loses the CURRENT chunk's progress, not everything
    ingested so far.
    """
    import time

    manifest = pd.read_csv(manifest_csv, dtype=str)
    total = len(manifest)
    progress_path = Path(f"{out_prefix}_progress.json")
    start_time = time.monotonic()

    ml_path = Path(f"{out_prefix}_data_ml.npy")
    ecg_meta_path = Path(f"{out_prefix}_metadata.csv")
    participant_meta_path = Path(f"{out_prefix}_participant_metadata.csv")
    failures_path = Path(f"{out_prefix}_failures.csv")
    quarantined_path = Path(f"{out_prefix}_quarantined.csv")

    # Master data is SHARDED into one file per chunk, never re-read and
    # re-written as a whole. Re-loading a growing single archive every
    # chunk (an earlier version of this function did that) defeats the
    # entire purpose of chunking, since memory use would grow right back
    # to holding everything at once by the final chunks. A small ecg_id ->
    # chunk-file index is kept instead (a few bytes per record, not the
    # signal itself), so a specific ECG can still be looked up later.
    master_chunk_dir = Path(f"{out_prefix}_master_chunks")
    master_chunk_dir.mkdir(exist_ok=True)
    master_index: dict[str, str] = {}  # ecg_id -> chunk filename
    chunk_idx = 0

    master_arrays: dict[str, np.ndarray] = {}   # accumulates ONLY across one chunk at a time, flushed to disk after each
    ml_chunks: list[np.ndarray] = []            # small float32 chunks -- much lighter than float64 masters
    all_meta_rows: list[dict] = []
    all_failures: list[dict] = []
    all_quarantined: list[dict] = []
    n_ok = 0

    def _write_progress(i: int):
        elapsed = time.monotonic() - start_time
        rate = i / elapsed if elapsed > 0 else 0.0
        remaining = (total - i) / rate if rate > 0 else None
        payload = {
            "processed": i, "total": total,
            "percent": round(100 * i / total, 2) if total else 0.0,
            "ok": n_ok, "failed": len(all_failures), "quarantined": len(all_quarantined),
            "elapsed_seconds": round(elapsed, 1), "records_per_second": round(rate, 2),
            "estimated_minutes_remaining": round(remaining / 60, 1) if remaining is not None else None,
        }
        try:
            progress_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write progress file: %s", exc)

    def _flush_master_chunk():
        """Writes THIS chunk's arrays to their own new file -- never reads
        or rewrites any prior chunk, so memory use per flush is bounded by
        chunk_size alone, no matter how many chunks came before."""
        nonlocal master_arrays, chunk_idx
        if not master_arrays:
            return
        chunk_filename = f"chunk_{chunk_idx:05d}.npz"
        chunk_path = master_chunk_dir / chunk_filename
        _save_npz_atomic(chunk_path, master_arrays)

        for ecg_key in master_arrays:
            master_index[ecg_key[len("ecg_"):]] = chunk_filename  # strip "ecg_" prefix -> plain ecg_id

        chunk_idx += 1
        master_arrays = {}

    chunk_raw: list[RawSignal] = []

    def _process_chunk():
        nonlocal chunk_raw, n_ok
        if not chunk_raw:
            return
        clean, quarantined = quarantine_non_finite(chunk_raw)
        all_quarantined.extend(quarantined)

        ml_signals = []
        finally_clean = []
        for r in clean:
            try:
                ml_signals.append(derive_ml_signal(r, crop_mode="start", ml_dtype=ml_dtype))
                finally_clean.append(r)
            except Exception as exc:  # noqa: BLE001 - one degenerate resample must not kill the batch
                all_quarantined.append({
                    "ecg_id": r.ecg_id, "participant_id": r.participant_id, "source": r.source,
                    "n_non_finite": None, "signal_length": len(r.signal),
                    "reason": f"ML derivation (resample/crop) failed: {exc}",
                })
                log.warning("quarantined %s: ML derivation failed: %s", r.ecg_id, exc)

        for r in finally_clean:
            master_arrays[f"ecg_{r.ecg_id}"] = r.signal
        if ml_signals:
            ml_chunks.append(build_ml_matrix(ml_signals))
        all_meta_rows.extend(build_ecg_metadata(finally_clean, ml_signals).to_dict("records"))
        n_ok += len(finally_clean)

        _flush_master_chunk()
        chunk_raw = []

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        study_id = str(row.study_id)
        subject_id = str(row.subject_id)
        try:
            chunk_raw.append(ingest_mimic_raw(mimic_root, subject_id=subject_id, study_id=study_id))
        except Exception as exc:  # noqa: BLE001
            all_failures.append({"subject_id": subject_id, "study_id": study_id, "error": str(exc)})
            log.warning("failed to ingest subject=%s study=%s: %s", subject_id, study_id, exc)

        if len(chunk_raw) >= chunk_size:
            _process_chunk()

        if i % progress_every == 0 or i == total:
            _write_progress(i)
            elapsed = time.monotonic() - start_time
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining_min = ((total - i) / rate / 60) if rate > 0 else None
            eta_str = f", ETA ~{remaining_min:.1f} min" if remaining_min is not None else ""
            log.info("progress: %d / %d (%.1f%%) (%d ok, %d failed, %d quarantined)%s",
                      i, total, 100 * i / total, n_ok, len(all_failures), len(all_quarantined), eta_str)

    _process_chunk()  # flush whatever's left in the final partial chunk

    log.info("ingested %d clean records, %d failures, %d quarantined, out of %d manifest rows",
              n_ok, len(all_failures), len(all_quarantined), total)

    if all_failures:
        pd.DataFrame(all_failures).to_csv(failures_path, index=False)
    if all_quarantined:
        pd.DataFrame(all_quarantined).to_csv(quarantined_path, index=False)
        log.warning("quarantined %d records for non-finite values -- see %s", len(all_quarantined), quarantined_path)

    if n_ok == 0:
        log.error("no clean records ingested — nothing to write")
        return

    ml_matrix = np.concatenate(ml_chunks, axis=0)
    verify_ml_matrix(ml_matrix)
    _save_npy_atomic(ml_path, ml_matrix)

    ecg_meta = pd.DataFrame(all_meta_rows)
    _atomic_write_bytes(ecg_meta_path, lambda tmp: ecg_meta.to_csv(tmp, index=False))

    participant_meta = pd.DataFrame({
        "Participant_ID": sorted(ecg_meta["Participant_ID"].unique()),
        "Age": pd.NA, "Sex": pd.NA, "Race": pd.NA, "Ethnicity": pd.NA, "Resting_ECG_Info": pd.NA,
    })
    _atomic_write_bytes(participant_meta_path, lambda tmp: participant_meta.to_csv(tmp, index=False))

    master_index_path = Path(f"{out_prefix}_master_index.json")
    _atomic_write_bytes(master_index_path, lambda tmp: tmp.write_text(json.dumps(master_index, indent=2)))

    # Checksum every individual master shard, not one monolithic file --
    # this lets a teammate verify any single chunk independently, and
    # verifying doesn't require loading the whole master dataset into memory.
    chunk_checksums = {
        f.name: _sha256_of_file(f) for f in sorted(master_chunk_dir.glob("chunk_*.npz"))
    }

    checksums = {
        "master_chunk_dir": str(master_chunk_dir),
        "master_index_file": str(master_index_path),
        "master_index_sha256": _sha256_of_file(master_index_path),
        "master_chunk_sha256": chunk_checksums,
        "n_master_chunks": len(chunk_checksums),
        "ml_data_file": str(ml_path), "ml_data_sha256": _sha256_of_file(ml_path),
        "ml_shape": list(ml_matrix.shape), "ml_dtype": str(ml_matrix.dtype),
        "ecg_metadata_file": str(ecg_meta_path), "ecg_metadata_sha256": _sha256_of_file(ecg_meta_path),
        "n_records": n_ok, "n_failures": len(all_failures), "n_quarantined": len(all_quarantined),
        "n_cropped": int(ecg_meta["Cropped"].sum()), "n_padded": int(ecg_meta["Padded"].sum()),
    }
    _atomic_write_bytes(Path(f"{out_prefix}_checksums.json"), lambda tmp: tmp.write_text(json.dumps(checksums, indent=2)))

    log.info("wrote %d master chunk(s) to %s, %s (%s %s), %s, %s, %s",
              len(chunk_checksums), master_chunk_dir, ml_path, ml_matrix.shape, ml_matrix.dtype,
              ecg_meta_path, participant_meta_path, master_index_path)


def run_akbilgic_batch(csv_path: str, clinical_dir: str, watch_dir: str, out_prefix: str, ml_dtype=np.float32) -> None:
    """Batch-ingests the public Akbilgic (SJLIFE) 243-participant paired
    dataset. Produces output in the exact same schema as run_batch(), under
    a different out_prefix, so the two can later be concatenated (e.g.
    pd.concat on the two ecg_metadata.csv files) for cross-vendor analysis.

    csv_path: shared_paired_data_243.csv
    clinical_dir: ClinicalECGs_full_243/
    watch_dir: AppleECGs_full_243/

    Applies the documented filename bug fix from the team plan (§6.1): the
    CSV's clinical path column reuses the "apple_ecg" filename prefix even
    for clinical files. Patched here rather than left for every caller to
    remember.
    """
    df = pd.read_csv(csv_path, dtype=str)
    raw_signals: list[RawSignal] = []
    failures: list[dict] = []

    # Column names CONFIRMED against the real shared_paired_data_243.csv header
    # on 9/25 (an earlier version of this function guessed lowercase names
    # like "participant_id"/"sex"/"race" from the team plan's prose description
    # alone -- those were wrong; this is the verified real header):
    #   ,sudoID,apple_loc_ECG_243,clinical_loc_ECG_243,gender_x,
    #   RestingECG_PatientDemographics_Race_,age_at_ecg,ethnic,
    #   RestingECG_OriginalRestingECGMeasurements_VentricularRate_,
    #   ecgAvgHR,Minutes_ABS_Btw_Clinical_Apple
    id_col = "sudoID" if "sudoID" in df.columns else df.columns[0]
    if df[id_col].duplicated().any():
        dupes = df.loc[df[id_col].duplicated(keep=False), id_col].tolist()
        raise ValueError(f"duplicate participant IDs in {csv_path}: {dupes[:10]} -- refusing to ingest "
                         f"(the master archive would silently overwrite arrays)")

    for _, row in df.iterrows():
        participant_id = str(row[id_col])  # e.g. "PID_132" -- kept as-is for traceability

        # CONFIRMED 9/25: sudoID values look like "PID_132", but the actual
        # files on disk are named "clinical_ecg_132.npy" / "apple_ecg_132.npy"
        # with NO prefix. Strip any non-digit prefix to get the bare number
        # the filenames actually use.
        numeric_id = re.sub(r"^\D+", "", participant_id)  # "PID_132" -> "132"
        if not _SAFE_ID_RE.match(numeric_id):
            failures.append({"participant_id": participant_id, "file": "N/A",
                              "error": f"could not extract a numeric id from {participant_id!r}"})
            log.warning("skipping participant=%s: no numeric id extractable", participant_id)
            continue

        clinical_filename = f"clinical_ecg_{numeric_id}.npy"
        watch_filename = f"apple_ecg_{numeric_id}.npy"
        clinical_path = Path(clinical_dir) / clinical_filename
        watch_path = Path(watch_dir) / watch_filename

        try:
            raw_signals.append(ingest_akbilgic_clinical_raw(str(clinical_path), participant_id))
        except Exception as exc:  # noqa: BLE001
            failures.append({"participant_id": participant_id, "file": str(clinical_path), "error": str(exc)})
            log.warning("failed to ingest clinical for participant=%s: %s", participant_id, exc)

        try:
            raw_signals.append(ingest_akbilgic_watch_raw(str(watch_path), participant_id))
        except Exception as exc:  # noqa: BLE001
            failures.append({"participant_id": participant_id, "file": str(watch_path), "error": str(exc)})
            log.warning("failed to ingest watch for participant=%s: %s", participant_id, exc)

    log.info("ingested %d raw records (clinical+watch), %d failures, from %d participants",
              len(raw_signals), len(failures), len(df))

    if failures:
        pd.DataFrame(failures).to_csv(f"{out_prefix}_failures.csv", index=False)

    if not raw_signals:
        log.error("no records ingested — nothing to write")
        return

    raw_signals, quarantined = quarantine_non_finite(raw_signals)
    if quarantined:
        pd.DataFrame(quarantined).to_csv(f"{out_prefix}_quarantined.csv", index=False)
        log.warning("quarantined %d Akbilgic record(s) for non-finite values -- see %s_quarantined.csv",
                    len(quarantined), out_prefix)

    # A participant with only one of the two recordings cannot form a cross-vendor
    # pair. Drop the orphan half explicitly and log it, rather than letting a
    # half-formed pair reach the pair generator.
    sources_by_pid: dict[str, set[str]] = {}
    for r in raw_signals:
        sources_by_pid.setdefault(r.participant_id, set()).add(r.source)
    orphans = {pid for pid, srcs in sources_by_pid.items()
               if srcs != {"akbilgic_clinical", "akbilgic_watch"}}
    if orphans:
        log.warning("dropping %d participant(s) missing one half of the pair: %s",
                    len(orphans), sorted(orphans))
        pd.DataFrame({"participant_id": sorted(orphans)}).to_csv(f"{out_prefix}_orphans.csv", index=False)
        raw_signals = [r for r in raw_signals if r.participant_id not in orphans]
    if not raw_signals:
        log.error("no complete clinical+watch pairs left -- nothing to write")
        return

    verify_master(raw_signals)
    ml_signals = [derive_ml_signal(r, crop_mode="middle" if r.source == "akbilgic_watch" else "start", ml_dtype=ml_dtype)
                  for r in raw_signals]
    ml_matrix = build_ml_matrix(ml_signals)
    verify_ml_matrix(ml_matrix)

    master_path = Path(f"{out_prefix}_master.npz")
    ml_path = Path(f"{out_prefix}_data_ml.npy")
    ecg_meta_path = Path(f"{out_prefix}_metadata.csv")
    participant_meta_path = Path(f"{out_prefix}_participant_metadata.csv")

    build_master_archive(raw_signals, master_path)
    _save_npy_atomic(ml_path, ml_matrix)

    ecg_meta = build_ecg_metadata(raw_signals, ml_signals)
    _atomic_write_bytes(ecg_meta_path, lambda tmp: ecg_meta.to_csv(tmp, index=False))

    # Akbilgic's own CSV has real demographics -- pull them in using the
    # CONFIRMED real column names (see the header comment above id_col).
    part_meta = build_participant_metadata_skeleton(raw_signals)
    demo_col_map = [
        ("gender_x", "Sex"),
        ("RestingECG_PatientDemographics_Race_", "Race"),
        ("ethnic", "Ethnicity"),
        ("age_at_ecg", "Age"),
        ("RestingECG_OriginalRestingECGMeasurements_VentricularRate_", "Resting_ECG_Info"),
    ]
    for col_src, col_dst in demo_col_map:
        if col_src in df.columns:
            lookup = dict(zip("akbilgic_" + df[id_col].astype(str), df[col_src]))
            part_meta[col_dst] = part_meta["Participant_ID"].map(lookup).fillna(part_meta[col_dst])
        else:
            log.warning("expected Akbilgic column %r not found in CSV -- %s will stay empty", col_src, col_dst)
    part_meta["Age"] = pd.to_numeric(part_meta["Age"], errors="coerce")  # read as text from the CSV; age bands need a number
    _atomic_write_bytes(participant_meta_path, lambda tmp: part_meta.to_csv(tmp, index=False))

    checksums = {
        "master_file": str(master_path),
        "master_sha256": _sha256_of_file(master_path),
        "ml_data_file": str(ml_path),
        "ml_data_sha256": _sha256_of_file(ml_path),
        "ml_shape": list(ml_matrix.shape),
        "ml_dtype": str(ml_matrix.dtype),
        "ecg_metadata_file": str(ecg_meta_path),
        "ecg_metadata_sha256": _sha256_of_file(ecg_meta_path),
        "n_records": len(raw_signals),
        "n_failures": len(failures),
        "n_cropped": int(ecg_meta["Cropped"].sum()),
        "n_padded": int(ecg_meta["Padded"].sum()),
    }
    _atomic_write_bytes(Path(f"{out_prefix}_checksums.json"),
                         lambda tmp: tmp.write_text(json.dumps(checksums, indent=2)))

    log.info("wrote %s, %s (%s %s), %s, %s", master_path, ml_path, ml_matrix.shape, ml_matrix.dtype,
              ecg_meta_path, participant_meta_path)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    def dominant_freq(sig, rate):
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / rate)
        mags = np.abs(np.fft.rfft(sig))
        return freqs[np.argmax(mags[1:]) + 1]

    # (1) resample_signal: 1:1 rate is a true no-op, no filtering artifact, dtype preserved
    same_rate = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = resample_signal(same_rate, 250.0, 250.0, dtype=np.float64)
    assert np.array_equal(out, same_rate) and out.dtype == np.float64, "1:1 resample should be an exact no-op"

    # (2) 500->250 preserves dominant frequency
    t500 = np.arange(5000) / 500.0
    sine500 = np.sin(2 * np.pi * 1.0 * t500).astype(np.float64)
    resampled = resample_signal(sine500, 500.0, 250.0, dtype=np.float64)
    assert len(resampled) == 2500
    assert abs(dominant_freq(sine500, 500.0) - dominant_freq(resampled, 250.0)) < 0.05

    # (3) crop_or_pad: cropped and padded are mutually exclusive and separately flagged
    long_sig = np.ones(3000, dtype=np.float64)
    cropped_sig, cropped, padded, pad_n = crop_or_pad(long_sig, 2500, crop_mode="middle")
    assert len(cropped_sig) == 2500 and cropped and not padded and pad_n == 0

    short_sig = np.ones(100, dtype=np.float64)
    padded_sig, cropped2, padded2, pad_n2 = crop_or_pad(short_sig, 2500, crop_mode="start")
    assert len(padded_sig) == 2500 and not cropped2 and padded2 and pad_n2 == 2400
    assert padded_sig[100:].sum() == 0.0

    exact_sig = np.ones(2500, dtype=np.float64)
    exact_out, cropped3, padded3, pad_n3 = crop_or_pad(exact_sig, 2500)
    assert not cropped3 and not padded3 and pad_n3 == 0

    # (4) master vs ML separation: deriving ML does not mutate the RawSignal
    raw = RawSignal(signal=np.ones(15360, dtype=np.float64), ecg_id="test_watch",
                     participant_id="p1", pair_id="p1_pair", source="akbilgic_watch",
                     native_rate_hz=512.0, native_length=15360)
    original_signal = raw.signal.copy()
    ml = derive_ml_signal(raw, crop_mode="middle")
    assert np.array_equal(raw.signal, original_signal), "deriving ML must not mutate the master RawSignal"
    assert len(ml.signal) == 2500 and ml.signal.dtype == np.float32

    verify_master([raw])

    # (5) path validation rejects traversal-style ids
    p = build_mimic_record_path("MimicDataAWS", "13158794", "48346352")
    assert str(p) == "MimicDataAWS/files/p1315/p13158794/s48346352/48346352"
    for bad_id in ("../etc/passwd", "13158794/../../x", "", "abc123"):
        try:
            build_mimic_record_path("MimicDataAWS", bad_id, "1")
            raise AssertionError(f"expected rejection of {bad_id!r}")
        except ValueError:
            pass

    # (6) metadata + master archive round-trip
    raw2 = RawSignal(signal=np.full(5000, 2.0, dtype=np.float64), ecg_id="test_mimic",
                      participant_id="p2", pair_id=None, source="mimic",
                      native_rate_hz=500.0, native_length=5000)
    ml2 = derive_ml_signal(raw2, crop_mode="start")
    meta = build_ecg_metadata([raw, raw2], [ml, ml2])
    assert set(meta.columns) >= {"ECG_ID", "Participant_ID", "Pair_ID", "Source",
                                  "Original_Hz", "Final_Hz", "Original_samples",
                                  "Final_samples", "Cropped", "Padded"}
    assert len(meta) == 2

    part_meta = build_participant_metadata_skeleton([raw, raw2])
    assert set(part_meta["Participant_ID"]) == {"p1", "p2"}

    log.info("selftest passed: no-op resample exact, freq-preservation OK, "
              "crop/pad mutually exclusive, master untouched by ML derivation, "
              "path validation OK, metadata schema OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--manifest", help="MIMIC mode: CSV with study_id,subject_id columns")
    parser.add_argument("--mimic-root", help="MIMIC mode: root dir containing the 'files/' MIMIC tree")
    parser.add_argument("--akbilgic-csv", help="Akbilgic mode: shared_paired_data_243.csv")
    parser.add_argument("--clinical-dir", help="Akbilgic mode: ClinicalECGs_full_243/ directory")
    parser.add_argument("--watch-dir", help="Akbilgic mode: AppleECGs_full_243/ directory")
    parser.add_argument("--out-prefix", default="ecg")
    parser.add_argument("--ml-dtype", default="float32", choices=["float32", "float64"],
                         help="dtype for the derived ML matrix only — the master archive is always float64")
    args = parser.parse_args()

    ml_dtype = np.float32 if args.ml_dtype == "float32" else np.float64

    if args.selftest:
        _selftest()
    elif args.manifest and args.mimic_root:
        run_batch(args.manifest, args.mimic_root, args.out_prefix, ml_dtype=ml_dtype)
    elif args.akbilgic_csv and args.clinical_dir and args.watch_dir:
        run_akbilgic_batch(args.akbilgic_csv, args.clinical_dir, args.watch_dir, args.out_prefix, ml_dtype=ml_dtype)
    else:
        parser.print_help()
