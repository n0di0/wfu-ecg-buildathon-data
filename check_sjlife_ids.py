#!/usr/bin/env python3
"""check_sjlife_ids.py -- compares participant IDs in shared_paired_data_243.csv
against the .npy files actually on disk, in both directions.

Usage:
    python3 check_sjlife_ids.py /path/to/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository

Read-only: it changes nothing. It prints, for each folder:
  - IDs in the CSV that have no file on disk   (these become ingest failures)
  - files on disk whose ID is not in the CSV   (unreferenced files)
"""
import re
import sys
from pathlib import Path

import pandas as pd

repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
csv_path = repo / "shared_paired_data_243.csv"
df = pd.read_csv(csv_path, dtype=str)
csv_ids = [re.sub(r"^\D+", "", str(s)) for s in df["sudoID"]]
print(f"CSV rows: {len(df)} | unique IDs: {len(set(csv_ids))}")

for folder, prefix in [("ClinicalECGs_full_243", "clinical_ecg_"), ("AppleECGs_full_243", "apple_ecg_")]:
    names = sorted(p.name for p in (repo / folder).iterdir())
    disk = {n[len(prefix):-4] for n in names if n.startswith(prefix) and n.endswith(".npy")}
    other = [n for n in names if not (n.startswith(prefix) and n.endswith(".npy"))]
    print(f"\n{folder}: {len(names)} entries, {len(disk)} match the {prefix}N.npy pattern")
    if other:
        print("  entries that don't match the pattern:", other)
    print("  in CSV, no file on disk:", sorted(set(csv_ids) - disk, key=lambda x: int(x) if x.isdigit() else -1))
    print("  on disk, not in CSV:   ", sorted(disk - set(csv_ids), key=lambda x: int(x) if x.isdigit() else -1))
