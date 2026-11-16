"""Audit the PMFC project dataset for any variables usable by a drone weight-optimization component.

Writes outputs/data_audit.json and outputs/data_audit_columns.csv.
No values are invented; this script only reports what exists.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES = [ROOT / "pmfc_fuel_cell.xlsx", ROOT / "pemfc_efficiency_clean.xlsx"]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Terms that must exist somewhere for a weight-optimization problem to be solvable.
REQUIRED_TERMS = [
    "mass", "weight", "kg", "gram", "thrust", "payload", "battery", "motor",
    "propeller", "prop", "frame", "endurance", "power_density", "cell_count",
    "capacity", "mah", "esc", "arm", "fuselage", "wing", "airframe", "drone", "uav",
]

rows, audit = [], {"files": {}, "required_term_hits": {}, "notes_verbatim": None}

for f in FILES:
    xl = pd.ExcelFile(f)
    audit["files"][f.name] = {"sheets": {}}
    for sheet in xl.sheet_names:
        df = pd.read_excel(f, sheet_name=sheet)
        cols = [str(c) for c in df.columns]
        audit["files"][f.name]["sheets"][sheet] = {
            "shape": list(df.shape),
            "n_columns": len(cols),
            "columns": cols,
        }
        for i, c in enumerate(cols):
            rows.append({"file": f.name, "sheet": sheet, "column": c,
                         "dtype": str(df.iloc[:, i].dtype),
                         "non_null": int(df.iloc[:, i].notna().sum())})

# Search column names for any mass/thrust/payload-related term
all_text = []
for fname, meta in audit["files"].items():
    for sheet, smeta in meta["sheets"].items():
        all_text += [(fname, sheet, c) for c in smeta["columns"]]

for term in REQUIRED_TERMS:
    hits = [(f, s, c) for (f, s, c) in all_text if re.search(term, c, re.I)]
    audit["required_term_hits"][term] = hits

# Pull the project's own notes (Clean_Data workbook, 'Notes' sheet)
notes = pd.read_excel(FILES[1], sheet_name="Notes", header=None)
note_lines = [str(v) for v in notes.iloc[:, 0].dropna().tolist()]
audit["notes_verbatim"] = note_lines

pd.DataFrame(rows).to_csv(OUT / "data_audit_columns.csv", index=False)
(OUT / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))

print("=== SHEETS / SHAPES ===")
for fname, meta in audit["files"].items():
    for sheet, smeta in meta["sheets"].items():
        print(f"{fname:32s} {sheet:18s} {tuple(smeta['shape'])}")

print("\n=== REQUIRED DRONE TERM HITS (column names) ===")
for term, hits in audit["required_term_hits"].items():
    print(f"{term:14s} -> {len(hits)} hit(s) {hits if hits else ''}")

print("\n=== PROJECT NOTES: drone / mass statement ===")
for line in note_lines:
    if re.search(r"drone|weight|mass|Stage", line, re.I):
        print("-", line)
