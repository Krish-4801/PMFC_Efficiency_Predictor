"""Central configuration for the drone weight-optimization component.

All paths, physical constants, and design scenarios live here so every
notebook/script uses the same values.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs" / "drone_weight"
FIGURES = OUTPUTS / "figures"
REPORTS = ROOT / "reports"

for _p in (DATA_PROCESSED, OUTPUTS, FIGURES, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Physical constants (SI unless stated)
# ----------------------------------------------------------------------------
RHO_AIR = 1.225          # kg/m^3, ISA sea level
G = 9.80665              # m/s^2
LIPO_NOMINAL_V = 3.7     # V per cell
USABLE_BATTERY_FRACTION = 0.80  # depth of discharge actually used in flight

# Design requirement: minimum static thrust-to-weight ratio.
# 2.0 is the conventional minimum for controllability margin in multirotors
# (1.5-2.5 typical). Treated as a *design requirement*, not a data-derived value.
MIN_TWR = 2.0
N_ROTORS = 4

# Hard cap to avoid extrapolating the weight model beyond the observed fleet.
MAX_AUW_KG = 2.5

# ----------------------------------------------------------------------------
# Dataset provenance
# ----------------------------------------------------------------------------
DATASET = {
    "name": "FPV-DB: open dataset of FPV drone component specifications",
    "publisher": "FPV-DB / Zenodo",
    "doi": "10.5281/zenodo.23613812",
    "url": "https://github.com/fpvdb/fpv-db-data",
    "license": "CC BY 4.0",
    "accessed": "2026-11-16",
    "tables": {
        "quads": "quads.csv",
        "motors": "motors.csv",
        "batteries": "batteries.csv",
        "props": "props.csv",
        "stacks": "stacks.csv",
        "cameras": "cameras.csv",
        "vtx": "vtx.csv",
    },
}

# ----------------------------------------------------------------------------
# Optimization scenarios.
#
# Bare-frame mass is NOT present in FPV-DB (see reports/METHODOLOGY.md).
# It is therefore an explicit design input. `airframe_g` is the mass of the
# frame + flight controller/ESC + video system + wiring, estimated with the
# data-residual method documented in the methodology report; it is NOT a
# measured column in the dataset. Sensitivity is run on this parameter.
# ----------------------------------------------------------------------------
SCENARIOS = {
    "3inch": {
        "label": "3-inch micro class",
        "min_prop_in": 2.5,
        "max_prop_in": 3.5,
        "payload_kg": 0.20,
        "airframe_g": 95.0,
        "endurance_min": 8.0,
    },
    "5inch": {
        "label": "5-inch freestyle / cinematic class",
        "min_prop_in": 4.5,
        "max_prop_in": 5.5,
        "payload_kg": 0.50,
        "airframe_g": 245.0,
        "endurance_min": 10.0,
    },
    "7inch": {
        "label": "7-inch long-range / heavy-lift class",
        "min_prop_in": 6.5,
        "max_prop_in": 7.5,
        "payload_kg": 0.75,
        "airframe_g": 450.0,
        "endurance_min": 12.0,
    },
}
PRIMARY_SCENARIO = "5inch"

RANDOM_STATE = 42
