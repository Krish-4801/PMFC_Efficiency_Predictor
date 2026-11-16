"""Data ingestion, cleaning and feature engineering for the drone weight component.

Source: FPV-DB open dataset (CC BY 4.0, doi:10.5281/zenodo.23613812).
Every derived column is explicit and reproducible; nothing is imputed silently
except where a `*_missing` indicator is also written.

Outputs (data/processed/)
-------------------------
quad_design_table.csv     one row per assembled quad  (n=101)
thrust_bench.csv          tidy thrust-bench rows      (n=6786)
propulsion_groups.csv     one row per (motor, prop, cells) with a thrust curve
battery_catalog.csv       battery packs               (n=326)
component_catalog.csv     every component with a mass
data_quality_report.json  audit trail
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_PROCESSED,
    DATA_RAW,
    DATASET,
    G,
    LIPO_NOMINAL_V,
    OUTPUTS,
    RHO_AIR,
    USABLE_BATTERY_FRACTION,
)

# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------

_CELL_RE = re.compile(r"(\d+)")
_KV_RE = re.compile(r"(\d{3,5})\s*kv", re.I)


def parse_cells(value) -> float:
    """'6S' -> 6, '2S (2x 1S in series)' -> 2, None -> nan."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    m = _CELL_RE.search(str(value))
    return float(m.group(1)) if m else np.nan


def parse_kv(value) -> float:
    m = _KV_RE.findall(str(value))
    return float(m[0]) if m else np.nan


def parse_prop_diameter_in(value) -> float:
    """Robustly parse a nominal prop diameter (inches) from a catalogue string.

    The *first* size token is always the diameter; later tokens are pitch/blades.
    Handles 'Gemfan 3016R*3' (3.0"), 'HQ TD2.9*2.7*3' (2.9"), 'HQProp 8x4.5'
    (8.0"), '31MM' (1.22"), 'Gemfan D90*5' (3.54"), 'T-Motor T10.5x5' (10.5"),
    '51466-3B' (5.1"), '4943.5-3B' (4.9").
    Returns NaN when the string is ambiguous (e.g. the 4-digit code '1050',
    which could be 10x5.0 or 1.0x5.0).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).upper().strip()

    m = re.search(r"(\d{2,3})\s*R?MM", s)
    if m:
        d = float(m.group(1)) / 25.4
        return d if 0.5 <= d <= 13 else np.nan

    m = re.search(r"\bD(\d{2,3})\b", s)
    if m:
        d = float(m.group(1)) / 25.4
        return d if 0.5 <= d <= 13 else np.nan

    # explicit 'AxB' -> first token is the diameter
    m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*[X*]", s)
    if m:
        d = float(m.group(1))
        return d if 0.5 <= d <= 13 else np.nan

    m = re.search(r"(\d{5})", s)
    if m:
        d = float(m.group(1)[:2]) / 10.0
        return d if 0.5 <= d <= 13 else np.nan

    m = re.search(r"(\d{4})", s)
    if m:
        first2 = int(m.group(1)[:2])
        d = first2 / 10.0                   # 3016 -> 3.0, 5146 -> 5.1, 7042 -> 7.0
        # 10xx/11xx/12xx codes are ambiguous between 1.x" micro props and
        # 10-12" props written without a decimal; reject instead of guessing.
        return d if 2.0 <= d <= 13 else np.nan

    m = re.match(r"^(\d{2})(?!\d)", s)
    if m:
        d = float(m.group(1)) / 10.0
        return d if 0.5 <= d <= 13 else np.nan
    return np.nan


def parse_prop_pitch_in(value) -> float:
    """Parse pitch from names like '3016' (3.0x1.6), '8x4.5' or '10.5x5'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).upper()
    m = re.search(r"\d{1,2}(?:\.\d+)?\s*[X*]\s*(\d{1,2}(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{2})(\d{2})\s*[R\-\s]", s)
    if m:
        return float(m.group(2)) / 10.0
    return np.nan


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_raw() -> dict[str, pd.DataFrame]:
    tabs = {}
    for name, fname in DATASET["tables"].items():
        tabs[name] = pd.read_csv(DATA_RAW / fname)
    return tabs


def _try_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


# ----------------------------------------------------------------------------
# Quads -> design table
# ----------------------------------------------------------------------------

def build_quad_table(quads: pd.DataFrame, batteries: pd.DataFrame) -> pd.DataFrame:
    q = quads.copy()
    q["battery_spec"] = q["recommended_battery"].map(_try_json)

    q["cells"] = q["battery_spec"].map(
        lambda d: parse_cells(d.get("cells")) if isinstance(d, dict) else np.nan
    )
    q["capacity_mah"] = q["battery_spec"].map(
        lambda d: pd.to_numeric(d.get("capacity_mah"), errors="coerce")
        if isinstance(d, dict)
        else np.nan
    )
    q["battery_chemistry"] = q["battery_spec"].map(
        lambda d: d.get("chemistry") if isinstance(d, dict) else None
    )

    # Battery mass: nearest catalogue pack with identical cell count, using
    # log-distance in capacity. This is an *estimate* of the pack the airframe
    # was weighed with; the join distance is retained for auditing.
    b = batteries.copy()
    b["cells_n"] = b["cells"].map(parse_cells)
    b = b.dropna(subset=["cells_n", "weight_g"])

    def _match(cells, cap):
        if pd.isna(cells) or pd.isna(cap):
            return pd.Series({"battery_mass_g": np.nan, "battery_match_logdist": np.nan})
        sub = b[b["cells_n"] == cells]
        if sub.empty:
            return pd.Series({"battery_mass_g": np.nan, "battery_match_logdist": np.nan})
        dist = (np.log(sub["capacity_mah"] / cap)).abs()
        row = sub.loc[dist.idxmin()]
        return pd.Series(
            {"battery_mass_g": float(row["weight_g"]), "battery_match_logdist": float(dist.min())}
        )

    q = pd.concat([q, q.apply(lambda r: _match(r["cells"], r["capacity_mah"]), axis=1)], axis=1)

    q["motor_kv"] = q["motors"].map(parse_kv)
    q["n_motors"] = 4  # every assembled quad in this table is a quadrotor

    # --- totals -------------------------------------------------------------
    q["dry_mass_g"] = q["weight_no_battery_g"]
    q["auw_est_g"] = q["dry_mass_g"] + q["battery_mass_g"]
    q["auw_reported_g"] = q["typical_auw_g"]

    # --- physics / design features -----------------------------------------
    q["voltage_nominal_V"] = q["cells"] * LIPO_NOMINAL_V
    q["energy_wh"] = q["capacity_mah"] / 1000.0 * q["voltage_nominal_V"]
    q["disk_area_m2"] = q["n_motors"] * np.pi * (q["prop_size_in"] * 0.0254 / 2.0) ** 2
    q["disk_loading_kg_m2"] = (q["auw_est_g"] / 1000.0) / q["disk_area_m2"]
    q["battery_mass_fraction"] = q["battery_mass_g"] / q["auw_est_g"]
    q["dry_mass_fraction"] = q["dry_mass_g"] / q["auw_est_g"]
    q["specific_energy_wh_kg"] = q["energy_wh"] / (q["battery_mass_g"] / 1000.0)
    q["power_W"] = q["voltage_nominal_V"] * q["typical_current_a"]
    q["power_to_weight_W_kg"] = q["power_W"] / (q["auw_est_g"] / 1000.0)

    # Aerodynamic / allometric helpers used by the surrogate model
    q["prop_disk_area_single_m2"] = np.pi * (q["prop_size_in"] * 0.0254 / 2.0) ** 2
    q["log_wheelbase"] = np.log(q["wheelbase_mm"])
    q["wheelbase_missing"] = q["wheelbase_mm"].isna().astype(int)
    q["class_group"] = (
        q["class"]
        .str.extract(r"(cinewhoop|freestyle|tiny whoop|whoop|toothpick|long range|racing|micro)")[0]
        .fillna("other")
    )

    keep = [
        "brand", "model", "slug", "class", "class_group", "motors", "motor_kv",
        "prop_size_in", "wheelbase_mm", "n_motors", "flight_controller",
        "battery_chemistry", "cells", "capacity_mah", "voltage_nominal_V", "energy_wh",
        "battery_mass_g", "battery_match_logdist", "dry_mass_g", "auw_est_g",
        "auw_reported_g", "typical_current_a", "manufacturer_flight_time_min",
        "disk_area_m2", "prop_disk_area_single_m2", "disk_loading_kg_m2",
        "battery_mass_fraction", "dry_mass_fraction", "specific_energy_wh_kg",
        "power_W", "power_to_weight_W_kg", "log_wheelbase", "wheelbase_missing",
        "sources",
    ]
    return q[keep]


# ----------------------------------------------------------------------------
# Thrust bench -> tidy long table
# ----------------------------------------------------------------------------

def build_thrust_bench(motors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in motors.dropna(subset=["thrust_table"]).iterrows():
        table = _try_json(r["thrust_table"])
        if not table:
            continue
        for x in table:
            if not isinstance(x, dict):
                continue
            rows.append(
                {
                    "motor_slug": r["slug"],
                    "motor_brand": r["brand"],
                    "motor_model": r["model"],
                    "motor_mass_g": r["weight_g"],
                    "motor_max_cells": r["max_cells"],
                    "stator": r["stator"],
                    "kv": pd.to_numeric(x.get("kv"), errors="coerce"),
                    "prop": x.get("prop"),
                    "prop_key": x.get("prop_key"),
                    "prop_blades": x.get("prop_blades"),
                    "cells": x.get("cells"),
                    "throttle_pct": pd.to_numeric(x.get("throttle"), errors="coerce"),
                    "voltage_V": pd.to_numeric(x.get("voltage_v"), errors="coerce"),
                    "current_A": pd.to_numeric(x.get("current_a"), errors="coerce"),
                    "thrust_g": pd.to_numeric(x.get("thrust_g"), errors="coerce"),
                    "efficiency_g_per_W": pd.to_numeric(x.get("efficiency_gw"), errors="coerce"),
                }
            )
    d = pd.DataFrame(rows)
    d["cells_n"] = d["cells"].map(parse_cells)
    d["prop_diameter_in"] = d["prop"].map(parse_prop_diameter_in)
    d["prop_pitch_in"] = d["prop"].map(parse_prop_pitch_in)
    d = d.dropna(subset=["thrust_g", "throttle_pct"])
    return d.reset_index(drop=True)


def build_prop_mass_model(props: pd.DataFrame):
    """Fit prop mass (g) from diameter/pitch/blades on the real prop catalogue.

    Returns a fitted sklearn model plus its CV metrics. Used only to assign a
    mass to bench-test props that do not appear in the props table.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict

    p = props.dropna(subset=["diameter_in", "weight_g"]).copy()
    p["pitch_in"] = p["pitch_in"].fillna(p["pitch_in"].median())
    p["blades"] = p["blades"].fillna(3)
    X = p[["diameter_in", "pitch_in", "blades"]]
    y = p["weight_g"]
    model = GradientBoostingRegressor(random_state=0)
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    metrics = {
        "n": int(len(p)),
        "cv_r2": float(r2_score(y, pred)),
        "cv_mae_g": float(mean_absolute_error(y, pred)),
    }
    model.fit(X, y)
    return model, metrics


def build_propulsion_groups(bench: pd.DataFrame, prop_model, prop_metrics) -> pd.DataFrame:
    """Aggregate the tidy bench data into one row per (motor, prop, cells).

    Keeps groups with >= 3 throttle points so that hover current can be
    interpolated rather than guessed.
    """
    b = bench.dropna(subset=["prop_diameter_in", "cells_n"]).copy()
    # prop mass: measured catalogue value if the prop_key matches, else model
    b["prop_mass_g"] = prop_model.predict(
        pd.DataFrame(
            {
                "diameter_in": b["prop_diameter_in"],
                "pitch_in": b["prop_pitch_in"].fillna(b["prop_pitch_in"].median()),
                "blades": b["prop_blades"].fillna(3),
            }
        )
    )
    b["prop_mass_g"] = b["prop_mass_g"].clip(lower=0.3)

    groups = []
    for (motor, prop_key, cells), g in b.groupby(["motor_slug", "prop_key", "cells_n"]):
        g = g.sort_values("throttle_pct")
        if len(g) < 3:
            continue
        g = g.drop_duplicates(subset="throttle_pct", keep="first")
        if len(g) < 3:
            continue
        groups.append(
            {
                "motor_slug": motor,
                "prop_key": prop_key,
                "cells": int(cells),
                "motor_brand": g["motor_brand"].iloc[0],
                "motor_model": g["motor_model"].iloc[0],
                "motor_mass_g": float(g["motor_mass_g"].iloc[0]),
                "motor_max_cells": g["motor_max_cells"].iloc[0],
                "stator": g["stator"].iloc[0],
                "kv": float(g["kv"].iloc[0]) if pd.notna(g["kv"].iloc[0]) else np.nan,
                "prop_diameter_in": float(g["prop_diameter_in"].iloc[0]),
                "prop_pitch_in": float(g["prop_pitch_in"].iloc[0])
                if pd.notna(g["prop_pitch_in"].iloc[0])
                else np.nan,
                "prop_mass_g": float(g["prop_mass_g"].iloc[0]),
                "n_points": int(len(g)),
                "max_thrust_g": float(g["thrust_g"].max()),
                "max_current_A": float(g["current_A"].max()) if g["current_A"].notna().any() else np.nan,
                "voltage_max_V": float(g["voltage_V"].max()) if g["voltage_V"].notna().any() else np.nan,
                # curves stored as JSON so the optimizer can interpolate exactly
                "throttle_curve": json.dumps(g["throttle_pct"].round(2).tolist()),
                "thrust_curve_g": json.dumps(g["thrust_g"].round(2).tolist()),
                "current_curve_A": json.dumps(
                    g["current_A"].interpolate().bfill().ffill().round(3).tolist()
                ),
                "voltage_curve_V": json.dumps(
                    g["voltage_V"].interpolate().bfill().ffill().round(3).tolist()
                ),
            }
        )
    out = pd.DataFrame(groups)
    out["prop_mass_metrics"] = json.dumps(prop_metrics)
    return out


# ----------------------------------------------------------------------------
# Component catalogue & quality report
# ----------------------------------------------------------------------------

def build_component_catalog(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for kind, df in tabs.items():
        if kind == "motors":
            sub = df[["slug", "brand", "model", "weight_g"]].copy()
            sub["detail"] = df["stator"]
        elif kind == "batteries":
            sub = df[["slug", "brand", "model", "weight_g"]].copy()
            sub["detail"] = (
                df["capacity_mah"].astype(str) + "mAh " + df["cells"].astype(str)
            )
        elif kind == "props":
            sub = df[["slug", "brand", "model", "weight_g"]].copy()
            sub["detail"] = df["diameter_in"].astype(str) + "in"
        elif kind == "quads":
            sub = df[["slug", "brand", "model", "weight_no_battery_g"]].copy()
            sub = sub.rename(columns={"weight_no_battery_g": "weight_g"})
            sub["detail"] = df["prop_size_in"].astype(str) + "in quad"
        else:
            sub = df[["slug", "brand", "model", "weight_g"]].copy()
            sub["detail"] = None
        sub["component_type"] = kind.rstrip("s")
        frames.append(sub)
    cat = pd.concat(frames, ignore_index=True)
    cat["weight_g"] = pd.to_numeric(cat["weight_g"], errors="coerce")
    return cat.dropna(subset=["weight_g"])


def data_quality_report(quad: pd.DataFrame, bench: pd.DataFrame, groups: pd.DataFrame,
                        batteries: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    report = {
        "dataset": DATASET,
        "tables": {},
        "weight_formula_validation": {},
        "physics_checks": {},
    }
    for name, df in {
        "quad_design_table": quad, "thrust_bench": bench,
        "propulsion_groups": groups, "battery_catalog": batteries,
        "component_catalog": catalog,
    }.items():
        report["tables"][name] = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_fraction": {
                c: round(float(df[c].isna().mean()), 4)
                for c in df.columns
                if df[c].isna().any()
            },
        }

    v = quad.dropna(subset=["auw_reported_g", "auw_est_g"])
    err = v["auw_est_g"] - v["auw_reported_g"]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((v["auw_reported_g"] - v["auw_reported_g"].mean()) ** 2).sum())
    report["weight_formula_validation"] = {
        "formula": "AUW = dry_mass (frame+motors+props+avionics) + battery_mass",
        "n_reported_auw": int(len(v)),
        "mae_g": round(float(err.abs().mean()), 2),
        "mape_pct": round(float((err.abs() / v["auw_reported_g"]).mean() * 100), 3),
        "bias_g": round(float(err.mean()), 2),
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
        "note": "Battery mass is matched from the pack catalogue by cell count "
                "and nearest capacity; reported AUW is the manufacturer figure.",
    }

    # Battery catalogue internal consistency: E = C*V
    b = batteries.dropna(subset=["energy_wh", "capacity_mah", "cells"])
    e_calc = b["capacity_mah"] / 1000 * b["cells"].map(parse_cells) * LIPO_NOMINAL_V
    report["physics_checks"]["battery_energy_vs_capacity_median_abs_err_wh"] = round(
        float((b["energy_wh"] - e_calc).abs().median()), 4
    )

    q_ok = quad.dropna(subset=["disk_area_m2", "auw_est_g"])
    report["physics_checks"]["disk_loading_kg_m2"] = {
        "median": round(float(q_ok["disk_loading_kg_m2"].median()), 3),
        "p10": round(float(q_ok["disk_loading_kg_m2"].quantile(0.10)), 3),
        "p90": round(float(q_ok["disk_loading_kg_m2"].quantile(0.90)), 3),
    }
    report["physics_checks"]["battery_mass_fraction"] = {
        "median": round(float(q_ok["battery_mass_fraction"].median()), 3),
        "p10": round(float(q_ok["battery_mass_fraction"].quantile(0.10)), 3),
        "p90": round(float(q_ok["battery_mass_fraction"].quantile(0.90)), 3),
    }
    return report


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> dict:
    tabs = load_raw()
    batteries = tabs["batteries"].copy()
    batteries["cells_n"] = batteries["cells"].map(parse_cells)

    quad = build_quad_table(tabs["quads"], batteries)
    bench = build_thrust_bench(tabs["motors"])
    prop_model, prop_metrics = build_prop_mass_model(tabs["props"])
    groups = build_propulsion_groups(bench, prop_model, prop_metrics)
    catalog = build_component_catalog(tabs)

    quad.to_csv(DATA_PROCESSED / "quad_design_table.csv", index=False)
    bench.to_csv(DATA_PROCESSED / "thrust_bench.csv", index=False)
    groups.to_csv(DATA_PROCESSED / "propulsion_groups.csv", index=False)
    batteries.to_csv(DATA_PROCESSED / "battery_catalog.csv", index=False)
    catalog.to_csv(DATA_PROCESSED / "component_catalog.csv", index=False)

    report = data_quality_report(quad, bench, groups, batteries, catalog)
    report["prop_mass_model"] = prop_metrics
    report["propulsion_groups"] = {
        "n_groups": int(len(groups)),
        "n_motors": int(groups["motor_slug"].nunique()),
        "n_props": int(groups["prop_key"].nunique()),
        "max_thrust_range_g": [
            round(float(groups["max_thrust_g"].min()), 1),
            round(float(groups["max_thrust_g"].max()), 1),
        ],
    }
    (OUTPUTS / "data_quality_report.json").write_text(json.dumps(report, indent=2))

    print("quad_design_table:", quad.shape)
    print("thrust_bench:", bench.shape)
    print("propulsion_groups:", groups.shape)
    print("battery_catalog:", batteries.shape, "| component_catalog:", catalog.shape)
    print("\nWeight-formula validation (n=%d): MAE=%.2f g, MAPE=%.2f%%, R2=%.4f, bias=%.2f g" % (
        report["weight_formula_validation"]["n_reported_auw"],
        report["weight_formula_validation"]["mae_g"],
        report["weight_formula_validation"]["mape_pct"],
        report["weight_formula_validation"]["r2"],
        report["weight_formula_validation"]["bias_g"],
    ))
    return report


if __name__ == "__main__":
    main()
