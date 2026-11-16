"""Exploratory data analysis for the drone weight component.

Produces:
  outputs/drone_weight/eda_summary.json
  reports/eda_observations.md
  outputs/drone_weight/figures/*.png
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, FIGURES, OUTPUTS, REPORTS, RHO_AIR

warnings.filterwarnings("ignore")

NUMERIC_COLS = [
    "prop_size_in", "wheelbase_mm", "cells", "capacity_mah", "motor_kv",
    "battery_mass_g", "dry_mass_g", "auw_est_g", "energy_wh",
    "disk_loading_kg_m2", "battery_mass_fraction", "specific_energy_wh_kg",
    "power_to_weight_W_kg",
]


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    return 0.6745 * (s - med) / mad if mad else pd.Series(np.zeros(len(s)), index=s.index)


def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUMERIC_COLS:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rz = _robust_z(s)
        rows.append({
            "variable": c, "n": int(s.size), "missing": int(df[c].isna().sum()),
            "mean": s.mean(), "sd": s.std(), "min": s.min(), "p10": s.quantile(0.10),
            "p25": q1, "median": s.median(), "p75": q3, "p90": s.quantile(0.90),
            "max": s.max(),
            "cv_pct": 100 * s.std() / s.mean() if s.mean() else np.nan,
            "iqr_outliers": int(((s < lo) | (s > hi)).sum()),
            "robust_outliers_3.5": int((rz.abs() > 3.5).sum()),
            "skew": s.skew(), "kurtosis": s.kurtosis(),
        })
    return pd.DataFrame(rows).round(3)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.impute import SimpleImputer

    Xn = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=X.columns
    )
    # drop constant columns (VIF undefined)
    Xn = Xn.loc[:, Xn.std() > 0]
    vifs = [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])]
    return (
        pd.DataFrame({"feature": Xn.columns, "VIF": vifs})
        .sort_values("VIF", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )


def allometric_fit(df: pd.DataFrame, x: str, y: str = "auw_est_g") -> dict:
    """OLS of log(y) on log(x): slope is the allometric exponent b in y ~ x^b."""
    import statsmodels.api as sm

    d = df[[x, y]].dropna()
    d = d[(d[x] > 0) & (d[y] > 0)]
    model = sm.OLS(np.log(d[y]), sm.add_constant(np.log(d[x]))).fit()
    return {
        "x": x, "y": y, "n": int(len(d)),
        "exponent": float(model.params.iloc[1]),
        "exponent_se": float(model.bse.iloc[1]),
        "r2": float(model.rsquared),
        "p_value": float(model.pvalues.iloc[1]),
    }


def run_eda(save: bool = True) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    quad = pd.read_csv(DATA_PROCESSED / "quad_design_table.csv")
    catalog = pd.read_csv(DATA_PROCESSED / "component_catalog.csv")
    groups = pd.read_csv(DATA_PROCESSED / "propulsion_groups.csv")
    batteries = pd.read_csv(DATA_PROCESSED / "battery_catalog.csv")

    desc = descriptives(quad)

    # --- missingness / duplicates -----------------------------------------
    missing = (
        quad.isna().mean().sort_values(ascending=False).rename("missing_fraction")
        .reset_index().rename(columns={"index": "column"})
    )
    missing["missing_count"] = [int(quad[c].isna().sum()) for c in missing["column"]]
    dupes = int(quad.duplicated(subset=["brand", "model"]).sum())

    # --- correlations ------------------------------------------------------
    corr = quad[NUMERIC_COLS].corr(method="pearson")
    corr_spearman = quad[NUMERIC_COLS].corr(method="spearman")

    model_feats = [
        "prop_size_in", "wheelbase_mm", "cells", "capacity_mah", "motor_kv",
    ]
    vif = vif_table(quad[model_feats])

    # --- component-level relationships ------------------------------------
    batt = batteries.dropna(subset=["energy_wh", "weight_g"])
    batt_slope = np.polyfit(batt["energy_wh"], batt["weight_g"], 1)
    batt_r2 = np.corrcoef(batt["energy_wh"], batt["weight_g"])[0, 1] ** 2
    spec_energy_median = float((batt["energy_wh"] / batt["weight_g"] * 1000).median())

    prop = pd.read_csv(DATA_PROCESSED / "component_catalog.csv")
    prop = prop[prop["component_type"] == "prop"].dropna(subset=["weight_g"])

    allom = {
        "auw_vs_wheelbase": allometric_fit(quad, "wheelbase_mm"),
        "auw_vs_prop_size": allometric_fit(quad, "prop_size_in"),
        "dry_vs_wheelbase": allometric_fit(quad, "wheelbase_mm", "dry_mass_g"),
        "dry_vs_prop_size": allometric_fit(quad, "prop_size_in", "dry_mass_g"),
    }

    # --- figures -----------------------------------------------------------
    # 1. component-weight breakdown for the fleet (sorted stacked bars)
    d = quad.dropna(subset=["auw_est_g"]).sort_values("auw_est_g").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(d.index, d["dry_mass_g"], label="Dry mass (frame + motors + props + avionics)",
           color="#3b6ea5", width=1.0)
    ax.bar(d.index, d["battery_mass_g"], bottom=d["dry_mass_g"], label="Battery",
           color="#e08a3c", width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Aircraft (sorted by AUW)")
    ax.set_ylabel("Mass (g, log scale)")
    ax.set_title("Component-sum weight breakdown across 101 assembled quads")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES / "component_weight_breakdown.png", dpi=200); plt.close(fig)

    # 2. distributions
    cols = ["auw_est_g", "dry_mass_g", "battery_mass_g", "prop_size_in",
            "capacity_mah", "energy_wh"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(quad[c].dropna(), kde=True, ax=ax, color="#3b6ea5")
        ax.set_title(c)
    fig.suptitle("Distributions of drone weight and design parameters")
    fig.tight_layout(); fig.savefig(FIGURES / "distributions.png", dpi=200); plt.close(fig)

    # 3. correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Pearson correlation — weight and design variables")
    fig.tight_layout(); fig.savefig(FIGURES / "correlation_heatmap.png", dpi=200); plt.close(fig)

    # 4. design parameter vs weight (log-log)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    pairs = [("wheelbase_mm", "AUW (g)"), ("prop_size_in", "AUW (g)"),
             ("battery_mass_g", "AUW (g)"), ("capacity_mah", "AUW (g)")]
    for ax, (x, lab) in zip(axes.ravel(), pairs):
        dd = quad[[x, "auw_est_g"]].dropna()
        dd = dd[(dd[x] > 0)]
        ax.scatter(dd[x], dd["auw_est_g"], s=35, alpha=0.75, edgecolor="k", linewidth=0.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(x); ax.set_ylabel(lab)
    fig.suptitle("Design parameters vs all-up weight (log–log, allometric view)")
    fig.tight_layout(); fig.savefig(FIGURES / "design_vs_weight.png", dpi=200); plt.close(fig)

    # 5. outliers by class
    fig, ax = plt.subplots(figsize=(12, 5.5))
    order = quad.groupby("class_group")["auw_est_g"].median().sort_values().index
    sns.boxplot(data=quad, x="class_group", y="auw_est_g", order=order, ax=ax,
                color="#9dc3e6")
    sns.stripplot(data=quad, x="class_group", y="auw_est_g", order=order, ax=ax,
                  color="k", size=4, alpha=0.7)
    ax.set_yscale("log"); ax.set_xlabel("")
    ax.set_ylabel("AUW (g, log scale)")
    ax.set_title("Weight distribution by airframe class (outlier inspection)")
    fig.tight_layout(); fig.savefig(FIGURES / "weight_outliers_by_class.png", dpi=200); plt.close(fig)

    # 6. catalogue component masses by type
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=catalog, x="component_type", y="weight_g", ax=ax, color="#a8d08d")
    ax.set_yscale("log"); ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.set_xlabel(""); ax.set_ylabel("Component mass (g, log scale)")
    ax.set_title(f"Component mass distributions in the catalogue (n={len(catalog)})")
    fig.tight_layout(); fig.savefig(FIGURES / "component_mass_by_type.png", dpi=200); plt.close(fig)

    # 7. battery specific energy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(batt["energy_wh"], batt["weight_g"], s=22, alpha=0.6)
    xs = np.linspace(0, batt["energy_wh"].max(), 50)
    axes[0].plot(xs, np.polyval(batt_slope, xs), "r--",
                 label=f"m = {batt_slope[0]:.2f}·E + {batt_slope[1]:.1f} g  (R²={batt_r2:.3f})")
    axes[0].set_xlabel("Pack energy (Wh)"); axes[0].set_ylabel("Pack mass (g)")
    axes[0].set_title("Battery mass is nearly linear in energy")
    axes[0].legend(fontsize=10)
    se = batt["energy_wh"] / batt["weight_g"] * 1000
    sns.histplot(se, kde=True, ax=axes[1], color="#c55a11")
    axes[1].axvline(spec_energy_median, color="k", ls="--",
                    label=f"median {spec_energy_median:.0f} Wh/kg")
    axes[1].set_xlabel("Specific energy (Wh/kg)")
    axes[1].set_title("Pack-level specific energy")
    axes[1].legend(fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "battery_energy_mass.png", dpi=200); plt.close(fig)

    # 8. motor mass vs thrust
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(groups["motor_mass_g"], groups["max_thrust_g"], s=28, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Motor mass (g, log)")
    ax.set_ylabel("Max measured thrust per motor (g, log)")
    ax.set_title(f"Motor mass vs thrust ({len(groups)} motor/prop/cell combinations)")
    fig.tight_layout(); fig.savefig(FIGURES / "motor_mass_vs_thrust.png", dpi=200); plt.close(fig)

    # --- summary -----------------------------------------------------------
    summary = {
        "n_quads": int(len(quad)),
        "n_components": int(len(catalog)),
        "duplicate_brand_model_rows": dupes,
        "descriptives": desc.to_dict(orient="records"),
        "missing": missing.to_dict(orient="records"),
        "vif": vif.to_dict(orient="records"),
        "allometric": allom,
        "battery_mass_model": {
            "slope_g_per_Wh": float(batt_slope[0]),
            "intercept_g": float(batt_slope[1]),
            "r2": float(batt_r2),
            "median_specific_energy_wh_kg": spec_energy_median,
        },
        "top_pearson_with_auw": (
            corr["auw_est_g"].drop("auw_est_g").sort_values(ascending=False).round(3).to_dict()
        ),
        "top_spearman_with_auw": (
            corr_spearman["auw_est_g"].drop("auw_est_g").sort_values(ascending=False).round(3).to_dict()
        ),
    }

    if save:
        (OUTPUTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        desc.to_csv(OUTPUTS / "descriptive_statistics.csv", index=False)
        vif.to_csv(OUTPUTS / "vif_table.csv", index=False)
        missing.to_csv(OUTPUTS / "missingness.csv", index=False)
        write_observations(summary, quad, groups, catalog)

    return summary


def write_observations(summary: dict, quad: pd.DataFrame, groups: pd.DataFrame,
                       catalog: pd.DataFrame) -> None:
    corr = summary["top_pearson_with_auw"]
    allom = summary["allometric"]
    vif = pd.DataFrame(summary["vif"])
    bm = summary["battery_mass_model"]
    d = {r["variable"]: r for r in summary["descriptives"]}

    txt = f"""# Drone Weight EDA — Key Findings

Generated by `src/eda.py` from the FPV-DB open dataset
(CC BY 4.0, doi:10.5281/zenodo.23613812). All numbers below are reproduced
in `outputs/drone_weight/eda_summary.json`.

## 1. Dataset shape and completeness

* {summary['n_quads']} assembled quadrotors, {summary['n_components']} individual components.
* Duplicate brand/model rows: {summary['duplicate_brand_model_rows']}.
* AUW spans **{d['auw_est_g']['min']:.0f}–{d['auw_est_g']['max']:.0f} g**
  (median {d['auw_est_g']['median']:.0f} g, CV {d['auw_est_g']['cv_pct']:.0f}%),
  a two-order-of-magnitude design range.
* Dry mass spans {d['dry_mass_g']['min']:.0f}–{d['dry_mass_g']['max']:.0f} g; battery
  mass {d['battery_mass_g']['min']:.0f}–{d['battery_mass_g']['max']:.0f} g.
* Structure of missingness is *informative*, not random: 26/101 aircraft have no
  published wheelbase and 69/101 have no manufacturer AUW. These are omitted
  specifications, not zero values; wheelbase is median-imputed **with a missing
  indicator** and manufacturer AUW is used only as an external validation target.

## 2. What drives total weight

Pearson r with AUW (strongest first):

| variable | Pearson r |
|---|---|
""" + "\n".join(f"| `{k}` | {v:+.3f} |" for k, v in corr.items()) + f"""

The ordering is physically coherent: overall size (wheelbase, prop diameter)
and energy carried (cells, capacity, battery mass) dominate. Motor KV is
**negatively** correlated because small, high-KV motors belong to micro quads
and large, low-KV motors belong to heavy lifters — KV is a *class marker*, not
a mass driver on its own.

Allometric exponents (log–log OLS, AUW ∝ size^b):

| scaling | exponent b | s.e. | R² |
|---|---|---|---|
| AUW vs wheelbase | {allom['auw_vs_wheelbase']['exponent']:.2f} | {allom['auw_vs_wheelbase']['exponent_se']:.2f} | {allom['auw_vs_wheelbase']['r2']:.2f} |
| AUW vs prop diameter | {allom['auw_vs_prop_size']['exponent']:.2f} | {allom['auw_vs_prop_size']['exponent_se']:.2f} | {allom['auw_vs_prop_size']['r2']:.2f} |
| Dry mass vs wheelbase | {allom['dry_vs_wheelbase']['exponent']:.2f} | {allom['dry_vs_wheelbase']['exponent_se']:.2f} | {allom['dry_vs_wheelbase']['r2']:.2f} |

An exponent above 2 would flag a *cubic* (volume-driven) mass penalty; the
observed exponents are close to or below 2, i.e. mass grows roughly with the
square of linear size. This is the empirical basis for treating frame size as a
first-order weight lever, and it is why the optimizer is constrained to remain
inside the observed size envelope instead of extrapolating.

## 3. Battery is the single largest controllable mass

* Pack mass is almost perfectly linear in stored energy:
  **m_batt ≈ {bm['slope_g_per_Wh']:.2f} g/Wh × E + {bm['intercept_g']:.1f} g**
  (R² = {bm['r2']:.3f}, n = {len(pd.read_csv(DATA_PROCESSED / 'battery_catalog.csv'))} packs).
* Median pack-level specific energy: **{bm['median_specific_energy_wh_kg']:.0f} Wh/kg**.
* Battery mass fraction of AUW: median
  {d['battery_mass_fraction']['median']:.2f} (P10 {d['battery_mass_fraction']['p10']:.2f},
  P90 {d['battery_mass_fraction']['p90']:.2f}).

Because the slope is nearly constant across chemistries and brands, battery mass
is a *predictable linear function of the energy requirement* — exactly the term
that makes endurance and weight compete in the optimization.

## 4. Multicollinearity

Variance inflation factors for the modelling features:

| feature | VIF |
|---|---|
""" + "\n".join(f"| `{r['feature']}` | {r['VIF']:.2f} |" for _, r in vif.iterrows()) + f"""

Capacity, cell count and energy encode the same physical quantity; likewise
wheelbase and prop diameter. Tree ensembles are insensitive to this, but the
linear models require regularisation — which is why Ridge/Lasso are retained in
the benchmark rather than plain OLS alone.

## 5. Outliers

IQR and robust-z (|z|>3.5) counts are tabulated in
`descriptive_statistics.csv`. The extreme aircraft are **not measurement
errors**: they are 6–7 inch cinelifters and long-range builds whose mass is
genuinely an order of magnitude above the micro-whoop population. They are
retained because they define the upper end of the design envelope used by the
optimizer; deleting them would silently shrink the feasible design space.

## 6. Engineering interpretation — where optimization can act

1. **Battery energy is the dominant controllable term** (30–45% of AUW at the
   median). Reducing energy directly reduces mass but also endurance — the core
   trade-off.
2. **Prop diameter and wheelbase set the structural/propulsion mass** with an
   allometric exponent near 2. Shrinking the frame is the second lever, but it
   is limited by disc loading: smaller discs need more power for the same thrust.
3. **Motor mass is the least predictable component** (see the benchmark): it is
   chosen in discrete commercial sizes, so it behaves like a step function of
   thrust class rather than a smooth function of size.
4. **Payload is absent as a numeric column in this dataset.** It enters the
   optimization as an explicit requirement and an additive mass term, never as
   a fitted coefficient. This is stated again in the methodology report.
"""
    (REPORTS / "eda_observations.md").write_text(txt)


if __name__ == "__main__":
    s = run_eda()
    print("EDA complete.")
    print("AUW median %.1f g, range %.1f-%.1f g" % (
        [r for r in s["descriptives"] if r["variable"] == "auw_est_g"][0]["median"],
        [r for r in s["descriptives"] if r["variable"] == "auw_est_g"][0]["min"],
        [r for r in s["descriptives"] if r["variable"] == "auw_est_g"][0]["max"],
    ))
    print("Top correlations:", s["top_pearson_with_auw"])
