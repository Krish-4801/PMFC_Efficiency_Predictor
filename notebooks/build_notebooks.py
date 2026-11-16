"""Build and execute the three drone-weight notebooks.

Run:  python notebooks/build_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
KERNEL = "tanu"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# ---------------------------------------------------------------------------
# 01 — EDA
# ---------------------------------------------------------------------------
eda = nbf.v4.new_notebook()
eda.cells = [
    md("""
# Drone Weight — Exploratory Data Analysis

This notebook characterises the drone weight data used by the optimization component.

**Dataset:** FPV-DB open component catalogue (CC BY 4.0, doi:10.5281/zenodo.23613812).
**Target:** all-up weight (AUW) of 101 assembled quadrotors, reconstructed as
`dry mass + catalogue-matched battery mass`.

The PEMFC dataset in this repository was audited first and contains **no mass, thrust,
payload or battery variables**, so it cannot support this component (see
`outputs/data_audit.json` and the project's own `Notes` sheet).
"""),
    code("""
import sys, json
sys.path.insert(0, "..")
import pandas as pd, numpy as np
pd.set_option("display.width", 200)
from src.config import DATA_PROCESSED, OUTPUTS, FIGURES
from src.eda import run_eda, descriptives, vif_table

quad = pd.read_csv(DATA_PROCESSED / "quad_design_table.csv")
print("quad design table:", quad.shape)
quad[["brand","model","class_group","prop_size_in","wheelbase_mm",
      "dry_mass_g","battery_mass_g","auw_est_g"]].head(10)
"""),
    md("## 1. Missingness and duplicates"),
    code("""
miss = quad.isna().mean().sort_values(ascending=False)
print("duplicate brand/model rows:", int(quad.duplicated(subset=["brand","model"]).sum()))
print(miss[miss > 0].round(3).to_string())
"""),
    md("""
Missingness is **informative**, not random: 69 of 101 aircraft have no published AUW
and 26 have no wheelbase. These are omitted specifications. Wheelbase is median-imputed
with a `wheelbase_missing` indicator in the model; the published AUW is used only as an
external validation target.
"""),
    md("## 2. Descriptive statistics"),
    code("""
desc = descriptives(quad)
desc[["variable","n","missing","mean","sd","min","median","max","cv_pct","skew",
      "iqr_outliers","robust_outliers_3.5"]].round(2)
"""),
    md("""
AUW spans **25.6–2,211 g** with a median of 245 g and a coefficient of variation of 114 %.
The three IQR outliers are genuine 6–7 inch cinelifters and are retained: they define the
upper end of the design envelope used by the optimizer. Deleting them would shrink the
feasible design space.
"""),
    md("## 3. Correlation and multicollinearity"),
    code("""
cols = ["auw_est_g","dry_mass_g","battery_mass_g","energy_wh","wheelbase_mm",
        "prop_size_in","cells","capacity_mah","motor_kv","disk_loading_kg_m2"]
corr = quad[cols].corr()
corr["auw_est_g"].drop("auw_est_g").sort_values(ascending=False).round(3)
"""),
    code("""
from src.eda import vif_table
vif_table(quad[["prop_size_in","wheelbase_mm","cells","capacity_mah","motor_kv"]])
"""),
    md("""
Prop size, cell count and wheelbase are collinear (VIF 4.8–8.2) because they all encode
vehicle size. Tree models tolerate this; linear models require regularisation, which is
why Ridge and Lasso are kept in the benchmark.

Motor KV is **negatively** correlated with weight (r = −0.59): small high-KV motors belong
to micro quads, large low-KV motors to heavy lifters. KV is a class marker, not a mass driver.
"""),
    md("## 4. Allometric scaling"),
    code("""
from src.eda import allometric_fit
fits = [allometric_fit(quad, "wheelbase_mm"), allometric_fit(quad, "prop_size_in"),
        allometric_fit(quad, "wheelbase_mm", "dry_mass_g"),
        allometric_fit(quad, "prop_size_in", "dry_mass_g")]
pd.DataFrame(fits)[["x","y","n","exponent","exponent_se","r2","p_value"]].round(4)
"""),
    md("""
The exponents are close to 2 (2.12–2.29), i.e. mass grows roughly with the square of
linear size rather than with volume. This is the empirical justification for treating
frame/prop size as a first-order weight lever, and for constraining the optimizer to the
observed size envelope instead of extrapolating.
"""),
    md("## 5. Component-level relationships"),
    code("""
cat = pd.read_csv(DATA_PROCESSED / "component_catalog.csv")
print("components:", cat.shape[0])
cat.groupby("component_type")["weight_g"].agg(["count","median","min","max"]).round(2)
"""),
    code("""
batt = pd.read_csv(DATA_PROCESSED / "battery_catalog.csv")
a, b = np.polyfit(batt["energy_wh"], batt["weight_g"], 1)
r2 = np.corrcoef(batt["energy_wh"], batt["weight_g"])[0,1]**2
print(f"m_batt = {a:.3f} g/Wh x E + {b:.2f} g   (R2 = {r2:.3f}, n = {len(batt)})")
print(f"median pack specific energy = {(batt.energy_wh/batt.weight_g*1000).median():.0f} Wh/kg")
"""),
    md("""
Battery mass is almost perfectly linear in stored energy (R² = 0.972). This is the
mechanism that makes **endurance and weight compete directly**: every extra minute of
endurance requires energy, and energy costs mass at a nearly constant rate.
"""),
    md("## 6. Figures"),
    code("""
from IPython.display import Image, display
for name in ["component_weight_breakdown","distributions","correlation_heatmap",
             "design_vs_weight","weight_outliers_by_class","component_mass_by_type",
             "battery_energy_mass","motor_mass_vs_thrust"]:
    display(Image(filename=str(FIGURES / f"{name}.png"), width=820))
"""),
    md("""
**What the figures show**

* *Component breakdown* — dry mass and battery mass grow together across two orders of
  magnitude; battery share is roughly constant (median 36 %).
* *Distributions* — all mass variables are strongly right-skewed; log transforms are the
  appropriate scale.
* *Correlation heatmap* — size and energy dominate; temperature-like or KV variables do not.
* *Design vs weight* — log–log slopes near 2 confirm area-like scaling.
* *Outliers by class* — the heavy tail is a class effect, not measurement error.
* *Component masses by type* — batteries and stacks span the widest mass range, motors
  and props are tightly clustered; this is why battery choice is the main optimization lever.
* *Motor mass vs thrust* — measured bench data, spanning 18 g to 7.1 kg of thrust.
"""),
    md("## 7. Validation of the weight formula"),
    code("""
v = quad.dropna(subset=["auw_reported_g","auw_est_g"])
err = v.auw_est_g - v.auw_reported_g
print("n =", len(v))
print(f"MAE  = {err.abs().mean():.2f} g")
print(f"MAPE = {(err.abs()/v.auw_reported_g).mean()*100:.2f} %")
print(f"bias = {err.mean():+.2f} g")
print(f"R2   = {1-(err**2).sum()/((v.auw_reported_g-v.auw_reported_g.mean())**2).sum():.4f}")
"""),
    md("""
The identity `AUW = dry mass + catalogue battery mass` reproduces the 32 published AUW
values with 22.2 g MAE (7.45 %) and R² = 0.969. The small positive bias is expected: the
recommended pack is a range, and the manufacturer often weighed with a lighter one.
"""),
    md("## 8. Conclusions for the optimization"),
    code("""
print(open("../reports/eda_observations.md").read()[:3000])
"""),
]

# ---------------------------------------------------------------------------
# 02 — Modelling
# ---------------------------------------------------------------------------
model = nbf.v4.new_notebook()
model.cells = [
    md("""
# Drone Weight — Model Benchmark

Ten regressors predict **total all-up weight** from design parameters.
Protocol: 80/20 hold-out (test untouched during tuning), `RandomizedSearchCV` (5-fold)
inside the training set only, `RepeatedKFold(5×5)` on the full table for stability.
All preprocessing lives inside a scikit-learn `Pipeline`, so there is no leakage.

This stage answers *"how predictable is drone weight?"*. The optimization stage does
**not** use these models; it searches measured catalogue components directly.
"""),
    code("""
import sys
sys.path.insert(0, "..")
import pandas as pd
pd.set_option("display.width", 220)
from src.modeling import load_design_table, build_features
df = load_design_table()
X, y = build_features(df)
print("rows:", len(df), "| features:", X.shape[1])
print("target AUW (g):", y.min().round(1), "-", y.max().round(1), "| median", y.median().round(1))
X.head()
"""),
    md("## Run the benchmark (tuning is inside the training split)"),
    code("""
from src.modeling import run_benchmark, plots
metrics, artefacts = run_benchmark()
metrics[["rank","model","R2","MAE_g","RMSE_g","MAPE_pct","cv_R2_mean","cv_R2_sd"]].round(3)
"""),
    md("## Visual comparison"),
    code("""
plots(metrics, artefacts)
from IPython.display import Image, display
display(Image(filename="../outputs/drone_weight/figures/model_rmse_comparison.png", width=850))
display(Image(filename="../outputs/drone_weight/figures/predicted_vs_actual.png", width=850))
"""),
    md("""
## Interpretation

* **Linear regression is the most stable** (CV R² 0.902 ± 0.074) and has the lowest test
  RMSE. With 101 samples, the extra flexibility of boosted trees does not pay for itself.
* **CatBoost has the best absolute error** (MAE 39.4 g, MAPE 20.6 %): gradient boosting
  captures the step-like motor/prop classes better than a straight line, at the cost of
  higher variance across folds.
* **The test R² (0.98) is optimistic** relative to repeated CV (0.90). The honest
  generalisation estimate is the CV column, not the single split.
* **SVR fails** (CV R² 0.67 ± 0.25): the RBF width cannot be tuned reliably at n = 101.
* **Neural networks are not supported** by this dataset size; the small MLP is reported
  only to demonstrate that (R² 0.905, MAE 79 g).

**Engineering reading.** Weight is predictable to roughly ±40–80 g, i.e. 10–25 % of a
typical quad. That is good enough for early sizing but not for certification. It also
justifies using measured catalogue values, not model predictions, inside the optimizer.
"""),
    code("""
metrics.to_csv("../outputs/drone_weight/model_benchmark_notebook.csv", index=False)
print("saved")
"""),
]

# ---------------------------------------------------------------------------
# 03 — Optimization
# ---------------------------------------------------------------------------
opt = nbf.v4.new_notebook()
opt.cells = [
    md("""
# Drone Weight — Constrained Optimization

## Formulation

$$
\\min_{i,j,n}\\; W_{\\text{total}}
= W_{\\text{airframe}} + 4(m_{\\text{motor},i}+m_{\\text{prop},i})
+ n\\,m_{\\text{batt},j} + m_{\\text{payload}}
$$

subject to

$$
\\begin{aligned}
&\\text{(C1)}\\quad 4T_{\\max,i}/W_{\\text{total}} \\ge 2.0 && \\text{static thrust}\\\\
&\\text{(C2)}\\quad 60\\,nE_j\\eta / P_h \\ge t_{\\text{req}},\\; P_h=4V_iI_i(W/4) && \\text{endurance}\\\\
&\\text{(C3)}\\quad D_{\\min}\\le D_i\\le D_{\\max} && \\text{prop/frame}\\\\
&\\text{(C4)}\\quad S_i\\le S_{\\max,i} && \\text{voltage}\\\\
&\\text{(C5)}\\quad I_h \\le nC_{\\text{rate},j}C_j/1000 && \\text{discharge}\\\\
&\\text{(C6)}\\quad W_{\\text{total}}\\le 2500\\ \\text{g} && \\text{extrapolation cap}
\\end{aligned}
$$

Every mass, thrust and current value is measured/catalogued. `W_airframe` and the payload
are explicit design inputs; the frame mass is **not** present in the dataset and is varied
by ±30 % in the sensitivity study.
"""),
    code("""
import sys, json
sys.path.insert(0, "..")
import pandas as pd, numpy as np
pd.set_option("display.width", 240)
from src.config import SCENARIOS
from src.optimization import Catalogue, DesignProblem, component_table, sensitivity

cat = Catalogue.load()
print("propulsion groups:", len(cat.groups), "| batteries:", len(cat.batteries))
pd.DataFrame(SCENARIOS).T
"""),
    md("## Exact solution (global optimum by enumeration)"),
    code("""
results = {}
for name in SCENARIOS:
    p = DesignProblem(cat, SCENARIOS[name])
    best, designs = p.exhaustive()
    results[name] = (p, best, designs)
    print(f"{name:6s} candidates={len(designs):6d} feasible={designs.feasible.sum():6d} "
          f"optimum={best['w_total_g']:.1f} g  TWR={best['twr']:.2f}  "
          f"endurance={best['endurance_min']:.1f} min")
"""),
    code("""
for name,(p,best,designs) in results.items():
    print("="*70)
    print(name, "->", round(best["w_total_g"],1), "g")
    for k in ["motor_model","motor_mass_g","prop_key","prop_diameter_in","prop_mass_g",
              "cells","battery_model","battery_mass_g","n_packs","twr","endurance_min"]:
        print(f"  {k:22s} {best[k]}")
"""),
    md("## Component-wise result"),
    code("""
comp = pd.concat([component_table({"exact":{"scenario":n,"global_optimum":b}})
                  for n,(p,b,d) in results.items()], ignore_index=True)
comp.round(1)
"""),
    md("## Baseline comparison"),
    code("""
import json
res = json.load(open("../outputs/drone_weight/optimization_results.json"))
rows=[]
for k,v in res["scenarios"].items():
    o=v["global_optimum"]; b=v["baseline_median_feasible"]; t=v["baseline_typical_components"]
    rows.append({"scenario":k,"optimized_g":o["w_total_g"],
                 "median_feasible_g":b["w_total_g"],
                 "reduction_vs_median_feasible_pct":v["reduction_vs_median_feasible_pct"],
                 "real_class_plus_payload_g":t["real_class_auw_plus_payload_g"],
                 "real_class_n":t["real_class_n"]})
pd.DataFrame(rows).round(1)
"""),
    md("""
The reduction against the **median feasible catalogue build** is 22–43 %. That baseline is
a valid but heavily over-provisioned combination. Against a **real aircraft of the same prop
class plus the same payload**, the optimized 5-inch design is only 1.3 % lighter — a much more
credible statement. Both numbers are reported; neither is hidden.
"""),
    md("## Algorithm comparison (pre-computed, 8 seeds × 1,200 evaluations)"),
    code("""
comp = pd.read_csv("../outputs/drone_weight/algorithm_comparison.csv")
comp.groupby("algorithm")[["best_feasible_g","mean_feasible_g","sd_feasible_g",
                           "success_rate","gap_to_optimum_pct","mean_time_s"]].mean().round(3)
"""),
    code("""
comp.groupby(["scenario","algorithm"])["success_rate"].mean().unstack().round(2)
"""),
    md("""
**Conclusion.** Exhaustive enumeration is the appropriate method: 8k–47k candidates are
evaluated in under a second and give the certified global optimum with zero variance.
Among heuristics, GA and TPE reach the optimum most often; DE is the most consistent but
converges to slightly worse points; grid search is dominated. No heuristic beats exact
enumeration on any criterion.
"""),
    md("## Sensitivity (5-inch)"),
    code("""
sens = pd.read_csv("../outputs/drone_weight/sensitivity_analysis.csv")
piv = sens[sens.parameter!="baseline"].pivot_table(index="value", columns="parameter",
                                                   values="optimum_g")
piv.round(1)
"""),
    code("""
from IPython.display import Image, display
for f in ["convergence_curves","algorithm_comparison","original_vs_optimized",
          "optimized_component_breakdown","sensitivity_analysis","constraint_feasibility"]:
    display(Image(filename=f"../outputs/drone_weight/figures/{f}.png", width=850))
"""),
    md("""
### Reading the sensitivity plots

* **Payload dominates**: 0 g → 474 g, 500 g → 1,139 g, 1,500 g → 2,479 g. The response is
  super-linear because payload cascades into propulsion and battery mass.
* **Endurance** is convex: 5 min → 980 g, 12 min → 1,142 g, 20 min → 1,776 g. Beyond ~12 min
  each extra minute costs disproportionately.
* **TWR** between 1.5 and 3.0 does not change the optimum: the selected design already has
  TWR 3.34, so **energy, not thrust, is the binding constraint** at this payload.
* **Airframe mass** shifts the optimum almost one-for-one, as expected from an additive objective.
* The **feasible design space** plot shows the green feasible region and the red TWR = 2 line;
  the optimum sits on the endurance boundary, not the thrust boundary.
"""),
    md("""
## Limitations to carry into any decision

1. Frame mass is a design input, not a measured column (±30 % sensitivity provided).
2. Usable battery fraction η = 0.80 is an assumption; endurance scales linearly with it.
3. Bench thrust/current are steady-state at fixed voltage; voltage sag, heating and ESC losses
   are not modelled.
4. Payload is additive and drives C1/C2, but the dataset has no payload column to calibrate a
   payload-dependent mass cascade.
5. The catalogue is FPV/hobby class (≤ 2.2 kg); C6 caps results at 2.5 kg.
6. No structural stress model exists in the data; "structural integrity" is represented only by
   the frame-class prop range and the airframe mass input.
"""),
]

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    # Use the project kernel if it is registered, otherwise the default python3.
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        available = KernelSpecManager().find_kernel_specs()
    except Exception:
        available = {}
    kernel = KERNEL if KERNEL in available else "python3"

    for fname, nb in [
        ("01_drone_weight_eda.ipynb", eda),
        ("02_drone_weight_modeling.ipynb", model),
        ("03_drone_weight_optimization.ipynb", opt),
    ]:
        path = NB_DIR / fname
        nbf.write(nb, path)
        client = NotebookClient(nb, timeout=1800, kernel_name=kernel,
                                resources={"metadata": {"path": str(NB_DIR)}})
        client.execute()
        nbf.write(nb, path)
        print("executed", fname)
