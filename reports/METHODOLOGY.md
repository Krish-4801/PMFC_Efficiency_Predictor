# Drone Weight Optimization — Methodology and Mathematical Formulation

**Component:** drone weight optimization (the previously incomplete part of the project)
**Base repository:** `Eratin/PMFC_Efficiency_Predictor` — PEMFC voltage-efficiency pipeline, preserved unchanged
**Drone dataset:** FPV-DB open dataset, CC BY 4.0, DOI [10.5281/zenodo.23613812](https://doi.org/10.5281/zenodo.23613812)
**Code:** `src/data_pipeline.py`, `src/eda.py`, `src/modeling.py`, `src/optimization.py`
**Generated artefacts:** `outputs/drone_weight/`, figures in `outputs/drone_weight/figures/`

---

## 0. Scope decision and why a second dataset was required

The reference repository ships two workbooks (`pmfc_fuel_cell.xlsx`, `pemfc_efficiency_clean.xlsx`)
containing 14,217 fuel-cell operating points across 15 variables. An audit of **every sheet and
every column** (`audit_dataset.py`, `outputs/data_audit.json`) found:

* 0 columns containing `mass`, `weight`, `kg`, `thrust`, `payload`, `battery`, `motor`,
  `propeller`, `frame`, `endurance`, `power_density`, `cell_count`, or `capacity`;
* the project's own `Notes` sheet states: *"Stages 3-4 … (drone power/weight analysis and
  weight optimisation) — this sheet has zero data on fuel-cell mass, cell count, or power
  density … You will need a second, separate dataset."*

A drone weight problem cannot be fitted to electrochemical polarisation data without inventing
masses. The drone component therefore uses a **separate, open, citable dataset**, while the PEMFC
files and scripts are left untouched. This is the only route consistent with the project's own
instruction not to fabricate engineering specifications.

---

## 1. Dataset and variables

FPV-DB is a component-level catalogue assembled from manufacturer datasheets and bench tests.

| Table | Rows used | Content |
|---|---|---|
| `quads` | 101 | assembled quadrotors: dry mass, AUW, wheelbase, prop size, recommended pack |
| `motors` | 199 | mass, stator, KV, maximum cell count, and **6,786 measured thrust-bench rows** |
| `batteries` | 326 | capacity, cells, chemistry, energy, mass, C-rating |
| `props` | 195 | diameter, pitch, blades, mass |
| `stacks` / `cameras` / `vtx` | 112 / 52 / 169 | avionics masses |

After cleaning: **5,627 usable bench rows**, **392 propulsion groups** (motor + prop + cell count)
with at least three throttle points so hover current can be interpolated rather than assumed.

### 1.1 Units

| Symbol | Meaning | Unit |
|---|---|---|
| $m_i$ | component mass | g |
| $W$ | weight (mass) | g |
| $D$ | propeller diameter | in (converted to m where needed) |
| $C$ | pack capacity | mAh |
| $S$ | cell count | – |
| $E$ | pack energy | Wh |
| $T$ | thrust | g (gram-force) |
| $I$ | current | A |
| $V$ | voltage | V |
| $t$ | endurance | min |
| $g$ | gravitational acceleration | 9.80665 m/s² |

---

## 2. The existing weight calculation, verified

The catalogue reports two masses per aircraft: `weight_no_battery_g` and, for 32 aircraft,
`typical_auw_g`. The component-sum identity tested is

$$
W_{\text{AUW}} \;=\; W_{\text{dry}} \;+\; W_{\text{battery}},
\qquad
W_{\text{dry}} = W_{\text{frame}} + 4W_{\text{motor}} + 4W_{\text{prop}} + W_{\text{avionics}} .
$$

`W_dry` is a measured column (frame, motors, props and avionics lumped, as the manufacturer
weighs the aircraft). `W_battery` is **not** stored per aircraft; it is matched from the pack
catalogue by (i) identical cell count and (ii) nearest capacity on a log scale. The match
distance is retained (`battery_match_logdist`), median 0.000, 90th percentile 0.022, i.e. most
packs are exact catalogue entries.

**Validation against the 32 aircraft with a published AUW:**

| metric | value |
|---|---|
| n | 32 |
| MAE | **22.2 g** |
| MAPE | **7.45 %** |
| R² | **0.969** |
| bias | +19.4 g |

The positive bias means the recommended pack is on average slightly heavier than the pack the
manufacturer weighed with — expected, because the recommendation is a range. The identity is
therefore accepted as the project's weight model.

> **Not available in the data:** bare-frame mass, motor-only mass per assembled aircraft, ESC
> thermal limits, structural stress allowables, and a numeric payload column. Frame mass is
> consequently an **explicit design input** in the optimization (§6), not a fitted coefficient.

### 2.1 Battery mass sub-model

Across 326 packs, mass is almost perfectly linear in stored energy:

$$
W_{\text{battery}}(E) \;=\; 5.678\,E \;+\; 25.20 \quad [\text{g}], \qquad R^2 = 0.972 .
$$

Marginal specific energy is $1/5.678\ \text{Wh/g} = 176$ Wh/kg; the median pack-level specific
energy is 138 Wh/kg (the intercept captures packaging). This linearity is what makes energy
(hence endurance) and mass directly compete.

---

## 3. EDA — findings that shape the model

Full output: `outputs/drone_weight/eda_summary.json`, `reports/eda_observations.md`.

* **AUW** spans 25.6–2,211 g (median 245 g, CV 114 %, skew 2.98). Log-scale treatment is
  justified; three aircraft are flagged as IQR/robust-z outliers and **retained** because they are
  genuine 6–7 inch cinelifters that define the upper design envelope.
* **Missingness is informative**: 69/101 aircraft lack a published AUW, 26/101 lack a wheelbase.
  Wheelbase is median-imputed **with a `wheelbase_missing` indicator**; the reported AUW is used
  only as an external validation target, never as a feature.
* **Dominant correlations with AUW:** dry mass 0.958, battery mass 0.949, energy 0.916,
  wheelbase 0.861, capacity 0.857, prop diameter 0.825, cells 0.716, motor KV −0.587.
  KV is a *class marker* (small motors run high KV), not a mass driver.
* **Allometric scaling** (log–log OLS):

  | relation | exponent $b$ | s.e. | R² |
  |---|---|---|---|
  | AUW vs wheelbase | 2.29 | 0.12 | 0.824 |
  | AUW vs prop diameter | 2.12 | 0.10 | 0.806 |
  | dry mass vs wheelbase | 2.15 | 0.12 | 0.813 |

  An exponent near 2 indicates area-like (not cubic) mass growth. The optimizer is therefore
  constrained inside the observed size envelope instead of extrapolating.
* **Multicollinearity (VIF):** prop size 8.2, cells 6.5, wheelbase 4.8, KV 3.4, capacity 1.9.
  Tree models tolerate this; linear models need regularisation, which is why Ridge/Lasso are kept.
* **Battery mass fraction** of AUW: median 0.36 (P10 0.27, P90 0.48) — the largest controllable
  term. **Disk loading** median 12.0 kg/m² (3.1–31.8).

---

## 4. Prediction models (distinct from the optimization)

Target: `auw_est_g` (total weight). Features: prop diameter, wheelbase (+ missing indicator),
cells, capacity, motor KV (+ missing indicator), airframe class (one-hot). No target leakage:
`cell_voltage`-style circular variables do not exist here, and `dry_mass`/`battery_mass` are not
used as inputs.

Protocol: 80/20 hold-out (test untouched during tuning) → `RandomizedSearchCV` (5-fold) on the
training set only → final scoring on the test set; plus `RepeatedKFold(5×5)` on the full table for
stability. All preprocessing is inside a `Pipeline`.

| Rank | Model | Test R² | MAE (g) | RMSE (g) | MAPE (%) | CV R² (mean ± sd) |
|---|---|---|---|---|---|---|
| 1 | Linear Regression | 0.980 | 45.5 | **58.3** | 60.0 | 0.902 ± 0.074 |
| 2 | CatBoost | 0.976 | **39.4** | 63.4 | **20.6** | 0.847 ± 0.170 |
| 3 | XGBoost | 0.974 | 45.2 | 67.0 | 41.3 | 0.837 ± 0.171 |
| 4 | Gradient Boosting | 0.972 | 40.4 | 69.3 | 20.0 | 0.822 ± 0.180 |
| 5 | Random Forest | 0.968 | 46.2 | 73.9 | 26.2 | 0.814 ± 0.177 |
| 6 | Extra Trees | 0.964 | 47.5 | 78.6 | 25.1 | 0.861 ± 0.129 |
| 7 | Lasso | 0.962 | 59.6 | 81.0 | 60.2 | 0.880 ± 0.089 |
| 8 | Ridge | 0.960 | 55.5 | 82.4 | 49.9 | 0.869 ± 0.096 |
| 9 | MLP (32-32) | 0.905 | 79.4 | 127.4 | 44.6 | 0.803 ± 0.127 |
| 10 | SVR (RBF) | 0.540 | 96.0 | 280.2 | 30.5 | 0.668 ± 0.247 |

**Interpretation.** With n = 101, linear regression is the most stable (CV R² 0.90) and CatBoost
has the best absolute error. The gap between test R² (0.98) and CV R² (0.90) shows the test split
is optimistic — the honest generalisation estimate is the CV column. Neural networks are not
supported by 101 samples; the MLP is reported only to demonstrate that. SVR fails because the
kernel width cannot be tuned reliably at this sample size.

The **optimization does not use these black-box models**; it uses the measured component
catalogue directly, which avoids surrogate error entirely. The regression benchmark is retained
because it answers the separate question "how well can total weight be predicted from design
parameters?".

---

## 5. Why optimization rather than regression

Regression predicts the mass of an *existing* design. Optimization selects a *new* combination
of real parts. The optimization problem is therefore stated on the catalogue:

$$
\min_{i,j,k,n}\ W_{\text{total}}
$$

with $i$ = propulsion group (motor + prop + cells), $j$ = battery pack, $k$ implicit in $j$,
$n$ = number of parallel packs.

---

## 6. Optimization formulation

### 6.1 Decision variables

| Variable | Domain | Source |
|---|---|---|
| propulsion group $i$ | 392 measured (motor, prop, cells) groups, filtered to the scenario's prop range and $S \le S_{\max}$ | `propulsion_groups.csv` |
| battery pack $j$ | 326 catalogue packs with $S_j = S_i$ | `battery_catalog.csv` |
| parallel packs $n$ | $\{1,2,3\}$ | design choice |
| propeller diameter $D_i$ | scenario range $[D_{\min}, D_{\max}]$ | frame class |

Continuous variables (wheelbase, frame mass) are **not** free: the catalogue has no frame table,
so the frame is a fixed input per scenario (see §7).

### 6.2 Objective

$$
W_{\text{total}}(i,j,n) \;=\;
W_{\text{airframe}}
\;+\; 4\,(m_{\text{motor},i} + m_{\text{prop},i})
\;+\; n\,m_{\text{batt},j}
\;+\; m_{\text{payload}} .
$$

All masses are catalogue values except $W_{\text{airframe}}$ and $m_{\text{payload}}$, which are
requirement inputs.

### 6.3 Constraints

**C1 — static thrust-to-weight.**
$$
\frac{4\,T_{\max}(i)}{W_{\text{total}}} \;\ge\; \mathrm{TWR}_{\min} = 2.0 .
$$
$T_{\max}(i)$ is the largest measured thrust in the group's bench table.

**C2 — hover endurance.** The hover thrust per motor is $T_h = W_{\text{total}}/4$. Current is
interpolated from the measured curve $I_i(T)$ (linear between bench points; through the origin
below the first point):

$$
I_h = \operatorname{interp}\!\big(T_h;\ T_i,\ I_i\big), \qquad
P_h = 4\,V_i(T_h)\,I_h ,
$$
$$
t_{\text{hover}} \;=\; \frac{60\,n\,E_j\,\eta}{P_h} \;\ge\; t_{\text{req}},
\qquad \eta = 0.80 .
$$

$\eta$ is the usable depth of discharge; it is a stated assumption, not a data column. The
sensitivity of the optimum to $t_{\text{req}}$ is reported.

**C3 — propeller/frame compatibility.**
$$
D_{\min} \le D_i \le D_{\max}.
$$
The lower bound prevents the degenerate "tiny prop on a large frame" solution; the upper bound is
the physical clearance.

**C4 — voltage limit.**
$$
S_i \le S_{\max}(i).
$$

**C5 — battery discharge limit.**
$$
I_h \;\le\; n\,C_{\text{rate},j}\,\frac{C_j}{1000}.
$$

**C6 — extrapolation cap.**
$$
W_{\text{total}} \le 2500\ \text{g},
$$
the top of the observed AUW range (2,211 g) rounded up.

### 6.4 Penalty handling

For metaheuristics the constrained problem is scalarised as

$$
F(x) = W_{\text{total}}(x) + \lambda \sum_{c \in \{C1,\dots,C6\}}
\max\!\left(0,\ \frac{\text{bound}_c - \text{value}_c(x)}{\text{bound}_c}\right),
\qquad \lambda = 10^5 .
$$

A candidate is **feasible** only when every normalised violation is zero. All reported weights
are feasible weights; infeasible runs are counted separately.

### 6.5 Exact reference solution

The full candidate set (12,600 / 46,956 / 8,229 combinations for the three scenarios) is
enumerated, giving the **global optimum by construction**. Metaheuristics are then scored against
it — this is the benchmark that makes the algorithm comparison meaningful.

---

## 7. Scenarios and the frame-mass assumption

| Scenario | Prop range (in) | Payload | Airframe + avionics | Endurance |
|---|---|---|---|---|
| 3-inch micro | 2.5–3.5 | 200 g | 95 g | 8 min |
| 5-inch freestyle | 4.5–5.5 | 500 g | 245 g | 10 min |
| 7-inch long-range | 6.5–7.5 | 750 g | 450 g | 12 min |

`W_airframe` is **not measured in the dataset**. It was estimated by mass balance on the quad
table (dry mass minus median motor and prop masses for the class) and is varied by ±30 % in the
sensitivity study (`outputs/drone_weight/sensitivity_analysis.csv`). It is an input, and is
labelled as such everywhere.

---

## 8. Algorithms

| Algorithm | Implementation | Encoding |
|---|---|---|
| Exhaustive enumeration | vectorised catalogue scan | exact |
| Grid search | 8 × 4 × 3 coarse grid | integer |
| Random search | uniform integers | integer |
| Genetic algorithm | tournament (k=3), uniform crossover, 15 % mutation, elitism 2 | integer |
| Particle swarm | $w=0.72$, $c_1=c_2=1.5$, velocity clamp 25 % of span | continuous → rounded |
| Differential evolution | SciPy `differential_evolution`, `integrality=[1,1,1]`, popsize 15 | integer |
| Bayesian optimisation | Optuna TPE | integer |

All heuristics share a **constraint-aware decoder**: the group index maps into the scenario-valid
group list, and the battery index maps into the subset of packs whose cell count matches the
chosen group. This prevents wasted evaluations on structurally impossible combinations and makes
the comparison about search strategy rather than about who is luckier with the encoding.

Budget: 1,200 evaluations per run, 8 independent seeds per algorithm per scenario.

---

## 9. Results

### 9.1 Optimal designs

| Scenario | AUW | Motor | Prop | Battery | TWR | Endurance | vs median feasible |
|---|---|---|---|---|---|---|---|
| 3-inch | **460.1 g** | 4 × F1404 | 4 × HQProp 3520 | 2 × 650 mAh 4S | 3.04 | 8.6 min | **−43.4 %** |
| 5-inch | **1,138.5 g** | 4 × 2203.5 | 4 × Gemfan 5125 | 1 × 3000 mAh 6S | 3.34 | 11.3 min | **−31.8 %** |
| 7-inch | **1,715.7 g** | 4 × F90 2806.5 | 4 × Gemfan 7042 | 1 × 3000 mAh 6S | 4.93 | 12.4 min | **−22.0 %** |

Component-wise breakdown: `outputs/drone_weight/optimized_component_weights.csv`. Because
FPV-DB has no bare-frame column, the airframe input is split into a **catalogue-based avionics
allocation** (median FC/ESC stack 8.5 g + camera 7.5 g + VTX 13.0 g = 29.0 g) and a **frame
residual**. This is an allocation of a design input, not an independent measurement; the
airframe total is preserved exactly and the split is labelled as such in the CSV.

The headline percentage is relative to the **median feasible catalogue build** — a representative
combination that satisfies the same requirements. That baseline is heavily over-provisioned (many
large packs and motors meet the constraints), so the reduction is large. Against a real aircraft of
the same prop class plus the same payload, the comparison is much closer:

| Scenario | Optimized | Real median aircraft + payload | Difference |
|---|---|---|---|
| 3-inch | 460.1 g | 453.8 g (n=35) | +1.4 % |
| 5-inch | 1,138.5 g | 1,154.0 g (n=17) | −1.3 % |
| 7-inch | 1,715.7 g | 2,647.5 g (n=2) | −35.2 % |

**Honest reading:** the optimizer does not beat every commercial design; it removes the
over-provisioning that appears when components are chosen independently of the requirement. The
7-inch comparison rests on only two real aircraft and should not be over-interpreted.

### 9.2 Algorithm comparison

Averaged over the three scenarios and 8 seeds:

| Algorithm | Best feasible (g) | Mean (g) | SD (g) | Success rate | Gap to optimum | Time (s) |
|---|---|---|---|---|---|---|
| Genetic Algorithm | 1,106.4 | 1,197.5 | 66.2 | 0.17 | 0.35 % | 0.07 |
| Bayesian (TPE) | 1,106.4 | 1,189.1 | 49.1 | 0.21 | 0.35 % | 2.96 |
| Random Search | 1,121.6 | 1,180.4 | 44.3 | 0.08 | 1.81 % | 0.05 |
| Particle Swarm | 1,125.0 | 1,204.8 | 50.7 | 0.08 | 1.98 % | 0.06 |
| Differential Evolution | 1,149.0 | 1,208.4 | 36.1 | 0.08 | 4.09 % | 0.09 |
| Grid Search | 1,243.2 | 1,243.2 | 0.0 | 0.00 | 15.60 % | 0.01 |

**Which approach is actually appropriate?** For this problem, **exhaustive enumeration**. The
candidate space is 8k–47k evaluated in 0.02–0.5 s, and it yields the certified global optimum with
zero variance. The metaheuristics are useful only as a scalability study: GA and TPE reach the
optimum most often, DE is the most *consistent* (lowest SD) but converges to slightly worse
points, and grid search is dominated. None beat enumeration on any criterion. This is the correct
engineering conclusion, not a reason to prefer a fashionable optimizer.

### 9.3 Sensitivity

5-inch scenario (`sensitivity_analysis.csv`):

* **Payload** is the strongest driver: 0 g → 473.7 g; 500 g → 1,138.5 g; 1,500 g → 2,478.7 g
  (super-linear, because payload cascades into propulsion and battery).
* **Endurance**: 5 min → 979.5 g; 20 min → 1,775.7 g. The curve steepens above ~12 min.
* **TWR requirement**: 1.5 → 3.0 leaves the optimum unchanged at 1,138.5 g (the chosen design
  already has TWR 3.34). **Thrust margin is not the binding constraint at this payload/endurance;
  energy is.** Raising TWR above 3.34 would start to bind.
* **Airframe mass** ±30 %: 1,017–1,212 g, i.e. roughly one-for-one, as expected from the additive
  objective.

---

## 10. Assumptions and limitations (explicit)

1. **Frame mass is an input, not a measurement.** FPV-DB has no bare-frame column. Values are
   documented estimates; ±30 % sensitivity is provided.
2. **Usable battery fraction η = 0.80** is an assumption. It scales endurance linearly; the
   sensitivity to the endurance requirement is reported instead of hiding it.
3. **Thrust and current are bench values at fixed voltage**, interpolated at hover. Prop
   unloading, motor heating, ESC losses and voltage sag are not modelled.
4. **Payload is additive.** It is a requirement, not a fitted variable; the dataset has no numeric
   payload column. Payload does enter C1 and C2, so it correctly drives propulsion and battery size.
5. **Prop diameter is parsed from catalogue strings.** Ambiguous codes (e.g. `1050`, `1219`) are
   rejected rather than guessed; 392 of 867 bench groups survive.
6. **The dataset is FPV/hobby-class (≤ 2.2 kg).** C6 caps AUW at 2.5 kg; results above that would
   be extrapolation and are excluded.
7. **No structural stress or fatigue model.** "Structural integrity" is represented only by the
   frame-class prop range and the airframe mass input. A real structural check needs layup,
   thickness and load data that the catalogue does not contain.
8. **C-rate limit uses the pack's declared rating**, which is a manufacturer figure.

---

## 11. Visualisation and interpretation

All figures are 200 dpi under `outputs/drone_weight/figures/`.

| Figure | What it shows | What it reveals |
|---|---|---|
| `component_weight_breakdown.png` | dry vs battery mass for all 101 aircraft, sorted by AUW, log scale | battery share is roughly constant across two orders of magnitude; no mass class is anomalous |
| `distributions.png` | histograms + KDE of AUW, dry mass, battery mass, prop size, capacity, energy | strong right skew in every mass variable; supports log-scale analysis and warns against linear extrapolation |
| `correlation_heatmap.png` | 11×11 Pearson matrix | size and energy dominate; KV is a class marker; cell count and energy are redundant |
| `design_vs_weight.png` | four log–log scatters | slopes near 2 (area-like scaling); visible curvature would indicate a regime change — none is present |
| `weight_outliers_by_class.png` | box + strip plot by class group | the heavy tail is a class effect (cinelifters), not bad data; justifies retaining outliers |
| `component_mass_by_type.png` | catalogue mass ranges per component type | batteries and stacks span the widest range; props/motors are tight — battery is the main lever |
| `battery_energy_mass.png` | pack mass vs energy + specific-energy histogram | linear law m = 5.68E + 25.2 (R² 0.972); median 138 Wh/kg |
| `motor_mass_vs_thrust.png` | motor mass vs measured max thrust (392 groups) | thrust scales super-linearly with motor mass; this is why over-sizing motors is expensive |
| `model_rmse_comparison.png` | test RMSE per regressor | linear model is competitive; SVR is an outlier failure |
| `predicted_vs_actual.png` | held-out and out-of-fold parity plots | errors are proportional to mass (heteroscedastic); small quads are predicted relatively poorly |
| `convergence_curves.png` | best feasible AUW vs search progress | GA/TPE drop quickly; PSO and DE stall above the optimum; grid never reaches it |
| `algorithm_comparison.png` | mean feasible AUW per algorithm per scenario, with optimum lines | heuristics cluster 0–4 % above the exact optimum; grid is 5–16 % above |
| `original_vs_optimized.png` | median feasible vs optimized AUW | the saving is large against an over-provisioned catalogue build and small against a real aircraft |
| `optimized_component_breakdown.png` | stacked component masses per scenario | payload and battery dominate; propulsion is a small share, so over-sizing motors is not the main error |
| `sensitivity_analysis.png` | AUW vs TWR, endurance, payload, airframe scale | payload is the strongest driver; endurance is convex; TWR is non-binding below 3.34 |
| `constraint_feasibility.png` | feasible/infeasible designs in endurance–weight and TWR–weight planes | the optimum sits on the **endurance** boundary, not the thrust boundary |

---

## 12. Reference-to-method mapping

Full metadata, DOIs and the published-vs-adapted distinction are in
`reports/REFERENCES.md`. The mapping is:

| Method used here | Published basis | How ours differs |
|---|---|---|
| Additive mass build-up | Ong, Srigrarom & Hesse 2019 (10.2514/6.2019-2095) | per-component catalogue masses instead of class mass fractions |
| Empirical design trends / statistical sizing | Rand & Khromov 2021 (10.3390/aerospace8110321) | discrete COTS catalogue, exact enumeration |
| Component-mass regression from design inputs | Rubrichi, Gnoni & Tornese 2025 (10.1016/j.ifacol.2025.09.280) | 10 regressors, leak-free CV; used for prediction only |
| Battery–endurance trade-off | de Angelis et al. 2021 (10.1016/j.ast.2021.107057); Hwang et al. 2018 (10.3390/en11092221) | measured interpolated hover current instead of a figure of merit |
| Propulsion maps for sizing | Bershadsky, Haviland & Johnson 2016 (10.2514/6.2016-0581) | public bench tables + COTS selection layer |
| COTS component selection | May, Braga & Ferguson 2022 (10.1139/dsa-2021-0051); Dai, Quan & Cai 2021 (10.1109/TASE.2021.3081122) | objective is mass, not cost, with TWR and C-rate constraints |
| MDO problem structure | Martins & Lambe 2013 (10.2514/1.J051895) | monolithic MDF, no decomposition needed at this scale |
| Take-off-weight minimisation with GA | Babaei et al. 2018 (10.1016/j.cja.2018.04.018); Hosseini et al. 2017 (10.5028/jatm.v9i2.725) | multirotor, discrete catalogue, exact optimum as benchmark |
| PSO weight minimisation | Saib et al. 2023 (10.1177/09544100231212651) | used as a baseline, not the recommended method |
| DE / PSO / GA / BO algorithms | Storn & Price 1997; Kennedy & Eberhart 1995; Deb et al. 2002; Shahriari et al. 2016 | standard implementations, compared on one problem |

---

## 13. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py        # fetch FPV-DB CSVs + record SHA-256
python run_all.py                   # pipeline -> EDA -> models -> optimization -> figures
jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace
```

Outputs land in `outputs/drone_weight/`; the narrative EDA is in
`reports/eda_observations.md`; references are in `reports/REFERENCES.md`.
