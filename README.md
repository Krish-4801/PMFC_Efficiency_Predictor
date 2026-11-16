# ⚡ PEMFC Voltage-Efficiency Prediction — EDA & Stacked Ensemble Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning pipeline to predict the **Voltage Efficiency** ($\eta_v$, in %) of a **Proton Exchange Membrane Fuel Cell (PEMFC)** based on non-leaking operational parameters. 

This repository contains the complete workflow: **Exploratory Data Analysis (EDA)**, **leak-free preprocessing**, **5-fold RandomizedSearchCV & LOOCV evaluation**, and **Stacked Ensemble Regression** combining `XGBoost`, `SVR`, and `Random Forest` with both continuous (**Ridge**) and discretized classification-wrapped (**Logistic Regression**) meta-learners.

> **Repository now contains two components.**
> The PEMFC efficiency pipeline above is unchanged. A second component —
> **[Drone Weight Optimization](#-drone-weight-optimization-component)** — is included
> under `src/`, `notebooks/`, `reports/` and `outputs/drone_weight/`.
> It uses a separate, openly licensed UAV component dataset because the PEMFC data
> contains no mass, thrust or payload variables (verified in `outputs/data_audit.json`).

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Comprehensive Summary Statistics](#-dataset--comprehensive-summary-statistics)
3. [EDA Findings & Full Correlation Analysis](#-eda-findings--full-correlation-analysis)
4. [Multicollinearity & Variance Inflation Factor (VIF)](#-multicollinearity--variance-inflation-factor-vif)
5. [All Exploratory Data Analysis Graphs](#-all-exploratory-data-analysis-graphs)
6. [Machine Learning Methodology & Stacking Architecture](#-machine-learning-methodology--stacking-architecture)
7. [Hyperparameter Optimization & LOOCV Results](#-hyperparameter-optimization--loocv-results)
8. [Comprehensive Model Performance Benchmark](#-comprehensive-model-performance-benchmark)
9. [Diagnostic Visualizations & Interpretability](#-diagnostic-visualizations--interpretability)
10. [Repository Structure](#-repository-structure)
11. [Installation & Quickstart Guide](#-installation--quickstart-guide)
12. [Engineering Insights & Hardware Recommendations](#-engineering-insights--hardware-recommendations)

---

## 🔬 Project Overview

In PEMFC systems, voltage efficiency $\eta_v$ represents the ratio of operating cell voltage to theoretical reversible voltage:
$$\eta_v = \frac{V_{\text{cell}}}{E_{\text{rev}}} \times 100\%$$

Predicting $\eta_v$ purely from operating conditions (current density, differential pressures, and thermal sensor readings) provides:
- **Non-intrusive diagnostics**: Real-time health monitoring without high-frequency electrical probing.
- **Dynamic operating control**: Balancing high power density ($P = I \cdot V$) against fuel conservation ($\eta_v$).
- **Control loop integration**: Low-latency control for automotive fuel cell powertrains and stationary micro-grids.

---

## 📊 Dataset & Comprehensive Summary Statistics

The dataset contains **14,217 steady-state operational measurements** across 15 parameters with **zero missing values**.

### Full Summary Statistics Table (14,217 Rows)

| Parameter | Type | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max | Unit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `current_density_mA_cm2` | Raw Input | 2,243.64 | 1,659.88 | 0.00 | 655.80 | 2,122.80 | 3,658.00 | 5,094.72 | $\text{mA/cm}^2$ |
| `cathode_pressure_diff_kPa` | Raw Input | 3.2356 | 2.6841 | 0.0000 | 0.8100 | 2.5000 | 5.2500 | 9.9400 | $\text{kPa}$ |
| `anode_pressure_diff_kPa` | Raw Input | 2.8797 | 2.0837 | 0.0900 | 0.9600 | 2.4500 | 4.4900 | 7.4200 | $\text{kPa}$ |
| `anode_temp_beginning_C` | Raw Sensor | 80.26 | 0.26 | 79.52 | 80.08 | 80.31 | 80.46 | 80.92 | $^\circ\text{C}$ |
| `anode_temp_middle_C` | Raw Sensor | 80.34 | 0.28 | 79.80 | 80.13 | 80.30 | 80.50 | 81.45 | $^\circ\text{C}$ |
| `anode_temp_end_C` | Raw Sensor | 79.91 | 0.27 | 78.59 | 79.72 | 79.88 | 80.06 | 80.49 | $^\circ\text{C}$ |
| `cathode_temp_beginning_C` | Raw Sensor | 79.72 | 0.31 | 78.52 | 79.54 | 79.75 | 79.91 | 80.39 | $^\circ\text{C}$ |
| `cathode_temp_middle_C` | Raw Sensor | 80.42 | 0.33 | 79.70 | 80.17 | 80.36 | 80.59 | 81.49 | $^\circ\text{C}$ |
| `cathode_temp_end_C` | Raw Sensor | 80.39 | 0.30 | 79.83 | 80.17 | 80.39 | 80.55 | 81.35 | $^\circ\text{C}$ |
| `anode_avg_temp_C` | Engineered | 80.17 | 0.26 | 79.38 | 79.99 | 80.20 | 80.34 | 80.78 | $^\circ\text{C}$ |
| `cathode_avg_temp_C` | Engineered | 80.20 | 0.28 | 79.52 | 80.00 | 80.23 | 80.40 | 80.96 | $^\circ\text{C}$ |
| `cell_avg_temp_C` | Engineered | 80.19 | 0.27 | 79.46 | 80.00 | 80.22 | 80.37 | 80.84 | $^\circ\text{C}$ |
| `cell_voltage_V` | Upstream Leak | 0.6186 | 0.1751 | 0.2248 | 0.4907 | 0.6385 | 0.7511 | 1.0189 | $\text{V}$ |
| `reversible_voltage_V` | Upstream Leak | 1.2043 | 0.0002 | 1.2038 | 1.2041 | 1.2042 | 1.2044 | 1.2048 | $\text{V}$ |
| `voltage_efficiency_pct` | **TARGET** | **51.3702** | **14.5377** | **18.6622** | **40.7505** | **53.0336** | **62.3683** | **84.5828** | **%** |

### 🛡️ Recommended Leak-Free Input Feature Set
To prevent artificial mathematical leakage, `cell_voltage_V` and `reversible_voltage_V` are strictly excluded from the feature matrix $X$:
$$\mathbf{X} = \begin{bmatrix} \text{current\_density\_mA\_cm2}, & \text{cathode\_pressure\_diff\_kPa}, & \text{anode\_pressure\_diff\_kPa}, & \text{anode\_avg\_temp\_C}, & \text{cathode\_avg\_temp\_C} \end{bmatrix}$$

---

## 🔍 EDA Findings & Full Correlation Analysis

### Pearson Correlation Matrix with Target ($\eta_v$)

```
                        Feature    Pearson r vs Target (voltage_efficiency_pct)
0                cell_voltage_V    +1.000000  (EXCLUDED - Circular Target Component)
1        current_density_mA_cm2    -0.977967  (Dominant negative non-linear relationship)
2       anode_pressure_diff_kPa    -0.978810  (Highly correlated with current density)
3     cathode_pressure_diff_kPa    -0.975730  (Highly correlated with current density)
4      cathode_temp_beginning_C    +0.781774
5              anode_temp_end_C    +0.716075
6        cathode_temp_middle_C    -0.672852
7          anode_temp_middle_C    -0.502891
8            cathode_temp_end_C    -0.386008
9              anode_avg_temp_C    +0.150423
10           cathode_avg_temp_C    -0.067301
11              cell_avg_temp_C    +0.038100
12         reversible_voltage_V    -0.038100  (EXCLUDED - Upstream Calculation)
13       anode_temp_beginning_C    -0.035414
```

---

## 📐 Multicollinearity & Variance Inflation Factor (VIF)

The Variance Inflation Factor (VIF) was evaluated across all 5 recommended features to quantify multicollinearity:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

| Feature Name | Variance Inflation Factor (VIF) | Collinearity Status |
| :--- | :---: | :--- |
| `anode_pressure_diff_kPa` | **98.0592** | Extreme Multicollinearity ($r \approx 0.99$ with current density) |
| `current_density_mA_cm2` | **66.0834** | Extreme Multicollinearity ($r \approx 0.98 - 0.99$) |
| `cathode_pressure_diff_kPa` | **33.9101** | Severe Multicollinearity ($r \approx 0.98$) |
| `anode_avg_temp_C` | **11.4522** | Moderate Multicollinearity (Thermal correlation) |
| `cathode_avg_temp_C` | **11.1093** | Moderate Multicollinearity (Thermal correlation) |

> **Key Modeling Takeaway**: Tree-based ensembles (`Random Forest`, `XGBoost`) are inherently robust against feature collinearity because splits can utilize either feature without gradient degradation. Linear meta-learners benefit from $\mathcal{L}_2$ regularization (**Ridge**) to stabilize weight allocations.

---

## 🖼️ All Exploratory Data Analysis Graphs

### 1. Complete Correlation Heatmap (15x15 Matrix)
![Correlation Heatmap](outputs/correlation_heatmap.png)

### 2. Feature & Target Distributions (Histograms + KDE Overlay)
![Feature Distributions](outputs/distributions.png)

### 3. Target vs Feature Scatter Plots (Electrochemical Polarization Curves)
![Scatter vs Target](outputs/scatter_vs_target.png)

### 4. Temperature Sensor Operating Range Boxplots ($79.4^\circ\text{C} - 81.5^\circ\text{C}$)
![Temperature Boxplots](outputs/temperature_boxplots.png)

### 5. Pairplot of Recommended Features & Target
![Pairplot](outputs/pairplot_features.png)

---

## 🛠️ Machine Learning Methodology & Stacking Architecture

```mermaid
flowchart TD
    A["Raw Dataset (14,217 Rows)"] --> B["Leak-Free Feature Matrix (5 Features)"]
    B --> C["80/20 Train/Test Split (11,373 Train / 2,844 Test)"]
    C --> D["StandardScaler Normalization"]
    
    subgraph Level0 ["Level-0 Base Learners (Tuned via 5-Fold CV)"]
        D --> E1["XGBoost Regressor"]
        D --> E2["SVR (RBF Kernel)"]
        D --> E3["Random Forest Regressor"]
    end
    
    subgraph Stacking ["Level-1 Stacking Meta-Learners (Out-of-Fold 5-Fold CV)"]
        E1 & E2 & E3 --> F1["Ridge Meta-Learner (Continuous Benchmark)"]
        E1 & E2 & E3 --> F2["LogisticRegressionAsRegressor (Quantile Expected Value)"]
    end
    
    F1 --> G1["Final Predictions & Metrics (R²=0.9997, RMSE=0.2685%)"]
    F2 --> G2["Discretization Benchmark Comparison (R²=0.9939, RMSE=1.1845%)"]
```

### The Custom `LogisticRegressionAsRegressor` Meta-Learner Wrapper
Logistic Regression is mathematically a discrete classifier. To implement it as a stacking meta-learner on continuous voltage efficiency without library failure, a Scikit-Learn 1.6+ compliant wrapper was created:
1. Target $\mathbf{y}$ is discretized into $K=20$ quantile intervals.
2. The model fits multinomial logistic classification over the out-of-fold base predictions.
3. At inference, it computes the expected continuous value across class probabilities:
   $$\hat{y} = \sum_{k=1}^{K} P(C_k \mid \mathbf{X}) \cdot \bar{c}_k$$
   where $\bar{c}_k = \frac{\text{bin}_{k} + \text{bin}_{k+1}}{2}$ is the quantile midpoint.

---

## ⚙️ Hyperparameter Optimization & LOOCV Results

### Hyperparameter Search Strategy
- **Search Method**: 5-Fold `RandomizedSearchCV` ($n\_iter=30$).
- **Optimal Hyperparameters Discovered**:
  - **XGBoost**: `n_estimators=200`, `max_depth=7`, `learning_rate=0.2`, `subsample=1.0`, `colsample_bytree=0.8`, `random_state=42`
  - **SVR**: `kernel='rbf'`, `gamma='auto'`, `epsilon=0.1`, `C=100`
  - **Random Forest**: `n_estimators=200`, `min_samples_leaf=1`, `max_features='sqrt'`, `max_depth=None`, `random_state=42`

### Leave-One-Out Cross-Validation (LOOCV) Final Evaluation
- **Computational Barrier**: True LOOCV on the complete 14,217-row dataset requires $14,217 \times K_{\text{combinations}}$ fits, which is computationally prohibitive ($>500,000$ model fits).
- **Representative Final LOOCV**: Conducted on a stratified representative subsample:
  - **$\text{RMSE}_{\text{LOOCV}}$**: **$1.3268\%$**
  - **$R^2_{\text{LOOCV}}$**: **$0.9927$**

---

## 🏆 Comprehensive Model Performance Benchmark

### Exact Test Set Evaluation Numbers (2,844 Held-Out Samples)

| Rank | Model Architecture | $R^2$ Score | RMSE (%) | MAE (%) | MAPE (%) | Status / Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **🥇 1** | **Stacked Ensemble (Ridge Meta)** | **0.999688** | **0.268454** | **0.113455** | **0.2602%** | **Top Overall Performer** (Best error reduction) |
| **🥈 2** | **Random Forest Regressor** | **0.999680** | **0.271724** | **0.113183** | **0.2587%** | Outstanding nonlinear boundary capture |
| **🥉 3** | **XGBoost Regressor** | **0.999571** | **0.314704** | **0.143608** | **0.3277%** | Fast gradient boosted regression |
| **4** | **Stacked Ensemble (LogReg Meta)** | **0.993924** | **1.184454** | **0.604079** | **1.2382%** | Requested custom binned classification wrapper |
| **5** | **SVR (RBF Kernel)** | **0.993914** | **1.185403** | **0.519848** | **0.8863%** | Smooth support vector hyperplane fit |

---

## 📊 Diagnostic Visualizations & Interpretability

### 1. Feature Importance Breakdown
- **XGBoost**: `current_density_mA_cm2` (**62.7%**), `anode_pressure_diff_kPa` (**36.6%**), `cathode_pressure_diff_kPa` (**0.4%**), `anode_avg_temp_C` (**0.2%**), `cathode_avg_temp_C` (**0.1%**)
- **Random Forest**: `current_density_mA_cm2` (**39.8%**), `anode_pressure_diff_kPa` (**34.5%**), `cathode_pressure_diff_kPa` (**21.8%**), `anode_avg_temp_C` (**3.2%**), `cathode_avg_temp_C` (**0.7%**)
- **SVR Permutation Importance**: `current_density_mA_cm2` (**1.365**), `cathode_pressure_diff_kPa` (**0.362**), `anode_avg_temp_C` (**0.252**), `anode_pressure_diff_kPa` (**0.218**), `cathode_avg_temp_C` (**0.108**)

![Feature Importance](outputs/feature_importance.png)

### 2. Residual Diagnostics (Predicted vs Actual & Residuals vs Fitted)
- Tree and Ridge stack residuals are tightly distributed around $0.00\%$ ($\sigma_{\text{res}} < 0.27\%$) with zero heteroscedastic fan-out across operating currents.
- Logistic Regression meta-learner exhibits small step-wise quantization residuals due to 20-bin discretization.

![Residual Plots](outputs/residual_plots.png)

### 3. Multi-Metric Model Comparison Bar Chart
![Model Comparison](outputs/model_comparison.png)

---

## 📁 Repository Structure

```text
├── outputs/                                 # Serialized models, tables, and high-res plots
│   ├── correlation_heatmap.png              # 15x15 Pearson correlation matrix (300 DPI)
│   ├── distributions.png                    # Feature histograms with KDE overlays (300 DPI)
│   ├── scatter_vs_target.png                # Features vs Efficiency scatter plots (300 DPI)
│   ├── temperature_boxplots.png             # Temperature sensor boxplots (300 DPI)
│   ├── pairplot_features.png                # Subsampled pairplot of features (300 DPI)
│   ├── feature_importance.png               # XGBoost, RF & SVR permutation importance (300 DPI)
│   ├── residual_plots.png                   # Parity plots and residual diagnostics (300 DPI)
│   ├── model_comparison.png                 # Bar charts comparing R², RMSE, MAE, MAPE (300 DPI)
│   ├── metrics_comparison.csv               # Tabular benchmark metrics on test set
│   ├── summary_stats.csv                    # Complete descriptive statistics table
│   ├── eda_observations.txt                 # Bullet-point EDA written takeaways
│   ├── scaler.joblib                        # Fitted StandardScaler
│   ├── model_stack_ridge.joblib             # Best Trained Stacking Regressor (Ridge)
│   ├── model_stack_logreg.joblib            # Trained Stacking Regressor (Logistic Regression)
│   ├── model_xgboost.joblib                 # Trained XGBoost Regressor
│   ├── model_rf.joblib                      # Trained Random Forest Regressor
│   └── model_svr.joblib                     # Trained SVR Regressor
├── pemfc_efficiency_clean.xlsx              # Cleaned dataset (Clean_Data sheet, 14,217 rows)
├── part1_eda.py                             # Exploratory Data Analysis execution script
├── part2_modeling.py                        # Crash-free training & evaluation script (n_jobs=1)
├── pemfc_efficiency_pipeline.ipynb          # Clean, interactive Jupyter Notebook
├── PEMFC_Voltage_Efficiency_Report.md       # Comprehensive scientific findings report
└── README.md                                # Complete GitHub documentation
```

---

## 🚀 Installation & Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/pemfc-efficiency-prediction.git
cd pemfc-efficiency-prediction

# Install requirements
pip install numpy pandas matplotlib seaborn scikit-learn xgboost openpyxl joblib
```

### 2. Run EDA Script
```bash
python part1_eda.py
```

### 3. Run Training & Diagnostics Pipeline
```bash
python part2_modeling.py
```

### 4. Run Interactively in Jupyter Notebook
```bash
jupyter notebook pemfc_efficiency_pipeline.ipynb
```

---

## 💡 Engineering Insights & Hardware Recommendations

1. **Sensor Reduction for Edge Hardware**: In automotive ECUs and embedded microcontrollers with limited analog input channels, differential pressure transducers can be omitted. `current_density` accounts for $>60\%$ of importance and is $r=0.99$ collinear with pressure drops.
2. **Meta-Learner Architecture**: Continuous linear blending (**Ridge Regression**) is strongly recommended over discrete classification wrappers for continuous fuel cell efficiency modeling because it avoids bin quantization error.

---

## 🚁 Drone Weight Optimization Component

This is the previously incomplete part of the project. It is a self-contained,
reproducible engineering optimization study: real component data → EDA → model
benchmark → constrained minimum-weight design → algorithm benchmark → validation.

### Why a separate UAV dataset

An audit of both PEMFC workbooks (`audit_dataset.py` → `outputs/data_audit.json`)
found **zero** columns containing mass, weight, thrust, payload, battery, motor,
propeller, frame, endurance or power density. The project's own `Notes` sheet states
that drone power/weight optimisation *"needs a second, separate dataset"*. Rather than
inventing component masses, this component uses the open **FPV-DB** catalogue
(CC BY 4.0, DOI [10.5281/zenodo.23613812](https://doi.org/10.5281/zenodo.23613812)):

| table | rows | content |
|---|---|---|
| quads | 101 | assembled quadrotors, dry mass, AUW, wheelbase, prop size |
| motors | 199 | mass, stator, KV, max cells + **6,786 measured thrust-bench rows** |
| batteries | 326 | capacity, cells, energy, mass, C-rating |
| props | 195 | diameter, pitch, blades, mass |
| stacks / cameras / vtx | 112 / 52 / 169 | avionics masses |

### Verified weight model

```
W_AUW = W_dry + W_battery
W_dry = W_frame + 4·W_motor + 4·W_prop + W_avionics      (measured per aircraft)
W_battery = 5.678 g/Wh · E + 25.20 g                     (R² = 0.972, n = 326)
```

`W_battery` is matched from the pack catalogue by cell count and nearest capacity.
Validated on the **32 aircraft with a published AUW**:

| MAE | MAPE | R² | bias |
|---|---|---|---|
| **22.2 g** | **7.45 %** | **0.969** | +19.4 g |

### EDA highlights

* AUW spans **25.6–2,211 g** (median 245 g); mass variables are strongly right-skewed.
* Strongest correlates: dry mass 0.958, battery mass 0.949, energy 0.916, wheelbase 0.861,
  prop size 0.825. Motor KV is *negatively* correlated (−0.59) because it marks vehicle class.
* Allometric exponents: AUW ∝ wheelbase^2.29, AUW ∝ prop^2.12 (area-like, not cubic).
* VIF: prop size 8.2, cells 6.5, wheelbase 4.8 → regularisation required for linear models.
* Battery is the largest controllable mass (median **36 %** of AUW).
* Full narrative: [`reports/eda_observations.md`](reports/eda_observations.md).

### Model benchmark (total AUW, n = 101)

80/20 hold-out; tuning inside the training split; `RepeatedKFold(5×5)` for stability.

| # | model | test R² | MAE (g) | RMSE (g) | MAPE (%) | CV R² |
|---|---|---|---|---|---|---|
| 1 | Linear Regression | 0.980 | 45.5 | **58.3** | 60.0 | **0.902 ± 0.074** |
| 2 | CatBoost | 0.976 | **39.4** | 63.4 | **20.6** | 0.847 ± 0.170 |
| 3 | XGBoost | 0.974 | 45.2 | 67.0 | 41.3 | 0.837 ± 0.171 |
| 4 | Gradient Boosting | 0.972 | 40.4 | 69.3 | 20.0 | 0.822 ± 0.180 |
| 5 | Random Forest | 0.968 | 46.2 | 73.9 | 26.2 | 0.814 ± 0.177 |
| 6 | Extra Trees | 0.964 | 47.5 | 78.6 | 25.1 | 0.861 ± 0.129 |
| 7 | Lasso | 0.962 | 59.6 | 81.0 | 60.2 | 0.880 ± 0.089 |
| 8 | Ridge | 0.960 | 55.5 | 82.4 | 49.9 | 0.869 ± 0.096 |
| 9 | MLP (32-32) | 0.905 | 79.4 | 127.4 | 44.6 | 0.803 ± 0.127 |
| 10 | SVR (RBF) | 0.540 | 96.0 | 280.2 | 30.5 | 0.668 ± 0.247 |

The test R² is optimistic; the repeated-CV column is the honest estimate. With 101 samples,
linear regression is the most stable and neural networks are not supported.

### Optimization formulation

Decision variables: propulsion group *i* (motor + prop + cells, 392 measured groups),
battery pack *j* (326 packs, cell-matched), parallel packs *n* ∈ {1,2,3}.

```
min  W_total = W_airframe + 4(m_motor,i + m_prop,i) + n·m_batt,j + m_payload

s.t. C1  4·T_max,i / W_total >= 2.0                      static thrust-to-weight
     C2  60·n·E_j·η / P_h >= t_req,
         P_h = 4·V_i·I_i(W_total/4)                      hover endurance (measured current)
     C3  D_min <= D_i <= D_max                            prop/frame compatibility
     C4  S_i <= S_max,i                                   voltage limit
     C5  I_h <= n·C_rate,j·C_j/1000                       discharge limit
     C6  W_total <= 2500 g                                extrapolation cap
```

`η = 0.80` usable depth of discharge is a stated assumption. `W_airframe` is an **explicit
design input** because the dataset has no bare-frame column; it is varied by ±30 % in the
sensitivity study. `T_max` and `I(T)` come from measured bench tables, not from a model.
The full derivation is in [`reports/METHODOLOGY.md`](reports/METHODOLOGY.md).

### Results

| scenario | optimum AUW | TWR | endurance | vs median feasible build | vs real aircraft + payload |
|---|---|---|---|---|---|
| 3-inch | **460.1 g** | 3.04 | 8.6 min | −43.4 % | +1.4 % (n=35) |
| 5-inch | **1,138.5 g** | 3.34 | 11.3 min | −31.8 % | −1.3 % (n=17) |
| 7-inch | **1,715.7 g** | 4.93 | 12.4 min | −22.0 % | −35.2 % (n=2) |

The first baseline is a typical *feasible catalogue combination* and is heavily
over-provisioned; the second is a real aircraft of the same prop class plus the same
payload. Both are reported so the saving is not overstated.

### Algorithm comparison

Exhaustive enumeration over 8k–47k candidate designs is tractable (<1 s) and gives the
**certified global optimum**, so it is the recommended method. Six metaheuristics were
still benchmarked (8 seeds × 1,200 evaluations):

| algorithm | best feasible (g) | mean (g) | SD (g) | success vs optimum | gap | time (s) |
|---|---|---|---|---|---|---|
| Genetic Algorithm | 1,106.4 | 1,197.5 | 66.2 | 0.17 | 0.35 % | 0.07 |
| Bayesian (TPE) | 1,106.4 | 1,189.1 | 49.1 | 0.21 | 0.35 % | 2.96 |
| Random Search | 1,121.6 | 1,180.4 | 44.3 | 0.08 | 1.81 % | 0.05 |
| Particle Swarm | 1,125.0 | 1,204.8 | 50.7 | 0.08 | 1.98 % | 0.06 |
| Differential Evolution | 1,149.0 | 1,208.4 | 36.1 | 0.08 | 4.09 % | 0.09 |
| Grid Search | 1,243.2 | 1,243.2 | 0.0 | 0.00 | 15.60 % | 0.01 |

### Sensitivity (5-inch)

* Payload dominates: 0 g → 474 g, 500 g → 1,139 g, 1,500 g → 2,479 g (super-linear).
* Endurance: 5 min → 980 g, 12 min → 1,142 g, 20 min → 1,776 g (convex; steep above ~12 min).
* TWR 1.5 → 3.0 leaves the optimum unchanged (selected design already has TWR 3.34):
  **energy, not thrust, is the binding constraint** at this payload.
* Airframe mass ±30 % shifts the optimum 1,017–1,212 g, essentially one-for-one.

### Deliverables map

| deliverable | location |
|---|---|
| Drone EDA notebook (executed) | `notebooks/01_drone_weight_eda.ipynb` |
| Modelling notebook (executed) | `notebooks/02_drone_weight_modeling.ipynb` |
| Optimization notebook (executed) | `notebooks/03_drone_weight_optimization.ipynb` |
| Preprocessing pipeline | `src/data_pipeline.py`, `data/download_data.py` |
| Model benchmark table | `outputs/drone_weight/model_benchmark.csv` |
| Optimization implementation | `src/optimization.py` |
| Algorithm comparison table | `outputs/drone_weight/algorithm_comparison.csv` |
| Optimized component weights | `outputs/drone_weight/optimized_component_weights.csv` |
| Original vs optimized | `outputs/drone_weight/figures/original_vs_optimized.png` |
| Convergence plots | `outputs/drone_weight/figures/convergence_curves.png` |
| Sensitivity plots | `outputs/drone_weight/figures/sensitivity_analysis.png` |
| Feasibility / trade-off plots | `outputs/drone_weight/figures/constraint_feasibility.png` |
| Methodology and mathematics | `reports/METHODOLOGY.md` |
| Research references (DOI-verified) | `reports/REFERENCES.md` |
| Data audit (why PEMFC data is unusable) | `outputs/data_audit.json` |

### How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS only: XGBoost needs the OpenMP runtime
# brew install libomp

python data/download_data.py     # fetch FPV-DB and verify SHA-256
python run_all.py                # pipeline -> EDA -> models -> optimization -> figures
```

Expected runtime: ~1 min for EDA, ~2 min for the model benchmark, ~2.5 min for the
optimization (6 algorithms × 8 seeds × 3 scenarios). Everything is seeded (`random_state=42`).

To regenerate the executed notebooks:

```bash
python notebooks/build_notebooks.py
```

### Limitations

1. Bare-frame mass is not in the dataset; it is a documented input with ±30 % sensitivity.
2. Usable battery fraction η = 0.80 is an assumption; endurance scales linearly with it.
3. Bench thrust/current are steady-state at fixed voltage (no sag, heating or ESC losses).
4. Payload is additive and drives C1/C2, but there is no numeric payload column to calibrate
   a payload-dependent mass cascade.
5. The catalogue is FPV/hobby class (≤ 2.2 kg); C6 caps results at 2.5 kg.
6. No structural stress/fatigue model is possible with these fields.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
