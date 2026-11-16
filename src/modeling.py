"""Benchmark regression models for total drone weight (AUW).

Target: auw_est_g = dry mass + matched battery mass  (n = 101 assembled quads).

Protocol
--------
* 80/20 train/test split (test never touched during tuning).
* RandomizedSearchCV (5-fold) inside the training set only -> no leakage.
* Final metrics on the held-out test set: R2, MAE, RMSE, MAPE.
* Repeated 5-fold CV on the full table for stability (mean +/- sd of R2).
* All preprocessing lives inside a scikit-learn Pipeline.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, FIGURES, OUTPUTS, RANDOM_STATE

warnings.filterwarnings("ignore")

TARGET = "auw_est_g"

NUMERIC_FEATURES = [
    "prop_size_in",
    "wheelbase_mm",
    "wheelbase_missing",
    "cells",
    "capacity_mah",
    "motor_kv",
    "kv_missing",
]
CATEGORICAL_FEATURES = ["class_group"]


def load_design_table() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "quad_design_table.csv")


def build_features(df: pd.DataFrame):
    d = df.copy()
    d["kv_missing"] = d["motor_kv"].isna().astype(int)
    X = d[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = d[TARGET].astype(float)
    return X, y


def make_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num, NUMERIC_FEATURES),
                              ("cat", cat, CATEGORICAL_FEATURES)])


def model_zoo():
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor

    return {
        "Linear Regression": (
            LinearRegression(),
            {},
        ),
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"model__alpha": np.logspace(-3, 3, 25)},
        ),
        "Lasso": (
            Lasso(random_state=RANDOM_STATE, max_iter=20000),
            {"model__alpha": np.logspace(-3, 2, 25)},
        ),
        "SVR (RBF)": (
            SVR(),
            {
                "model__C": [0.3, 1, 3, 10, 30, 100],
                "model__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
                "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=4),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 4, 6, 8],
                "model__min_samples_leaf": [1, 2, 3],
                "model__max_features": ["sqrt", 0.7, 1.0],
            },
        ),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=4),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 4, 6, 8],
                "model__min_samples_leaf": [1, 2, 3],
                "model__max_features": ["sqrt", 0.7, 1.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                random_state=RANDOM_STATE,
                n_jobs=4,
                objective="reg:squarederror",
                tree_method="hist",
            ),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 1.0],
                "model__reg_lambda": [0.5, 1.0, 3.0],
            },
        ),
        "CatBoost": (
            CatBoostRegressor(
                random_state=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
            ),
            {
                "model__iterations": [300, 600],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__depth": [2, 3, 4],
                "model__l2_leaf_reg": [1, 3, 10],
            },
        ),
        "MLP (32-32)": (
            MLPRegressor(
                random_state=RANDOM_STATE,
                max_iter=4000,
                early_stopping=True,
                hidden_layer_sizes=(32, 32),
            ),
            {
                "model__alpha": [1e-4, 1e-3, 1e-2, 0.1],
                "model__learning_rate_init": [0.001, 0.01],
            },
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_g": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_g": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def run_benchmark(save: bool = True) -> tuple[pd.DataFrame, dict]:
    from sklearn.base import clone
    from sklearn.model_selection import (
        RandomizedSearchCV,
        RepeatedKFold,
        cross_val_predict,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline

    df = load_design_table()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    zoo = model_zoo()
    rows, fitted, oof = [], {}, {}
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    for name, (estimator, grid) in zoo.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", estimator)])

        if grid:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=grid,
                n_iter=min(25, int(np.prod([len(v) for v in grid.values()]))),
                scoring="r2",
                cv=5,
                random_state=RANDOM_STATE,
                n_jobs=4,
                refit=True,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params = {k.replace("model__", ""): str(v) for k, v in search.best_params_.items()}
        else:
            best = pipe.fit(X_train, y_train)
            best_params = {}

        pred_test = best.predict(X_test)
        test_metrics = regression_metrics(y_test, pred_test)

        # stability: repeated CV on the full table with the tuned pipeline
        scores = cross_validate(
            clone(best), X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error"),
            n_jobs=4,
        )
        oof_pred = cross_val_predict(clone(best), X, y, cv=RepeatedKFold(
            n_splits=5, n_repeats=1, random_state=RANDOM_STATE
        ), n_jobs=4)
        oof[name] = oof_pred

        rows.append(
            {
                "model": name,
                **test_metrics,
                "cv_R2_mean": float(scores["test_r2"].mean()),
                "cv_R2_sd": float(scores["test_r2"].std()),
                "cv_MAE_mean_g": float(-scores["test_neg_mean_absolute_error"].mean()),
                "best_params": json.dumps(best_params),
            }
        )
        fitted[name] = best
        print(f"{name:20s} test R2={test_metrics['R2']:.3f} "
              f"MAE={test_metrics['MAE_g']:.1f}g RMSE={test_metrics['RMSE_g']:.1f}g "
              f"| CV R2={scores['test_r2'].mean():.3f}+/-{scores['test_r2'].std():.3f}")

    metrics = pd.DataFrame(rows).sort_values("RMSE_g").reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))

    artefacts = {
        "X_test": X_test, "y_test": y_test, "oof": oof,
        "X_train": X_train, "y_train": y_train,
        "X_all": X, "y_all": y, "fitted": fitted,
    }

    if save:
        metrics.to_csv(OUTPUTS / "model_benchmark.csv", index=False)
        json.dump(
            {
                "target": TARGET,
                "n_rows": int(len(df)),
                "n_features": int(X.shape[1]),
                "split": "80/20 hold-out, random_state=42",
                "cv": "RepeatedKFold(5 folds x 5 repeats)",
                "metrics": metrics.to_dict(orient="records"),
            },
            open(OUTPUTS / "model_benchmark.json", "w"),
            indent=2,
        )
        pd.DataFrame(
            {"y_true": y_test, **{k: v for k, v in {"pred_" + n: fitted[n].predict(X_test)
                                                    for n in fitted}.items()}}
        ).to_csv(OUTPUTS / "test_predictions.csv", index=False)
        import joblib
        joblib.dump(fitted, OUTPUTS / "fitted_models.joblib")

    return metrics, artefacts


def plots(metrics: pd.DataFrame, artefacts: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    y_test = artefacts["y_test"]

    fig, ax = plt.subplots(figsize=(11, 5))
    m = metrics.sort_values("RMSE_g")
    ax.barh(m["model"], m["RMSE_g"], color="#3b6ea5")
    ax.set_xlabel("Test RMSE (g)")
    ax.set_title("Model benchmark — total drone weight (AUW)")
    for i, (v, r2) in enumerate(zip(m["RMSE_g"], m["R2"])):
        ax.text(v, i, f"  {v:.0f} g | R²={r2:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_rmse_comparison.png", dpi=200)
    plt.close(fig)

    best_name = metrics.iloc[0]["model"]
    pred = artefacts["oof"][best_name]
    y_all = artefacts["y_all"]
    fitted = artefacts["fitted"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(y_test, fitted[best_name].predict(artefacts["X_test"]),
                    s=45, alpha=0.8, edgecolor="k", linewidth=0.4)
    lim = [0, max(y_all.max(), 1) * 1.05]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel("Actual AUW (g)")
    axes[0].set_ylabel("Predicted AUW (g)")
    axes[0].set_title(f"Best model: {best_name} (held-out test set)")
    axes[1].scatter(pred, y_all, s=25, alpha=0.65, color="#7a4fa0")
    axes[1].plot(lim, lim, "r--", lw=1)
    axes[1].set_xlabel("Out-of-fold predicted AUW (g)")
    axes[1].set_ylabel("Actual AUW (g)")
    axes[1].set_title(f"Out-of-fold predictions (all {len(y_all)} quads)")
    fig.tight_layout()
    fig.savefig(FIGURES / "predicted_vs_actual.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    metrics, artefacts = run_benchmark()
    plots(metrics, artefacts)
    print("\nSaved:", OUTPUTS / "model_benchmark.csv")
