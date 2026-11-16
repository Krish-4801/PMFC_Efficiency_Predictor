"""End-to-end driver for the drone weight-optimization component.

    python run_all.py

Stages
------
1. data pipeline     -> data/processed/*
2. EDA               -> outputs/drone_weight/eda_summary.json + figures
3. model benchmark   -> outputs/drone_weight/model_benchmark.csv
4. optimization      -> outputs/drone_weight/optimization_results.json
5. summary table     -> printed + outputs/drone_weight/summary.md

The PEMFC scripts (`part1_eda.py`, `part2_modeling.py`) are NOT touched.
"""
from __future__ import annotations

import json
import time

from src import data_pipeline, eda, modeling, optimization
from src.config import OUTPUTS


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 72)
    print("1/4  data pipeline")
    print("=" * 72)
    data_pipeline.main()

    print("\n" + "=" * 72)
    print("2/4  exploratory data analysis")
    print("=" * 72)
    eda.run_eda()
    print("EDA figures written to", OUTPUTS / "figures")

    print("\n" + "=" * 72)
    print("3/4  model benchmark")
    print("=" * 72)
    metrics, artefacts = modeling.run_benchmark()
    modeling.plots(metrics, artefacts)
    print("best test model:", metrics.iloc[0]["model"])

    print("\n" + "=" * 72)
    print("4/4  constrained optimization")
    print("=" * 72)
    out = optimization.run_all()

    res = json.load(open(OUTPUTS / "optimization_results.json"))
    lines = ["# Drone weight optimization — summary\n",
             "| scenario | optimum AUW | median feasible | reduction | TWR | endurance |",
             "|---|---|---|---|---|---|"]
    for name, s in res["scenarios"].items():
        o = s["global_optimum"]
        b = s["baseline_median_feasible"]
        lines.append(
            f"| {name} | {o['w_total_g']:.1f} g | {b['w_total_g']:.1f} g | "
            f"{s['reduction_vs_median_feasible_pct']:.1f} % | {o['twr']:.2f} | "
            f"{o['endurance_min']:.1f} min |"
        )
    (OUTPUTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    print(f"\nDone in {time.perf_counter() - t0:.1f} s")
    print("Artefacts:", OUTPUTS)


if __name__ == "__main__":
    main()
