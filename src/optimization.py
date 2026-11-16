"""Constrained minimum-weight design of a multirotor from real catalog components.

Problem
-------
Choose a propulsion group (motor + propeller + cell count, each with a measured
thrust/current bench curve) and a battery pack (with parallel-pack count) that
minimise total mass for a required payload and hover endurance, inside a given
airframe envelope.

    minimise  W_total = W_airframe + N_rotor*(m_motor + m_prop)
                        + n_pack*m_battery + m_payload

    subject to
      (C1) N_rotor * T_max  >= TWR_min * W_total            (static thrust)
      (C2) t_hover          >= t_required                    (endurance)
      (C3) D_min <= D_prop  <= D_max                        (prop/frame class)
      (C4) cells            <= motor.max_cells               (voltage limit)
      (C5) I_hover          <= n_pack*C_rating*capacity/1000 (discharge limit)
      (C6) W_total          <= AUW_max                       (extrapolation cap)

(C2) uses the measured current at the thrust needed to hover, interpolated from
the bench curve - no assumed figure of merit.

Every mass, thrust and current value comes from the FPV-DB catalogue except
W_airframe, which is an explicit design input (the dataset has no bare-frame
column) and is varied in the sensitivity study.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    DATA_PROCESSED,
    FIGURES,
    LIPO_NOMINAL_V,
    MAX_AUW_KG,
    MIN_TWR,
    N_ROTORS,
    OUTPUTS,
    RANDOM_STATE,
    SCENARIOS,
    USABLE_BATTERY_FRACTION,
)

PENALTY = 1.0e5


# ----------------------------------------------------------------------------
# Catalogue preparation
# ----------------------------------------------------------------------------

def _parse_max_cells(value) -> float:
    import re
    if pd.isna(value):
        return np.nan
    nums = [int(x) for x in re.findall(r"\d+", str(value))]
    return float(max(nums)) if nums else np.nan


def _parse_c_rating(value) -> float:
    import re
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m else np.nan


@dataclass
class Catalogue:
    groups: pd.DataFrame
    batteries: pd.DataFrame
    group_curves: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls) -> "Catalogue":
        g = pd.read_csv(DATA_PROCESSED / "propulsion_groups.csv")
        b = pd.read_csv(DATA_PROCESSED / "battery_catalog.csv")
        g["motor_max_cells_n"] = g["motor_max_cells"].map(_parse_max_cells)
        b["c_rating_n"] = b["c_rating"].map(_parse_c_rating)
        b = b.dropna(subset=["weight_g", "energy_wh", "capacity_mah", "cells_n"])
        b["max_discharge_A"] = b["c_rating_n"] * b["capacity_mah"] / 1000.0
        curves = []
        for _, r in g.iterrows():
            T = np.array(json.loads(r["thrust_curve_g"]), dtype=float)
            I = np.array(json.loads(r["current_curve_A"]), dtype=float)
            V = np.array(json.loads(r["voltage_curve_V"]), dtype=float)
            order = np.argsort(T)
            T, I, V = T[order], I[order], V[order]
            keep = np.concatenate([[True], np.diff(T) > 1e-9])
            curves.append({"T": T[keep], "I": I[keep], "V": V[keep],
                           "T_max": float(T.max())})
        return cls(groups=g.reset_index(drop=True), batteries=b.reset_index(drop=True),
                   group_curves=curves)

    def candidates(self, scenario: dict) -> pd.DataFrame:
        g, b = self.groups, self.batteries
        sel = (g["prop_diameter_in"] <= scenario["max_prop_in"]) & g["max_thrust_g"].gt(0)
        if "min_prop_in" in scenario:
            sel &= g["prop_diameter_in"] >= scenario["min_prop_in"]
        rows = []
        for gi in np.where(sel)[0]:
            gr = g.iloc[gi]
            if pd.notna(gr["motor_max_cells_n"]) and gr["cells"] > gr["motor_max_cells_n"]:
                continue
            bb = b.index[b["cells_n"] == gr["cells"]]
            if len(bb) == 0:
                continue
            for bi in bb:
                for npack in (1, 2, 3):
                    rows.append((int(gi), int(bi), npack))
        return pd.DataFrame(rows, columns=["group_idx", "battery_idx", "n_packs"])


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

class DesignProblem:
    def __init__(self, cat: Catalogue, scenario: dict,
                 min_twr: float = MIN_TWR, max_auw_kg: float = MAX_AUW_KG,
                 airframe_g: float | None = None, payload_kg: float | None = None,
                 endurance_min: float | None = None):
        self.cat = cat
        self.min_prop_in = scenario.get("min_prop_in", 0.0)
        self.max_prop_in = scenario["max_prop_in"]
        self.airframe_g = scenario["airframe_g"] if airframe_g is None else airframe_g
        self.payload_g = 1000 * (scenario["payload_kg"] if payload_kg is None else payload_kg)
        self.endurance_min = scenario["endurance_min"] if endurance_min is None else endurance_min
        self.min_twr = min_twr
        self.max_auw_g = max_auw_kg * 1000
        self.cand = cat.candidates(scenario)
        self.g, self.b = cat.groups, cat.batteries
        self.curves = cat.group_curves
        self.n_evals = 0

        # constraint-aware decoding pools (used by all metaheuristics)
        self.valid_groups = np.unique(self.cand["group_idx"].values)
        self.battery_pool = {
            int(gi): self.cand.loc[self.cand["group_idx"] == gi, "battery_idx"].unique()
            for gi in self.valid_groups
        }

    # -- constraint-aware decoding ------------------------------------------
    def decode(self, x) -> tuple[int, int, int]:
        g = int(self.valid_groups[int(x[0]) % len(self.valid_groups)])
        pool = self.battery_pool[g]
        b = int(pool[int(x[1]) % len(pool)])
        n = int(np.clip(int(x[2]), 1, 3))
        return g, b, n

    # -- single design -------------------------------------------------------
    def evaluate(self, group_idx: int, battery_idx: int, n_packs: int,
                 detail: bool = False) -> dict:
        self.n_evals += 1
        gr = self.g.iloc[int(group_idx)]
        br = self.b.iloc[int(battery_idx)]
        curve = self.curves[int(group_idx)]

        w_total = (self.airframe_g
                   + N_ROTORS * (gr["motor_mass_g"] + gr["prop_mass_g"])
                   + n_packs * br["weight_g"]
                   + self.payload_g)

        twr = N_ROTORS * curve["T_max"] / w_total
        t_hover = w_total / N_ROTORS
        if t_hover <= curve["T"][0]:
            i_hover = curve["I"][0] * t_hover / curve["T"][0]
            v_hover = curve["V"][0]
        elif t_hover <= curve["T"][-1]:
            i_hover = float(np.interp(t_hover, curve["T"], curve["I"]))
            v_hover = float(np.interp(t_hover, curve["T"], curve["V"]))
        else:
            i_hover, v_hover = np.inf, curve["V"][-1]

        p_hover = N_ROTORS * v_hover * i_hover
        energy = n_packs * br["energy_wh"] * USABLE_BATTERY_FRACTION
        endurance = 60.0 * energy / p_hover if np.isfinite(p_hover) and p_hover > 0 else 0.0

        viol = [
            max(0.0, (self.min_twr - twr) / self.min_twr),
            max(0.0, (self.endurance_min - endurance) / self.endurance_min),
            max(0.0, (gr["prop_diameter_in"] - self.max_prop_in) / self.max_prop_in),
            max(0.0, (self.min_prop_in - gr["prop_diameter_in"]) / self.max_prop_in),
        ]
        mc = gr["motor_max_cells_n"]
        viol.append(0.0 if pd.isna(mc) else max(0.0, (gr["cells"] - mc) / mc))
        max_i = n_packs * br["max_discharge_A"]
        viol.append(0.0 if not np.isfinite(max_i) or max_i <= 0
                    else max(0.0, (i_hover - max_i) / max_i))
        viol.append(max(0.0, (w_total - self.max_auw_g) / self.max_auw_g))
        violation = float(sum(viol))

        out = {"w_total_g": float(w_total), "twr": float(twr),
               "endurance_min": float(endurance), "hover_current_A": float(i_hover),
               "violation": violation, "feasible": violation <= 1e-9}
        if detail:
            out.update({
                "group_idx": int(group_idx), "battery_idx": int(battery_idx),
                "n_packs": int(n_packs),
                "motor_slug": gr["motor_slug"], "motor_model": gr["motor_model"],
                "motor_mass_g": float(gr["motor_mass_g"]),
                "prop_key": gr["prop_key"], "prop_diameter_in": float(gr["prop_diameter_in"]),
                "prop_mass_g": float(gr["prop_mass_g"]),
                "cells": int(gr["cells"]), "kv": float(gr["kv"]),
                "max_thrust_per_motor_g": float(curve["T_max"]),
                "battery_model": br["model"], "battery_mass_g": float(br["weight_g"]),
                "battery_energy_wh": float(br["energy_wh"]),
                "battery_capacity_mah": float(br["capacity_mah"]),
                "battery_c_rating": br["c_rating"],
                "airframe_g": float(self.airframe_g), "payload_g": float(self.payload_g),
                "propulsion_mass_g": float(N_ROTORS * (gr["motor_mass_g"] + gr["prop_mass_g"])),
                "battery_total_g": float(n_packs * br["weight_g"]),
            })
        return out

    def score(self, x):
        """Return (penalised fitness, feasible weight-or-inf)."""
        d = self.evaluate(*self.decode(x))
        f = d["w_total_g"] + PENALTY * d["violation"]
        return f, (d["w_total_g"] if d["feasible"] else np.inf)

    def exhaustive(self) -> tuple[dict, pd.DataFrame]:
        recs = [self.evaluate(int(gi), int(bi), int(n), detail=True)
                for gi, bi, n in self.cand.itertuples(index=False)]
        df = pd.DataFrame(recs)
        feas = df[df["feasible"]]
        best = (feas.sort_values("w_total_g").iloc[0] if len(feas)
                else df.sort_values(["violation", "w_total_g"]).iloc[0]).to_dict()
        return best, df


# ----------------------------------------------------------------------------
# Algorithms (all share the same constraint-aware decoder)
# ----------------------------------------------------------------------------

def _bounds(problem: DesignProblem):
    return [(0, max(1, len(problem.valid_groups) - 1)),
            (0, 40), (1, 3)]


def run_random(problem, budget, seed):
    rng = np.random.default_rng(seed)
    b = _bounds(problem)
    best_x, best_f, best_w = None, np.inf, np.inf
    hist = []
    for _ in range(budget):
        x = [rng.integers(lo, hi + 1) for lo, hi in b]
        f, w = problem.score(x)
        if w < best_w or (best_x is None):
            best_x, best_w = x, min(best_w, w)
        best_f = min(best_f, f)
        hist.append(best_w)
    return best_x, best_f, hist


def run_ga(problem, budget, seed, pop_size=40):
    rng = np.random.default_rng(seed)
    b = _bounds(problem)
    lo = np.array([z[0] for z in b]); hi = np.array([z[1] for z in b])

    def rand_ind():
        return rng.integers(lo, hi + 1)

    pop = np.array([rand_ind() for _ in range(pop_size)])
    fits, feas_w = [], []
    for p in pop:
        f, w = problem.score(p); fits.append(f); feas_w.append(w)
    fits, feas_w = np.array(fits), np.array(feas_w)
    hist, best_w = [], np.inf
    gens = max(1, budget // pop_size)
    for _ in range(gens):
        order = np.argsort(fits)
        pop, fits, feas_w = pop[order], fits[order], feas_w[order]
        best_w = min(best_w, float(np.min(feas_w)))
        hist.append(best_w)
        new = [pop[0].copy(), pop[1].copy()]
        while len(new) < pop_size:
            idx = rng.integers(0, pop_size, 3)
            p1 = pop[idx[np.argmin(fits[idx])]]
            idx = rng.integers(0, pop_size, 3)
            p2 = pop[idx[np.argmin(fits[idx])]]
            child = np.where(rng.random(len(lo)) < 0.5, p1, p2).copy()
            mut = rng.random(len(lo)) < 0.15
            child[mut] = rng.integers(lo[mut], hi[mut] + 1)
            new.append(child)
        pop = np.array(new)
        fits, feas_w = [], []
        for p in pop:
            f, w = problem.score(p); fits.append(f); feas_w.append(w)
        fits, feas_w = np.array(fits), np.array(feas_w)
    i = int(np.argmin(fits))
    hist.append(min(best_w, float(feas_w[i])))
    return pop[i], float(fits[i]), hist


def run_pso(problem, budget, seed, swarm=40):
    rng = np.random.default_rng(seed)
    b = _bounds(problem)
    lo = np.array([z[0] for z in b], float); hi = np.array([z[1] for z in b], float)
    span = np.maximum(hi - lo, 1)

    x = lo + rng.random((swarm, len(lo))) * span
    v = (rng.random((swarm, len(lo))) - 0.5) * span * 0.1
    xi = np.rint(x).astype(int)
    fit = np.array([problem.score(p)[0] for p in xi])
    w_arr = np.array([problem.score(p)[1] for p in xi])
    pbest, pbest_fit = xi.copy(), fit.copy()
    j = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[j].copy(), float(pbest_fit[j])
    best_w = float(np.min(w_arr))
    hist = [best_w]
    iters = max(1, budget // swarm)
    for _ in range(iters):
        r1, r2 = rng.random((swarm, len(lo))), rng.random((swarm, len(lo)))
        v = 0.72 * v + 1.5 * r1 * (pbest - xi) + 1.5 * r2 * (gbest - xi)
        v = np.clip(v, -span * 0.25, span * 0.25)
        x = np.clip(x + v, lo, hi)
        xi = np.rint(x).astype(int)
        fit, w_arr = [], []
        for p in xi:
            f, w = problem.score(p); fit.append(f); w_arr.append(w)
        fit, w_arr = np.array(fit), np.array(w_arr)
        improved = fit < pbest_fit
        pbest[improved], pbest_fit[improved] = xi[improved], fit[improved]
        j = int(np.argmin(pbest_fit))
        if pbest_fit[j] < gbest_fit:
            gbest_fit, gbest = float(pbest_fit[j]), pbest[j].copy()
        best_w = min(best_w, float(np.min(w_arr)))
        hist.append(best_w)
    return gbest, gbest_fit, hist


def run_de(problem, budget, seed):
    from scipy.optimize import differential_evolution

    b = _bounds(problem)
    popsize = 15
    maxiter = max(1, budget // (popsize * len(b)))
    hist, state = [], {"best": np.inf}

    def obj(x):
        f, w = problem.score(np.rint(x).astype(int))
        if w < state["best"]:
            state["best"] = w
        hist.append(state["best"])
        return f

    res = differential_evolution(obj, b, seed=seed, popsize=popsize, maxiter=maxiter,
                                 integrality=[1, 1, 1], polish=False, updating="immediate", tol=0)
    return np.rint(res.x).astype(int), float(res.fun), hist


def run_tpe(problem, budget, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    b = _bounds(problem)
    hist, state = [], {"best": np.inf}

    def objective(trial):
        x = [trial.suggest_int(f"x{i}", z[0], z[1]) for i, z in enumerate(b)]
        f, w = problem.score(x)
        if w < state["best"]:
            state["best"] = w
        hist.append(state["best"])
        return f

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget)
    x = [study.best_params[f"x{i}"] for i in range(len(b))]
    return np.array(x), float(study.best_value), hist


def run_grid(problem, budget, seed):
    """Coarse grid over group rank x battery rank x pack count (all decoded)."""
    b = _bounds(problem)
    grid = [np.linspace(z[0], z[1], 8 if i == 0 else 4).astype(int)
            for i, z in enumerate(b)]
    hist, best_f, best_x, best_w = [], np.inf, None, np.inf
    for x0 in grid[0]:
        for x1 in grid[1]:
            for x2 in grid[2]:
                f, w = problem.score([x0, x1, x2])
                if w < best_w:
                    best_w, best_x = w, [x0, x1, x2]
                if f < best_f:
                    best_f = f
                hist.append(best_w)
    return best_x, best_f, hist


ALGORITHMS = {
    "Random Search": run_random,
    "Grid Search": run_grid,
    "Genetic Algorithm": run_ga,
    "Particle Swarm": run_pso,
    "Differential Evolution": run_de,
    "Bayesian (TPE)": run_tpe,
}


# ----------------------------------------------------------------------------
# Scenario driver
# ----------------------------------------------------------------------------

def solve_scenario(name: str, cat: Catalogue, n_runs: int = 8,
                   budget: int = 1200, seed: int = RANDOM_STATE) -> dict:
    scenario = SCENARIOS[name]
    problem = DesignProblem(cat, scenario)
    t0 = time.perf_counter()
    best_exact, all_designs = problem.exhaustive()
    t_exact = time.perf_counter() - t0

    feas = all_designs[all_designs["feasible"]]
    # Baseline A: typical component selection (median mass of each component
    # family inside the scenario's valid propulsion set). This isolates the
    # benefit of optimisation from the combinatorial size of the candidate set.
    vg = problem.valid_groups
    med_motor = float(cat.groups.loc[vg, "motor_mass_g"].median())
    med_prop = float(cat.groups.loc[vg, "prop_mass_g"].median())
    cells_ok = cat.groups.loc[vg, "cells"].unique()
    med_batt = float(cat.batteries.loc[cat.batteries["cells_n"].isin(cells_ok), "weight_g"].median())
    typical_components_g = (problem.airframe_g + N_ROTORS * (med_motor + med_prop)
                            + med_batt + problem.payload_g)
    baseline = {
        "w_total_g": typical_components_g,
        "median_motor_mass_g": med_motor,
        "median_prop_mass_g": med_prop,
        "median_battery_mass_g": med_batt,
        "definition": "airframe + 4*(median motor + median prop) + 1 median battery + payload",
    }
    # Baseline C: the median *feasible* design (a representative build that
    # actually meets the requirements). Used for the headline reduction.
    median_feasible = feas.sort_values("w_total_g").iloc[len(feas) // 2].to_dict() if len(feas) else None
    # Baseline B: real aircraft of this prop class from the quad table, plus payload.
    quads = pd.read_csv(DATA_PROCESSED / "quad_design_table.csv")
    cls = quads[(quads["prop_size_in"] >= scenario.get("min_prop_in", 0))
                & (quads["prop_size_in"] <= scenario["max_prop_in"])]
    real_class_g = (float(cls["auw_est_g"].median()) + problem.payload_g) if len(cls) else np.nan
    baseline["real_class_auw_plus_payload_g"] = real_class_g
    baseline["real_class_n"] = int(len(cls))
    opt_w = best_exact["w_total_g"]

    rows, curves = [], {}
    for algo_name, fn in ALGORITHMS.items():
        run_w, run_time, hit, evals, run_hist = [], [], [], [], []
        for s in range(n_runs):
            p = DesignProblem(cat, scenario)
            t0 = time.perf_counter()
            x, f, hist = fn(p, budget, seed + s)
            dt = time.perf_counter() - t0
            d = p.evaluate(*p.decode(x), detail=True)
            run_w.append(d["w_total_g"] if d["feasible"] else np.nan)
            run_time.append(dt)
            evals.append(p.n_evals)
            hit.append(bool(d["feasible"]) and abs(d["w_total_g"] - opt_w) < 1e-6)
            run_hist.append(hist)
        curves[algo_name] = run_hist[0]
        valid = np.array(run_w, float)
        found = np.isfinite(valid)
        rows.append({
            "scenario": name,
            "scenario_label": scenario["label"],
            "algorithm": algo_name,
            "best_feasible_g": float(np.nanmin(valid)) if found.any() else np.nan,
            "mean_feasible_g": float(np.nanmean(valid)) if found.any() else np.nan,
            "sd_feasible_g": float(np.nanstd(valid)) if found.any() else np.nan,
            "infeasible_runs": int((~found).sum()),
            "success_rate": float(np.mean(hit)),
            "mean_time_s": float(np.mean(run_time)),
            "sd_time_s": float(np.std(run_time)),
            "evals_per_run": float(np.mean(evals)),
            "gap_to_optimum_pct": float(100 * (np.nanmin(valid) - opt_w) / opt_w)
            if found.any() else np.nan,
        })

    exact = {
        "scenario": name, "scenario_label": scenario["label"],
        "airframe_g": scenario["airframe_g"], "payload_g": scenario["payload_kg"] * 1000,
        "endurance_required_min": scenario["endurance_min"],
        "prop_range_in": [scenario.get("min_prop_in", 0), scenario["max_prop_in"]],
        "n_candidates": int(len(all_designs)), "n_feasible": int(len(feas)),
        "exhaustive_time_s": float(t_exact),
        "global_optimum": best_exact,
        "baseline_typical_components": baseline,
        "baseline_median_feasible": median_feasible,
        "reduction_vs_typical_pct": float(100 * (typical_components_g - opt_w) / typical_components_g),
        "reduction_vs_median_feasible_pct": float(
            100 * (median_feasible["w_total_g"] - opt_w) / median_feasible["w_total_g"])
        if median_feasible else None,
    }
    return {"scenario": name, "exact": exact, "comparison": pd.DataFrame(rows),
            "designs": all_designs, "curves": curves, "problem": problem}


def sensitivity(cat: Catalogue, name: str = "5inch") -> pd.DataFrame:
    scenario = SCENARIOS[name]
    rows = []
    base = DesignProblem(cat, scenario)
    b0, _ = base.exhaustive()
    rows.append({"parameter": "baseline", "value": np.nan,
                 "optimum_g": b0["w_total_g"], "feasible": b0["feasible"]})
    sweeps = {
        "min_twr": ([1.5, 1.75, 2.0, 2.25, 2.5, 3.0], "min_twr"),
        "endurance_min": ([5, 8, 10, 12, 15, 20, 25], "endurance_min"),
        "payload_kg": ([0.0, 0.25, 0.5, 0.75, 1.0, 1.5], "payload_kg"),
        "airframe_scale": ([0.7, 0.85, 1.0, 1.15, 1.3], "airframe_g"),
    }
    for pname, (values, kw) in sweeps.items():
        for v in values:
            kwargs = {kw: (scenario["airframe_g"] * v if kw == "airframe_g" else v)}
            p = DesignProblem(cat, scenario, **kwargs)
            b, _ = p.exhaustive()
            rows.append({"parameter": pname, "value": v,
                         "optimum_g": b["w_total_g"], "feasible": bool(b["feasible"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def component_table(result: dict, avionics_g: float = 29.0) -> pd.DataFrame:
    """Component-wise mass table.

    FPV-DB has no bare-frame column, so the scenario airframe input is split into
    a catalogue-based avionics allocation (median FC/ESC stack + camera + VTX,
    n=53/47/116 => 29.0 g) and a frame residual. The split is an *allocation* of a
    design input, not an independent measurement; the airframe total is preserved
    exactly.
    """
    e = result["exact"]
    o = e["global_optimum"]
    frame_g = max(0.0, o["airframe_g"] - avionics_g)
    rows = [
        ("Frame (residual allocation)", frame_g, "airframe input minus avionics"),
        ("Avionics (FC/ESC + camera + VTX)", avionics_g, "catalogue median allocation"),
        ("Motors", N_ROTORS * o["motor_mass_g"], f"{N_ROTORS} x {o['motor_model']}"),
        ("Propellers", N_ROTORS * o["prop_mass_g"], f"{N_ROTORS} x {o['prop_key']}"),
        ("Battery", o["battery_total_g"], f"{o['n_packs']} x {o['battery_model']}"),
        ("Payload", o["payload_g"], "requirement"),
    ]
    df = pd.DataFrame(rows, columns=["component", "mass_g", "detail"])
    df["scenario"] = e["scenario"]
    df["total_g"] = df["mass_g"].sum()
    df["share_pct"] = 100 * df["mass_g"] / df["total_g"]
    return df


def run_all(n_runs: int = 8, budget: int = 1200, save: bool = True) -> dict:
    cat = Catalogue.load()
    results = {name: solve_scenario(name, cat, n_runs=n_runs, budget=budget)
               for name in SCENARIOS}
    sens = sensitivity(cat)
    comp = pd.concat([component_table(r) for r in results.values()], ignore_index=True)
    comparison = pd.concat([r["comparison"] for r in results.values()], ignore_index=True)

    if save:
        comp.to_csv(OUTPUTS / "optimized_component_weights.csv", index=False)
        comparison.to_csv(OUTPUTS / "algorithm_comparison.csv", index=False)
        sens.to_csv(OUTPUTS / "sensitivity_analysis.csv", index=False)
        payload = {
            "method": "exhaustive enumeration over catalogue components, "
                      "verified against six metaheuristics",
            "n_rotors": N_ROTORS,
            "usable_battery_fraction": USABLE_BATTERY_FRACTION,
            "scenarios": {name: r["exact"] for name, r in results.items()},
            "sensitivity": sens.to_dict(orient="records"),
        }
        (OUTPUTS / "optimization_results.json").write_text(
            json.dumps(payload, indent=2, default=str))
        for name, r in results.items():
            r["designs"].to_csv(OUTPUTS / f"all_designs_{name}.csv", index=False)
        plot_results(results, cat, sens)
    return {"results": results, "sensitivity": sens, "components": comp,
            "comparison": comparison}


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def plot_results(results: dict, cat: Catalogue, sens: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")

    primary = results["5inch"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, hist in primary["curves"].items():
        h = np.asarray(hist, float)
        if len(h) == 0:
            continue
        x = np.linspace(0, 100, len(h))
        ax.plot(x, h, label=name, lw=1.8)
    ax.set_xlabel("Search progress (% of evaluation budget)")
    ax.set_ylabel("Best feasible AUW found (g)")
    ax.set_title("Convergence - 5-inch scenario (representative run)")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES / "convergence_curves.png", dpi=200); plt.close(fig)

    comp = pd.concat([r["comparison"] for r in results.values()], ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=comp, x="algorithm", y="mean_feasible_g", hue="scenario", ax=ax)
    for name in sorted(comp["scenario"].unique()):
        opt = results[name]["exact"]["global_optimum"]["w_total_g"]
        ax.axhline(opt, ls=":", lw=1, color="k", alpha=0.35)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.set_ylabel("Mean feasible AUW (g)")
    ax.set_title("Algorithm stability vs exhaustive optimum (multiple seeds)")
    fig.tight_layout(); fig.savefig(FIGURES / "algorithm_comparison.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    comp_df = pd.concat([component_table(r) for r in results.values()], ignore_index=True)
    pivot = comp_df.pivot_table(index="scenario", columns="component", values="mass_g",
                                aggfunc="sum").fillna(0)
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Mass (g)")
    ax.set_title("Optimised component-wise mass breakdown")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "optimized_component_breakdown.png", dpi=200); plt.close(fig)

    rows = [{"scenario": n, "baseline_g": r["exact"]["baseline_median_feasible"]["w_total_g"],
             "optimized_g": r["exact"]["global_optimum"]["w_total_g"]}
            for n, r in results.items()]
    cmp = pd.DataFrame(rows).melt(id_vars="scenario", var_name="design", value_name="mass_g")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=cmp, x="scenario", y="mass_g", hue="design", ax=ax)
    ax.set_ylabel("AUW (g)")
    ax.set_title("Typical feasible build (median) vs constrained optimum")
    fig.tight_layout(); fig.savefig(FIGURES / "original_vs_optimized.png", dpi=200); plt.close(fig)

    metrics = sens[sens["parameter"] != "baseline"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, pname in zip(axes.ravel(),
                         ["min_twr", "endurance_min", "payload_kg", "airframe_scale"]):
        d = metrics[metrics["parameter"] == pname]
        ax.plot(d["value"], d["optimum_g"], "o-", color="#3b6ea5")
        ax.set_xlabel(pname); ax.set_ylabel("Optimum AUW (g)")
        ax.set_title(f"Sensitivity: {pname}")
        ax.grid(alpha=0.4)
    fig.suptitle("Design-requirement sensitivity (5-inch scenario)")
    fig.tight_layout(); fig.savefig(FIGURES / "sensitivity_analysis.png", dpi=200); plt.close(fig)

    d = results["5inch"]["designs"]
    f = d[d["feasible"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(d["endurance_min"], d["w_total_g"], s=8, alpha=0.25, label="infeasible")
    axes[0].scatter(f["endurance_min"], f["w_total_g"], s=10, alpha=0.5, color="tab:green",
                    label="feasible")
    axes[0].set_xlim(0, 60); axes[0].set_ylim(0, d["w_total_g"].quantile(0.98))
    axes[0].set_xlabel("Hover endurance (min)"); axes[0].set_ylabel("AUW (g)")
    axes[0].set_title("Feasible design space")
    axes[0].legend(fontsize=9)
    axes[1].scatter(d["twr"], d["w_total_g"], s=8, alpha=0.25, label="infeasible")
    axes[1].scatter(f["twr"], f["w_total_g"], s=10, alpha=0.5, color="tab:green",
                    label="feasible")
    axes[1].axvline(MIN_TWR, color="r", ls="--", lw=1)
    axes[1].set_xlim(0, 12); axes[1].set_ylim(0, d["w_total_g"].quantile(0.98))
    axes[1].set_xlabel("Static thrust-to-weight ratio"); axes[1].set_ylabel("AUW (g)")
    axes[1].set_title("Thrust margin vs weight")
    axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES / "constraint_feasibility.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    out = run_all()
    print(out["comparison"].to_string(index=False))
