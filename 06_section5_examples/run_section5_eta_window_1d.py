#!/usr/bin/env python3
"""Penalty-strength / 'sweet point' study for Section 5.

For a fixed Section-5 truth and sample size, this driver multiplies the
scenario's n-dependent penalty schedule by lambda.  It visualizes the tradeoff
between function RMSE and redundant-layer identity stabilization.  This is a
finite-sample diagnostic for the tuning-window discussion in Theorems 5.1--5.2;
it does not estimate the unknown constants in the asymptotic inequalities.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import json as json

import numpy as np
import matplotlib.pyplot as plt

import penalized_dgp_recovery_1d as dgp
from section5_scenarios import ONE_D_SCENARIOS, scenario_1d
from section5_utils import (
    aggregate_long,
    custom_active_layer_error_1d,
    eta_scaled,
    make_random_dataset_1d,
    parse_float_grid,
    parse_int_grid,
    parse_text_grid,
    write_csv,
    write_json,
)


def base_config(meta: Dict, n_train: int, fast: bool) -> Dict:
    cfg = dgp.default_config_1d()
    cfg.update({
        "domain": (-1.0, 1.0),
        "penalty_domain": (-2.0, 2.0),
        "diagnostic_domain": (-2.0, 2.0),
        "n_train": int(n_train),
        "n_test": 301,
        "noise_sd": float(meta["noise_sd"]),
        "rho": float(meta["rho"]),
        "kernel_kind": str(meta["kernel_kind"]),
        "init_lengthscale": float(meta["init_lengthscale"]),
        "hard_identity_enabled": False,
        "active_tol": 0.08,
        "penalty_grid_size": 121,
        "recovery_grid_size": 321,
    })
    if fast:
        cfg.update({"n_iter": 80, "burn": 25, "thin": 5})
    else:
        cfg.update({"n_iter": 300, "burn": 100, "thin": 8})
    return cfg


def metrics(meta: Dict, result: Dict) -> Dict:
    rec = result["recovery"]
    redundant = list(meta["redundant_layers"])
    active = list(meta["active_nonfinal_layers"])
    out = {
        "rmse": float(result["rmse"]),
        "redundant_identity_error": float(np.mean(rec["layer_l2_identity"][redundant])),
        "redundant_partial_error": float(np.mean(rec["partial_l2_truth"][redundant])),
        "effective_depth": float(rec["estimated_effective_depth"]),
    }
    if active:
        out["active_layer_truth_error"] = float(np.mean([
            custom_active_layer_error_1d(dgp, result, meta["layer_fns"], j) for j in active
        ]))
        out["active_layer_identity_distance"] = float(np.mean(rec["layer_l2_identity"][active]))
    else:
        out["active_layer_truth_error"] = float("nan")
        out["active_layer_identity_distance"] = float("nan")
    return out


def plot_window(agg: List[Dict], scenario: str, outdir: Path) -> None:
    sub = sorted([r for r in agg if r["scenario"] == scenario], key=lambda r: float(r["lambda"]))
    lam = np.asarray([float(r["lambda"]) for r in sub])
    rmse = np.asarray([r["rmse_mean"] for r in sub], float)
    rmse_se = np.asarray([r["rmse_se"] for r in sub], float)
    ide = np.asarray([r["redundant_identity_error_mean"] for r in sub], float)
    ide_se = np.asarray([r["redundant_identity_error_se"] for r in sub], float)
    par = np.asarray([r["redundant_partial_error_mean"] for r in sub], float)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].errorbar(lam, rmse, yerr=rmse_se, marker="o")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Function-level fit")
    axes[1].errorbar(lam, ide, yerr=ide_se, marker="o")
    axes[1].set_ylabel("redundant identity error")
    axes[1].set_title("Layer anchoring")
    axes[2].plot(ide, rmse, marker="o")
    for x, y, l in zip(ide, rmse, lam):
        axes[2].annotate(f"{l:g}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[2].set_xlabel("redundant identity error")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("Prediction–parsimony tradeoff")
    for ax in axes[:2]:
        ax.set_xlabel("penalty multiplier lambda")
        ax.grid(alpha=0.25)
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"Section 5 penalty-window diagnostic: {scenario.replace('_', ' ')}")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.83, wspace=0.29)
    fig.savefig(outdir / f"section5_eta_window_{scenario}.png", dpi=260)
    fig.savefig(outdir / f"section5_eta_window_{scenario}.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(lam, par, marker="o")
    ax.set_xlabel("penalty multiplier lambda")
    ax.set_ylabel("redundant partial-composition error")
    ax.set_title("Partial-composition stabilization")
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(outdir / f"section5_eta_partial_{scenario}.png", dpi=260)
    fig.savefig(outdir / f"section5_eta_partial_{scenario}.pdf")
    plt.close(fig)


def run(args) -> None:
    if args.scenario not in ONE_D_SCENARIOS:
        raise ValueError(f"choose --scenario from {ONE_D_SCENARIOS}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lambdas = parse_float_grid(args.lambda_grid)
    seeds = parse_int_grid(args.seeds)
    init_modes = parse_text_grid(args.init_modes)
    if args.fast:
        lambdas = lambdas[: min(4, len(lambdas))]
        seeds = seeds[:1]
        init_modes = init_modes[:1]

    n_train = int(args.n_train)
    meta = scenario_1d(args.scenario, n_train)
    cfg0 = base_config(meta, n_train, args.fast)
    eta_nominal = eta_scaled(meta["eta_base"], meta["eta_exponents"], n_train, n_ref=30.0)
    rows: List[Dict] = []

    total = len(seeds) * len(init_modes) * (1 + sum(1 for l in lambdas if l > 0))
    counter = 0
    for seed in seeds:
        data_seed = 300_000 + 1009 * seed + 37 * n_train
        data = make_random_dataset_1d(
            dgp, meta["layer_fns"], n_train, cfg0["n_test"], cfg0["noise_sd"], data_seed, cfg0["domain"]
        )
        for init_idx, init_mode in enumerate(init_modes):
            chain_seed = 400_000 + 10007 * seed + 131 * n_train + 17 * init_idx
            counter += 1
            print(f"[{counter}/{total}] lambda=0 seed={seed} init={init_mode}", flush=True)
            regular_samples = dgp.run_chain_1d(cfg0, data, chain_seed, "baseline", init_mode)
            regular = dgp.summarize_run_1d(regular_samples, data, meta["layer_fns"], cfg0)
            rec = {
                "scenario": meta["name"], "n_train": n_train, "seed": seed, "init_mode": init_mode,
                "lambda": 0.0, "eta": "[0,0,0,0]",
            }
            rec.update(metrics(meta, regular))
            rows.append(rec)

            for lam in lambdas:
                if lam <= 0:
                    continue
                cfg = dict(cfg0)
                cfg["eta"] = [float(lam) * float(v) for v in eta_nominal]
                counter += 1
                print(f"[{counter}/{total}] lambda={lam:g} seed={seed} init={init_mode}", flush=True)
                samples = dgp.run_chain_1d(cfg, data, chain_seed, "penalized", init_mode)
                result = dgp.summarize_run_1d(samples, data, meta["layer_fns"], cfg)
                rec = {
                    "scenario": meta["name"], "n_train": n_train, "seed": seed, "init_mode": init_mode,
                    "lambda": float(lam), "eta": json.dumps(cfg["eta"]),
                }
                rec.update(metrics(meta, result))
                rows.append(rec)

    metric_names = [
        "rmse", "redundant_identity_error", "redundant_partial_error", "effective_depth",
        "active_layer_truth_error", "active_layer_identity_distance",
    ]
    agg = aggregate_long(rows, ["scenario", "n_train", "lambda"], metric_names)
    write_csv(rows, outdir / "section5_eta_window_runs.csv")
    write_csv(agg, outdir / "section5_eta_window_aggregate.csv")
    write_json({
        "scenario": meta["name"], "n_train": n_train, "lambda_grid": lambdas,
        "nominal_eta_at_lambda_1": eta_nominal, "seeds": seeds, "init_modes": init_modes,
        "fast": bool(args.fast),
        "interpretation": "lambda is a finite-sample multiplier; the asymptotic theorem does not determine its numerical constant.",
    }, outdir / "section5_eta_window_config.json")
    plot_window(agg, meta["name"], outdir)
    print(f"Done. Results written to {outdir.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="holder_correct", choices=ONE_D_SCENARIOS)
    p.add_argument("--n-train", type=int, default=45)
    p.add_argument("--lambda-grid", default="0,0.2,0.4,0.7,1.0,1.5,2.5")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--init-modes", default="identity,smooth_random")
    p.add_argument("--outdir", default="section5_eta_window_results")
    p.add_argument("--fast", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
