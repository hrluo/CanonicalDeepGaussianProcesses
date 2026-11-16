#!/usr/bin/env python3
"""Finite-sample illustrations for the one-dimensional examples in Section 5.

This driver studies four mechanisms from Section 5:
  * correctly specified Hölder smooth truth;
  * oversmooth prior with an active penultimate rescaling layer;
  * a non-differentiable cusp;
  * a discontinuous step-function stress test.

It compares an empirical-Bayes single GP, an unaugmented four-layer DGP, and
the identity-augmented four-layer DGP.  The dashed log-log lines in the output
are *reference slopes* from the Section-5 statements; the script is not a
numerical proof of those asymptotic rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

import penalized_dgp_recovery_1d as dgp
from section5_scenarios import ONE_D_SCENARIOS, scenario_1d
from section5_utils import (
    aggregate_long,
    custom_active_layer_error_1d,
    eta_scaled,
    make_random_dataset_1d,
    parse_int_grid,
    parse_text_grid,
    single_gp_1d,
    write_csv,
    write_json,
)


def _config_for_scenario(meta: Dict, n_train: int, fast: bool) -> Dict:
    cfg = dgp.default_config_1d()
    cfg.update({
        "domain": (-1.0, 1.0),
        "penalty_domain": (-2.0, 2.0),
        "diagnostic_domain": (-2.0, 2.0),
        "n_train": int(n_train),
        "n_test": 301,
        "noise_sd": float(meta["noise_sd"]),
        "rho": float(meta["rho"]),
        "eta": eta_scaled(meta["eta_base"], meta["eta_exponents"], n_train, n_ref=30.0),
        "kernel_kind": str(meta["kernel_kind"]),
        "init_lengthscale": float(meta["init_lengthscale"]),
        "hard_identity_enabled": False,
        "active_tol": 0.08,
        "penalty_grid_size": 121,
        "recovery_grid_size": 321,
        "predictive_draws_per_state": 1,
    })
    if fast:
        cfg.update({"n_iter": 80, "burn": 25, "thin": 5})
    else:
        cfg.update({"n_iter": 300, "burn": 100, "thin": 8})
    return cfg


def _result_row(meta: Dict, n_train: int, seed: int, init_mode: str, model: str,
                result: Dict, eta: List[float]) -> Dict:
    rec = result["recovery"]
    redundant = list(meta["redundant_layers"])
    active = list(meta["active_nonfinal_layers"])
    row = {
        "scenario": meta["name"],
        "n_train": int(n_train),
        "seed": int(seed),
        "init_mode": init_mode,
        "model": model,
        "kernel_kind": meta["kernel_kind"],
        "rho": float(meta["rho"]),
        "eta": json.dumps([float(x) for x in eta]),
        "rmse": float(result["rmse"]),
        "redundant_identity_error": float(np.mean(rec["layer_l2_identity"][redundant])),
        "redundant_partial_error": float(np.mean(rec["partial_l2_truth"][redundant])),
        "effective_depth": float(rec["estimated_effective_depth"]),
    }
    if active:
        vals = [custom_active_layer_error_1d(dgp, result, meta["layer_fns"], j) for j in active]
        row["active_layer_truth_error"] = float(np.mean(vals))
        row["active_layer_identity_distance"] = float(np.mean(rec["layer_l2_identity"][active]))
    else:
        row["active_layer_truth_error"] = float("nan")
        row["active_layer_identity_distance"] = float("nan")
    return row


def _single_gp_row(meta: Dict, n_train: int, seed: int, fit: Dict) -> Dict:
    return {
        "scenario": meta["name"],
        "n_train": int(n_train),
        "seed": int(seed),
        "init_mode": "",
        "model": "single_gp",
        "kernel_kind": meta["kernel_kind"],
        "rho": float(meta["rho"]),
        "eta": "[]",
        "rmse": float(fit["rmse"]),
        "redundant_identity_error": float("nan"),
        "redundant_partial_error": float("nan"),
        "effective_depth": float("nan"),
        "active_layer_truth_error": float("nan"),
        "active_layer_identity_distance": float("nan"),
    }


def _plot_rates(meta: Dict, agg: List[Dict], outdir: Path) -> None:
    sub = [r for r in agg if r["scenario"] == meta["name"]]
    models = ["single_gp", "unaugmented", "augmented"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    for model in models:
        rows = sorted([r for r in sub if r["model"] == model], key=lambda r: int(r["n_train"]))
        if not rows:
            continue
        n = np.asarray([r["n_train"] for r in rows], float)
        rmse = np.asarray([r["rmse_mean"] for r in rows], float)
        se = np.asarray([r["rmse_se"] for r in rows], float)
        axes[0].errorbar(n, rmse, yerr=se, marker="o", label=model.replace("_", " "))

    aug_rows = sorted([r for r in sub if r["model"] == "augmented"], key=lambda r: int(r["n_train"]))
    if aug_rows:
        n = np.asarray([r["n_train"] for r in aug_rows], float)
        y = np.asarray([r["rmse_mean"] for r in aug_rows], float)
        rate = float(meta["rate_exponent"])
        ref = y[0] * (n / n[0]) ** (-rate)
        axes[0].plot(n, ref, linestyle="--", label=f"reference n^(-{rate:.3f})")

    for model in ["unaugmented", "augmented"]:
        rows = sorted([r for r in sub if r["model"] == model], key=lambda r: int(r["n_train"]))
        n = np.asarray([r["n_train"] for r in rows], float)
        val = np.asarray([r["redundant_identity_error_mean"] for r in rows], float)
        se = np.asarray([r["redundant_identity_error_se"] for r in rows], float)
        axes[1].errorbar(n, val, yerr=se, marker="o", label=model)

    if aug_rows:
        k = float(meta["eta_exponents"][meta["redundant_layers"][0]])
        r = float(meta["rate_exponent"])
        radius_exp = 1.0 - 2.0 * r - k
        vals = np.asarray([r0["redundant_identity_error_mean"] for r0 in aug_rows], float)
        n = np.asarray([r0["n_train"] for r0 in aug_rows], float)
        ref = vals[0] * (n / n[0]) ** radius_exp
        axes[1].plot(n, ref, linestyle="--", label=f"radius-order slope {radius_exp:.3f}")

    for model in ["unaugmented", "augmented"]:
        rows = sorted([r for r in sub if r["model"] == model], key=lambda r: int(r["n_train"]))
        n = np.asarray([r["n_train"] for r in rows], float)
        val = np.asarray([r["redundant_partial_error_mean"] for r in rows], float)
        se = np.asarray([r["redundant_partial_error_se"] for r in rows], float)
        axes[2].errorbar(n, val, yerr=se, marker="o", label=model)

    axes[0].set_title("Function-level prediction")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("Redundant-layer identity error")
    axes[1].set_ylabel("mean L2 distance to identity")
    axes[2].set_title("Redundant partial-composition error")
    axes[2].set_ylabel("mean L2 error")
    for ax in axes:
        ax.set_xlabel("n")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(meta["title"])
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.84, wspace=0.27)
    fig.savefig(outdir / f"section5_rate_{meta['name']}.png", dpi=260)
    fig.savefig(outdir / f"section5_rate_{meta['name']}.pdf")
    plt.close(fig)


def _plot_prediction_detail(meta: Dict, data: Dict, single: Dict, regular: Dict, augmented: Dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True)
    models = [
        ("Single GP", single["pred_mean"], single["pred_var"]),
        ("Unaugmented DGP", regular["pred_mean"], regular["pred_var"]),
        ("Identity-augmented DGP", augmented["pred_mean"], augmented["pred_var"]),
    ]
    for ax, (title, mean, var) in zip(axes, models):
        sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
        ax.plot(data["x_test"], data["y_true_test"], linewidth=2, label="truth")
        ax.plot(data["x_test"], mean, linewidth=1.7, label="posterior mean")
        ax.fill_between(data["x_test"], mean - 1.645 * sd, mean + 1.645 * sd, alpha=0.2, label="90% pointwise band")
        ax.scatter(data["x_train"], data["y_train"], s=13, alpha=0.35, label="training data")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("f(x)")
    axes[0].legend(fontsize=7)
    fig.suptitle(meta["title"] + " — largest n")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.83, wspace=0.14)
    fig.savefig(outdir / f"section5_prediction_{meta['name']}.png", dpi=260)
    fig.savefig(outdir / f"section5_prediction_{meta['name']}.pdf")
    plt.close(fig)


def _plot_layers_detail(meta: Dict, regular: Dict, augmented: Dict, outdir: Path) -> None:
    grid = np.linspace(-2.0, 2.0, 321)
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0), sharex=True)
    for j in range(3):
        truth = np.asarray(meta["layer_fns"][j](grid), float)
        for col, (title, res) in enumerate([("Unaugmented DGP", regular), ("Identity-augmented DGP", augmented)]):
            mean, _, maps = dgp.layer_mean_map_1d(res["samples"], j, grid)
            lo, hi = np.quantile(maps, [0.05, 0.95], axis=0)
            ax = axes[j, col]
            ax.fill_between(grid, lo, hi, alpha=0.18, label="90% retained-state envelope")
            ax.plot(grid, mean, linewidth=1.8, label="posterior mean")
            ax.plot(grid, grid, linestyle="--", linewidth=1.0, label="identity")
            if not np.allclose(truth, grid):
                ax.plot(grid, truth, linestyle=":", linewidth=1.5, label="target layer")
            ax.set_title(f"{title}: layer {j+1}")
            ax.set_ylabel(f"f{j+1}(z)")
            ax.grid(alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel("layer input z")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(meta["title"] + " — hidden-layer stabilization")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.92, hspace=0.27, wspace=0.20)
    fig.savefig(outdir / f"section5_layers_{meta['name']}.png", dpi=260)
    fig.savefig(outdir / f"section5_layers_{meta['name']}.pdf")
    plt.close(fig)


def run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scenarios = ONE_D_SCENARIOS if args.scenario == "all" else parse_text_grid(args.scenario)
    for name in scenarios:
        if name not in ONE_D_SCENARIOS:
            raise ValueError(f"Unknown scenario {name!r}; choose from {ONE_D_SCENARIOS}")
    n_grid = parse_int_grid(args.n_grid)
    seeds = parse_int_grid(args.seeds)
    init_modes = parse_text_grid(args.init_modes)
    if args.fast:
        n_grid = n_grid[: min(2, len(n_grid))]
        seeds = seeds[:1]
        init_modes = init_modes[:1]

    rows: List[Dict] = []
    detail = {}
    total = sum(len(n_grid) * (len(seeds) + len(seeds) * len(init_modes) * 2) for _ in scenarios)
    counter = 0

    for scenario_name in scenarios:
        for n_train in n_grid:
            meta = scenario_1d(scenario_name, n_train)
            cfg = _config_for_scenario(meta, n_train, args.fast)
            for seed in seeds:
                data_seed = 100_000 + 1009 * seed + 37 * n_train + 7919 * ONE_D_SCENARIOS.index(scenario_name)
                data = make_random_dataset_1d(
                    dgp, meta["layer_fns"], n_train, cfg["n_test"], cfg["noise_sd"], data_seed, cfg["domain"]
                )
                counter += 1
                print(f"[{counter}/{total}] {scenario_name} n={n_train} seed={seed} single GP", flush=True)
                sg = single_gp_1d(
                    dgp, data["x_train"], data["y_train"], data["x_test"], data["y_true_test"],
                    cfg["noise_sd"], cfg["kernel_kind"]
                )
                rows.append(_single_gp_row(meta, n_train, seed, sg))

                for init_idx, init_mode in enumerate(init_modes):
                    chain_seed = 200_000 + 10007 * seed + 131 * n_train + 17 * init_idx + 5003 * ONE_D_SCENARIOS.index(scenario_name)
                    counter += 2
                    print(f"[{counter-1}/{total}] {scenario_name} n={n_train} seed={seed} init={init_mode} unaugmented", flush=True)
                    regular_samples = dgp.run_chain_1d(cfg, data, chain_seed, "baseline", init_mode)
                    regular = dgp.summarize_run_1d(regular_samples, data, meta["layer_fns"], cfg)
                    print(f"[{counter}/{total}] {scenario_name} n={n_train} seed={seed} init={init_mode} augmented", flush=True)
                    augmented_samples = dgp.run_chain_1d(cfg, data, chain_seed, "penalized", init_mode)
                    augmented = dgp.summarize_run_1d(augmented_samples, data, meta["layer_fns"], cfg)
                    rows.append(_result_row(meta, n_train, seed, init_mode, "unaugmented", regular, cfg["eta"]))
                    rows.append(_result_row(meta, n_train, seed, init_mode, "augmented", augmented, cfg["eta"]))

                    if n_train == max(n_grid) and seed == seeds[0] and init_mode == init_modes[0]:
                        detail[scenario_name] = (meta, data, sg, regular, augmented)

    metrics = [
        "rmse", "redundant_identity_error", "redundant_partial_error", "effective_depth",
        "active_layer_truth_error", "active_layer_identity_distance",
    ]
    agg = aggregate_long(rows, ["scenario", "n_train", "model"], metrics)
    write_csv(rows, outdir / "section5_1d_runs.csv")
    write_csv(agg, outdir / "section5_1d_aggregate.csv")
    write_json({
        "scenarios": scenarios,
        "n_grid": n_grid,
        "seeds": seeds,
        "init_modes": init_modes,
        "fast": bool(args.fast),
        "interpretation": "Reference slopes are visual order guides from Section 5, not fitted or claimed asymptotic exponents.",
    }, outdir / "section5_1d_config.json")

    for scenario_name in scenarios:
        meta = scenario_1d(scenario_name, max(n_grid))
        _plot_rates(meta, agg, outdir)
        if scenario_name in detail:
            meta_d, data, sg, regular, augmented = detail[scenario_name]
            _plot_prediction_detail(meta_d, data, sg, regular, augmented, outdir)
            _plot_layers_detail(meta_d, regular, augmented, outdir)

    print(f"Done. Results written to {outdir.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="all", help=f"all or comma-separated names from {ONE_D_SCENARIOS}")
    p.add_argument("--n-grid", default="20,30,45,70")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--init-modes", default="identity,smooth_random")
    p.add_argument("--outdir", default="section5_1d_results")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
