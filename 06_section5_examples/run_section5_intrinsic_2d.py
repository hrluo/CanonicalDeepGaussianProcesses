#!/usr/bin/env python3
"""Finite-sample illustration of the intrinsic-structure example in Section 5.

The truth depends on t=(x1+x2)/2.  The first two hidden layers are identity,
the third hidden layer maps (x1,x2) to (t,0), and the final layer is scalar.
This is a fixed-two-dimensional representation of the one-dimensional active
structure used in the conditional Section-5 statement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

import penalized_dgp_recovery_2d as dgp
from section5_scenarios import intrinsic_2d_scenario
from section5_utils import (
    aggregate_long,
    custom_active_layer_error_2d,
    eta_scaled,
    parse_int_grid,
    parse_text_grid,
    single_gp_2d,
    write_csv,
    write_json,
)


def config_for(meta: Dict, n_train: int, fast: bool) -> Dict:
    cfg = dgp.default_config_2d()
    cfg.update({
        "domain": ((-1.0, 1.0), (-1.0, 1.0)),
        "penalty_domain": ((-2.0, 2.0), (-2.0, 2.0)),
        "diagnostic_domain": ((-2.0, 2.0), (-2.0, 2.0)),
        "n_train": int(n_train),
        "test_side": 25,
        "noise_sd": float(meta["noise_sd"]),
        "rho": float(meta["rho"]),
        "eta": eta_scaled(meta["eta_base"], meta["eta_exponents"], n_train, n_ref=70.0),
        "kernel_kind": str(meta["kernel_kind"]),
        "init_lengthscale": float(meta["init_lengthscale"]),
        "hard_identity_enabled": False,
        "active_tol": 0.10,
        "penalty_grid_side": 19,
        "recovery_grid_side": 31,
        "predictive_draws_per_state": 1,
    })
    if fast:
        cfg.update({"n_iter": 70, "burn": 22, "thin": 4, "test_side": 17, "recovery_grid_side": 21})
    else:
        cfg.update({"n_iter": 260, "burn": 90, "thin": 7})
    return cfg


def dgp_row(meta: Dict, n_train: int, seed: int, init_mode: str, model: str, result: Dict, eta: List[float]) -> Dict:
    rec = result["recovery"]
    projection_error = custom_active_layer_error_2d(dgp, result, meta["layer_fns"], 2)
    Xg = np.asarray(rec["grid_points"], float)
    est3 = np.asarray(rec["est_layer_maps"][2], float)
    target3 = np.asarray(meta["layer_fns"][2](Xg), float)
    coord1_error = float(np.sqrt(np.mean((est3[:, 0] - target3[:, 0]) ** 2)))
    leakage = float(np.sqrt(np.mean(est3[:, 1] ** 2)))
    return {
        "scenario": meta["name"], "n_train": int(n_train), "seed": int(seed), "init_mode": init_mode,
        "model": model, "kernel_kind": meta["kernel_kind"], "rho": meta["rho"], "eta": json.dumps(eta),
        "rmse": float(result["rmse"]),
        "redundant_identity_error": float(np.mean(rec["layer_l2_identity"][[0, 1]])),
        "redundant_partial_error": float(np.mean(rec["partial_l2_truth"][[0, 1]])),
        "active_projection_error": projection_error,
        "active_projection_coord1_error": coord1_error,
        "active_projection_leakage": leakage,
        "active_layer_identity_distance": float(rec["layer_l2_identity"][2]),
        "effective_depth": float(rec["estimated_effective_depth"]),
    }


def single_row(meta: Dict, n_train: int, seed: int, sg: Dict) -> Dict:
    return {
        "scenario": meta["name"], "n_train": int(n_train), "seed": int(seed), "init_mode": "",
        "model": "single_gp", "kernel_kind": meta["kernel_kind"], "rho": meta["rho"], "eta": "[]",
        "rmse": float(sg["rmse"]),
        "redundant_identity_error": float("nan"), "redundant_partial_error": float("nan"),
        "active_projection_error": float("nan"), "active_projection_coord1_error": float("nan"),
        "active_projection_leakage": float("nan"), "active_layer_identity_distance": float("nan"),
        "effective_depth": float("nan"),
    }


def plot_rate(meta: Dict, agg: List[Dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for model in ["single_gp", "unaugmented", "augmented"]:
        rows = sorted([r for r in agg if r["model"] == model], key=lambda r: int(r["n_train"]))
        n = np.asarray([r["n_train"] for r in rows], float)
        y = np.asarray([r["rmse_mean"] for r in rows], float)
        se = np.asarray([r["rmse_se"] for r in rows], float)
        axes[0].errorbar(n, y, yerr=se, marker="o", label=model.replace("_", " "))
    aug = sorted([r for r in agg if r["model"] == "augmented"], key=lambda r: int(r["n_train"]))
    if aug:
        n = np.asarray([r["n_train"] for r in aug], float)
        y = np.asarray([r["rmse_mean"] for r in aug], float)
        rate = float(meta["rate_exponent"])
        axes[0].plot(n, y[0] * (n / n[0]) ** (-rate), linestyle="--", label=f"reference n^(-{rate:.3f})")

    for model in ["unaugmented", "augmented"]:
        rows = sorted([r for r in agg if r["model"] == model], key=lambda r: int(r["n_train"]))
        n = np.asarray([r["n_train"] for r in rows], float)
        axes[1].errorbar(n, [r["redundant_identity_error_mean"] for r in rows],
                         yerr=[r["redundant_identity_error_se"] for r in rows], marker="o", label=model)
        axes[2].errorbar(n, [r["active_projection_error_mean"] for r in rows],
                         yerr=[r["active_projection_error_se"] for r in rows], marker="o", label=model)

    titles = ["Function-level prediction", "Redundant layers", "Active rank-one representation"]
    ylabels = ["RMSE", "identity error", "layer-3 projection error"]
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel(yl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(meta["title"])
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.84, wspace=0.27)
    fig.savefig(outdir / "section5_intrinsic_2d_rates.png", dpi=260)
    fig.savefig(outdir / "section5_intrinsic_2d_rates.pdf")
    plt.close(fig)


def plot_surfaces(meta: Dict, data: Dict, sg: Dict, regular: Dict, augmented: Dict, outdir: Path) -> None:
    g1, g2 = data["grid_x1"], data["grid_x2"]
    surfaces = [
        ("truth", np.asarray(data["y_true_test"]).reshape(g1.shape)),
        ("single GP", np.asarray(sg["pred_mean"]).reshape(g1.shape)),
        ("unaugmented DGP", np.asarray(regular["pred_mean"]).reshape(g1.shape)),
        ("identity-augmented DGP", np.asarray(augmented["pred_mean"]).reshape(g1.shape)),
    ]
    vals = np.concatenate([s.ravel() for _, s in surfaces])
    vmin, vmax = float(vals.min()), float(vals.max())
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 8.5))
    for ax, (title, surf) in zip(axes.ravel(), surfaces):
        im = ax.imshow(surf, origin="lower", extent=[-1, 1, -1, 1], aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    fig.suptitle("Section 5 intrinsic-structure example — final regression surface")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.08, top=0.91, hspace=0.23, wspace=0.18)
    fig.savefig(outdir / "section5_intrinsic_2d_surfaces.png", dpi=260)
    fig.savefig(outdir / "section5_intrinsic_2d_surfaces.pdf")
    plt.close(fig)


def plot_projection(meta: Dict, regular: Dict, augmented: Dict, outdir: Path) -> None:
    g1, g2, Xg = dgp.make_grid_2d(((-2.0, 2.0), (-2.0, 2.0)), 31)
    target = np.asarray(meta["layer_fns"][2](Xg), float)
    reg, _, _ = dgp.layer_mean_map_2d(regular["samples"], 2, Xg)
    aug, _, _ = dgp.layer_mean_map_2d(augmented["samples"], 2, Xg)
    arrays = [("target", target), ("unaugmented", reg), ("augmented", aug)]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.7))
    for col, (title, arr) in enumerate(arrays):
        for coord in [0, 1]:
            surf = arr[:, coord].reshape(g1.shape)
            ax = axes[coord, col]
            im = ax.imshow(surf, origin="lower", extent=[-2, 2, -2, 2], aspect="equal")
            ax.set_title(f"{title}: layer 3 coord {coord+1}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Learned active projection f3(x): target image is ( (x1+x2)/2, 0 )")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.90, hspace=0.25, wspace=0.26)
    fig.savefig(outdir / "section5_intrinsic_2d_projection.png", dpi=260)
    fig.savefig(outdir / "section5_intrinsic_2d_projection.pdf")
    plt.close(fig)


def run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_grid = parse_int_grid(args.n_grid)
    seeds = parse_int_grid(args.seeds)
    init_modes = parse_text_grid(args.init_modes)
    if args.fast:
        n_grid = n_grid[: min(2, len(n_grid))]
        seeds = seeds[:1]
        init_modes = init_modes[:1]

    rows: List[Dict] = []
    detail = None
    total = len(n_grid) * (len(seeds) + 2 * len(seeds) * len(init_modes))
    counter = 0
    for n_train in n_grid:
        meta = intrinsic_2d_scenario(n_train)
        cfg = config_for(meta, n_train, args.fast)
        for seed in seeds:
            rng = np.random.default_rng(500_000 + 1009 * seed + 41 * n_train)
            data = dgp.make_dataset_2d(meta["layer_fns"], cfg["domain"], n_train, cfg["test_side"], cfg["noise_sd"], rng)
            counter += 1
            print(f"[{counter}/{total}] n={n_train} seed={seed} single GP", flush=True)
            sg = single_gp_2d(dgp, data["X_train"], data["y_train"], data["X_test"], data["y_true_test"], cfg["noise_sd"], cfg["kernel_kind"])
            rows.append(single_row(meta, n_train, seed, sg))

            for init_idx, init_mode in enumerate(init_modes):
                chain_seed = 600_000 + 10007 * seed + 137 * n_train + 17 * init_idx
                counter += 2
                print(f"[{counter-1}/{total}] n={n_train} seed={seed} init={init_mode} unaugmented", flush=True)
                rs = dgp.run_chain_2d(cfg, data, chain_seed, "baseline", init_mode)
                regular = dgp.summarize_run_2d(rs, data, meta["layer_fns"], cfg)
                print(f"[{counter}/{total}] n={n_train} seed={seed} init={init_mode} augmented", flush=True)
                ps = dgp.run_chain_2d(cfg, data, chain_seed, "penalized", init_mode)
                augmented = dgp.summarize_run_2d(ps, data, meta["layer_fns"], cfg)
                rows.append(dgp_row(meta, n_train, seed, init_mode, "unaugmented", regular, cfg["eta"]))
                rows.append(dgp_row(meta, n_train, seed, init_mode, "augmented", augmented, cfg["eta"]))
                if n_train == max(n_grid) and seed == seeds[0] and init_mode == init_modes[0]:
                    detail = (meta, data, sg, regular, augmented)

    metrics = [
        "rmse", "redundant_identity_error", "redundant_partial_error", "active_projection_error",
        "active_projection_coord1_error", "active_projection_leakage", "active_layer_identity_distance",
        "effective_depth",
    ]
    agg = aggregate_long(rows, ["scenario", "n_train", "model"], metrics)
    write_csv(rows, outdir / "section5_intrinsic_2d_runs.csv")
    write_csv(agg, outdir / "section5_intrinsic_2d_aggregate.csv")
    write_json({
        "n_grid": n_grid, "seeds": seeds, "init_modes": init_modes, "fast": bool(args.fast),
        "interpretation": "The active rank-one layer is a fixed-2D realization of the conditional intrinsic representation; reference slope is a visual guide only.",
    }, outdir / "section5_intrinsic_2d_config.json")
    meta = intrinsic_2d_scenario(max(n_grid))
    plot_rate(meta, agg, outdir)
    if detail is not None:
        plot_surfaces(*detail, outdir)
        plot_projection(detail[0], detail[3], detail[4], outdir)
    print(f"Done. Results written to {outdir.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-grid", default="40,70,100")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--init-modes", default="identity,smooth_random")
    p.add_argument("--outdir", default="section5_intrinsic_2d_results")
    p.add_argument("--fast", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
