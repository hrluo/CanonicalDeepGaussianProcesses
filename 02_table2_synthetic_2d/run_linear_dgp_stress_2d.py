#!/usr/bin/env python3
"""
Contrasting 2D stress-test suite for regular DGP vs linear identity-augmented DGP.

Place this file in the same folder as one of:
    penalized_dgp_recovery_2d(2).py
    penalized_dgp_recovery_2d.py

Then run examples such as:
    python run_linear_dgp_stress_2d_contrasts.py --example shallow_ridges --outdir out_ridges
    python run_linear_dgp_stress_2d_contrasts.py --example shallow_checkerboard --outdir out_checkerboard
    python run_linear_dgp_stress_2d_contrasts.py --example localized_bumps_sparse --outdir out_bumps
    python run_linear_dgp_stress_2d_contrasts.py --example active_radial_warp --outdir out_radial_warp
    python run_linear_dgp_stress_2d_contrasts.py --example all --outdir out_all_2d --fast

Purpose
-------
The main paper-facing contrast is an over-specified 4-layer DGP:
    f1 = id, f2 = id, f3 = id, f4 active.

A regular DGP can invent unnecessary hidden 2D warps. The linear augmented
DGP should shrink redundant non-final layers toward identity and place the
signal in the final layer.

The suite also includes active-warp examples:
    f1 = id, f2 = id, f3 active 2D warp, f4 active final layer.

Those examples check that the linear penalty is not merely forcing all hidden
layers to identity. With smaller eta on layer 3, the augmented DGP should allow
a useful late hidden layer to activate while shrinking earlier redundant layers.

Outputs
-------
Each example writes:
    stress_2d_summary.csv          per seed/init results
    stress_2d_config.json          config for the example
    *.png                          prediction, hidden maps, partial-composition,
                                   and recovery heatmaps

When --example all is used, the root output folder also receives:
    all_stress_2d_runs.csv
    all_stress_2d_aggregate.csv
    example_descriptions.json

Primary diagnostics
-------------------
For the paper, emphasize:
    1. redundant identity error:
       mean_{j=1,2,3} || f_j_hat - id ||_L2 for shallow examples,
       or mean_{j=1,2} for active-warp examples;
    2. partial-composition error;
    3. effective depth;
    4. RMSE, as a secondary diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import matplotlib.pyplot as plt


def _load_same_folder_module() -> Any:
    here = Path(__file__).resolve().parent
    candidates = [ 
        here / "penalized_dgp_recovery_2d.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("penalized_dgp_recovery_2d_local", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not import {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "Could not find penalized_dgp_recovery_2d(2).py or penalized_dgp_recovery_2d.py "
        "in the same folder as this driver."
    )


m = _load_same_folder_module()


def _identity2(X):
    X = m.as_points(X)
    return X.copy()


def _clip_state(H, lo=-1.35, hi=1.35):
    H = m.as_points(H)
    return np.clip(H, lo, hi)


def truth_shallow_ridges():
    """Shallow nonlinear ridge/bump surface; layers 1--3 are redundant."""
    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        return (
            0.75 * np.sin(2.3 * np.pi * u) * np.cos(1.4 * np.pi * v)
            + 0.55 * np.exp(-18.0 * ((u - 0.25) ** 2 + (v + 0.20) ** 2))
            - 0.35 * np.exp(-22.0 * ((u + 0.45) ** 2 + (v - 0.35) ** 2))
        )
    return [_identity2, _identity2, _identity2, f4]


def truth_shallow_checkerboard():
    """Shallow checkerboard/ridge final surface; layers 1--3 are redundant."""
    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        return (
            0.58 * np.sin(2.7 * np.pi * u) * np.sin(2.1 * np.pi * v)
            + 0.25 * np.cos(4.1 * np.pi * u + 0.4 * np.sin(np.pi * v))
            - 0.18 * v
        )
    return [_identity2, _identity2, _identity2, f4]


def truth_localized_bumps_sparse():
    """Shallow localized bumps with sparse/noisy 2D observations."""
    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        bump1 = np.exp(-34.0 * ((u - 0.28) ** 2 + (v + 0.18) ** 2))
        bump2 = np.exp(-45.0 * ((u + 0.40) ** 2 + (v - 0.42) ** 2))
        bump3 = np.exp(-30.0 * ((u + 0.05) ** 2 + (v + 0.55) ** 2))
        return (
            0.80 * bump1
            - 0.55 * bump2
            + 0.35 * bump3
            + 0.22 * np.sin(2.0 * np.pi * u)
            - 0.15 * np.cos(2.2 * np.pi * v)
        )
    return [_identity2, _identity2, _identity2, f4]


def truth_boundary_transition():
    """Shallow oblique boundary transition; layers 1--3 are redundant."""
    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        boundary = 1.0 / (1.0 + np.exp(-11.0 * (u + 0.55 * v + 0.12)))
        return 1.05 * boundary - 0.52 + 0.20 * np.sin(2.0 * np.pi * v) + 0.18 * u * v
    return [_identity2, _identity2, _identity2, f4]


def truth_anisotropic_ridge_sparse():
    """Shallow anisotropic oblique ridge; layers 1--3 are redundant."""
    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        r = 0.82 * u - 0.57 * v
        s = 0.57 * u + 0.82 * v
        return 0.85 * np.sin(2.8 * np.pi * r) + 0.30 * np.exp(-20.0 * (s - 0.20) ** 2) - 0.18 * r
    return [_identity2, _identity2, _identity2, f4]


def truth_active_radial_warp():
    """Layers 1--2 redundant; layer 3 radial warp and final layer active."""
    def f3(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        r2 = u ** 2 + v ** 2
        scale = 0.72 + 0.42 * np.exp(-1.8 * r2)
        return _clip_state(np.column_stack([scale * u, scale * v]))

    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        return 0.75 * np.sin(2.6 * np.pi * u) + 0.35 * np.cos(2.2 * np.pi * v) + 0.20 * u * v
    return [_identity2, _identity2, f3, f4]


def truth_active_shear_warp():
    """Layers 1--2 redundant; layer 3 shear/swirl warp and final layer active."""
    def f3(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        out1 = u + 0.22 * np.sin(np.pi * v)
        out2 = v + 0.18 * np.sin(np.pi * u) * np.cos(np.pi * v)
        return _clip_state(np.column_stack([out1, out2]))

    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        return (
            0.65 * np.sin(2.0 * np.pi * (u + 0.25 * v))
            + 0.32 * np.exp(-16.0 * ((u - 0.15) ** 2 + (v + 0.15) ** 2))
            - 0.22 * v
        )
    return [_identity2, _identity2, f3, f4]


def truth_canceling_affine_trap():
    """Non-identifiable affine shrink/inverse decomposition; use mainly for discussion."""
    A = np.array([[0.82, 0.22], [-0.16, 0.88]])
    b = np.array([0.12, -0.08])
    Ainv = np.linalg.inv(A)

    def f1(X):
        X = m.as_points(X)
        return _clip_state(X @ A.T + b)

    def f2(H):
        H = m.as_points(H)
        return _clip_state((H - b) @ Ainv.T)

    def f4(H):
        H = m.as_points(H)
        u, v = H[:, 0], H[:, 1]
        return 0.70 * np.sin(2.0 * np.pi * u) * np.cos(1.7 * np.pi * v)
    return [f1, f2, _identity2, f4]


ExampleFn = Callable[[], List[Callable[[np.ndarray], np.ndarray]]]

EXAMPLES: Dict[str, Dict[str, Any]] = {
    "shallow_ridges": {
        "truth": truth_shallow_ridges,
        "description": "Shallow nonlinear ridge/bump surface; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.70, "eta": [52.0, 30.0, 14.0, 0.0], "init_lengthscale": 0.34},
    },
    "shallow_checkerboard": {
        "truth": truth_shallow_checkerboard,
        "description": "Shallow checkerboard/ridge final surface; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.68, "eta": [55.0, 32.0, 15.0, 0.0], "init_lengthscale": 0.30},
    },
    "localized_bumps_sparse": {
        "truth": truth_localized_bumps_sparse,
        "description": "Shallow localized bumps with sparse/noisy observations.",
        "redundant_layers": 3,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.65, "eta": [60.0, 36.0, 16.0, 0.0], "init_lengthscale": 0.28},
    },
    "boundary_transition": {
        "truth": truth_boundary_transition,
        "description": "Shallow oblique boundary transition; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.70, "eta": [55.0, 32.0, 15.0, 0.0], "init_lengthscale": 0.32},
    },
    "anisotropic_ridge_sparse": {
        "truth": truth_anisotropic_ridge_sparse,
        "description": "Shallow anisotropic oblique ridge; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.67, "eta": [58.0, 34.0, 16.0, 0.0], "init_lengthscale": 0.30},
    },
    "active_radial_warp": {
        "truth": truth_active_radial_warp,
        "description": "Layers 1--2 redundant; layer 3 radial warp and final layer active.",
        "redundant_layers": 2,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.70, "eta": [55.0, 32.0, 5.0, 0.0], "init_lengthscale": 0.34},
    },
    "active_shear_warp": {
        "truth": truth_active_shear_warp,
        "description": "Layers 1--2 redundant; layer 3 shear/swirl warp and final layer active.",
        "redundant_layers": 2,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.70, "eta": [55.0, 32.0, 5.0, 0.0], "init_lengthscale": 0.34},
    },
    "canceling_affine_trap": {
        "truth": truth_canceling_affine_trap,
        "description": "Non-identifiable affine shrink/inverse decomposition; use for discussion.",
        "redundant_layers": 3,
        "config": {"n_train": 100, "noise_sd": 0.1, "rho": 0.70, "eta": [55.0, 32.0, 15.0, 0.0], "init_lengthscale": 0.34},
    },
}


def available_examples() -> List[str]:
    return list(EXAMPLES.keys())


def make_config(example_name: str, fast: bool = False) -> Dict[str, Any]:
    if example_name not in EXAMPLES:
        raise ValueError(f"Unknown example {example_name}")

    config = m.default_config_2d()
    config["domain"] = ((-1.0, 1.0), (-1.0, 1.0))
    config["penalty_domain"] = ((-2.0, 2.0), (-2.0, 2.0))
    config["diagnostic_domain"] = ((-2.0, 2.0), (-2.0, 2.0))
    config["test_side"] = 23
    config["p_norm"] = 2.0
    config["init_variance"] = 1.0
    config["layer_noise2"] = 1e-4
    config["hard_identity_enabled"] = False  # pure linear augmented posterior
    config.update(EXAMPLES[example_name]["config"])

    if fast:
        config["n_train"] = min(int(config["n_train"]), 26)
        config["test_side"] = 15
        config["n_iter"] = 60
        config["burn"] = 20
        config["thin"] = 5
        config["penalty_grid_side"] = 7
        config["recovery_grid_side"] = 13
    else:
        config["n_iter"] = 400
        config["burn"] = 70
        config["thin"] = 6
        config["penalty_grid_side"] = 11
        config["recovery_grid_side"] = 17
    return config


def _safe_get_recovery(run: Dict[str, Any], key: str, default=None):
    rec = run.get("recovery", {})
    return rec.get(key, default)


def _mean_first_k(values: Iterable[float], k: int) -> float:
    x = np.asarray(list(values), float)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x[: min(k, x.size)]))


def summarize_single(single: Dict[str, Any], redundant_layers: int) -> Dict[str, float]:
    baseline = single["baseline"]
    penalized = single["penalized"]
    b_layer = np.asarray(_safe_get_recovery(baseline, "layer_l2_identity", []), float)
    p_layer = np.asarray(_safe_get_recovery(penalized, "layer_l2_identity", []), float)
    b_partial = np.asarray(_safe_get_recovery(baseline, "partial_l2_truth", []), float)
    p_partial = np.asarray(_safe_get_recovery(penalized, "partial_l2_truth", []), float)
    b_id = _mean_first_k(b_layer, redundant_layers)
    p_id = _mean_first_k(p_layer, redundant_layers)
    b_part = _mean_first_k(b_partial, redundant_layers)
    p_part = _mean_first_k(p_partial, redundant_layers)
    return {
        "baseline_rmse": float(baseline.get("rmse", np.nan)),
        "penalized_rmse": float(penalized.get("rmse", np.nan)),
        "rmse_gain_baseline_minus_penalized": float(baseline.get("rmse", np.nan) - penalized.get("rmse", np.nan)),
        "baseline_mean_redundant_identity_error": b_id,
        "penalized_mean_redundant_identity_error": p_id,
        "identity_gain_baseline_minus_penalized": b_id - p_id,
        "baseline_mean_partial_truth_error": b_part,
        "penalized_mean_partial_truth_error": p_part,
        "partial_gain_baseline_minus_penalized": b_part - p_part,
        "baseline_effective_depth": float(_safe_get_recovery(baseline, "estimated_effective_depth", np.nan)),
        "penalized_effective_depth": float(_safe_get_recovery(penalized, "estimated_effective_depth", np.nan)),
    }


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    numeric_keys = [
        k for k, v in rows[0].items()
        if k not in {"example", "description", "seed", "init_mode"}
        and isinstance(v, (int, float, np.floating))
    ]
    out = []
    for ex in sorted(set(row["example"] for row in rows)):
        sub = [r for r in rows if r["example"] == ex]
        agg = {"example": ex, "description": EXAMPLES[ex]["description"], "n_runs": len(sub)}
        for key in numeric_keys:
            vals = np.asarray([r[key] for r in sub], float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                agg[f"{key}_mean"] = float("nan")
                agg[f"{key}_se"] = float("nan")
            else:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(agg)
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(single: Dict[str, Any], study: Dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fig in single.get("figs", {}).items():
        fig.savefig(outdir / f"single_{name}.png", dpi=300)
    for name, fig in study.get("figs", {}).items():
        fig.savefig(outdir / f"study_{name}.png", dpi=300)
    plt.close("all")


def default_heatmap_limits() -> Dict[str, tuple]:
    return {
        "hidden_maps": (-1.40, 1.40),
        "final_surface": (-1.60, 1.60),
        "final_composition": (-1.60, 1.60),
        "layer_l2_truth": (0.0, 2.00),
        "partial_l2_truth": (0.0, 2.00),
    }


def run_one_example(example_name: str, root_outdir: Path, fast: bool, seeds: List[int], init_modes: List[str]) -> List[Dict[str, Any]]:
    meta = EXAMPLES[example_name]
    layer_fns = meta["truth"]()
    config = make_config(example_name, fast=fast)
    redundant_layers = int(meta["redundant_layers"])
    outdir = root_outdir / example_name
    outdir.mkdir(parents=True, exist_ok=True)
    heatmap_limits = default_heatmap_limits()
    single = m.run_single_experiment_2d(config, layer_fns, seed=seeds[0], init_mode=init_modes[0], heatmap_limits=heatmap_limits)
    study = m.run_recovery_study_2d(config, layer_fns, seeds=seeds, init_modes=init_modes, heatmap_limits=heatmap_limits)
    save_figures(single, study, outdir)
    rows = []
    for r in study["results"]:
        row = {"example": example_name, "description": meta["description"], "seed": r["seed"], "init_mode": r["init_mode"]}
        row.update(summarize_single(r, redundant_layers=redundant_layers))
        rows.append(row)
    write_csv(rows, outdir / "stress_2d_summary.csv")
    with (outdir / "stress_2d_config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\n[{example_name}] {meta['description']}")
    print(f"Saved outputs to: {outdir.resolve()}")
    print(f"Summary CSV: {(outdir / 'stress_2d_summary.csv').resolve()}")
    return rows


def print_example_list() -> None:
    print("Available examples:")
    for name, meta in EXAMPLES.items():
        print(f"  {name:28s} - {meta['description']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=available_examples() + ["all"], default="shallow_ridges", help="Which contrast example to run. Use 'all' to run every example.")
    parser.add_argument("--outdir", default="stress_2d_contrasts_out")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--list-examples", action="store_true")
    args = parser.parse_args()
    if args.list_examples:
        print_example_list()
        return
    root_outdir = Path(args.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)
    if args.fast:
        seeds = [0]
        init_modes = ["smooth_random"]
    else:
        seeds = [0, 1, 2]
        init_modes = ["smooth_random", "identity"]
    examples = available_examples() if args.example == "all" else [args.example]
    all_rows = []
    for ex in examples:
        all_rows.extend(run_one_example(ex, root_outdir, args.fast, seeds, init_modes))
    write_csv(all_rows, root_outdir / "all_stress_2d_runs.csv")
    write_csv(aggregate_rows(all_rows), root_outdir / "all_stress_2d_aggregate.csv")
    with (root_outdir / "example_descriptions.json").open("w") as f:
        json.dump({name: {"description": meta["description"], "config": meta["config"]} for name, meta in EXAMPLES.items()}, f, indent=2, default=str)
    print("\nCombined outputs:")
    print(f"  Per-run CSV:   {(root_outdir / 'all_stress_2d_runs.csv').resolve()}")
    print(f"  Aggregate CSV: {(root_outdir / 'all_stress_2d_aggregate.csv').resolve()}")
    print("\nBest paper-facing 2D examples are usually:")
    print("  1. localized_bumps_sparse")
    print("  2. boundary_transition")
    print("  3. active_radial_warp")
    print("Use shallow_ridges as the simplest theory-aligned 2D sanity check.")


if __name__ == "__main__":
    main()
