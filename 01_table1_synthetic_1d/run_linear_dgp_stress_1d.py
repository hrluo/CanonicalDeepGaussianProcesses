#!/usr/bin/env python3
"""
Contrasting 1D stress-test suite for regular DGP vs linear identity-augmented DGP.

Place this file in the same folder as one of:
    penalized_dgp_recovery_1d(2).py
    penalized_dgp_recovery_1d.py

Then run examples such as:
    python run_linear_dgp_stress_1d_contrasts.py --example shallow_oscillatory --outdir out_shallow_osc
    python run_linear_dgp_stress_1d_contrasts.py --example shallow_bump --outdir out_bump
    python run_linear_dgp_stress_1d_contrasts.py --example multiscale_sparse --outdir out_multiscale
    python run_linear_dgp_stress_1d_contrasts.py --example active_penultimate_warp --outdir out_warp
    python run_linear_dgp_stress_1d_contrasts.py --example all --outdir out_all --fast

Purpose
-------
The most convincing paper-facing contrast is an over-specified DGP:
    f1 = id, f2 = id, f3 = id, f4 active.

A regular DGP can invent unnecessary hidden warps. The linear augmented DGP
should shrink redundant non-final layers toward identity and place the signal
in the final layer.

This suite also includes one active-penultimate example:
    f1 = id, f2 = id, f3 active warp, f4 active final layer.

That example checks that the linear penalty is not just forcing all hidden
layers to identity. It should allow the later, weakly penalized non-final layer
to activate while shrinking the earlier redundant layers.

Outputs
-------
Each run writes:
    stress_1d_summary.csv          per seed/init results
    stress_1d_aggregate.csv        aggregate means and standard errors
    stress_1d_config.json          base config for the example
    *.png                          prediction, layer, partial-composition, and heatmap figures

Primary diagnostics
-------------------
For the paper, emphasize:
    1. redundant identity error:
       mean_{j=1,2,3} || f_j_hat - id ||_L2
       for shallow examples, or mean_{j=1,2} for active_penultimate_warp;

    2. partial-composition error:
       whether intermediate compositions match the intended parsimonious structure;

    3. effective depth:
       regular DGP tends to over-activate layers; augmented DGP should be shallower;

    4. RMSE:
       useful, but secondary, because the theory is primarily about representation.
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
        here / "penalized_dgp_recovery_1d.py",
    ]

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "penalized_dgp_recovery_1d_local",
                path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not import {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not find penalized_dgp_recovery_1d(2).py or "
        "penalized_dgp_recovery_1d.py in the same folder as this driver."
    )


m = _load_same_folder_module()


def _identity(x):
    return np.asarray(x, float)


def truth_shallow_oscillatory():
    return [
        _identity,
        _identity,
        _identity,
        lambda z: (
            0.80 * np.sin(2.0 * np.pi * np.asarray(z, float))
            + 0.25 * np.cos(4.0 * np.pi * np.asarray(z, float))
        ),
    ]


def truth_shallow_bump():
    return [
        _identity,
        _identity,
        _identity,
        lambda z: (
            np.tanh(5.0 * (np.asarray(z, float) + 0.35))
            - np.tanh(5.0 * (np.asarray(z, float) - 0.25))
            + 0.15 * np.sin(5.0 * np.pi * np.asarray(z, float))
        ),
    ]


def truth_multiscale_sparse():
    def f4(z):
        z = np.asarray(z, float)
        local = np.exp(-35.0 * (z - 0.20) ** 2)
        return (
            0.55 * np.sin(2.0 * np.pi * z)
            + 0.30 * local * np.sin(9.0 * np.pi * z)
            - 0.20 * np.exp(-45.0 * (z + 0.45) ** 2)
        )

    return [_identity, _identity, _identity, f4]


def truth_edge_transition():
    def f4(z):
        z = np.asarray(z, float)
        return (
            1.10 / (1.0 + np.exp(-12.0 * (z + 0.62)))
            - 0.55
            + 0.18 * np.sin(3.0 * np.pi * z)
        )

    return [_identity, _identity, _identity, f4]


def truth_active_penultimate_warp():
    def f3(z):
        z = np.asarray(z, float)
        return np.tanh(1.9 * z) / np.tanh(1.9)

    def f4(u):
        u = np.asarray(u, float)
        return 0.85 * np.sin(2.4 * np.pi * u) + 0.25 * u

    return [_identity, _identity, f3, f4]


def truth_canceling_symmetry_trap():
    def warp(z):
        z = np.asarray(z, float)
        return z + 0.22 * np.sin(np.pi * z)

    def inv_warp_approx(u):
        u = np.asarray(u, float)
        z = u.copy()
        for _ in range(8):
            z = u - 0.22 * np.sin(np.pi * z)
        return z

    def f4(z):
        z = np.asarray(z, float)
        return 0.75 * np.sin(2.0 * np.pi * z)

    return [warp, inv_warp_approx, _identity, f4]


ExampleFn = Callable[[], List[Callable[[np.ndarray], np.ndarray]]]


EXAMPLES: Dict[str, Dict[str, Any]] = {
    "shallow_oscillatory": {
        "truth": truth_shallow_oscillatory,
        "description": "Shallow oscillatory final layer; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {
            "n_train": 30,
            "noise_sd": 0.10,
            "rho": 0.70,
            "eta": [40.0, 24.0, 12.0, 0.0],
            "init_lengthscale": 0.24,
        },
    },
    "shallow_bump": {
        "truth": truth_shallow_bump,
        "description": "Shallow steep bump/transition final layer; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {
            "n_train": 30,
            "noise_sd": 0.10,
            "rho": 0.70,
            "eta": [45.0, 26.0, 12.0, 0.0],
            "init_lengthscale": 0.24,
        },
    },
    "multiscale_sparse": {
        "truth": truth_multiscale_sparse,
        "description": "Shallow multiscale final layer with sparse/noisy observations.",
        "redundant_layers": 3,
        "config": {
            "n_train": 30,
            "noise_sd": 0.10,
            "rho": 0.65,
            "eta": [50.0, 30.0, 14.0, 0.0],
            "init_lengthscale": 0.20,
        },
    },
    "edge_transition": {
        "truth": truth_edge_transition,
        "description": "Shallow boundary transition; layers 1--3 redundant.",
        "redundant_layers": 3,
        "config": {
            "n_train": 30,
            "noise_sd": 0.10,
            "rho": 0.70,
            "eta": [45.0, 25.0, 12.0, 0.0],
            "init_lengthscale": 0.22,
        },
    },
    "active_penultimate_warp": {
        "truth": truth_active_penultimate_warp,
        "description": "Layers 1--2 redundant; layer 3 and final layer active.",
        "redundant_layers": 2,
        "config": {
            "n_train": 30,
            "noise_sd": 0.10,
            "rho": 0.70,
            "eta": [45.0, 26.0, 4.0, 0.0],
            "init_lengthscale": 0.26,
        },
    },
    "canceling_symmetry_trap": {
        "truth": truth_canceling_symmetry_trap,
        "description": "Non-identifiable decomposition with canceling warp/inverse warp.",
        "redundant_layers": 3,
        "config": {
            "n_train": 30,
            "noise_sd": 0.10,
            "rho": 0.70,
            "eta": [45.0, 25.0, 12.0, 0.0],
            "init_lengthscale": 0.24,
        },
    },
}


def available_examples() -> List[str]:
    return list(EXAMPLES.keys())


def make_config(example_name: str, fast: bool = False) -> Dict[str, Any]:
    if example_name not in EXAMPLES:
        raise ValueError(f"Unknown example {example_name}")

    config = m.default_config_1d()

    config["domain"] = (-1.0, 1.0)
    config["penalty_domain"] = (-2.0, 2.0)
    config["diagnostic_domain"] = (-2.0, 2.0)
    config["n_test"] = 250
    config["p_norm"] = 2.0
    config["init_variance"] = 1.0
    config["layer_noise2"] = 1e-4

    # Pure linear augmented posterior.
    # Do not turn this on for the paper-facing comparison; it is an extra
    # deterministic heuristic, not the linear augmented posterior.
    config["hard_identity_enabled"] = False

    config.update(EXAMPLES[example_name]["config"])

    if fast:
        config["n_iter"] = 80
        config["burn"] = 25
        config["thin"] = 5
        config["recovery_grid_size"] = 151
        config["penalty_grid_size"] = 61
        config["n_test"] = 151
    else:
        config["n_iter"] = 300
        config["burn"] = 100
        config["thin"] = 8
        config["recovery_grid_size"] = 301
        config["penalty_grid_size"] = 101

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
        "rmse_gain_baseline_minus_penalized": float(
            baseline.get("rmse", np.nan) - penalized.get("rmse", np.nan)
        ),
        "baseline_mean_redundant_identity_error": b_id,
        "penalized_mean_redundant_identity_error": p_id,
        "identity_gain_baseline_minus_penalized": b_id - p_id,
        "baseline_mean_partial_truth_error": b_part,
        "penalized_mean_partial_truth_error": p_part,
        "partial_gain_baseline_minus_penalized": b_part - p_part,
        "baseline_effective_depth": float(
            _safe_get_recovery(baseline, "estimated_effective_depth", np.nan)
        ),
        "penalized_effective_depth": float(
            _safe_get_recovery(penalized, "estimated_effective_depth", np.nan)
        ),
    }


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    numeric_keys = [
        k
        for k, v in rows[0].items()
        if k not in {"example", "description", "seed", "init_mode"}
        and isinstance(v, (int, float, np.floating))
    ]

    out = []
    example_names = sorted(set(row["example"] for row in rows))

    for ex in example_names:
        sub = [r for r in rows if r["example"] == ex]
        agg = {
            "example": ex,
            "description": EXAMPLES[ex]["description"],
            "n_runs": len(sub),
        }

        for key in numeric_keys:
            vals = np.asarray([r[key] for r in sub], float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                agg[f"{key}_mean"] = float("nan")
                agg[f"{key}_se"] = float("nan")
            else:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_se"] = (
                    float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                    if vals.size > 1
                    else 0.0
                )

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


def run_one_example(
    example_name: str,
    root_outdir: Path,
    fast: bool,
    seeds: List[int],
    init_modes: List[str],
) -> List[Dict[str, Any]]:
    meta = EXAMPLES[example_name]
    layer_fns = meta["truth"]()
    config = make_config(example_name, fast=fast)
    redundant_layers = int(meta["redundant_layers"])

    outdir = root_outdir / example_name
    outdir.mkdir(parents=True, exist_ok=True)

    heatmap_limits = {
        "layer_l2_truth": (0.0, 1.50),
        "partial_l2_truth": (0.0, 1.50),
    }

    single = m.run_single_experiment_1d(
        config,
        layer_fns,
        seed=seeds[0],
        init_mode=init_modes[0],
    )

    study = m.run_recovery_study_1d(
        config,
        layer_fns,
        seeds=seeds,
        init_modes=init_modes,
        heatmap_limits=heatmap_limits,
    )

    save_figures(single, study, outdir)

    rows = []
    for r in study["results"]:
        row = {
            "example": example_name,
            "description": meta["description"],
            "seed": r["seed"],
            "init_mode": r["init_mode"],
        }
        row.update(summarize_single(r, redundant_layers=redundant_layers))
        rows.append(row)

    write_csv(rows, outdir / "stress_1d_summary.csv")

    with (outdir / "stress_1d_config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\n[{example_name}] {meta['description']}")
    print(f"Saved outputs to: {outdir.resolve()}")
    print(f"Summary CSV: {(outdir / 'stress_1d_summary.csv').resolve()}")

    return rows


def print_example_list() -> None:
    print("Available examples:")
    for name, meta in EXAMPLES.items():
        print(f"  {name:26s} - {meta['description']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        choices=available_examples() + ["all"],
        default="shallow_oscillatory",
        help="Which contrast example to run. Use 'all' to run every example.",
    )
    parser.add_argument("--outdir", default="stress_1d_contrasts_out")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--list-examples", action="store_true")
    args = parser.parse_args()

    if args.list_examples:
        print_example_list()
        return

    root_outdir = Path(args.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        seeds = [0, 1]
        init_modes = ["smooth_random"]
    else:
        seeds = [0, 1, 2, 3, 4]
        init_modes = ["smooth_random", "identity"]

    examples = available_examples() if args.example == "all" else [args.example]

    all_rows = []
    for ex in examples:
        rows = run_one_example(
            example_name=ex,
            root_outdir=root_outdir,
            fast=args.fast,
            seeds=seeds,
            init_modes=init_modes,
        )
        all_rows.extend(rows)

    write_csv(all_rows, root_outdir / "all_stress_1d_runs.csv")
    aggregate = aggregate_rows(all_rows)
    write_csv(aggregate, root_outdir / "all_stress_1d_aggregate.csv")

    with (root_outdir / "example_descriptions.json").open("w") as f:
        json.dump(
            {
                name: {
                    "description": meta["description"],
                    "config": meta["config"],
                }
                for name, meta in EXAMPLES.items()
            },
            f,
            indent=2,
            default=str,
        )

    print("\nCombined outputs:")
    print(f"  Per-run CSV:   {(root_outdir / 'all_stress_1d_runs.csv').resolve()}")
    print(f"  Aggregate CSV: {(root_outdir / 'all_stress_1d_aggregate.csv').resolve()}")

    print("\nBest paper-facing examples are usually:")
    print("  1. shallow_bump")
    print("  2. multiscale_sparse")
    print("  3. active_penultimate_warp")
    print("Use shallow_oscillatory as the simplest theory-aligned sanity check.")


if __name__ == "__main__":
    main()
