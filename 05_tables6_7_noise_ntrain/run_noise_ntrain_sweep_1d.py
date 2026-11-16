#!/usr/bin/env python3
"""
Reproduce Appendix F.4 and Tables 6--7 of the manuscript using the shared
1D MCMC implementation in ``penalized_dgp_recovery_1d.py``.

Paper protocol implemented here
-------------------------------
Truth: 1D edge-transition, with f1=f2=f3=id and only f4 active.
Observed domain: [-1, 1].
Identity-penalty / layer-diagnostic domain: [-2, 2].
Training sizes: 10, 15, 20, 30, 40, 60.
Noise SDs: 0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00.
Fractional likelihood power: rho = 0.70.
Identity norm: p = 2.
Penalty weights: eta = (45, 25, 12, 0).
Hard thresholding: disabled.
Replicates per grid cell: 3 random seeds x 2 initializations = 6.

The two initialization modes are the same two used by the full 1D stress-test
suite in the supplied 0808 code base: ``smooth_random`` and ``identity``.
The MCMC effort is also inherited from that full stress-test configuration:
300 iterations, burn-in 100, thinning 8, penalty grid 101, recovery grid 301.
The manuscript does not print these implementation-level MCMC constants, so
keeping them synchronized with the shared stress driver is important.

Outputs
-------
noise_ntrain_runs.csv
    252 paired fits (42 cells x 3 seeds x 2 initializations).
noise_ntrain_cells.csv
    42 cell means/SEs over the six replicates in each (n_train, sigma) cell.
table6_by_noise.csv
    Table-6 aggregation: for each sigma, average over all 6 n_train values,
    3 seeds, and 2 initializations (36 paired fits per row).
table7_by_ntrain.csv
    Table-7 aggregation: for each n_train, average over all 7 sigma values,
    3 seeds, and 2 initializations (42 paired fits per row).
noise_ntrain_config.json
    Exact configuration and aggregation metadata.

No figures are produced by this driver.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _load_core() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "penalized_dgp_recovery_1d.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing shared 1D core: {path}")
    spec = importlib.util.spec_from_file_location("penalized_dgp_recovery_1d_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load_core()


NTRAIN_GRID = [10, 15, 20, 30, 40, 60]
NOISE_GRID = [0.00, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
SEEDS = [0, 1, 2]
INIT_MODES = ["smooth_random", "identity"]


def _identity(x):
    return np.asarray(x, float)


def edge_transition_layers():
    def f4(z):
        z = np.asarray(z, float)
        return (
            1.10 / (1.0 + np.exp(-12.0 * (z + 0.62)))
            - 0.55
            + 0.18 * np.sin(3.0 * np.pi * z)
        )

    return [_identity, _identity, _identity, f4]


def paper_config(n_train: int, noise_sd: float) -> Dict[str, Any]:
    config = m.default_config_1d()

    # Manuscript-level settings.
    config["domain"] = (-1.0, 1.0)
    config["penalty_domain"] = (-2.0, 2.0)
    config["diagnostic_domain"] = (-2.0, 2.0)
    config["n_train"] = int(n_train)
    config["n_test"] = 250
    config["noise_sd"] = float(noise_sd)
    config["rho"] = 0.70
    config["eta"] = [45.0, 25.0, 12.0, 0.0]
    config["p_norm"] = 2.0
    config["hard_identity_enabled"] = False

    # Same implementation settings as the non-fast 1D stress driver in 0808.
    config["init_lengthscale"] = 0.22
    config["init_variance"] = 1.0
    config["layer_noise2"] = 1e-4
    config["n_iter"] = 300
    config["burn"] = 100
    config["thin"] = 8
    config["penalty_grid_size"] = 101
    config["recovery_grid_size"] = 301

    return config


def mean_first(values: Iterable[float], k: int = 3) -> float:
    arr = np.asarray(list(values), float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr[: min(k, arr.size)]))


def summarize_pair(baseline: Dict[str, Any], augmented: Dict[str, Any]) -> Dict[str, float]:
    b_rec = baseline["recovery"]
    a_rec = augmented["recovery"]

    # The edge-transition truth has identity layers 1--3, so this is exactly
    # the manuscript's Delta_id definition on the enlarged diagnostic grid.
    b_id = mean_first(b_rec["layer_l2_identity"], 3)
    a_id = mean_first(a_rec["layer_l2_identity"], 3)

    # The first three target partial compositions are also identity.
    b_partial = mean_first(b_rec["partial_l2_truth"], 3)
    a_partial = mean_first(a_rec["partial_l2_truth"], 3)

    return {
        "regular_rmse": float(baseline["rmse"]),
        "augmented_rmse": float(augmented["rmse"]),
        "regular_delta_id": b_id,
        "augmented_delta_id": a_id,
        "regular_delta_partial": b_partial,
        "augmented_delta_partial": a_partial,
        "regular_effective_depth": float(b_rec["estimated_effective_depth"]),
        "augmented_effective_depth": float(a_rec["estimated_effective_depth"]),
    }


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean_se(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size))


METRICS = [
    "regular_rmse",
    "augmented_rmse",
    "regular_delta_id",
    "augmented_delta_id",
    "regular_delta_partial",
    "augmented_delta_partial",
    "regular_effective_depth",
    "augmented_effective_depth",
]


def aggregate_group(sub: Sequence[Dict[str, Any]], prefix: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(prefix)
    out["n_runs"] = len(sub)
    for metric in METRICS:
        mu, se = mean_se(r[metric] for r in sub)
        out[metric] = mu
        out[f"{metric}_se"] = se
    return out


def aggregate_cells(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for n_train in NTRAIN_GRID:
        for sigma in NOISE_GRID:
            sub = [r for r in rows if r["n_train"] == n_train and np.isclose(r["sigma"], sigma)]
            out.append(aggregate_group(sub, {"n_train": n_train, "sigma": sigma}))
    return out


def aggregate_table6(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for sigma in NOISE_GRID:
        sub = [r for r in rows if np.isclose(r["sigma"], sigma)]
        out.append(aggregate_group(sub, {"sigma": sigma}))
    return out


def aggregate_table7(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for n_train in NTRAIN_GRID:
        sub = [r for r in rows if r["n_train"] == n_train]
        out.append(aggregate_group(sub, {"n_train": n_train}))
    return out


def run_sweep(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    layer_fns = edge_transition_layers()
    rows: List[Dict[str, Any]] = []

    total = len(NTRAIN_GRID) * len(NOISE_GRID) * len(SEEDS) * len(INIT_MODES)
    counter = 0

    for n_train in NTRAIN_GRID:
        for sigma in NOISE_GRID:
            config = paper_config(n_train=n_train, noise_sd=sigma)

            for seed in SEEDS:
                # One dataset per (n_train, sigma, seed), shared by both init modes
                # and by the regular/augmented fits.
                data_rng = np.random.default_rng(seed)
                data = m.make_dataset_1d(
                    layer_fns=layer_fns,
                    domain=config["domain"],
                    n_train=config["n_train"],
                    n_test=config["n_test"],
                    noise_sd=config["noise_sd"],
                    rng=data_rng,
                )

                for init_mode in INIT_MODES:
                    counter += 1
                    print(
                        f"[{counter}/{total}] n_train={n_train} sigma={sigma:.2f} "
                        f"seed={seed} init={init_mode}",
                        flush=True,
                    )

                    baseline_samples = m.run_chain_1d(
                        config, data, seed=seed + 101,
                        algorithm="baseline", init_mode=init_mode,
                    )
                    augmented_samples = m.run_chain_1d(
                        config, data, seed=seed + 202,
                        algorithm="penalized", init_mode=init_mode,
                    )

                    baseline = m.summarize_run_1d(baseline_samples, data, layer_fns, config)
                    augmented = m.summarize_run_1d(augmented_samples, data, layer_fns, config)

                    row: Dict[str, Any] = {
                        "n_train": int(n_train),
                        "sigma": float(sigma),
                        "seed": int(seed),
                        "init_mode": init_mode,
                    }
                    row.update(summarize_pair(baseline, augmented))
                    rows.append(row)

    write_csv(rows, outdir / "noise_ntrain_runs.csv")
    write_csv(aggregate_cells(rows), outdir / "noise_ntrain_cells.csv")
    write_csv(aggregate_table6(rows), outdir / "table6_by_noise.csv")
    write_csv(aggregate_table7(rows), outdir / "table7_by_ntrain.csv")

    config_record = {
        "manuscript_section": "Appendix F.4 / Tables 6--7",
        "truth": "edge_transition",
        "domain": [-1.0, 1.0],
        "penalty_domain": [-2.0, 2.0],
        "diagnostic_domain": [-2.0, 2.0],
        "n_train_grid": NTRAIN_GRID,
        "noise_sd_grid": NOISE_GRID,
        "seeds": SEEDS,
        "init_modes": INIT_MODES,
        "rho": 0.70,
        "p_norm": 2.0,
        "eta": [45.0, 25.0, 12.0, 0.0],
        "hard_identity_enabled": False,
        "n_test": 250,
        "n_iter": 300,
        "burn": 100,
        "thin": 8,
        "penalty_grid_size": 101,
        "recovery_grid_size": 301,
        "table6_aggregation": "within each sigma: 6 n_train x 3 seeds x 2 initializations = 36 paired fits",
        "table7_aggregation": "within each n_train: 7 sigma x 3 seeds x 2 initializations = 42 paired fits",
    }
    with (outdir / "noise_ntrain_config.json").open("w") as f:
        json.dump(config_record, f, indent=2)

    print("Done.")
    print(outdir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="noise_ntrain_table6_7")
    args = parser.parse_args()
    run_sweep(Path(args.outdir))


if __name__ == "__main__":
    main()
