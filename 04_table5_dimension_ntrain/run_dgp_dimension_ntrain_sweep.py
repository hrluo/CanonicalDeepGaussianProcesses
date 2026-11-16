#!/usr/bin/env python3
"""
Sweep input dimension d and sample size n_train for regular DGP vs
linear identity-augmented DGP.

This script is self-contained. It does not require the earlier 1D/2D recovery
scripts, because those scripts are specialized to d=1 and d=2. Here we use a
finite-rank random-Fourier-feature DGP so that dimensions
    d = 1, 2, 3, 5, 10, 50, 100
can be handled in one experiment.

Model
-----
Both methods use the same over-specified residual DGP architecture:

    h_0 = x,
    h_j = h_{j-1} + B_j phi_j(h_{j-1}),  j=1,2,3,
    yhat = beta^T phi_4(h_3) + b.

The hidden maps are vector-valued. Setting B_j = 0 gives the identity map.
The regular DGP has no identity penalty. The augmented DGP adds

    sum_j eta_j || g_j(x_tilde) - x_tilde ||

on a fixed Monte Carlo reference sample x_tilde ~ Unif([-2,2]^d), where
g_j(u)=u+B_j phi_j(u). This matches Appendix F.3 of the manuscript.

Default truth
-------------
The default truth is sparse-edge:

    y = g(x_1)                         for d=1,
    y = g(x_1, x_2)                    for d>=2,

and all remaining coordinates are nuisance. The true hidden layers are
identities. Thus this experiment asks:

    As d increases, does the regular DGP waste capacity by warping nuisance
    directions, and does the augmented DGP remain closer to the parsimonious
    identity-hidden-layer representation?

Outputs
-------
The output folder contains:

    dimension_ntrain_runs.csv
    dimension_ntrain_aggregate.csv
    dimension_ntrain_config.json

and heatmaps:

    heatmap_rmse_gain_mean.png
    heatmap_identity_gain_mean.png
    heatmap_depth_reduction_mean.png

Positive gain means the augmented DGP is better:

    rmse_gain      = regular RMSE - augmented RMSE
    identity_gain  = regular hidden identity error - augmented hidden identity error
    depth_reduction = regular effective depth - augmented effective depth

Example
-------
Fast smoke test:

    python run_dgp_dimension_ntrain_sweep.py --fast --outdir dim_sweep_fast

Fuller run:

    python run_dgp_dimension_ntrain_sweep.py \
        --d-grid 1,2,3,5,10,50,100 \
        --n-train-grid 25,50,100 \
        --seeds 0,1,2 \
        --lambda-dim 2.0 \
        --anchor-size 1024 \
        --outdir dim_sweep_full

For very high dimensions, start with --fast or fewer seeds, then scale up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires PyTorch. Install it with `pip install torch` "
        "or run in an environment where torch is available."
    ) from exc


# ---------------------------------------------------------------------
# Reproducibility and utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_int_grid(text: str) -> List[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("Empty integer grid.")
    if any(v <= 0 for v in vals):
        raise ValueError("All grid values must be positive.")
    return vals


def parse_float_grid(text: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("Empty float grid.")
    return vals


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Synthetic truths
# ---------------------------------------------------------------------

def make_X(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)


def truth_sparse_edge(X: np.ndarray) -> np.ndarray:
    """
    Sparse low-dimensional truth embedded in d dimensions.

    For d=1:
        y depends only on x1.

    For d>=2:
        y depends only on (x1, x2). All other coordinates are nuisance.
    """
    X = np.asarray(X, dtype=np.float32)
    u = X[:, 0]

    if X.shape[1] == 1:
        y = (
            1.10 / (1.0 + np.exp(-12.0 * (u + 0.62)))
            - 0.55
            + 0.18 * np.sin(3.0 * np.pi * u)
        )
        return y.astype(np.float32)

    v = X[:, 1]
    boundary = 1.0 / (1.0 + np.exp(-9.0 * (u + 0.55 * v + 0.12)))
    y = (
        1.05 * boundary
        - 0.52
        + 0.20 * np.sin(2.0 * np.pi * v)
        + 0.18 * u * v
    )
    return y.astype(np.float32)


def truth_sparse_bumps(X: np.ndarray) -> np.ndarray:
    """
    Sparse localized bumps embedded in d dimensions.
    For d>=2 the truth depends only on the first two coordinates.
    """
    X = np.asarray(X, dtype=np.float32)
    u = X[:, 0]

    if X.shape[1] == 1:
        y = (
            0.55 * np.sin(2.0 * np.pi * u)
            + 0.30 * np.exp(-35.0 * (u - 0.20) ** 2) * np.sin(9.0 * np.pi * u)
            - 0.20 * np.exp(-45.0 * (u + 0.45) ** 2)
        )
        return y.astype(np.float32)

    v = X[:, 1]
    bump1 = np.exp(-34.0 * ((u - 0.28) ** 2 + (v + 0.18) ** 2))
    bump2 = np.exp(-45.0 * ((u + 0.40) ** 2 + (v - 0.42) ** 2))
    bump3 = np.exp(-30.0 * ((u + 0.05) ** 2 + (v + 0.55) ** 2))
    y = (
        0.80 * bump1
        - 0.55 * bump2
        + 0.35 * bump3
        + 0.22 * np.sin(2.0 * np.pi * u)
        - 0.15 * np.cos(2.2 * np.pi * v)
    )
    return y.astype(np.float32)


def truth_dense_additive(X: np.ndarray) -> np.ndarray:
    """
    Dense additive truth.

    This is a useful robustness case. Unlike sparse_edge, many coordinates
    matter. The augmented DGP may still improve representation, but its RMSE
    advantage is expected to be less systematic.
    """
    X = np.asarray(X, dtype=np.float32)
    d = X.shape[1]
    k = min(d, 20)
    weights = 1.0 / np.sqrt(np.arange(1, k + 1, dtype=np.float32))
    terms = np.sin(np.pi * X[:, :k]) * weights[None, :]
    y = np.sum(terms, axis=1) / np.sqrt(np.sum(weights ** 2))
    if d >= 2:
        y = y + 0.15 * X[:, 0] * X[:, 1]
    return y.astype(np.float32)


TRUTHS = {
    "sparse_edge": truth_sparse_edge,
    "sparse_bumps": truth_sparse_bumps,
    "dense_additive": truth_dense_additive,
}


def make_dataset(
    d: int,
    n_train: int,
    n_test: int,
    noise_sd: float,
    truth_name: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_train = make_X(n_train, d, rng)
    X_test = make_X(n_test, d, rng)

    f_train = TRUTHS[truth_name](X_train)
    f_test = TRUTHS[truth_name](X_test)

    y_train = f_train + noise_sd * rng.standard_normal(n_train).astype(np.float32)

    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
        f_test.astype(np.float32),
        f_train.astype(np.float32),
    )


# ---------------------------------------------------------------------
# Finite-rank residual DGP
# ---------------------------------------------------------------------

class RFFMap(nn.Module):
    """
    Fixed random Fourier features for an RBF-like kernel.
    """

    def __init__(self, in_dim: int, n_features: int, lengthscale: float):
        super().__init__()
        W = torch.randn(in_dim, n_features) / max(lengthscale, 1e-6)
        b = 2.0 * math.pi * torch.rand(n_features)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / n_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(X @ self.W + self.b)


class ResidualRFFDGP(nn.Module):
    """
    Vector-valued residual DGP with fixed RFF bases and trainable coefficients.
    """

    def __init__(
        self,
        d: int,
        n_hidden_layers: int = 3,
        n_features_hidden: int = 64,
        n_features_output: int = 128,
        lengthscale_hidden: float = 1.0,
        lengthscale_output: float = 1.0,
        residual_scale: float = 0.35,
        init_sd: float = 0.02,
    ):
        super().__init__()
        self.d = d
        self.n_hidden_layers = n_hidden_layers
        self.residual_scale = residual_scale

        self.hidden_features = nn.ModuleList(
            [
                RFFMap(d, n_features_hidden, lengthscale_hidden)
                for _ in range(n_hidden_layers)
            ]
        )
        self.hidden_coef = nn.ParameterList(
            [
                nn.Parameter(init_sd * torch.randn(n_features_hidden, d))
                for _ in range(n_hidden_layers)
            ]
        )

        self.output_features = RFFMap(d, n_features_output, lengthscale_output)
        self.output_coef = nn.Parameter(init_sd * torch.randn(n_features_output))
        self.output_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        X: torch.Tensor,
        return_hidden: bool = False,
    ):
        H = X
        hidden_states = [H]
        residuals = []

        for feat, coef in zip(self.hidden_features, self.hidden_coef):
            R = self.residual_scale * (feat(H) @ coef)
            H = H + R
            hidden_states.append(H)
            residuals.append(R)

        yhat = self.output_features(H) @ self.output_coef + self.output_bias

        if return_hidden:
            return yhat, hidden_states, residuals
        return yhat

    def layer_residuals_on_reference(self, X: torch.Tensor):
        """Evaluate B_j phi_j(X) for each hidden map g_j on common inputs X."""
        return [
            self.residual_scale * (feat(X) @ coef)
            for feat, coef in zip(self.hidden_features, self.hidden_coef)
        ]

    def coefficient_norm2(self) -> torch.Tensor:
        out = torch.sum(self.output_coef ** 2) + self.output_bias ** 2
        for coef in self.hidden_coef:
            out = out + torch.sum(coef ** 2)
        return out


@dataclass
class FitResult:
    rmse: float
    mae: float
    identity_error: float
    layer_identity_errors: List[float]
    effective_depth: float
    final_loss: float


def fit_model(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_test_np: np.ndarray,
    f_test_np: np.ndarray,
    X_anchor_np: np.ndarray,
    d: int,
    seed: int,
    augmented: bool,
    noise_sd: float,
    n_hidden_layers: int,
    n_features_hidden: int,
    n_features_output: int,
    lengthscale_hidden: float,
    lengthscale_output: float,
    residual_scale: float,
    prior_weight: float,
    eta: Sequence[float],
    epochs: int,
    lr: float,
    depth_threshold: float,
    verbose: bool = False,
) -> FitResult:
    set_seed(seed)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    f_test = torch.tensor(f_test_np, dtype=torch.float32)
    X_anchor = torch.tensor(X_anchor_np, dtype=torch.float32)

    model = ResidualRFFDGP(
        d=d,
        n_hidden_layers=n_hidden_layers,
        n_features_hidden=n_features_hidden,
        n_features_output=n_features_output,
        lengthscale_hidden=lengthscale_hidden,
        lengthscale_output=lengthscale_output,
        residual_scale=residual_scale,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = float(X_train.shape[0])
    sigma2 = max(float(noise_sd) ** 2, 1e-4)

    eta_tensor = torch.tensor(list(eta), dtype=torch.float32)
    if len(eta_tensor) < n_hidden_layers:
        eta_tensor = torch.cat(
            [eta_tensor, torch.zeros(n_hidden_layers - len(eta_tensor))]
        )
    eta_tensor = eta_tensor[:n_hidden_layers]

    last_loss = float("nan")

    for t in range(epochs):
        opt.zero_grad()

        yhat, hidden_states, residuals = model(X_train, return_hidden=True)
        mse_term = 0.5 * torch.mean((yhat - y_train) ** 2) / sigma2

        # Average-posterior scaling: as n grows, the likelihood dominates
        # the coefficient prior, while the identity penalty remains a tunable
        # augmentation strength.
        prior_term = 0.5 * prior_weight * model.coefficient_norm2() / n

        identity_term = torch.zeros(())
        if augmented:
            per_layer = []
            # Appendix F.3: evaluate each residual map g_j(u)-u on the
            # same fixed reference sample from Unif([-2,2]^d), not only
            # along the observed training trajectories.
            for R in model.layer_residuals_on_reference(X_anchor):
                per_layer.append(torch.sqrt(torch.mean(R ** 2) + 1e-12))
            per_layer_tensor = torch.stack(per_layer)
            identity_term = torch.sum(eta_tensor * per_layer_tensor)

        loss = mse_term + prior_term + identity_term
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().cpu())

        if verbose and (t % max(1, epochs // 5) == 0 or t == epochs - 1):
            print(f"      epoch={t:5d} loss={last_loss:.6f}")

    with torch.no_grad():
        yhat_test, hidden_states_test, residuals_test = model(X_test, return_hidden=True)

        rmse = float(torch.sqrt(torch.mean((yhat_test - f_test) ** 2)).cpu())
        mae = float(torch.mean(torch.abs(yhat_test - f_test)).cpu())

        layer_errs = []
        for R in residuals_test:
            # RMS displacement per coordinate on test inputs.
            layer_errs.append(float(torch.sqrt(torch.mean(R ** 2)).cpu()))

        identity_error = float(np.mean(layer_errs)) if layer_errs else float("nan")
        effective_depth = float(np.sum(np.asarray(layer_errs) > depth_threshold))

    return FitResult(
        rmse=rmse,
        mae=mae,
        identity_error=identity_error,
        layer_identity_errors=layer_errs,
        effective_depth=effective_depth,
        final_loss=last_loss,
    )


# ---------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------

def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return []

    group_keys = ["truth", "d", "n_train"]
    skip = set(group_keys + ["seed"])

    numeric_keys = [
        k for k, v in rows[0].items()
        if k not in skip and isinstance(v, (int, float, np.floating))
    ]

    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, sub in sorted(groups.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        agg = {
            "truth": key[0],
            "d": key[1],
            "n_train": key[2],
            "n_runs": len(sub),
        }
        for metric in numeric_keys:
            vals = np.asarray([r[metric] for r in sub], float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                agg[f"{metric}_mean"] = float("nan")
                agg[f"{metric}_se"] = float("nan")
            else:
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_se"] = (
                    float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                    if vals.size > 1
                    else 0.0
                )
        out.append(agg)

    return out


def matrix_from_aggregate(
    aggregate: List[Dict],
    d_grid: Sequence[int],
    n_grid: Sequence[int],
    metric: str,
) -> np.ndarray:
    idx = {
        (int(row["d"]), int(row["n_train"])): row
        for row in aggregate
    }
    mat = np.full((len(d_grid), len(n_grid)), np.nan)
    for i, d in enumerate(d_grid):
        for j, n_train in enumerate(n_grid):
            row = idx.get((int(d), int(n_train)))
            if row is not None:
                mat[i, j] = row.get(metric, np.nan)
    return mat


def save_heatmap(
    mat: np.ndarray,
    d_grid: Sequence[int],
    n_grid: Sequence[int],
    title: str,
    path: Path,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(mat, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(n_grid)))
    ax.set_xticklabels([str(x) for x in n_grid])
    ax.set_yticks(np.arange(len(d_grid)))
    ax.set_yticklabels([str(x) for x in d_grid])

    ax.set_xlabel("n_train")
    ax.set_ylabel("input dimension d")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_all_heatmaps(
    aggregate: List[Dict],
    d_grid: Sequence[int],
    n_grid: Sequence[int],
    outdir: Path,
) -> None:
    specs = [
        (
            "rmse_gain_mean",
            "RMSE gain",
            "regular RMSE - augmented RMSE",
            "heatmap_rmse_gain_mean.png",
        ),
        (
            "identity_gain_mean",
            "Hidden identity-error gain",
            "regular identity error - augmented identity error",
            "heatmap_identity_gain_mean.png",
        ),
        (
            "depth_reduction_mean",
            "Effective-depth reduction",
            "regular depth - augmented depth",
            "heatmap_depth_reduction_mean.png",
        ),
    ]

    for metric, title, label, filename in specs:
        mat = matrix_from_aggregate(aggregate, d_grid, n_grid, metric)
        save_heatmap(mat, d_grid, n_grid, title, outdir / filename, label)


# ---------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------

def run_sweep(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d_grid = parse_int_grid(args.d_grid)
    n_grid = parse_int_grid(args.n_train_grid)
    seeds = parse_int_grid(args.seeds)

    if args.truth not in TRUTHS:
        raise ValueError(f"Unknown truth {args.truth}. Choices: {sorted(TRUTHS)}")

    if args.fast:
        epochs = min(args.epochs, 250)
        n_features_hidden = min(args.n_features_hidden, 48)
        n_features_output = min(args.n_features_output, 96)
        seeds = seeds[: max(1, min(len(seeds), 2))]
        n_test = min(args.n_test, 1000)
    else:
        epochs = args.epochs
        n_features_hidden = args.n_features_hidden
        n_features_output = args.n_features_output
        n_test = args.n_test

    # Appendix F.3 uses eta = lambda_dim * (1, 0.60, 0.25).
    eta = [float(args.lambda_dim), 0.60 * float(args.lambda_dim), 0.25 * float(args.lambda_dim)]

    rows: List[Dict] = []

    total = len(d_grid) * len(n_grid) * len(seeds)
    counter = 0

    for d in d_grid:
        # A mild lengthscale scaling avoids immediate distance concentration
        # at high d while keeping the same random-feature model family.
        if args.lengthscale_hidden <= 0:
            lengthscale_hidden = math.sqrt(float(d))
        else:
            lengthscale_hidden = args.lengthscale_hidden

        if args.lengthscale_output <= 0:
            lengthscale_output = math.sqrt(float(d))
        else:
            lengthscale_output = args.lengthscale_output

        for n_train in n_grid:
            for seed in seeds:
                counter += 1
                print(
                    f"[{counter}/{total}] truth={args.truth} d={d} "
                    f"n_train={n_train} seed={seed}",
                    flush=True,
                )

                X_train, y_train, X_test, f_test, f_train = make_dataset(
                    d=d,
                    n_train=n_train,
                    n_test=n_test,
                    noise_sd=args.noise_sd,
                    truth_name=args.truth,
                    seed=seed,
                )
                anchor_rng = np.random.default_rng(1_000_000 + 1009 * seed + 37 * d + n_train)
                X_anchor = anchor_rng.uniform(
                    -2.0, 2.0, size=(int(args.anchor_size), d)
                ).astype(np.float32)

                # Same initialization/random features for regular and augmented.
                base_seed = 10_000 + 997 * seed + 31 * d + n_train

                regular = fit_model(
                    X_train_np=X_train,
                    y_train_np=y_train,
                    X_test_np=X_test,
                    f_test_np=f_test,
                    X_anchor_np=X_anchor,
                    d=d,
                    seed=base_seed,
                    augmented=False,
                    noise_sd=args.noise_sd,
                    n_hidden_layers=args.n_hidden_layers,
                    n_features_hidden=n_features_hidden,
                    n_features_output=n_features_output,
                    lengthscale_hidden=lengthscale_hidden,
                    lengthscale_output=lengthscale_output,
                    residual_scale=args.residual_scale,
                    prior_weight=args.prior_weight,
                    eta=eta,
                    epochs=epochs,
                    lr=args.lr,
                    depth_threshold=args.depth_threshold,
                    verbose=args.verbose,
                )

                augmented = fit_model(
                    X_train_np=X_train,
                    y_train_np=y_train,
                    X_test_np=X_test,
                    f_test_np=f_test,
                    X_anchor_np=X_anchor,
                    d=d,
                    seed=base_seed,
                    augmented=True,
                    noise_sd=args.noise_sd,
                    n_hidden_layers=args.n_hidden_layers,
                    n_features_hidden=n_features_hidden,
                    n_features_output=n_features_output,
                    lengthscale_hidden=lengthscale_hidden,
                    lengthscale_output=lengthscale_output,
                    residual_scale=args.residual_scale,
                    prior_weight=args.prior_weight,
                    eta=eta,
                    epochs=epochs,
                    lr=args.lr,
                    depth_threshold=args.depth_threshold,
                    verbose=args.verbose,
                )

                row = {
                    "truth": args.truth,
                    "d": int(d),
                    "n_train": int(n_train),
                    "seed": int(seed),
                    "noise_sd": float(args.noise_sd),
                    "regular_rmse": regular.rmse,
                    "augmented_rmse": augmented.rmse,
                    "rmse_gain": regular.rmse - augmented.rmse,
                    "regular_mae": regular.mae,
                    "augmented_mae": augmented.mae,
                    "mae_gain": regular.mae - augmented.mae,
                    "regular_identity_error": regular.identity_error,
                    "augmented_identity_error": augmented.identity_error,
                    "identity_gain": regular.identity_error - augmented.identity_error,
                    "regular_effective_depth": regular.effective_depth,
                    "augmented_effective_depth": augmented.effective_depth,
                    "depth_reduction": regular.effective_depth - augmented.effective_depth,
                    "regular_final_loss": regular.final_loss,
                    "augmented_final_loss": augmented.final_loss,
                    "regular_layer_identity_errors": json.dumps(regular.layer_identity_errors),
                    "augmented_layer_identity_errors": json.dumps(augmented.layer_identity_errors),
                }
                rows.append(row)

    runs_path = outdir / "dimension_ntrain_runs.csv"
    agg_path = outdir / "dimension_ntrain_aggregate.csv"

    write_csv(rows, runs_path)
    aggregate = aggregate_rows(rows)
    write_csv(aggregate, agg_path)
    save_all_heatmaps(aggregate, d_grid, n_grid, outdir)

    with (outdir / "dimension_ntrain_config.json").open("w") as f:
        json.dump(
            {
                "truth": args.truth,
                "d_grid": d_grid,
                "n_train_grid": n_grid,
                "seeds": seeds,
                "noise_sd": args.noise_sd,
                "n_test": n_test,
                "n_hidden_layers": args.n_hidden_layers,
                "n_features_hidden": n_features_hidden,
                "n_features_output": n_features_output,
                "epochs": epochs,
                "lr": args.lr,
                "eta": eta,
                "eta_shape": [1.0, 0.60, 0.25],
                "lambda_dim": args.lambda_dim,
                "anchor_domain": [-2.0, 2.0],
                "anchor_size": args.anchor_size,
                "prior_weight": args.prior_weight,
                "residual_scale": args.residual_scale,
                "depth_threshold": args.depth_threshold,
                "lengthscale_hidden": args.lengthscale_hidden,
                "lengthscale_output": args.lengthscale_output,
                "lengthscale_rule": "sqrt(d) when lengthscale argument <= 0",
                "fast": bool(args.fast),
            },
            f,
            indent=2,
            default=str,
        )

    print("\nDone.")
    print(f"Per-run CSV:   {runs_path.resolve()}")
    print(f"Aggregate CSV: {agg_path.resolve()}")
    print("Heatmaps:")
    print(f"  {outdir / 'heatmap_rmse_gain_mean.png'}")
    print(f"  {outdir / 'heatmap_identity_gain_mean.png'}")
    print(f"  {outdir / 'heatmap_depth_reduction_mean.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--truth",
        choices=sorted(TRUTHS.keys()),
        default="sparse_edge",
        help="Synthetic truth.",
    )

    parser.add_argument(
        "--d-grid",
        default="1,2,3,5,10,50,100",
        help="Comma-separated input dimensions.",
    )
    parser.add_argument(
        "--n-train-grid",
        default="25,50,100",
        help="Comma-separated training sizes.",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--outdir", default="dimension_ntrain_sweep")
    parser.add_argument("--noise-sd", type=float, default=0.10)
    parser.add_argument("--n-test", type=int, default=2000)

    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--n-features-hidden", type=int, default=64)
    parser.add_argument("--n-features-output", type=int, default=128)

    # If lengthscale is <= 0, use sqrt(d), which is safer in high dimension.
    parser.add_argument("--lengthscale-hidden", type=float, default=-1.0)
    parser.add_argument("--lengthscale-output", type=float, default=-1.0)

    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--lambda-dim",
        type=float,
        default=2.0,
        help=(
            "Common scale for Appendix F.3 eta=lambda_dim*(1,0.60,0.25). "
            "The manuscript states this scale is selected on a separate pilot "
            "run and then frozen; override this value if your frozen pilot scale differs."
        ),
    )
    parser.add_argument(
        "--anchor-size",
        type=int,
        default=1024,
        help="Number of fixed Unif([-2,2]^d) Monte Carlo anchor points per run.",
    )

    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--depth-threshold", type=float, default=0.025)

    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
