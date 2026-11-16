#!/usr/bin/env python3
"""Shared helpers for the Section-5 finite-sample illustrations.

These helpers deliberately use only NumPy and Matplotlib plus the local DGP
modules in this folder.  They do not import code from sibling experiment
folders.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_int_grid(text: str) -> List[int]:
    vals = [int(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("empty integer grid")
    return vals


def parse_float_grid(text: str) -> List[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("empty float grid")
    return vals


def parse_text_grid(text: str) -> List[str]:
    vals = [v.strip() for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("empty text grid")
    return vals


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def aggregate_long(rows: List[Dict], group_keys: Sequence[str], metrics: Sequence[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out: List[Dict] = []
    for key, sub in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n_runs"] = len(sub)
        for metric in metrics:
            vals = np.asarray([float(r.get(metric, np.nan)) for r in sub], float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rec[f"{metric}_mean"] = float(vals.mean())
                rec[f"{metric}_se"] = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            else:
                rec[f"{metric}_mean"] = float("nan")
                rec[f"{metric}_se"] = float("nan")
        out.append(rec)
    return out


def eta_scaled(base: Sequence[float], exponents: Sequence[float], n: int, n_ref: float = 30.0) -> List[float]:
    if len(base) != len(exponents):
        raise ValueError("base and exponents must have equal length")
    ratio = float(n) / float(n_ref)
    return [float(b) * ratio ** float(k) if float(b) != 0.0 else 0.0 for b, k in zip(base, exponents)]


def make_random_dataset_1d(dgp1, layer_fns, n_train: int, n_test: int, noise_sd: float, seed: int,
                           domain=(-1.0, 1.0)) -> Dict:
    """Random-design analogue of the paper's 1D regression model."""
    rng = np.random.default_rng(int(seed))
    x_train = np.sort(rng.uniform(domain[0], domain[1], size=int(n_train)))
    x_test = np.linspace(domain[0], domain[1], int(n_test))
    truth_train = dgp1.compose_layer_functions_1d(layer_fns, x_train)
    truth_test = dgp1.compose_layer_functions_1d(layer_fns, x_test)
    y_train = truth_train[-1] + rng.normal(0.0, float(noise_sd), size=x_train.size)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_true_test": truth_test[-1],
        "truth_train_states": truth_train,
        "truth_test_states": truth_test,
    }


def _standardize_y(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, float)
    ym = float(np.mean(y))
    ys = float(np.std(y))
    ys = max(ys, 1e-10)
    return (y - ym) / ys, ym, ys


def single_gp_1d(dgp1, x_train, y_train, x_test, y_true_test, noise_sd: float,
                 kernel_kind: str, ell_grid: Sequence[float] = (0.08, 0.12, 0.18, 0.26, 0.38, 0.55, 0.8),
                 var_grid: Sequence[float] = (0.5, 1.0, 2.0)) -> Dict:
    """Simple empirical-Bayes single GP comparator using the same local kernel family."""
    x_train = np.asarray(x_train, float)
    x_test = np.asarray(x_test, float)
    y_std, ym, ys = _standardize_y(y_train)
    noise2 = max(float(noise_sd) ** 2 / (ys ** 2), 1e-8)
    best = None
    for ell in ell_grid:
        for var in var_grid:
            K = dgp1.gp_kernel_1d(x_train, x_train, float(ell), float(var), kernel_kind)
            K = K + noise2 * np.eye(x_train.size)
            try:
                score = dgp1.mvn_logpdf_cov(y_std, np.zeros_like(y_std), K)
            except np.linalg.LinAlgError:
                continue
            if best is None or score > best[0]:
                best = (float(score), float(ell), float(var))
    if best is None:
        raise RuntimeError("single-GP empirical-Bayes grid failed")
    _, ell, var = best
    mean_std, var_std = dgp1.gp_mean_var_1d(x_test, x_train, y_std, ell, var, noise2, kernel_kind)
    mean = mean_std * ys + ym
    var_post = var_std * ys ** 2
    rmse = float(np.sqrt(np.mean((mean - np.asarray(y_true_test, float)) ** 2)))
    return {"rmse": rmse, "pred_mean": mean, "pred_var": var_post, "ell": ell, "var": var}


def single_gp_2d(dgp2, X_train, y_train, X_test, y_true_test, noise_sd: float,
                 kernel_kind: str, ell_grid: Sequence[float] = (0.18, 0.26, 0.38, 0.55, 0.8, 1.15),
                 var_grid: Sequence[float] = (0.5, 1.0, 2.0)) -> Dict:
    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    y_std, ym, ys = _standardize_y(y_train)
    noise2 = max(float(noise_sd) ** 2 / (ys ** 2), 1e-8)
    best = None
    for ell in ell_grid:
        for var in var_grid:
            K = dgp2.gp_kernel_2d(X_train, X_train, float(ell), float(var), kernel_kind)
            K = K + noise2 * np.eye(X_train.shape[0])
            try:
                score = dgp2.mvn_logpdf_cov(y_std, np.zeros_like(y_std), K)
            except np.linalg.LinAlgError:
                continue
            if best is None or score > best[0]:
                best = (float(score), float(ell), float(var))
    if best is None:
        raise RuntimeError("single-GP empirical-Bayes grid failed")
    _, ell, var = best
    mean_std, var_std = dgp2.gp_mean_var_scalar(X_test, X_train, y_std, ell, var, noise2, kernel_kind)
    mean = mean_std * ys + ym
    var_post = var_std * ys ** 2
    rmse = float(np.sqrt(np.mean((mean - np.asarray(y_true_test, float)) ** 2)))
    return {"rmse": rmse, "pred_mean": mean, "pred_var": var_post, "ell": ell, "var": var}


def custom_active_layer_error_1d(dgp1, result: Dict, layer_fns, layer_idx: int) -> float:
    grid = np.asarray(result["recovery"]["grid"], float)
    est = np.asarray(result["recovery"]["est_layer_maps"][layer_idx], float)
    truth = np.asarray(layer_fns[layer_idx](grid), float)
    return dgp1.lp_norm(est - truth, 2.0)


def custom_active_layer_error_2d(dgp2, result: Dict, layer_fns, layer_idx: int) -> float:
    grid = np.asarray(result["recovery"]["grid_points"], float)
    est = np.asarray(result["recovery"]["est_layer_maps"][layer_idx], float)
    truth = np.asarray(layer_fns[layer_idx](grid), float)
    return dgp2.pointwise_lp_norm(est - truth, 2.0)
