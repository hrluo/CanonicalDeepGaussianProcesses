#!/usr/bin/env python3
"""Truth constructions and finite-sample tuning schedules for Section 5.

The experiments are illustrations of the structural mechanisms in Section 5;
they are not numerical proofs of the asymptotic Matérn contraction theorems.
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def identity_1d(x):
    return np.asarray(x, float)


def identity_2d(X):
    return np.asarray(X, float)


def _holder_correct_target(x):
    x = np.asarray(x, float)
    return 0.62 * np.sin(1.5 * np.pi * x) + 0.22 * np.cos(np.pi * x) + 0.10 * x


def _oversmooth_target(x):
    x = np.asarray(x, float)
    return 0.55 * np.abs(x + 0.18) + 0.18 * np.sin(2.0 * np.pi * x) - 0.18


def _cusp_target(x):
    x = np.asarray(x, float)
    return -np.abs(x)


def _step_target(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.0, -1.0, 1.0)


def scenario_1d(name: str, n_train: int) -> Dict[str, Any]:
    """Return one Section-5 1D scenario.

    ``rate_exponent`` is the exponent r in the reference order n^{-r} stated
    in the corresponding Section-5 example.  It is used only to draw a slope
    guide on log-log plots.
    """
    name = str(name).lower()
    n = int(n_train)

    if name == "holder_correct":
        beta = 1.5
        rate = beta / (2.0 * beta + 1.0)
        lower_eta = 1.0 / (2.0 * beta + 1.0)
        upper_eta = (beta + 1.0) / (2.0 * beta + 1.0)
        eta_exp = 0.5 * (lower_eta + upper_eta)
        return {
            "name": name,
            "title": "Hölder smooth, correctly specified",
            "layer_fns": [identity_1d, identity_1d, identity_1d, _holder_correct_target],
            "kernel_kind": "matern32",
            "theory_beta": beta,
            "theory_alpha": beta,
            "rate_exponent": rate,
            "eta_base": [36.0, 22.0, 12.0, 0.0],
            "eta_exponents": [eta_exp, eta_exp, eta_exp, 0.0],
            "redundant_layers": [0, 1, 2],
            "active_nonfinal_layers": [],
            "init_lengthscale": 0.34,
            "rho": 0.70,
            "noise_sd": 0.10,
            "notes": "All non-final layers are redundant; reference slope is the correctly specified Hölder rate.",
        }

    if name == "holder_oversmooth":
        beta, alpha = 1.0, 2.0
        rate = beta / (2.0 * beta + 1.0)
        tau = n ** ((alpha - beta) / ((2.0 * beta + 1.0) * alpha))

        def f3(x, tau=tau):
            return tau * np.asarray(x, float)

        def f4(u, tau=tau):
            return _oversmooth_target(np.asarray(u, float) / tau)

        return {
            "name": name,
            "title": "Hölder smooth, oversmooth prior: active rescaling layer",
            "layer_fns": [identity_1d, identity_1d, f3, f4],
            "kernel_kind": "matern52",
            "theory_beta": beta,
            "theory_alpha": alpha,
            "rate_exponent": rate,
            "eta_base": [40.0, 24.0, 4.0, 0.0],
            "eta_exponents": [0.58, 0.58, 0.10, 0.0],
            "redundant_layers": [0, 1],
            "active_nonfinal_layers": [2],
            "init_lengthscale": 0.40,
            "rho": 0.70,
            "noise_sd": 0.10,
            "tau": float(tau),
            "notes": "The penultimate truth is the scaling tau_n x from the Section-5 rescaling construction; its penalty is deliberately weak.",
        }

    if name == "cusp":
        alpha = 1.5
        rate = alpha / (2.0 * alpha + 1.0)
        tau = 1.45 * (n / 30.0) ** 0.08

        def f3(x, tau=tau):
            return tau * np.asarray(x, float)

        def f4(u, tau=tau):
            return _cusp_target(np.asarray(u, float) / tau)

        return {
            "name": name,
            "title": "Non-differentiable cusp",
            "layer_fns": [identity_1d, identity_1d, f3, f4],
            "kernel_kind": "matern32",
            "theory_beta": 1.0,
            "theory_alpha": alpha,
            "rate_exponent": rate,
            "eta_base": [40.0, 24.0, 4.0, 0.0],
            "eta_exponents": [0.46, 0.46, 0.10, 0.0],
            "redundant_layers": [0, 1],
            "active_nonfinal_layers": [2],
            "init_lengthscale": 0.30,
            "rho": 0.70,
            "noise_sd": 0.10,
            "tau": float(tau),
            "notes": "Finite-sample local-structure stress test; the reference slope is the conditional rate in the revised Section-5 statement.",
        }

    if name == "step":
        alpha = 2.0
        rate = alpha / (2.0 * alpha + 3.0)
        tau = 1.50 * (n / 30.0) ** 0.06

        def f3(x, tau=tau):
            return tau * np.asarray(x, float)

        def f4(u, tau=tau):
            return _step_target(np.asarray(u, float) / tau)

        return {
            "name": name,
            "title": "Discontinuous step: proxy-motivated stress test",
            "layer_fns": [identity_1d, identity_1d, f3, f4],
            "kernel_kind": "matern52",
            "theory_beta": None,
            "theory_alpha": alpha,
            "rate_exponent": rate,
            "eta_base": [45.0, 28.0, 2.5, 0.0],
            "eta_exponents": [0.60, 0.60, 0.05, 0.0],
            "redundant_layers": [0, 1],
            "active_nonfinal_layers": [2],
            "init_lengthscale": 0.24,
            "rho": 0.70,
            "noise_sd": 0.10,
            "tau": float(tau),
            "notes": "Continuous Matérn DGPs can only approximate the jump in L2; this is a finite-sample stress test, not a verification of the conditional prior-mass construction.",
        }

    raise ValueError(f"Unknown Section-5 1D scenario {name!r}")


ONE_D_SCENARIOS = ["holder_correct", "holder_oversmooth", "cusp", "step"]


def intrinsic_2d_scenario(n_train: int) -> Dict[str, Any]:
    beta = 1.0
    rate = beta / (2.0 * beta + 1.0)

    def f3(X):
        X = np.asarray(X, float)
        t = 0.5 * (X[:, 0] + X[:, 1])
        return np.column_stack([t, np.zeros_like(t)])

    def f4(H):
        H = np.asarray(H, float)
        u = H[:, 0]
        return 0.82 * np.sin(np.pi * u) + 0.22 * u

    return {
        "name": "intrinsic_2d",
        "title": "Intrinsic one-dimensional structure in two ambient dimensions",
        "layer_fns": [identity_2d, identity_2d, f3, f4],
        "kernel_kind": "matern32",
        "theory_beta": beta,
        "rate_exponent": rate,
        "eta_base": [42.0, 26.0, 3.0, 0.0],
        "eta_exponents": [0.50, 0.50, 0.10, 0.0],
        "redundant_layers": [0, 1],
        "active_nonfinal_layers": [2],
        "init_lengthscale": 0.48,
        "rho": 0.70,
        "noise_sd": 0.10,
        "notes": "The third hidden map has rank-one image (t,0), t=(x1+x2)/2; this is a fixed-2D implementation of the conditional intrinsic representation.",
    }
