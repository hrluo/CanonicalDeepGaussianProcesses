
"""
Penalized deep GP decomposition-recovery study (1D input).

This script keeps the same paper-facing comparison:
    Algorithm 1 = regular DGP
    Algorithm 2 = penalized DGP with identity shrinkage on non-final layers

It is redesigned around:
    1) truth-vs-estimate decomposition recovery
    2) multi-seed, multi-initialization robustness
    3) concise user-editable configuration in the main block

The code is intentionally written so the user can edit:
    - the true layer functions
    - the domain limits
    - train/test sample sizes
    - seeds and initialization modes
    - shared heatmap color limits

No plotting function uses tight_layout().
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# GP and MCMC utilities
# ---------------------------------------------------------------------

def rbf_kernel(x, z, ell, var):
    x = np.asarray(x, float)[:, None]
    z = np.asarray(z, float)[None, :]
    d2 = (x - z) ** 2
    return var * np.exp(-0.5 * d2 / max(ell ** 2, 1e-12))


def chol_stable(K, jitter=1e-8, max_tries=8):
    K = np.asarray(K, float)
    eye = np.eye(K.shape[0])
    cur = float(jitter)
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(K + cur * eye)
        except np.linalg.LinAlgError:
            cur *= 10.0
    raise np.linalg.LinAlgError("Cholesky failed")


def mvn_logpdf_chol(x, mean, L):
    x = np.asarray(x, float)
    mean = np.asarray(mean, float)
    r = x - mean
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    n = x.size
    return -0.5 * (n * np.log(2 * np.pi) + logdet + r @ alpha)


def mvn_logpdf_cov(x, mean, K, jitter=1e-8):
    return mvn_logpdf_chol(x, mean, chol_stable(K, jitter=jitter))


def invgamma_sample(shape, scale, rng):
    return 1.0 / rng.gamma(shape=shape, scale=1.0 / scale)


def ess_mvn(current, mean, L_prior, loglike_fn, rng, max_steps=1000):
    cur = np.asarray(current, float)
    mean = np.asarray(mean, float)
    nu = mean + L_prior @ rng.normal(size=cur.size)
    logy = float(loglike_fn(cur)) + np.log(rng.uniform())
    theta = rng.uniform(0.0, 2.0 * np.pi)
    theta_min, theta_max = theta - 2.0 * np.pi, theta
    gc, gn = cur - mean, nu - mean
    for _ in range(max_steps):
        prop = mean + gc * np.cos(theta) + gn * np.sin(theta)
        if float(loglike_fn(prop)) >= logy:
            return prop
        if theta < 0.0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)
    return cur


def gp_posterior_sample_latent(y, m_prior, K_prior, noise_diag, rng, rho_like=1.0):
    y = np.asarray(y, float)
    m_prior = np.asarray(m_prior, float)
    noise_diag = np.asarray(noise_diag, float)
    S = np.maximum(noise_diag / max(rho_like, 1e-12), 1e-12)
    L = chol_stable(K_prior, jitter=1e-8)
    sinv = 1.0 / S
    A = (sinv[:, None] ** 0.5) * L
    M = np.eye(y.size) + A.T @ A
    LM = chol_stable(M, jitter=1e-10)
    b = L.T @ (sinv * (y - m_prior))
    mu_z = np.linalg.solve(LM.T, np.linalg.solve(LM, b))
    z = mu_z + np.linalg.solve(LM.T, rng.normal(size=y.size))
    return m_prior + L @ z


def gp_mean_var_1d(x_star, x_train, y_train, ell, var, noise2):
    x_star = np.asarray(x_star, float)
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    K = rbf_kernel(x_train, x_train, ell, var) + (noise2 + 1e-10) * np.eye(x_train.size)
    L = chol_stable(K, jitter=1e-8)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    Ks = rbf_kernel(x_star, x_train, ell, var)
    mean = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    kss = var * np.ones(x_star.size)
    var_post = np.maximum(kss - np.sum(v * v, axis=0), 1e-12)
    return mean, var_post


def gp_sample_1d(x_star, x_train, y_train, ell, var, noise2, rng):
    mean, var_post = gp_mean_var_1d(x_star, x_train, y_train, ell, var, noise2)
    return mean + np.sqrt(var_post) * rng.normal(size=x_star.size)


def lp_norm(values, p=2.0):
    values = np.asarray(values, float)
    a = np.abs(values)
    if p == np.inf:
        return float(np.max(a))
    return float((np.mean(a ** p)) ** (1.0 / p))


def map_identity_distance_grid(grid, x_train, y_train, ell, var, noise2, p=2.0):
    mean, _ = gp_mean_var_1d(grid, x_train, y_train, ell, var, noise2)
    return lp_norm(mean - grid, p)


def latent_identity_distance(x_in, x_out, p=2.0):
    return lp_norm(np.asarray(x_out, float) - np.asarray(x_in, float), p)


def resolve_limits(arrays, user_limits=None, symmetric=False):
    if user_limits is not None:
        return tuple(user_limits)
    vals = np.concatenate([np.ravel(np.asarray(a, float)) for a in arrays if a is not None])
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    if np.isclose(lo, hi):
        pad = 1e-6 if lo == 0.0 else 0.05 * abs(lo)
        return (lo - pad, hi + pad)
    return (lo, hi)


# ---------------------------------------------------------------------
# Truth, data, and initialization
# ---------------------------------------------------------------------

def compose_layer_functions_1d(layer_fns, x):
    out = [np.asarray(x, float)]
    cur = out[0]
    for fn in layer_fns:
        cur = np.asarray(fn(cur), float)
        out.append(cur)
    return out


def make_dataset_1d(layer_fns, domain, n_train, n_test, noise_sd, rng):
    x_train = np.linspace(domain[0], domain[1], int(n_train))
    x_test = np.linspace(domain[0], domain[1], int(n_test))
    truth_train = compose_layer_functions_1d(layer_fns, x_train)
    truth_test = compose_layer_functions_1d(layer_fns, x_test)
    y_train = truth_train[-1] + rng.normal(0.0, noise_sd, size=x_train.size)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_true_test": truth_test[-1],
        "truth_train_states": truth_train,
        "truth_test_states": truth_test,
    }


def random_smooth_warp_1d(x, rng, scale=0.18):
    x = np.asarray(x, float)
    a, b, c, d = rng.normal(size=4)
    return x + scale * (
        0.65 * np.sin(1.6 * x + a) +
        0.45 * np.tanh(1.25 * x + b) +
        0.20 * c * x +
        0.15 * d
    )


def initial_hidden_states_1d(x, y_stdzd, rng, mode="identity", truth_states=None):
    x = np.asarray(x, float)
    y_stdzd = np.asarray(y_stdzd, float)
    if mode == "identity":
        return [x.copy(), x.copy(), x.copy(), x.copy(), y_stdzd.copy()]

    if mode == "smooth_random":
        h1 = random_smooth_warp_1d(x, rng)
        h2 = random_smooth_warp_1d(h1, rng)
        h3 = random_smooth_warp_1d(h2, rng)
        return [x.copy(), h1, h2, h3, y_stdzd.copy()]

    if mode == "truth_perturbed":
        if truth_states is None:
            raise ValueError("truth_states is required for truth_perturbed initialization")
        h1 = np.asarray(truth_states[1], float) + 0.05 * rng.normal(size=x.size)
        h2 = np.asarray(truth_states[2], float) + 0.05 * rng.normal(size=x.size)
        h3 = np.asarray(truth_states[3], float) + 0.05 * rng.normal(size=x.size)
        return [x.copy(), h1, h2, h3, y_stdzd.copy()]

    raise ValueError(f"Unknown init mode: {mode}")


# ---------------------------------------------------------------------
# Penalized DGP state and updates
# ---------------------------------------------------------------------

def default_hard_identity_layers(n_layers=4):
    flags = [False] * n_layers
    for j in range(min(3, n_layers)):
        flags[j] = True
    return flags


def default_hard_identity_thresholds(n_layers=4, threshold=0.05):
    vals = [None] * n_layers
    for j in range(min(3, n_layers)):
        vals[j] = float(threshold)
    return vals


def clone_state(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, list):
            out[k] = [item.copy() if isinstance(item, np.ndarray) else item for item in v]
        else:
            out[k] = v
    return out


def normalize_hard_threshold_config(state):
    n_layers = 4
    state["hard_identity_enabled"] = bool(state.get("hard_identity_enabled", False))
    state["hard_identity_layers"] = list(state.get("hard_identity_layers", default_hard_identity_layers(n_layers)))
    state["hard_identity_thresholds"] = list(state.get("hard_identity_thresholds", default_hard_identity_thresholds(n_layers)))
    state["hard_identity_active"] = list(state.get("hard_identity_active", [False] * n_layers))
    state["hard_identity_sticky"] = bool(state.get("hard_identity_sticky", True))
    state["hard_identity_metric"] = state.get("hard_identity_metric", "latent")
    state["hard_identity_check_timing"] = state.get("hard_identity_check_timing", "after_each_layer")
    state["hard_identity_last_distances"] = list(state.get("hard_identity_last_distances", [np.nan] * n_layers))
    return state


def force_identity_layers_inplace(state):
    for j, active in enumerate(state["hard_identity_active"]):
        if active:
            state["h"][j + 1] = state["h"][j].copy()
    return state


def current_layer_identity_distance(state, layer_idx):
    if state["hard_identity_metric"] == "latent":
        return latent_identity_distance(state["h"][layer_idx], state["h"][layer_idx + 1], state["p_norm"])
    ell = float(np.exp(state["log_ell"][layer_idx]))
    var = float(np.exp(state["log_var"][layer_idx]))
    return map_identity_distance_grid(
        state["penalty_grid"],
        state["h"][layer_idx],
        state["h"][layer_idx + 1],
        ell,
        var,
        float(state["layer_noise2"][layer_idx]),
        state["p_norm"],
    )


def maybe_force_identity(state, layer_idx=None):
    normalize_hard_threshold_config(state)
    if not state["hard_identity_enabled"]:
        state["hard_identity_last_distances"] = [np.nan] * 4
        return state

    layers = range(4) if layer_idx is None else [layer_idx]
    sticky = bool(state["hard_identity_sticky"])
    for j in layers:
        if not state["hard_identity_layers"][j]:
            continue
        threshold = state["hard_identity_thresholds"][j]
        if threshold is None:
            continue
        dist = current_layer_identity_distance(state, j)
        state["hard_identity_last_distances"][j] = dist
        hit = bool(dist <= float(threshold))
        if hit:
            state["hard_identity_active"][j] = True
        elif state["hard_identity_active"][j] and not sticky:
            state["hard_identity_active"][j] = False
    return force_identity_layers_inplace(state)


def mh_update_hypers(h_in, h_out, log_ell, log_var, layer_noise2, rng,
                     grid=None, eta=0.0, p_norm=2.0, step_ell=0.05, step_var=0.05):
    prop_le = log_ell + step_ell * rng.normal()
    prop_lv = log_var + step_var * rng.normal()

    def logprior(le, lv):
        return -0.5 * ((le - np.log(0.35)) ** 2 + (lv - np.log(1.0)) ** 2)

    def target(le, lv):
        ell, var = float(np.exp(le)), float(np.exp(lv))
        K = rbf_kernel(h_in, h_in, ell, var) + layer_noise2 * np.eye(h_in.size)
        val = mvn_logpdf_cov(h_out, np.zeros_like(h_out), K, jitter=1e-8) + logprior(le, lv)
        if grid is not None and eta > 0.0:
            val -= eta * map_identity_distance_grid(grid, h_in, h_out, ell, var, layer_noise2, p_norm)
        return float(val)

    cur = target(log_ell, log_var)
    prop = target(prop_le, prop_lv)
    if np.log(rng.uniform()) < (prop - cur):
        return prop_le, prop_lv, True
    return log_ell, log_var, False


def mcmc_sweep_1d(state, rng):
    normalize_hard_threshold_config(state)
    force_identity_layers_inplace(state)

    x, y = state["x"], state["y_stdzd"]
    h = state["h"]
    log_ell, log_var = state["log_ell"], state["log_var"]
    layer_noise2 = state["layer_noise2"]
    rho, eta, p_norm = float(state["rho"]), list(state["eta"]), float(state["p_norm"])
    grid = state["penalty_grid"]

    # top layer
    if state["hard_identity_active"][3]:
        h[4] = h[3].copy()
    else:
        ell4, var4 = float(np.exp(log_ell[3])), float(np.exp(log_var[3]))
        K4 = rbf_kernel(h[3], h[3], ell4, var4) + layer_noise2[3] * np.eye(x.size)
        h[4] = gp_posterior_sample_latent(
            y=y,
            m_prior=np.zeros_like(y),
            K_prior=K4,
            noise_diag=state["sigma2"] * np.ones_like(y),
            rng=rng,
            rho_like=rho,
        )

    # hidden layers h3, h2, h1
    for j in [3, 2, 1]:
        idx = j - 1
        if state["hard_identity_active"][idx]:
            h[j] = h[j - 1].copy()
            continue

        ellj, varj = float(np.exp(log_ell[idx])), float(np.exp(log_var[idx]))
        Kj = rbf_kernel(h[j - 1], h[j - 1], ellj, varj) + layer_noise2[idx] * np.eye(x.size)
        Lj = chol_stable(Kj, jitter=1e-8)

        def loglike(hj_cand):
            elln, varn = float(np.exp(log_ell[j])), float(np.exp(log_var[j]))
            Kn = rbf_kernel(hj_cand, hj_cand, elln, varn) + layer_noise2[j] * np.eye(x.size)
            ll = mvn_logpdf_cov(h[j + 1], np.zeros_like(h[j + 1]), Kn, jitter=1e-8)
            if eta[idx] > 0.0:
                ll -= eta[idx] * map_identity_distance_grid(
                    grid, h[j - 1], hj_cand, ellj, varj, layer_noise2[idx], p_norm
                )
            return float(ll)

        h[j] = ess_mvn(h[j], np.zeros_like(h[j]), Lj, loglike, rng)
        if state["hard_identity_enabled"] and state["hard_identity_check_timing"] == "after_each_layer":
            maybe_force_identity(state, layer_idx=idx)

    resid = y - h[4]
    a0, b0 = state["noise_a0"], state["noise_b0"]
    state["sigma2"] = invgamma_sample(
        a0 + 0.5 * rho * y.size,
        b0 + 0.5 * rho * float(np.sum(resid ** 2)),
        rng,
    )

    for j in range(4):
        if state["hard_identity_active"][j]:
            continue
        z_use = grid if j < 3 else None
        eta_j = eta[j] if j < 3 else 0.0
        log_ell[j], log_var[j], ok = mh_update_hypers(
            h[j], h[j + 1], log_ell[j], log_var[j], layer_noise2[j], rng,
            grid=z_use, eta=eta_j, p_norm=p_norm,
            step_ell=state["hyp_step_ell"], step_var=state["hyp_step_var"]
        )
        state["acc_hyp"] += int(ok)

    if state["hard_identity_enabled"] and state["hard_identity_check_timing"] != "after_each_layer":
        maybe_force_identity(state)

    state["h"], state["log_ell"], state["log_var"] = h, log_ell, log_var
    return state


def make_initial_state_1d(x, y_train, truth_train_states, rng, config, algorithm, init_mode):
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train))
    y_stdzd = (y_train - y_mean) / max(y_std, 1e-12)
    h_init = initial_hidden_states_1d(x, y_stdzd, rng, mode=init_mode, truth_states=truth_train_states)
    penalty_domain = config.get("penalty_domain", config["domain"])
    penalty_grid = np.linspace(penalty_domain[0], penalty_domain[1], int(config["penalty_grid_size"]))
    state = dict(
        x=np.asarray(x, float),
        y_stdzd=np.asarray(y_stdzd, float),
        y_mean=y_mean,
        y_std=max(y_std, 1e-12),
        h=[np.asarray(v, float) for v in h_init],
        log_ell=[np.log(config["init_lengthscale"])] * 4,
        log_var=[np.log(config["init_variance"])] * 4,
        layer_noise2=[config["layer_noise2"]] * 4,
        sigma2=config["init_sigma2"],
        noise_a0=config["noise_a0"],
        noise_b0=config["noise_b0"],
        rho=config["rho"],
        eta=[0.0, 0.0, 0.0, 0.0],
        p_norm=config["p_norm"],
        penalty_grid=penalty_grid,
        hyp_step_ell=config["hyp_step_ell"],
        hyp_step_var=config["hyp_step_var"],
        acc_hyp=0,
        hard_identity_enabled=False,
        hard_identity_layers=default_hard_identity_layers(),
        hard_identity_thresholds=default_hard_identity_thresholds(),
        hard_identity_active=[False] * 4,
        hard_identity_sticky=True,
        hard_identity_metric="latent",
        hard_identity_check_timing="after_each_layer",
        hard_identity_last_distances=[np.nan] * 4,
    )
    if algorithm == "penalized":
        state["eta"] = list(config["eta"])
        state["hard_identity_enabled"] = bool(config["hard_identity_enabled"])
        state["hard_identity_layers"] = list(config["hard_identity_layers"])
        state["hard_identity_thresholds"] = list(config["hard_identity_thresholds"])
        state["hard_identity_sticky"] = bool(config["hard_identity_sticky"])
        state["hard_identity_metric"] = config["hard_identity_metric"]
        state["hard_identity_check_timing"] = config["hard_identity_check_timing"]
    return normalize_hard_threshold_config(state)


def run_chain_1d(config, data, seed, algorithm, init_mode):
    rng = np.random.default_rng(seed)
    state = make_initial_state_1d(
        data["x_train"], data["y_train"], data["truth_train_states"], rng, config, algorithm, init_mode
    )
    kept = []
    for it in range(int(config["n_iter"])):
        mcmc_sweep_1d(state, rng)
        if it >= int(config["burn"]) and ((it - int(config["burn"])) % int(config["thin"]) == 0):
            kept.append({
                "h": [v.copy() for v in state["h"]],
                "log_ell": list(state["log_ell"]),
                "log_var": list(state["log_var"]),
                "layer_noise2": list(state["layer_noise2"]),
                "sigma2": float(state["sigma2"]),
                "y_mean": float(state["y_mean"]),
                "y_std": float(state["y_std"]),
                "hard_identity_active": list(state["hard_identity_active"]),
                "p_norm": state["p_norm"],
            })
    return kept


# ---------------------------------------------------------------------
# Posterior summaries and recovery metrics
# ---------------------------------------------------------------------

def posterior_predictive_1d(samples, x_test, rng=None, draws_per_state=1):
    if rng is None:
        rng = np.random.default_rng(0)
    x_test = np.asarray(x_test, float)
    draws = []
    for st in samples:
        for _ in range(int(draws_per_state)):
            cur = x_test.copy()
            for j in range(4):
                if st.get("hard_identity_active", [False] * 4)[j]:
                    continue
                ell, var = float(np.exp(st["log_ell"][j])), float(np.exp(st["log_var"][j]))
                cur = gp_sample_1d(cur, st["h"][j], st["h"][j + 1], ell, var, st["layer_noise2"][j], rng)
            draws.append(cur * st["y_std"] + st["y_mean"])
    draws = np.stack(draws, axis=0)
    return draws.mean(axis=0), draws.var(axis=0)


def layer_mean_map_1d(samples, layer_idx, grid):
    grid = np.asarray(grid, float)
    maps = []
    for st in samples:
        if st.get("hard_identity_active", [False] * 4)[layer_idx]:
            maps.append(grid.copy())
            continue
        ell, var = float(np.exp(st["log_ell"][layer_idx])), float(np.exp(st["log_var"][layer_idx]))
        mean, _ = gp_mean_var_1d(grid, st["h"][layer_idx], st["h"][layer_idx + 1], ell, var, st["layer_noise2"][layer_idx])
        maps.append(mean)
    maps = np.stack(maps, axis=0)
    return maps.mean(axis=0), maps.var(axis=0), maps


def partial_compositions_from_maps_1d(grid, mean_maps):
    vals = [np.asarray(grid, float)]
    cur = vals[0].copy()
    for layer_map in mean_maps:
        cur = np.interp(cur, grid, layer_map, left=layer_map[0], right=layer_map[-1])
        vals.append(cur)
    return vals


def state_active_indicator_1d(st, grid, active_tol):
    active = []
    for j in range(3):
        if st.get("hard_identity_active", [False] * 4)[j]:
            active.append(False)
            continue
        ell, var = float(np.exp(st["log_ell"][j])), float(np.exp(st["log_var"][j]))
        mean, _ = gp_mean_var_1d(grid, st["h"][j], st["h"][j + 1], ell, var, st["layer_noise2"][j])
        active.append(lp_norm(mean - grid, 2.0) > active_tol)
    return np.array(active, bool)


def recovery_metrics_1d(samples, layer_fns, domain, active_tol=0.08, grid_size=301):
    grid = np.linspace(domain[0], domain[1], int(grid_size))
    truth_states = compose_layer_functions_1d(layer_fns, grid)
    mean_maps = []
    metrics = {
        "layer_l2_truth": [],
        "layer_l2_identity": [],
        "partial_l2_truth": [],
        "active_prob": [],
        "estimated_effective_depth": np.nan,
        "true_effective_depth": np.nan,
    }

    for j in range(4):
        mean_j, _, maps_j = layer_mean_map_1d(samples, j, grid)
        mean_maps.append(mean_j)
        truth_j = truth_states[j + 1]
        metrics["layer_l2_truth"].append(lp_norm(mean_j - truth_j, 2.0))
        metrics["layer_l2_identity"].append(lp_norm(mean_j - grid, 2.0))
        if j < 3:
            state_active = []
            for k, st in enumerate(samples):
                if st.get("hard_identity_active", [False] * 4)[j]:
                    state_active.append(False)
                else:
                    state_active.append(lp_norm(maps_j[k] - grid, 2.0) > active_tol)
            metrics["active_prob"].append(float(np.mean(state_active)))
        else:
            metrics["active_prob"].append(np.nan)

    truth_partials = truth_states
    est_partials = partial_compositions_from_maps_1d(grid, mean_maps)
    for j in range(1, 5):
        metrics["partial_l2_truth"].append(lp_norm(est_partials[j] - truth_partials[j], 2.0))

    true_active = [lp_norm(truth_states[j + 1] - grid, 2.0) > active_tol for j in range(3)]
    metrics["true_effective_depth"] = int(np.sum(true_active))
    metrics["estimated_effective_depth"] = float(np.sum(metrics["active_prob"][:3]))
    for key in ["layer_l2_truth", "layer_l2_identity", "partial_l2_truth", "active_prob"]:
        metrics[key] = np.asarray(metrics[key], float)
    metrics["grid"] = grid
    metrics["truth_states_grid"] = truth_states
    metrics["est_partial_grid"] = est_partials
    metrics["est_layer_maps"] = mean_maps
    return metrics


def summarize_run_1d(samples, data, layer_fns, config):
    pred_mean, pred_var = posterior_predictive_1d(samples, data["x_test"], draws_per_state=config["predictive_draws_per_state"])
    diagnostic_domain = config.get("diagnostic_domain", config.get("penalty_domain", config["domain"]))
    recovery = recovery_metrics_1d(
        samples, layer_fns, diagnostic_domain, active_tol=config["active_tol"], grid_size=config["recovery_grid_size"]
    )
    rmse = float(np.sqrt(np.mean((pred_mean - data["y_true_test"]) ** 2)))
    return {"pred_mean": pred_mean, "pred_var": pred_var, "rmse": rmse, "recovery": recovery, "samples": samples}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_prediction_comparison_1d(data, baseline, penalized):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, title, res in zip(axes, ["Algorithm 1: regular DGP", "Algorithm 2: penalized DGP"], [baseline, penalized]):
        sd = np.sqrt(np.maximum(res["pred_var"], 0.0))
        ax.plot(data["x_test"], data["y_true_test"], linewidth=2, label="truth")
        ax.plot(data["x_test"], res["pred_mean"], linewidth=2, label="posterior mean")
        ax.fill_between(data["x_test"], res["pred_mean"] - 2 * sd, res["pred_mean"] + 2 * sd, alpha=0.2)
        ax.scatter(data["x_train"], data["y_train"], s=18, alpha=0.45)
        ax.set_title(f"{title}\nRMSE={res['rmse']:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.87, wspace=0.24)
    return fig


def plot_layer_truth_vs_estimate_1d(layer_fns, baseline, penalized, domain):
    grid = np.asarray(baseline["recovery"]["grid"], float)
    truth = baseline["recovery"]["truth_states_grid"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for j in range(4):
        for col, title, res in zip([0, 1], ["Algorithm 1", "Algorithm 2"], [baseline, penalized]):
            ax = axes[j, col]
            est = np.asarray(res["recovery"]["est_layer_maps"][j], float)
            ax.plot(grid, truth[j + 1], linewidth=2, label="truth")
            ax.plot(grid, est, linewidth=2, label="posterior mean")
            ax.plot(grid, grid, "--", linewidth=1, label="identity")
            ax.set_title(f"{title}: layer {j + 1}")
            ax.set_xlabel(f"input to layer {j + 1}")
            ax.set_ylabel(f"output of layer {j + 1}")
            ax.set_xlim(domain[0], domain[1])
            if j == 0 and col == 0:
                ax.legend(loc="best")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.96, hspace=0.34, wspace=0.24)
    return fig


def plot_partial_compositions_1d(baseline, penalized, domain):
    grid = np.asarray(baseline["recovery"]["grid"], float)
    truth = baseline["recovery"]["truth_states_grid"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for j in range(1, 5):
        for col, title, res in zip([0, 1], ["Algorithm 1", "Algorithm 2"], [baseline, penalized]):
            ax = axes[j - 1, col]
            est = np.asarray(res["recovery"]["est_partial_grid"][j], float)
            ax.plot(grid, truth[j], linewidth=2, label="truth partial composition")
            ax.plot(grid, est, linewidth=2, label="estimated partial composition")
            ax.set_title(f"{title}: G_{j}")
            ax.set_xlabel("x")
            ax.set_ylabel(f"G_{j}(x)")
            ax.set_xlim(domain[0], domain[1])
            if j == 1 and col == 0:
                ax.legend(loc="best")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.96, hspace=0.34, wspace=0.24)
    return fig


def plot_recovery_heatmaps_1d(study, metric_key="layer_l2_truth", value_limits=None):
    labels = study["run_labels"]
    baseline_mat = np.asarray(study["matrices"]["baseline"][metric_key], float)
    penalized_mat = np.asarray(study["matrices"]["penalized"][metric_key], float)
    vmin, vmax = resolve_limits([baseline_mat, penalized_mat], user_limits=value_limits, symmetric=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.35 * len(labels) + 1.8)))
    for ax, title, mat in zip(
        axes,
        ["Algorithm 1: regular DGP", "Algorithm 2: penalized DGP"],
        [baseline_mat, penalized_mat]
    ):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{metric_key}")
        ax.set_xlabel("layer")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(j + 1) for j in range(mat.shape[1])])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label(metric_key)
    fig.subplots_adjust(left=0.16, right=0.93, bottom=0.11, top=0.88, wspace=0.24)
    return fig


def plot_effective_depth_robustness_1d(study):
    labels = np.asarray(study["run_labels"])
    baseline = np.asarray(study["vectors"]["baseline"]["estimated_effective_depth"], float)
    penalized = np.asarray(study["vectors"]["penalized"]["estimated_effective_depth"], float)
    truth = float(study["truth"]["true_effective_depth"])
    x = np.arange(labels.size)

    fig, ax = plt.subplots(1, 1, figsize=(12, max(4.2, 0.30 * labels.size + 1.8)))
    ax.plot(x, baseline, "o-", label="Algorithm 1")
    ax.plot(x, penalized, "o-", label="Algorithm 2")
    ax.axhline(truth, linestyle="--", linewidth=1.5, label="true effective depth")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("effective depth")
    ax.set_title("Multi-seed / multi-initialization effective-depth recovery")
    ax.legend(loc="best")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.88)
    return fig


# ---------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------

def run_single_experiment_1d(config, layer_fns, seed=0, init_mode="identity"):
    rng = np.random.default_rng(seed)
    data = make_dataset_1d(
        layer_fns=layer_fns,
        domain=config["domain"],
        n_train=config["n_train"],
        n_test=config["n_test"],
        noise_sd=config["noise_sd"],
        rng=rng,
    )
    baseline_samples = run_chain_1d(config, data, seed=seed + 101, algorithm="baseline", init_mode=init_mode)
    penalized_samples = run_chain_1d(config, data, seed=seed + 202, algorithm="penalized", init_mode=init_mode)
    baseline = summarize_run_1d(baseline_samples, data, layer_fns, config)
    penalized = summarize_run_1d(penalized_samples, data, layer_fns, config)
    figs = {
        "prediction": plot_prediction_comparison_1d(data, baseline, penalized),
        "layers": plot_layer_truth_vs_estimate_1d(
            layer_fns, baseline, penalized, config.get("diagnostic_domain", config.get("penalty_domain", config["domain"]))
        ),
        "partials": plot_partial_compositions_1d(
            baseline, penalized, config.get("diagnostic_domain", config.get("penalty_domain", config["domain"]))
        ),
    }
    return {"data": data, "baseline": baseline, "penalized": penalized, "figs": figs}


def run_recovery_study_1d(config, layer_fns, seeds, init_modes, heatmap_limits=None):
    results = []
    run_labels = []
    for init_mode in init_modes:
        for seed in seeds:
            single = run_single_experiment_1d(config, layer_fns, seed=seed, init_mode=init_mode)
            results.append({"seed": seed, "init_mode": init_mode, **single})
            run_labels.append(f"{init_mode} | seed {seed}")

    metric_keys = ["layer_l2_truth", "layer_l2_identity", "partial_l2_truth", "active_prob"]
    vector_keys = ["rmse", "estimated_effective_depth"]
    matrices = {"baseline": {}, "penalized": {}}
    vectors = {"baseline": {}, "penalized": {}}

    for alg in ["baseline", "penalized"]:
        for key in metric_keys:
            matrices[alg][key] = np.vstack([r[alg]["recovery"][key] for r in results])
        vectors[alg]["rmse"] = np.asarray([r[alg]["rmse"] for r in results], float)
        vectors[alg]["estimated_effective_depth"] = np.asarray(
            [r[alg]["recovery"]["estimated_effective_depth"] for r in results], float
        )

    truth = results[0]["baseline"]["recovery"]
    figs = {
        "layer_truth_error": plot_recovery_heatmaps_1d(
            {"run_labels": run_labels, "matrices": matrices, "vectors": vectors, "truth": truth},
            metric_key="layer_l2_truth",
            value_limits=None if heatmap_limits is None else heatmap_limits.get("layer_l2_truth"),
        ),
        "partial_truth_error": plot_recovery_heatmaps_1d(
            {"run_labels": run_labels, "matrices": matrices, "vectors": vectors, "truth": truth},
            metric_key="partial_l2_truth",
            value_limits=None if heatmap_limits is None else heatmap_limits.get("partial_l2_truth"),
        ),
        "effective_depth": plot_effective_depth_robustness_1d(
            {"run_labels": run_labels, "matrices": matrices, "vectors": vectors, "truth": truth}
        ),
    }
    return {
        "results": results,
        "run_labels": run_labels,
        "matrices": matrices,
        "vectors": vectors,
        "truth": truth,
        "figs": figs,
    }


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

def default_config_1d():
    return {
        "domain": (-1.0, 1.0),
        "penalty_domain": (-2.0, 2.0),
        "diagnostic_domain": (-2.0, 2.0),
        "n_train": 45,
        "n_test": 250,
        "noise_sd": 0.08,
        "n_iter": 180,
        "burn": 60,
        "thin": 6,
        "rho": 1.0,
        "eta": [8.0, 4.0, 1.0, 0.0],
        "p_norm": 2.0,
        "penalty_grid_size": 101,
        "layer_noise2": 1e-4,
        "init_sigma2": 0.1 ** 2,
        "noise_a0": 2.0,
        "noise_b0": 0.05,
        "init_lengthscale": 0.35,
        "init_variance": 1.0,
        "hyp_step_ell": 0.05,
        "hyp_step_var": 0.05,
        "predictive_draws_per_state": 1,
        "recovery_grid_size": 301,
        "active_tol": 0.08,
        "hard_identity_enabled": False,
        "hard_identity_layers": [True, True, True, False],
        "hard_identity_thresholds": [0.05, 0.05, 0.05, None],
        "hard_identity_sticky": True,
        "hard_identity_metric": "latent",
        "hard_identity_check_timing": "after_each_layer",
    }


def default_truth_1d():
    return [
        lambda x: x + 0.35 * np.sin(2.0 * np.pi * x),
        lambda z: np.tanh(2.15 * z) + 0.10 * z,
        lambda z: z,
        lambda z: z,
    ]


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # User-editable main section
    # ---------------------------------------------------------------
    layer_fns = default_truth_1d()

    config = default_config_1d()
    config["domain"] = (-1.0, 1.0)
    config["n_train"] = 25
    config["n_test"] = 250
    config["hard_identity_enabled"] = True
    config["hard_identity_thresholds"] = [0.04, 0.04, 0.04, None]

    seeds = [0, 1, 2]
    init_modes = ["identity", "smooth_random", "truth_perturbed"]

    heatmap_limits = {
        "layer_l2_truth": (0.0, 1.00),
        "partial_l2_truth": (0.0, 1.00),
    }

    single = run_single_experiment_1d(config, layer_fns, seed=0, init_mode="identity")
    study = run_recovery_study_1d(config, layer_fns, seeds=seeds, init_modes=init_modes, heatmap_limits=heatmap_limits)

    # Example displays:
    single["figs"]["prediction"]
    single["figs"]["layers"]
    single["figs"]["partials"]
    study["figs"]["layer_truth_error"]
    study["figs"]["partial_truth_error"]
    study["figs"]["effective_depth"]
