
"""
Penalized deep GP decomposition-recovery study (2D input).

This script keeps the same paper-facing comparison:
    Algorithm 1 = regular DGP
    Algorithm 2 = penalized DGP with identity shrinkage on non-final layers

It is redesigned around:
    1) truth-vs-estimate heatmap comparisons
    2) multi-seed, multi-initialization robustness
    3) concise user-editable configuration in the main block

All heatmaps accept shared color limits through function arguments or through the
heatmap_limits dictionary passed into the study runner.

No plotting function uses tight_layout().
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# GP and MCMC utilities
# ---------------------------------------------------------------------

def as_points(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("Expected a vector or a 2D array of points")


def rbf_kernel(X, Z, ell, var):
    X = as_points(X)
    Z = as_points(Z)
    diff = X[:, None, :] - Z[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
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
    return -0.5 * (x.size * np.log(2 * np.pi) + logdet + r @ alpha)


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


def gp_mean_var_scalar(X_star, X_train, y_train, ell, var, noise2):
    X_star = as_points(X_star)
    X_train = as_points(X_train)
    y_train = np.asarray(y_train, float)
    K = rbf_kernel(X_train, X_train, ell, var) + (noise2 + 1e-10) * np.eye(X_train.shape[0])
    L = chol_stable(K, jitter=1e-8)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    Ks = rbf_kernel(X_star, X_train, ell, var)
    mean = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    kss = var * np.ones(X_star.shape[0])
    var_post = np.maximum(kss - np.sum(v * v, axis=0), 1e-12)
    return mean, var_post


def gp_mean_var_multioutput(X_star, X_train, Y_train, ell, var, noise2):
    X_star = as_points(X_star)
    X_train = as_points(X_train)
    Y_train = np.asarray(Y_train, float)
    if Y_train.ndim != 2:
        raise ValueError("Y_train must be matrix-valued here")
    K = rbf_kernel(X_train, X_train, ell, var) + (noise2 + 1e-10) * np.eye(X_train.shape[0])
    L = chol_stable(K, jitter=1e-8)
    Ks = rbf_kernel(X_star, X_train, ell, var)
    v = np.linalg.solve(L, Ks.T)
    kss = var * np.ones(X_star.shape[0])
    base_var = np.maximum(kss - np.sum(v * v, axis=0), 1e-12)
    means = []
    for q in range(Y_train.shape[1]):
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train[:, q]))
        means.append(Ks @ alpha)
    mean = np.column_stack(means)
    var_post = np.column_stack([base_var.copy() for _ in range(Y_train.shape[1])])
    return mean, var_post


def gp_sample_scalar(X_star, X_train, y_train, ell, var, noise2, rng):
    mean, var_post = gp_mean_var_scalar(X_star, X_train, y_train, ell, var, noise2)
    return mean + np.sqrt(var_post) * rng.normal(size=mean.size)


def gp_sample_multioutput(X_star, X_train, Y_train, ell, var, noise2, rng):
    mean, var_post = gp_mean_var_multioutput(X_star, X_train, Y_train, ell, var, noise2)
    return mean + np.sqrt(var_post) * rng.normal(size=mean.shape)


def gp_output_logpdf(Y, K):
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        return mvn_logpdf_cov(Y, np.zeros_like(Y), K, jitter=1e-8)
    total = 0.0
    for q in range(Y.shape[1]):
        total += mvn_logpdf_cov(Y[:, q], np.zeros(Y.shape[0]), K, jitter=1e-8)
    return float(total)


def pack_outputs(Y):
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        return Y.copy()
    return np.concatenate([Y[:, q] for q in range(Y.shape[1])], axis=0)


def unpack_outputs(v, n_obs, out_dim):
    v = np.asarray(v, float)
    if out_dim == 1:
        return v.copy()
    return np.column_stack([v[q * n_obs:(q + 1) * n_obs] for q in range(out_dim)])


def pointwise_lp_norm(values, p=2.0):
    values = np.asarray(values, float)
    if values.ndim == 1:
        row_norms = np.abs(values)
    else:
        row_norms = np.linalg.norm(values, axis=1)
    if p == np.inf:
        return float(np.max(row_norms))
    return float((np.mean(row_norms ** p)) ** (1.0 / p))


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
# Truth, data, initialization
# ---------------------------------------------------------------------

def compose_layer_functions_2d(layer_fns, X):
    out = [as_points(X)]
    cur = out[0]
    for fn in layer_fns[:-1]:
        cur = as_points(fn(cur))
        out.append(cur)
    cur = np.asarray(layer_fns[-1](cur), float)
    out.append(cur)
    return out


def make_grid_2d(domain, side):
    x1 = np.linspace(domain[0][0], domain[0][1], int(side))
    x2 = np.linspace(domain[1][0], domain[1][1], int(side))
    g1, g2 = np.meshgrid(x1, x2, indexing="xy")
    pts = np.column_stack([g1.ravel(), g2.ravel()])
    return g1, g2, pts


def make_dataset_2d(layer_fns, domain, n_train, test_side, noise_sd, rng):
    x1 = rng.uniform(domain[0][0], domain[0][1], size=int(n_train))
    x2 = rng.uniform(domain[1][0], domain[1][1], size=int(n_train))
    X_train = np.column_stack([x1, x2])
    g1, g2, X_test = make_grid_2d(domain, test_side)
    truth_train = compose_layer_functions_2d(layer_fns, X_train)
    truth_test = compose_layer_functions_2d(layer_fns, X_test)
    y_train = truth_train[-1] + rng.normal(0.0, noise_sd, size=X_train.shape[0])
    return {
        "X_train": X_train,
        "X_test": X_test,
        "grid_x1": g1,
        "grid_x2": g2,
        "y_train": y_train,
        "y_true_test": truth_test[-1],
        "truth_train_states": truth_train,
        "truth_test_states": truth_test,
    }


def random_smooth_warp_2d(X, rng, scale=0.18):
    X = as_points(X)
    a = rng.normal(size=6)
    u = X[:, 0]
    v = X[:, 1]
    out1 = u + scale * (0.55 * np.sin(1.4 * u + a[0]) + 0.35 * np.cos(1.2 * v + a[1]) + 0.18 * a[2] * v)
    out2 = v + scale * (0.50 * np.sin(1.1 * v + a[3]) + 0.35 * np.cos(1.3 * u + a[4]) + 0.18 * a[5] * u)
    return np.column_stack([out1, out2])


def initial_hidden_states_2d(X, y_stdzd, rng, mode="identity", truth_states=None):
    X = as_points(X)
    y_stdzd = np.asarray(y_stdzd, float)
    if mode == "identity":
        return [X.copy(), X.copy(), X.copy(), X.copy(), y_stdzd.copy()]

    if mode == "smooth_random":
        h1 = random_smooth_warp_2d(X, rng)
        h2 = random_smooth_warp_2d(h1, rng)
        h3 = random_smooth_warp_2d(h2, rng)
        return [X.copy(), h1, h2, h3, y_stdzd.copy()]

    if mode == "truth_perturbed":
        if truth_states is None:
            raise ValueError("truth_states is required for truth_perturbed initialization")
        h1 = as_points(truth_states[1]) + 0.05 * rng.normal(size=truth_states[1].shape)
        h2 = as_points(truth_states[2]) + 0.05 * rng.normal(size=truth_states[2].shape)
        h3 = as_points(truth_states[3]) + 0.05 * rng.normal(size=truth_states[3].shape)
        return [X.copy(), h1, h2, h3, y_stdzd.copy()]

    raise ValueError(f"Unknown init mode: {mode}")


# ---------------------------------------------------------------------
# State, penalty, threshold logic
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
        if active and np.asarray(state["h"][j]).shape == np.asarray(state["h"][j + 1]).shape:
            state["h"][j + 1] = np.asarray(state["h"][j], float).copy()
    return state


def map_identity_distance_grid(grid_points, X_train, Y_train, ell, var, noise2, p=2.0):
    grid_points = as_points(grid_points)
    Y_train = np.asarray(Y_train, float)
    if Y_train.ndim == 1:
        mean, _ = gp_mean_var_scalar(grid_points, X_train, Y_train, ell, var, noise2)
        target = grid_points[:, 0]
        return pointwise_lp_norm(mean - target, p)
    mean, _ = gp_mean_var_multioutput(grid_points, X_train, Y_train, ell, var, noise2)
    return pointwise_lp_norm(mean - grid_points, p)


def latent_identity_distance(X_in, X_out, p=2.0):
    X_in = np.asarray(X_in, float)
    X_out = np.asarray(X_out, float)
    if X_in.shape != X_out.shape:
        return np.inf
    return pointwise_lp_norm(X_out - X_in, p)


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
        return -0.5 * ((le - np.log(0.45)) ** 2 + (lv - np.log(1.0)) ** 2)

    def target(le, lv):
        ell, var = float(np.exp(le)), float(np.exp(lv))
        K = rbf_kernel(h_in, h_in, ell, var) + layer_noise2 * np.eye(as_points(h_in).shape[0])
        val = gp_output_logpdf(h_out, K) + logprior(le, lv)
        if grid is not None and eta > 0.0:
            val -= eta * map_identity_distance_grid(grid, h_in, h_out, ell, var, layer_noise2, p_norm)
        return float(val)

    cur = target(log_ell, log_var)
    prop = target(prop_le, prop_lv)
    if np.log(rng.uniform()) < (prop - cur):
        return prop_le, prop_lv, True
    return log_ell, log_var, False


def mcmc_sweep_2d(state, rng):
    normalize_hard_threshold_config(state)
    force_identity_layers_inplace(state)

    X = state["X"]
    y = state["y_stdzd"]
    h = state["h"]
    log_ell, log_var = state["log_ell"], state["log_var"]
    layer_noise2 = state["layer_noise2"]
    rho, eta, p_norm = float(state["rho"]), list(state["eta"]), float(state["p_norm"])
    grid = state["penalty_grid"]

    # top scalar output layer
    ell4, var4 = float(np.exp(log_ell[3])), float(np.exp(log_var[3]))
    K4 = rbf_kernel(h[3], h[3], ell4, var4) + layer_noise2[3] * np.eye(X.shape[0])
    h[4] = gp_posterior_sample_latent(
        y=y,
        m_prior=np.zeros_like(y),
        K_prior=K4,
        noise_diag=state["sigma2"] * np.ones_like(y),
        rng=rng,
        rho_like=rho,
    )

    # hidden vector-valued layers h3, h2, h1
    for j in [3, 2, 1]:
        idx = j - 1
        if state["hard_identity_active"][idx]:
            h[j] = as_points(h[j - 1]).copy()
            continue

        ellj, varj = float(np.exp(log_ell[idx])), float(np.exp(log_var[idx]))
        Kj = rbf_kernel(h[j - 1], h[j - 1], ellj, varj) + layer_noise2[idx] * np.eye(X.shape[0])
        Lj = chol_stable(Kj, jitter=1e-8)
        out_dim = as_points(h[j]).shape[1]

        def loglike(hj_vec):
            hj_cand = unpack_outputs(hj_vec, n_obs=X.shape[0], out_dim=out_dim)
            elln, varn = float(np.exp(log_ell[j])), float(np.exp(log_var[j]))
            Kn = rbf_kernel(hj_cand, hj_cand, elln, varn) + layer_noise2[j] * np.eye(X.shape[0])
            ll = gp_output_logpdf(h[j + 1], Kn)
            if eta[idx] > 0.0:
                ll -= eta[idx] * map_identity_distance_grid(
                    grid, h[j - 1], hj_cand, ellj, varj, layer_noise2[idx], p_norm
                )
            return float(ll)

        cur = pack_outputs(h[j])
        mean = np.zeros_like(cur)
        L_big = np.kron(np.eye(out_dim), Lj)
        h[j] = unpack_outputs(ess_mvn(cur, mean, L_big, loglike, rng), n_obs=X.shape[0], out_dim=out_dim)

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


def make_initial_state_2d(data, rng, config, algorithm, init_mode):
    y_train = np.asarray(data["y_train"], float)
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train))
    y_stdzd = (y_train - y_mean) / max(y_std, 1e-12)
    h_init = initial_hidden_states_2d(
        data["X_train"], y_stdzd, rng, mode=init_mode, truth_states=data["truth_train_states"]
    )
    penalty_domain = config.get("penalty_domain", config["domain"])
    _, _, penalty_grid = make_grid_2d(penalty_domain, config["penalty_grid_side"])
    state = dict(
        X=as_points(data["X_train"]),
        y_stdzd=np.asarray(y_stdzd, float),
        y_mean=y_mean,
        y_std=max(y_std, 1e-12),
        h=[np.asarray(v, float) for v in h_init[:-1]] + [np.asarray(h_init[-1], float)],
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


def run_chain_2d(config, data, seed, algorithm, init_mode):
    rng = np.random.default_rng(seed)
    state = make_initial_state_2d(data, rng, config, algorithm, init_mode)
    kept = []
    for it in range(int(config["n_iter"])):
        mcmc_sweep_2d(state, rng)
        if it >= int(config["burn"]) and ((it - int(config["burn"])) % int(config["thin"]) == 0):
            kept.append({
                "h": [np.asarray(v, float).copy() for v in state["h"]],
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

def posterior_mean_path_2d(st, X_query):
    cur = as_points(X_query)
    partials = [cur.copy()]
    for j in range(3):
        if st.get("hard_identity_active", [False] * 4)[j]:
            partials.append(cur.copy())
            continue
        ell, var = float(np.exp(st["log_ell"][j])), float(np.exp(st["log_var"][j]))
        cur, _ = gp_mean_var_multioutput(cur, st["h"][j], st["h"][j + 1], ell, var, st["layer_noise2"][j])
        partials.append(cur.copy())
    ell4, var4 = float(np.exp(st["log_ell"][3])), float(np.exp(st["log_var"][3]))
    final, _ = gp_mean_var_scalar(cur, st["h"][3], st["h"][4], ell4, var4, st["layer_noise2"][3])
    partials.append(final)
    return partials


def posterior_predictive_2d(samples, X_test, draws_per_state=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    X_test = as_points(X_test)
    draws = []
    for st in samples:
        for _ in range(int(draws_per_state)):
            cur = X_test.copy()
            for j in range(3):
                if st.get("hard_identity_active", [False] * 4)[j]:
                    continue
                ell, var = float(np.exp(st["log_ell"][j])), float(np.exp(st["log_var"][j]))
                cur = gp_sample_multioutput(cur, st["h"][j], st["h"][j + 1], ell, var, st["layer_noise2"][j], rng)
            ell4, var4 = float(np.exp(st["log_ell"][3])), float(np.exp(st["log_var"][3]))
            y_draw = gp_sample_scalar(cur, st["h"][3], st["h"][4], ell4, var4, st["layer_noise2"][3], rng)
            draws.append(y_draw * st["y_std"] + st["y_mean"])
    draws = np.stack(draws, axis=0)
    return draws.mean(axis=0), draws.var(axis=0)


def layer_mean_map_2d(samples, layer_idx, X_query):
    X_query = as_points(X_query)
    maps = []
    for st in samples:
        if layer_idx < 3 and st.get("hard_identity_active", [False] * 4)[layer_idx]:
            maps.append(X_query.copy())
            continue
        ell, var = float(np.exp(st["log_ell"][layer_idx])), float(np.exp(st["log_var"][layer_idx]))
        if layer_idx < 3:
            mean, _ = gp_mean_var_multioutput(X_query, st["h"][layer_idx], st["h"][layer_idx + 1], ell, var, st["layer_noise2"][layer_idx])
        else:
            mean, _ = gp_mean_var_scalar(X_query, st["h"][layer_idx], st["h"][layer_idx + 1], ell, var, st["layer_noise2"][layer_idx])
        maps.append(mean)
    maps = np.stack(maps, axis=0)
    return maps.mean(axis=0), maps.var(axis=0), maps


def posterior_mean_partials_2d(samples, X_query):
    paths = [posterior_mean_path_2d(st, X_query) for st in samples]
    out = []
    for j in range(5):
        vals = [np.asarray(p[j], float) for p in paths]
        out.append(np.mean(np.stack(vals, axis=0), axis=0))
    return out


def recovery_metrics_2d(samples, layer_fns, domain, grid_side=35, active_tol=0.10):
    g1, g2, X_grid = make_grid_2d(domain, grid_side)
    truth_states = compose_layer_functions_2d(layer_fns, X_grid)
    est_layers = []
    est_partials = posterior_mean_partials_2d(samples, X_grid)

    layer_l2_truth = []
    layer_l2_identity = []
    active_prob = []
    for j in range(4):
        mean_j, _, maps_j = layer_mean_map_2d(samples, j, X_grid)
        est_layers.append(mean_j)
        layer_l2_truth.append(pointwise_lp_norm(mean_j - truth_states[j + 1], 2.0))
        if j < 3:
            layer_l2_identity.append(pointwise_lp_norm(mean_j - X_grid, 2.0))
            state_active = []
            for k in range(maps_j.shape[0]):
                state_active.append(pointwise_lp_norm(maps_j[k] - X_grid, 2.0) > active_tol)
            active_prob.append(float(np.mean(state_active)))
        else:
            layer_l2_identity.append(np.nan)
            active_prob.append(np.nan)

    partial_l2_truth = []
    for j in range(1, 5):
        partial_l2_truth.append(pointwise_lp_norm(est_partials[j] - truth_states[j], 2.0))

    true_active = [pointwise_lp_norm(truth_states[j + 1] - X_grid, 2.0) > active_tol for j in range(3)]
    return {
        "grid_x1": g1,
        "grid_x2": g2,
        "grid_points": X_grid,
        "truth_states_grid": truth_states,
        "est_layer_maps": est_layers,
        "est_partial_grid": est_partials,
        "layer_l2_truth": np.asarray(layer_l2_truth, float),
        "layer_l2_identity": np.asarray(layer_l2_identity, float),
        "partial_l2_truth": np.asarray(partial_l2_truth, float),
        "active_prob": np.asarray(active_prob, float),
        "true_effective_depth": int(np.sum(true_active)),
        "estimated_effective_depth": float(np.nansum(active_prob[:3])),
    }


def summarize_run_2d(samples, data, layer_fns, config):
    pred_mean, pred_var = posterior_predictive_2d(
        samples, data["X_test"], draws_per_state=config["predictive_draws_per_state"]
    )
    recovery = recovery_metrics_2d(
        samples, layer_fns, config.get("diagnostic_domain", config.get("penalty_domain", config["domain"])),
        grid_side=config["recovery_grid_side"], active_tol=config["active_tol"]
    )
    rmse = float(np.sqrt(np.mean((pred_mean - data["y_true_test"]) ** 2)))
    return {"pred_mean": pred_mean, "pred_var": pred_var, "rmse": rmse, "recovery": recovery, "samples": samples}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_final_surface_comparison_2d(data, baseline, penalized, value_limits=None):
    truth = data["y_true_test"].reshape(data["grid_x1"].shape)
    mean1 = baseline["pred_mean"].reshape(data["grid_x1"].shape)
    mean2 = penalized["pred_mean"].reshape(data["grid_x1"].shape)
    vmin, vmax = resolve_limits([truth, mean1, mean2], user_limits=value_limits, symmetric=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    titles = ["Truth", "Algorithm 1: regular DGP", "Algorithm 2: penalized DGP"]
    for ax, title, surface in zip(axes, titles, [truth, mean1, mean2]):
        im = ax.imshow(
            surface,
            origin="lower",
            extent=[data["grid_x1"].min(), data["grid_x1"].max(), data["grid_x2"].min(), data["grid_x2"].max()],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("surface value")
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.13, top=0.88, wspace=0.24)
    return fig


def plot_hidden_layer_heatmaps_2d(single, value_limits=None):
    rec1 = single["baseline"]["recovery"]
    rec2 = single["penalized"]["recovery"]
    g1, g2 = rec1["grid_x1"], rec1["grid_x2"]

    arrays = []
    for j in range(3):
        arrays.extend([
            rec1["truth_states_grid"][j + 1][:, 0].reshape(g1.shape),
            rec1["truth_states_grid"][j + 1][:, 1].reshape(g1.shape),
            rec1["est_layer_maps"][j][:, 0].reshape(g1.shape),
            rec1["est_layer_maps"][j][:, 1].reshape(g1.shape),
            rec2["est_layer_maps"][j][:, 0].reshape(g1.shape),
            rec2["est_layer_maps"][j][:, 1].reshape(g1.shape),
        ])
    vmin, vmax = resolve_limits(arrays, user_limits=value_limits, symmetric=False)

    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    col_titles = [
        "Truth c1", "Truth c2",
        "Alg 1 c1", "Alg 1 c2",
        "Alg 2 c1", "Alg 2 c2",
    ]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)
    for j in range(3):
        surfaces = [
            rec1["truth_states_grid"][j + 1][:, 0].reshape(g1.shape),
            rec1["truth_states_grid"][j + 1][:, 1].reshape(g1.shape),
            rec1["est_layer_maps"][j][:, 0].reshape(g1.shape),
            rec1["est_layer_maps"][j][:, 1].reshape(g1.shape),
            rec2["est_layer_maps"][j][:, 0].reshape(g1.shape),
            rec2["est_layer_maps"][j][:, 1].reshape(g1.shape),
        ]
        for c, surface in enumerate(surfaces):
            ax = axes[j, c]
            im = ax.imshow(
                surface,
                origin="lower",
                extent=[g1.min(), g1.max(), g2.min(), g2.max()],
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            if c == 0:
                ax.set_ylabel(f"layer {j + 1}\nx2")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("hidden-map value")
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.06, top=0.92, wspace=0.16, hspace=0.28)
    return fig


def plot_partial_heatmaps_2d(single, value_limits=None):
    rec1 = single["baseline"]["recovery"]
    rec2 = single["penalized"]["recovery"]
    g1, g2 = rec1["grid_x1"], rec1["grid_x2"]

    truth = rec1["truth_states_grid"][-1].reshape(g1.shape)
    est1 = rec1["est_partial_grid"][-1].reshape(g1.shape)
    est2 = rec2["est_partial_grid"][-1].reshape(g1.shape)
    vmin, vmax = resolve_limits([truth, est1, est2], user_limits=value_limits, symmetric=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, title, surface in zip(
        axes,
        ["Truth final composition", "Alg 1 final composition", "Alg 2 final composition"],
        [truth, est1, est2]
    ):
        im = ax.imshow(
            surface,
            origin="lower",
            extent=[g1.min(), g1.max(), g2.min(), g2.max()],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("composition value")
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.13, top=0.88, wspace=0.24)
    return fig


def plot_recovery_heatmaps_2d(study, metric_key="layer_l2_truth", value_limits=None):
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
        ax.set_xlabel("layer / partial")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(j + 1) for j in range(mat.shape[1])])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label(metric_key)
    fig.subplots_adjust(left=0.16, right=0.93, bottom=0.11, top=0.88, wspace=0.24)
    return fig


def plot_effective_depth_robustness_2d(study):
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

def run_single_experiment_2d(config, layer_fns, seed=0, init_mode="identity", heatmap_limits=None):
    rng = np.random.default_rng(seed)
    data = make_dataset_2d(
        layer_fns=layer_fns,
        domain=config["domain"],
        n_train=config["n_train"],
        test_side=config["test_side"],
        noise_sd=config["noise_sd"],
        rng=rng,
    )
    baseline_samples = run_chain_2d(config, data, seed=seed + 101, algorithm="baseline", init_mode=init_mode)
    penalized_samples = run_chain_2d(config, data, seed=seed + 202, algorithm="penalized", init_mode=init_mode)
    baseline = summarize_run_2d(baseline_samples, data, layer_fns, config)
    penalized = summarize_run_2d(penalized_samples, data, layer_fns, config)

    figs = {
        "final_surface": plot_final_surface_comparison_2d(
            data, baseline, penalized,
            value_limits=None if heatmap_limits is None else heatmap_limits.get("final_surface")
        ),
        "hidden_layers": plot_hidden_layer_heatmaps_2d(
            {"data": data, "baseline": baseline, "penalized": penalized},
            value_limits=None if heatmap_limits is None else heatmap_limits.get("hidden_maps")
        ),
        "final_composition": plot_partial_heatmaps_2d(
            {"data": data, "baseline": baseline, "penalized": penalized},
            value_limits=None if heatmap_limits is None else heatmap_limits.get("final_composition")
        ),
    }
    return {"data": data, "baseline": baseline, "penalized": penalized, "figs": figs}


def run_recovery_study_2d(config, layer_fns, seeds, init_modes, heatmap_limits=None):
    results = []
    run_labels = []
    for init_mode in init_modes:
        for seed in seeds:
            single = run_single_experiment_2d(
                config, layer_fns, seed=seed, init_mode=init_mode, heatmap_limits=heatmap_limits
            )
            results.append({"seed": seed, "init_mode": init_mode, **single})
            run_labels.append(f"{init_mode} | seed {seed}")

    metric_keys = ["layer_l2_truth", "partial_l2_truth", "active_prob"]
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
        "layer_truth_error": plot_recovery_heatmaps_2d(
            {"run_labels": run_labels, "matrices": matrices, "vectors": vectors, "truth": truth},
            metric_key="layer_l2_truth",
            value_limits=None if heatmap_limits is None else heatmap_limits.get("layer_l2_truth"),
        ),
        "partial_truth_error": plot_recovery_heatmaps_2d(
            {"run_labels": run_labels, "matrices": matrices, "vectors": vectors, "truth": truth},
            metric_key="partial_l2_truth",
            value_limits=None if heatmap_limits is None else heatmap_limits.get("partial_l2_truth"),
        ),
        "effective_depth": plot_effective_depth_robustness_2d(
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
# Defaults
# ---------------------------------------------------------------------

def default_config_2d():
    return {
        "domain": ((-1.0, 1.0), (-1.0, 1.0)),
        "penalty_domain": ((-2.0, 2.0), (-2.0, 2.0)),
        "diagnostic_domain": ((-2.0, 2.0), (-2.0, 2.0)),
        "n_train": 100,
        "test_side": 35,
        "noise_sd": 0.05,
        "n_iter": 180,
        "burn": 60,
        "thin": 6,
        "rho": 1.0,
        "eta": [8.0, 4.0, 1.0, 0.0],
        "p_norm": 2.0,
        "penalty_grid_side": 21,
        "layer_noise2": 1e-4,
        "init_sigma2": 0.1 ** 2,
        "noise_a0": 2.0,
        "noise_b0": 0.05,
        "init_lengthscale": 0.45,
        "init_variance": 1.0,
        "hyp_step_ell": 0.05,
        "hyp_step_var": 0.05,
        "predictive_draws_per_state": 1,
        "recovery_grid_side": 35,
        "active_tol": 0.10,
        "hard_identity_enabled": False,
        "hard_identity_layers": [True, True, True, False],
        "hard_identity_thresholds": [0.06, 0.06, 0.06, None],
        "hard_identity_sticky": True,
        "hard_identity_metric": "latent",
        "hard_identity_check_timing": "after_each_layer",
    }


def default_truth_2d():
    def f1(X):
        X = as_points(X)
        return np.column_stack([
            X[:, 0] + 0.35 * np.sin(np.pi * X[:, 1]),
            X[:, 1] + 0.25 * np.cos(np.pi * X[:, 0]),
        ])

    def f2(H):
        H = as_points(H)
        return np.column_stack([
            np.tanh(1.20 * H[:, 0] + 0.40 * H[:, 1]),
            np.tanh(-0.35 * H[:, 0] + 1.10 * H[:, 1]),
        ])

    def f3(H):
        H = as_points(H)
        return H

    def f4(H):
        H = as_points(H)
        return np.tanh(H[:, 0] + 0.55 * H[:, 1])

    return [f1, f2, f3, f4]


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # User-editable main section
    # ---------------------------------------------------------------
    layer_fns = default_truth_2d()

    config = default_config_2d()
    config["domain"] = ((-1.0, 1.0), (-1.0, 1.0))
    config["n_train"] = 25
    config["test_side"] = 15
    config["hard_identity_enabled"] = True
    config["hard_identity_thresholds"] = [0.05, 0.05, 0.05, None]

    seeds = [0, 1, 2]
    init_modes = ["identity", "smooth_random", "truth_perturbed"]

    heatmap_limits = {
        "hidden_maps": (-1.6, 1.6),
        "final_surface": (-1.3, 1.3),
        "final_composition": (-1.3, 1.3),
        "layer_l2_truth": (0.0, 1.50),
        "partial_l2_truth": (0.0, 1.50),
    }

    single = run_single_experiment_2d(config, layer_fns, seed=0, init_mode="identity", heatmap_limits=heatmap_limits)
    study = run_recovery_study_2d(config, layer_fns, seeds=seeds, init_modes=init_modes, heatmap_limits=heatmap_limits)

    # Example displays:
    single["figs"]["final_surface"]
    single["figs"]["hidden_layers"]
    single["figs"]["final_composition"]
    study["figs"]["layer_truth_error"]
    study["figs"]["partial_truth_error"]
    study["figs"]["effective_depth"]
