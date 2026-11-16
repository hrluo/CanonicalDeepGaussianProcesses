# Section 5 finite-sample illustrations

This folder supplements Section 5 (function-level contraction and layerwise
parsimony) of the draft.  It is deliberately separate from the Section 6 table
scripts: the purpose here is to **illustrate the mechanisms and conditional
examples in Section 5**, not to replace the reported numerical tables.

All code needed by this folder is local.  It does not import research code from
another experiment folder.

## What the experiments study

### 1. Correctly specified Hölder-smooth case

`run_section5_1d.py --scenario holder_correct`

The fitted model has four layers but the first three truth layers are identity.
The driver varies `n`, scales the identity penalties with `n`, and compares:

- a one-layer empirical-Bayes GP;
- an unaugmented four-layer DGP;
- the identity-augmented four-layer DGP.

The output overlays the Section-5 order `n^{-beta/(2 beta + 1)}` as a **reference
slope only**.  The finite-sample simulation is not an estimator or proof of the
asymptotic contraction exponent.

### 2. Oversmooth-prior rescaling construction

`run_section5_1d.py --scenario holder_oversmooth`

The target uses a Hölder-1 function and the local kernel is smoother.  The
penultimate truth layer is the scaling map `tau_n x`, matching the rescaling
view used in the conditional Section-5 construction; its penalty is kept much
smaller than those of the two redundant earlier layers.  The plots show whether
augmentation can simultaneously anchor the early layers and leave the useful
late hidden layer active.

### 3. Non-differentiable cusp

`run_section5_1d.py --scenario cusp`

The target is `f*(x)=-|x|`.  The first two layers are redundant and the third
layer is allowed to remain active.  This is a finite-sample local-structure
stress test corresponding to the conditional cusp statement.

### 4. Discontinuous step

`run_section5_1d.py --scenario step`

The target is the step function in Section 5.  Continuous Matérn GP layers can
only approximate a jump in `L2`; therefore this script is a **proxy-motivated
stress test** and not a verification of the conditional prior-mass construction.

### 5. Penalty-window / sweet-point diagnostic

`run_section5_eta_window_1d.py`

At fixed `n`, the driver multiplies the scenario-specific penalty schedule by a
common scalar `lambda`.  It plots prediction error against redundant-layer
identity error.  This is intended to visualize the finite-sample analogue of
the tuning-window discussion in Theorems 5.1--5.2: too little penalty gives
weak canonicalization, while too much penalty can damage function-level fit.

### 6. Intrinsic one-dimensional structure in two ambient dimensions

`run_section5_intrinsic_2d.py`

The truth depends on

    t = (x1 + x2) / 2,

with hidden layer 3 mapping `(x1,x2)` to `(t,0)`.  The first two hidden layers
are redundant.  The script measures both their identity errors and the error of
the learned active rank-one map.  This is a fixed-two-dimensional realization
of the conditional intrinsic-representation example; it does not claim that
the general theorem alone proves automatic dimension discovery.

## Kernel option added locally

The main Section 6 MCMC code uses an RBF kernel.  For Section 5, where the text
uses Matérn regularity to organize the examples, the **local copies** of
`penalized_dgp_recovery_1d.py` and `penalized_dgp_recovery_2d.py` add the
following dependency-free kernel options:

- `rbf`;
- `matern12`;
- `matern32`;
- `matern52`.

These half-integer Matérn kernels are used as finite-sample smoothness
comparators.  The kernel labels should not be read as a numerical identification
of the manuscript's abstract regularity parameter `alpha`.

## Conditional-mean penalty and blocked update

Like the numerical code in Section 6, this implementation approximates the
function-space identity penalty with a finite-grid penalty on retained-state GP
conditional-mean maps.  The local copies additionally include the adjacent
penalty contribution in the hidden-state ESS update: when `h_j` changes, the
identity penalty of the next non-final layer is recomputed because that layer's
conditional-mean map uses `h_j` as its GP input locations.

This makes the Section-5 drivers target a coherent conditional-mean-regularized
surrogate.  It is still distinct from an exact latent-path implementation of the
formal function-space posterior.

## Recommended full runs

From this folder:

```bash
python run_section5_1d.py \
  --scenario all \
  --n-grid 20,30,45,70 \
  --seeds 0,1,2 \
  --init-modes identity,smooth_random \
  --outdir results_1d
```

```bash
python run_section5_eta_window_1d.py \
  --scenario holder_correct \
  --n-train 45 \
  --lambda-grid 0,0.2,0.4,0.7,1.0,1.5,2.5 \
  --seeds 0,1,2 \
  --init-modes identity,smooth_random \
  --outdir results_eta_window
```

```bash
python run_section5_intrinsic_2d.py \
  --n-grid 40,70,100 \
  --seeds 0,1,2 \
  --init-modes identity,smooth_random \
  --outdir results_intrinsic_2d
```

For a short code-path test, append `--fast`.  The `--fast` results are not
intended for the manuscript.

## Outputs

### `run_section5_1d.py`

- `section5_1d_runs.csv`: run-level results;
- `section5_1d_aggregate.csv`: means and Monte Carlo standard errors;
- `section5_1d_config.json`;
- `section5_rate_<scenario>.png/.pdf`;
- `section5_prediction_<scenario>.png/.pdf`;
- `section5_layers_<scenario>.png/.pdf`.

The rate figure has three panels: function RMSE, redundant-layer identity error,
and redundant partial-composition error.

### `run_section5_eta_window_1d.py`

- `section5_eta_window_runs.csv`;
- `section5_eta_window_aggregate.csv`;
- `section5_eta_window_config.json`;
- `section5_eta_window_<scenario>.png/.pdf`;
- `section5_eta_partial_<scenario>.png/.pdf`.

### `run_section5_intrinsic_2d.py`

- `section5_intrinsic_2d_runs.csv`;
- `section5_intrinsic_2d_aggregate.csv`;
- `section5_intrinsic_2d_config.json`;
- `section5_intrinsic_2d_rates.png/.pdf`;
- `section5_intrinsic_2d_surfaces.png/.pdf`;
- `section5_intrinsic_2d_projection.png/.pdf`.

## Interpretation

The most defensible empirical questions for Section 5 are:

1. Does augmentation preserve broadly comparable function-level prediction as
   `n` grows?
2. Do redundant early layers move closer to identity?
3. When the scenario has a useful late hidden map, does a weak late penalty
   allow that map to remain active?
4. Is there a finite-sample penalty regime that improves representation without
   materially worsening prediction?
5. In the intrinsic example, can the active late layer recover the rank-one
   projection while the earlier layers remain near identity?

The simulations should be presented as finite-sample support for these
mechanisms.  They do not verify the conditional small-ball or active-block
prior-mass assumptions appearing in the Section-5 corollaries.
