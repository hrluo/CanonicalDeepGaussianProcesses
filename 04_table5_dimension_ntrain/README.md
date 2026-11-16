# Appendix F.3 / Table 5: dimension and sample-size sweep

`run_dgp_dimension_ntrain_sweep.py` is the supplied script aligned to the manuscript's Appendix F.3 protocol.

Default manuscript grid:

- d = 1, 2, 3, 5, 10, 50, 100
- n_train = 25, 50, 100
- seeds = 0, 1, 2
- noise SD = 0.10
- sparse edge-transition truth

The identity penalty is evaluated on a fixed Monte Carlo reference sample from
`Unif([-2,2]^d)`, rather than only along training trajectories.  The layerwise
shape is eta = lambda_dim * (1, 0.60, 0.25).

Run:

```bash
python run_dgp_dimension_ntrain_sweep.py --outdir results
```

The draft states that `lambda_dim` is selected in a separate pilot validation
run and frozen across all 21 cells, but it does not print the numeric pilot
value.  The supplied script used a leading penalty of 2.0; accordingly the
paper-facing default here is `--lambda-dim 2.0`, yielding eta=(2.0,1.2,0.5).
Override it if the archived pilot value is different.

Likewise, the manuscript specifies a fixed Monte Carlo reference sample but
not its size M.  This release exposes `--anchor-size` and uses 1024 by default.
Those two values are therefore implementation defaults, not quantities stated
numerically in the manuscript.
