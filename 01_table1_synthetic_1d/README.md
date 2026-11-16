# Table 1: one-dimensional synthetic stress tests

Files:

- `penalized_dgp_recovery_1d.py`: shared dense-MCMC DGP implementation minimally revised so the penalty and layer diagnostics are on
  `[-2,2]` while observations and RMSE remain on `[-1,1]`.
- `run_linear_dgp_stress_1d.py`: supplied 1D stress-suite driver.

Full run:

```bash
python run_linear_dgp_stress_1d.py --example all --outdir results
```

The non-fast driver uses five seeds and two initialization modes
(`smooth_random`, `identity`) for ten runs per design.  It uses the design-
specific rho and eta schedules in the manuscript appendix.  Hard identity
thresholding is disabled.

The identity-error summary was changed to use the code's explicit
`layer_l2_identity` diagnostic.  This matters for the canceling-symmetry trap,
where the arbitrary supplied warp/inverse pair is not the canonical identity
representative.

The historical 0808 driver predates the latest manuscript's expanded UQ panel;
its primary paper-facing outputs are RMSE, identity error, partial-composition
error and effective depth.  The full 42-cell Appendix F.4 experiment is in
folder 05 and does not depend on any separate/legacy implementation.
