# Table 2: two-dimensional synthetic stress tests

Files:

- `penalized_dgp_recovery_2d.py`: shared dense-MCMC 2D implementation minimally revised for `[-2,2]^2` penalty/diagnostic evaluation.
- `run_linear_dgp_stress_2d.py`: supplied eight-design stress-suite driver.

Run all eight designs:

```bash
python run_linear_dgp_stress_2d.py --example all --outdir results
```

The full driver uses three seeds and two initializations, hence six runs per
design.  It uses n=100, noise SD 0.10 and a 23x23 test grid, with the
manuscript's design-specific rho/eta schedules.  Hard identity thresholding is
disabled.
