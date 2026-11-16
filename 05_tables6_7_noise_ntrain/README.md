# Appendix F.4 / Tables 6--7: full 42-cell sweep

This folder contains the new missing paper-facing driver
`run_noise_ntrain_sweep_1d.py`.  It uses **only** the shared 1D MCMC core (copied locally and minimally revised for the enlarged penalty
and diagnostic domain).

The driver implements the manuscript protocol literally:

- truth: edge transition;
- observed domain: [-1,1];
- penalty and layer-diagnostic domain: [-2,2];
- n_train = {10,15,20,30,40,60};
- sigma = {0,0.05,0.10,0.20,0.40,0.70,1.00};
- rho = 0.70;
- p = 2;
- eta = (45,25,12,0);
- hard thresholding disabled;
- seeds = {0,1,2};
- initializations = {smooth_random, identity}.

Thus there are 42 cells and 6 paired replicates per cell, or 252 regular-vs-
augmented comparisons.

Run:

```bash
python run_noise_ntrain_sweep_1d.py --outdir results
```

Outputs:

- `noise_ntrain_runs.csv`: all 252 paired replicates;
- `noise_ntrain_cells.csv`: the 42 cell summaries;
- `table6_by_noise.csv`: aggregation exactly as Table 6;
- `table7_by_ntrain.csv`: aggregation exactly as Table 7;
- `noise_ntrain_config.json`: exact run configuration.

Table 6 averages each noise level over all six sample sizes, three seeds and
two initializations (36 paired fits per row).  Table 7 averages each sample
size over all seven noise levels, three seeds and two initializations (42
paired fits per row).

No figures are produced by this driver.
