# Canonical Layer Inference for Deep Gaussian Processes

**Content** This is the code repository for the research manuscript **“Canonical Layer Inference for Deep Gaussian Processes via Symmetry-Breaking Augmentation”** by **Yichen Zhu** and **Hengrui Luo**. The repository contains the reproducibility code used for the numerical experiments in the manuscript. This code repository has been assisted by OpenAI ChatGPT 6.0 during the implementation, managing and debugging, the authors take responsibilities for the correctness.

- The experiment folders provide the paper-facing pipelines for the main numerical studies:
  - `01_table1_synthetic_1d/` — one-dimensional synthetic stress tests.
  - `02_table2_synthetic_2d/` — two-dimensional synthetic stress tests.
  - `03_table3_uci/` — UCI real-data comparison.
  - `04_table5_dimension_ntrain/` — ambient-dimension and sample-size sweep.
  - `05_tables6_7_noise_ntrain/` — full noise-level and sample-size sweep.
- The synthetic experiments use separate domains for prediction and for the layerwise identity diagnostics:
  - observations and final prediction are evaluated on `[-1,1]^d`;
  - the identity penalty is evaluated on `[-2,2]^d`;
  - layerwise diagnostics are evaluated on `[-2,2]^d`.
- The paper-facing drivers do not use SCAD penalties or hard identity thresholding.
- Each experiment folder is self-contained with respect to project code. When a shared 1D or 2D MCMC core is needed, the same minimally revised implementation is copied into the corresponding folder.
- This repository contains source code only. It does **not** include precomputed outputs, figures, reference-output folders, SCAD variants, DeepGP3 cross-checks, R alternatives, or legacy self-contained variants.

A convenient environment is Python 3.10+. Install the required packages by

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The UCI and dimension-sweep scripts require PyTorch, pandas, and scikit-learn. The MCMC synthetic scripts use NumPy and Matplotlib.

The recommended run order is

```bash
cd 01_table1_synthetic_1d
python run_linear_dgp_stress_1d.py --example all --outdir results

cd ../02_table2_synthetic_2d
python run_linear_dgp_stress_2d.py --example all --outdir results

cd ../03_table3_uci
python run_uci_table3.py --outdir results

cd ../04_table5_dimension_ntrain
python run_dgp_dimension_ntrain_sweep.py --outdir results

cd ../05_tables6_7_noise_ntrain
python run_noise_ntrain_sweep_1d.py --outdir results
```

The final command runs the full manuscript protocol for the noise/sample-size study: 42 configurations, three random seeds, two initializations, and both regular and augmented MCMC fits in every replicate. It is therefore the most computationally intensive experiment in this package.

**Abstract** Deep Gaussian processes (DGPs) represent a regression function as a composition of Gaussian-process layers. Although this compositional structure can be useful for modeling complex functions, the likelihood generally identifies only the final composition and does not uniquely determine the intermediate layers. The manuscript studies this structural non-identifiability and introduces a symmetry-breaking augmentation that favors canonical layer representations by penalizing unnecessary departures from the identity in redundant hidden layers. The theoretical analysis separates function-level learning from layerwise canonicalization, while the numerical experiments examine the resulting prediction–parsimony tradeoff across synthetic stress tests, real-data examples, ambient dimensions, sample sizes, and noise levels.

**Citation** We provide this code for reproducibility and research purposes. If you use this repository, please cite the manuscript using the following BibTeX entry:

```bibtex
@misc{zhu2026canonicallayerinferencedeep,
      title={Canonical Layer Inference for Deep Gaussian Processes via Symmetry-Breaking Augmentation}, 
      author={Yichen Zhu and Hengrui Luo},
      year={2026},
      eprint={2610.23842},
      archivePrefix={arXiv},
      primaryClass={math.ST},
      url={https://arxiv.org/abs/2610.23842}, 
}
```

Thank you for your interest in this work. Please feel free to reach out if you have questions about the code or the numerical experiments.
