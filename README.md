# One Operator to Rule Them All? Boundary-Indexed Operator Families in Neural PDE Solvers

This repository contains the code used in the paper:

“One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers”  
(ICLR 2026 Workshop on AI & PDE)

The goal of this project is to empirically demonstrate that standard neural PDE solvers do not learn a single boundary-agnostic solution operator when boundary conditions vary, but instead learn a boundary-distribution–indexed family of operators. The experiments are designed to isolate this effect in a controlled setting.

---

## Overview

We study the 2D Poisson equation on the unit square with mixed boundary conditions and show that:

- Neural operators generalize well *within* the boundary-condition distribution they are trained on.
- The same models fail sharply under boundary-condition distribution shift, even when the PDE, forcing distribution, and resolution are unchanged.
- Removing boundary-condition inputs causes models trained via empirical risk minimization to collapse to conditional expectations rather than valid solution operators.

All experiments are implemented in a single self-contained script and produce all figures and tables used in the paper.

---

## PDE Setup

The experiments consider the Poisson equation on [0,1] x [0,1] with mixed boundary conditions:
- Dirichlet on the left and bottom boundaries
- Neumann on the right and top boundaries

Boundary functions are sampled from parameterized families of smooth functions constructed via truncated Fourier series. Two distinct boundary-condition distributions are used to study cross-distribution generalization.

---

## Code Structure

- `bc_operator_family_experiments.py`  
  Main script that:
  - Generates synthetic PDE data
  - Trains neural operators under different boundary distributions
  - Evaluates cross-distribution generalization
  - Performs boundary extrapolation experiments
  - Demonstrates conditional-expectation behavior under boundary ablation
  - Exports figures, CSV tables, LaTeX-ready tables, and logs

There are no additional scripts or configuration files.

---

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib

A GPU is recommended but not strictly required.

---

## Running the Experiments

### Basic run (GPU)
```bash
python bc_operator_family_experiments.py --device cuda --train_steps 2500 --batch 12
````

### CPU-only run

```bash
python bc_operator_family_experiments.py --device cpu
```

### Windows OpenMP workaround (if needed)

If you encounter an OpenMP duplicate runtime warning on Windows, run:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

before executing the script.

---

## Outputs

After running the script, the output directory (default: `bc_operator_family_outputs/`) will contain:

### Logs and metrics

* `run_log.txt` — full training and evaluation log
* `metrics.json` — structured summary of all results

### Figures

* `error_vs_delta.png` — boundary extrapolation via Dirichlet shifts
* `error_vs_freq.png` — boundary extrapolation via increased boundary frequency
* `training_curve_*.png` — training dynamics for each model
* `condexp_compare.png` — conditional expectation demonstration
* `example_heatmaps.png` — same forcing, different boundary conditions

### Tables

* `cross_dist_table.csv` — cross-distribution generalization results
* `bc_ablation_table.csv` — boundary ablation comparison
* `sweep_delta.csv`, `sweep_freq.csv` — raw sweep results
* `tables/` — LaTeX-ready tables (`.tex`) used directly in the paper

These outputs correspond exactly to the figures and tables reported in the manuscript.

---

## Reproducibility

* All experiments use fixed random seeds.
* The forcing distribution is held constant across all runs.
* Only the boundary-condition distribution is varied when studying generalization.
* Ground-truth solutions are computed via a finite-difference discretization with a Jacobi iterative solver.

---

## Citation

If you use this code or find the results useful, please cite the paper:

```bibtex
@inproceedings{shikhman2026boundary,
  title={One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers},
  author={Shikhman, Lennon J.},
  booktitle={ICLR Workshop on Artificial Intelligence and Partial Differential Equations},
  year={2026}
}
```

---

## Contact

For questions or comments, please open an issue or contact the author directly.

---

*This repository is intended to support the paper’s claims and emphasize clarity and reproducibility over architectural novelty.*

```
