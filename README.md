# Neural Network Safety Verification for Traffic Sign Recognition

Course project for CSMATH 2026.

Paper title:

> 自动驾驶交通标志识别中的神经网络安全验证：基于抽象解释与线性松弛的可证鲁棒性分析

This repository contains a small, reproducible study of certified robustness for ReLU classifiers on a three-class GTSRB traffic sign subset. The project compares interval bound propagation (IBP), linear programming relaxation (LP), and mixed-integer linear programming refinement (MILP).

## Repository Structure

```text
.
├── paper.tex                    # LaTeX source of the course paper
├── references.bib               # BibTeX references
├── experiments/
│   └── certify_gtsrb.py         # Training and certification script
├── outputs/                     # Result JSON files and figures used by the paper
└── README.md
```

The local `data/`, `build/`, and final submission archives are intentionally ignored by Git. The GTSRB archive is large and should not be committed.

## Method Summary

The experiment trains a small one-hidden-layer ReLU MLP and verifies local robustness under an `L_inf` perturbation ball.

The verification methods are:

- `IBP`: fast interval bound propagation.
- `LP relaxation`: ReLU convex outer relaxation solved with linear programming.
- `MILP refinement`: ReLU activation states encoded with binary variables for tighter bounds.

## Data

The script supports two data modes:

- `gtsrb`: uses a local GTSRB archive under `data/data.zip`.
- `digits`: uses `sklearn.datasets.load_digits` as a lightweight sanity check.

For GTSRB, place a Kaggle-style archive containing `Train.csv`, `Test.csv`, `Train/`, and `Test/` at:

```text
data/data.zip
```

The repository does not include the dataset because it is large.

## Environment

Python dependencies:

```bash
python3 -m pip install numpy scipy scikit-learn pillow requests matplotlib
```

LaTeX build dependency:

```bash
brew install tectonic
```

## Reproduce Experiments

Run the main GTSRB epsilon sweep:

```bash
python3 experiments/certify_gtsrb.py \
  --dataset gtsrb \
  --classes 0,1,2 \
  --image-size 16 \
  --max-per-class 120 \
  --hidden 32 \
  --n-eval 10 \
  --eps-list 0.002,0.005,0.01 \
  --mode sweep \
  --out-dir outputs
```

Run a lightweight sanity check without downloading GTSRB:

```bash
python3 experiments/certify_gtsrb.py \
  --dataset digits \
  --classes 0,1,2 \
  --hidden 16 \
  --n-eval 10 \
  --eps 0.02 \
  --out-dir outputs_digits
```

## Build Paper

```bash
tectonic paper.tex --outdir build
```

The generated PDF is:

```text
build/paper.pdf
```

## Main Results

For the GTSRB three-class subset with `n=10`, hidden width 32:

| epsilon | IBP | LP relaxation | MILP |
|---:|---:|---:|---:|
| 0.002 | 0.9 | 0.9 | 0.9 |
| 0.005 | 0.5 | 0.9 | 0.9 |
| 0.010 | 0.0 | 0.5 | 0.6 |

The key observation is that IBP is fast but conservative, while LP and MILP provide tighter certificates for boundary samples.

## GitHub

Suggested public repository:

```text
https://github.com/yanyichen2024/traffic-sign-verification
```

