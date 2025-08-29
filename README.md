# SP-CCI: Synthetic-Powered Conformal Counterfactual Inference

This repository contains tidy, reproducible code to run experiments for the paper:

> **Synthetic Counterfactual Labels for Efficient Conformal Counterfactual Inference**

SP-CCI augments Conformal Counterfactual Inference (CCI) with synthetic counterfactual labels while preserving finite-sample coverage via a debiased, risk-controlled calibration procedure inspired by **Prediction-Powered Inference (PPI)** and **Risk-Controlling Prediction Sets (RCPS)**.

**Key features**
- Baseline **CCI** implementation.
- **SP-CCI** with synthetic label augmentation and debiased miscoverage estimator.
- Experiments on **Synthetic data (Lei–Candès setup)** and **IHDP**; optional **wireless handover** (Sionna RT) scaffold.
- Figure scripts to reproduce plots like *Coverage* and *Prediction Width*.


---

## Repository Structure

```
sp-cci/
├── pyproject.toml
├── requirements.txt
├── requirements-rt.txt
├── pre-commit-config.yaml
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── Figures/
│   └── .gitkeep
├── scripts/
│   ├── run_synthetic.py
│   ├── run_ihdp.py
│   ├── make_figures.py
│   └── utils.py
└── sp_cci/
    ├── __init__.py
    ├── cci.py
    ├── spcci.py
    ├── metrics.py
    ├── data_synthetic.py
    ├── data_ihdp.py
    └── plotting.py
```

- **sp_cci/**: library code (CCI, SP-CCI, data generators, metrics)
- **scripts/**: CLIs to run experiments & render figures
- **data/**: place external datasets here (IHDP, etc.) — see instructions below
- **results/**, **outputs/**, **Figures/**: experiment logs, tables, and plots

---

## Installation

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Optional (wireless ray-tracing experiment scaffold):
```bash
pip install -r requirements-rt.txt
```

### 2) Pre-commit hooks (recommended)
```bash
pip install pre-commit
pre-commit install
```

---

## Datasets

### Synthetic (no download needed)
Generated on-the-fly

### IHDP (semi-synthetic)

### IHDP Dataset Source

This repo uses the widely adopted semi-synthetic IHDP dataset. Please cite the original study and the commonly used semi-synthetic construction:

- **Hill, J. (2011).** *Bayesian nonparametric modeling for causal inference.* Journal of Computational and Graphical Statistics, 20(1), 217–240.
- **Shalit, U., Johansson, F. D., & Sontag, D. (2017).** *Estimating individual treatment effect: Generalization bounds and algorithms.* In ICML 2017. (NPCI/CFR version of IHDP used broadly in CATE literature.)
Place the processed IHDP table(s) under:
```
data/ihdp/ihdp.csv
```
You can adapt `sp_cci/data_ihdp.py` to your specific file name/format.

> The code does **not** auto-download IHDP. Ensure you have rights to use the dataset.

### Wireless (optional; Sionna RT)
The wireless demo is scaffolded only. It requires TensorFlow + Sionna and a compatible CUDA setup.

---

## Usage

### 1) Synthetic experiment
```bash
python scripts/run_synthetic.py   --n 5000 --rho 0.0 --alpha 0.15 --delta 0.1   --runs 50 --quality HQ   --out results/synth_hq.json
```

### 2) IHDP experiment
```bash
python scripts/run_ihdp.py   --alpha 0.15 --delta 0.1 --runs 50   --quality MQ   --ihdp-path data/ihdp/ihdp.csv   --out results/ihdp_mq.json
```

### 3) Make figures
```bash
python scripts/make_figures.py   --inputs results/synth_hq.json results/ihdp_mq.json   --outdir Figures
```
This will produce PDFs like `Coverage.pdf` and `prediction_width.pdf`.

---

## Methods (short)

- **CCI** (weighted conformal prediction) constructs intervals for \(Y(1)\) using treated calibration data and propensity weighting.
- **SP-CCI** augments treated calibration with synthetic counterfactual labels \(\hat{Y}(1)\) from a learned generator; a PPI-style debiasing step combined with RCPS selects the minimal widening \(\eta\) s.t. an upper confidence bound on miscoverage is ≤ \(lpha\).


---

## Citation

If this repository helps your research, please cite the paper.

---

## License

MIT. See `LICENSE`.
