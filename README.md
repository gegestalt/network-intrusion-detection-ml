# NSL-KDD Network Intrusion Detection — an ML study

An end-to-end machine-learning study applying classical and deep models to
**network intrusion detection** on the **NSL-KDD** benchmark, framed from a SOC
(Security Operations Centre) analyst's perspective.

> **Status:** work in progress. Phase 0 (setup + data) complete. This README is
> expanded into the full write-up (problem statement, results, limitations) in
> Phase 5.

## Project layout

```
nsl-kdd-ids/
├── data/                # NSL-KDD files (gitignored; fetched by script)
│   └── SOURCE.md        # provenance: source URLs + SHA-256 hashes
├── notebooks/           # exploratory analysis (Phase 1)
├── src/                 # reusable, tested pipeline + training code
│   └── download_data.py # programmatic, verified dataset download
├── results/
│   ├── figures/         # saved plots
│   └── metrics.md       # model comparison tables
├── .claude/skills/      # project workflow conventions (ids-ml-workflow)
├── requirements.txt     # pinned dependencies (Python 3.14)
└── README.md
```

## Quick start (reproduce)

```bash
# 1. create + activate the environment
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# macOS only: XGBoost/LightGBM need OpenMP ->  brew install libomp

# 2. install pinned dependencies
pip install -r requirements.txt

# 3. download + verify the dataset (~22 MB)
python src/download_data.py
```

## Dataset

**NSL-KDD** — Canadian Institute for Cybersecurity, University of New Brunswick.
A de-duplicated revision of KDD Cup '99. We use the **official**
`KDDTrain+` / `KDDTest+` split (see *Approach* — the official split is
deliberately hard because the test set contains attack types unseen in
training). Full provenance and hashes: [`data/SOURCE.md`](data/SOURCE.md).

## Approach (roadmap)

| Phase | Contents |
| --- | --- |
| 0 | Repo, env, pinned deps, verified data download ✅ |
| 1 | EDA: class balance, attack families, correlations, plots |
| 2 | Preprocessing pipeline (`ColumnTransformer`); binary + 5-class labels |
| 3 | Random Forest baseline + tuning + feature importance |
| 4 | MLP (PyTorch) comparison, same evaluation protocol |
| 5 | Full documentation, limitations, next steps |

## Why PyTorch (not TensorFlow/Keras)

PyTorch is the dominant framework in current ML research and gives explicit,
readable control over the training loop, dropout, and early stopping — which
reads well in a portfolio and makes the RF-vs-MLP comparison transparent. It
also installed cleanly on Python 3.14 with Apple-Silicon **MPS** GPU support.

## License / attribution

Code: MIT (see `LICENSE`, added in Phase 5). Dataset © CIC/UNB — cite Tavallaee
et al., *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA 2009.
