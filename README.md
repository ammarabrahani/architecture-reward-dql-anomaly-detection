# Architecture and reward function of Deep Q-Learning algorithm for anomaly detection — Research Code

**Paper:** Architecture and reward function of Deep Q-Learning algorithm for anomaly detection  
**Authors:** Ammar Yousuf Abrahani, Giovani Estrada  
**Institution:** National College of Ireland, Dublin

---

## Repository Structure

```
📁 notebooks/
├── 01_synthetic_dql_ae.ipynb          # Synthetic dataset experiments
├── 02_kaggle_credit_card_ae.ipynb     # Kaggle Credit Card — Full pipeline
├── 03_unsw_nb15_dql_ae.ipynb          # UNSW-NB15 — Full pipeline
├── 04_ton_iot_dql_ae.ipynb            # TON_IoT — Full pipeline
└── 05_kaggle_dql_with_seeds.ipynb     # Kaggle Credit Card — earlier draft, superseded by 02
📁 data/
├── creditcard.csv                      # Kaggle Credit Card (download separately)
├── UNSW_NB15_training-set.parquet     # UNSW-NB15 training set
├── UNSW_NB15_testing-set.parquet      # UNSW-NB15 testing set
└── train_test_network.csv             # TON_IoT dataset
```

---

## Datasets

| Dataset | Source | Records | Features | Anomaly % |
|---|---|---|---|---|
| Synthetic | Scikit-learn (`make_classification`) | 5,000 | 20 | 15% |
| Kaggle Credit Card | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 | 29 | 0.17% |
| UNSW-NB15 | [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) | 257,673 | 32 | 63.91% |
| TON_IoT | [UNSW](https://research.unsw.edu.au/projects/toniot-datasets) | 211,043 | 33 | 76.31% |

> **Note:** Anomaly % reflects train/test split files. All nine attack categories are grouped into a single anomaly label against normal traffic.

---

## Notebook Contents

### `01_synthetic_dql_ae.ipynb`
- Simple Autoencoder on synthetic datasets
- DQL convergence study (20 episodes)
- 5-seed statistical validation
- Symmetric vs Asymmetric reward comparison

### `02_kaggle_credit_card_ae.ipynb` ⭐ Complete Pipeline
- **Step 1:** Imports
- **Step 2:** Load & Preprocess (`creditcard.csv`)
- **Step 3:** Isolation Forest → Table VII
- **Step 4:** AE Simple/Wide/Deep comparison → **Table II**
- **Step 5:** DQL Simple/Wide/Deep comparison → **Table III**
- **Step 6:** DQL Convergence Plot → **Fig 4**
- **Step 7:** 5-Seed Statistical Validation → **Table VI**
- **Step 8:** Symmetric vs Asymmetric Reward → **Table IV**
- **Step 9:** Extended Metrics (FPR, ROC-AUC, PR-AUC) → **Table V**
- **Step 10:** Final Summary → **Table VII**

### `03_unsw_nb15_dql_ae.ipynb` ⭐ Complete Pipeline
- Isolation Forest, Simple/Wide/Deep AE, DQL → Tables II, III
- 5-Seed validation → Table VI
- Symmetric vs Asymmetric reward → Table IV
- Extended metrics (FPR, ROC-AUC, PR-AUC) → Table V

### `04_ton_iot_dql_ae.ipynb` ⭐ Complete Pipeline
- Isolation Forest, Simple/Wide/Deep AE, DQL → Tables II, III
- Extended metrics → Table V
- Symmetric vs Asymmetric reward → Table IV

### `05_kaggle_dql_with_seeds.ipynb`
Earlier iteration of the Kaggle Credit Card pipeline, kept for reference. Superseded by `02_kaggle_credit_card_ae.ipynb`, which adds the extended-metrics step (Table V) and consolidated final summary.

---

## Key Results

### Table II — Autoencoder Architecture Comparison (Macro F1-Score)

| Architecture | Synthetic | Kaggle | UNSW-NB15 | TON_IoT |
|---|---|---|---|---|
| Simple AE | 0.489 | 0.488 | 0.420 | 0.313 |
| Wide AE | 0.521 | 0.489 | 0.421 | 0.322 |
| **Deep AE** | **0.922** | **0.490** | 0.407 | 0.319 |

### Table III — DQL Architecture Comparison (Macro F1-Score)

| Architecture | Synthetic | Kaggle | UNSW-NB15 | TON_IoT |
|---|---|---|---|---|
| DQL Simple | 0.910 | 0.524 | 0.806 | 0.916 |
| DQL Wide | 0.915 | 0.553 | **0.858** | 0.983 |
| **DQL Deep** | **0.920** | 0.574 | 0.859 | **0.984** |

### Table IV — Symmetric vs Asymmetric Reward (DQL Deep)

| Dataset | Symmetric | Asymmetric | Improvement |
|---|---|---|---|
| Synthetic | 0.867 | 0.853 | -1.61% |
| Kaggle CC | 0.607 | 0.552 | -9.08% |
| TON_IoT | 0.969 | 0.975 | +0.58% |
| UNSW-NB15 | 0.831 | 0.852 | +2.56% |

### Table VI — DQL Deep, 5-Seed Mean ± Std Dev

| Dataset | Macro F1 | Precision | Recall |
|---|---|---|---|
| Synthetic | 0.857 ± 0.013 | 0.821 ± 0.011 | 0.915 ± 0.018 |
| Kaggle CC | 0.605 ± 0.080 | 0.574 ± 0.070 | 0.921 ± 0.043 |
| UNSW-NB15 | 0.854 ± 0.007 | 0.863 ± 0.018 | 0.850 ± 0.005 |
| TON_IoT | 0.970 ± 0.010 | 0.968 ± 0.010 | 0.973 ± 0.015 |

### Table VII — Full Model Comparison (Macro F1-Score)

| Model | Synthetic | Kaggle | UNSW-NB15 | TON_IoT |
|---|---|---|---|---|
| Isolation Forest | 0.934 | **0.661** | 0.318 | 0.281 |
| Deep AE | 0.922 | 0.490 | 0.404 | 0.319 |
| DQL Wide | 0.915 | 0.550 | **0.860** | 0.983 |
| **DQL Deep** | **0.920** | 0.574 | 0.859 | **0.984** |

---

## Requirements

```
tensorflow >= 2.10
scikit-learn >= 1.0
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
pyarrow >= 10.0  (for .parquet files)
```

Install:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib pyarrow
```

---

## How to Run

1. Clone/download the repository
2. Place dataset files in `data/` folder
3. Run notebooks in order (01 → 04); `05` is an earlier draft kept for reference and is not required
4. Each notebook is self-contained — run all cells sequentially

> **Important:** For Kaggle notebook, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place in same folder.

---

## Citation

If you use this code, please cite:

```
@inproceedings{abrahani2026dql,
  title={Architecture and reward function of Deep Q-Learning algorithm for anomaly detection},
  author={Abrahani, Ammar Yousuf and Estrada, Giovani},
  booktitle={Proc. 13th International Conference on Soft Computing and Machine Intelligence (ISCMI)},
  year={2026},
  address={Vienna, Austria}
}
```

---

## Disclaimer

This code was developed for academic research purposes at the National College of Ireland. Please cite properly if reused.
