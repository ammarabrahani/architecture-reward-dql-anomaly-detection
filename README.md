# architecture-reward-dql-anomaly-detection
Architecture and reward function of Deep Q-Learning algorithm for anomaly detection

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

All notebooks live in [`notebooks/`](notebooks/):

| Notebook | Dataset |
| --- | --- |
| `01_synthetic_dql_ae.ipynb` | Synthetic data (generated in-notebook, no download needed) |
| `02_kaggle_credit_card_ae.ipynb` | Kaggle Credit Card Fraud |
| `03_unsw_nb15_dql_ae.ipynb` | UNSW-NB15 |
| `04_ton_iot_dql_ae.ipynb` | TON_IoT |
| `05_kaggle_dql_with_seeds.ipynb` | Kaggle Credit Card Fraud (seeded reward comparison) |

## Data

Datasets are not committed to this repo (size/licensing). Download the following into `data/raw/` before running the corresponding notebook:

- **Kaggle Credit Card Fraud** — [kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) → `data/raw/creditcard.csv`
- **UNSW-NB15** — [research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) → `data/raw/UNSW_NB15_training-set.parquet`, `data/raw/UNSW_NB15_testing-set.parquet`
- **TON_IoT** — [research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets) → `data/raw/train_test_network.csv`
