# Credit Card Fraud Detection

Anomaly detection and machine learning pipeline for identifying fraudulent credit card
transactions, built on the Kaggle Credit Card Fraud Detection dataset
(284,807 European cardholder transactions, 492 labeled as fraud).

📦 **Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
*(not included in this repo — file is ~150 MB; see [Setup](#setup) below to download it)*

## Overview

Fraud makes up just **0.17%** of transactions in this dataset, so the core challenge is
extreme class imbalance — a naive "always legitimate" model scores 99.8% accuracy while
catching zero fraud. This project handles that with SMOTE oversampling and class-weighting,
trains three models, and evaluates them on metrics that actually matter for rare-event
detection (Precision-Recall AUC rather than accuracy).

**Best result:** XGBoost — 88.6% precision, 78.4% recall, PR-AUC 0.835.

## Project Structure

```
fraud_detection/
│
├── creditcard.csv          # Input data (not included — download separately, see below)
├── 01_eda.py                # Exploratory data analysis + plots
├── 02_modeling.py           # Feature engineering, training, evaluation, plots
├── results.json             # Saved metrics for all models (generated on run)
├── plots/                   # Generated figures (created on run)
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### 2. Get the data

> **Note:** `creditcard.csv` (~150 MB) is **not included in this repo** due to its size.
> Download it separately from Kaggle:
>
> 🔗 **https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**

After downloading, place `creditcard.csv` in the project root (same folder as the scripts).
You'll need a free Kaggle account to download it; if you have the Kaggle CLI set up, you can
alternatively run:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p . --unzip
```

### 3. Create the output folder

```bash
mkdir -p plots
```

## Usage

Run the EDA script first, then the modeling script:

```bash
python 01_eda.py
python 02_modeling.py
```

- `01_eda.py` generates 4 exploratory plots (class imbalance, amount distribution, time-of-day
  patterns, feature correlation) in `plots/`.
- `02_modeling.py` trains all three models, prints metrics to the console, saves them to
  `results.json`, and generates 5 evaluation plots (PR curves, ROC curves, confusion matrices,
  feature importance, threshold tradeoff) in `plots/`.

## Methodology

| Step | Approach |
|---|---|
| Feature engineering | Standardize `Amount` and `Time`; derive `Hour` of day |
| Train/test split | Stratified 70/30 split, preserving the fraud ratio in both sets |
| Class imbalance | SMOTE oversampling (training set only, to 10% fraud) + class-weighting |
| Models | Logistic Regression, Random Forest, XGBoost |
| Evaluation | PR-AUC (primary), ROC-AUC, Precision, Recall, F1, confusion matrix |

## Results

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.718 | 0.965 | 0.331 | 0.831 | 0.473 |
| Random Forest | 0.784 | 0.977 | 0.598 | 0.804 | 0.686 |
| **XGBoost** | **0.835** | 0.974 | **0.886** | 0.784 | **0.832** |

All metrics computed on a held-out test set (30% of data, ~85,443 transactions, 148 fraud
cases) never used during training.

XGBoost is the recommended model: it catches 78% of fraud while keeping false positives to
just 15 out of 85,295 legitimate transactions — the best precision/recall balance of the
three, which matters most for production systems where false alarms cost analyst time and
customer trust.

The most predictive features (consistent across Random Forest and XGBoost importance
rankings, and correlation analysis) are **V14, V17, V12, V10, and V4**.

## Notes & Limitations

- The dataset covers only two days of transactions, so seasonal/longer-term fraud patterns
  aren't captured.
- Features `V1`–`V28` are PCA-anonymized, limiting interpretability of *why* a transaction is
  flagged.
- SMOTE synthesizes fraud examples by interpolation, which can occasionally generate
  unrealistic samples — worth comparing against anomaly-detection approaches (Isolation
  Forest, Autoencoders) as a next step.

## Next Steps

- Explore unsupervised anomaly detection (Isolation Forest, Autoencoders, One-Class SVM) as a
  complementary layer that doesn't require labeled fraud.
- Incorporate the real dollar cost of false negatives vs. false positives to pick an optimal
  decision threshold (see `plots/09_threshold_tradeoff.png`).
- Deploy as a streaming scoring service with a feedback loop to capture confirmed
  fraud/false-positive labels for periodic retraining.
