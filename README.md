# RobotOps Copilot — Predictive Maintenance Model

A machine learning pipeline for predicting equipment failures using the **AI4I 2020 Predictive Maintenance Dataset** (UCI). The primary model is **XGBoost** with optional **TabPFN** support.

## Overview

This project builds a failure prediction model that:

1. **Loads** the AI4I 2020 dataset automatically via `ucimlrepo`
2. **Engineers features** — temperature differential, mechanical power, wear-torque interaction, and binary risk flags
3. **Trains** an XGBoost classifier (or TabPFN if configured) with class imbalance handling
4. **Evaluates** performance with ROC-AUC, classification report, confusion matrix, and risk score distribution
5. **Explains** predictions using SHAP beeswarm plots
6. **Ranks** machines by failure risk into tiers (Critical / High / Medium / Low)

## Installation

```bash
pip install xgboost ucimlrepo scikit-learn shap pandas numpy matplotlib
```

### Optional: TabPFN

TabPFN is a gated model on HuggingFace. To enable it:

1. Run `hf auth login`
2. Accept terms at https://huggingface.co/Prior-Labs/tabpfn_2_5
3. Uncomment the TabPFN import block in `predictive_maintenance_tabpfn.py`

## Usage

```bash
python predictive_maintenance_tabpfn.py
```

## Outputs

| File | Description |
|------|-------------|
| `maintenance_evaluation.png` | Confusion matrix and risk score distribution plot |
| `shap_beeswarm.png` | SHAP feature importance plot |
| `maintenance_predictions.csv` | Per-machine failure risk predictions with risk tiers |

## Engineered Features

| Feature | Description |
|---------|-------------|
| `temp_diff` | Process temperature minus air temperature |
| `power_kw` | Mechanical power derived from torque and RPM |
| `wear_torque` | Interaction between tool wear and torque |
| `high_torque` | Binary flag for torque > 50 Nm |
| `low_speed` | Binary flag for RPM < 1380 |
| `wear_high` | Binary flag for tool wear > 180 min |

## Risk Tiers

| Tier | Threshold |
|------|-----------|
| CRITICAL | >= 80% failure probability |
| HIGH | >= 50% |
| MEDIUM | >= 25% |
| LOW | < 25% |

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository, ID 601) — 10,000 data points with 6 features reflecting real-world industrial conditions. The dataset is downloaded automatically on first run.
