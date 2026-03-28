# TabPFN Test Set - Project Notes

## Overview
A test project for evaluating **TabPFN** (Tabular Prior-Fitted Network) — a transformer-based model for tabular classification — on a robot sensor navigation dataset.

## Repository
- **Branch:** `main`
- **Single commit:** `a57450a` — initial push
- **Python:** 3.13.5 (via Anaconda)
- **Virtual env:** `.venv/`

## Files

### `model.py`
Main script that:
1. Loads `sensor_readings_24.data` using pandas
2. Splits into train/test (80/20, stratified)
3. Caps training set at **1,000 rows** for Mac MPS memory stability
4. Trains a `TabPFNClassifier` (from the `tabpfn` package)
5. Predicts in **batches of 10** to avoid MPS out-of-memory
6. Reports accuracy

Key env settings:
- `PYTORCH_ENABLE_MPS_FALLBACK = "1"` (Apple Silicon fallback)
- `CUDA_VISIBLE_DEVICES = ""` (disables CUDA)

### `sensor_readings_24.data`
- **Format:** CSV (no header), 5,456 rows x 25 columns
- **Features:** 24 sensor readings (`sensor_1` through `sensor_24`), all float values
- **Target:** Navigation direction (last column)
- **Class distribution:**
  - `Move-Forward` — 2,205
  - `Sharp-Right-Turn` — 2,097
  - `Slight-Right-Turn` — 826
  - `Slight-Left-Turn` — 328
- **Source:** Appears to be the UCI "Wall-Following Robot Navigation" dataset

## Key Dependencies
- `tabpfn` — TabPFN classifier
- `scikit-learn` — train/test split, accuracy scoring
- `pandas`, `numpy` — data handling
- `torch` — PyTorch backend (used by TabPFN)
- `huggingface_hub` — model download

## Notes
- The project is configured for **Apple Silicon (MPS)** with fallbacks and small batch sizes to manage memory
- Training is limited to 1,000 samples; can be increased if memory allows
- The dataset is imbalanced (Move-Forward and Sharp-Right-Turn dominate)
