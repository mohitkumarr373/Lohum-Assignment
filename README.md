
# LOHUM Problem Statement — Battery Analysis

This repository implements the full pipeline for the LOHUM Summer Intern assignment:

1. Identify **Feature 10** as an electrical quantity derived from voltage/current.
2. Perform **within-OEM** and **across-OEM** comparative analysis.
3. Build predictive models for **State of Health (SOH)** and **State of Power (SOP)** for a selected OEM with 5-fold cross-validation.

## Repository Structure

- `data/`
  - `OEM1/` — 10 text files for OEM 1
  - `OEM2/` — 10 text files for OEM 2
  - `OEM3/` — 10 text files for OEM 3
- `data_loader.py` — utilities to load and combine raw `.txt` files.
- `feature_engineering.py` — target computation (SOH, SOP) and predictive feature extraction.
- `soh_sop_model.py` — training + evaluation script for SOH and SOP models.
- `Analysis_Report.pdf` — written report explaining methodology and findings.
- `models/` — saved model artifacts (`SOH_model.pkl`, `SOP_model.pkl`) after I run training.
- `metrics.csv` — model performance metrics (generated after training).
- `requirements.txt` — Python dependencies.

## How to Place the Data

Create the following structure in the repository root:

```text
data/
  OEM1/  # first folder of 10 text files
  OEM2/  # second folder of 10 text files
  OEM3/  # third folder of 10 text files
```


## Setup

```bash
pip install -r requirements.txt
```

## Running the Models

By default, the script trains on OEM 2 (can be changed via the `oem_id` argument):

```bash
python soh_sop_model.py
```

Or explicitly:

```bash
python -m soh_sop_model
```

The script will:

1. Load OEM data using `data_loader.py`.
2. Compute targets (SOH, SOP) using `feature_engineering.py`.
3. Extract early-discharge predictive features.
4. Train Random Forest models for SOH and SOP with 5-fold cross-validation.
5. Save:
   - `models/SOH_model.pkl`
   - `models/SOP_model.pkl`
   - `metrics.csv` with RMSE, MAE, MAPE for each target.
