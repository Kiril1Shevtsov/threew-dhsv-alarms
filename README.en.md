# DHSV Alarms · 3W

**English documentation** · [Русская версия](README.md)

A reproducible study of event-level alarm threshold selection using oil-well data from **Petrobras 3W**. The task is to detect labelled **undesirable DHSV closure** events while assessing not only individual prediction windows but also the burden of false alarms.

This repository contains models, saved predictions, metric calculations and a small command-line interface for processing Parquet records. It is an independent research project by Kirill Shevtsov and Danila Litvinov, not an official Petrobras product.

## Main result

On the **development set of 45 records, 13 wells and 22 DHSV events**, event-level threshold selection for HistGradientBoosting reduced false alarm episodes from **55 to 3**, retaining **20 out of 22** detections within five minutes after the labelled event onset.

| HGB variant | Detected within ≤300 s | Detected with no false episodes anywhere in the record | False episodes | Episodes per day |
|---|---:|---:|---:|---:|
| Threshold 0.5 | 20/22 | 16/22 | 55 | 10.85 |
| Event-level threshold selection | 20/22 | 18/22 | 3 | 0.59 |

The normal-operation denominator is **121.675 hours**. Thresholds are selected within the training wells; the outer evaluation well is excluded from both training and threshold selection. These records were used during method development: the figures characterize this experimental protocol, not universal superiority over other solutions.

### Additional evaluation

The fixed models were evaluated on **15 other wells**, containing normal intervals and failure classes 5, 7 and 9. For HGB at threshold 0.999:

- **0 false episodes** over **52.221 hours** of labelled normal operation;
- **2 of 15** records produced a DHSV signal during a different failure;
- this set contains **0 DHSV events**. It therefore assesses false alarms and diagnostic specificity; the 20/22 detection result belongs to the development set above.

Normal intervals here come from records containing failures. The additional sample is restricted to files smaller than 1 MB, one per well, selected by the smallest Git blob SHA. These selection rules matter when comparing experiments. Zero observed false episodes does not imply zero operational risk.

[All variants and results](results/) · [Experimental protocol](docs/methodology.en.md) · [Data provenance](DATA_SOURCES.en.md)

## Quick start

Python 3.12. A separate virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/check_results.py
```

The last command recalculates 930 record–model–alarm-policy combinations from the included predictions. It requires no network access and does not train models.

### Predicting a record

```bash
python scripts/download_data.py
python scripts/check_data.py
python scripts/predict.py raw/dataset/5/WELL-00015_20170620122925.parquet --output predictions.csv
```

The output contains timestamps, model scores, thresholds and confirmed alarms. HGB and the event-level threshold are used by default. Other options include:

```bash
python scripts/predict.py path/to/record.parquet --model calibrated_RF --policy window_q99 --output predictions.csv
```

The input must have a chronological 1 Hz index and six sensors: `P-PDG`, `P-TPT`, `T-TPT`, `P-MON-CKP`, `T-JUS-CKP`, `QGL`. The first hour is used for normalization; predictions start afterwards. `score` is a classifier score, **not a calibrated probability of physical failure**.

### Retraining

```bash
python scripts/train.py --output runs/hgb
```

Random Forest models are supplied pre-trained: their predictions and event metrics can be reproduced, but this repository does not include a separate RF retraining script.

The script retrains HGB in an outer leave-one-well-out loop and selects thresholds through inner three-fold grouped evaluation. It uses the included causal feature matrix and writes new results separately from the published tables. A complete run takes substantially longer than checking saved predictions. The full HGB cycle was rerun with this script when preparing the repository: the event-level results matched the published tables. [Full-cycle log](results/hgb_training.log).

## How the alarm works

1. Features are computed from the six sensors over the preceding 180 seconds, with a 15-second stride.
2. The normalized variant uses the first-hour median and scale without labels.
3. The classifier produces a numerical score.
4. Three consecutive threshold exceedances confirm an alarm.
5. Evaluation measures alarm episodes, the fraction of normal time under alarm and detection delay. These are different characteristics; one alone is insufficient.

The event-level threshold is selected from a grid subject to limits on false episodes and time under alarm. These are empirical tuning constraints, not guarantees for future data.

## Repository layout

```text
src/                 Causal features, models and event-level metrics
scripts/             Data download, inference, training and result recalculation
tests/               Regression tests
models/              Saved models and thresholds
data/                Predictions, features, splits and source manifest
results/             Aggregate and per-well results
docs/                Experimental protocol
```

## Intended use

This code supports research and method comparison on historical data. It does not control equipment, replace safety systems or establish industrial readiness. Results should not be compared directly with another benchmark without aligning the data, splits and alarm definitions.

We are interested in useful next steps: other failure types, an agreed benchmark, robustness to sensor changes and joint assessment of missed-event and false-alarm costs.

## Licenses

Project code is licensed under **Apache-2.0**. The original 3W data and included derived tables/features are licensed under **CC BY 4.0**, with attribution to Petrobras and the dataset authors. Models are distributed under Apache-2.0; their training data are described in [DATA_SOURCES.en.md](DATA_SOURCES.en.md).

Raw Parquet files are not included: the downloader retrieves them from a precisely pinned 3W revision and checks their Git blob SHA. Do not load arbitrary third-party pickle/joblib files: deserialization can execute code.
