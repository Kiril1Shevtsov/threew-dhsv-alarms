# Experimental protocol

[English overview](../README.en.md) · [Русская версия](methodology.md)

## Data

3W version 2.0.0, commit `914af287b1cf3e5207dce0da1e9dc08361c9adeb`.

Development set: all 22 real DHSV records in the selected revision and 23 normal-operation records, covering 13 wells. Features use 180-second windows with a 15-second stride after the first hour. Labels, well identifiers and failure onset times are excluded from the features. Labels are used only for training and evaluation.

The outer loop excludes an entire well. Within the training portion, three-fold GroupKFold produces predictions for threshold selection. In the original experiment, saved outer predictions from the same classifier on the same 45 records were reused; inner models were trained for threshold selection. This is a development experiment, not a new untouched confirmatory set.

The additional set contains 15 records from 15 wells disjoint from development. Selection used classes 5, 7 and 9, files smaller than 1 MB, and the smallest Git blob SHA per well. Selection and final models/thresholds were fixed before computing results on this set. It contains normal intervals and other failure types, with no DHSV-positive events. Metrics for these two tasks are reported separately.

## Models and features

Random Forest: 200 trees, minimum leaf size 5, `max_features=sqrt`, balanced classes. HistGradientBoosting: 150 iterations, up to 15 leaves, learning rate 0.1, L2 regularization 1.0. Seed: 20261109.

For each sensor: mean, standard deviation, minimum, maximum, last value, change across the window and missing-value fraction. The `calibrated` variant name means **sensor normalization**, not probability calibration. Missing values are filled using training-set medians, with additional missingness indicators. First-hour normalization is computed without labels; predictions begin after that hour.

## Event-level alarm policy

Three consecutive threshold exceedances, 15 seconds apart, confirm an alarm. A false episode is a contiguous period of confirmed alarm within labelled normal operation in one record. “Detection within 300 seconds” is measured from the start of the transition to a positive label; it does not mean prediction 300 seconds before the event.

“Clean detection” additionally requires no false episodes anywhere in the record, including intervals after the event, as defined by `clean_300`. For a record that starts already in failure, detection delay is not equated with elapsed time from the beginning of the file: that onset is treated as censored. Any signal during another failure type is recorded separately.

The policies compared are a fixed threshold of 0.5, a threshold based on the 99th percentile of negative scores, and event-level selection with budgets of 1, 2 or 6 false episodes per day and normal-time alarm occupancy ≤1%. Among eligible candidates, selection maximizes clean detections within 300 seconds, then ordinary detections; it next minimizes false episodes and chooses the higher threshold. A threshold above 1 disables alarms; it does not signify detector success.

## Reproducibility

`data/development_features.npz` contains the `raw` and `calibrated` feature matrices; row order matches `data/development.csv`. `data/thresholds.csv` contains the outer threshold for each model, policy and excluded well. `data/splits.json` describes inner splits. `models/` contains final models trained on development data; `original_*` variants are controls trained only on the 22 DHSV records.

`check_results.py` recalculates event tables from predictions. `predict.py` applies saved models to original Parquet files. `train.py` reruns the nested HGB loop; unlike the original calculation, it recomputes outer predictions itself. Retraining does not use the additional set for tuning. RF models are supplied pre-trained; their retraining script is not included.

The historical dataset contains correlated windows. They must not be treated as independent trials for confidence intervals. Historical interpolation to 1 Hz does not establish actual sensor-arrival latency in an industrial system.
