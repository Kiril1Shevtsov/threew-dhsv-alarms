# Data sources and rights

[English overview](README.en.md) · [Русская версия](DATA_SOURCES.md)

Source: **Petróleo Brasileiro S.A. (Petrobras), The 3W Project**.

- Repository: https://github.com/petrobras/3W
- Data version: 2.0.0.
- Pinned revision: `914af287b1cf3e5207dce0da1e9dc08361c9adeb`.
- Upstream licenses: https://github.com/petrobras/3W#licenses
- Original Parquet files: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Upstream code: Apache-2.0. The training and analysis code included here was written separately; 3W Toolkit code is not included.

Derived feature matrices, predictions, metadata and numerical tables in `data/` and `results/` are distributed under CC BY 4.0. Transformations include record and time-window selection, normalization, feature and prediction computation, and event-level metric aggregation. Raw Parquet files are not modified during download. The file list and original Git blob SHAs are recorded in `data/manifest.json`.

Source citation:

Vaz Vargas, R.E., de Melo Junior, A.J., de Campos Lima, C.B. et al.
*3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells.* Scientific Data 13, 949 (2026).
https://doi.org/10.1038/s41597-026-07225-z

Dataset description: https://arxiv.org/abs/2507.01048

Petrobras and the dataset authors have not endorsed this project. The models and results produced here are not Petrobras results.

Dependencies are installed separately and are not vendored: NumPy (BSD), pandas (BSD), SciPy (BSD), scikit-learn (BSD), joblib (BSD), Apache Arrow/pyarrow (Apache-2.0). Consult the respective project distributions for exact license terms.
