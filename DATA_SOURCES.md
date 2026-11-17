# Источники и права

Источник: **Petróleo Brasileiro S.A. (Petrobras), The 3W Project**.

- Репозиторий: https://github.com/petrobras/3W
- Версия данных: 2.0.0.
- Зафиксированная ревизия: `914af287b1cf3e5207dce0da1e9dc08361c9adeb`.
- Лицензии upstream: https://github.com/petrobras/3W#licenses
- Исходные Parquet: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Код upstream: Apache-2.0. Включённый код обучения и анализа написан отдельно; код 3W Toolkit не включён.

Производные матрицы признаков, прогнозы, метаданные и численные таблицы в `data/` и `results/` распространяются под CC BY 4.0. Преобразования: выбор записей и временных окон, нормировка, вычисление признаков и прогнозов, агрегация событийных метрик. Сырые Parquet при загрузке не изменяются. Список файлов и исходные Git blob SHA находится в `data/manifest.json`.

Цитирование источника:

Vaz Vargas, R.E., de Melo Junior, A.J., de Campos Lima, C.B. et al.
*3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells.* Scientific Data 13, 949 (2026).
https://doi.org/10.1038/s41597-026-07225-z

Описание данных: https://arxiv.org/abs/2507.01048

Petrobras и авторы датасета не заявляли одобрения данного проекта. Полученные здесь модели и результаты не являются результатами Petrobras.

Зависимости устанавливаются отдельно, не включены в репозиторий: NumPy (BSD), pandas (BSD), SciPy (BSD), scikit-learn (BSD), joblib (BSD), Apache Arrow/pyarrow (Apache-2.0). Для точных условий см. дистрибутивы соответствующих проектов.
