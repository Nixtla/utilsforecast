# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/evaluation.ipynb.

# %% auto 0
__all__ = ['evaluate']

# %% ../nbs/evaluation.ipynb 3
import inspect
import re
import reprlib
from typing import Callable, List, Optional

import pandas as pd

from .compat import DataFrame, pl
from .utils import ensure_dtypes

# %% ../nbs/evaluation.ipynb 4
def _function_name(f: Callable):
    if hasattr(f, "func"):
        # partial fn
        name = f.func.__name__
    else:
        name = f.__name__
    return name

# %% ../nbs/evaluation.ipynb 5
@ensure_dtypes("df")
def evaluate(
    df: DataFrame,
    metrics: List[Callable],
    models: Optional[List[str]] = None,
    train_df: Optional[DataFrame] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> DataFrame:
    """Evaluate forecast using different metrics.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Forecasts to evaluate.
        Must have `id_col`, `time_col`, `target_col` and models' predictions.
    metrics : list of callable
        Functions with arguments `df`, `models`, `id_col`, `target_col` and optionally `train_df`.
    models : list of str, optional (default=None)
        Names of the models to evaluate.
        If `None` will use every column in the dataframe after removing id, time and target.
    train_df : pandas DataFrame, optional (default=None)
        Training set. Used to evaluate metrics such as `mase`.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestep, its values can be timestamps or integers.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars DataFrame
        Metrics with one row per (id, metric) combination and one column per model.
    """
    if models is None:
        model_cols = [
            c
            for c in df.columns
            if c not in [id_col, time_col, target_col]
            and not re.search(r"-(?:lo|hi)-\d+", c)
        ]
    else:
        model_cols = models

    # y_train
    metric_requires_y_train = {
        _function_name(m): "train_df" in inspect.signature(m).parameters
        for m in metrics
    }
    y_train_metrics = [
        m for m, requires_yt in metric_requires_y_train.items() if requires_yt
    ]
    if y_train_metrics:
        if train_df is None:
            raise ValueError(
                f"The following metrics require y_train: {y_train_metrics}. "
                "Please provide `train_df`."
            )
        if isinstance(train_df, pd.DataFrame):
            train_df = train_df.sort_values([id_col, time_col])
        else:
            train_df = train_df.sort([id_col, time_col])
        missing_series = set(df[id_col].unique()) - set(train_df[id_col].unique())
        if missing_series:
            raise ValueError(
                f"The following series are missing from the train_df: {reprlib.repr(missing_series)}"
            )

    results_per_metric = []
    for metric in metrics:
        metric_name = _function_name(metric)
        kwargs = dict(df=df, models=model_cols, id_col=id_col, target_col=target_col)
        if metric_requires_y_train[metric_name]:
            kwargs["train_df"] = train_df
        result = metric(**kwargs)
        if isinstance(result, pd.DataFrame):
            result["metric"] = metric_name
        else:
            result = result.with_columns(pl.lit(metric_name).alias("metric"))
        results_per_metric.append(result)
    if isinstance(df, pd.DataFrame):
        df = pd.concat(results_per_metric).reset_index(drop=True)
        out_cols = [c for c in df.columns if c not in (id_col, "metric")]
        df = df[[id_col, "metric", *out_cols]]
    else:
        df = pl.concat(results_per_metric, how="diagonal")
        out_cols = [c for c in df.columns if c not in (id_col, "metric")]
        df = df.select([id_col, "metric", *out_cols])
    return df
