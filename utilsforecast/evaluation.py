# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/evaluation.ipynb.

# %% auto 0
__all__ = ['evaluate']

# %% ../nbs/evaluation.ipynb 3
import inspect
import re
import reprlib
from typing import Callable, Dict, List, Optional, get_origin

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp
from .compat import DataFrame, pl

# %% ../nbs/evaluation.ipynb 4
def _function_name(f: Callable):
    if hasattr(f, "func"):
        # partial fn
        name = f.func.__name__
    else:
        name = f.__name__
    return name


def _quantiles_from_levels(level) -> np.ndarray:
    """Returns quantiles associated to `level` and the sorte columns of `model_name`"""
    level = sorted(level)
    alphas = [100 - lv for lv in level]
    quantiles = [alpha / 200 for alpha in reversed(alphas)]
    quantiles.extend([1 - alpha / 200 for alpha in alphas])
    return np.array(quantiles)


def _models_from_levels(model_name, level) -> List[str]:
    cols = [f"{model_name}-lo-{lv}" for lv in reversed(level)]
    cols.extend([f"{model_name}-hi-{lv}" for lv in level])
    return cols

# %% ../nbs/evaluation.ipynb 5
def evaluate(
    df: DataFrame,
    metrics: List[Callable],
    models: Optional[List[str]] = None,
    train_df: Optional[DataFrame] = None,
    level: Optional[List[int]] = None,
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
    level : list of int, optional (default=None)
        Prediction interval levels. Used to compute losses that rely on quantiles.
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

    # interval cols
    if level is not None:
        expected_cols = {
            f"{m}-{side}-{lvl}"
            for m in model_cols
            for side in ("lo", "hi")
            for lvl in level
        }
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"The following columns are required for level={level} "
                f"and are missing: {missing}"
            )
    else:
        requires_level = [
            m
            for m in metrics
            if get_origin(inspect.signature(m).parameters["models"].annotation) is dict
        ]
        if requires_level:
            raise ValueError(
                f"The following metrics require setting `level`: {requires_level}"
            )

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
        metric_params = inspect.signature(metric).parameters
        if "q" in metric_params or metric_params["models"].annotation is Dict[str, str]:
            for lvl in level:
                quantiles = _quantiles_from_levels([lvl])
                for q, side in zip(quantiles, ["lo", "hi"]):
                    kwargs["models"] = {
                        model: f"{model}-{side}-{lvl}" for model in model_cols
                    }
                    if "q" in metric_params:
                        # this is for calibration, since it uses the predictions for q
                        # but doesn't use it
                        kwargs["q"] = q
                    result = metric(**kwargs)
                    result = ufp.assign_columns(result, "metric", f"{metric_name}_q{q}")
                    results_per_metric.append(result)
        elif "quantiles" in metric_params:
            quantiles = _quantiles_from_levels(level)
            kwargs["quantiles"] = quantiles
            kwargs["models"] = {
                model: _models_from_levels(model, level) for model in model_cols
            }
            result = metric(**kwargs)
            result = ufp.assign_columns(result, "metric", metric_name)
            results_per_metric.append(result)
        elif "level" in metric_params:
            for lvl in level:
                kwargs["level"] = lvl
                result = metric(**kwargs)
                result = ufp.assign_columns(
                    result, "metric", f"{metric_name}_level{lvl}"
                )
                results_per_metric.append(result)
        else:
            result = metric(**kwargs)
            result = ufp.assign_columns(result, "metric", metric_name)
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
