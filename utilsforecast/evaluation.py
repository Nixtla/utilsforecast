"""Model performance evaluation"""

__all__ = ['evaluate']


import inspect
import re
import reprlib
from typing import Callable, Dict, List, Optional, get_origin

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp

from .compat import AnyDFType, DFType, DistributedDFType, pl, pl_DataFrame
from .losses import _get_group_cols


def _function_name(f: Callable):
    if hasattr(f, "func"):
        # partial fn
        name = f.func.__name__
    else:
        name = f.__name__
    return name


def _quantiles_from_levels(level: List[int]) -> np.ndarray:
    """Returns quantiles associated to `level` and the sorte columns of `model_name`"""
    level = sorted(level)
    alphas = [100 - lv for lv in level]
    quantiles = [alpha / 200 for alpha in reversed(alphas)]
    quantiles.extend([1 - alpha / 200 for alpha in alphas])
    return np.array(quantiles)


def _models_from_levels(model_name: str, level: List[int]) -> List[str]:
    level = sorted(level)
    cols = [f"{model_name}-lo-{lv}" for lv in reversed(level)]
    cols.extend([f"{model_name}-hi-{lv}" for lv in level])
    return cols


def _get_model_cols(
    cols: List[str],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
) -> List[str]:
    return [
        c
        for c in cols
        if c not in [id_col, time_col, target_col, cutoff_col]
        and not re.search(r"-(?:lo|hi)-\d+", c)
    ]


def _evaluate_wrapper(
    df: pd.DataFrame,
    metrics: List[Callable],
    models: Optional[List[str]],
    level: Optional[List[int]],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
    agg_fn: Optional[str],
) -> pd.DataFrame:
    group_cols = _get_group_cols(df, id_col, cutoff_col)
    if "_in_sample" in df:
        in_sample_mask = df["_in_sample"]
        train_df = df.loc[in_sample_mask, [*group_cols, time_col, target_col]]
        df = df.loc[~in_sample_mask].drop(columns="_in_sample")
    else:
        train_df = None
    return evaluate(
        df=df,
        metrics=metrics,
        models=models,
        train_df=train_df,
        level=level,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        agg_fn=agg_fn,
    )


def _distributed_evaluate(
    df: DistributedDFType,
    metrics: List[Callable],
    models: Optional[List[str]],
    train_df: Optional[DFType],
    level: Optional[List[int]],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
    agg_fn: Optional[str],
) -> DistributedDFType:
    import fugue.api as fa

    if agg_fn is not None:
        raise ValueError("`agg_fn` is not supported in distributed")
    df_cols = fa.get_column_names(df)
    group_cols: list[str] = _get_group_cols(df, id_col, cutoff_col)
    if train_df is not None:
        # align columns in order to vstack them
        def assign_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
            return df.assign(**cols)
        train_cols = [*group_cols, time_col, target_col]
        extra_cols = [c for c in df_cols if c not in train_cols]
        train_df = fa.select_columns(train_df, train_cols)
        train_df = fa.transform(
            train_df,
            using=assign_cols,
            schema=(
                "*," + str(fa.get_schema(df).extract(extra_cols)) + ",_in_sample:bool"
            ),
            params={
                "cols": {
                    **{c: float("nan") for c in extra_cols},
                    "_in_sample": True,
                },
            },
        )
        df = fa.transform(
            df,
            using=assign_cols,
            schema="*,_in_sample:bool",
            params={"cols": {"_in_sample": False}},
        )
        df = fa.union(train_df, df)

    if models is None:
        model_cols = _get_model_cols(df_cols, id_col, time_col, target_col, cutoff_col)
    else:
        model_cols = models
    models_schema = ",".join(f"{m}:double" for m in model_cols)
    result_schema = fa.get_schema(df).extract(*group_cols) + "metric:str" + models_schema
    return fa.transform(
        df,
        using=_evaluate_wrapper,
        schema=result_schema,
        params=dict(
            metrics=metrics,
            models=models,
            level=level,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
            agg_fn=agg_fn,
        ),
        partition={"by": group_cols, "algo": "coarse"},
    )


def evaluate(
    df: AnyDFType,
    metrics: List[Callable],
    models: Optional[List[str]] = None,
    train_df: Optional[AnyDFType] = None,
    level: Optional[List[int]] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    agg_fn: Optional[str] = None,
) -> AnyDFType:
    """Evaluate forecast using different metrics.

    Args:
        df (pandas, polars, dask or spark DataFrame): Forecasts to evaluate.
            Must have `id_col`, `time_col`, `target_col` and models' predictions.
        metrics (list of callable): Functions with arguments `df`, `models`,
            `id_col`, `target_col` and optionally `train_df`.
        models (list of str, optional): Names of the models to evaluate.
            If `None` will use every column in the dataframe after removing
            id, time and target. Defaults to None.
        train_df (pandas, polars, dask or spark DataFrame, optional): Training set.
            Used to evaluate metrics such as `mase`. Defaults to None.
        level (list of int, optional): Prediction interval levels. Used to compute
            losses that rely on quantiles. Defaults to None.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its values
            can be timestamps or integers. Defaults to 'ds'.
        target_col (str, optional): Column that contains the target.
            Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for
            each forecast cross-validation fold. Defaults to 'cutoff'.
        agg_fn (str, optional): Statistic to compute on the scores by id to reduce
            them to a single number. Defaults to None.

    Returns:
        pandas, polars, dask or spark DataFrame: Metrics with one row per
            (id, metric) combination and one column per model. If `agg_fn` is
            not `None`, there is only one row per metric.
    """
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        return _distributed_evaluate(
            df=df,
            metrics=metrics,
            models=models,
            train_df=train_df,
            level=level,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
            agg_fn=agg_fn,
        )
    if models is None:
        model_cols = _get_model_cols(df.columns, id_col, time_col, target_col, cutoff_col)
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
        train_df = ufp.sort(train_df, by=[id_col, time_col])
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
            kwargs["cutoff_col"] = cutoff_col
            kwargs["time_col"] = time_col
        metric_params = inspect.signature(metric).parameters
        if "baseline" in metric_params:
            metric_name = f"{metric_name}_{metric_params['baseline'].default}"
        if "q" in metric_params or metric_params["models"].annotation is Dict[str, str]:
            assert level is not None  # we've already made sure of this above
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
            assert level is not None  # we've already made sure of this above
            quantiles = _quantiles_from_levels(level)
            kwargs["quantiles"] = quantiles
            kwargs["models"] = {
                model: _models_from_levels(model, level) for model in model_cols
            }
            result = metric(**kwargs)
            result = ufp.assign_columns(result, "metric", metric_name)
            results_per_metric.append(result)
        elif "level" in metric_params:
            assert level is not None  # we've already made sure of this above
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
    else:
        df = pl.concat(results_per_metric, how="diagonal")
    
    if cutoff_col in df.columns:
        id_cols = [id_col, cutoff_col, "metric"]
    else:
        id_cols = [id_col, "metric"]

    model_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + model_cols]
    if agg_fn is not None:
        group_cols = id_cols[1:] # exclude id_col
        df = ufp.group_by_agg(
            df,
            by=group_cols,
            aggs={m: agg_fn for m in model_cols},
            maintain_order=True,
        )
    return df

class ParetoFrontier:
    """Utilities for Pareto frontier analysis."""
    
    @staticmethod
    def is_dominated(candidate: np.ndarray, others: np.ndarray, directions: np.ndarray) -> bool:
        """Checks if a candidate solution is dominated by any of the others.
        
        Args:
            candidate (np.ndarray): Metric values for the candidate.
            others (np.ndarray): Metric values for other models.
            directions (np.ndarray): 1 for minimization, -1 for maximization.
        """
        # A solution B dominates A if B is at least as good as A in all objectives
        # AND strictly better in at least one objective.
        
        # Adjust for direction (multiply by 1 for min, -1 for max so it's always 'less is better')
        c = candidate * directions
        o = others * directions
        
        # d is True if others[i] <= candidate in all metrics
        better_or_equal = np.all(o <= c, axis=1)
        # s is True if others[i] < candidate in at least one metric
        strictly_better = np.any(o < c, axis=1)
        
        return np.any(better_or_equal & strictly_better)

    @classmethod
    def find_non_dominated(
        cls, 
        performance_df: AnyDFType, 
        metrics: Optional[List[str]] = None,
        maximization: Optional[List[str]] = None
    ) -> AnyDFType:
        """Returns the non-dominated models (Pareto frontier).
        
        Args:
            performance_df (AnyDFType): Output from evaluate.
            metrics (List[str], optional): Metrics to consider. Defaults to all model columns if None.
            maximization (List[str], optional): Metrics where 'more is better'.
        """
        is_pandas = isinstance(performance_df, pd.DataFrame)
        
        if is_pandas:
            df = performance_df
            columns = df.columns.tolist()
        else:
            # Polars
            df = performance_df.to_pandas()
            columns = df.columns.tolist()
            
        # Determine metric columns: usually the columns excluding 'unique_id', 'cutoff', 'metric'
        # In `evaluate` output with agg_fn, ID/cutoff are usually gone, leaving 'metric' and model names.
        # But if `performance_df` is transposed or formatted differently, user passes specific `metrics`.
        if metrics is None:
            # We assume it's metric-centric cols 
            metrics = [c for c in columns if c not in ["unique_id", "metric", "cutoff", "ds", "y"]]
        
        if len(metrics) == 0:
            return performance_df
            
        data = df[metrics].values
        directions = np.ones(len(metrics))
        if maximization:
            for i, m in enumerate(metrics):
                if m in maximization:
                    directions[i] = -1
        
        is_pareto = []
        for i in range(len(data)):
            others = np.delete(data, i, axis=0)
            if len(others) == 0:
                is_pareto.append(True)
                continue
            dominated = cls.is_dominated(data[i], others, directions)
            is_pareto.append(not dominated)
            
        if is_pandas:
            return performance_df.iloc[is_pareto].copy()
        else:
            # Polars
            import polars as pl
            
            # Using row indices to filter Polars dataframe
            indices = [i for i, val in enumerate(is_pareto) if val]
            return performance_df[indices]

    @staticmethod
    def plot_pareto_2d(
        performance_df: AnyDFType,
        metric_x: str,
        metric_y: str,
        maximize_x: bool = False,
        maximize_y: bool = False,
        show_dominated: bool = True,
        title: str = "Pareto Frontier"
    ):
        """Plots the 2D Pareto frontier."""
        import warnings
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib is required for plotting.")
            return None

        pareto_df = ParetoFrontier.find_non_dominated(
            performance_df, 
            metrics=[metric_x, metric_y],
            maximization=([metric_x] if maximize_x else []) + ([metric_y] if maximize_y else [])
        )
        
        is_pandas = isinstance(performance_df, pd.DataFrame)
        if not is_pandas:
            perf_pd = performance_df.to_pandas()
            pareto_pd = pareto_df.to_pandas()
        else:
            perf_pd = performance_df
            pareto_pd = pareto_df
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if show_dominated:
            ax.scatter(
                perf_pd[metric_x], 
                perf_pd[metric_y], 
                color='grey', alpha=0.5, label='Dominated'
            )
            for idx, row in perf_pd.iterrows():
                label = row.get("model", idx) if "model" in perf_pd.columns else idx
                ax.annotate(label, (row[metric_x], row[metric_y]), alpha=0.7)
                
        ax.scatter(
            pareto_pd[metric_x], 
            pareto_pd[metric_y], 
            color='red', s=100, label='Pareto Optimal'
        )
        
        # Sort pareto points for a nice line
        pareto_sorted = pareto_pd.sort_values(metric_x)
        ax.plot(pareto_sorted[metric_x], pareto_sorted[metric_y], 'r--', alpha=0.5)
        
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
