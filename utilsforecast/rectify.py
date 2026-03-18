"""Rectify multi-step forecast correction utilities."""

__all__ = ["compute_rectify_residuals", "align_rectify_features", "rectify"]


from typing import Any, Dict, List, Tuple, Union

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT


def compute_rectify_residuals(
    df: IntoDataFrameT,
    forecasts_df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str | None = None,
) -> IntoDataFrameT:
    """Compute per-horizon residuals between actuals and base forecasts.

    Args:
        df (pandas or polars DataFrame): Actuals with columns
            [id_col, time_col, target_col].
        forecasts_df (pandas or polars DataFrame): Base forecasts
            with columns [id_col, time_col, *models].
        models (list of str): Columns that identify the models predictions
            in forecasts_df.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        target_col (str, optional): Column that contains the target.
            Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the CV fold
            cutoff. When provided, join and horizon computation are
            grouped by (id_col, cutoff_col). Required for cross-validation
            output where (id_col, time_col) pairs repeat across folds.
            Defaults to None.

    Returns:
        pandas or polars DataFrame: DataFrame with columns
            [id_col, time_col, 'horizon', *models] where each model column
            contains the residual (actual - forecast).
    """
    actuals = nw.from_native(df)
    forecasts = nw.from_native(forecasts_df)
    join_on = [id_col, time_col]
    actuals_cols = [id_col, time_col, target_col]
    forecasts_cols = [id_col, time_col, *models]
    if cutoff_col is not None:
        join_on = [id_col, time_col, cutoff_col]
        actuals_cols.append(cutoff_col)
        forecasts_cols.append(cutoff_col)
    merged = actuals.select(actuals_cols).join(
        forecasts.select(forecasts_cols),
        on=join_on,
        how="inner",
    )
    residual_exprs = [
        (nw.col(target_col) - nw.col(model)).alias(model) for model in models
    ]
    partition_by = [id_col] if cutoff_col is None else [id_col, cutoff_col]
    horizon_expr = (
        nw.col(time_col)
        .cum_count()
        .over(*partition_by, order_by=time_col)
        .cast(nw.Int32)
        .alias("horizon")
    )
    meta_cols = [id_col, time_col]
    if cutoff_col is not None:
        meta_cols.append(cutoff_col)
    result = merged.select(
        [*meta_cols, horizon_expr, *residual_exprs]
    ).sort(id_col, time_col)
    return nw.to_native(result)


def align_rectify_features(
    residuals_df: IntoDataFrameT,
    features: np.ndarray,
    id_col: str = "unique_id",
    time_col: str = "ds",
    mode: str = "per_horizon",
) -> Union[Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray]]], Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Align feature matrices with horizon-indexed residuals.

    Args:
        residuals_df (pandas or polars DataFrame): Output of
            compute_rectify_residuals.
        features (numpy ndarray): Feature matrix with shape
            (n_samples, n_features), row-aligned with residuals_df.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        mode (str, optional): 'per_horizon' or 'horizon_aware'.
            Defaults to 'per_horizon'.

    Returns:
        dict or tuple: Training data for rectification models.
    """
    raise NotImplementedError


def rectify(
    df: IntoDataFrameT,
    models: List[str],
    correction_models: Dict[int, Dict[str, Any]],
    features: np.ndarray,
    id_col: str = "unique_id",
    time_col: str = "ds",
    mode: str = "per_horizon",
) -> IntoDataFrameT:
    """Apply rectification corrections to base forecasts.

    Args:
        df (pandas or polars DataFrame): Base forecasts with
            columns [id_col, time_col, *models].
        models (list of str): Columns that identify the models predictions.
        correction_models (dict): Fitted correction models.
            In per_horizon mode: {horizon: {model_name: fitted_model}}.
            In horizon_aware mode: {model_name: fitted_model}.
        features (numpy ndarray): Feature matrix with shape
            (n_samples, n_features), row-aligned with df.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        mode (str, optional): 'per_horizon' or 'horizon_aware'.
            Defaults to 'per_horizon'.

    Returns:
        pandas or polars DataFrame: Corrected forecasts with same schema
            as input df.
    """
    raise NotImplementedError
