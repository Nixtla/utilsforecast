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

    Returns:
        pandas or polars DataFrame: DataFrame with columns
            [id_col, time_col, 'horizon', *models] where each model column
            contains the residual (actual - forecast).
    """
    actuals = nw.from_native(df)
    forecasts = nw.from_native(forecasts_df)
    merged = actuals.select([id_col, time_col, target_col]).join(
        forecasts.select([id_col, time_col, *models]),
        on=[id_col, time_col],
        how="inner",
    )
    residual_exprs = [
        (nw.col(target_col) - nw.col(model)).alias(model) for model in models
    ]
    sorted_merged = merged.sort(id_col, time_col)
    horizon_expr = (
        nw.col(time_col).cum_count().over(id_col).cast(nw.Int32).alias("horizon")
    )
    result = sorted_merged.select(
        [id_col, time_col, horizon_expr, *residual_exprs]
    )
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
