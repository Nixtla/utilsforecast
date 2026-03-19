"""Rectify multi-step forecast correction utilities."""

__all__ = ["compute_rectify_residuals", "align_rectify_features", "rectify"]


from typing import Any, Dict, List, Tuple, Union

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT

from utilsforecast.validation import validate_format


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
    validate_format(df, id_col=id_col, time_col=time_col, target_col=target_col)
    validate_format(forecasts_df, id_col=id_col, time_col=time_col, target_col=None)
    missing_models = sorted(set(models) - set(forecasts_df.columns))
    if missing_models:
        raise ValueError(f"forecasts_df is missing model columns: {missing_models}")
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
    models: List[str] | None = None,
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
        models (list of str, optional): Columns that identify the model
            residuals. If None, all columns except id_col, time_col, and
            'horizon' are used.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        mode (str, optional): 'per_horizon' or 'horizon_aware'.
            Defaults to 'per_horizon'.

    Returns:
        dict or tuple: Training data for rectification models.
            In per_horizon mode: {horizon: (X, {model_name: residuals})}.
            In horizon_aware mode: (X_with_horizon_col, {model_name: residuals}).
    """
    residuals = nw.from_native(residuals_df)
    if models is not None:
        model_cols = models
    else:
        model_cols = [
            c for c in residuals.columns if c not in (id_col, time_col, "horizon")
        ]
    horizons = residuals["horizon"].to_numpy()
    if mode == "per_horizon":
        result: Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
        for h in sorted(set(horizons)):
            mask = horizons == h
            X_h = features[mask]
            y_dict = {m: residuals[m].to_numpy()[mask] for m in model_cols}
            result[h] = (X_h, y_dict)
        return result
    elif mode == "horizon_aware":
        X_aug = np.column_stack([features, horizons])
        y_dict = {m: residuals[m].to_numpy() for m in model_cols}
        return X_aug, y_dict
    else:
        raise ValueError(f"mode must be 'per_horizon' or 'horizon_aware', got '{mode}'")


def rectify(
    df: IntoDataFrameT,
    models: List[str],
    correction_models: Union[Dict[int, Dict[str, Any]], Dict[str, Any]],
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
    validate_format(df, id_col=id_col, time_col=time_col, target_col=None)
    missing_models = sorted(set(models) - set(df.columns))
    if missing_models:
        raise ValueError(f"df is missing model columns: {missing_models}")
    frame = nw.from_native(df)
    horizon_expr = (
        nw.col(time_col)
        .cum_count()
        .over(id_col, order_by=time_col)
        .cast(nw.Int32)
        .alias("_horizon")
    )
    with_horizon = frame.with_columns(horizon_expr)
    horizons = with_horizon["_horizon"].to_numpy()
    corrections = np.zeros((len(frame), len(models)))
    if mode == "per_horizon":
        for h in sorted(set(horizons)):
            mask = horizons == h
            X_h = features[mask]
            for j, model in enumerate(models):
                corrections[mask, j] = correction_models[h][model].predict(X_h)
    elif mode == "horizon_aware":
        X_aug = np.column_stack([features, horizons])
        for j, model in enumerate(models):
            corrections[:, j] = correction_models[model].predict(X_aug)
    else:
        raise ValueError(f"mode must be 'per_horizon' or 'horizon_aware', got '{mode}'")
    corrected = with_horizon.drop("_horizon")
    for j, model in enumerate(models):
        corrected = corrected.with_columns(
            (nw.col(model) + corrections[:, j]).alias(model)
        )
    return nw.to_native(corrected)
