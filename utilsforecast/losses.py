# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/losses.ipynb.

# %% auto 0
__all__ = ['mae', 'mse', 'rmse', 'mape', 'smape', 'mase', 'rmae', 'quantile_loss', 'mqloss', 'coverage', 'calibration',
           'scaled_crps']

# %% ../nbs/losses.ipynb 3
from functools import wraps
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .compat import DataFrame

# %% ../nbs/losses.ipynb 6
def _divide_no_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Auxiliary funtion to handle divide by 0"""
    out_dtype = np.result_type(np.float32, a.dtype, b.dtype)
    return np.divide(a, b, out=np.zeros(a.shape, dtype=out_dtype), where=b != 0)

# %% ../nbs/losses.ipynb 7
def _metric_protections(
    y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray] = None
) -> None:
    if weights is None:
        return
    if np.sum(weights) <= 0:
        raise ValueError("Sum of weights must be positive")
    if y.shape != y_hat.shape:
        raise ValueError(
            f"Wrong y_hat dimension. y_hat shape={y_hat.shape}, y shape={y.shape}"
        )
    if weights.shape != y.shape:
        raise ValueError(
            f"Wrong weight dimension. weights shape={weights.shape}, y shape={y.shape}"
        )

# %% ../nbs/losses.ipynb 13
def _base_docstring(*args, **kwargs) -> Callable:
    base_docstring = """

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    model_cols : list of str
        Columns that identify the models predictions.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with the {name} for each id.
    """

    def docstring_decorator(f: Callable):
        f.__doc__ = f.__doc__ + base_docstring.format(name=f.__name__.upper())
        return f

    return docstring_decorator(*args, **kwargs)

# %% ../nbs/losses.ipynb 14
@_base_docstring
def mae(
    df: DataFrame,
    model_cols: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Mean Absolute Error (MAE)

    MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series."""
    if isinstance(df, pd.DataFrame):
        res = (
            (df[model_cols].sub(df[target_col], axis=0))
            .abs()
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:
        exprs = [
            (pl.col(target_col) - pl.col(model_col)).abs().mean().alias(model_col)
            for model_col in model_cols
        ]
        res = df.group_by(id_col).agg(exprs)
    return res

# %% ../nbs/losses.ipynb 20
@_base_docstring
def mse(
    df: DataFrame,
    model_cols: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Mean Squared Error (MSE)

    MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series."""
    if isinstance(df, pd.DataFrame):
        res = (
            (df[model_cols].sub(df[target_col], axis=0))
            .pow(2)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:
        exprs = [
            (pl.col(target_col) - pl.col(model_col)).pow(2).mean().alias(model_col)
            for model_col in model_cols
        ]
        res = df.group_by(id_col).agg(*exprs)
    return res

# %% ../nbs/losses.ipynb 26
@_base_docstring
def rmse(
    df: DataFrame,
    model_cols: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> Union[float, np.ndarray]:
    """Root Mean Squared Error (RMSE)

    RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.
    RMSE has a direct connection to the L2 norm."""
    res = mse(df, model_cols, id_col, target_col)
    if isinstance(res, pd.DataFrame):
        res[model_cols] = res[model_cols].pow(0.5)
    else:
        import polars as pl

        res = res.with_columns(*[pl.col(c).pow(0.5) for c in model_cols])
    return res

# %% ../nbs/losses.ipynb 33
@_base_docstring
def mape(
    df: DataFrame,
    model_cols: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> Union[float, np.ndarray]:
    """Mean Absolute Percentage Error (MAPE)

    MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error."""
    if isinstance(df, pd.DataFrame):
        res = (
            df[model_cols]
            .sub(df[target_col], axis=0)
            .abs()
            .div(df[target_col].abs(), axis=0)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:
        exprs = [
            (pl.col(target_col).sub(pl.col(model_col)).abs() / pl.col(target_col).abs())
            .mean()
            .alias(model_col)
            for model_col in model_cols
        ]
        res = df.group_by(id_col).agg(*exprs)
    return res

# %% ../nbs/losses.ipynb 38
@_base_docstring
def smape(
    df: DataFrame,
    model_cols: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> Union[float, np.ndarray]:
    """Symmetric Mean Absolute Percentage Error (SMAPE)

    SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined when the target is zero."""
    if isinstance(df, pd.DataFrame):
        delta_y = df[model_cols].sub(df[target_col], axis=0).abs()
        scale = df[model_cols].abs().add(df[target_col].abs(), axis=0)
        raw = delta_y.div(scale).fillna(0)
        res = raw.groupby(df[id_col], observed=True).mean()
        res.index.name = id_col
        res = res.reset_index()
    else:
        exprs = [
            (
                pl.col(model_col)
                .sub(pl.col(target_col))
                .abs()
                .truediv(pl.col(model_col).abs().add(pl.col(target_col).abs()))
            )
            .fill_nan(0)
            .alias(model_col)
            for model_col in model_cols
        ]
        res = df.select([id_col, *exprs]).group_by(id_col).mean()
    return res

# %% ../nbs/losses.ipynb 45
@_base_docstring
def mase(
    df: DataFrame,
    model_cols: List[str],
    seasonality: int,
    train_df: DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Mean Absolute Scaled Error (MASE)

    MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition."""
    if isinstance(df, pd.DataFrame):
        res = (
            df[model_cols]
            .sub(df[target_col], axis=0)
            .abs()
            .groupby(df[id_col], observed=True)
            .mean()
        )
        # assume train_df is sorted
        lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
        scale = (
            (train_df[target_col] - lagged)
            .abs()
            .groupby(train_df[id_col], observed=True)
            .mean()
        )
        res = res.div(scale, axis=0)
        res.index.name = id_col
        res = res.reset_index()
    else:
        exprs = [
            (pl.col(target_col).sub(pl.col(model_col)).abs()).mean().alias(model_col)
            for model_col in model_cols
        ]
        res = df.group_by(id_col).agg(*exprs)
        # assume train_df is sorted
        expr = (
            (pl.col(target_col).sub(pl.col(target_col).shift(seasonality)).abs())
            .mean()
            .alias("scale")
        )
        scale = train_df.group_by(id_col).agg(expr)
        res = res.join(scale, on=id_col, how="left").select(
            [
                id_col,
                *[
                    (pl.col(model_col) / pl.col("scale")).alias(model_col)
                    for model_col in model_cols
                ],
            ]
        )
    return res

# %% ../nbs/losses.ipynb 54
def rmae(
    df: DataFrame,
    model_cols1: List[str],
    model_cols2: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Relative Mean Absolute Error (RMAE)

    Calculates the RAME between two sets of forecasts (from two different forecasting methods).
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator."""
    numerator = mae(df, model_cols1, id_col, target_col)
    denominator = mae(df, model_cols2, id_col, target_col)
    if isinstance(numerator, pd.DataFrame):
        res = numerator.merge(denominator, on=id_col, suffixes=("", "_denominator"))
        out_cols = [id_col]
        for m1, m2 in zip(model_cols1, model_cols2):
            col_name = f"{m1}_div_{m2}"
            res[col_name] = res[m1] / res[f"{m2}_denominator"]
            out_cols.append(col_name)
        res = res[out_cols]
    else:
        res = numerator.join(denominator, on=id_col, suffix="_denominator")
        res = res.select(
            [
                id_col,
                *[
                    pl.col(m1)
                    .truediv(pl.col(f"{m2}_denominator"))
                    .alias(f"{m1}_div_{m2}")
                    for m1, m2 in zip(model_cols1, model_cols2)
                ],
            ]
        )
    return res

# %% ../nbs/losses.ipynb 61
@_base_docstring
def quantile_loss(
    df: DataFrame,
    model_cols: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Quantile Loss (QL)

    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median."""
    _metric_protections(y, y_hat, weights)

    delta_y = y - y_hat
    loss = np.maximum(q * delta_y, (q - 1) * delta_y)

    if weights is not None:
        quantile_loss = np.average(
            loss[~np.isnan(loss)], weights=weights[~np.isnan(loss)], axis=axis
        )
    else:
        quantile_loss = np.nanmean(loss, axis=axis)

    return quantile_loss

# %% ../nbs/losses.ipynb 65
def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Multi-Quantile loss (MQL)

    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution $\mathbf{\hat{F}}_{\\tau}$ with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    Parameters
    ----------
    y : numpy array
        Observed values.
    y_hat : numpy array
        Predicted values.
    quantiles : numpy array
        Quantiles to compare against.
    weights : numpy array, optional (default=None)
        Weights for weighted average.
    axis : int, optional (default=None)
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the elements of
        the input array. If axis is negative it counts from the last to first.

    Returns
    -------
    numpy array or double
        MQL along the specified axis.

    References
    ----------
    [1] https://www.jstor.org/stable/2629907
    """
    if weights is None:
        weights = np.ones(y.shape)

    _metric_protections(y, y_hat, weights)
    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_rep - y_hat
    mqloss = np.maximum(quantiles * error, (quantiles - 1) * error)

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss

# %% ../nbs/losses.ipynb 68
def coverage(
    y: np.ndarray,
    y_hat_lo: np.ndarray,
    y_hat_hi: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Coverage of y with y_hat_lo and y_hat_hi.

    Parameters
    ----------
    y : numpy array
        Observed values.
    y_hat_lo : numpy array
        Lower prediction interval.
    y_hat_hi : numpy array
        Higher prediction interval.

    Returns
    -------
    numpy array or double
        Coverage of y_hat

    References
    ----------
    [1] https://www.jstor.org/stable/2629907
    """
    return 100 * np.logical_and(y >= y_hat_lo, y <= y_hat_hi).mean()

# %% ../nbs/losses.ipynb 71
def calibration(
    y: np.ndarray,
    y_hat_hi: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Fraction of y that is lower than y_hat_hi.

    Parameters
    ----------
    y : numpy array
        Observed values.
    y_hat_hi : numpy array
        Higher prediction interval.

    Returns
    -------
    numpy array or double
        Calibration of y_hat

    References
    ----------
    [1] https://www.jstor.org/stable/2629907
    """
    return (y <= y_hat_hi).mean()

# %% ../nbs/losses.ipynb 74
def scaled_crps(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.
    This metric averages percentual weighted absolute deviations as
    defined by the quantile losses.


    Parameters
    ----------
    y : numpy array
        Observed values.
    y_hat : numpy array
        Predicted values.
    quantiles : numpy array
        Quantiles to compare against.
    weights : numpy array, optional (default=None)
        Weights for weighted average.
    axis : int, optional (default=None)
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the elements of
        the input array. If axis is negative it counts from the last to first.

    Returns
    -------
    numpy array or double.
        Scaled crps along the specified axis.

    References
    ----------
    [1] https://proceedings.mlr.press/v139/rangapuram21a.html
    """
    eps = np.finfo(float).eps
    norm = np.sum(np.abs(y))
    loss = mqloss(y=y, y_hat=y_hat, quantiles=quantiles, weights=weights, axis=axis)
    loss = 2 * loss * np.sum(np.ones(y.shape)) / (norm + eps)
    return loss
