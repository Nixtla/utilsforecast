"""Loss functions for model evaluation."""

__all__ = ['mae', 'mse', 'rmse', 'bias', 'cfe', 'pis', 'spis', 'mape', 'smape', 'mase', 'rmae', 'nd', 'msse', 'rmsse',
           'quantile_loss', 'scaled_quantile_loss', 'mqloss', 'scaled_mqloss', 'coverage', 'calibration', 'scaled_crps',
           'tweedie_deviance', 'linex']


from typing import Callable, Dict, List, Union

import narwhals as nw
import numpy as np
import pandas as pd

import utilsforecast.processing as ufp

from .compat import DFType, pl, pl_Expr


def _base_docstring(*args, **kwargs) -> Callable:
    base_docstring = """

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actual values and predictions.
        models (list of str): Columns that identify the models predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """

    def docstring_decorator(f: Callable):
        if f.__doc__ is not None:
            f.__doc__ += base_docstring
        return f

    return docstring_decorator(*args, **kwargs)


def _scale_loss(
    loss_df: DFType,
    scale_type: str,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    cutoff_col: str = "cutoff"
) -> DFType:
    """
    Args:
        loss_df (pandas or polars DataFrame): Input dataframe with id, actuals, predictions and losses results.
        scale_type (str): Type of scaling. Possible values are 'absolute_error' or 'squared_error'.
        models (list of str): Columns that identify the models predictions.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        time_col (str, optional): Column that contains the time values. Defaults to 'ds'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://robjhyndman.com/papers/mase.pdf
    """
    loss_nw = nw.from_native(loss_df)
    train_nw = nw.from_native(train_df)

    lagged = nw.col(target_col).shift(seasonality).over(id_col)

    if scale_type == "absolute_error":
        scale_expr = [(nw.col(target_col) - lagged).abs().alias("scale")]
    else:
        scale_expr = [((nw.col(target_col) - lagged) ** 2).alias("scale")]

    group_cols: list[str]
    if cutoff_col in loss_df.columns:
        scale = train_nw.select([id_col, time_col] + scale_expr)
        group_cols = [cutoff_col, id_col]
        scale = scale.with_columns(time_col, nw.col("scale").rolling_mean(window_size=int(1e12), min_samples=1).over(id_col).alias("scale"))
        full_nw = loss_nw.join(scale, left_on=group_cols, right_on=[time_col, id_col], how="left")
        full_nw = full_nw.with_columns(nw.when(nw.col("scale") == 0).then(float("nan")).otherwise(nw.col("scale")).alias("scale"))
    else:
        scale = train_nw.select([id_col] + scale_expr)
        group_cols = [id_col]
        scale = train_nw.select([id_col] + scale_expr)
        scale = scale.group_by(id_col).agg([nw.col("scale").mean()])
        full_nw = loss_nw.join(scale, on=group_cols, how="left")

    res = full_nw.select(
        [
            *group_cols,
            *[
                (nw.col(model) / nw.col("scale")).fill_nan(0).alias(model)
                for model in models
            ],
        ]
    )
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def mae(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Mean Absolute Error (MAE)

    MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series."""
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(target_col) - nw.col(m)).abs().alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def mse(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Mean Squared Error (MSE)

    MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series."""
    
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([((nw.col(target_col) - nw.col(m)) ** 2).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def rmse(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Root Mean Squared Error (RMSE)

    RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.
    RMSE has a direct connection to the L2 norm."""

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([((nw.col(target_col) - nw.col(m)) ** 2).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().sqrt().alias(m) for m in models])
    res = res.to_native()

    return res

@_base_docstring
def bias(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """Forecast estimator bias.

    Defined as prediction - actual"""

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(m) - nw.col(target_col)).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def cfe(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """
    Cumulative Forecast Error (CFE)

    Total signed forecast error per series. Positive values mean under forecast; negative mean over forecast.
    """
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(m) - nw.col(target_col)).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).sum().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def pis(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """
    Compute the raw Absolute Periods In Stock (PIS) for one or multiple models.

    The PIS metric sums the absolute forecast errors per series without any scaling,
    yielding a scale-dependent measure of bias.
    """
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(m) - nw.col(target_col)).abs().alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).sum().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def spis(
    df: DFType,
    train_df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    cutoff_col: str = "cutoff"
) -> DFType:
    """
    Compute the scaled Absolute Periods In Stock (sAPIS) for one or multiple models.

    The sPIS metric scales the sum of absolute forecast errors by the mean in-sample demand,
    yielding a scale-independent bias measure that can be aggregated across series.
    """

    loss_df = pis(df, models, id_col, target_col, cutoff_col)

    loss_nw = nw.from_native(loss_df)
    train_nw = nw.from_native(train_df)
    scale_expr = [nw.col(target_col).alias("scale")]

    group_cols: list[str]
    if cutoff_col in loss_df.columns:
        scale = train_nw.select([id_col, time_col] + scale_expr)
        group_cols = [cutoff_col, id_col]
        scale = scale.with_columns(time_col, nw.col("scale").rolling_mean(window_size=int(1e12), min_samples=1).over(id_col).alias("scale"))
        full_nw = loss_nw.join(scale, left_on=group_cols, right_on=[time_col, id_col], how="left")
        full_nw = full_nw.with_columns(nw.when(nw.col("scale") == 0).then(float("nan")).otherwise(nw.col("scale")).alias("scale"))
    else:
        scale = train_nw.select([id_col] + scale_expr)
        group_cols = [id_col]
        scale = train_nw.select([id_col] + scale_expr)
        scale = scale.group_by(id_col).agg([nw.col("scale").mean()])
        full_nw = loss_nw.join(scale, on=group_cols, how="left")

    res = full_nw.select(
        [
            *group_cols,
            *[
                (nw.col(model) / nw.col("scale")).fill_nan(0).alias(model)
                for model in models
            ],
        ]
    )
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res

    
@_base_docstring
def linex(
    df: DFType,
    models: List[str],
    a: float = 1.0,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """
    Linex Loss

    The Linex (Linear Exponential) loss penalizes over- and under-forecasting
    asymmetrically depending on the parameter a.

    - If a > 0, under-forecasting (y > y_hat) is penalized more.
    - If a < 0, over-forecasting (y_hat > y) is penalized more.
    - a must not be 0.
    """
    if np.isclose(a, 0.0):
        raise ValueError("Parameter a in Linex loss must be non-zero.")

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([((a * (nw.col(m) - nw.col(target_col))).exp() - (a * (nw.col(m) - nw.col(target_col))) - 1).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res

def _zero_to_nan(series: Union[pd.Series, "pl.Expr"]) -> Union[pd.Series, "pl.Expr"]:
    if isinstance(series, pd.Series):
        res = series.replace(0, np.nan)
    else:
        res = pl.when(series == 0).then(float("nan")).otherwise(series.abs())
    return res

@_base_docstring
def mape(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """Mean Absolute Percentage Error (MAPE)

    MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error."""
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(m) - nw.col(target_col)).abs() / (nw.when(nw.col(target_col) == 0.0).then(float("nan")).otherwise(nw.col(target_col).abs())).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res


@_base_docstring
def smape(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """Symmetric Mean Absolute Percentage Error (SMAPE)

    SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 100% which is desirable compared to normal MAPE that
    may be undetermined when the target is zero."""

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(m) - nw.col(target_col)).abs() / (nw.when((nw.col(target_col).abs() + nw.col(m).abs()) == 0.0).then(float("nan")).otherwise(nw.col(target_col).abs() + nw.col(m).abs())).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res

def mase(
    df: DFType,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    cutoff_col: str = "cutoff"
) -> DFType:
    """Mean Absolute Scaled Error (MASE)

    MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        time_col (str, optional): Column that contains the time values. Defaults to 'ds'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://robjhyndman.com/papers/mase.pdf
    """
    mean_abs_err = mae(df, models, id_col, target_col, cutoff_col)
        
    return _scale_loss(
        loss_df=mean_abs_err,
        scale_type="absolute_error",
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )


def rmae(
    df: DFType,
    models: List[str],
    baseline: str,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Relative Mean Absolute Error (RMAE)

    Calculates the RAME between two sets of forecasts (from two different forecasting methods).
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        baseline (str): Column that identifies the baseline model predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    numerator = mae(df, models, id_col, target_col)
    denominator = mae(df, [baseline], id_col, target_col)
    if ufp.is_nan(denominator[baseline]).any():
        raise ValueError(f"baseline model ({baseline}) contains NaNs.")
    denominator = ufp.rename(denominator, {baseline: f"{baseline}_denominator"})
    res = ufp.join(numerator, denominator, on=id_col)
    if isinstance(numerator, pd.DataFrame):
        for model in models:
            res[model] = (
                res[model].div(_zero_to_nan(res[f"{baseline}_denominator"])).fillna(0)
            )
        res = res[[id_col, *models]]
    else:

        def gen_expr(model, baseline) -> pl_Expr:
            denominator = _zero_to_nan(pl.col(f"{baseline}_denominator"))
            return pl.col(model).truediv(denominator).fill_nan(0).alias(model)

        exprs: List[pl_Expr] = [gen_expr(m, baseline) for m in models]
        res = res.select([id_col, *exprs])
    return res


def nd(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """Normalized Deviation (ND)

    ND measures the relative prediction
    accuracy of a forecasting method by calculating the
    sum of the absolute deviation of the prediction and the true
    value at a given time and dividing it by the sum of the absolute
    value of the ground truth.

    Args:
        df: Input dataframe with id, times, actuals and predictions.
        models: Columns that identify the models predictions.
        id_col: Column that identifies each serie. Defaults to 'unique_id'.
        target_col: Column that contains the target. Defaults to 'y'.
        cutoff_col: Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        Dataframe with one row per id and one column per model.
    """
    group_cols: list[str]
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    nom = res.with_columns([(nw.col(target_col) - nw.col(m)).abs().alias(m) for m in models])
    nom = nom.group_by(group_cols).agg([nw.col(m).sum().alias(m) for m in models])

    denom = res.with_columns([nw.col(target_col).abs()])
    denom = denom.group_by(group_cols).agg([nw.col(target_col).sum().alias(target_col)])

    res = nom.join(denom, on=group_cols, how="left")

    res = res.select(
        [
            *group_cols,
            *[
                (nw.col(model) / (nw.col(target_col))).fill_nan(0).alias(model)
                for model in models
            ],
        ]
    )
    res = res.to_native()

    return res    


def msse(
    df: DFType,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Mean Squared Scaled Error (MSSE)

    MSSE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean squared errors
    of the prediction and the observed value against the mean
    squared errors of the seasonal naive model.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        time_col (str, optional): Column that contains the time values. Defaults to 'ds'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://otexts.com/fpp3/accuracy.html
    """
    mean_sq_err = mse(df=df, models=models, id_col=id_col, target_col=target_col)
    return _scale_loss(
        loss_df=mean_sq_err,
        scale_type="squared_error",
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )


def rmsse(
    df: DFType,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    res = msse(
        df,
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    res_nw = nw.from_native(res)
    res_nw = res_nw.with_columns([(nw.col(m) ** 2).alias(m) for m in models])
    res = res_nw.to_native()

    return res


rmsse.__doc__ = msse.__doc__.replace(  # type: ignore[union-attr]
    "Mean Squared Scaled Error (MSSE)", "Root Mean Squared Scaled Error (RMSSE)"
)


def quantile_loss(
    df: DFType,
    models: Dict[str, str],
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Quantile Loss (QL)

    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions for the specified quantile.
        q (float, optional): Quantile for the predictions' comparison. Defaults to 0.5.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(target_col) - nw.col(m)).alias(m) for m in models])
    res = res.with_columns([(nw.when(nw.col(m) >= 0).then(q * nw.col(m)).otherwise((q - 1) * nw.col(m))).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res

def scaled_quantile_loss(
    df: DFType,
    models: Dict[str, str],
    seasonality: int,
    train_df: DFType,
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Scaled Quantile Loss (SQL)

    SQL measures the deviation of a quantile forecast scaled by
    the mean absolute errors of the seasonal naive model.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median.
    This was the official measure used in the M5 Uncertainty competition
    with seasonality = 1.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions for the specified quantile.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        q (float, optional): Quantile for the predictions' comparison. Defaults to 0.5.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        time_col (str, optional): Column that contains the time values. Defaults to 'ds'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    q_loss = quantile_loss(
        df=df, models=models, q=q, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col
    )
    return _scale_loss(
        loss_df=q_loss,
        scale_type="absolute_error",
        models=list(models.keys()),
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )


def mqloss(
    df: DFType,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """Multi-Quantile loss (MQL)

    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    # Not the most efficient implementation
    quantile_preds = {}
    for q, idx in zip(quantiles, range(len(quantiles))):  # Assumes quantiles are ordered
        quantile_preds[q] = {
            model: forecasts[idx]
            for model, forecasts in models.items()
        }

    res = (
        nw.concat(
            [
                nw.from_native(quantile_loss(df, models=quantile_preds[q], q=q, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col))
                for q in quantiles
            ]
        )
    )    
    res = res.group_by(group_cols).agg([nw.col(col).mean().alias(col) for col in res.columns if col not in group_cols])
    res = res.to_native()

    return res


def scaled_mqloss(
    df: DFType,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Scaled Multi-Quantile loss (SMQL)

    SMQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values
    scaled by the mean absolute errors of the seasonal naive model.
    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.
    This was the official measure used in the M5 Uncertainty competition
    with seasonality = 1.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        time_col (str, optional): Column that contains the time values. Defaults to 'ds'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    mq_loss = mqloss(
        df=df, models=models, quantiles=quantiles, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col
    )
    return _scale_loss(
        loss_df=mq_loss,
        scale_type="absolute_error",
        models=list(models.keys()),
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )


def coverage(
    df: DFType,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Coverage of y with y_hat_lo and y_hat_hi.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        level (int): Confidence level used for intervals.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(target_col).is_between(nw.col(f"{m}-lo-{level}"), nw.col(f"{m}-hi-{level}"))).alias(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res


def calibration(
    df: DFType,
    models: Dict[str, str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """
    Fraction of y that is lower than the model's predictions.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    res = nw.from_native(df)
    res = res.with_columns([(nw.col(target_col) <= nw.col(m)).alias(m) for m in list(models.items())])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in list(models.items())])
    res = res.to_native()

    return res


def scaled_crps(
    df: DFType,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DFType:
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.
    This metric averages percentual weighted absolute deviations as
    defined by the quantile losses.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://proceedings.mlr.press/v139/rangapuram21a.html
    """
    eps: np.float64 = np.finfo(np.float64).eps
    quantiles = np.asarray(quantiles)
    loss = mqloss(df, models, quantiles, id_col, target_col, cutoff_col)
    loss_nw = nw.from_native(loss)

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]

    df_nw = nw.from_native(df)
    grouped_df = df_nw.group_by(group_cols)
    sizes = grouped_df.len()
    norm = grouped_df.agg(nw.col(target_col).abs().sum().alias("norm"))
    loss_nw = loss_nw.join(sizes, on=group_cols, how="left")
    loss_nw = loss_nw.join(norm, on=group_cols, how="left")
    loss_nw = loss_nw.with_columns([((2 * nw.col(m) * nw.col("len")) / (nw.col("norm") + eps)).alias(m) for m in list(models.keys())])
    res = loss_nw.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in list(models.keys())])
    res = res.to_native()
    
    return res


def tweedie_deviance(
    df: DFType,
    models: List[str],
    power: float = 1.5,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff"
) -> DFType:
    """
    Compute the Tweedie deviance loss for one or multiple models, grouped by an identifier.

    Each group's deviance is calculated using the mean_tweedie_deviance function, which
    measures the deviation between actual and predicted values under the Tweedie distribution.

    The `power` parameter defines the specific compound distribution:
      - 1: Poisson
      - (1, 2): Compound Poisson-Gamma
      - 2: Gamma
      - >2: Inverse Gaussian

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        power (float, optional): Tweedie power parameter. Determines the compound distribution. Defaults to 1.5.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cv. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: DataFrame with one row per id and one column per model, containing the mean Tweedie deviance.

    References:
        [1] https://en.wikipedia.org/wiki/Tweedie_distribution

    """
    if power < 0:
        raise ValueError("Power must be non-negative.")
    elif power >= 2 and np.any(df[[target_col]] <= 0):
        raise ValueError(
            f"Power {power} requires all target values to be strictly positive."
        )

    if np.any(df[models] <= 0):
        raise ValueError(
            "All predictions must be strictly positive for Tweedie deviance."
        )

    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]


    if power == 0:

        def gen_expr(model):
            return ((nw.col(model) - nw.col(target_col)) ** 2).alias(model)

    elif power == 1:

        def gen_expr(model):
            return (
                nw.when(nw.col(target_col) == 0)
                .then(2 * nw.col(model))
                .otherwise(
                    2
                    * (
                        nw.col(target_col)
                        * (nw.col(target_col).log() - nw.col(model).log())
                        - (nw.col(target_col) - nw.col(model))
                    )
                )
                .alias(model)
            )

    elif power == 2:

        def gen_expr(model):
            return (
                2
                * (nw.col(model).log() - nw.col(target_col).log())
                 + (nw.col(target_col) / (nw.col(model)))
                - 1
            ).alias(model)

    else:

        def gen_expr(model):
            return (
                2
                * (
                    nw.col(target_col)
                    .clip(0)
                    ** (2 - power)
                    / ((1 - power) * (2 - power))
                )
                + (
                    nw.col(target_col)
                    * (nw.col(model) ** (1 - power))
                    / (1 - power)
                )
                + (nw.col(model) ** (2 - power) / (2 - power))
            ).alias(model)

    res = nw.from_native(df)
    res = res.with_columns([gen_expr(m) for m in models])
    res = res.group_by(group_cols).agg([nw.col(m).mean().alias(m) for m in models])
    res = res.to_native()

    return res
