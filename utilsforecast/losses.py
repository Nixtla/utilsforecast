"""Loss functions for model evaluation."""

__all__ = [
    "mae",
    "mse",
    "rmse",
    "bias",
    "cfe",
    "pis",
    "spis",
    "mape",
    "wape",
    "smape",
    "mase",
    "rmae",
    "nd",
    "msse",
    "rmsse",
    "quantile_loss",
    "scaled_quantile_loss",
    "mqloss",
    "scaled_mqloss",
    "coverage",
    "winkler_score",
    "calibration",
    "scaled_crps",
    "tweedie_deviance",
    "linex",
    "excess_volatility",
    "forecast_percentage_change",
]

import warnings
from typing import Callable, Dict, List, Union

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT


def _get_group_cols(df: IntoDataFrameT, id_col: str, cutoff_col: str) -> list[str]:
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]
    return group_cols


def _base_docstring(*args, **kwargs) -> Callable:
    base_docstring = """

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actual values and predictions.
        models (list of str): Columns that identify the models predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """

    def docstring_decorator(f: Callable):
        if f.__doc__ is not None:
            f.__doc__ += base_docstring
        return f

    return docstring_decorator(*args, **kwargs)


def _nw_agg_expr(
    df: IntoDataFrameT,
    models: Union[list[str], list[tuple[str, str]]],
    id_col: str,
    cutoff_col: str,
    gen_expr: Callable[[Union[str, tuple[str, str]]], nw.Expr],
    agg: str = "mean",
) -> IntoDataFrameT:
    exprs = [gen_expr(model) for model in models]
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    return (
        nw.from_native(df)
        .select([*group_cols, *exprs])
        .group_by(*group_cols)
        .agg(getattr(nw.all(), agg)())
        .sort(*group_cols)
        .to_native()
    )


def _create_train_with_cutoffs(
    train_df: IntoDataFrameT,
    df: IntoDataFrameT,
    id_col: str,
    time_col: str,
    cutoff_col: str,
) -> IntoDataFrameT:
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    train_df = nw.from_native(train_df)

    if cutoff_col in group_cols:
        cutoffs_df = nw.from_native(df).select(*group_cols).unique()
        train_df = train_df.join(cutoffs_df, on=id_col, how="inner").filter(
            nw.col(time_col) <= nw.col(cutoff_col)
        )

    return train_df


def _scale_loss(
    df: IntoDataFrameT,
    models: List[str],
    scales: IntoDataFrameT,
    id_col: str,
    cutoff_col: str,
) -> IntoDataFrameT:
    exprs = [(nw.col(m) / nw.col("scale")).alias(m) for m in models]
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    return (
        nw.from_native(df)
        .join(nw.from_native(scales), on=group_cols)
        .select([*group_cols, *exprs])
        .to_native()
    )


@_base_docstring
def mae(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Mean Absolute Error (MAE)

    MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series."""
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        cutoff_col=cutoff_col,
        gen_expr=lambda m: (nw.col(target_col) - nw.col(m)).abs().alias(m),
    )


@_base_docstring
def mse(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Mean Squared Error (MSE)

    MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series."""
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        cutoff_col=cutoff_col,
        gen_expr=lambda m: ((nw.col(target_col) - nw.col(m)) ** 2).alias(m),
    )


@_base_docstring
def rmse(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Root Mean Squared Error (RMSE)

    RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.
    RMSE has a direct connection to the L2 norm."""

    df = mse(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return (
        nw.from_native(df).with_columns(*[nw.col(m).sqrt() for m in models]).to_native()
    )


@_base_docstring
def bias(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Forecast estimator bias.

    Defined as prediction - actual"""
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        cutoff_col=cutoff_col,
        gen_expr=lambda m: (nw.col(m) - nw.col(target_col)).alias(m),
    )


@_base_docstring
def cfe(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """
    Cumulative Forecast Error (CFE)

    Total signed forecast error per series. Positive values mean under forecast; negative mean over forecast.
    """
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        cutoff_col=cutoff_col,
        gen_expr=lambda m: (nw.col(m) - nw.col(target_col)).alias(m),
        agg="sum",
    )


@_base_docstring
def pis(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """
    Compute the raw Absolute Periods In Stock (PIS) for one or multiple models.

    The PIS metric sums the absolute forecast errors per series without any scaling,
    yielding a scale-dependent measure of bias.
    """
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        cutoff_col=cutoff_col,
        gen_expr=lambda m: (nw.col(m) - nw.col(target_col)).abs().alias(m),
        agg="sum",
    )


def spis(
    df: IntoDataFrameT,
    models: List[str],
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> IntoDataFrameT:
    """
    Compute the scaled Absolute Periods In Stock (sAPIS) for one or multiple models.

    The sPIS metric scales the sum of absolute forecast errors by the mean in-sample demand,
    yielding a scale-independent bias measure that can be aggregated across series.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """

    def gen_expr(_m):
        return nw.col(target_col).alias("scale")

    df = nw.from_native(df)
    train_df = _create_train_with_cutoffs(
        train_df=train_df,
        df=df,
        id_col=id_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )
    scales = _nw_agg_expr(
        df=train_df,
        models=["unused"],
        id_col=id_col,
        cutoff_col=cutoff_col,
        gen_expr=gen_expr,
    )
    raw = pis(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return _scale_loss(
        df=raw, models=models, scales=scales, id_col=id_col, cutoff_col=cutoff_col
    )


def _zero_to_nan(series):
    return nw.when(series == 0).then(float("nan")).otherwise(series)


@_base_docstring
def mape(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Mean Absolute Percentage Error (MAPE)

    MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error."""

    def gen_expr(model):
        abs_err = (nw.col(target_col) - nw.col(model)).abs()
        abs_target = _zero_to_nan(nw.col(target_col)).abs()
        return (abs_err / abs_target).alias(model)

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


@_base_docstring
def smape(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Symmetric Mean Absolute Percentage Error (SMAPE)

    SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 100% which is desirable compared to normal MAPE that
    may be undetermined when the target is zero."""

    def gen_expr(model):
        abs_err = (nw.col(model) - nw.col(target_col)).abs()
        denominator = nw.col(model).abs() + nw.col(target_col).abs()
        return (
            nw.when(denominator == 0)
            .then(0.0)
            .otherwise(abs_err / denominator)
            .alias(model)
        )

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def mase(
    df: IntoDataFrameT,
    models: List[str],
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://robjhyndman.com/papers/mase.pdf
    """
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)

    def scale_expr(_m):
        lagged = nw.col(target_col).shift(seasonality).over(group_cols)
        return (nw.col(target_col) - lagged).abs().alias("scale")

    mae_df = mae(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    train_df = _create_train_with_cutoffs(
        train_df=train_df,
        df=df,
        id_col=id_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )

    scales = _nw_agg_expr(
        df=train_df,
        models=["unused"],
        id_col=id_col,
        gen_expr=scale_expr,
        cutoff_col=cutoff_col,
    )
    return _scale_loss(
        df=mae_df,
        models=models,
        scales=scales,
        id_col=id_col,
        cutoff_col=cutoff_col,
    )


def rmae(
    df: IntoDataFrameT,
    models: List[str],
    baseline: str,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    df = nw.from_native(df)
    if df[baseline].is_null().any():
        raise ValueError(f"baseline model ({baseline}) contains NaNs.")
    mae_df = mae(df, models, id_col, target_col, cutoff_col)
    scales = (
        nw.from_native(mae(df, [baseline], id_col, target_col, cutoff_col))
        .rename({baseline: "scale"})
        .with_columns(scale=_zero_to_nan(nw.col("scale")))
    )
    return _scale_loss(
        df=mae_df,
        models=models,
        scales=scales,
        id_col=id_col,
        cutoff_col=cutoff_col,
    )


@_base_docstring
def nd(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Normalized Deviation (ND)

    ND measures the relative prediction
    accuracy of a forecasting method by calculating the
    sum of the absolute deviation of the prediction and the true
    value at a given time and dividing it by the sum of the absolute
    value of the ground truth."""

    def gen_expr(model):
        return ((nw.col(target_col) - nw.col(model)).abs()).alias(model)

    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)

    return (
        nw.from_native(df)
        .select(
            *group_cols,
            nw.col(target_col).abs().alias("scale"),
            *[gen_expr(m) for m in models],
        )
        .group_by(*group_cols)
        .agg(nw.all().sum())
        .select(
            *group_cols, *[(nw.col(m) / _zero_to_nan(nw.col("scale"))) for m in models]
        )
        .sort(*group_cols)
        .to_native()
    )


@_base_docstring
def wape(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Weighted Absolute Percentage Error (WAPE)

    WAPE = sum(|y - ŷ|) / sum(|y|)

    Unlike MAPE, which averages per-point percentage errors, WAPE normalizes
    the total absolute error by the total absolute value of the actuals. This
    makes it robust to zero actuals and gives more weight to higher-volume
    points.

    WAPE is mathematically identical to the Normalized Deviation (`nd`); it is
    also known as wMAPE or the MAD/Mean ratio. It is exposed under the WAPE
    name -- the most common name for this metric in business forecasting -- and
    delegates to `nd`."""
    return nd(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )


def msse(
    df: IntoDataFrameT,
    models: List[str],
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://otexts.com/fpp3/accuracy.html
    """
    mse_df = mse(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    train_df = _create_train_with_cutoffs(
        train_df=train_df,
        df=df,
        id_col=id_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )
    train_group_cols = _get_group_cols(
        df=train_df, id_col=id_col, cutoff_col=cutoff_col
    )
    baseline = train_df.with_columns(
        scale=nw.col(target_col).shift(seasonality).over(*train_group_cols)
    )
    scales = mse(
        df=baseline,
        models=["scale"],
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return _scale_loss(
        df=mse_df,
        scales=scales,
        models=models,
        id_col=id_col,
        cutoff_col=cutoff_col,
    )


def rmsse(
    df: IntoDataFrameT,
    models: List[str],
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> IntoDataFrameT:
    res = msse(
        df,
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        time_col=time_col,
    )
    return (
        nw.from_native(res)
        .with_columns(*[nw.col(m).sqrt() for m in models])
        .to_native()
    )


rmsse.__doc__ = msse.__doc__.replace(  # type: ignore[union-attr]
    "Mean Squared Scaled Error (MSSE)", "Root Mean Squared Scaled Error (RMSSE)"
)


def quantile_loss(
    df: IntoDataFrameT,
    models: Dict[str, str],
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """

    def gen_expr(model):
        model_name, pred_col = model
        delta_y = nw.col(target_col) - nw.col(pred_col)
        return nw.max_horizontal(
            (q * delta_y).alias("a"), ((q - 1) * delta_y).alias("b")
        ).alias(model_name)

    return _nw_agg_expr(
        df=df,
        models=list(models.items()),
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def scaled_quantile_loss(
    df: IntoDataFrameT,
    models: Dict[str, str],
    seasonality: int,
    train_df: IntoDataFrameT,
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    qloss_df = quantile_loss(
        df=df,
        models=models,
        q=q,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    train_df = _create_train_with_cutoffs(
        train_df=train_df,
        df=df,
        id_col=id_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )
    train_group_cols = _get_group_cols(
        df=train_df, id_col=id_col, cutoff_col=cutoff_col
    )
    baseline = train_df.with_columns(
        scale=nw.col(target_col).shift(seasonality).over(*train_group_cols)
    )
    scales = mae(
        df=baseline,
        models=["scale"],
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return _scale_loss(
        df=qloss_df,
        scales=scales,
        models=list(models.keys()),
        id_col=id_col,
        cutoff_col=cutoff_col,
    )


def mqloss(
    df: IntoDataFrameT,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """

    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)

    # Not the most efficient implementation
    # Sort quantiles and reorder forecasts to match
    # This ensures correct pairing regardless of input order
    sorted_indices = np.argsort(quantiles)
    sorted_quantiles = quantiles[sorted_indices]

    # Reorder forecast columns according to sorted quantile indices
    sorted_models = {
        model: [forecasts[i] for i in sorted_indices]
        for model, forecasts in models.items()
    }

    # Map each sorted quantile to its corresponding forecast column
    quantile_preds = {}
    for q, idx in zip(sorted_quantiles, range(len(sorted_quantiles))):
        quantile_preds[q] = {
            model: forecasts[idx] for model, forecasts in sorted_models.items()
        }

    res = nw.concat(
        [
            nw.from_native(
                quantile_loss(
                    df,
                    models=quantile_preds[q],
                    q=q,
                    id_col=id_col,
                    target_col=target_col,
                    cutoff_col=cutoff_col,
                )
            )
            for q in sorted_quantiles
        ]
    )
    res = res.group_by(group_cols).agg(
        [nw.col(col).mean().alias(col) for col in res.columns if col not in group_cols]
    )
    res = res.to_native()

    return res


def scaled_mqloss(
    df: IntoDataFrameT,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    mql_df = mqloss(
        df=df,
        models=models,
        quantiles=quantiles,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    train_df = _create_train_with_cutoffs(
        train_df=train_df,
        df=df,
        id_col=id_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )
    train_group_cols = _get_group_cols(
        df=train_df, id_col=id_col, cutoff_col=cutoff_col
    )
    baseline = train_df.with_columns(
        scale=nw.col(target_col).shift(seasonality).over(*train_group_cols)
    )
    scales = mae(
        df=baseline,
        models=["scale"],
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return _scale_loss(
        df=mql_df,
        scales=scales,
        models=list(models.keys()),
        id_col=id_col,
        cutoff_col=cutoff_col,
    )


def coverage(
    df: IntoDataFrameT,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Coverage of y with y_hat_lo and y_hat_hi.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        level (int): Confidence level used for intervals.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """

    def gen_expr(model):
        return (
            nw.col(target_col)
            .is_between(nw.col(f"{model}-lo-{level}"), nw.col(f"{model}-hi-{level}"))
            .alias(model)
        )

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def winkler_score(
    df: IntoDataFrameT,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Winkler Score

    The Winkler score evaluates a prediction interval by rewarding narrow
    intervals and penalizing observations that fall outside them. For a
    100(1-alpha)% prediction interval [lo, hi] and an observation y, with
    alpha = 1 - level/100, it is defined as:

    - The interval width (hi - lo), if the observation lies within the interval
      (between lo and hi, inclusive).
    - (hi - lo) + (2/alpha)(lo - y), if the observation lies below the interval.
    - (hi - lo) + (2/alpha)(y - hi), if the observation lies above the interval.

    Lower scores are better. Intervals that are both narrow and well calibrated
    are rewarded, while observations outside the interval incur on a penalty.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        level (int): Confidence level used for intervals. Must satisfy 0 < level < 100.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://otexts.com/fpppy/05-toolbox.html#winkler-score
        [2] Winkler, R. L. (1972). A decision-theoretic approach to interval estimation.
    """
    if level <= 0 or level >= 100:
        raise ValueError(f"`level` must satisfy 0 < level < 100, but got {level}.")
    alpha = (100 - level) / 100

    def gen_expr(model):
        lo = nw.col(f"{model}-lo-{level}")
        hi = nw.col(f"{model}-hi-{level}")
        y = nw.col(target_col)
        width = hi - lo
        penalty = (
            nw.when(y < lo)
            .then((2 / alpha) * (lo - y))
            .otherwise(nw.when(y > hi).then((2 / alpha) * (y - hi)).otherwise(0.0))
        )
        return (width + penalty).alias(model)

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def calibration(
    df: IntoDataFrameT,
    models: Dict[str, str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """
    Fraction of y that is lower than the model's predictions.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """

    def gen_expr(model):
        model_name, q_preds = model
        return (nw.col(target_col) <= nw.col(q_preds)).alias(model_name)

    return _nw_agg_expr(
        df=df,
        models=list(models.items()),
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def scaled_crps(
    df: IntoDataFrameT,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://proceedings.mlr.press/v139/rangapuram21a.html
    """
    df = nw.from_native(df)
    eps = np.finfo(np.float64).eps
    quantiles = np.asarray(quantiles)
    loss = mqloss(
        df=df,
        models=models,
        quantiles=quantiles,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )

    def gen_expr(model):
        return (2 * nw.col(model) * nw.col("counts") / (nw.col("norm") + eps)).alias(
            model
        )

    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    stats = (
        df.with_columns(nw.col(target_col).abs())
        .group_by(*group_cols)
        .agg(
            counts=nw.col(id_col).len(),
            norm=nw.col(target_col).sum(),
        )
    )
    return _nw_agg_expr(
        df=nw.from_native(loss).join(stats, on=group_cols),
        models=list(models.keys()),
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def tweedie_deviance(
    df: IntoDataFrameT,
    models: List[str],
    power: float = 1.5,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
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
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: DataFrame with one row per id and one column per model, containing the mean Tweedie deviance.

    References:
        [1] https://en.wikipedia.org/wiki/Tweedie_distribution

    """
    if power < 0:
        raise ValueError("Power must be non-negative.")
    df = nw.from_native(df)
    if power >= 2 and (df[target_col] <= 0).any():
        raise ValueError(
            f"Power {power} requires all target values to be strictly positive."
        )
    if any((df[m] <= 0).any() for m in models):
        raise ValueError(
            "All predictions must be strictly positive for Tweedie deviance."
        )

    target = nw.col(target_col)

    def gen_expr(model):
        pred = nw.col(model)
        if power == 0:
            deviance = (target - pred) ** 2
        elif power == 1:
            # y * log(y / mu) is taken as 0 at y == 0, its limit there
            deviance = 2 * (
                nw.when(target > 0).then(target * (target / pred).log()).otherwise(0.0)
                - target
                + pred
            )
        elif power == 2:
            ratio = target / pred
            deviance = 2 * (ratio - ratio.log() - 1)
        else:
            # every term of the unit deviance carries the factor of 2
            deviance = 2 * (
                target.clip(0) ** (2 - power) / ((1 - power) * (2 - power))
                + pred ** (2 - power) / (2 - power)
                - target * pred ** (1 - power) / (1 - power)
            )
        return deviance.alias(model)

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


@_base_docstring
def linex(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    a: float = 1.0,
) -> IntoDataFrameT:
    """Linex Loss (Linear Exponential)

    The Linex loss penalizes over- and under-forecasting
    asymmetrically depending on the parameter a.

    - If a > 0, under-forecasting ($y > \hat{y}$) is penalized more.
    - If a < 0, over-forecasting ($\hat{y} > y$) is penalized more.
    - a must not be 0.

    Args:
        a (float, optional): Asymmetry parameter. Must be non-zero. Defaults to 1.0.
    """
    if np.isclose(a, 0.0):
        raise ValueError("Parameter a in Linex loss must be non-zero.")

    def gen_expr(model):
        error = nw.col(target_col) - nw.col(model)
        return ((error * a).exp() - error * a - 1).alias(model)

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )


def _consecutive_forecast_pairs(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
) -> tuple[nw.DataFrame, nw.DataFrame]:
    """Pair each forecast with its revision from the next cutoff, for the same date.

    For every model column `m`, the pairs hold `m` (the forecast issued at one cutoff)
    and `m_update` (the forecast issued at the next cutoff of the same serie), joined on
    `id_col` and `time_col`, with `cutoff_col` set to the cutoff of the revision. Dates
    covered by only one cutoff are dropped. Also returns every (id, cutoff) in `df`.
    """
    df = nw.from_native(df)
    if cutoff_col not in df.columns:
        raise ValueError(
            f"Forecast volatility compares forecasts across cutoffs, but `df` has no "
            f"'{cutoff_col}' column. Pass cross-validation output."
        )
    cutoffs = (
        df.select(id_col, cutoff_col)
        .unique()
        .with_columns(
            _fcd=nw.col(cutoff_col).rank(method="dense").over(id_col).cast(nw.Int64)
        )
    )
    _warn_uneven_cutoffs(cutoffs, id_col=id_col, cutoff_col=cutoff_col)
    df = df.join(cutoffs, on=[id_col, cutoff_col])
    before = df.select(id_col, time_col, "_fcd", target_col, *models)
    update = df.select(
        id_col,
        time_col,
        cutoff_col,
        (nw.col("_fcd") - 1).alias("_fcd"),
        *[nw.col(m).alias(f"{m}_update") for m in models],
    )
    pairs = before.join(update, on=[id_col, time_col, "_fcd"])
    if pairs.shape[0] == 0:
        raise ValueError(
            "No target date is forecasted from two consecutive cutoffs, so forecast "
            "volatility is undefined. Run cross-validation with more than one window "
            "and a `step_size` smaller than the horizon."
        )
    return pairs, cutoffs.select(cutoff_col, id_col)


def _warn_uneven_cutoffs(cutoffs: nw.DataFrame, id_col: str, cutoff_col: str) -> None:
    """Warn about series whose cutoffs are unevenly spaced, e.g. a missing window."""
    previous = cutoffs.select(
        id_col,
        (nw.col("_fcd") + 1).alias("_fcd"),
        nw.col(cutoff_col).alias("_previous_cutoff"),
    )
    gaps = (
        cutoffs.join(previous, on=[id_col, "_fcd"])
        .select(id_col, (nw.col(cutoff_col) - nw.col("_previous_cutoff")).alias("_gap"))
        .group_by(id_col)
        .agg(nw.col("_gap").n_unique())
        .filter(nw.col("_gap") > 1)
        .sort(id_col)
    )
    if gaps.shape[0] > 0:
        uneven = gaps[id_col].to_list()
        warnings.warn(
            f"Cutoffs are not evenly spaced for {len(uneven)} series (e.g. "
            f"{uneven[:3]}), so some revisions span more than one step. Each forecast "
            "is compared with the next available cutoff on the dates both forecast.",
            UserWarning,
            stacklevel=4,
        )


def _with_all_cutoffs(
    res: nw.DataFrame, cutoffs: nw.DataFrame, id_col: str, cutoff_col: str
) -> IntoDataFrameT:
    """Give every (id, cutoff) a row, leaving the first cutoff of each serie empty."""
    return (
        cutoffs.join(res, on=[cutoff_col, id_col], how="left")
        .sort(cutoff_col, id_col)
        .to_native()
    )


def _valid(expr: nw.Expr) -> nw.Expr:
    """True where `expr` is neither null nor NaN, since polars keeps NaN as a value."""
    return ~(expr.is_null() | expr.is_nan())


def _pinball_expr(target: nw.Expr, pred: nw.Expr, q: float) -> nw.Expr:
    error = target - pred
    return nw.max_horizontal(q * error, (q - 1) * error)


def excess_volatility(
    df: IntoDataFrameT,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Scaled Excess Volatility (sEV)

    Measures harmful forecast volatility across consecutive cutoffs. When forecast
    windows overlap, the same target date is forecasted from several cutoffs. Each
    revision is charged for how far it moves the forecast (revision cost) and credited
    for the accuracy it buys, so only revisions that move away from the truth, or
    overshoot it, are penalized.

    For a forecast $\\hat{y}_1$ and its revision $\\hat{y}_2$ from the next cutoff:
    $EV = QL(\\hat{y}_2, \\hat{y}_1) - (QL(y, \\hat{y}_1) - QL(y, \\hat{y}_2))$,
    averaged over `quantiles`. Values above 0 mean the revisions cost more than the
    accuracy they bought.

    Each row sums the EV of the revisions made at one cutoff and divides it by a
    per-serie scale, $\\sum |y|$ over the serie's pairs over its number of scored
    cutoffs, so a serie's rows average to its sEV, $\\sum EV / \\sum |y|$. Averaging
    across series weights them equally, unlike the pooled sEV of [1]. A missing
    cutoff is skipped, comparing with the next one on the same dates, with a warning.

    Args:
        df (pandas or polars DataFrame): Cross-validation output with id, times, cutoffs, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep. Defaults to 'ds'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and cutoff and one column per model.
            A row scores the revisions made at that cutoff, so the first cutoff of each serie is null.

    References:
        [1] https://openreview.net/forum?id=dXdycy7WCX
    """
    quantiles = np.asarray(quantiles)
    cols = [c for forecasts in models.values() for c in forecasts]
    pairs, cutoffs = _consecutive_forecast_pairs(
        df=df,
        models=cols,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    y = nw.col(target_col)
    exprs = []
    for model, forecasts in models.items():
        ev = sum(
            _pinball_expr(nw.col(f"{c}_update"), nw.col(c), q)
            - _pinball_expr(y, nw.col(c), q)
            + _pinball_expr(y, nw.col(f"{c}_update"), q)
            for c, q in zip(forecasts, quantiles)
        ) / len(quantiles)
        # only score and scale by the dates where this model's pair is complete
        valid = _valid(ev)
        exprs.append(nw.when(valid).then(ev).alias(model))
        exprs.append(nw.when(valid).then(y.abs()).alias(f"_{model}_norm"))
        exprs.append(nw.when(valid).then(1).alias(f"_{model}_pairs"))
    eps = np.finfo(np.float64).eps
    res = (
        pairs.select(cutoff_col, id_col, *exprs)
        .group_by(cutoff_col, id_col)
        .agg(nw.all().sum())
    )

    def scaled(model):
        # sum of |y| over the serie's pairs, split evenly across its scored cutoffs
        scored = nw.col(f"_{model}_pairs") > 0
        n_scored = scored.cast(nw.Int64).sum().over(id_col)
        scale = nw.col(f"_{model}_norm").sum().over(id_col) / n_scored
        return nw.when(scored).then(nw.col(model) / (scale + eps)).alias(model)

    res = res.select(cutoff_col, id_col, *[scaled(m) for m in models.keys()])
    return _with_all_cutoffs(res, cutoffs, id_col, cutoff_col)


def forecast_percentage_change(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Symmetric Forecast Percentage Change (sFPC)

    Measures the relative change of the forecasts across consecutive cutoffs. When
    forecast windows overlap, the same target date is forecasted from several
    cutoffs. For a forecast $\\hat{y}_1$ and its revision $\\hat{y}_2$ from the next
    cutoff, sFPC averages $200 |\\hat{y}_2 - \\hat{y}_1| / (|\\hat{y}_1| + |\\hat{y}_2|)$
    over the pairs revised at each cutoff. Unlike `excess_volatility`, every revision
    is penalized, even the ones that improve accuracy, so the two are best read
    together. A missing cutoff is skipped, comparing with the next one on the same
    dates, with a warning.

    To evaluate a probabilistic forecast pass the median column, since the outer
    quantiles are expected to shift inward as the uncertainty shrinks.

    Args:
        df (pandas or polars DataFrame): Cross-validation output with id, times, cutoffs, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep. Defaults to 'ds'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and cutoff and one column per model.
            A row scores the revisions made at that cutoff, so the first cutoff of each serie is null.

    References:
        [1] https://openreview.net/forum?id=dXdycy7WCX
    """
    pairs, cutoffs = _consecutive_forecast_pairs(
        df=df,
        models=models,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )

    def gen_expr(model):
        before = nw.col(model)
        update = nw.col(f"{model}_update")
        denominator = update.abs() + before.abs()
        change = (
            nw.when(denominator == 0)
            .then(0.0)
            .otherwise(200 * (update - before).abs() / denominator)
        )
        return nw.when(_valid(change)).then(change).alias(model)

    res = _nw_agg_expr(
        df=pairs,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
        cutoff_col=cutoff_col,
    )
    return _with_all_cutoffs(nw.from_native(res), cutoffs, id_col, cutoff_col)
