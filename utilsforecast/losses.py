"""Loss functions for model evaluation."""

__all__ = ['mae', 'mse', 'rmse', 'bias', 'cfe', 'pis', 'spis', 'mape', 'smape', 'mase', 'rmae', 'nd', 'msse', 'rmsse',
           'quantile_loss', 'scaled_quantile_loss', 'mqloss', 'scaled_mqloss', 'coverage', 'calibration', 'scaled_crps',
           'tweedie_deviance']

from typing import Callable, Optional, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT


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


def _nw_agg_expr(
    df: IntoDataFrameT,
    models: Union[list[str], list[tuple[str, str]]],
    id_col: str,
    gen_expr: Callable[[Union[str, tuple[str, str]]], nw.Expr],
    agg: str = "mean",
) -> IntoDataFrameT:
    exprs = [gen_expr(model) for model in models]
    return (
        nw.from_native(df)
        .select([id_col, *exprs])
        .group_by(id_col)
        .agg(getattr(nw.all(), agg)())
        .to_native()   
    )


def _scale_loss(
    df: IntoDataFrameT,
    models: list[str],
    scales: IntoDataFrameT,
    id_col: str,
) -> IntoDataFrameT:
    exprs = [(nw.col(m) / nw.col("scale")).alias(m) for m in models]
    return (
        nw.from_native(df)
        .join(nw.from_native(scales), on=id_col)
        .select([id_col, *exprs])
        .to_native()
    )


@_base_docstring
def mae(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
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
        gen_expr=lambda m: (nw.col(target_col) - nw.col(m)).abs().alias(m)
    )


@_base_docstring
def mse(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
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
        gen_expr=lambda m: ((nw.col(target_col) - nw.col(m)) ** 2).alias(m)
    )


@_base_docstring
def rmse(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
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
    df = mse(df=df, models=models, id_col=id_col, target_col=target_col)
    return (
        nw.from_native(df)
        .with_columns(*[nw.col(m).sqrt() for m in models])
        .to_native()
    )


@_base_docstring
def bias(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
    """Forecast estimator bias.

    Defined as prediction - actual"""
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=lambda m: (nw.col(m) - nw.col(target_col)).alias(m)
    )


@_base_docstring
def cfe(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
    """
    Cumulative Forecast Error (CFE)

    Total signed forecast error per series. Positive values mean under forecast; negative mean over forecast.
    """
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=lambda m: (nw.col(m) - nw.col(target_col)).alias(m),
        agg="sum",
    )


@_base_docstring
def pis(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
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
        gen_expr=lambda m: (nw.col(m) - nw.col(target_col)).abs().alias(m),
        agg="sum",
    )


@_base_docstring
def spis(
    df: IntoDataFrameT,
    df_train: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
    """
    Compute the scaled Absolute Periods In Stock (sAPIS) for one or multiple models.

    The sPIS metric scales the sum of absolute forecast errors by the mean in-sample demand,
    yielding a scale-independent bias measure that can be aggregated across series.
    """
    df = nw.from_native(df)
    scales = df.group_by(id_col).agg(nw.col(target_col).mean().alias("scale"))
    raw = pis(df=df, models=models, id_col=id_col, target_col=target_col)
    return _scale_loss(df=raw, models=models, scales=scales, id_col=id_col)


def _zero_to_nan(series):
    return nw.when(series == 0).then(float("nan")).otherwise(series)


@_base_docstring
def mape(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
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
    )


@_base_docstring
def smape(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
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
        denominator = _zero_to_nan(
            nw.col(model).abs() + nw.col(target_col).abs()
        )
        return (abs_err / denominator).alias(model).fill_null(0)

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
    )


def mase(
    df: IntoDataFrameT,
    models: list[str],
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://robjhyndman.com/papers/mase.pdf
    """
    def scale_expr(_m):
        lagged = nw.col(target_col).shift(seasonality).over(id_col)
        return (nw.col(target_col) - lagged).abs().alias("scale")   
    
    mae_df = mae(df=df, models=models, id_col=id_col, target_col=target_col)
    scales = _nw_agg_expr(
        df=train_df,
        models=["unused"],
        id_col=id_col,
        gen_expr=scale_expr,
    )
    return _scale_loss(
        df=mae_df,
        models=models,
        scales=scales,
        id_col=id_col,
    )


def rmae(
    df: IntoDataFrameT,
    models: list[str],
    baseline: str,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    df = nw.from_native(df)
    if df[baseline].is_null().any():
        raise ValueError(f"baseline model ({baseline}) contains NaNs.")
    mae_df = mae(df, models, id_col, target_col)
    scales = (
        nw.from_native(mae(df, [baseline], id_col, target_col))
        .rename({baseline: "scale"})
        .with_columns(scale=_zero_to_nan(nw.col("scale")))
    )
    return _scale_loss(
        df=mae_df,
        models=models,
        scales=scales,
        id_col=id_col,
    )


def nd(
    df: IntoDataFrameT,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
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

    Returns:
        Dataframe with one row per id and one column per model.
    """
    def gen_expr(model):
        return (
            (nw.col(target_col) - nw.col(model)).abs() /
            _zero_to_nan(nw.col(target_col).abs())
        ).alias(model)
    
    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
    )


def msse(
    df: IntoDataFrameT,
    models: list[str],
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://otexts.com/fpp3/accuracy.html
    """
    mse_df = mse(df=df, models=models, id_col=id_col, target_col=target_col)
    baseline = (
        nw.from_native(train_df)
        .with_columns(scale=nw.col(target_col).shift(seasonality).over(id_col))
    )
    scales = mse(
        df=baseline, models=["scale"], id_col=id_col, target_col=target_col
    )
    return _scale_loss(
        df=mse_df,
        scales=scales,
        models=models,
        id_col=id_col,
    )


def rmsse(
    df: IntoDataFrameT,
    models: list[str],
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
    res = msse(
        df,
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
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
    models: dict[str, str],
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    def gen_expr(model):
        model_name, pred_col = model
        delta_y = nw.col(target_col) - nw.col(pred_col)
        return nw.max_horizontal(
            (q * delta_y).alias("a"),
            ((q - 1) * delta_y).alias("b")
        ).alias(model_name)

    return _nw_agg_expr(
        df=df,
        models=list(models.items()),
        id_col=id_col,
        gen_expr=gen_expr,
    )


def scaled_quantile_loss(
    df: IntoDataFrameT,
    models: dict[str, str],
    seasonality: int,
    train_df: IntoDataFrameT,
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    qloss_df = quantile_loss(
        df=df, models=models, q=q, id_col=id_col, target_col=target_col
    )
    baseline = (
        nw.from_native(train_df)
        .with_columns(scale=nw.col(target_col).shift(seasonality).over(id_col))
    )
    scales = mae(
        df=baseline, models=["scale"], id_col=id_col, target_col=target_col
    )
    return _scale_loss(
        df=qloss_df,
        scales=scales,
        models=models,
        id_col=id_col,
    )


def mqloss(
    df: IntoDataFrameT,
    models: dict[str, list[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    df = nw.from_native(df)
    losses = {id_col: df[id_col]}
    error = np.empty((df.shape[0], quantiles.size))
    for model, predictions in models.items():
        for j, q_preds in enumerate(predictions):
            error[:, j] = (df[target_col] - df[q_preds]).to_numpy()
        loss = np.maximum(error * quantiles, error * (quantiles - 1)).mean(axis=1)
        losses[model] = loss
    return _nw_agg_expr(
        df=nw.from_dict(losses),
        models=list(models.keys()),
        id_col=id_col,
        gen_expr=lambda m: nw.col(m),
    )


def scaled_mqloss(
    df: IntoDataFrameT,
    models: dict[str, list[str]],
    quantiles: np.ndarray,
    seasonality: int,
    train_df: IntoDataFrameT,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    mql_df = mqloss(
        df=df, models=models, quantiles=quantiles, id_col=id_col, target_col=target_col
    )
    baseline = (
        nw.from_native(train_df)
        .with_columns(scale=nw.col(target_col).shift(seasonality).over(id_col))
    )
    scales = mae(
        df=baseline, models=["scale"], id_col=id_col, target_col=target_col
    )
    return _scale_loss(
        df=mql_df,
        scales=scales,
        models=list(models.keys()),
        id_col=id_col,
    )


def coverage(
    df: IntoDataFrameT,
    models: list[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
    """Coverage of y with y_hat_lo and y_hat_hi.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        level (int): Confidence level used for intervals.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    def gen_expr(model):
        return (
            nw.col(target_col)
            .is_between(
                nw.col(f"{model}-lo-{level}"), nw.col(f"{model}-hi-{level}")
            )
            .alias(model)
        )

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
    )


def calibration(
    df: IntoDataFrameT,
    models: dict[str, str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> IntoDataFrameT:
    """
    Fraction of y that is lower than the model's predictions.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

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
    )


def scaled_crps(
    df: IntoDataFrameT,
    models: dict[str, list[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
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

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://proceedings.mlr.press/v139/rangapuram21a.html
    """
    df = nw.from_native(df)
    eps = np.finfo(np.float64).eps
    quantiles = np.asarray(quantiles)
    loss = mqloss(
        df=df, models=models, quantiles=quantiles, id_col=id_col, target_col=target_col
    )
    def gen_expr(model):
        return (
            2 * nw.col(model) * nw.col("counts") / (nw.col("norm") + eps)
        ).alias(model)

    stats = (
        df
        .with_columns(target_col=nw.col(target_col).abs())
        .group_by(id_col)
        .agg(
            counts=nw.col(id_col).len(),
            norm=nw.col(target_col).sum(),
        )
    )
    return _nw_agg_expr(
        df=nw.from_native(loss).join(stats, on=id_col),
        models=list(models.keys()),
        id_col=id_col,
        gen_expr=gen_expr,
    )


def tweedie_deviance(
    df: IntoDataFrameT,
    models: list[str],
    power: float = 1.5,
    id_col: str = "unique_id",
    target_col: str = "y",
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
                + (nw.col(target_col) / nw.col(model))
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
                - (
                    nw.col(target_col)
                    * (nw.col(model) ** (1 - power))
                    / (1 - power)
                )
                + (nw.col(model) ** (2 - power) / (2 - power))
            ).alias(model)

    return _nw_agg_expr(
        df=df,
        models=models,
        id_col=id_col,
        gen_expr=gen_expr,
    )
