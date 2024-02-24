# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/losses.ipynb.

# %% auto 0
__all__ = ['mae', 'mse', 'rmse', 'mape', 'smape', 'mase', 'rmae', 'quantile_loss', 'mqloss', 'coverage', 'calibration',
           'scaled_crps']

# %% ../nbs/losses.ipynb 3
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .compat import DataFrame, pl_DataFrame, pl
from .processing import group_by

# %% ../nbs/losses.ipynb 11
def _base_docstring(*args, **kwargs) -> Callable:
    base_docstring = """

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, actual values and predictions.
    models : list of str
        Columns that identify the models predictions.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.
    """

    def docstring_decorator(f: Callable):
        if f.__doc__ is not None:
            f.__doc__ += base_docstring
        return f

    return docstring_decorator(*args, **kwargs)

# %% ../nbs/losses.ipynb 12
def _pl_agg_expr(
    df: pl_DataFrame,
    models: List[str],
    id_col: str,
    gen_expr: Callable[[str], "pl.Expr"],
) -> pl_DataFrame:
    exprs = [gen_expr(model) for model in models]
    df = df.select([id_col, *exprs])
    res = group_by(df, id_col).mean()
    return res

# %% ../nbs/losses.ipynb 13
@_base_docstring
def mae(
    df: DataFrame,
    models: List[str],
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
            (df[models].sub(df[target_col], axis=0))
            .abs()
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return pl.col(target_col).sub(pl.col(model)).abs().alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 19
@_base_docstring
def mse(
    df: DataFrame,
    models: List[str],
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
            (df[models].sub(df[target_col], axis=0))
            .pow(2)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return pl.col(target_col).sub(pl.col(model)).pow(2).alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 24
@_base_docstring
def rmse(
    df: DataFrame,
    models: List[str],
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
    res = mse(df, models, id_col, target_col)
    if isinstance(res, pd.DataFrame):
        res[models] = res[models].pow(0.5)
    else:
        res = res.with_columns(*[pl.col(c).pow(0.5) for c in models])
    return res

# %% ../nbs/losses.ipynb 30
def _zero_to_nan(series: Union[pd.Series, "pl.Expr"]) -> Union[pd.Series, "pl.Expr"]:
    if isinstance(series, pd.Series):
        res = series.replace(0, np.nan)
    else:
        res = pl.when(series == 0).then(float("nan")).otherwise(series.abs())
    return res

# %% ../nbs/losses.ipynb 31
@_base_docstring
def mape(
    df: DataFrame,
    models: List[str],
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
            df[models]
            .sub(df[target_col], axis=0)
            .abs()
            .div(_zero_to_nan(df[target_col].abs()), axis=0)
            .fillna(0)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            abs_err = pl.col(target_col).sub(pl.col(model)).abs()
            abs_target = _zero_to_nan(pl.col(target_col))
            ratio = abs_err.truediv(abs_target).alias(model)
            return ratio.fill_nan(0)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 35
@_base_docstring
def smape(
    df: DataFrame,
    models: List[str],
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
    0% and 100% which is desirable compared to normal MAPE that
    may be undetermined when the target is zero."""
    if isinstance(df, pd.DataFrame):
        delta_y = df[models].sub(df[target_col], axis=0).abs()
        scale = df[models].abs().add(df[target_col].abs(), axis=0)
        raw = delta_y.div(scale).fillna(0)
        res = raw.groupby(df[id_col], observed=True).mean()
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            abs_err = pl.col(model).sub(pl.col(target_col)).abs()
            denominator = _zero_to_nan(
                pl.col(model).abs().add(pl.col(target_col)).abs()
            )
            ratio = abs_err.truediv(denominator).alias(model)
            return ratio.fill_nan(0)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 41
def mase(
    df: DataFrame,
    models: List[str],
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
    used in the M4 Competition.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    seasonality : int
        Main frequency of the time series;
        Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
    train_df : pandas or polars DataFrame
        Training dataframe with id and actual values. Must be sorted by time.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.

    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """
    mean_abs_err = mae(df, models, id_col, target_col)
    if isinstance(train_df, pd.DataFrame):
        mean_abs_err = mean_abs_err.set_index(id_col)
        # assume train_df is sorted
        lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
        scale = train_df[target_col].sub(lagged).abs()
        scale = scale.groupby(train_df[id_col], observed=True).mean()
        res = mean_abs_err.div(_zero_to_nan(scale), axis=0).fillna(0)
        res.index.name = id_col
        res = res.reset_index()
    else:
        # assume train_df is sorted
        lagged = pl.col(target_col).shift(seasonality).over(id_col)
        scale_expr = pl.col(target_col).sub(lagged).abs().alias("scale")
        scale = train_df.select([id_col, scale_expr])
        try:
            scale = scale.group_by(id_col).mean()
        except AttributeError:
            scale = scale.groupby(id_col).mean()
        scale = scale.with_columns(_zero_to_nan(pl.col("scale")))

        def gen_expr(model):
            return pl.col(model).truediv(pl.col("scale")).fill_nan(0).alias(model)

        full_df = mean_abs_err.join(scale, on=id_col, how="left")
        res = _pl_agg_expr(full_df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 46
def rmae(
    df: DataFrame,
    models: List[str],
    baseline_models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Relative Mean Absolute Error (RMAE)

    Calculates the RAME between two sets of forecasts (from two different forecasting methods).
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    baseline_models : list of str
        Columns that identify the baseline models predictions.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.
    """
    numerator = mae(df, models, id_col, target_col)
    denominator = mae(df, baseline_models, id_col, target_col)
    if isinstance(numerator, pd.DataFrame):
        res = numerator.merge(denominator, on=id_col, suffixes=("", "_denominator"))
        out_cols = [id_col]
        for model, baseline in zip(models, baseline_models):
            col_name = f"{model}_div_{baseline}"
            res[col_name] = (
                res[model].div(_zero_to_nan(res[f"{baseline}_denominator"])).fillna(0)
            )
            out_cols.append(col_name)
        res = res[out_cols]
    else:

        def gen_expr(model, baseline):
            denominator = _zero_to_nan(pl.col(f"{baseline}_denominator"))
            return (
                pl.col(model)
                .truediv(denominator)
                .fill_nan(0)
                .alias(f"{model}_div_{baseline}")
            )

        res = numerator.join(denominator, on=id_col, suffix="_denominator")
        exprs = [gen_expr(m1, m2) for m1, m2 in zip(models, baseline_models)]
        res = res.select([id_col, *exprs])
    return res

# %% ../nbs/losses.ipynb 52
def quantile_loss(
    df: DataFrame,
    models: List[str],
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Quantile Loss (QL)

    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    q : float (default=0.5)
        Quantile for the predictions' comparison.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.
    """
    if isinstance(df, pd.DataFrame):
        # we multiply by -1 because we want errors defined by y - y_hat
        delta_y = df[models].sub(df[target_col], axis=0) * (-1)
        res = (
            np.maximum(q * delta_y, (q - 1) * delta_y)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            # we multiply by -1 because we want errors defined by y - y_hat
            delta_y = pl.col(model).sub(pl.col(target_col)) * (-1)
            try:
                col_max = pl.max_horizontal([q * delta_y, (q - 1) * delta_y])
            except AttributeError:
                col_max = pl.max([q * delta_y, (q - 1) * delta_y])
            return col_max.alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 58
def mqloss(
    df: DataFrame,
    models: List[str],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
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
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    quantiles : numpy array
        Quantiles to compare against.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.

    References
    ----------
    [1] https://www.jstor.org/stable/2629907
    """
    res: Optional[DataFrame] = None
    quantiles = np.asarray(quantiles)
    for model in models:
        error = (df[target_col].to_numpy() - df[model].to_numpy()).reshape(-1, 1)
        loss = np.maximum(error * quantiles, error * (quantiles - 1)).mean(axis=1)
        result = type(df)({model: loss})
        if isinstance(result, pd.DataFrame):
            result = result.groupby(df[id_col], observed=True).mean()
        else:
            result = result.with_columns(df[id_col])
            result = group_by(result, id_col).mean()
        if res is None:
            res = result
            if isinstance(res, pd.DataFrame):
                res.index.name = id_col
                res = res.reset_index()
        else:
            if isinstance(res, pd.DataFrame):
                res = pd.concat([res, result], axis=1)
            else:
                res = res.join(result, on=id_col)
    return res

# %% ../nbs/losses.ipynb 63
def coverage(
    df: DataFrame,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Coverage of y with y_hat_lo and y_hat_hi.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    level : int
        Confidence level used for intervals.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.

    References
    ----------
    [1] https://www.jstor.org/stable/2629907
    """
    if isinstance(df, pd.DataFrame):
        out = np.empty((df.shape[0], len(models)))
        for j, model in enumerate(models):
            out[:, j] = df[target_col].between(
                df[f"{model}-lo-{level}"], df[f"{model}-hi-{level}"]
            )
        res = (
            pd.DataFrame(out, columns=models).groupby(df[id_col], observed=True).mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return (
                pl.col(target_col)
                .is_between(
                    pl.col(f"{model}-lo-{level}"), pl.col(f"{model}-hi-{level}")
                )
                .alias(model)
            )

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 67
def calibration(
    df: DataFrame,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """
    Fraction of y that is lower than y_hat_hi.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    level : int
        Confidence level used for intervals.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.

    References
    ----------
    [1] https://www.jstor.org/stable/2629907
    """
    if isinstance(df, pd.DataFrame):
        out = np.empty((df.shape[0], len(models)))
        for j, model in enumerate(models):
            out[:, j] = df[target_col].le(df[f"{model}-hi-{level}"])
        res = (
            pd.DataFrame(out, columns=models).groupby(df[id_col], observed=True).mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return pl.col(target_col).le(pl.col(f"{model}-hi-{level}")).alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res

# %% ../nbs/losses.ipynb 71
def scaled_crps(
    df: DataFrame,
    models: List[str],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.
    This metric averages percentual weighted absolute deviations as
    defined by the quantile losses.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    quantiles : numpy array
        Quantiles to compare against.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.

    References
    ----------
    [1] https://proceedings.mlr.press/v139/rangapuram21a.html
    """
    eps = np.finfo(float).eps
    quantiles = np.asarray(quantiles)
    loss = mqloss(df, models, quantiles, id_col, target_col)
    if isinstance(loss, pd.DataFrame):
        loss = loss.set_index(id_col)
        assert isinstance(df, pd.DataFrame)
        norm = df[target_col].abs().groupby(df[id_col], observed=True).sum()
        sizes = df[id_col].value_counts()
        scales = sizes * (sizes + 1) / 2
        res = 2 * loss.mul(scales, axis=0).div(norm + eps, axis=0)
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return (
                2 * pl.col(model) * pl.col("counts") / (pl.col("norm") + eps)
            ).alias(model)

        grouped_df = group_by(df, id_col)
        norm = grouped_df.agg(pl.col(target_col).abs().sum().alias("norm"))
        sizes = df[id_col].value_counts()
        sizes.columns = [id_col, "counts"]
        sizes = sizes.with_columns(pl.col("counts") * (pl.col("counts") + 1) / 2)
        res = _pl_agg_expr(
            loss.join(sizes, on=id_col).join(norm, on=id_col), models, id_col, gen_expr
        )
    return res
