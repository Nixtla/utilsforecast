"""Create exogenous regressors for your models"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/feature_engineering.ipynb.

# %% auto 0
__all__ = ['fourier', 'trend', 'time_features', 'future_exog_to_historic', 'pipeline']

# %% ../nbs/feature_engineering.ipynb 3
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp
from .compat import DFType, DataFrame, pl, pl_DataFrame, pl_Expr
from .validation import validate_format, validate_freq

# %% ../nbs/feature_engineering.ipynb 4
_Features = Tuple[List[str], np.ndarray, np.ndarray]


def _add_features(
    df: DFType,
    freq: Union[str, int],
    h: int,
    id_col: str,
    time_col: str,
    f: Callable[[np.ndarray, int], _Features],
) -> Tuple[DFType, DFType]:
    # validations
    if not isinstance(h, int) or h < 0:
        raise ValueError("`h` must be a non-negative integer")
    validate_format(df, id_col, time_col, None)
    validate_freq(df[time_col], freq)

    # decompose series
    id_counts = ufp.counts_by_id(df, id_col)
    uids = id_counts[id_col]
    sizes = id_counts["counts"].to_numpy()

    # compute values
    cols, vals, future_vals = f(sizes=sizes, h=h)  # type: ignore

    # assign back to df
    sort_idxs = ufp.maybe_compute_sort_indices(df, id_col, time_col)
    times = df[time_col]
    if sort_idxs is not None:
        restore_idxs = np.empty_like(sort_idxs)
        restore_idxs[sort_idxs] = np.arange(sort_idxs.size)
        vals = vals[restore_idxs]
        times = ufp.take_rows(times, sort_idxs)
    last_times = ufp.take_rows(times, sizes.cumsum() - 1)
    df = ufp.copy_if_pandas(df, deep=False)
    transformed = ufp.assign_columns(df, cols, vals)

    if h == 0:
        return transformed, type(df)({})

    # future vals
    future_df = ufp.make_future_dataframe(
        uids=uids,
        last_times=last_times,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    future_df = ufp.assign_columns(future_df, cols, future_vals)
    return transformed, future_df


def _assign_slices(
    sizes: np.ndarray,
    feats: np.ndarray,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    max_samples, n_feats = feats.shape
    vals = np.empty((sizes.sum(), n_feats), dtype=np.float32)
    future_vals = np.empty((h * sizes.size, n_feats))
    start = 0
    for i, size in enumerate(sizes):
        vals[start : start + size, :] = feats[max_samples - size - h : max_samples - h]
        future_vals[i * h : (i + 1) * h] = feats[max_samples - h :]
        start += size
    return vals, future_vals


def _fourier(
    sizes: np.ndarray,
    h: int,
    season_length: int,
    k: int,
) -> _Features:
    # taken from: https://github.com/tblume1992/TSUtilities/blob/main/TSUtilities/TSFeatures/fourier_seasonality.py
    x = 2 * np.pi * np.arange(1, k + 1) / season_length
    x = x.astype(np.float32)
    t = np.arange(1, sizes.max() + 1 + h, dtype=np.float32)
    x = x * t[:, None]
    terms = np.hstack([np.sin(x), np.cos(x)])
    cols = [f"{op}{i+1}_{season_length}" for op in ("sin", "cos") for i in range(k)]
    vals, future_vals = _assign_slices(sizes=sizes, feats=terms, h=h)
    return cols, vals, future_vals


def _trend(sizes: np.ndarray, h: int) -> _Features:
    t = np.arange(1, sizes.max() + 1 + h, dtype=np.float32).reshape(-1, 1)
    cols = ["trend"]
    vals, future_vals = _assign_slices(sizes=sizes, feats=t, h=h)
    return cols, vals, future_vals

# %% ../nbs/feature_engineering.ipynb 5
def fourier(
    df: DFType,
    freq: Union[str, int],
    season_length: int,
    k: int,
    h: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Tuple[DFType, DFType]:
    """Compute fourier seasonal terms for training and forecasting

    Parameters
    ----------
    df : pandas or polars DataFrame
        Dataframe with ids, times and values for the exogenous regressors.
    freq : str or int
        Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
    season_length : int
        Number of observations per unit of time. Ex: 24 Hourly data.
    k : int
        Maximum order of the fourier terms
    h : int (default=0)
        Forecast horizon.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestep, its values can be timestamps or integers.

    Returns
    -------
    transformed_df : pandas or polars DataFrame
        Original DataFrame with the computed features
    future_df : pandas or polars DataFrame
        DataFrame with future values
    """
    f = partial(_fourier, season_length=season_length, k=k)
    return _add_features(
        df=df,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        f=f,
    )

# %% ../nbs/feature_engineering.ipynb 12
def trend(
    df: DFType,
    freq: Union[str, int],
    h: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Tuple[DFType, DFType]:
    """Add a trend column with consecutive integers for training and forecasting

    Parameters
    ----------
    df : pandas or polars DataFrame
        Dataframe with ids, times and values for the exogenous regressors.
    freq : str or int
        Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
    h : int (default=0)
        Forecast horizon.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestep, its values can be timestamps or integers.

    Returns
    -------
    transformed_df : pandas or polars DataFrame
        Original DataFrame with the computed features
    future_df : pandas or polars DataFrame
        DataFrame with future values
    """
    return _add_features(
        df=df,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        f=_trend,
    )

# %% ../nbs/feature_engineering.ipynb 15
def _compute_time_feature(
    times: Union[pd.Index, pl_Expr],
    feature: Union[str, Callable],
) -> Tuple[
    Union[str, List[str]],
    Union[pd.DataFrame, pl_Expr, List[pl_Expr], pd.Index, np.ndarray],
]:
    if callable(feature):
        feat_vals = feature(times)
        if isinstance(feat_vals, pd.DataFrame):
            feat_name = feat_vals.columns.tolist()
            feat_vals = feat_vals.to_numpy()
        else:
            feat_name = feature.__name__
    else:
        feat_name = feature
        if isinstance(times, pd.DatetimeIndex):
            if feature in ("week", "weekofyear"):
                times = times.isocalendar()
            feat_vals = getattr(times, feature).to_numpy()
        else:
            feat_vals = getattr(times.dt, feature)()
    return feat_name, feat_vals


def _add_time_features(
    df: DFType,
    features: List[Union[str, Callable]],
    time_col: str = "ds",
) -> DFType:
    df = ufp.copy_if_pandas(df, deep=False)
    unique_times = df[time_col].unique()
    if isinstance(df, pd.DataFrame):
        times = pd.Index(unique_times)
        time2pos = {time: i for i, time in enumerate(times)}
        restore_idxs = df[time_col].map(time2pos).to_numpy()
        for feature in features:
            name, vals = _compute_time_feature(times, feature)
            df[name] = vals[restore_idxs]
    elif isinstance(df, pl_DataFrame):
        exprs = []
        for feature in features:
            name, vals = _compute_time_feature(pl.col(time_col), feature)
            if isinstance(vals, list):
                exprs.extend(vals)
            else:
                assert isinstance(vals, pl_Expr)
                exprs.append(vals.alias(name))
        feats = unique_times.to_frame().with_columns(*exprs)
        df = df.join(feats, on=time_col, how="left")
    return df

# %% ../nbs/feature_engineering.ipynb 16
def time_features(
    df: DFType,
    freq: Union[str, int],
    features: List[Union[str, Callable]],
    h: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Tuple[DFType, DFType]:
    """Compute timestamp-based features for training and forecasting

    Parameters
    ----------
    df : pandas or polars DataFrame
        Dataframe with ids, times and values for the exogenous regressors.
    freq : str or int
        Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
    features : list of str or callable
        Features to compute. Can be string aliases of timestamp attributes or functions to apply to the times.
    h : int (default=0)
        Forecast horizon.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestep, its values can be timestamps or integers.

    Returns
    -------
    transformed_df : pandas or polars DataFrame
        Original DataFrame with the computed features
    future_df : pandas or polars DataFrame
        DataFrame with future values
    """
    transformed = _add_time_features(df=df, features=features, time_col=time_col)
    if h == 0:
        return transformed, type(df)({})
    times_by_id = ufp.group_by_agg(df, id_col, {time_col: "max"}, maintain_order=True)
    times_by_id = ufp.sort(times_by_id, id_col)
    future = ufp.make_future_dataframe(
        uids=times_by_id[id_col],
        last_times=times_by_id[time_col],
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    future = _add_time_features(df=future, features=features, time_col=time_col)
    return transformed, future

# %% ../nbs/feature_engineering.ipynb 19
def future_exog_to_historic(
    df: DFType,
    freq: Union[str, int],
    features: List[str],
    h: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Tuple[DFType, DFType]:
    """Turn future exogenous features into historic by shifting them `h` steps.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Dataframe with ids, times and values for the exogenous regressors.
    freq : str or int
        Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
    features : list of str
        Features to be converted into historic.
    h : int (default=0)
        Forecast horizon.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestep, its values can be timestamps or integers.

    Returns
    -------
    transformed_df : pandas or polars DataFrame
        Original DataFrame with the computed features
    future_df : pandas or polars DataFrame
        DataFrame with future values
    """
    if h == 0:
        return df, type(df)({})
    new_feats = ufp.copy_if_pandas(df[[id_col, time_col, *features]])
    new_feats = ufp.assign_columns(
        new_feats,
        time_col,
        ufp.offset_times(new_feats[time_col], freq=freq, n=h),
    )
    df = ufp.drop_columns(df, features)
    df = ufp.join(df, new_feats, on=[id_col, time_col], how="left")
    times_by_id = ufp.group_by_agg(df, id_col, {time_col: "max"}, maintain_order=True)
    times_by_id = ufp.sort(times_by_id, id_col)
    future = ufp.make_future_dataframe(
        uids=times_by_id[id_col],
        last_times=times_by_id[time_col],
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    future = ufp.join(future, new_feats, on=[id_col, time_col], how="left")
    return df, future

# %% ../nbs/feature_engineering.ipynb 25
def pipeline(
    df: DFType,
    features: List[Callable],
    freq: Union[str, int],
    h: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Tuple[DFType, DFType]:
    """Compute several features for training and forecasting

    Parameters
    ----------
    df : pandas or polars DataFrame
        Dataframe with ids, times and values for the exogenous regressors.
    features : list of callable
        List of features to compute. Must take only df, freq, h, id_col and time_col (other arguments must be fixed).
    freq : str or int
        Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
    h : int (default=0)
        Forecast horizon.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestep, its values can be timestamps or integers.

    Returns
    -------
    transformed_df : pandas or polars DataFrame
        Original DataFrame with the computed features
    future_df : pandas or polars DataFrame
        DataFrame with future values
    """
    transformed: Optional[DataFrame] = None
    future: Optional[DataFrame] = None
    for f in features:
        f_transformed, f_future = f(
            df=df, freq=freq, h=h, id_col=id_col, time_col=time_col
        )
        if transformed is None:
            transformed = f_transformed
            future = f_future
        else:
            feat_cols = [c for c in f_future.columns if c not in (id_col, time_col)]
            transformed = ufp.horizontal_concat([transformed, f_transformed[feat_cols]])
            future = ufp.horizontal_concat([future, f_future[feat_cols]])
    return transformed, future
