# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/validation.ipynb.

# %% auto 0
__all__ = ['ensure_shallow_copy', 'ensure_time_dtype', 'validate_format', 'validate_freq']

# %% ../nbs/validation.ipynb 2
import re
from typing import Union

import numpy as np
import pandas as pd

from .compat import DataFrame, Series, pl_DataFrame, pl_Series, pl

# %% ../nbs/validation.ipynb 5
def _get_np_dtype(s: Union[Series, pd.Index]) -> type:
    if isinstance(s, (pd.Series, pd.Index)):
        dtype = s.dtype.type
    else:
        dtype = s.head(1).to_numpy().dtype.type
    return dtype

# %% ../nbs/validation.ipynb 8
def _is_int_dtype(dtype: type) -> bool:
    return np.issubdtype(dtype, np.integer)


def _is_dt_dtype(dtype: type) -> bool:
    return np.issubdtype(dtype, np.datetime64)

# %% ../nbs/validation.ipynb 9
def _is_dt_or_int(s: Series) -> bool:
    dtype = _get_np_dtype(s)
    return _is_dt_dtype(dtype) or _is_int_dtype(dtype)

# %% ../nbs/validation.ipynb 10
def ensure_shallow_copy(df: pd.DataFrame) -> pd.DataFrame:
    from packaging.version import Version

    if Version(pd.__version__) < Version("1.4"):
        # https://github.com/pandas-dev/pandas/pull/43406
        df = df.copy()
    return df

# %% ../nbs/validation.ipynb 11
def ensure_time_dtype(df: DataFrame, time_col: str = "ds") -> DataFrame:
    """Make sure that `time_col` contains timestamps or integers.
    If it contains strings, try to cast them as timestamps."""
    times = df[time_col]
    if _is_dt_or_int(times):
        return df
    parse_err_msg = (
        f"Failed to parse '{time_col}' from string to datetime. "
        "Please make sure that it contains valid timestamps or integers."
    )
    if isinstance(times, pd.Series) and pd.api.types.is_object_dtype(times):
        try:
            times = pd.to_datetime(times)
        except ValueError:
            raise ValueError(parse_err_msg)
        df = ensure_shallow_copy(df.copy(deep=False))
        df[time_col] = times
    elif isinstance(times, pl_Series) and times.dtype == pl.Utf8:
        try:
            times = times.str.to_datetime()
        except pl.exceptions.ComputeError:
            raise ValueError(parse_err_msg)
        df = df.with_columns(times)
    else:
        raise ValueError(f"'{time_col}' should have valid timestamps or integers.")
    return df

# %% ../nbs/validation.ipynb 14
def validate_format(
    df: DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> None:
    """Ensure DataFrame has expected format.

    Parameters
    ----------
    df : pandas or polars DataFrame
        DataFrame with time series in long format.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    time_col : str (default='ds')
        Column that identifies each timestamp.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    None
    """
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        raise ValueError(
            f"`df` must be either pandas or polars dataframe, got {type(df)}"
        )

    # required columns
    missing_cols = sorted({id_col, time_col, target_col} - set(df.columns))
    if missing_cols:
        raise ValueError(f"The following columns are missing: {missing_cols}")

    # time col
    if not _is_dt_or_int(df[time_col]):
        times_dtype = df[time_col].head(1).to_numpy().dtype
        raise ValueError(
            f"The time column ('{time_col}') should have either timestamps or integers, got '{times_dtype}'."
        )

    # target col
    target = df[target_col]
    if isinstance(target, pd.Series):
        is_numeric = np.issubdtype(target.dtype.type, np.number)
    else:
        is_numeric = target.is_numeric()
    if not is_numeric:
        raise ValueError(
            f"The target column ('{target_col}') should have a numeric data type, got '{target.dtype}')"
        )

# %% ../nbs/validation.ipynb 19
def validate_freq(
    times: Series,
    freq: Union[str, int],
) -> None:
    time_dtype = times.head(1).to_numpy().dtype
    if _is_int_dtype(time_dtype) and not isinstance(freq, int):
        raise ValueError(
            "Time column contains integers but the specified frequency is not an integer. "
            "Please provide a valid integer, e.g. `freq=1`"
        )
    if _is_dt_dtype(time_dtype) and isinstance(freq, int):
        raise ValueError(
            "Time column contains timestamps but the specified frequency is an integer. "
            "Please provide a valid pandas or polars offset, e.g. `freq='D'` or `freq='1d'`."
        )
    # try to catch pandas frequency in polars dataframe
    if isinstance(times, pl_Series) and isinstance(freq, str):
        missing_n = re.search(r"\d+", freq) is None
        uppercase = re.sub("\d+", "", freq).isupper()
        if missing_n or uppercase:
            raise ValueError(
                "You must specify a valid polars offset when using polars dataframes. "
                "You can find the available offsets in "
                "https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.dt.offset_by.html"
            )
