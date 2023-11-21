# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/validation.ipynb.

# %% auto 0
__all__ = ['ensure_shallow_copy', 'ensure_time_dtype', 'validate_format']

# %% ../nbs/validation.ipynb 2
import numpy as np
import pandas as pd

from .compat import DataFrame, Series, pl_DataFrame, pl_Series, pl

# %% ../nbs/validation.ipynb 5
def _is_dt_or_int(s: Series) -> bool:
    dtype = s.head(1).to_numpy().dtype
    is_dt = np.issubdtype(dtype, np.datetime64)
    is_int = np.issubdtype(dtype, np.integer)
    return is_dt or is_int

# %% ../nbs/validation.ipynb 6
def ensure_shallow_copy(df: pd.DataFrame) -> pd.DataFrame:
    from packaging.version import Version

    if Version(pd.__version__) < Version("1.4"):
        # https://github.com/pandas-dev/pandas/pull/43406
        df = df.copy()
    return df

# %% ../nbs/validation.ipynb 7
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

# %% ../nbs/validation.ipynb 10
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