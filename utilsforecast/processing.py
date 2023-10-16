# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/processing.ipynb.

# %% auto 0
__all__ = ['assign_columns', 'take_rows', 'filter_with_mask', 'is_nan', 'is_none', 'is_nan_or_none', 'vertical_concat',
           'horizontal_concat', 'copy_if_pandas', 'join', 'drop_index_if_pandas', 'rename', 'sort', 'offset_dates',
           'group_by', 'DataFrameProcessor']

# %% ../nbs/processing.ipynb 2
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from .compat import DataFrame, Series, pl, pl_DataFrame, pl_Series
from .validation import validate_format

# %% ../nbs/processing.ipynb 4
def _polars_categorical_to_numerical(serie: pl_Series) -> pl_Series:
    if serie.dtype == pl.Categorical:
        serie = serie.to_physical()
    return serie

# %% ../nbs/processing.ipynb 5
def assign_columns(
    df: DataFrame,
    names: Union[str, List[str]],
    values: Union[np.ndarray, pd.Series, pl_Series],
) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df[names] = values
    else:
        is_scalar = isinstance(values, str) or not hasattr(values, "__len__")
        if is_scalar:
            assert isinstance(names, str)
            vals: Union[pl_DataFrame, pl_Series, pl.Expr] = pl.lit(values).alias(names)
        elif isinstance(values, pl_Series):
            assert isinstance(names, str)
            vals = values.alias(names)
        else:
            if isinstance(names, str):
                names = [names]
            vals = pl.from_numpy(values, schema=names)
        df = df.with_columns(vals)
    return df

# %% ../nbs/processing.ipynb 8
def take_rows(df: Union[DataFrame, Series], idxs: np.ndarray) -> DataFrame:
    if isinstance(df, (pd.DataFrame, pd.Series)):
        df = df.iloc[idxs]
    else:
        df = df[idxs]
    return df

# %% ../nbs/processing.ipynb 10
def filter_with_mask(
    df: Union[Series, DataFrame, pd.Index],
    mask: Union[np.ndarray, pd.Series, pl_Series],
) -> DataFrame:
    if isinstance(df, (pd.DataFrame, pd.Series, pd.Index)):
        out = df[mask]
    else:
        out = df.filter(mask)  # type: ignore
    return out

# %% ../nbs/processing.ipynb 11
def is_nan(s: Series) -> Series:
    if isinstance(s, pd.Series):
        out = s.isna()
    else:
        out = s.is_nan()
    return out

# %% ../nbs/processing.ipynb 13
def is_none(s: Series) -> Series:
    if isinstance(s, pd.Series):
        out = is_nan(s)
    else:
        out = s.is_null()
    return out

# %% ../nbs/processing.ipynb 15
def is_nan_or_none(s: Series) -> Series:
    return is_nan(s) | is_none(s)

# %% ../nbs/processing.ipynb 17
def vertical_concat(dfs: List[DataFrame]) -> DataFrame:
    if not dfs:
        raise ValueError("Can't concatenate empty list.")
    if isinstance(dfs[0], pd.DataFrame):
        out = pd.concat(dfs)
    elif isinstance(dfs[0], pl_DataFrame):
        out = pl.concat(dfs)
    else:
        raise ValueError(f"Got list of unexpected types: {type(dfs[0])}.")
    return out

# %% ../nbs/processing.ipynb 19
def horizontal_concat(dfs: List[DataFrame]) -> DataFrame:
    if not dfs:
        raise ValueError("Can't concatenate empty list.")
    if isinstance(dfs[0], pd.DataFrame):
        out = pd.concat(dfs, axis=1)
    elif isinstance(dfs[0], pl_DataFrame):
        out = pl.concat(dfs, how="horizontal")
    else:
        raise ValueError(f"Got list of unexpected types: {type(dfs[0])}.")
    return out

# %% ../nbs/processing.ipynb 21
def copy_if_pandas(df: DataFrame, deep: bool = False) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.copy(deep=deep)
    return df

# %% ../nbs/processing.ipynb 22
def join(
    df1: DataFrame, df2: DataFrame, on: Union[str, List[str]], how: str = "inner"
) -> DataFrame:
    if isinstance(df1, pd.DataFrame):
        out = df1.merge(df2, on=on, how=how)
    else:
        out = df1.join(df2, on=on, how=how)  # type: ignore
    return out

# %% ../nbs/processing.ipynb 23
def drop_index_if_pandas(df: DataFrame) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.reset_index(drop=True)
    return df

# %% ../nbs/processing.ipynb 24
def rename(df: DataFrame, mapping: Dict[str, str]) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns=mapping, copy=False)
    else:
        df = df.rename(mapping)
    return df

# %% ../nbs/processing.ipynb 25
def sort(df: DataFrame, by: Union[str, List[str]]) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        out = df.sort_values(by)
    else:
        out = df.sort(by)
    return out

# %% ../nbs/processing.ipynb 26
def offset_dates(
    dates: Union[pd.Index, pl_Series],
    freq: Union[int, str, BaseOffset],
    n: int,
):
    if isinstance(dates, (pd.DatetimeIndex, pd.Series, pd.Index)) and isinstance(
        freq, (int, BaseOffset)
    ):
        out = dates + n * freq
    elif isinstance(dates, pl_Series) and isinstance(freq, int):
        out = dates + n * freq
    elif isinstance(dates, pl_Series) and isinstance(freq, str):
        freq_n, freq_offset = re.findall(r"(\d+)(\w+)", freq)[0]
        total_n = int(freq_n) * n
        out = dates.dt.offset_by(f"{total_n}{freq_offset}")
    else:
        raise ValueError(
            f"Can't process the following combination {(type(dates), type(freq))}"
        )
    return out

# %% ../nbs/processing.ipynb 27
def group_by(df: Union[Series, DataFrame], by, maintain_order=False):
    if isinstance(df, (pd.Series, pd.DataFrame)):
        out = df.groupby(by, observed=True, sort=not maintain_order)
    else:
        if isinstance(df, pl_Series):
            df = df.to_frame()
        try:
            out = df.group_by(by, maintain_order=maintain_order)
        except AttributeError:
            out = df.groupby(by, maintain_order=maintain_order)
    return out

# %% ../nbs/processing.ipynb 28
class DataFrameProcessor:
    def __init__(
        self,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        """Class to  extract common structures from pandas and polars dataframes.

        Parameters
        ----------
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        """
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def counts_by_id(self, df: DataFrame) -> DataFrame:
        if isinstance(df, pd.DataFrame):
            id_counts = df.groupby(self.id_col, observed=True).size()
            id_counts = id_counts.reset_index()
            id_counts.columns = [self.id_col, "counts"]
        else:
            id_counts = df[self.id_col].value_counts().sort(self.id_col)
        return id_counts

    def value_cols_to_numpy(self, df: DataFrame) -> np.ndarray:
        exclude_cols = [self.id_col, self.time_col, self.target_col]
        value_cols = [col for col in df.columns if col not in exclude_cols]
        # ensure target is the first column
        value_cols = [self.target_col] + value_cols
        if isinstance(df, pd.DataFrame):
            dtypes = df.dtypes
            cat_cols = [
                c for c in value_cols if isinstance(dtypes[c], pd.CategoricalDtype)
            ]
            if cat_cols:
                df = df.copy(deep=False)
                for col in cat_cols:
                    df[col] = df[col].cat.codes
            data = df[value_cols].to_numpy()
        else:
            try:
                expr = pl.all().map_batches(_polars_categorical_to_numerical)
            except AttributeError:
                expr = pl.all().map(_polars_categorical_to_numerical)

            data = df[value_cols].select(expr).to_numpy()
        return data

    def maybe_compute_sort_indices(self, df: DataFrame) -> Optional[np.ndarray]:
        """Compute indices that would sort dataframe

        Parameters
        ----------
        df : pandas or polars DataFrame
            Input dataframe with id, times and target values.

        Returns
        -------
        numpy array or None
            Array with indices to sort the dataframe or None if it's already sorted.
        """
        if isinstance(df, pd.DataFrame):
            idx = pd.MultiIndex.from_arrays([df[self.id_col], df[self.time_col]])
        else:
            import polars as pl

            sort_idxs = df.select(
                pl.arg_sort_by([self.id_col, self.time_col]).alias("idx")
            )["idx"].to_numpy()
            idx = pd.Index(sort_idxs)
        if idx.is_monotonic_increasing:
            return None
        if isinstance(df, pd.DataFrame):
            sort_idxs = idx.argsort()
        return sort_idxs

    def process(
        self, df: DataFrame
    ) -> Tuple[Series, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract components from dataframe

        Parameters
        ----------
        df : pandas or polars DataFrame
            Input dataframe with id, times and target values.

        Returns
        -------
        ids : pandas or polars Serie
            serie with the sorted unique ids present in the data.
        last_times : numpy array
            array with the last time for each serie.
        data : numpy ndarray
            1d array with target values.
        indptr : numpy ndarray
            1d array with indices to the start and end of each serie.
        sort_idxs : numpy array or None
            array with the indices that would sort the original data.
            If the data is already sorted this is `None`.
        """
        # validations
        validate_format(df, self.id_col, self.time_col, self.target_col)

        # ids
        id_counts = self.counts_by_id(df)
        uids = id_counts[self.id_col]

        # indices
        indptr = np.append(
            np.int64(0),
            id_counts["counts"].to_numpy().cumsum().astype(np.int64),
        )
        last_idxs = indptr[1:] - 1

        # data
        data = self.value_cols_to_numpy(df)
        # ensure float dtype
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        # ensure 2d
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # check if we need to sort
        sort_idxs = self.maybe_compute_sort_indices(df)
        if sort_idxs is not None:
            data = data[sort_idxs]
            last_idxs = sort_idxs[last_idxs]
        times = df[self.time_col].to_numpy()[last_idxs]
        return uids, times, data, indptr, sort_idxs
