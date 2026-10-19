__all__ = [
    "to_numpy",
    "counts_by_id",
    "maybe_compute_sort_indices",
    "assign_columns",
    "drop_columns",
    "take_rows",
    "filter_with_mask",
    "is_nan",
    "is_none",
    "is_nan_or_none",
    "match_if_categorical",
    "vertical_concat",
    "horizontal_concat",
    "copy_if_pandas",
    "join",
    "drop_index_if_pandas",
    "rename",
    "sort",
    "offset_times",
    "offset_dates",
    "time_ranges",
    "repeat",
    "cv_times",
    "group_by",
    "group_by_agg",
    "is_in",
    "between",
    "fill_null",
    "cast",
    "value_cols_to_numpy",
    "make_future_dataframe",
    "anti_join",
    "ensure_sorted",
    "ProcessedDF",
    "process_df",
    "DataFrameProcessor",
    "backtest_splits",
    "add_insample_levels",
]


import re
import reprlib
import typing
import warnings
from typing import (
    Any,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from utilsforecast.validation import (
    _is_dt_dtype,
    _is_int_dtype,
    ensure_shallow_copy,
    validate_format,
)

from .compat import DataFrame, Series, pl, pl_DataFrame, pl_Series


def _polars_categorical_to_numerical(serie: pl_Series) -> pl_Series:
    if serie.dtype == pl.Categorical:
        serie = serie.to_physical()
    return serie


def to_numpy(df: DataFrame) -> np.ndarray:
    if isinstance(df, pd.DataFrame):
        cat_cols = [
            c
            for c, dtype in df.dtypes.items()
            if isinstance(dtype, pd.CategoricalDtype)
        ]
        if cat_cols:
            df = df.copy(deep=False)
            df = ensure_shallow_copy(df)
            for col in cat_cols:
                df[col] = df[col].cat.codes
        out = df.to_numpy()
    else:
        expr = pl.all().map_batches(_polars_categorical_to_numerical)
        out = df.select(expr).to_numpy(order="c")
    return out


def counts_by_id(df: DataFrame, id_col: str) -> DataFrame:
    id_counts: DataFrame
    if isinstance(df, pd.DataFrame):
        counts = df[id_col].value_counts(sort=False, dropna=True)
        ids = counts.index
        if isinstance(ids, pd.CategoricalIndex):
            # there's no observed argument in value_counts
            # so this can return unseen categories
            observed = (counts > 0).to_numpy()
            counts = counts[observed]
            sort_idxs = ids.codes[observed].argsort()
        else:
            sort_idxs = ids.argsort()
        id_counts = counts.iloc[sort_idxs].reset_index()
    else:
        id_counts = df[id_col].value_counts().sort(id_col)
    id_counts.columns = [id_col, "counts"]
    return id_counts


def maybe_compute_sort_indices(
    df: DataFrame, id_col: str, time_col: str
) -> Optional[np.ndarray]:
    """Compute indices that would sort the dataframe

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times and target values.

    Returns:
        numpy array or None: Array with indices to sort the dataframe or None if it's already sorted.
    """
    ids: Union[pl_Series, np.ndarray]
    times: Union[pl_Series, np.ndarray]
    if isinstance(df, pd.DataFrame):
        pd_ids = df[id_col]
        if isinstance(pd_ids.dtype, pd.CategoricalDtype):
            # we sort categoricals by their codes, this is also done in counts_by_id
            pd_ids = pd_ids.cat.codes
        # pandas series alignment makes this slow, cast to numpy
        ids = pd_ids.to_numpy()
        times = df[time_col].to_numpy()
    else:
        ids = df[id_col]
        times = df[time_col]
    ids_are_sorted = (ids[:-1] <= ids[1:]).all()
    if ids_are_sorted:
        times_are_sorted = (
            (
                (times[:-1] < times[1:])  # times are ascending
                | (ids[:-1] != ids[1:])  # except when the id changes
            ).all()
        )
        if times_are_sorted:
            return None
    if isinstance(df, pd.DataFrame):
        if pd.api.types.is_string_dtype(df.dtypes[id_col]):
            # MultiIndex.argsort is faster than lexsort for strings
            sort_idxs = pd.MultiIndex.from_frame(df[[id_col, time_col]]).argsort()
        else:
            sort_idxs = np.lexsort((times, ids))
    else:
        sort_idxs = (
            df.select(pl.arg_sort_by([id_col, time_col])).to_series(0).to_numpy()
        )
    return sort_idxs


def _is_sorted(df: DataFrame, id_col: str, time_col: str) -> bool:
    ids: Union[pl_Series, np.ndarray]
    times: Union[pl_Series, np.ndarray]
    if isinstance(df, pd.DataFrame):
        pd_ids = df[id_col]
        if pd_ids.hasnans:
            return False
        if isinstance(pd_ids.dtype, pd.CategoricalDtype):
            pd_ids = pd_ids.cat.codes
        ids = pd_ids.to_numpy()
        times = df[time_col].to_numpy()
    else:
        ids = df[id_col]
        if ids.has_nulls():
            return False
        times = df[time_col]
    try:
        ids_are_sorted = (ids[:-1] <= ids[1:]).all()
    except TypeError:
        return False
    if not ids_are_sorted:
        return False
    try:
        times_are_sorted = ((times[:-1] < times[1:]) | (ids[:-1] != ids[1:])).all()
    except TypeError:
        return False
    return bool(times_are_sorted)


def assign_columns(
    df: DataFrame,
    names: Union[str, List[str]],
    values: Union[np.ndarray, pd.Series, pl_Series, List[float], str, float],
) -> DataFrame:
    if isinstance(values, list) and (
        len(values) != df.shape[0] or not isinstance(names, str)
    ):
        raise ValueError("Only single column assignment is supported for lists.")
    if isinstance(df, pd.DataFrame):
        df[names] = values  # type: ignore[index, assignment]
    else:
        is_scalar = isinstance(values, str) or not hasattr(values, "__len__")
        if is_scalar:
            assert isinstance(names, str)
            vals: Union[pl_DataFrame, pl_Series, pl.Expr] = pl.lit(values).alias(names)
        elif isinstance(values, pl_Series):
            assert isinstance(names, str)
            vals = values.alias(names)
        else:
            if isinstance(values, np.ndarray):
                if isinstance(names, str):
                    names = [names]
                vals = pl.from_numpy(values, schema=names, orient="row")
            elif isinstance(values, list):
                assert isinstance(names, str)
                vals = pl_Series(name=names, values=values)
        df = df.with_columns(vals)
    return df


def drop_columns(df: DataFrame, columns: Union[str, List[str]]) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.drop(columns=columns)
    else:
        df = df.drop(columns)
    return df


def take_rows(df: Union[DataFrame, Series, np.ndarray], idxs: np.ndarray) -> DataFrame:
    if isinstance(df, (pd.DataFrame, pd.Series)):
        df = df.iloc[idxs]
    else:
        df = df[idxs]
    return df  # type: ignore[return-value]


def filter_with_mask(
    df: Union[Series, DataFrame, pd.Index, np.ndarray],
    mask: Union[np.ndarray, pd.Series, pl_Series],
) -> DataFrame:
    if isinstance(df, (pd.DataFrame, pd.Series, pd.Index, np.ndarray)):
        out = df[mask]  # type: ignore[index]
    else:
        out = df.filter(mask)  # type: ignore
    return out  # type: ignore[return-value]


def is_nan(s: Series) -> Series:
    out: Series
    if isinstance(s, pd.Series):
        out = s.isna()
    elif s.dtype.is_float():
        out = s.is_nan()
    else:
        # polars only supports is_nan on numeric columns and only floats can hold NaN
        out = pl.repeat(False, s.len(), eager=True).alias(s.name)
    return out


def is_none(s: Series) -> Series:
    if isinstance(s, pd.Series):
        out = is_nan(s)
    else:
        out = s.is_null()
    return out


def is_nan_or_none(s: Series) -> Series:
    return is_nan(s) | is_none(s)


def _match_pd_categoricals(
    s1: Union[pd.Series, pd.Index], s2: pd.Series
) -> Tuple[Union[pd.Series, pd.Index], pd.Series]:
    s1_dtype = s1.dtype
    assert isinstance(s1_dtype, pd.CategoricalDtype)
    cat1 = s1_dtype.categories
    s2_dtype = s2.dtype
    if isinstance(s2_dtype, pd.CategoricalDtype):
        cat2 = set(s2_dtype.categories)
    else:
        cat2 = set(s2.dropna().unique().astype(cat1.dtype))
    missing = cat2 - set(cat1)
    new_dtype = s1_dtype
    if missing:
        # we assume the original is s1, so we extend its categories
        new_dtype = pd.CategoricalDtype(
            categories=cat1.tolist() + sorted(missing), ordered=s1_dtype.ordered
        )
        s1 = s1.astype(new_dtype)
    # rebuild from the values, astype keeps s2's order if only the order differs
    new_s2 = pd.Series(
        pd.Categorical(s2, dtype=new_dtype), index=s2.index, name=s2.name
    )
    if new_s2.isna().sum() != s2.isna().sum():
        bad = s2[new_s2.isna() & s2.notna()].unique()[:5]
        raise ValueError(
            f"Values with dtype {s2.dtype} don't match the categories with dtype "
            f"{cat1.dtype}, e.g. {bad.tolist()}."
        )
    return s1, new_s2


def _match_pl_categoricals(s1: pl_Series, s2: pl_Series) -> Tuple[pl_Series, pl_Series]:
    with pl.StringCache():
        cat1 = s1.cat.get_categories()
        if s2.dtype == pl.Categorical:
            cat2 = s2.cat.get_categories()
        else:
            cat2 = s2.unique().sort().cast(cat1.dtype)
        # populate cache, keep original categories first
        pl.concat([cat1, cat2]).cast(pl.Categorical)
        s1 = s1.cast(pl.Utf8).cast(pl.Categorical)
        s2 = s2.cast(pl.Utf8).cast(pl.Categorical)
    return s1, s2


@overload
def match_if_categorical(s1: pd.Index, s2: Series) -> Tuple[pd.Index, Series]: ...
@overload
def match_if_categorical(s1: Series, s2: Series) -> Tuple[Series, Series]: ...
def match_if_categorical(
    s1: Union[Series, pd.Index], s2: Series
) -> Tuple[Union[Series, pd.Index], Series]:
    if isinstance(s1, (pd.Series, pd.Index)) and isinstance(
        s1.dtype, pd.CategoricalDtype
    ):
        assert isinstance(s2, pd.Series)
        return _match_pd_categoricals(s1, s2)
    if isinstance(s1, pl_Series) and s1.dtype == pl.Categorical:
        assert isinstance(s2, pl_Series)
        return _match_pl_categoricals(s1, s2)
    return s1, s2


def vertical_concat(
    dfs: List[Union[DataFrame, Series]], match_categories: bool = True
) -> Union[DataFrame, Series]:
    if not dfs:
        raise ValueError("Can't concatenate empty list.")
    out: Union[DataFrame, Series]
    if isinstance(dfs[0], pd.Series):
        out = pd.concat(typing.cast(List[pd.Series], dfs)).reset_index(drop=True)
    elif isinstance(dfs[0], pl_Series):
        out = pl.concat(typing.cast(List[pl_Series], dfs))
    elif isinstance(dfs[0], pd.DataFrame):
        pd_dfs = typing.cast(List[pd.DataFrame], dfs)
        cat_cols = [
            c
            for c, dtype in pd_dfs[0].dtypes.items()
            if isinstance(dtype, pd.CategoricalDtype)
        ]
        if match_categories and cat_cols:
            if len(pd_dfs) > 2:
                raise NotImplementedError(
                    "Categorical replacement for more than two dataframes"
                )
            assert len(pd_dfs) == 2
            df1, df2 = pd_dfs
            df1 = df1.copy(deep=False)
            df2 = df2.copy(deep=False)
            for col in cat_cols:
                s1, s2 = _match_pd_categoricals(df1[col], df2[col])
                df1[col] = s1
                df2[col] = s2
            pd_dfs = [df1, df2]
        out = pd.concat(pd_dfs).reset_index(drop=True)
    else:
        pl_dfs = typing.cast(List[pl_DataFrame], dfs)
        all_cols = pl_dfs[0].columns
        pl_cat_cols = [
            all_cols[i]
            for i, dtype in enumerate(pl_dfs[0].dtypes)
            if dtype == pl.Categorical
        ]
        if match_categories and pl_cat_cols:
            if len(pl_dfs) > 2:
                raise NotImplementedError(
                    "Categorical replacement for more than two dataframes"
                )
            assert len(pl_dfs) == 2
            pl_df1, pl_df2 = pl_dfs
            for col in pl_cat_cols:
                pl_s1, pl_s2 = _match_pl_categoricals(pl_df1[col], pl_df2[col])
                pl_df1 = pl_df1.with_columns(pl_s1)
                pl_df2 = pl_df2.with_columns(pl_s2)
            pl_dfs = [pl_df1, pl_df2]
        out = pl.concat(pl_dfs)
    return out


def horizontal_concat(dfs: List[DataFrame]) -> DataFrame:
    if not dfs:
        raise ValueError("Can't concatenate empty list.")
    out: DataFrame
    if isinstance(dfs[0], pd.DataFrame):
        out = pd.concat(typing.cast(List[pd.DataFrame], dfs), axis=1)
    elif isinstance(dfs[0], pl_DataFrame):
        out = pl.concat(typing.cast(List[pl_DataFrame], dfs), how="horizontal")
    else:
        raise ValueError(f"Got list of unexpected types: {type(dfs[0])}.")
    return out


def copy_if_pandas(df: DataFrame, deep: bool = False) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.copy(deep=deep)
        if not deep:
            df = ensure_shallow_copy(df)
    return df


def join(
    df1: Union[DataFrame, Series],
    df2: Union[DataFrame, Series],
    on: Union[str, List[str]],
    how: str = "inner",
) -> DataFrame:
    if isinstance(df1, (pd.Series, pl_Series)):
        df1 = df1.to_frame()
    if isinstance(df2, (pd.Series, pl_Series)):
        df2 = df2.to_frame()
    if isinstance(df1, pd.DataFrame):
        out = df1.merge(df2, on=on, how=how)  # type: ignore[arg-type]
    else:
        out = df1.join(df2, on=on, how=how)  # type: ignore
    return out


def drop_index_if_pandas(df: DataFrame) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.reset_index(drop=True)
    return df


def rename(df: DataFrame, mapping: Dict[str, str]) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns=mapping, copy=False)
    else:
        df = df.rename(mapping)
    return df


def sort(df: DataFrame, by: Optional[Union[str, List[str]]] = None) -> DataFrame:
    out: Union[DataFrame, Series, pd.Index]
    if isinstance(df, pd.DataFrame):
        out = df.sort_values(by).reset_index(drop=True)  # type: ignore[arg-type]
    elif isinstance(df, (pd.Series, pd.Index)):
        out = df.sort_values()
        if isinstance(out, pd.Series):
            out = out.reset_index(drop=True)
    elif isinstance(df, pl_DataFrame):
        out = df.sort(by)
    else:
        out = df.sort()
    return out  # type: ignore[return-value]


def _multiply_pl_freq(
    freq: str, n: Union[int, np.ndarray, Series]
) -> Union[str, pl_Series]:
    freq_n, freq_offset = re.findall(r"(\d+)(\w+)", freq)[0]
    freq_n = int(freq_n)
    if isinstance(n, int):
        total_n = freq_n * n
        out: Union[str, pl_Series] = f"{total_n}{freq_offset}"
    else:
        if not isinstance(n, pl_Series) or not n.dtype.is_integer():
            raise ValueError("`n` must be an integer or a polars series of integers.")
        out = (n * freq_n).cast(pl.Utf8) + freq_offset
    return out


def _ensure_month_ends(
    times: pl_Series, orig_times: pl_Series, freq: Union[str, int, BaseOffset]
) -> pl_Series:
    if not isinstance(freq, str) or "mo" not in freq:
        return times
    next_days = orig_times.dt.offset_by("1d")
    month_ends = (next_days.dt.month() != orig_times.dt.month()).all()
    if month_ends:
        times = times.dt.month_end()
    return times


_NANOS_PER_UNIT = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def _dt_unit(times: Union[pd.Series, pd.Index]) -> str:
    dtype = times.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return dtype.unit
    return np.datetime_data(dtype)[0]  # type: ignore[arg-type]


def _has_time_part(times: Union[pd.Series, pd.Index]) -> bool:
    values = times.to_numpy()
    values = values[~np.isnat(values)]
    return bool((values != values.astype("datetime64[D]")).any())


@overload
def _arrow_times_to_numpy(times: pd.Series) -> pd.Series: ...
@overload
def _arrow_times_to_numpy(times: pd.Index) -> pd.Index: ...
def _arrow_times_to_numpy(
    times: Union[pd.Series, pd.Index],
) -> Union[pd.Series, pd.Index]:
    import pyarrow as pa

    dtype = times.dtype
    assert isinstance(dtype, pd.ArrowDtype)
    arrow_dtype = dtype.pyarrow_dtype
    is_date64 = pa.types.is_date64(arrow_dtype)
    if pa.types.is_date(arrow_dtype):
        # pandas converts arrow dates to numpy elementwise, but not arrow timestamps
        unit = "s" if pa.types.is_date32(arrow_dtype) else "ms"
        arrow_dtype = pa.timestamp(unit)
        times = times.astype(pd.ArrowDtype(arrow_dtype))
    unit = arrow_dtype.unit
    numpy_dtype = np.dtype(f"datetime64[{unit}]")
    if arrow_dtype.tz is None:
        times = times.astype(numpy_dtype)
        # date64 stores milliseconds, so it can hold times that aren't at midnight
        if is_date64 and _has_time_part(times):
            raise ValueError(
                "Expected the 'date64' values to be at midnight, "
                "but some of them have a time part."
            )
        return times
    # pandas converts tz-aware arrow timestamps elementwise, so we convert the UTC values
    utc = times.astype(pd.ArrowDtype(pa.timestamp(unit))).astype(numpy_dtype)
    if isinstance(utc, pd.Series):
        return utc.dt.tz_localize("UTC").dt.tz_convert(arrow_dtype.tz)
    return pd.DatetimeIndex(utc).tz_localize("UTC").tz_convert(arrow_dtype.tz)


@overload
def _numpy_times_to_arrow(times: pd.Series, dtype: "pd.ArrowDtype") -> pd.Series: ...
@overload
def _numpy_times_to_arrow(times: pd.Index, dtype: "pd.ArrowDtype") -> pd.Index: ...
def _numpy_times_to_arrow(
    times: Union[pd.Series, pd.Index], dtype: "pd.ArrowDtype"
) -> Union[pd.Series, pd.Index]:
    import pyarrow as pa

    if pa.types.is_date(dtype.pyarrow_dtype):
        if _has_time_part(times):
            raise ValueError(
                f"Cannot represent the offset times as '{dtype}' "
                "because some of them aren't at midnight."
            )
        return times.astype(dtype)
    arrow_dtype = pa.timestamp(_dt_unit(times), tz=dtype.pyarrow_dtype.tz)
    return times.astype(pd.ArrowDtype(arrow_dtype))


def _offset_times_array_n(
    times: Union[pd.Series, pd.Index], freq: BaseOffset, n: np.ndarray
) -> Union[pd.Series, pd.Index]:
    n = np.asarray(n)
    if n.shape != (len(times),):
        raise ValueError(f"Expected n to have shape ({len(times)},), got {n.shape}.")
    if not np.issubdtype(n.dtype, np.integer):
        raise TypeError(f"Expected n to have an integer dtype, got {n.dtype}.")
    if len(times) == 0:
        return times.copy()
    if n.dtype.kind == "u" and n.max() > np.iinfo(np.int64).max:
        raise OverflowError("Expected n to fit in int64.")
    # a python int multiplier keeps n's dtype, so narrower ints would overflow
    n = n.astype(np.int64, copy=False)
    if isinstance(freq, pd.offsets.Tick):
        # pandas uses the finer of the times' and the offset's resolutions
        unit = min(
            _dt_unit(times),
            pd.Timedelta(freq).unit,  # type: ignore[arg-type]
            key=_NANOS_PER_UNIT.__getitem__,
        )
        step = freq.nanos // _NANOS_PER_UNIT[unit]
        bound = np.iinfo(np.int64).max // max(abs(step), 1)
        if n.max() > bound or n.min() < -bound:
            raise pd.errors.OutOfBoundsTimedelta(
                f"{freq} multiplied by n overflows timedelta64[{unit}]."
            )
        deltas = (n * step).astype(f"timedelta64[{unit}]")
        return times + deltas
    # pandas only vectorizes non-fixed offsets for a scalar n, so we apply each
    # distinct n to its own run of times and then restore the original order
    order = np.argsort(n, kind="stable")
    ks, starts = np.unique(n[order], return_index=True)
    ends = np.append(starts[1:], n.size)
    dt_index = pd.DatetimeIndex(times)
    with warnings.catch_warnings():
        # offsets without a vectorized implementation still fall back elementwise
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        runs = [
            dt_index[order[s:e]] + int(k) * freq for k, s, e in zip(ks, starts, ends)
        ]
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    out = runs[0].append(runs[1:]).take(inv_order)
    if isinstance(times, pd.Series):
        return pd.Series(out, index=times.index, name=times.name)
    return out.rename(times.name)


def offset_times(
    times: Union[Series, pd.Index],
    freq: Union[int, str, BaseOffset],
    n: Union[int, np.ndarray, Series],
) -> Union[Series, pd.Index]:
    out: Union[Series, pd.Index]
    if isinstance(times, (pd.Series, pd.Index)):
        if isinstance(freq, str):
            freq = pd.tseries.frequencies.to_offset(freq)
        ints = _is_int_dtype(times) and isinstance(freq, int)
        dts = _is_dt_dtype(times) and isinstance(freq, BaseOffset)
        if not ints and not dts:
            raise ValueError(
                f"Cannot offset times with data type: '{times.dtype}' "
                f"using a frequency of type: '{type(freq)}'."
            )
        orig_dtype = times.dtype
        arrow_dtype = (
            orig_dtype if dts and isinstance(orig_dtype, pd.ArrowDtype) else None
        )
        if arrow_dtype is not None:
            times = _arrow_times_to_numpy(times)
        if dts and np.ndim(n) > 0:
            out = _offset_times_array_n(
                times, typing.cast(BaseOffset, freq), np.asarray(n)
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                out = times + n * freq
        if arrow_dtype is not None:
            out = _numpy_times_to_arrow(out, arrow_dtype)
    elif isinstance(times, pl_Series) and isinstance(freq, int):
        out = times + n * freq
    elif isinstance(times, pl_Series) and isinstance(freq, str):
        total_offset = _multiply_pl_freq(freq, n)
        out = times.dt.offset_by(total_offset)
        out = _ensure_month_ends(out, times, freq)
    else:
        raise ValueError(
            f"Cannot offset times of type: '{type(times)}' "
            f"using a frequency of type: '{type(freq)}'."
        )
    return out


def offset_dates(
    dates: Union[Series, pd.Index],
    freq: Union[int, str, BaseOffset],
    n: Union[int, Series],
) -> Union[Series, pd.Index]:
    warnings.warn(
        "`offset_dates` has been renamed to `offset_times`", category=DeprecationWarning
    )
    return offset_times(dates, freq, n)


def time_ranges(
    starts: Union[Series, pd.Index],
    freq: Union[int, str, BaseOffset],
    periods: int,
) -> Series:
    out: Series
    if isinstance(starts, pd.Series):
        starts = pd.Index(starts)
    if isinstance(starts, pd.Index):
        if _is_int_dtype(starts):
            starts_np = starts.to_numpy(copy=False)  # may be pyarrow
            steps = (np.arange(periods) * typing.cast(int, freq)).astype(
                starts_np.dtype
            )
            values = np.repeat(starts_np, periods) + np.tile(steps, starts_np.size)
            out = pd.Series(values, dtype=starts.dtype)
        elif _is_dt_dtype(starts):
            if isinstance(freq, str):
                freq = pd.tseries.frequencies.to_offset(freq)
            # other freqs are left for offset_times to reject with the original dtype
            starts_dtype = starts.dtype
            arrow_dtype = (
                starts_dtype
                if isinstance(freq, BaseOffset)
                and isinstance(starts_dtype, pd.ArrowDtype)
                else None
            )
            dt_starts = starts if arrow_dtype is None else _arrow_times_to_numpy(starts)
            per_period = [
                typing.cast(pd.Index, offset_times(dt_starts, freq, i))
                for i in range(periods)
            ]
            if per_period:
                series_major = np.arange(starts.size * periods).reshape(periods, -1)
                times = per_period[0].append(per_period[1:])
                times = times.take(series_major.ravel(order="F"))
            else:
                times = dt_starts[:0]
            if arrow_dtype is not None:
                times = _numpy_times_to_arrow(times, arrow_dtype)
            out = pd.Series(times.rename(None))
        else:
            raise ValueError(
                f"`starts` must be integers or timestamps, got '{starts.dtype}'."
            )
    else:
        assert isinstance(starts, pl_Series)
        if starts.dtype.is_integer():
            step = typing.cast(int, freq)
            ends = starts + step * periods
            out = pl.int_ranges(starts, ends, step, eager=True).explode()
        else:
            interval = typing.cast(str, freq)
            ends = typing.cast(pl_Series, offset_times(starts, interval, periods - 1))
            if starts.dtype == pl.Date:
                ranges = pl.date_ranges(starts, ends, interval=interval, eager=True)
            else:
                ranges = pl.datetime_ranges(starts, ends, interval=interval, eager=True)
            out = _ensure_month_ends(ranges.explode(), starts, interval)
        out = out.alias(starts.name)
    return out


def repeat(
    s: Union[Series, pd.Index, np.ndarray], n: Union[int, np.ndarray, Series]
) -> Union[Series, pd.Index, np.ndarray]:
    out: Union[Series, pd.Index, np.ndarray]
    if isinstance(s, pl_Series):
        if isinstance(n, np.ndarray):
            n = pl_Series(n)
        out = (
            pl.DataFrame(s.alias("x"))
            .select(pl.col("x").repeat_by(n))["x"]  # type: ignore[arg-type]
            .explode()
            .alias(s.name)
        )
    else:
        out = np.repeat(s, n)
        if isinstance(out, pd.Series):
            out = out.reset_index(drop=True)
    return out


def cv_times(
    times: np.ndarray,
    uids: Union[Series, pd.Index],
    indptr: np.ndarray,
    h: int,
    test_size: int,
    step_size: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> DataFrame:
    if test_size < h:
        raise ValueError("`test_size` should be greater or equal to `h`.")
    n, resid = divmod(test_size - h, step_size)
    if resid != 0:
        raise ValueError("`test_size - h` should be a multiple `step_size`")
    n_windows = n + 1
    df_constructor: Union[Type[pd.DataFrame], Type[pl_DataFrame]]
    if isinstance(uids, pl_Series):
        df_constructor = pl_DataFrame
    else:
        df_constructor = pd.DataFrame
    sizes = np.diff(indptr)
    out_times = []
    out_cutoffs = []
    out_ids: List[Union[DataFrame, Series]] = []
    for i in range(n_windows):
        offset = test_size - i * step_size + 1
        use_series = sizes >= offset
        cutoff_idxs = indptr[1:][use_series] - offset
        valid_idxs = np.repeat(cutoff_idxs + 1, h) + np.tile(
            np.arange(h), cutoff_idxs.size
        )
        out_times.append(times[valid_idxs])
        out_cutoffs.append(np.repeat(times[cutoff_idxs], h))
        mask: Union[np.ndarray, pl_Series] = use_series
        if isinstance(uids, pl_Series):
            mask = pl_Series(use_series)
        window_uids = typing.cast(Series, filter_with_mask(uids, mask))
        out_ids.append(typing.cast(Series, repeat(window_uids, h)))
    return df_constructor(
        {
            id_col: vertical_concat(out_ids),
            time_col: np.hstack(out_times),
            "cutoff": np.hstack(out_cutoffs),
        }
    )


def group_by(df: Union[Series, DataFrame], by, maintain_order=False):
    if isinstance(df, (pd.Series, pd.DataFrame)):
        return df.groupby(by, observed=True, sort=not maintain_order)
    if isinstance(df, pl_Series):
        df = df.to_frame()
    return df.group_by(by, maintain_order=maintain_order)


def group_by_agg(df: DataFrame, by, aggs, maintain_order=False) -> DataFrame:
    if isinstance(df, pd.DataFrame):
        out = group_by(df, by, maintain_order).agg(aggs).reset_index()
    else:
        out = group_by(df, by, maintain_order).agg(
            *[getattr(pl.col(c), agg)() for c, agg in aggs.items()]
        )
    return out


def is_in(s: Series, collection) -> Series:
    out: Series
    if isinstance(s, pl_Series):
        out = s.is_in(collection)
    else:
        out = s.isin(collection)
    return out


def between(s: Series, lower: Series, upper: Series) -> Series:
    out: Series
    if isinstance(s, pd.Series):
        out = s.between(lower, upper)  # type: ignore[arg-type]
    else:
        out = s.is_between(lower, upper)  # type: ignore[arg-type]
    return out


def fill_null(df: DataFrame, mapping: Dict[str, Any]) -> DataFrame:
    out: DataFrame
    if isinstance(df, pd.DataFrame):
        out = df.fillna(mapping)
    else:
        out = df.with_columns(*[pl.col(col).fill_null(v) for col, v in mapping.items()])
    return out


def cast(s: Series, dtype: type) -> Series:
    if isinstance(s, pd.Series):
        s = s.astype(dtype)
    else:
        s = s.cast(dtype)
    return s


def value_cols_to_numpy(
    df: DataFrame, id_col: str, time_col: str, target_col: Optional[str]
) -> np.ndarray:
    exclude_cols = [id_col, time_col]
    if target_col is not None:
        exclude_cols.append(target_col)
    value_cols = [col for col in df.columns if col not in exclude_cols]
    if target_col is not None:
        value_cols = [target_col, *value_cols]
    data = to_numpy(df[value_cols])
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32)
    return data


def make_future_dataframe(
    uids: Series,
    last_times: Union[Series, pd.Index],
    freq: Union[int, str, BaseOffset],
    h: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> DataFrame:
    starts = offset_times(last_times, freq, 1)
    df_constructor: Union[Type[pd.DataFrame], Type[pl_DataFrame]]
    if isinstance(uids, pl_Series):
        df_constructor = pl_DataFrame
    else:
        df_constructor = pd.DataFrame
    return df_constructor(
        {
            id_col: repeat(uids, h),
            time_col: time_ranges(starts, freq=freq, periods=h),
        }
    )


def anti_join(df1: DataFrame, df2: DataFrame, on: Union[str, List[str]]) -> DataFrame:
    out: DataFrame
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        out = df1.merge(df2, on=on, how="left", indicator=True)
        out = out[out["_merge"] == "left_only"].drop(columns="_merge")
        out = out.reset_index(drop=True)
    elif isinstance(df1, pl_DataFrame) and isinstance(df2, pl_DataFrame):
        out = join(df1, df2, on=on, how="anti")
    else:
        raise ValueError(
            "df1 and df2 must be pandas or polars dataframes of the same type. "
            f"Got type(df1): '{type(df1)}', type(df2): '{type(df2)}'"
        )
    return out


def ensure_sorted(df: DataFrame, id_col: str, time_col: str) -> DataFrame:
    sort_idxs = maybe_compute_sort_indices(df=df, id_col=id_col, time_col=time_col)
    if sort_idxs is not None:
        df = take_rows(df=df, idxs=sort_idxs)
    return df


class ProcessedDF(NamedTuple):
    uids: Series
    last_times: np.ndarray
    data: np.ndarray
    indptr: np.ndarray
    sort_idxs: Optional[np.ndarray]


def process_df(
    df: DataFrame,
    id_col: str,
    time_col: str,
    target_col: Optional[str],
) -> ProcessedDF:
    """Extract components from dataframe

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times and target values.

    Returns:
        ProcessedDF: A named tuple containing:
            - ids (pandas or polars Series): serie with the sorted unique ids present in the data.
            - last_times (numpy array): array with the last time for each serie.
            - data (numpy ndarray): 2d array with target plus features values.
            - indptr (numpy ndarray): 1d array with indices to the start and end of each serie.
            - sort_idxs (numpy array or None): array with the indices that would sort the original data.
              If the data is already sorted this is `None`.
    """
    # validations
    validate_format(df, id_col, time_col, target_col)

    # ids
    id_counts = counts_by_id(df, id_col)
    uids = id_counts[id_col]

    # indices
    sizes = id_counts["counts"].to_numpy()
    indptr = np.append(0, sizes.cumsum()).astype(np.int32)
    last_idxs = indptr[1:] - 1

    # data
    data = value_cols_to_numpy(df, id_col, time_col, target_col)

    # check if we need to sort
    sort_idxs = maybe_compute_sort_indices(df, id_col, time_col)
    if sort_idxs is not None:
        data = data[sort_idxs]
        last_idxs = sort_idxs[last_idxs]
    times = df[time_col].to_numpy()[last_idxs]
    return ProcessedDF(uids, times, data, indptr, sort_idxs)


class DataFrameProcessor:
    def __init__(
        self,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def process(
        self, df: DataFrame
    ) -> Tuple[Series, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return process_df(df, self.id_col, self.time_col, self.target_col)


def _find_boundaries(
    times: np.ndarray,
    indptr: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    if indptr.size <= 1:
        return np.empty(0, dtype=np.int64)
    sizes = np.diff(indptr)
    counts = np.add.reduceat(times <= np.repeat(bounds, sizes), indptr[:-1])
    return indptr[:-1] + counts.astype(np.int64, copy=False)


def _ranges_to_indexer(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    lens = np.maximum(stops - starts, 0)
    total = int(lens.sum())
    offsets = np.zeros(lens.size + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return (
        np.repeat(starts, lens)
        + np.arange(total, dtype=np.int64)
        - np.repeat(offsets[:-1], lens)
    )


def _single_split_sorted(
    df: DataFrame,
    uids: Series,
    indptr: np.ndarray,
    times: np.ndarray,
    last_times: Series,
    i_window: int,
    n_windows: int,
    h: int,
    id_col: str,
    freq: Union[int, str, pd.offsets.BaseOffset],
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
    allow_partial_horizons: bool = False,
) -> Tuple[DataFrame, np.ndarray, np.ndarray]:
    if step_size is None:
        step_size = h
    if allow_partial_horizons:
        test_size = step_size * n_windows
    else:
        test_size = h + step_size * (n_windows - 1)
    offset = test_size - i_window * step_size
    train_ends = offset_times(last_times, freq, -offset)
    valid_ends = offset_times(train_ends, freq, h)
    train_stops = _find_boundaries(times, indptr, np.asarray(train_ends))
    valid_stops = _find_boundaries(times, indptr, np.asarray(valid_ends))
    if input_size is None:
        train_starts = indptr[:-1].astype(np.int64, copy=True)
    else:
        train_start_times = offset_times(train_ends, freq, -input_size)
        train_starts = _find_boundaries(times, indptr, np.asarray(train_start_times))
    train_sizes = train_stops - train_starts
    zeros_mask = train_sizes == 0
    if zeros_mask.all():
        raise ValueError(
            "All series are too short for the cross validation settings, "
            f"at least {offset + 1} samples are required.\n"
            "Please reduce `n_windows` or `h`."
        )
    if zeros_mask.any():
        ids = filter_with_mask(uids, zeros_mask)
        warnings.warn(
            "The following series are too short for the window "
            f"and will be dropped: {reprlib.repr(list(ids))}"
        )
        valid_stops = valid_stops.copy()
        valid_stops[zeros_mask] = train_stops[zeros_mask]
    cutoffs: DataFrame
    if isinstance(df, pd.DataFrame):
        cutoffs = pd.DataFrame(
            {
                id_col: typing.cast(pd.Series, uids).reset_index(drop=True),
                "cutoff": pd.Series(train_ends).reset_index(drop=True),
            }
        )
    else:
        cutoffs = pl_DataFrame({id_col: uids, "cutoff": train_ends})
    train_idxs = _ranges_to_indexer(train_starts, train_stops)
    valid_idxs = _ranges_to_indexer(train_stops, valid_stops)
    return cutoffs, train_idxs, valid_idxs


def _single_split(
    df: DataFrame,
    i_window: int,
    n_windows: int,
    h: int,
    id_col: str,
    time_col: str,
    freq: Union[int, str, pd.offsets.BaseOffset],
    max_dates: Series,
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
    allow_partial_horizons: bool = False,
) -> Tuple[DataFrame, Series, Series]:
    if step_size is None:
        step_size = h
    if allow_partial_horizons:
        test_size = step_size * n_windows
    else:
        test_size = h + step_size * (n_windows - 1)
    offset = test_size - i_window * step_size
    ids = df[id_col]
    times = df[time_col]
    train_ends = offset_times(max_dates, freq, -offset)
    valid_ends = offset_times(train_ends, freq, h)
    train_mask = times <= train_ends
    valid_mask = (times > train_ends) & (times <= valid_ends)
    if input_size is not None:
        train_starts = offset_times(train_ends, freq, -input_size)
        train_mask &= times > train_starts
    train_sizes: DataFrame
    if isinstance(train_mask, pd.Series):
        train_sizes = (
            train_mask.groupby(typing.cast(pd.Series, ids), observed=True, sort=False)
            .sum()
            .reset_index()
        )
    else:
        tmp_df = pl.DataFrame({id_col: ids, time_col: train_mask})
        train_sizes = group_by_agg(
            tmp_df, id_col, {time_col: "sum"}, maintain_order=True
        )
    zeros_mask = train_sizes[time_col].eq(0)
    if zeros_mask.all():
        raise ValueError(
            "All series are too short for the cross validation settings, "
            f"at least {offset + 1} samples are required.\n"
            "Please reduce `n_windows` or `h`."
        )
    elif zeros_mask.any():
        short_ids = filter_with_mask(train_sizes[id_col], zeros_mask)
        warnings.warn(
            "The following series are too short for the window "
            f"and will be dropped: {reprlib.repr(list(short_ids))}"
        )
        dropped_ids = is_in(ids, short_ids)
        valid_mask &= ~dropped_ids
    cutoffs: DataFrame
    if isinstance(train_ends, pd.Series):
        cutoffs = (
            train_ends.set_axis(typing.cast(pd.Series, ids))
            .groupby(id_col, observed=True)
            .head(1)
            .rename("cutoff")
            .reset_index()
        )
    else:
        cutoffs = typing.cast(pl_Series, train_ends).to_frame().with_columns(ids)
        cutoffs = (
            group_by(cutoffs, id_col)
            .agg(pl.col(time_col).head(1))
            .explode(pl.col(time_col))
            .rename({time_col: "cutoff"})
        )
    return cutoffs, train_mask, valid_mask


def backtest_splits(
    df: DataFrame,
    n_windows: int,
    h: int,
    id_col: str,
    time_col: str,
    freq: Union[int, str, pd.offsets.BaseOffset],
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
    allow_partial_horizons: bool = False,
) -> Generator[Tuple[DataFrame, DataFrame, DataFrame], None, None]:
    if _is_sorted(df=df, id_col=id_col, time_col=time_col):
        id_counts = counts_by_id(df, id_col)
        uids = id_counts[id_col]
        sizes = id_counts["counts"].to_numpy()
        indptr = np.append(0, sizes.cumsum()).astype(np.int64)
        times = df[time_col].to_numpy()
        last_times = typing.cast(Series, take_rows(df[time_col], indptr[1:] - 1))
        for i in range(n_windows):
            cutoffs, train_idxs, valid_idxs = _single_split_sorted(
                df=df,
                uids=uids,
                indptr=indptr,
                times=times,
                last_times=last_times,
                i_window=i,
                n_windows=n_windows,
                h=h,
                id_col=id_col,
                freq=freq,
                step_size=step_size,
                input_size=input_size,
                allow_partial_horizons=allow_partial_horizons,
            )
            train = take_rows(df, train_idxs)
            valid = take_rows(df, valid_idxs)
            yield cutoffs, train, valid
        return
    max_dates: Series
    if isinstance(df, pd.DataFrame):
        max_dates = df.groupby(id_col, observed=True)[time_col].transform("max")
    else:
        max_dates = df.select(pl.col(time_col).max().over(id_col))[time_col]
    for i in range(n_windows):
        cutoffs, train_mask, valid_mask = _single_split(
            df,
            i_window=i,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
            max_dates=max_dates,
            step_size=step_size,
            input_size=input_size,
            allow_partial_horizons=allow_partial_horizons,
        )
        train = filter_with_mask(df, train_mask)
        valid = filter_with_mask(df, valid_mask)
        yield cutoffs, train, valid


def add_insample_levels(
    df: DataFrame,
    models: List[str],
    level: List[Union[int, float]],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DataFrame:
    import operator

    from scipy.stats import norm

    df = copy_if_pandas(df, deep=False)
    cuts = norm.ppf(0.5 + np.asarray(level) / 200).reshape(1, -1)
    stds_df: DataFrame
    if isinstance(df, pd.DataFrame):
        errors = df[models].sub(df[target_col], axis=0)
        stds_df = errors.groupby(df[id_col], observed=True).transform("std")
    else:
        exprs = (pl.col(m).sub(pl.col(target_col)).std().over(id_col) for m in models)
        stds_df = df.select(exprs)
    stds = to_numpy(stds_df)
    preds = to_numpy(df[models])
    vals = np.empty_like(preds, shape=(preds.shape[0], len(models) * 2 * len(level)))
    cols = []
    k = 0
    for i, model in enumerate(models):
        widths = cuts * stds[:, [i]]
        for side, op in {"lo": operator.sub, "hi": operator.add}.items():
            for j, lvl in enumerate(level):
                cols.append(f"{model}-{side}-{lvl}")
                vals[:, k] = op(preds[:, i], widths[:, j])
                k += 1
    return assign_columns(df, cols, vals)
