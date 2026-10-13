import warnings

import numpy as np
import pandas as pd
import pytest

from utilsforecast.processing import offset_times

_ARRAY_N_TIMES = pd.to_datetime(
    ["2020-01-01", "2020-01-31", "2020-02-29", "2021-03-15", "2021-12-31"]
)


def _offset_times_elementwise(times, freq, n):
    return [t + int(k) * freq for t, k in zip(times, n)]


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
@pytest.mark.parametrize("as_index", [False, True])
@pytest.mark.parametrize("tz", [None, "America/New_York"])
@pytest.mark.parametrize(
    "freq",
    [pd.DateOffset(milliseconds=500), pd.DateOffset(microseconds=500)],
)
def test_offset_times_with_array_n_preserves_subsecond_offsets(
    unit, as_index, tz, freq
):
    times = pd.DatetimeIndex(_ARRAY_N_TIMES, name="ds").as_unit(unit)
    if tz is not None:
        times = times.tz_localize(tz)
    if not as_index:
        times = pd.Series(times, index=[5, 6, 7, 8, 9], name="ds")
    n = np.array([2, -1, 0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = offset_times(times, freq, n)
    assert list(result) == _offset_times_elementwise(times, freq, n)
    assert result.name == times.name
    if not as_index:
        pd.testing.assert_index_equal(result.index, times.index)


@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_offset_times_with_array_n_promotes_pyarrow_resolution(tz):
    dtype = "timestamp[s][pyarrow]" if tz is None else f"timestamp[s, tz={tz}][pyarrow]"
    times = pd.Series(_ARRAY_N_TIMES, name="ds").astype(dtype)
    freq = pd.DateOffset(milliseconds=500)
    n = np.array([2, -1, 0, 1, 2])
    result = offset_times(times, freq, n)
    assert list(result) == _offset_times_elementwise(times, freq, n)
    assert isinstance(result.dtype, pd.ArrowDtype)
    assert result.dtype.pyarrow_dtype.unit in ("ms", "us", "ns")
    assert result.dtype.pyarrow_dtype.tz == tz


@pytest.mark.parametrize("as_index", [False, True])
def test_offset_times_with_empty_array_n(as_index):
    times = pd.DatetimeIndex([], dtype="datetime64[s]", name="ds")
    if not as_index:
        times = pd.Series(times, name="ds")
    result = offset_times(
        times, pd.DateOffset(milliseconds=500), np.array([], dtype=np.int64)
    )
    assert result.empty
    assert result.dtype == times.dtype
    assert result.name == times.name
