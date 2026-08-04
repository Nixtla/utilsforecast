import warnings
from datetime import date, datetime
from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utilsforecast.preprocessing import (
    _vectorized_serie_ranges,
    fill_gaps,
    id_time_grid,
)


@pytest.fixture
def pandas_datetime_df():
    """DataFrame with datetime ds column for pandas."""
    return pd.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": pd.to_datetime(["2020", "2021", "2023", "2021", "2022"]),
            "y": np.arange(5),
        }
    )


@pytest.fixture
def pandas_int_df():
    """DataFrame with integer ds column for pandas."""
    return pd.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": [2020, 2021, 2023, 2021, 2022],
            "y": np.arange(5),
        }
    )


@pytest.fixture
def polars_datetime_df():
    """DataFrame with datetime ds column for polars."""
    return pl.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": [
                datetime(2020, 1, 1),
                datetime(2022, 1, 1),
                datetime(2023, 1, 1),
                datetime(2021, 1, 1),
                datetime(2022, 1, 1),
            ],
            "y": np.arange(5),
        }
    )


@pytest.fixture
def polars_date_df():
    """DataFrame with date ds column for polars."""
    return pl.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": [
                date(2020, 1, 1),
                date(2022, 1, 1),
                date(2023, 1, 1),
                date(2021, 1, 1),
                date(2022, 1, 1),
            ],
            "y": np.arange(5),
        }
    )


@pytest.fixture
def polars_int_df():
    """DataFrame with integer ds column for polars."""
    return pl.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": [2020, 2021, 2023, 2021, 2022],
            "y": np.arange(5),
        }
    )


@pytest.fixture
def polars_ms_df(polars_datetime_df):
    """DataFrame with millisecond precision datetime for polars."""
    return polars_datetime_df.with_columns(pl.col("ds").cast(pl.Datetime(time_unit="ms")))


@pytest.fixture
def warning_df():
    """DataFrame where last value doesn't meet frequency (year start)."""
    return pd.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": pd.to_datetime(["2020-01", "2021-01", "2023-01", "2021-01", "2022-02"]),
            "y": np.arange(5),
        }
    )


# --- Helper functions ---


def get_pandas_freqs():
    """Get list of pandas frequency aliases with version compatibility."""
    freqs = [
        "YE",
        "YS",
        "ME",
        "MS",
        "W",
        "W-TUE",
        "D",
        "s",
        "ms",
        1,
        2,
        "20D",
        "30s",
        "2YE",
        "3YS",
        "30min",
        "B",
        "1h",
        "QS-NOV",
        "QE",
    ]
    try:
        pd.tseries.frequencies.to_offset("YE")
    except ValueError:
        # Older pandas version - use old aliases
        freqs = [
            f.replace("YE", "Y").replace("ME", "M").replace("h", "H").replace("QE", "Q")
            for f in freqs
            if isinstance(f, str)
        ]
    return freqs


def get_polars_freqs():
    """Get list of polars frequency aliases."""
    return ["1d", "1w", "1mo", "1q", "1y"]


def get_integer_freqs():
    """Get list of integer frequencies."""
    return list(range(1, 11))


# --- Basic fill_gaps tests ---


class TestFillGapsBasic:
    """Basic tests for fill_gaps function."""

    def test_fill_gaps_pandas_datetime_per_serie(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and per_serie end."""
        result = fill_gaps(pandas_datetime_df, freq="YS", end="per_serie")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > len(pandas_datetime_df)

    def test_fill_gaps_pandas_datetime_specific_end(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and specific end date."""
        result = fill_gaps(pandas_datetime_df, freq="YS", end="2024")
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_pandas_datetime_global_start(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and global start."""
        result = fill_gaps(pandas_datetime_df, freq="YS", start="global")
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_pandas_datetime_specific_start(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and specific start date."""
        result = fill_gaps(pandas_datetime_df, freq="YS", start="2019")
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_pandas_int(self, pandas_int_df):
        """Test fill_gaps with pandas integer ds column."""
        result = fill_gaps(pandas_int_df, freq=1, start=2019, end=2024)
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_polars_datetime(self, polars_datetime_df):
        """Test fill_gaps with polars datetime."""
        result = fill_gaps(
            polars_datetime_df,
            freq="1y",
            start=datetime(2019, 1, 1),
            end=datetime(2024, 1, 1),
        )
        assert isinstance(result, pl.DataFrame)

    def test_fill_gaps_polars_datetime_ms(self, polars_ms_df):
        """Test fill_gaps preserves millisecond precision for polars."""
        result = fill_gaps(
            polars_ms_df,
            freq="1y",
            start=datetime(2019, 1, 1),
            end=datetime(2024, 1, 1),
        )
        assert result.schema["ds"].time_unit == "ms"

    def test_fill_gaps_polars_date(self, polars_date_df):
        """Test fill_gaps with polars date type."""
        result = fill_gaps(
            polars_date_df,
            freq="1y",
            start=date(2020, 1, 1),
            end=date(2024, 1, 1),
        )
        assert isinstance(result, pl.DataFrame)

    def test_fill_gaps_polars_int(self, polars_int_df):
        """Test fill_gaps with polars integer ds column."""
        result = fill_gaps(polars_int_df, freq=1, start=2019, end=2024)
        assert isinstance(result, pl.DataFrame)


class TestFillGapsWarning:
    """Test fill_gaps warning behavior."""

    def test_fill_gaps_warns_on_lost_values(self, warning_df):
        """Test that fill_gaps warns when values are lost due to frequency mismatch."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fill_gaps(warning_df, "YS")
            assert len(w) > 0
            assert "values were lost" in str(w[0].message)


# --- Comprehensive fill_gaps tests ---


N_PERIODS = 100


def generate_test_dates(freq, n_periods):
    """Generate test dates for a given frequency."""
    if isinstance(freq, (pd.offsets.BaseOffset, str)):
        if isinstance(freq, str):
            try:
                delta = pd.Timedelta(freq)
                if delta.days > 0:
                    tz = None
                else:
                    tz = "Europe/Berlin"
            except ValueError:
                tz = None
        else:
            tz = None
        dates = pd.date_range("1950-01-01", periods=n_periods, freq=freq, tz=tz)
    else:
        dates = np.arange(0, freq * n_periods, freq, dtype=np.int64)
    return dates


def create_test_data(dates, n_periods, include_start, include_end):
    """Create test DataFrame with random date selections."""
    base_idxs = []
    if include_start:
        base_idxs.append(0)
    if include_end:
        base_idxs.append(dates.size - 1)
    base_idxs = np.array(base_idxs, dtype=np.int64)
    date_idxs = np.hstack(
        [
            np.append(
                base_idxs,
                np.random.choice(
                    np.arange(1, dates.size - 1),
                    size=n_periods // 2 - len(base_idxs),
                    replace=False,
                ),
            )
            for _ in range(2)
        ],
    )
    return pd.DataFrame(
        {
            "unique_id": np.repeat([1, 2], n_periods // 2),
            "ds": dates[date_idxs],
            "y": np.arange(n_periods, dtype=np.float64),
        }
    )


def verify_fill_results(data, filled, freq, start, end):
    """Verify fill_gaps results are correct."""
    data_starts_ends = data.groupby("unique_id", observed=True)["ds"].agg(["min", "max"])
    global_start = data_starts_ends["min"].min()
    global_end = data_starts_ends["max"].max()
    filled_starts_ends = filled.groupby("unique_id", observed=True)["ds"].agg(["min", "max"])

    # Inferred frequency is the expected
    first_serie = filled[filled["unique_id"] == 1]
    if isinstance(freq, str):
        if first_serie["ds"].dt.tz is not None:
            first_serie = first_serie.copy()
            first_serie["ds"] = first_serie["ds"].dt.tz_convert("UTC")
        inferred_freq = pd.infer_freq(first_serie["ds"].dt.tz_localize(None))
        assert inferred_freq == pd.tseries.frequencies.to_offset(freq)
    else:
        assert all(first_serie["ds"].diff().value_counts().index == [freq])

    # Fill keeps original data
    assert filled["y"].count() == N_PERIODS

    # Check starts
    if start == "per_serie":
        pd.testing.assert_series_equal(
            data_starts_ends["min"],
            filled_starts_ends["min"],
            check_dtype=False,
        )
    else:  # global or specific
        min_dates = filled_starts_ends["min"].unique()
        assert min_dates.size == 1
        expected_start = global_start if start == "global" else start
        assert min_dates[0] == expected_start

    # Check ends
    if end == "per_serie":
        pd.testing.assert_series_equal(
            data_starts_ends["max"],
            filled_starts_ends["max"],
            check_dtype=False,
        )
    else:  # global or specific
        max_dates = filled_starts_ends["max"].unique()
        assert max_dates.size == 1
        expected_end = global_end if end == "global" else end
        assert max_dates[0] == expected_end


class TestFillGapsComprehensive:
    """Comprehensive tests for fill_gaps with various frequencies and options."""

    @pytest.mark.parametrize("freq", get_pandas_freqs())
    @pytest.mark.parametrize("start_type", ["global", "per_serie", "specific"])
    @pytest.mark.parametrize("end_type", ["global", "per_serie", "specific"])
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_fill_gaps_comprehensive(self, freq, start_type, end_type, include_start, include_end):
        """Test fill_gaps with various frequency, start, and end combinations."""
        dates = generate_test_dates(freq, N_PERIODS)

        if isinstance(freq, (pd.offsets.BaseOffset, str)):
            offset = pd.tseries.frequencies.to_offset(freq)
        else:
            offset = freq

        global_start = dates[0]
        global_end = dates[-1]

        # Map start_type to actual start value
        if start_type == "global":
            start = "global"
        elif start_type == "per_serie":
            start = "per_serie"
        else:
            start = global_start - offset

        # Map end_type to actual end value
        if end_type == "global":
            end = "global"
        elif end_type == "per_serie":
            end = "per_serie"
        else:
            end = global_end + offset

        data = create_test_data(dates, N_PERIODS, include_start, include_end)
        filled = fill_gaps(data, freq, start=start, end=end)
        verify_fill_results(data, filled, freq, start, end)


# --- Error tests for incompatible frequency and time column ---


def create_error_test_data(dates, n_periods, include_start, include_end, lib):
    """Create test data for error tests."""
    base_idxs = []
    if include_start:
        base_idxs.append(0)
    if include_end:
        base_idxs.append(np.size(dates) - 1)
    base_idxs = np.array(base_idxs, dtype=np.int64)
    date_idxs = np.hstack(
        [
            np.append(
                base_idxs,
                np.random.choice(
                    np.arange(1, np.size(dates) - 1),
                    size=n_periods // 2 - len(base_idxs),
                    replace=False,
                ),
            )
            for _ in range(2)
        ],
    )
    if lib == "pandas":
        return pd.DataFrame(
            {
                "unique_id": np.repeat([1, 2], n_periods // 2),
                "ds": dates[date_idxs],
                "y": np.arange(n_periods, dtype=np.float64),
            }
        )
    else:  # polars
        return pl.DataFrame(
            {
                "unique_id": np.repeat([1, 2], n_periods // 2),
                "ds": dates[date_idxs],
                "y": np.arange(n_periods, dtype=np.float64),
            }
        )


class TestFillGapsIncompatibleFrequency:
    """Test fill_gaps raises errors for incompatible frequency and time column."""

    @pytest.fixture
    def pandas_freq_data(self):
        """Generate date data for pandas frequency tests."""
        freqs_pd = [
            "YE", "YS", "ME", "MS", "W", "W-TUE", "D", "s", "ms",
            "20D", "30s", "2YE", "3YS", "30min", "B", "1h", "QS-NOV", "QE",
        ]
        # Handle version compatibility
        try:
            pd.tseries.frequencies.to_offset("YE")
        except ValueError:
            freqs_pd = [
                f.replace("YE", "Y").replace("ME", "M").replace("h", "H").replace("QE", "Q")
                for f in freqs_pd
            ]
        return freqs_pd

    @pytest.fixture
    def polars_freq_data(self):
        """Generate date data for polars frequency tests."""
        return ["1d", "1w", "1mo", "1q", "1y"]

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_pandas_int_freq_with_datetime_data(self, pandas_freq_data, freq_int, include_start, include_end):
        """Test that integer freq with datetime data raises ValueError for pandas."""
        for freq_str in pandas_freq_data:
            dates_str = pd.date_range("1950-01-01", periods=N_PERIODS, freq=freq_str)
            offset = pd.tseries.frequencies.to_offset(freq_str)
            first_date = dates_str[0] - offset
            last_date = dates_str[-1] + offset

            starts = ["global", "per_serie", first_date]
            ends = ["global", "per_serie", last_date]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_str, N_PERIODS, include_start, include_end, "pandas")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_int, start=start, end=end)

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_pandas_str_freq_with_int_data(self, pandas_freq_data, freq_int, include_start, include_end):
        """Test that string freq with int data raises ValueError for pandas."""
        dates_int = np.arange(1, (N_PERIODS * freq_int) + 1, freq_int)

        for freq_str in pandas_freq_data:
            # Use integer boundaries for int data
            first_int = dates_int[0] - freq_int
            last_int = dates_int[-1] + freq_int

            starts = ["global", "per_serie", first_int]
            ends = ["global", "per_serie", last_int]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_int, N_PERIODS, include_start, include_end, "pandas")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_str, start=start, end=end)

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_polars_int_freq_with_datetime_data(self, polars_freq_data, freq_int, include_start, include_end):
        """Test that integer freq with datetime data raises ValueError for polars."""
        for freq_str in polars_freq_data:
            pl_dt = pl.date(1950, 1, 1)
            dates_str = pl.date_range(
                pl_dt,
                pl_dt.dt.offset_by(f"{N_PERIODS}{freq_str[1:]}"),
                interval=freq_str,
                eager=True,
            )
            first_date = dates_str.dt.offset_by(f"-{freq_str}")[0]
            last_date = dates_str.dt.offset_by(freq_str)[-1]

            starts = ["global", "per_serie", first_date]
            ends = ["global", "per_serie", last_date]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_str, N_PERIODS, include_start, include_end, "polars")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_int, start=start, end=end)

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_polars_str_freq_with_int_data(self, polars_freq_data, freq_int, include_start, include_end):
        """Test that string freq with int data raises ValueError for polars."""
        dates_int = np.arange(1, (N_PERIODS * freq_int) + 1, freq_int)

        for freq_str in polars_freq_data:
            # Use integer boundaries for int data
            first_int = dates_int[0] - freq_int
            last_int = dates_int[-1] + freq_int

            starts = ["global", "per_serie", first_int]
            ends = ["global", "per_serie", last_int]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_int, N_PERIODS, include_start, include_end, "polars")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_str, start=start, end=end)


# ========================================
# id_time_grid: vectorized per-serie range construction (perf)
# ========================================
#
# `id_time_grid`'s pandas path used to build each serie's timestamps with a
# python-level loop (`np.hstack([np.arange(s, e, delta) for s, e in
# zip(starts, ends)])`). It's now a single vectorized computation, extracted
# into `_vectorized_serie_ranges` so it's a pure, directly-testable unit
# (also called by `id_time_grid` itself -- these tests exercise the actual
# production code, not a copy of the algorithm). These tests pin it against
# that literal inline loop on adversarial series (varying lengths,
# single-point series, multi-step deltas, datetime and integer frequencies).


def _reference_arange_hstack(starts, ends, delta):
    """The original per-serie python loop `id_time_grid` used to build."""
    return np.hstack([np.arange(s, e, delta) for s, e in zip(starts, ends)])


class TestIdTimeGridVectorizedRanges:
    def test_matches_reference_loop_int_freq(self):
        rng = np.random.default_rng(123)
        for _ in range(25):
            n = rng.integers(1, 40)
            delta = int(rng.integers(1, 5))
            # starts/ends land on multiples of delta, like real timestamps do
            starts = (rng.integers(0, 100, size=n) * delta).astype(np.int64)
            n_steps = rng.integers(1, 25, size=n)
            ends = starts + n_steps * delta
            expected = _reference_arange_hstack(starts, ends, delta)
            actual, _sizes = _vectorized_serie_ranges(starts, ends, delta)
            np.testing.assert_array_equal(actual, expected)

    def test_matches_reference_loop_datetime_freq(self):
        rng = np.random.default_rng(321)
        for _ in range(25):
            n = rng.integers(1, 40)
            base = np.datetime64("2020-01-01")
            delta = np.timedelta64(int(rng.integers(1, 4)), "D")
            starts = base + (rng.integers(0, 500, size=n).astype(np.int64) * delta)
            n_steps = rng.integers(1, 25, size=n)
            ends = starts + n_steps.astype(np.int64) * delta
            expected = _reference_arange_hstack(starts, ends, delta)
            actual, _sizes = _vectorized_serie_ranges(starts, ends, delta)
            np.testing.assert_array_equal(actual, expected)

    def test_id_time_grid_int_freq_end_to_end(self):
        df = pd.DataFrame(
            {
                "unique_id": np.repeat(["a", "b", "c"], [5, 1, 10]),
                "ds": np.concatenate(
                    [np.arange(5), np.array([7]), np.arange(10)]
                ),
            }
        )
        # explicit per_serie/per_serie so each id spans only its own
        # [min, max] inclusive at step 1 (the default `end` is "global")
        grid = id_time_grid(df, freq=1, start="per_serie", end="per_serie")
        for uid, sub in df.groupby("unique_id"):
            g = grid[grid["unique_id"] == uid]["ds"].to_numpy()
            np.testing.assert_array_equal(
                g, np.arange(sub["ds"].min(), sub["ds"].max() + 1)
            )

    def test_id_time_grid_business_day_freq(self):
        df = pd.DataFrame(
            {
                "unique_id": np.repeat(["a", "b"], [10, 8]),
                "ds": np.concatenate(
                    [
                        pd.bdate_range("2020-01-01", periods=10),
                        pd.bdate_range("2020-01-06", periods=8),
                    ]
                ),
            }
        )
        grid = id_time_grid(df, freq="B")
        assert (pd.DatetimeIndex(grid["ds"]).dayofweek < 5).all()

    def test_id_time_grid_single_point_series(self):
        # start == end for every serie: each should contribute exactly its
        # own single timestamp when start/end are 'per_serie'.
        df = pd.DataFrame(
            {
                "unique_id": ["a", "b", "c"],
                "ds": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-10"]),
            }
        )
        grid = id_time_grid(df, freq="D", start="per_serie", end="per_serie")
        # id_time_grid always returns datetime64[ns]; compare values only
        # (dtype -- ns vs pandas' default us -- is intentional and unrelated
        # to the ranges themselves)
        pd.testing.assert_series_equal(
            grid.sort_values("unique_id")["ds"].reset_index(drop=True),
            df.sort_values("unique_id")["ds"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_id_time_grid_monthly_freq(self):
        df = pd.DataFrame(
            {
                "unique_id": np.repeat(["a", "b"], [6, 4]),
                "ds": np.concatenate(
                    [
                        pd.date_range("2020-01-01", periods=6, freq="MS"),
                        pd.date_range("2020-03-01", periods=4, freq="MS"),
                    ]
                ),
            }
        )
        grid = id_time_grid(df, freq="MS")
        for uid, sub in df.groupby("unique_id"):
            g = pd.DatetimeIndex(grid[grid["unique_id"] == uid]["ds"])
            expected = pd.date_range(sub["ds"].min(), sub["ds"].max(), freq="MS")
            # values only, see the dtype note in test_id_time_grid_single_point_series
            np.testing.assert_array_equal(g.to_numpy(), expected.to_numpy())


# ========================================
# id_time_grid: negative-range validation
# ========================================
#
# A negative time range (computed `end` before `start` for a particular
# serie) happens when a fixed `start`/`end` bound doesn't align with that
# serie's own timestamps -- e.g. a fixed `start` later than a serie's own
# `end` (`end="per_serie"`). On `main`, this crashed with an opaque,
# accidental `ValueError: repeats may not contain negative values` from
# `np.repeat(times_by_id.index, sizes)`, only because `sizes` happened to be
# negative in that case, not because it validated anything -- the crash
# depended on the vagaries of `np.repeat`'s error handling and would have
# behaved differently for a near-zero negative gap (see the exactly-zero
# test below). `id_time_grid` now validates this explicitly upfront and
# raises a clear `ValueError` naming the offending series.


class TestIdTimeGridNegativeRangeValidation:
    def test_negative_range_datetime_raises_clear_error(self):
        # serie "a" ends 2020-01-05; a fixed start of 2020-01-10 is after it
        df = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": pd.to_datetime(
                    ["2020-01-01", "2020-01-05", "2020-06-01", "2020-06-10"]
                ),
            }
        )
        with pytest.raises(ValueError, match="negative time range") as exc_info:
            id_time_grid(df, freq="D", start="2020-01-10", end="per_serie")
        message = str(exc_info.value)
        assert "'a'" in message
        assert "1 serie" in message

    def test_negative_range_int_raises_clear_error(self):
        # serie "a" ends at 5; a fixed start of 50 is after it
        df = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": [0, 5, 100, 110],
            }
        )
        with pytest.raises(ValueError, match="negative time range") as exc_info:
            id_time_grid(df, freq=1, start=50, end="per_serie")
        message = str(exc_info.value)
        assert "'a'" in message
        assert "1 serie" in message

    def test_negative_range_names_every_offending_serie(self):
        # both series have a fixed start after their own end
        df = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": pd.to_datetime(
                    ["2020-01-01", "2020-01-05", "2020-01-02", "2020-01-06"]
                ),
            }
        )
        with pytest.raises(ValueError, match="negative time range") as exc_info:
            id_time_grid(df, freq="D", start="2020-01-10", end="per_serie")
        message = str(exc_info.value)
        assert "2 serie" in message

    def test_zero_length_range_is_legitimate_not_an_error(self):
        # start == end for every serie (single-point series): a legitimate
        # empty/single-point range, not a negative one -- must not raise.
        df = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "ds": pd.to_datetime(["2020-01-01", "2020-01-05"]),
            }
        )
        grid = id_time_grid(df, freq="D", start="per_serie", end="per_serie")
        assert len(grid) == 2

    def test_normal_ranges_unaffected_by_validation(self):
        # sanity: typical per_serie/global usage (positive ranges for every
        # serie) still produces the expected grid and doesn't raise.
        df = pd.DataFrame(
            {
                "unique_id": np.repeat(["a", "b"], [3, 3]),
                "ds": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-05",
                        "2020-01-03",
                        "2020-01-04",
                        "2020-01-10",
                    ]
                ),
            }
        )
        grid = id_time_grid(df, freq="D")
        for uid, sub in df.groupby("unique_id"):
            g = pd.DatetimeIndex(grid[grid["unique_id"] == uid]["ds"])
            expected = pd.date_range(sub["ds"].min(), df["ds"].max(), freq="D")
            np.testing.assert_array_equal(g.to_numpy(), expected.to_numpy())
