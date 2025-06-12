import datetime
import warnings
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastcore.test import test_eq, test_fail
from polars import DataFrame as pl_DataFrame
from polars import Series as pl_Series

from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.data import generate_series
from utilsforecast.processing import (
    DataFrameProcessor,
    _multiply_pl_freq,
    add_insample_levels,
    anti_join,
    assign_columns,
    backtest_splits,
    between,
    cast,
    cv_times,
    fill_null,
    group_by,
    group_by_agg,
    horizontal_concat,
    is_in,
    is_nan,
    is_nan_or_none,
    is_none,
    make_future_dataframe,
    offset_times,
    process_df,
    repeat,
    sort,
    take_rows,
    time_ranges,
    to_numpy,
    vertical_concat,
)

if POLARS_INSTALLED:
    import polars as pl
    import polars.testing


@pytest.fixture(params=["pandas"] + (["polars"] if POLARS_INSTALLED else []))
def engine(request):
    return request.param


def test_assign_columns(engine):
    series = generate_series(2, engine=engine)
    x = np.random.rand(series.shape[0])
    series = assign_columns(series, "x", x)
    series = assign_columns(series, ["y", "z"], np.vstack([x, x]).T)
    series = assign_columns(series, "ones", 1)
    series = assign_columns(series, "zeros", np.zeros(series.shape[0]))
    series = assign_columns(series, "as", "a")
    series = assign_columns(series, "bs", series.shape[0] * ["b"])

    # Select and compare data
    if engine == "pandas":
        np.testing.assert_allclose(series[["x", "y", "z"]].values, np.vstack([x, x, x]).T)
        np.testing.assert_equal(series["ones"].values, np.ones(series.shape[0]))
        np.testing.assert_equal(series["as"].values, np.full(series.shape[0], "a"))
        np.testing.assert_equal(series["bs"].values, np.full(series.shape[0], "b"))
    else:  # polars
        np.testing.assert_allclose(series.select(["x", "y", "z"]).to_numpy(), np.vstack([x, x, x]).T)
        np.testing.assert_equal(series["ones"].to_numpy(), np.ones(series.shape[0]))
        np.testing.assert_equal(series["as"].to_numpy(), np.full(series.shape[0], "a"))
        np.testing.assert_equal(series["bs"].to_numpy(), np.full(series.shape[0], "b"))


def test_take_rows(engine):
    series = generate_series(2, engine=engine)
    subset = take_rows(series, np.array([0, 2]))
    assert subset.shape[0] == 2


def test_is_nan():
    np.testing.assert_equal(
        is_nan(pd.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, True]),
    )
    np.testing.assert_equal(
        is_nan(pl.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, None]),
    )


def test_is_none():
    np.testing.assert_equal(
        is_none(pd.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, True]),
    )
    if POLARS_INSTALLED:
        np.testing.assert_equal(
            is_none(pl.Series([np.nan, 1.0, None])).to_numpy(),
            np.array([False, False, True]),
        )


def test_is_nan_or_none():
    np.testing.assert_equal(
        is_nan_or_none(pd.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, True]),
    )
    if POLARS_INSTALLED:
        np.testing.assert_equal(
            is_nan_or_none(pl.Series([np.nan, 1.0, None])).to_numpy(),
            np.array([True, False, True]),
        )

def test_vertical_concat_pd():
    df1 = pd.DataFrame({"x": ["a", "b", "c"]}, dtype="category")
    df2 = pd.DataFrame({"x": ["f", "b", "a"]}, dtype="category")
    pd.testing.assert_series_equal(
        vertical_concat([df1, df2])["x"],
        pd.Series(
            ["a", "b", "c", "f", "b", "a"],
            name="x",
            dtype=pd.CategoricalDtype(categories=["a", "b", "c", "f"]),
        ),
    )


def test_vertical_concat_pl():
    df1 = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.Categorical})
    df2 = pl.DataFrame({"x": ["f", "b", "a"]}, schema={"x": pl.Categorical})
    out = vertical_concat([df1, df2])["x"]
    assert out.equals(pl.Series("x", ["a", "b", "c", "f", "b", "a"]))
    assert out.to_physical().equals(pl.Series("x", [0, 1, 2, 3, 1, 0]))
    assert out.cat.get_categories().equals(pl.Series("x", ["a", "b", "c", "f"]))

for engine in engines:
    series = generate_series(2, engine=engine)
    doubled = vertical_concat([series, series])
    assert doubled.shape[0] == 2 * series.shape[0]
for engine in engines:
    series = generate_series(2, engine=engine)
    renamer = {c: f"{c}_2" for c in series.columns}
    if engine == "pandas":
        series2 = series.rename(columns=renamer)
    else:
        series2 = series.rename(renamer)
    doubled = horizontal_concat([series, series2])
    assert doubled.shape[1] == 2 * series.shape[1]
pd.testing.assert_frame_equal(
    sort(pd.DataFrame({"x": [3, 1, 2]}), "x"), pd.DataFrame({"x": [1, 2, 3]})
)
pd.testing.assert_frame_equal(
    sort(pd.DataFrame({"x": [3, 1, 2]}), ["x"]), pd.DataFrame({"x": [1, 2, 3]})
)
pd.testing.assert_series_equal(sort(pd.Series([3, 1, 2])), pd.Series([1, 2, 3]))
pd.testing.assert_index_equal(sort(pd.Index([3, 1, 2])), pd.Index([1, 2, 3]))
pl.testing.assert_frame_equal(
    sort(pl.DataFrame({"x": [3, 1, 2]}), "x"),
    pl.DataFrame({"x": [1, 2, 3]}),
)
pl.testing.assert_frame_equal(
    sort(pl.DataFrame({"x": [3, 1, 2]}), ["x"]),
    pl.DataFrame({"x": [1, 2, 3]}),
)
pl.testing.assert_series_equal(
    sort(pl.Series("x", [3, 1, 2])), pl.Series("x", [1, 2, 3])
)
test_eq(_multiply_pl_freq("1d", 4), "4d")
test_eq(_multiply_pl_freq("2d", 4), "8d")
pl.testing.assert_series_equal(
    _multiply_pl_freq("1d", pl_Series([1, 2])),
    pl_Series(["1d", "2d"]),
)
pl.testing.assert_series_equal(
    _multiply_pl_freq("4m", pl_Series([2, 4])),
    pl_Series(["8m", "16m"]),
)
pd.testing.assert_index_equal(
    offset_times(
        pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
        pd.offsets.MonthEnd(),
        1,
    ),
    pd.Index(pd.to_datetime(["2020-02-29", "2020-03-31", "2020-04-30"])),
)
pd.testing.assert_index_equal(
    offset_times(
        pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        pd.offsets.MonthBegin(),
        1,
    ),
    pd.Index(pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"])),
)
pl.testing.assert_series_equal(
    offset_times(
        pl_Series([dt(2020, 1, 31), dt(2020, 2, 28), dt(2020, 3, 31)]), "1mo", 1
    ),
    pl_Series([dt(2020, 2, 29), dt(2020, 3, 28), dt(2020, 4, 30)]),
)
pl.testing.assert_series_equal(
    offset_times(
        pl_Series([dt(2020, 1, 31), dt(2020, 2, 29), dt(2020, 3, 31)]), "1mo", 1
    ),
    pl_Series([dt(2020, 2, 29), dt(2020, 3, 31), dt(2020, 4, 30)]),
)
# datetimes
dates = pd.to_datetime(["2000-01-01", "2010-10-10"])
pd.testing.assert_series_equal(
    time_ranges(dates, freq="D", periods=3),
    pd.Series(
        pd.to_datetime(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
                "2010-10-10",
                "2010-10-11",
                "2010-10-12",
            ]
        )
    ),
)
pd.testing.assert_series_equal(
    time_ranges(dates, freq="2D", periods=3),
    pd.Series(
        pd.to_datetime(
            [
                "2000-01-01",
                "2000-01-03",
                "2000-01-05",
                "2010-10-10",
                "2010-10-12",
                "2010-10-14",
            ]
        )
    ),
)
pd.testing.assert_series_equal(
    time_ranges(dates, freq="4D", periods=3),
    pd.Series(
        pd.to_datetime(
            [
                "2000-01-01",
                "2000-01-05",
                "2000-01-09",
                "2010-10-10",
                "2010-10-14",
                "2010-10-18",
            ]
        )
    ),
)
pd.testing.assert_series_equal(
    time_ranges(
        pd.to_datetime(["2000-01-01", "2010-10-01"]),
        freq=2 * pd.offsets.MonthBegin(),
        periods=2,
    ),
    pd.Series(pd.to_datetime(["2000-01-01", "2000-03-01", "2010-10-01", "2010-12-01"])),
)
pd.testing.assert_series_equal(
    time_ranges(
        pd.to_datetime(["2000-01-01", "2010-01-01"]).tz_localize("US/Eastern"),
        freq=2 * pd.offsets.YearBegin(),
        periods=2,
    ),
    pd.Series(
        pd.to_datetime(
            ["2000-01-01", "2002-01-01", "2010-01-01", "2012-01-01"]
        ).tz_localize("US/Eastern")
    ),
)
pd.testing.assert_series_equal(
    time_ranges(
        pd.to_datetime(["2000-12-31", "2010-12-31"]),
        freq=2 * pd.offsets.YearEnd(),
        periods=2,
    ),
    pd.Series(pd.to_datetime(["2000-12-31", "2002-12-31", "2010-12-31", "2012-12-31"])),
)
# ints
dates = pd.Series([1, 10])
pd.testing.assert_series_equal(
    time_ranges(dates, freq=1, periods=3), pd.Series([1, 2, 3, 10, 11, 12])
)
pd.testing.assert_series_equal(
    time_ranges(dates, freq=2, periods=3), pd.Series([1, 3, 5, 10, 12, 14])
)
pd.testing.assert_series_equal(
    time_ranges(dates, freq=4, periods=3), pd.Series([1, 5, 9, 10, 14, 18])
)
# datetimes
dates = pl.Series([dt(2000, 1, 1), dt(2010, 10, 10)])
pl.testing.assert_series_equal(
    time_ranges(dates, freq="1d", periods=3),
    pl.Series(
        [
            dt(2000, 1, 1),
            dt(2000, 1, 2),
            dt(2000, 1, 3),
            dt(2010, 10, 10),
            dt(2010, 10, 11),
            dt(2010, 10, 12),
        ]
    ),
)
pl.testing.assert_series_equal(
    time_ranges(dates, freq="2d", periods=3),
    pl.Series(
        [
            dt(2000, 1, 1),
            dt(2000, 1, 3),
            dt(2000, 1, 5),
            dt(2010, 10, 10),
            dt(2010, 10, 12),
            dt(2010, 10, 14),
        ]
    ),
)
pl.testing.assert_series_equal(
    time_ranges(dates, freq="4d", periods=3),
    pl.Series(
        [
            dt(2000, 1, 1),
            dt(2000, 1, 5),
            dt(2000, 1, 9),
            dt(2010, 10, 10),
            dt(2010, 10, 14),
            dt(2010, 10, 18),
        ]
    ),
)
pl.testing.assert_series_equal(
    time_ranges(pl.Series([dt(2010, 2, 28), dt(2000, 1, 31)]), "1mo", 3),
    pl.Series(
        [
            dt(2010, 2, 28),
            dt(2010, 3, 31),
            dt(2010, 4, 30),
            dt(2000, 1, 31),
            dt(2000, 2, 29),
            dt(2000, 3, 31),
        ]
    ),
)
# dates
dates = pl.Series([datetime.date(2000, 1, 1), datetime.date(2010, 10, 10)])
pl.testing.assert_series_equal(
    time_ranges(dates, freq="1d", periods=2),
    pl.Series(
        [
            datetime.date(2000, 1, 1),
            datetime.date(2000, 1, 2),
            datetime.date(2010, 10, 10),
            datetime.date(2010, 10, 11),
        ]
    ),
)
# ints
dates = pl.Series([1, 10])
pl.testing.assert_series_equal(
    time_ranges(dates, freq=1, periods=3),
    pl.Series([1, 2, 3, 10, 11, 12]),
)
pl.testing.assert_series_equal(
    time_ranges(dates, freq=2, periods=3),
    pl.Series([1, 3, 5, 10, 12, 14]),
)
pl.testing.assert_series_equal(
    time_ranges(dates, freq=4, periods=3),
    pl.Series([1, 5, 9, 10, 14, 18]),
)
pd.testing.assert_index_equal(
    repeat(pd.CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c"]), 2),
    pd.CategoricalIndex(["a", "a", "b", "b", "c", "c"], categories=["a", "b", "c"]),
)
pd.testing.assert_series_equal(repeat(pd.Series([1, 2]), 2), pd.Series([1, 1, 2, 2]))
pd.testing.assert_series_equal(
    repeat(pd.Series([1, 2]), pd.Series([2, 3])),
    pd.Series([1, 1, 2, 2, 2]),
)
np.testing.assert_array_equal(
    repeat(np.array([np.datetime64("2000-01-01"), np.datetime64("2010-10-10")]), 2),
    np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2000-01-01"),
            np.datetime64("2010-10-10"),
            np.datetime64("2010-10-10"),
        ]
    ),
)
np.testing.assert_array_equal(
    repeat(np.array([1, 2]), np.array([2, 3])),
    np.array([1, 1, 2, 2, 2]),
)
s = pl.Series(["a", "b", "c"], dtype=pl.Categorical)
pl.testing.assert_series_equal(repeat(s, 2), pl.concat([s, s]).sort())
pl.testing.assert_series_equal(repeat(pl.Series([2, 4]), 2), pl.Series([2, 2, 4, 4]))
pl.testing.assert_series_equal(
    repeat(pl.Series([1, 2]), np.array([2, 3])),
    pl.Series([1, 1, 2, 2, 2]),
)
times = np.arange(51, dtype=np.int64)
uids = pd.Series(["id_0"])
indptr = np.array([0, 51])
h = 3
test_size = 5
actual = cv_times(
    times=times,
    uids=uids,
    indptr=indptr,
    h=h,
    test_size=test_size,
    step_size=1,
)
expected = pd.DataFrame(
    {
        "unique_id": 9 * ["id_0"],
        "ds": np.hstack([[46, 47, 48], [47, 48, 49], [48, 49, 50]], dtype=np.int64),
        "cutoff": np.repeat(np.array([45, 46, 47], dtype=np.int64), h),
    }
)
pd.testing.assert_frame_equal(actual, expected)

# step_size=2
actual = cv_times(
    times=times,
    uids=uids,
    indptr=indptr,
    h=h,
    test_size=test_size,
    step_size=2,
)
expected = pd.DataFrame(
    {
        "unique_id": 6 * ["id_0"],
        "ds": np.hstack([[46, 47, 48], [48, 49, 50]], dtype=np.int64),
        "cutoff": np.repeat(np.array([45, 47], dtype=np.int64), h),
    }
)
pd.testing.assert_frame_equal(actual, expected)
pd.testing.assert_frame_equal(
    group_by_agg(pd.DataFrame({"x": [1, 1, 2], "y": [1, 1, 1]}), "x", {"y": "sum"}),
    pd.DataFrame({"x": [1, 2], "y": [2, 1]}),
)
pd.testing.assert_frame_equal(
    group_by_agg(
        pl.DataFrame({"x": [1, 1, 2], "y": [1, 1, 1]}),
        "x",
        {"y": "sum"},
        maintain_order=True,
    ).to_pandas(),
    pd.DataFrame({"x": [1, 2], "y": [2, 1]}),
)
np.testing.assert_equal(
    is_in(pd.Series([1, 2, 3]), [1]), np.array([True, False, False])
)
np.testing.assert_equal(
    is_in(pl.Series([1, 2, 3]), [1]), np.array([True, False, False])
)
np.testing.assert_equal(
    between(pd.Series([1, 2, 3]), pd.Series([0, 1, 4]), pd.Series([4, 1, 2])),
    np.array([True, False, False]),
)
np.testing.assert_equal(
    between(pl.Series([1, 2, 3]), pl.Series([0, 1, 4]), pl.Series([4, 1, 2])),
    np.array([True, False, False]),
)
pd.testing.assert_frame_equal(
    fill_null(pd.DataFrame({"x": [1, np.nan], "y": [np.nan, 2]}), {"x": 2, "y": 1}),
    pd.DataFrame({"x": [1, 2], "y": [1, 2]}, dtype="float64"),
)
pl.testing.assert_frame_equal(
    fill_null(pl.DataFrame({"x": [1, None], "y": [None, 2]}), {"x": 2, "y": 1}),
    pl.DataFrame({"x": [1, 2], "y": [1, 2]}),
)
pd.testing.assert_series_equal(
    cast(pd.Series([1, 2, 3]), "int16"), pd.Series([1, 2, 3], dtype="int16")
)
pd.testing.assert_series_equal(
    cast(pl.Series("x", [1, 2, 3]), pl.Int16).to_pandas(),
    pd.Series([1, 2, 3], name="x", dtype="int16"),
)
pd.testing.assert_frame_equal(
    make_future_dataframe(
        pd.Series([1, 2]), pd.to_datetime(["2000-01-01", "2010-10-10"]), freq="D", h=2
    ),
    pd.DataFrame(
        {
            "unique_id": [1, 1, 2, 2],
            "ds": pd.to_datetime(
                ["2000-01-02", "2000-01-03", "2010-10-11", "2010-10-12"]
            ),
        }
    ),
)
pl.testing.assert_frame_equal(
    make_future_dataframe(
        pl.Series([1, 2]),
        pl.Series([dt(2000, 1, 1), dt(2010, 10, 10)]),
        freq="1d",
        h=2,
        id_col="uid",
        time_col="dates",
    ),
    pl.DataFrame(
        {
            "uid": [1, 1, 2, 2],
            "dates": [
                dt(2000, 1, 2),
                dt(2000, 1, 3),
                dt(2010, 10, 11),
                dt(2010, 10, 12),
            ],
        }
    ),
)
pd.testing.assert_frame_equal(
    anti_join(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1]}), on="x"),
    pd.DataFrame({"x": [2]}),
)
test_eq(
    anti_join(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}), on="x").shape[0],
    0,
)
pl.testing.assert_frame_equal(
    anti_join(pl_DataFrame({"x": [1, 2]}), pl_DataFrame({"x": [1]}), on="x"),
    pl_DataFrame({"x": [2]}),
)
test_eq(
    anti_join(pl_DataFrame({"x": [1]}), pl_DataFrame({"x": [1]}), on="x").shape[0],
    0,
)
horizon = 3
test_size = 5
for equal_ends in [True, False]:
    n_series = 2
    series = generate_series(n_series, equal_ends=equal_ends)
    freq = pd.tseries.frequencies.to_offset("D")
    uids, last_times, data, indptr, _ = process_df(series, "unique_id", "ds", "y")
    times = series["ds"].to_numpy()
    df_dates = cv_times(
        times=times,
        uids=uids,
        indptr=indptr,
        h=horizon,
        test_size=test_size,
        step_size=1,
    )
    test_eq(len(df_dates), n_series * horizon * (test_size - horizon + 1))
static_features = ["static_0", "static_1"]
for n_static_features in [0, 2]:
    series_pd = generate_series(
        1_000, n_static_features=n_static_features, equal_ends=False, engine="pandas"
    )
    for i in range(n_static_features):
        series_pd[f"static_{i}"] = (
            series_pd[f"static_{i}"].map(lambda x: f"x_{x}").astype("category")
        )
    scrambled_series_pd = series_pd.sample(frac=1.0)
    dfp = DataFrameProcessor("unique_id", "ds", "y")
    uids, times, data, indptr, _ = dfp.process(scrambled_series_pd)
    test_eq(times, series_pd.groupby("unique_id", observed=True)["ds"].max().values)
    test_eq(uids, np.sort(series_pd["unique_id"].unique()))
    for i in range(n_static_features):
        series_pd[f"static_{i}"] = series_pd[f"static_{i}"].cat.codes
    test_eq(data, series_pd[["y"] + static_features[:n_static_features]].to_numpy())
    test_eq(
        np.diff(indptr), series_pd.groupby("unique_id", observed=True).size().values
    )
# test process_df with target_col=None
series_pd = generate_series(10, n_static_features=2, equal_ends=False, engine="pandas")
series_pd = series_pd.rename(columns={"y": "exog_0"})
_, _, data, indptr, _ = process_df(series_pd, "unique_id", "ds", None)
np.testing.assert_allclose(
    data,
    to_numpy(series_pd.drop(columns=["unique_id", "ds"])),
)
for n_static_features in [0, 2]:
    series_pl = generate_series(
        1_000, n_static_features=n_static_features, equal_ends=False, engine="polars"
    )
    scrambled_series_pl = series_pl.sample(fraction=1.0, shuffle=True)
    dfp = DataFrameProcessor("unique_id", "ds", "y")
    uids, times, data, indptr, _ = dfp.process(scrambled_series_pl)
    grouped = group_by(series_pl, "unique_id")
    test_eq(times, grouped.agg(pl.col("ds").max()).sort("unique_id")["ds"].to_numpy())
    test_eq(uids, series_pl["unique_id"].unique().sort())
    test_eq(
        data,
        series_pl.select(
            pl.col(c).map_batches(lambda s: s.to_physical())
            for c in ["y"] + static_features[:n_static_features]
        ).to_numpy(),
    )
    test_eq(np.diff(indptr), grouped.count().sort("unique_id")["count"].to_numpy())
short_series = generate_series(100, max_length=50)
backtest_results = list(
    backtest_splits(
        short_series,
        n_windows=1,
        h=49,
        id_col="unique_id",
        time_col="ds",
        freq=pd.offsets.Day(),
    )
)[0]
test_fail(
    lambda: list(
        backtest_splits(
            short_series,
            n_windows=1,
            h=50,
            id_col="unique_id",
            time_col="ds",
            freq=pd.offsets.Day(),
        )
    ),
    contains="at least 51 samples are required",
)
some_short_series = generate_series(100, min_length=20, max_length=100)
with warnings.catch_warnings(record=True) as issued_warnings:
    warnings.simplefilter("always", UserWarning)
    splits = list(
        backtest_splits(
            some_short_series,
            n_windows=1,
            h=50,
            id_col="unique_id",
            time_col="ds",
            freq=pd.offsets.Day(),
        )
    )
    assert any("will be dropped" in str(w.message) for w in issued_warnings)
short_series_int = short_series.copy()
short_series_int["ds"] = short_series.groupby("unique_id", observed=True).transform(
    "cumcount"
)
backtest_int_results = list(
    backtest_splits(
        short_series_int, n_windows=1, h=40, id_col="unique_id", time_col="ds", freq=1
    )
)[0]


def test_backtest_splits(df, n_windows, h, step_size, input_size):
    max_dates = df.groupby("unique_id", observed=True)["ds"].max()
    day_offset = pd.offsets.Day()
    common_kwargs = dict(
        n_windows=n_windows,
        h=h,
        id_col="unique_id",
        time_col="ds",
        freq=pd.offsets.Day(),
        step_size=step_size,
        input_size=input_size,
    )
    permuted_df = df.sample(frac=1.0)
    splits = backtest_splits(df, **common_kwargs)
    splits_on_permuted = list(backtest_splits(permuted_df, **common_kwargs))
    if step_size is None:
        step_size = h
    test_size = h + step_size * (n_windows - 1)
    for window, (cutoffs, train, valid) in enumerate(splits):
        offset = test_size - window * step_size
        expected_max_train_dates = max_dates - day_offset * offset
        max_train_dates = train.groupby("unique_id", observed=True)["ds"].max()
        pd.testing.assert_series_equal(max_train_dates, expected_max_train_dates)
        pd.testing.assert_frame_equal(
            cutoffs, max_train_dates.rename("cutoff").reset_index()
        )

        if input_size is not None:
            expected_min_train_dates = expected_max_train_dates - day_offset * (
                input_size - 1
            )
            min_train_dates = train.groupby("unique_id", observed=True)["ds"].min()
            pd.testing.assert_series_equal(min_train_dates, expected_min_train_dates)

        expected_min_valid_dates = expected_max_train_dates + day_offset
        min_valid_dates = valid.groupby("unique_id", observed=True)["ds"].min()
        pd.testing.assert_series_equal(min_valid_dates, expected_min_valid_dates)

        expected_max_valid_dates = expected_max_train_dates + day_offset * h
        max_valid_dates = valid.groupby("unique_id", observed=True)["ds"].max()
        pd.testing.assert_series_equal(max_valid_dates, expected_max_valid_dates)

        if window == n_windows - 1:
            pd.testing.assert_series_equal(max_valid_dates, max_dates)

        _, permuted_train, permuted_valid = splits_on_permuted[window]
        pd.testing.assert_frame_equal(
            train, permuted_train.sort_values(["unique_id", "ds"])
        )
    pd.testing.assert_frame_equal(
        valid, permuted_valid.sort_values(["unique_id", "ds"])
    )


n_series = 20
min_length = 100
max_length = 1000
series = generate_series(
    n_series, freq="D", min_length=min_length, max_length=max_length
)

for step_size in (None, 1, 2):
    for input_size in (None, 4):
        test_backtest_splits(
            series, n_windows=3, h=14, step_size=step_size, input_size=input_size
        )
h = 10
series_pl = generate_series(
    n_series, freq="D", min_length=min_length, max_length=max_length, engine="polars"
)
splits = backtest_splits(
    series_pl, n_windows=3, h=h, id_col="unique_id", time_col="ds", freq="1d"
)
for cutoffs, train, valid in splits:
    train_ends = train.group_by("unique_id", maintain_order=True).agg(
        pl.col("ds").max()
    )
    valid_starts = valid.group_by("unique_id", maintain_order=True).agg(
        pl.col("ds").min()
    )
    valid_ends = valid.group_by("unique_id", maintain_order=True).agg(
        pl.col("ds").max()
    )
    expected_valid_starts = offset_times(train_ends["ds"], "1d", 1)
    expected_valid_ends = offset_times(train_ends["ds"], "1d", h)
    pl.testing.assert_series_equal(valid_starts["ds"], expected_valid_starts)
    pl.testing.assert_series_equal(valid_ends["ds"], expected_valid_ends)
series = generate_series(100, n_models=2)
models = ["model0", "model1"]
levels = [80, 95]
with_levels = add_insample_levels(series, models, levels)
for model in models:
    for lvl in levels:
        assert (
            with_levels[f"{model}-lo-{lvl}"].lt(with_levels[f"{model}-hi-{lvl}"]).all()
        )
series_pl = generate_series(100, n_models=2, engine="polars")
with_levels_pl = add_insample_levels(series_pl, ["model0", "model1"], [80, 95])
pd.testing.assert_frame_equal(
    with_levels.drop(columns="unique_id"),
    with_levels_pl.to_pandas().drop(columns="unique_id"),
)
