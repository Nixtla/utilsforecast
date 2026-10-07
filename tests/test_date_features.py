import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import utilsforecast.date_features as dtf


def _pandas_reference(dates: pd.Series, name: str) -> np.ndarray:
    if name == "week_of_year":
        return dates.dt.isocalendar().week.to_numpy()
    attr = {"day_of_week": "dayofweek", "day_of_year": "dayofyear"}.get(name, name)
    return getattr(dates.dt, attr).to_numpy()


def test_available_lists_every_exported_feature():
    features = dtf.available()
    assert [f.name for f in features] == [
        name for name in dtf.__all__ if name != "available"
    ]
    for feature in features:
        assert getattr(dtf, feature.name) is feature


def test_docs_table_matches_available():
    doc = Path(__file__).parents[1] / "docs" / "date_features.html.md"
    rows = [
        [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        for line in doc.read_text().splitlines()
        if line.startswith("| `")
    ]
    assert rows == [
        [f.name, f.description, np.dtype(f.dtype).name] for f in dtf.available()
    ]


@pytest.mark.parametrize(
    "dates",
    [
        # spans 1900 and 2100 (not leap years), 2000 (leap) and 53-week ISO years
        pd.Series(pd.date_range("1899-12-01", "2101-02-01", freq="D")),
        pd.Series(
            pd.date_range("1999-12-01", "2031-01-01", freq="7h", tz="America/New_York")
        ),
        pd.Series(
            pd.date_range("1999-12-01", "2031-01-01", freq="7h", tz="Asia/Tokyo")
        ),
    ],
    ids=["naive", "tz-behind-utc", "tz-ahead-of-utc"],
)
@pytest.mark.parametrize(
    "container", ["pandas_series", "pandas_index", "polars", "polars_date"]
)
def test_compute_matches_pandas_attributes(dates, container):
    if container == "pandas_index":
        inp = pd.DatetimeIndex(dates)
    elif container == "polars":
        inp = pl.from_pandas(dates)
    elif container == "polars_date":
        if dates.dt.tz is not None or (dates.dt.hour != 0).any():
            pytest.skip("polars Date has no time or time zone")
        inp = pl.from_pandas(dates).cast(pl.Date)
    else:
        inp = dates
    for feature in dtf.available():
        if container == "polars_date" and feature in (dtf.hour, dtf.minute, dtf.second):
            continue
        vals = feature.compute(inp)
        assert vals.dtype == feature.dtype
        np.testing.assert_array_equal(
            vals, _pandas_reference(dates, feature.name), err_msg=feature.name
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_null_dates_raise(engine):
    dates = pd.Series(pd.to_datetime(["2020-01-01", None, "2020-03-01"]))
    if engine == "polars":
        dates = pl.from_pandas(dates)
    with pytest.raises(ValueError, match="'month', found 1 null dates"):
        dtf.month.compute(dates)


@pytest.mark.parametrize(
    "dates",
    [
        pd.Series([1, 2, 3]),
        pl.Series([1, 2, 3]),
        pd.Series(pd.date_range("2020-01-01", periods=3).date),
    ],
    ids=["pandas_int", "polars_int", "pandas_date_objects"],
)
def test_non_datetime_dates_raise(dates):
    with pytest.raises(ValueError, match="'month', dates must be datetimes"):
        dtf.month.compute(dates)


def test_pickling_keeps_identity():
    assert pickle.loads(pickle.dumps(dtf.month)) is dtf.month
    assert copy.deepcopy(dtf.day_of_week) is dtf.day_of_week
