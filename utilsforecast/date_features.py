"""Backend-agnostic calendar features"""

__all__ = [
    "CalendarFeature",
    "available",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "day_of_week",
    "day_of_year",
    "days_in_month",
    "hour",
    "minute",
    "second",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]


import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalendarFeature:
    """Calendar feature with the same values and dtype for pandas and polars dates.

    The features are listed by `available()`, custom features can be passed to
    `time_features` as functions.

    Attributes:
        name (str): Name of the feature, used as the column name.
        description (str): Description of the values.
        dtype (type): numpy dtype of the computed values.
        values (range, optional): Values to one-hot encode the feature with, None if it isn't one-hot encoded.
    """

    name: str
    description: str = field(repr=False)
    dtype: type = field(repr=False)
    values: Optional[range] = field(repr=False)
    _compute: Callable[[np.ndarray], np.ndarray] = field(repr=False, compare=False)

    def compute(self, dates) -> np.ndarray:
        """Compute the feature values.

        Args:
            dates (pandas or polars Series, or pandas DatetimeIndex): Dates to compute the feature from.
                Time zone aware dates use their local time.

        Returns:
            numpy.ndarray: Feature values, one per date.

        Raises:
            ValueError: If the dates aren't datetimes or there are null dates.
        """
        return _compute_features(dates, [self])[0]

    def __reduce__(self):
        if _FEATURES.get(self.name) is not self:
            raise pickle.PicklingError(
                f"Can't pickle {self.name!r}, only the features listed by `available()` can be pickled."
            )
        return _from_name, (self.name,)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _to_datetime64(dates) -> np.ndarray:
    if isinstance(dates, pd.Index):
        dates = pd.Series(dates)
    if isinstance(dates, pd.Series) and isinstance(dates.dtype, pd.ArrowDtype):
        if dates.dtype.kind == "M":
            # removing the time zone from pyarrow dates returns UTC times
            tz = getattr(dates.dtype.pyarrow_dtype, "tz", None)
            if tz is None:
                dates = dates.astype(dates.dtype.numpy_dtype)
            else:
                unit = dates.dtype.pyarrow_dtype.unit
                dates = dates.astype(pd.DatetimeTZDtype(unit=unit, tz=tz))
    dates = nw.from_native(dates, series_only=True)
    native = dates.to_native()
    # narwhals reports some pandas time zones (e.g. America/New_York) as Unknown
    is_datetime = isinstance(dates.dtype, (nw.Date, nw.Datetime)) or (
        isinstance(native, pd.Series) and pd.api.types.is_datetime64_any_dtype(native)
    )
    if not is_datetime:
        raise ValueError(
            f"Can't compute calendar features, dates must be datetimes, got {dates.dtype}."
        )
    n_nulls = dates.null_count()
    if n_nulls:
        raise ValueError(
            f"Can't compute calendar features, found {n_nulls} null dates."
        )
    if dates.dtype != nw.Date:
        # not checking the dtype's time zone because narwhals reports some
        # (e.g. America/New_York) as Unknown
        dates = dates.dt.replace_time_zone(None)
    datetimes = dates.to_numpy()
    unit, _ = np.datetime_data(datetimes.dtype)
    per_second = {"ms": 10**3, "us": 10**6, "ns": 10**9}.get(unit)
    if per_second is None:
        return datetimes
    # numpy's astype overflows when flooring dates close to the minimum int64
    seconds = datetimes.view(np.int64) // per_second
    return seconds.astype("datetime64[s]")


def _compute_features(dates, features: List[CalendarFeature]) -> List[np.ndarray]:
    if not features:
        return []
    datetimes = _to_datetime64(dates)
    return [f._compute(datetimes).astype(f.dtype) for f in features]


def _elapsed(dates: np.ndarray, unit: str, period: str) -> np.ndarray:
    # numpy floors when casting to a coarser unit, so this also works before 1970
    start = dates.astype(f"datetime64[{period}]")
    return (dates.astype(f"datetime64[{unit}]") - start).astype(np.int64)


def _month(dates: np.ndarray) -> np.ndarray:
    return dates.astype("datetime64[M]").astype(np.int64) % 12 + 1


def _day_of_week(dates: np.ndarray) -> np.ndarray:
    # 1970-01-01 was a thursday
    return (dates.astype("datetime64[D]").astype(np.int64) + 3) % 7


def _days_in_month(dates: np.ndarray) -> np.ndarray:
    month = dates.astype("datetime64[M]")
    return ((month + 1).astype("datetime64[D]") - month).astype(np.int64)


def _week(dates: np.ndarray) -> np.ndarray:
    # the ISO week is the week of the year of the thursday of that week
    days = dates.astype("datetime64[D]")
    thursday = days + (3 - _day_of_week(days)).astype("timedelta64[D]")
    return _elapsed(thursday, "D", "Y") // 7 + 1


def _is_month_end(dates: np.ndarray) -> np.ndarray:
    return _elapsed(dates, "D", "M") + 1 == _days_in_month(dates)


year = CalendarFeature(
    "year",
    "Year.",
    np.int16,
    None,
    lambda d: d.astype("datetime64[Y]").astype(np.int64) + 1970,
)
quarter = CalendarFeature(
    "quarter",
    "Quarter of the year, from 1 to 4.",
    np.int8,
    range(1, 5),
    lambda d: (_month(d) - 1) // 3 + 1,
)
month = CalendarFeature(
    "month", "Month of the year, from 1 to 12.", np.int8, range(1, 13), _month
)
week = CalendarFeature(
    "week", "ISO week of the year, from 1 to 53.", np.int8, range(1, 54), _week
)
day = CalendarFeature(
    "day",
    "Day of the month, from 1 to 31.",
    np.int8,
    range(1, 32),
    lambda d: _elapsed(d, "D", "M") + 1,
)
day_of_week = CalendarFeature(
    "day_of_week",
    "Day of the week, from 0 (Monday) to 6 (Sunday).",
    np.int8,
    range(7),
    _day_of_week,
)
day_of_year = CalendarFeature(
    "day_of_year",
    "Day of the year, from 1 to 366.",
    np.int16,
    range(1, 367),
    lambda d: _elapsed(d, "D", "Y") + 1,
)
days_in_month = CalendarFeature(
    "days_in_month",
    "Number of days in the month, from 28 to 31.",
    np.int8,
    range(28, 32),
    _days_in_month,
)
hour = CalendarFeature(
    "hour",
    "Hour of the day, from 0 to 23.",
    np.int8,
    range(24),
    lambda d: _elapsed(d, "h", "D"),
)
minute = CalendarFeature(
    "minute",
    "Minute of the hour, from 0 to 59.",
    np.int8,
    range(60),
    lambda d: _elapsed(d, "m", "h"),
)
second = CalendarFeature(
    "second",
    "Second of the minute, from 0 to 59.",
    np.int8,
    range(60),
    lambda d: _elapsed(d, "s", "m"),
)
is_month_start = CalendarFeature(
    "is_month_start",
    "1 on the first day of the month, 0 otherwise.",
    np.int8,
    None,
    lambda d: _elapsed(d, "D", "M") == 0,
)
is_month_end = CalendarFeature(
    "is_month_end",
    "1 on the last day of the month, 0 otherwise.",
    np.int8,
    None,
    _is_month_end,
)
is_quarter_start = CalendarFeature(
    "is_quarter_start",
    "1 on the first day of the quarter, 0 otherwise.",
    np.int8,
    None,
    lambda d: (_elapsed(d, "D", "M") == 0) & (_month(d) % 3 == 1),
)
is_quarter_end = CalendarFeature(
    "is_quarter_end",
    "1 on the last day of the quarter, 0 otherwise.",
    np.int8,
    None,
    lambda d: _is_month_end(d) & (_month(d) % 3 == 0),
)
is_year_start = CalendarFeature(
    "is_year_start",
    "1 on January 1st, 0 otherwise.",
    np.int8,
    None,
    lambda d: _elapsed(d, "D", "Y") == 0,
)
is_year_end = CalendarFeature(
    "is_year_end",
    "1 on December 31st, 0 otherwise.",
    np.int8,
    None,
    lambda d: _is_month_end(d) & (_month(d) == 12),
)

_FEATURES: Dict[str, CalendarFeature] = {
    f.name: f
    for f in [
        year,
        quarter,
        month,
        week,
        day,
        day_of_week,
        day_of_year,
        days_in_month,
        hour,
        minute,
        second,
        is_month_start,
        is_month_end,
        is_quarter_start,
        is_quarter_end,
        is_year_start,
        is_year_end,
    ]
}


def _from_name(name: str) -> CalendarFeature:
    return _FEATURES[name]


def available() -> List[CalendarFeature]:
    """List the available calendar features.

    Returns:
        list of CalendarFeature: Available features.
    """
    return list(_FEATURES.values())
