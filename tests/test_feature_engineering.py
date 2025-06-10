from functools import partial, reduce

import numpy as np
import pandas as pd
import polars as pl
import polars.testing

from utilsforecast.data import generate_series
from utilsforecast.feature_engineering import (
    fourier,
    future_exog_to_historic,
    pipeline,
    time_features,
    trend,
)

series = generate_series(5, equal_ends=True)
series_pl = generate_series(5, equal_ends=True, engine='polars')


def test_fourier_transform():
    series = generate_series(5, equal_ends=True)

    transformed_df, future_df = fourier(series, freq='D', season_length=7, k=2, h=1)
    transformed_df2, future_df2 = fourier(
        series.sample(frac=1.0), freq='D', season_length=7, k=2, h=1
    )
    pd.testing.assert_frame_equal(
        transformed_df,
        transformed_df2.sort_values(['unique_id', 'ds']).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(future_df, future_df2)

    transformed_pl, future_pl = fourier(series_pl, freq='1d', season_length=7, k=2, h=1)
    transformed_pl2, future_pl2 = fourier(
        series_pl.sample(fraction=1.0), freq='1d', season_length=7, k=2, h=1
    )
    pl.testing.assert_frame_equal(transformed_pl, transformed_pl2)
    pl.testing.assert_frame_equal(future_pl, future_pl2)
    pd.testing.assert_frame_equal(
        transformed_df.drop(columns=['unique_id']),
        transformed_pl.drop('unique_id').to_pandas(),
    )
    pd.testing.assert_frame_equal(
        future_df.drop(columns=['unique_id']), future_pl.drop('unique_id').to_pandas()
    )
    series = generate_series(5, equal_ends=True)
    transformed_df, future_df = trend(series, freq='D', h=1)
    transformed_df
    future_df
    transformed_df, future_df = time_features(
        series, freq='D', features=['month', 'day', 'week'], h=1
    )
    series_with_prices = series.assign(price=np.random.rand(len(series))).sample(
        frac=1.0
    )
    series_with_prices
    transformed_df, future_df = future_exog_to_historic(
        df=series_with_prices,
        freq='D',
        features=['price'],
        h=2,
    )
    pd.testing.assert_frame_equal(
        (
            series_with_prices.sort_values(['unique_id', 'ds'])
            .groupby('unique_id', observed=True)
            .tail(2)[['unique_id', 'price']]
            .reset_index(drop=True)
        ),
        future_df[['unique_id', 'price']],
    )
    series_with_prices_pl = pl.from_pandas(
        series_with_prices.astype({'unique_id': 'int64'})
    )
    transformed_pl, future_pl = future_exog_to_historic(
        df=series_with_prices_pl,
        freq='1d',
        features=['price'],
        h=2,
    )
    pd.testing.assert_frame_equal(
        future_pl.to_pandas(), future_df.astype({'unique_id': 'int64'})
    )


def is_weekend(times):
    if isinstance(times, pd.Index):
        dow = times.weekday + 1  # monday=0 in pandas and 1 in polars
    else:
        dow = times.dt.weekday()
    return dow >= 6


def even_days_and_months(times):
    if isinstance(times, pd.Index):
        out = pd.DataFrame(
            {
                'even_day': (times.weekday + 1) % 2 == 0,
                'even_month': times.month % 2 == 0,
            }
        )
    else:
        # for polars you can return a list of expressions
        out = [
            (times.dt.weekday() % 2 == 0).alias('even_day'),
            (times.dt.month() % 2 == 0).alias('even_month'),
        ]
    return out


features = [
    trend,
    partial(fourier, season_length=7, k=1),
    partial(fourier, season_length=28, k=1),
    partial(time_features, features=['day', is_weekend, even_days_and_months]),
]
transformed_df, future_df = pipeline(
    series,
    features=features,
    freq='D',
    h=1,
)


def reduce_join(dfs, on):
    return reduce(
        lambda left, right: left.merge(right, on=on, how='left'),
        dfs,
    )


def test_pipeline():
    individual_results = [f(series, freq='D', h=1) for f in features]
    expected_transformed = reduce_join(
        [r[0] for r in individual_results], on=['unique_id', 'ds', 'y']
    )
    expected_future = reduce_join(
        [r[1] for r in individual_results], on=['unique_id', 'ds']
    )
    pd.testing.assert_frame_equal(transformed_df, expected_transformed)
    pd.testing.assert_frame_equal(future_df, expected_future)
    transformed_pl, future_pl = pipeline(
        series_pl,
        features=features,
        freq='1d',
        h=1,
    )
    pd.testing.assert_frame_equal(
        transformed_pl.drop('unique_id').to_pandas(),
        transformed_df.drop(columns='unique_id'),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        future_pl.drop('unique_id').to_pandas(),
        future_df.drop(columns='unique_id'),
        check_dtype=False,
    )
