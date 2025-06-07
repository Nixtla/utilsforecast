import matplotlib.pyplot as plt
from utilsforecast.data import generate_series
from utilsforecast.plotting import plot_series

level = [80, 95]
series = generate_series(
    4, freq='D', equal_ends=True, with_trend=True, n_models=2, level=level
)
test_pd = series.groupby('unique_id', observed=True).tail(10).copy()
train_pd = series.drop(test_pd.index)
plt.style.use('ggplot')
fig = plot_series(
    train_pd,
    forecasts_df=test_pd,
    ids=[0, 3],
    plot_random=False,
    level=level,
    max_insample_length=50,
    engine='matplotlib',
    plot_anomalies=True,
)
fig.savefig('imgs/plotting.png', bbox_inches='tight')
import warnings
from itertools import product

from fastcore.test import test_fail

from utilsforecast.compat import POLARS_INSTALLED

if POLARS_INSTALLED:
    import polars as pl
try:
    import plotly

    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False
try:
    import plotly_resampler

    PLOTLY_RESAMPLER_INSTALLED = True
except ImportError:
    PLOTLY_RESAMPLER_INSTALLED = False
bools = [True, False]
polars = bools if POLARS_INSTALLED else [False]
anomalies = bools
randoms = bools
forecasts = bools
ids = [[0], [3, 1], None]
levels = [[80], None]
max_insample_lengths = [None, 50]
engines = ['matplotlib']
if POLARS_INSTALLED:
    train_pl = pl.DataFrame(train_pd.to_records(index=False))
    test_pl = pl.DataFrame(test_pd.to_records(index=False))
if PLOTLY_INSTALLED:
    engines.append('plotly')
if PLOTLY_RESAMPLER_INSTALLED:
    engines.append('plotly-resampler')
iterable = product(
    polars, ids, anomalies, levels, max_insample_lengths, engines, randoms, forecasts
)

for (
    as_polars,
    ids,
    plot_anomalies,
    level,
    max_insample_length,
    engine,
    plot_random,
    with_forecasts,
) in iterable:
    if POLARS_INSTALLED and as_polars:
        train = train_pl
        test = test_pl if with_forecasts else None
    else:
        train = train_pd
        test = test_pd if with_forecasts else None
    fn = lambda: plot_series(
        train,
        forecasts_df=test,
        ids=ids,
        plot_random=plot_random,
        plot_anomalies=plot_anomalies,
        level=level,
        max_insample_length=max_insample_length,
        engine=engine,
    )
    if level is None and plot_anomalies:
        test_fail(fn, contains='specify the `level` argument')
    elif level is not None and plot_anomalies and not with_forecasts:
        test_fail(fn, contains='provide a `forecasts_df` with prediction')
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='The behavior of DatetimeProperties.to_pydatetime is deprecated',
                category=FutureWarning,
            )
            fn()
