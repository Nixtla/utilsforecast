# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/compat.ipynb.

# %% auto 0
__all__ = ['DataFrame', 'Series']

# %% ../nbs/compat.ipynb 1
from typing import Union

import pandas as pd

# %% ../nbs/compat.ipynb 2
try:
    import polars as pl
    from polars import DataFrame as pl_DataFrame
    from polars import Series as pl_Series

    POLARS_INSTALLED = True
except ImportError:
    pl = None
    pl_DataFrame = type(None)
    pl_Series = type(None)

    POLARS_INSTALLED = False

try:
    import plotly  # noqa: F401

    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False

try:
    import plotly_resampler  # noqa: F401

    PLOTLY_RESAMPLER_INSTALLED = True
except ImportError:
    PLOTLY_RESAMPLER_INSTALLED = False

DataFrame = Union[pd.DataFrame, pl_DataFrame]
Series = Union[pd.Series, pl_Series]
