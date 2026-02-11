"""Synthetic time series generation for controlled benchmarking.

This module provides :class:`TimeSeriesSimulator`, a utility for generating
synthetic time series with configurable statistical distributions, trend,
seasonality and noise components.  The output is a panel DataFrame compatible
with the Nixtla ecosystem (``unique_id``, ``ds``, ``y``), making it
straightforward to feed into ``StatsForecast``, ``MLForecast`` or any other
framework that consumes the standard Nixtla format.

Custom distribution callables allow practitioners to encode domain-specific
data-generation processes—promotional spikes, regime changes, bimodal demand
patterns—so that model benchmarks reflect realistic scenarios.
"""

__all__ = ["TimeSeriesSimulator"]

from typing import Callable, Dict, List, Literal, Optional, Union, overload

import numpy as np
import pandas as pd

from .compat import pl, pl_DataFrame

# ---------------------------------------------------------------------------
# Built-in distribution registry
# ---------------------------------------------------------------------------

_BUILTIN_DISTRIBUTIONS = frozenset(
    {
        "normal",
        "poisson",
        "exponential",
        "gamma",
        "uniform",
        "binomial",
        "lognormal",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_from_distribution(
    distribution: Union[str, Callable[..., np.ndarray]],
    dist_params: Dict,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *length* values from *distribution*.

    Parameters
    ----------
    distribution : str or callable
        Either the name of a built-in distribution or a callable with
        signature ``(size: int, rng: numpy.random.Generator) -> ndarray``.
    dist_params : dict
        Keyword arguments forwarded to the built-in distribution sampler.
    length : int
        Number of data points to generate.
    rng : numpy.random.Generator
        Random number generator used for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of sampled values with shape ``(length,)``.
    """
    if callable(distribution):
        return np.asarray(distribution(length, rng), dtype=np.float64)

    generators = {
        "normal": lambda: rng.normal(
            dist_params.get("loc", 0.0),
            dist_params.get("scale", 1.0),
            size=length,
        ),
        "poisson": lambda: rng.poisson(
            dist_params.get("lam", 5.0),
            size=length,
        ).astype(np.float64),
        "exponential": lambda: rng.exponential(
            dist_params.get("scale", 1.0),
            size=length,
        ),
        "gamma": lambda: rng.gamma(
            dist_params.get("shape", 2.0),
            dist_params.get("scale", 2.0),
            size=length,
        ),
        "uniform": lambda: rng.uniform(
            dist_params.get("low", 0.0),
            dist_params.get("high", 1.0),
            size=length,
        ),
        "binomial": lambda: rng.binomial(
            dist_params.get("n", 10),
            dist_params.get("p", 0.5),
            size=length,
        ).astype(np.float64),
        "lognormal": lambda: rng.lognormal(
            dist_params.get("mean", 0.0),
            dist_params.get("sigma", 1.0),
            size=length,
        ),
    }
    if distribution not in generators:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Expected one of {sorted(_BUILTIN_DISTRIBUTIONS)} or a callable."
        )
    return generators[distribution]()


def _generate_trend(
    trend: Union[str, Callable[..., np.ndarray]],
    trend_params: Dict,
    length: int,
) -> np.ndarray:
    """Build an additive trend component.

    Parameters
    ----------
    trend : str or callable
        One of ``'linear'``, ``'quadratic'``, ``'exponential'`` or a callable
        that receives an array of time indices and returns trend values.
    trend_params : dict
        Parameters for the built-in trend functions.
    length : int
        Number of data points.

    Returns
    -------
    numpy.ndarray
        Trend values with shape ``(length,)``.
    """
    t = np.arange(length, dtype=np.float64)

    if callable(trend):
        return np.asarray(trend(t), dtype=np.float64)

    if trend == "linear":
        slope = trend_params.get("slope", 1.0)
        intercept = trend_params.get("intercept", 0.0)
        return slope * t + intercept

    if trend == "quadratic":
        a = trend_params.get("a", 0.01)
        b = trend_params.get("b", 0.0)
        c = trend_params.get("c", 0.0)
        return a * t**2 + b * t + c

    if trend == "exponential":
        base = trend_params.get("base", 1.01)
        scale = trend_params.get("scale", 1.0)
        return scale * (np.power(base, t) - 1.0)

    raise ValueError(
        f"Unknown trend '{trend}'. "
        "Expected 'linear', 'quadratic', 'exponential' or a callable."
    )


def _generate_seasonality(
    seasonality: Union[int, List[int]],
    seasonality_strength: Union[float, List[float]],
    length: int,
) -> np.ndarray:
    """Build an additive seasonal component.

    Parameters
    ----------
    seasonality : int or list of int
        Period(s) expressed in the number of time steps.
    seasonality_strength : float or list of float
        Amplitude(s) of the sinusoidal seasonal component(s).
    length : int
        Number of data points.

    Returns
    -------
    numpy.ndarray
        Seasonal values with shape ``(length,)``.
    """
    t = np.arange(length, dtype=np.float64)
    periods = [seasonality] if isinstance(seasonality, int) else list(seasonality)
    strengths: List[float]
    if isinstance(seasonality_strength, (int, float)):
        strengths = [float(seasonality_strength)]
    else:
        strengths = [float(s) for s in seasonality_strength]

    if len(strengths) == 1 and len(periods) > 1:
        strengths = strengths * len(periods)
    elif len(strengths) != len(periods):
        raise ValueError(
            f"seasonality_strength must be a single value or match the number "
            f"of seasonal periods ({len(periods)}), got {len(strengths)}."
        )

    seasonal = np.zeros(length, dtype=np.float64)
    for period, strength in zip(periods, strengths):
        seasonal += strength * np.sin(2.0 * np.pi * t / period)
    return seasonal


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TimeSeriesSimulator:
    """Generate synthetic time series with specified distributions.

    ``TimeSeriesSimulator`` produces panel data that follows a configurable
    statistical distribution (or a user-supplied callable) with optional trend,
    seasonality and noise components.  This is useful for systematically
    benchmarking forecasting models against data with known properties.

    Args:
        length (int): Number of time steps per series.  Defaults to 100.
            When *seasonality* is provided and *length* is less than
            ``3 * max(seasonality)``, it is automatically raised so that
            at least three full seasonal cycles are present.
        distribution (str or callable): Distribution for the base values.

            Built-in options:

            * ``'normal'`` — Normal / Gaussian (params: *loc*, *scale*)
            * ``'poisson'`` — Poisson (params: *lam*)
            * ``'exponential'`` — Exponential (params: *scale*)
            * ``'gamma'`` — Gamma (params: *shape*, *scale*)
            * ``'uniform'`` — Uniform (params: *low*, *high*)
            * ``'binomial'`` — Binomial (params: *n*, *p*)
            * ``'lognormal'`` — Log-normal (params: *mean*, *sigma*)

            If a callable is passed it must have the signature
            ``(size: int, rng: numpy.random.Generator) -> numpy.ndarray``
            and return an array of *size* values.  Defaults to ``'normal'``.
        dist_params (dict, optional): Keyword arguments forwarded to the
            built-in distribution sampler.  Ignored when *distribution* is a
            callable.
        trend (str or callable, optional): Additive trend component.

            Built-in options: ``'linear'``, ``'quadratic'``,
            ``'exponential'``.  A callable must accept a 1-D array of time
            indices and return an array of the same length.
        trend_params (dict, optional): Parameters for built-in trends.

            * ``'linear'``: *slope* (default 1.0), *intercept* (default 0.0)
            * ``'quadratic'``: *a* (default 0.01), *b* (default 0.0),
              *c* (default 0.0)
            * ``'exponential'``: *base* (default 1.01), *scale* (default 1.0)
        seasonality (int or list of int, optional): Seasonal period(s)
            expressed in the number of time steps.
        seasonality_strength (float or list of float): Amplitude(s) of
            the sinusoidal seasonal component(s).  A single value is
            broadcast to every period.  Defaults to 1.0.
        noise_std (float): Standard deviation of additive Gaussian noise
            applied on top of all other components.  Defaults to 0.0.
        freq (str): Pandas frequency alias for the time index.
            Defaults to ``'D'``.
        start (str): Start date for the time index.
            Defaults to ``'2020-01-01'``.
        seed (int, optional): Seed for ``numpy.random.default_rng`` to
            ensure reproducibility.

    Examples:
        Generate a basic panel with three normally distributed series:

        >>> from utilsforecast.synthetic import TimeSeriesSimulator
        >>> sim = TimeSeriesSimulator(
        ...     length=100,
        ...     distribution="normal",
        ...     dist_params={"loc": 100, "scale": 10},
        ...     seed=42,
        ... )
        >>> df = sim.simulate(n_series=3)
        >>> df.shape
        (300, 3)

        Generate demand data with custom promotional spikes:

        >>> def demand_with_spikes(size, rng):
        ...     base = rng.gamma(shape=5, scale=10, size=size)
        ...     mask = rng.random(size) < 0.05
        ...     base[mask] *= rng.uniform(2.5, 5.0, size=mask.sum())
        ...     return base
        >>> sim = TimeSeriesSimulator(
        ...     length=365,
        ...     distribution=demand_with_spikes,
        ...     trend="linear",
        ...     trend_params={"slope": 0.05},
        ...     seasonality=7,
        ...     seasonality_strength=10.0,
        ...     seed=42,
        ... )
        >>> df = sim.simulate(n_series=10)

        Generate series with multiple seasonalities:

        >>> sim = TimeSeriesSimulator(
        ...     length=365,
        ...     distribution="gamma",
        ...     dist_params={"shape": 2, "scale": 50},
        ...     seasonality=[7, 30],
        ...     seasonality_strength=[10.0, 5.0],
        ...     seed=42,
        ... )
        >>> df = sim.simulate()
    """

    def __init__(
        self,
        length: int = 100,
        distribution: Union[str, Callable[..., np.ndarray]] = "normal",
        dist_params: Optional[Dict] = None,
        trend: Optional[Union[str, Callable[..., np.ndarray]]] = None,
        trend_params: Optional[Dict] = None,
        seasonality: Optional[Union[int, List[int]]] = None,
        seasonality_strength: Union[float, List[float]] = 1.0,
        noise_std: float = 0.0,
        freq: str = "D",
        start: str = "2020-01-01",
        seed: Optional[int] = None,
    ):
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        if isinstance(distribution, str) and distribution not in _BUILTIN_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                f"Expected one of {sorted(_BUILTIN_DISTRIBUTIONS)} or a callable."
            )

        # Ensure enough room for at least three full seasonal cycles so that
        # downstream models can learn the pattern.
        if seasonality is not None:
            max_period = (
                max(seasonality)
                if isinstance(seasonality, (list, tuple))
                else seasonality
            )
            min_length = 3 * max_period
            if length < min_length:
                length = min_length

        self.length = length
        self.distribution = distribution
        self.dist_params: Dict = dist_params if dist_params is not None else {}
        self.trend = trend
        self.trend_params: Dict = trend_params if trend_params is not None else {}
        self.seasonality = seasonality
        self.seasonality_strength = seasonality_strength
        self.noise_std = noise_std
        self.freq = freq
        self.start = start
        self.seed = seed

    # -- public API ---------------------------------------------------------

    @overload
    def simulate(
        self,
        n_series: int = ...,
        engine: Literal["pandas"] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def simulate(
        self,
        n_series: int = ...,
        engine: Literal["polars"] = ...,
    ) -> pl_DataFrame: ...

    def simulate(
        self,
        n_series: int = 1,
        engine: Literal["pandas", "polars"] = "pandas",
    ) -> Union[pd.DataFrame, pl_DataFrame]:
        """Generate synthetic time series panel data.

        Args:
            n_series (int): Number of independent series to generate.
                Each series receives a unique integer ``unique_id``.
                Defaults to 1.
            engine (str): Output format — ``'pandas'`` or ``'polars'``.
                Defaults to ``'pandas'``.

        Returns:
            pandas or polars DataFrame: Panel with columns
                [``unique_id``, ``ds``, ``y``].

        Raises:
            ValueError: If *engine* is not ``'pandas'`` or ``'polars'``.
            ValueError: If polars output is requested but polars is not
                installed.
        """
        available_engines = ("pandas", "polars")
        if engine not in available_engines:
            raise ValueError(
                f"'{engine}' is not a valid engine; "
                f"available options: {list(available_engines)}"
            )
        if engine == "polars" and pl is None:
            raise ImportError(
                "polars is required for engine='polars'. "
                "Install it with: pip install polars"
            )

        rng = np.random.default_rng(self.seed)
        dates = pd.date_range(start=self.start, periods=self.length, freq=self.freq)

        # Pre-compute deterministic components (shared across series)
        trend_component: Optional[np.ndarray] = None
        if self.trend is not None:
            trend_component = _generate_trend(
                self.trend, self.trend_params, self.length
            )

        seasonal_component: Optional[np.ndarray] = None
        if self.seasonality is not None:
            seasonal_component = _generate_seasonality(
                self.seasonality, self.seasonality_strength, self.length
            )

        # Build arrays for the panel
        all_ids = np.repeat(np.arange(n_series), self.length)
        all_dates = np.tile(dates.values, n_series)
        all_values = np.empty(n_series * self.length, dtype=np.float64)

        for i in range(n_series):
            start_idx = i * self.length
            end_idx = start_idx + self.length
            values = _generate_from_distribution(
                self.distribution, self.dist_params, self.length, rng
            )
            if trend_component is not None:
                values = values + trend_component
            if seasonal_component is not None:
                values = values + seasonal_component
            if self.noise_std > 0:
                values = values + rng.normal(0.0, self.noise_std, size=self.length)
            all_values[start_idx:end_idx] = values

        if engine == "pandas":
            return pd.DataFrame(
                {"unique_id": all_ids, "ds": all_dates, "y": all_values}
            )

        # polars path
        return pl.DataFrame(
            {
                "unique_id": all_ids,
                "ds": all_dates,
                "y": all_values,
            }
        )
