"""Tests for utilsforecast.synthetic.TimeSeriesSimulator."""

import numpy as np
import pandas as pd
import pytest

from utilsforecast.synthetic import TimeSeriesSimulator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_simulator():
    """Simulator with default normal distribution and fixed seed."""
    return TimeSeriesSimulator(length=100, seed=42)


@pytest.fixture
def gamma_simulator():
    """Gamma-distributed simulator with trend and weekly seasonality."""
    return TimeSeriesSimulator(
        length=180,
        distribution="gamma",
        dist_params={"shape": 5, "scale": 10},
        trend="linear",
        trend_params={"slope": 0.2, "intercept": 0.0},
        seasonality=7,
        seasonality_strength=15.0,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


def test_basic_generation(basic_simulator):
    """Simulate produces a DataFrame with the expected schema and row count."""
    df = basic_simulator.simulate(n_series=2)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"unique_id", "ds", "y"}
    assert len(df) == 200
    assert df["unique_id"].nunique() == 2


def test_single_series_default():
    """Default n_series=1 returns one series."""
    sim = TimeSeriesSimulator(length=50, seed=0)
    df = sim.simulate()
    assert len(df) == 50
    assert df["unique_id"].nunique() == 1


def test_column_dtypes():
    """Output dtypes are datetime for ds and float for y."""
    sim = TimeSeriesSimulator(length=30, seed=0)
    df = sim.simulate()
    assert np.issubdtype(df["ds"].dtype, np.datetime64)
    assert np.issubdtype(df["y"].dtype, np.floating)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility():
    """Same seed produces identical results."""
    df1 = TimeSeriesSimulator(length=50, seed=42).simulate(n_series=3)
    df2 = TimeSeriesSimulator(length=50, seed=42).simulate(n_series=3)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    """Different seeds produce different values."""
    df1 = TimeSeriesSimulator(length=50, seed=42).simulate()
    df2 = TimeSeriesSimulator(length=50, seed=123).simulate()
    assert not df1["y"].equals(df2["y"])


# ---------------------------------------------------------------------------
# Built-in distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    ["normal", "poisson", "exponential", "gamma", "uniform", "binomial", "lognormal"],
)
def test_all_builtin_distributions(dist):
    """Every built-in distribution name produces a valid DataFrame."""
    sim = TimeSeriesSimulator(length=50, distribution=dist, seed=42)
    df = sim.simulate()
    assert len(df) == 50
    assert df["y"].notna().all()


def test_normal_distribution_statistics():
    """Normal(100, 10) mean and std are within expected tolerance."""
    sim = TimeSeriesSimulator(
        length=5000,
        distribution="normal",
        dist_params={"loc": 100, "scale": 10},
        seed=42,
    )
    df = sim.simulate()
    assert abs(df["y"].mean() - 100) < 2
    assert abs(df["y"].std() - 10) < 2


def test_poisson_distribution_mean():
    """Poisson(lam=10) mean should approximate lambda."""
    sim = TimeSeriesSimulator(
        length=5000,
        distribution="poisson",
        dist_params={"lam": 10},
        seed=42,
    )
    df = sim.simulate()
    assert abs(df["y"].mean() - 10) < 1


def test_gamma_distribution_mean():
    """Gamma(shape=2, scale=5) mean should approximate shape*scale=10."""
    sim = TimeSeriesSimulator(
        length=5000,
        distribution="gamma",
        dist_params={"shape": 2, "scale": 5},
        seed=42,
    )
    df = sim.simulate()
    assert abs(df["y"].mean() - 10) < 2


def test_uniform_distribution_bounds():
    """Uniform(0, 100) values lie in [0, 100] with mean ≈ 50."""
    sim = TimeSeriesSimulator(
        length=5000,
        distribution="uniform",
        dist_params={"low": 0, "high": 100},
        seed=42,
    )
    df = sim.simulate()
    assert df["y"].min() >= 0
    assert df["y"].max() <= 100
    assert abs(df["y"].mean() - 50) < 5


# ---------------------------------------------------------------------------
# Custom distribution callable
# ---------------------------------------------------------------------------


def test_custom_distribution_callable():
    """A user-supplied callable is used for sampling."""

    def beta_scaled(size, rng):
        return rng.beta(2, 5, size=size) * 100

    sim = TimeSeriesSimulator(length=200, distribution=beta_scaled, seed=42)
    df = sim.simulate()
    assert len(df) == 200
    # Beta(2,5) mean ≈ 0.286 → scaled mean ≈ 28.6
    assert 20 < df["y"].mean() < 40


def test_custom_distribution_with_spikes():
    """Custom demand-with-spikes produces values above 2× the mean."""

    def demand_with_spikes(size, rng):
        base = rng.gamma(shape=5, scale=10, size=size)
        mask = rng.random(size) < 0.05
        base[mask] *= rng.uniform(2.5, 5.0, size=mask.sum())
        return base

    sim = TimeSeriesSimulator(length=1000, distribution=demand_with_spikes, seed=42)
    df = sim.simulate()
    assert df["y"].max() > df["y"].mean() * 2


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_linear_trend():
    """Linear trend causes later values to be larger than earlier values."""
    sim = TimeSeriesSimulator(
        length=100,
        distribution="normal",
        dist_params={"loc": 0, "scale": 0.1},
        trend="linear",
        trend_params={"slope": 1.0, "intercept": 0.0},
        seed=42,
    )
    df = sim.simulate()
    assert df["y"].iloc[-10:].mean() > df["y"].iloc[:10].mean() + 50


def test_quadratic_trend():
    """Quadratic trend is present in the output."""
    sim = TimeSeriesSimulator(
        length=100,
        distribution="normal",
        dist_params={"loc": 0, "scale": 0.1},
        trend="quadratic",
        trend_params={"a": 0.01},
        seed=42,
    )
    df = sim.simulate()
    assert df["y"].iloc[-1] > df["y"].iloc[0]


def test_exponential_trend():
    """Exponential trend makes the series grow over time."""
    sim = TimeSeriesSimulator(
        length=100,
        distribution="normal",
        dist_params={"loc": 0, "scale": 0.1},
        trend="exponential",
        trend_params={"base": 1.05, "scale": 1.0},
        seed=42,
    )
    df = sim.simulate()
    assert df["y"].iloc[-1] > df["y"].iloc[0]


def test_custom_trend():
    """A callable trend function is applied correctly."""

    def log_trend(t):
        return np.log1p(t) * 10

    sim = TimeSeriesSimulator(
        length=100,
        distribution="normal",
        dist_params={"loc": 0, "scale": 0.1},
        trend=log_trend,
        seed=42,
    )
    df = sim.simulate()
    assert len(df) == 100
    assert df["y"].iloc[-1] > df["y"].iloc[0]


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------


def test_single_seasonality():
    """Values one period apart share the same seasonal effect."""
    sim = TimeSeriesSimulator(
        length=21,
        distribution="normal",
        dist_params={"loc": 100, "scale": 0.01},
        seasonality=7,
        seasonality_strength=10.0,
        seed=42,
    )
    df = sim.simulate()
    # Day 0 and day 7 have the same sin component
    assert abs(df["y"].iloc[0] - df["y"].iloc[7]) < 1


def test_multiple_seasonalities():
    """Multiple seasonal periods produce a valid DataFrame."""
    sim = TimeSeriesSimulator(
        length=100,
        distribution="normal",
        dist_params={"loc": 100, "scale": 0.1},
        seasonality=[7, 30],
        seasonality_strength=[5.0, 10.0],
        seed=42,
    )
    df = sim.simulate()
    assert len(df) == 100


def test_seasonality_auto_length_adjustment():
    """Length is raised when shorter than 3 × max seasonal period."""
    sim = TimeSeriesSimulator(length=10, seasonality=30, seed=42)
    df = sim.simulate()
    assert len(df) == 90  # 3 * 30


def test_seasonality_auto_length_adjustment_warns():
    """Auto length adjustment emits a warning explaining the change."""
    with pytest.warns(UserWarning, match="auto-adjusted from 10 to 90"):
        sim = TimeSeriesSimulator(length=10, seasonality=30, seed=42)
        df = sim.simulate()
        assert len(df) == 90


def test_single_strength_broadcast():
    """A single seasonality_strength is broadcast to all periods."""
    sim = TimeSeriesSimulator(
        length=100,
        seasonality=[7, 30],
        seasonality_strength=5.0,
        seed=42,
    )
    df = sim.simulate()
    assert len(df) == 100


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def test_noise_increases_variance():
    """Additional Gaussian noise increases the standard deviation."""
    kwargs = dict(
        length=5000,
        distribution="normal",
        dist_params={"loc": 100, "scale": 1},
        seed=42,
    )
    df_clean = TimeSeriesSimulator(**kwargs, noise_std=0.0).simulate()
    df_noisy = TimeSeriesSimulator(**kwargs, noise_std=10.0).simulate()
    assert df_noisy["y"].std() > df_clean["y"].std()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_distribution_name():
    """Unknown distribution string raises ValueError at construction."""
    with pytest.raises(ValueError, match="Unknown distribution"):
        TimeSeriesSimulator(distribution="invalid", seed=42)


def test_invalid_trend_name():
    """Unknown trend string raises ValueError at simulate time."""
    sim = TimeSeriesSimulator(trend="invalid", seed=42)
    with pytest.raises(ValueError, match="Unknown trend"):
        sim.simulate()


def test_invalid_length():
    """length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="length must be >= 1"):
        TimeSeriesSimulator(length=0)


def test_mismatched_seasonality_strength():
    """Mismatch between periods and strengths raises ValueError."""
    sim = TimeSeriesSimulator(
        seasonality=[7, 30],
        seasonality_strength=[1.0, 2.0, 3.0],
        seed=42,
    )
    with pytest.raises(ValueError, match="seasonality_strength must"):
        sim.simulate()


def test_invalid_engine():
    """Invalid engine name raises ValueError."""
    sim = TimeSeriesSimulator(length=10, seed=42)
    with pytest.raises(ValueError, match="not a valid engine"):
        sim.simulate(engine="spark")


# ---------------------------------------------------------------------------
# Polars support
# ---------------------------------------------------------------------------


def test_polars_output():
    """engine='polars' returns a polars DataFrame with correct schema."""
    pl = pytest.importorskip("polars")
    sim = TimeSeriesSimulator(length=50, seed=42)
    df = sim.simulate(n_series=2, engine="polars")
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"unique_id", "ds", "y"}
    assert len(df) == 100


def test_polars_pandas_values_match():
    """Pandas and polars outputs contain identical values."""
    pytest.importorskip("polars")
    sim = TimeSeriesSimulator(length=50, seed=42)
    df_pd = sim.simulate(engine="pandas")
    df_pl = sim.simulate(engine="polars").to_pandas()
    # ds columns may differ slightly in dtype; compare y values
    np.testing.assert_array_equal(df_pd["y"].values, df_pl["y"].values)


# ---------------------------------------------------------------------------
# Integration – output format compatibility
# ---------------------------------------------------------------------------


def test_output_compatible_with_generate_series():
    """Output schema matches the convention used by generate_series."""
    sim = TimeSeriesSimulator(length=50, seed=42)
    df = sim.simulate(n_series=3)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert np.issubdtype(df["ds"].dtype, np.datetime64)
    assert np.issubdtype(df["y"].dtype, np.floating)


def test_combined_components(gamma_simulator):
    """A simulator with distribution + trend + seasonality works end to end."""
    df = gamma_simulator.simulate(n_series=5)
    assert len(df) == 180 * 5
    assert df["unique_id"].nunique() == 5
    assert df["y"].notna().all()


def test_custom_freq_and_start():
    """Custom freq and start are respected in the time index."""
    sim = TimeSeriesSimulator(
        length=12,
        freq="MS",
        start="2023-01-01",
        seed=42,
    )
    df = sim.simulate()
    assert df["ds"].iloc[0] == pd.Timestamp("2023-01-01")
    assert df["ds"].iloc[-1] == pd.Timestamp("2023-12-01")
