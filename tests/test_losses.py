import inspect
import math
import warnings

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

import utilsforecast.losses as ufl
from utilsforecast.data import generate_series


warnings.filterwarnings("ignore", message="Unknown section References")


# @pytest.fixture(scope="module")
def setup_series(engine):
    models = ["model0", "model1"]
    series = generate_series(10, n_models=2, level=[80], engine=engine)
    return series, models


@pytest.fixture
def quantile_test_data():
    df = pd.DataFrame(
        {
            "unique_id": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "overestimation": [2.0, 3.0, 4.0],  # y + 1
            "underestimation": [0.0, 1.0, 2.0],  # y - 1
        }
    )
    df["unique_id"] = df["unique_id"].astype("category")
    df = pd.concat([df, df.assign(unique_id=2)]).reset_index(drop=True)

    ql_models_test = ["overestimation", "underestimation"]
    quantiles = np.array([0.1, 0.9])

    return df, ql_models_test, quantiles


@pytest.fixture
def quantile_models():
    return {
        0.1: {
            "model0": "model0-lo-80",
            "model1": "model1-lo-80",
        },
        0.9: {
            "model0": "model0-hi-80",
            "model1": "model1-hi-80",
        },
    }


@pytest.fixture
def multi_quantile_models():
    return {
        "model0": ["model0-lo-80", "model0-hi-80"],
        "model1": ["model1-lo-80", "model1-hi-80"],
    }


def manual_loop(df, models, loss_fn, seasonality=None, baseline=None):
    df = nw.from_native(df)
    results = []
    requires_train = "y_train" in inspect.signature(loss_fn).parameters
    for uid, uid_df in df.group_by("unique_id"):
        uid_res = {"unique_id": uid}
        y_true = uid_df["y"].to_numpy()
        kwargs = {}
        if requires_train:
            kwargs["y_train"] = y_true
        if seasonality is not None:
            kwargs["seasonality"] = seasonality
        if baseline is not None:
            kwargs["y_pred_baseline"] = uid_df[baseline].to_numpy()
        for model in models:
            y_pred = uid_df[model].to_numpy()
            uid_res[model] = loss_fn(y_true, y_pred, **kwargs)
        results.append(uid_res)
    return pd.DataFrame(results)


def mae_single(y_true, y_pred, **kwargs):
    return np.abs(y_true - y_pred).mean()


def mse_single(y_true, y_pred, **kwargs):
    return np.square(y_true - y_pred).mean()


def rmse_single(y_true, y_pred, **kwargs):
    return np.sqrt(np.square(y_true - y_pred).mean())


def bias_single(y_true, y_pred, **kwargs):
    return np.mean(y_pred - y_true)


def mape_single(y_true, y_pred, **kwargs):
    return np.mean(np.abs(y_true - y_pred) / y_true)


def smape_single(y_true, y_pred, **kwargs):
    return np.mean(np.abs(y_true - y_pred) / (y_true + y_pred))


def mase_single(y_true, y_pred, y_train, seasonality, **kwargs):
    scale = np.abs(y_train[:-seasonality] - y_train[seasonality:]).mean()
    return np.abs(y_true - y_pred).mean() / scale


def rmae_single(y_true, y_pred, baseline, **kwargs):
    num = np.abs(y_true - y_pred).mean()
    den = np.abs(y_true - baseline).mean()
    return num / den


def msse_single(y_true, y_pred, seasonality, y_train, **kwargs):
    num = np.square(y_true - y_pred).mean()
    den = np.square(y_train[:-seasonality] - y_train[seasonality:]).mean()
    return num / den


def rmsse_single(y_true, y_pred, seasonality, y_train, **kwargs):
    num = np.square(y_true - y_pred).mean()
    den = np.square(y_train[:-seasonality] - y_train[seasonality:]).mean()
    return math.sqrt(num / den)


def nd_single(y_true, y_pred, **kwargs):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "utils_fn,single_fn",
    [
        (ufl.mae, mae_single),
        (ufl.mse, mse_single),
        (ufl.rmse, rmse_single),
        (ufl.bias, bias_single),
        (ufl.mape, mape_single),
        (ufl.smape, smape_single),
        (ufl.mase, mase_single),
        (ufl.rmae, rmae_single),
        (ufl.msse, msse_single),
        (ufl.rmsse, rmsse_single),
        (ufl.nd, nd_single),
    ],
)
def test_loss(engine, utils_fn, single_fn):
    series, models = setup_series(engine)
    seasonality = 7
    baseline = models[0]
    loss_params = inspect.signature(utils_fn).parameters
    kwargs = {}
    if "seasonality" in loss_params:
        kwargs["seasonality"] = seasonality
    if "train_df" in loss_params:
        kwargs["train_df"] = series
    if "baseline" in loss_params:
        kwargs["baseline"] = baseline
    actual = utils_fn(series, models, **kwargs)

    df = nw.from_native(series)
    results = []
    for uid, uid_df in df.group_by("unique_id"):
        uid_res = {"unique_id": uid}
        y_true = uid_df["y"].to_numpy()
        for model in models:
            y_pred = uid_df[model].to_numpy()
            uid_res[model] = single_fn(
                y_true,
                y_pred,
                y_train=y_true,
                seasonality=7,
                baseline=uid_df[baseline].to_numpy(),
            )
        results.append(uid_res)
    expected = pd.DataFrame(results)

    np.testing.assert_allclose(actual[models], expected[models])


class TestQuantileLoss:
    def test_quantile_loss_basic(self, quantile_test_data):
        df, ql_models_test, quantiles = quantile_test_data

        for q in quantiles:
            ql_df = quantile_loss(
                df, models=dict(zip(ql_models_test, ql_models_test)), q=q
            )
            # For overestimation, delta_y = y - y_hat = -1 so ql = max(-q, -(q-1))
            expected_over = max(-q, -(q - 1))
            assert all(expected_over == ql for ql in ql_df["overestimation"])
            # For underestimation, delta_y = y - y_hat = 1, so ql = max(q, q-1)
            expected_under = max(q, q - 1)
            assert all(expected_under == ql for ql in ql_df["underestimation"])

    def test_quantile_loss_pandas_polars(self, setup_series, quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        for q in quantiles:
            pd_vs_pl(
                quantile_loss(series, quantile_models[q], q=q),
                quantile_loss(series_pl, quantile_models[q], q=q),
                models,
            )

    def test_scaled_quantile_loss(self, setup_series, quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        for q in quantiles:
            pd_vs_pl(
                scaled_quantile_loss(
                    series, quantile_models[q], seasonality=1, train_df=series, q=q
                ),
                scaled_quantile_loss(
                    series_pl,
                    quantile_models[q],
                    seasonality=1,
                    train_df=series_pl,
                    q=q,
                ),
                models,
            )


class TestMultiQuantileLoss:
    def test_multi_quantile_loss_calculation(
        self, setup_series, multi_quantile_models, quantile_models
    ):
        series, _, models = setup_series
        quantiles = np.array([0.1, 0.9])

        expected = (
            pd.concat(
                [
                    quantile_loss(series, models=quantile_models[q], q=q)
                    for q in quantiles
                ]
            )
            .groupby("unique_id", observed=True, as_index=False)
            .mean()
        )

        actual = mqloss(series, models=multi_quantile_models, quantiles=quantiles)
        pd.testing.assert_frame_equal(actual, expected)

    def test_multi_quantile_loss_pandas_polars(
        self, setup_series, multi_quantile_models
    ):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        pd_vs_pl(
            mqloss(series, multi_quantile_models, quantiles=quantiles),
            mqloss(series_pl, multi_quantile_models, quantiles=quantiles),
            models,
        )

    def test_multi_quantile_loss_output_shape(
        self, setup_series, multi_quantile_models
    ):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        for series_df in [series, series_pl]:
            if isinstance(series_df, pd.DataFrame):
                df_test = series_df.assign(
                    unique_id=lambda df: df["unique_id"].astype(str)
                )
            else:
                df_test = series_df.with_columns(pl.col("unique_id").cast(pl.Utf8))

            mql_df = mqloss(df_test, multi_quantile_models, quantiles=quantiles)

            # Check shape
            expected_shape = (series["unique_id"].nunique(), 1 + len(models))
            assert mql_df.shape == expected_shape

            # Check for null values
            if isinstance(mql_df, pd.DataFrame):
                null_vals = mql_df.isna().sum().sum()
            else:
                null_vals = mql_df.select(pl.all().is_null().sum()).sum_horizontal()
            assert null_vals.item() == 0

    def test_scaled_multi_quantile_loss(self, setup_series, multi_quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        pd_vs_pl(
            scaled_mqloss(
                series,
                multi_quantile_models,
                quantiles=quantiles,
                seasonality=1,
                train_df=series,
            ),
            scaled_mqloss(
                series_pl,
                multi_quantile_models,
                quantiles=quantiles,
                seasonality=1,
                train_df=series_pl,
            ),
            models,
        )


class TestProbabilisticMetrics:
    def test_coverage(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            coverage(series, models, 80),
            coverage(series_pl, models, 80),
            models,
        )

    def test_calibration(self, setup_series, quantile_models):
        series, series_pl, models = setup_series
        pd_vs_pl(
            calibration(series, quantile_models[0.1]),
            calibration(series_pl, quantile_models[0.1]),
            models,
        )

    def test_scaled_crps(self, setup_series, multi_quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        pd_vs_pl(
            scaled_crps(series, multi_quantile_models, quantiles),
            scaled_crps(series_pl, multi_quantile_models, quantiles),
            models,
        )


class TestTweedieDeviance:

    @pytest.mark.parametrize("power", [0, 1, 1.5, 2, 3])
    def test_non_zero_handling(self, setup_series, power):
        series, series_pl, models = setup_series
        # for power in [0, 1, 1.5, 2, 3]:
        # Test Pandas vs Polars
        td_pd = tweedie_deviance(series, models, target_col="y", power=power)
        td_pl = tweedie_deviance(series_pl, models, target_col="y", power=power)
        pd_vs_pl(
            td_pd,
            td_pl,
            models,
        )
        # Test for NaNs
        assert (
            not td_pd[models].isna().any().any()
        ), f"NaNs found in pd DataFrame for power {power}"
        assert (
            not td_pl.select(pl.col(models).is_null().any()).sum_horizontal().item()
        ), f"NaNs found in pl DataFrame for power {power}"
        # Test for infinites
        is_infinite = td_pd[models].isin([np.inf, -np.inf]).any().any()
        assert not is_infinite, f"Infinities found in pd DataFrame for power {power}"
        is_infinite_pl = (
            td_pl.select(pl.col(models).is_infinite().any()).sum_horizontal().item()
        )
        assert not is_infinite_pl, f"Infinities found in pl DataFrame for power {power}"

    @pytest.mark.parametrize("power", [0, 1, 1.5])
    def test_zero_handling(self, setup_series, power):
        series, series_pl, models = setup_series
        # Test zero handling (skip power >=2 since it requires all y > 0)
        series.loc[0, "y"] = 0.0  # Set a zero value to test the zero handling
        series.loc[49, "y"] = 0.0  # Set another zero value to test the zero handling
        series_pl[0, "y"] = 0.0  # Set a zero value to test the zero handling
        series_pl[49, "y"] = 0.0  # Set another zero value to test the zero handling

        # Test Pandas vs Polars
        td_pd = tweedie_deviance(series, models, target_col="y", power=power)
        td_pl = tweedie_deviance(series_pl, models, target_col="y", power=power)
        pd_vs_pl(
            td_pd,
            td_pl,
            models,
        )
        # Test for NaNs
        assert (
            not td_pd[models].isna().any().any()
        ), f"NaNs found in pd DataFrame for power {power}"
        assert (
            not td_pl.select(pl.col(models).is_null().any()).sum_horizontal().item()
        ), f"NaNs found in pl DataFrame for power {power}"
        # Test for infinites
        is_infinite = td_pd[models].isin([np.inf, -np.inf]).any().any()
        assert not is_infinite, f"Infinities found in pd DataFrame for power {power}"
        is_infinite_pl = (
            td_pl.select(pl.col(models).is_infinite().any()).sum_horizontal().item()
        )
        assert not is_infinite_pl, f"Infinities found in pl DataFrame for power {power}"
