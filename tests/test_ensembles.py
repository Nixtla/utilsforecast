from functools import partial

import numpy as np
import pandas as pd
import pytest

from utilsforecast.ensembles import (
    add_conformal_error_intervals,
    add_mean_ensemble,
    apply_ensemble,
    fit_ensemble,
    fit_conformal_error_intervals,
    fit_greedy_ensemble,
)
from utilsforecast.losses import mase, rmse


def test_add_mean_ensemble_pandas():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "y": [0.0, 0.0],
            "model_a": [1.0, 2.0],
            "model_b": [3.0, 4.0],
            "model_c": [5.0, 6.0],
        }
    )
    result = add_mean_ensemble(df, ensemble_name="mean")
    np.testing.assert_allclose(result["mean"].to_numpy(), np.array([3.0, 4.0]))


def test_add_mean_ensemble_polars():
    pl = pytest.importorskip("polars", reason="polars is not installed")
    polars_testing = pytest.importorskip(
        "polars.testing", reason="polars is not installed"
    )
    df = pl.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "y": [0.0, 0.0],
            "model_a": [1.0, 2.0],
            "model_b": [3.0, 4.0],
        }
    )
    result = add_mean_ensemble(df, ensemble_name="mean")
    expected = df.with_columns(mean=pl.Series("mean", [2.0, 3.0]))
    polars_testing.assert_frame_equal(result, expected)


def test_fit_greedy_ensemble_global_allows_repeated_models():
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "cutoff": [0, 0],
            "y": [0.0, 10.0],
            "model_a": [0.0, 0.0],
            "model_b": [10.0, 10.0],
        }
    )
    weights = fit_greedy_ensemble(cv_df, metric=rmse, max_iters=3)
    np.testing.assert_allclose(weights.loc[0, ["model_a", "model_b"]], [2 / 3, 1 / 3])

    forecast_df = cv_df.drop(columns=["y", "cutoff"])
    result = apply_ensemble(forecast_df, weights, ensemble_name="greedy")
    np.testing.assert_allclose(
        result["greedy"].to_numpy(),
        (2 * forecast_df["model_a"].to_numpy() + forecast_df["model_b"].to_numpy()) / 3,
    )


def test_fit_greedy_ensemble_global_averages_scores_across_cutoffs():
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a"],
            "ds": [1, 2, 3, 4],
            "cutoff": [0, 0, 2, 2],
            "y": [0.0, 1.0, 2.0, 3.0],
            "model_a": [0.0, 1.0, 0.0, 0.0],
            "model_b": [1.0, 1.0, 2.0, 3.0],
        }
    )
    weights = fit_greedy_ensemble(cv_df, metric=rmse, max_iters=1)
    np.testing.assert_allclose(weights.loc[0, ["model_a", "model_b"]], [0.0, 1.0])


def test_fit_ensemble_mean():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "model_a": [1.0, 2.0],
            "model_b": [3.0, 4.0],
            "model_c": [5.0, 6.0],
        }
    )
    weights = fit_ensemble(df, method="mean")
    expected = pd.DataFrame(
        [{"model_a": 1.0, "model_b": 1.0, "model_c": 1.0}]
    )
    pd.testing.assert_frame_equal(weights, expected)


def test_fit_greedy_ensemble_local():
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "cutoff": [0, 0, 0, 0],
            "y": [0.0, 10.0, 1.0, 9.0],
            "model_a": [0.0, 0.0, 8.0, 8.0],
            "model_b": [10.0, 10.0, 1.0, 9.0],
        }
    )
    weights = fit_greedy_ensemble(cv_df, metric=rmse, kind="local", max_iters=1)
    expected = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "model_a": [1.0, 0.0],
            "model_b": [0.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(weights, expected)

    forecast_df = cv_df.drop(columns=["y", "cutoff"])
    result = apply_ensemble(forecast_df, weights, ensemble_name="local_greedy")
    np.testing.assert_allclose(
        result["local_greedy"].to_numpy(),
        np.array([0.0, 0.0, 1.0, 9.0]),
    )


def test_fit_greedy_ensemble_local_averages_scores_across_cutoffs():
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "cutoff": [0, 1, 0, 1],
            "y": [0.0, 0.0, 1.0, 1.0],
            "model_a": [0.0, 0.0, 0.0, 0.0],
            "model_b": [1.0, 1.0, 1.0, 1.0],
        }
    )
    weights = fit_greedy_ensemble(cv_df, metric=rmse, kind="local", max_iters=1)
    expected = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "model_a": [1.0, 0.0],
            "model_b": [0.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(weights, expected)


def test_fit_ensemble_greedy_aliases():
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "cutoff": [0, 0, 0, 0],
            "y": [0.0, 10.0, 1.0, 9.0],
            "model_a": [0.0, 0.0, 8.0, 8.0],
            "model_b": [10.0, 10.0, 1.0, 9.0],
        }
    )
    global_weights = fit_ensemble(
        cv_df.iloc[:2],
        method="greedy_global",
        metric=rmse,
        max_iters=3,
    )
    np.testing.assert_allclose(
        global_weights.loc[0, ["model_a", "model_b"]],
        [2 / 3, 1 / 3],
    )

    local_weights = fit_ensemble(
        cv_df,
        method="greedy_local",
        metric=rmse,
        max_iters=1,
    )
    expected_local = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "model_a": [1.0, 0.0],
            "model_b": [0.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(local_weights, expected_local)

    local_weights_via_kind = fit_ensemble(
        cv_df,
        method="greedy",
        kind="local",
        metric=rmse,
        max_iters=1,
    )
    pd.testing.assert_frame_equal(local_weights_via_kind, expected_local)


def test_fit_greedy_ensemble_with_train_df():
    train_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a"],
            "ds": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [5, 6],
            "cutoff": [4, 4],
            "y": [5.0, 6.0],
            "model_a": [5.0, 6.0],
            "model_b": [4.0, 7.0],
        }
    )
    weights = fit_greedy_ensemble(
        cv_df,
        metric=partial(mase, seasonality=1),
        train_df=train_df,
        max_iters=1,
    )
    np.testing.assert_allclose(weights.loc[0, ["model_a", "model_b"]], [1.0, 0.0])


def test_fit_greedy_ensemble_rejects_nonstandard_schema_names():
    cv_df = pd.DataFrame(
        {
            "series_id": ["a", "a"],
            "timestamp": [1, 2],
            "cv_cutoff": [0, 0],
            "target": [0.0, 1.0],
            "model_a": [0.0, 1.0],
            "model_b": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="standard Nixtla schema"):
        fit_greedy_ensemble(
            cv_df,
            metric=rmse,
            id_col="series_id",
            time_col="timestamp",
            target_col="target",
            cutoff_col="cv_cutoff",
        )


def test_conformal_intervals_are_fit_from_ensemble_residuals():
    cv_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a"],
            "cutoff": [1, 1, 2, 2],
            "ds": [1, 2, 3, 4],
            "y": [10.0, 20.0, 10.0, 20.0],
            "Ensemble": [11.0, 18.0, 12.0, 19.0],
        }
    )
    future_df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [5, 6],
            "Ensemble": [30.0, 40.0],
        }
    )
    conformal = fit_conformal_error_intervals(cv_df, models=["Ensemble"])
    result = add_conformal_error_intervals(future_df, conformal, level=[80])
    np.testing.assert_allclose(result["Ensemble-lo-80"].to_numpy(), [28.2, 38.2])
    np.testing.assert_allclose(result["Ensemble-hi-80"].to_numpy(), [31.8, 41.8])
