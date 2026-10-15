import pandas as pd
import polars as pl
import pytest

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import coverage, mae


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "model", ["model-hi-2025-v2", "baseline-lo-10-tuned", "plain-model"]
)
@pytest.mark.parametrize("level", [None, 90, 90.5, 1e-5])
def test_evaluate_infers_models_with_interval_like_names(engine, model, level):
    data = {
        "unique_id": ["a", "a"],
        "ds": [1, 2],
        "y": [10.0, 20.0],
        "baseline": [10.0, 18.0],
        model: [12.0, 18.0],
    }
    models = ["baseline", model]
    metrics = [mae]
    if level is not None:
        for name in models:
            data[f"{name}-lo-{level}"] = [8.0, 17.0]
            data[f"{name}-hi-{level}"] = [14.0, 22.0]
        metrics.append(coverage)
        level = [level]
    df = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)

    inferred = evaluate(df, metrics=metrics, level=level)
    explicit = evaluate(df, metrics=metrics, models=models, level=level)

    assert set(inferred.columns) == {"unique_id", "metric", *models}
    if engine == "polars":
        inferred = inferred.to_pandas()
        explicit = explicit.to_pandas()
    pd.testing.assert_frame_equal(inferred, explicit)
    errors = inferred.loc[inferred["metric"] == "mae"]
    assert errors[model].iloc[0] == 2.0
    assert errors["baseline"].iloc[0] == 1.0
