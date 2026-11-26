import numpy as np
import pandas as pd
import polars as pl
import pytest

from utilsforecast.losses import mse, rmse


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,target,forecast",
    [(np.int8, 120, -120), (np.uint8, 0, 200), (np.int32, 60000, 0)],
)
@pytest.mark.parametrize("loss", [mse, rmse])
def test_squared_losses_promote_integer_operands(engine, dtype, target, forecast, loss):
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a"],
            "cutoff": [1, 1, 2, 2],
            "y": np.array([target, forecast, forecast, forecast], dtype=dtype),
            "model": np.array([forecast, target, forecast, forecast], dtype=dtype),
        }
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)

    result = loss(frame, models=["model"])
    if engine == "polars":
        result = result.to_pandas()
    result = result.sort_values("cutoff")
    error = abs(float(target) - float(forecast))
    expected = error**2 if loss is mse else error
    np.testing.assert_allclose(result["model"], [expected, 0.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype,y,pred", [("int8", -128, 100), ("uint8", 1, 200)])
def test_absolute_and_percentage_losses_match_float_inputs(engine, dtype, y, pred):
    from utilsforecast import losses

    data = pd.DataFrame(
        {
            "unique_id": ["a"],
            "y": np.array([y], dtype=dtype),
            "model": np.array([pred], dtype=dtype),
        }
    )
    reference = data.astype({"y": "float64", "model": "float64"})
    if engine == "polars":
        data, reference = pl.from_pandas(data), pl.from_pandas(reference)
    for loss in (losses.mae, losses.bias, losses.nd, losses.mape, losses.smape):
        np.testing.assert_allclose(
            loss(data, ["model"])["model"].to_numpy(),
            loss(reference, ["model"])["model"].to_numpy(),
            err_msg=loss.__name__,
        )
