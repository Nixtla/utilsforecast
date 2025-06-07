import re
import warnings

import pandas as pd
import numpy as np
from nbdev import show_doc
from utilsforecast.losses import (
    mae,
    mse,
    rmse,
    bias,
    mape,
    smape,
    rmae,
    mase,
    msse,
    rmsse,
    quantile_loss,
    scaled_quantile_loss,
    mqloss,
    scaled_mqloss,
    coverage,
    calibration,
    scaled_crps,
)
import polars as pl
from utilsforecast.compat import POLARS_INSTALLED

if POLARS_INSTALLED:
    import polars as pl

warnings.filterwarnings('ignore', message='Unknown section References')
from utilsforecast.data import generate_series

models = ['model0', 'model1']
series = generate_series(10, n_models=2, level=[80])
series_pl = generate_series(10, n_models=2, level=[80], engine='polars')
## 1. Scale-dependent Errors
### Mean Absolute Error (MAE)


# $$
# \mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} |y_{\tau} - \hat{y}_{\tau}|
# $$
def pd_vs_pl(pd_df, pl_df, models):
    np.testing.assert_allclose(
        pd_df[models].to_numpy(),
        pl_df.sort('unique_id').select(models).to_numpy(),
    )


pd_vs_pl(
    mae(series, models),
    mae(series_pl, models),
    models,
)
### Mean Squared Error

# $$
# \mathrm{MSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} (y_{\tau} - \hat{y}_{\tau})^{2}
# $$
pd_vs_pl(
    mse(series, models),
    mse(series_pl, models),
    models,
)
### Root Mean Squared Error

# $$
# \mathrm{RMSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \sqrt{\frac{1}{H} \sum^{t+H}_{\tau=t+1} (y_{\tau} - \hat{y}_{\tau})^{2}}
# $$

pd_vs_pl(
    rmse(series, models),
    rmse(series_pl, models),
    models,
)
pd_vs_pl(
    bias(series, models),
    bias(series_pl, models),
    models,
)
## 2. Percentage Errors
### Mean Absolute Percentage Error

# $$
# \mathrm{MAPE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{|y_{\tau}|}
# $$

pd_vs_pl(
    mape(series, models),
    mape(series_pl, models),
    models,
)
### Symmetric Mean Absolute Percentage Error

# $$
# \mathrm{SMAPE}_{2}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{|y_{\tau}|+|\hat{y}_{\tau}|}
# $$
pd_vs_pl(
    smape(series, models),
    smape(series_pl, models),
    models,
)
## 3. Scale-independent Errors
### Mean Absolute Scaled Error

# $$
# \mathrm{MASE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau}) =
# \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{\mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau})}
# $$

pd_vs_pl(
    mase(series, models, 7, series),
    mase(series_pl, models, 7, series_pl),
    models,
)
### Relative Mean Absolute Error

# $$
# \mathrm{RMAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}, \mathbf{\hat{y}}^{base}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{\mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{base}_{\tau})}
# $$
pd_vs_pl(
    rmae(series, models, models[0]),
    rmae(series_pl, models, models[0]),
    models,
)
### Mean Squared Scaled Error

# $$
# \mathrm{MSSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau}) =
# \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{(y_{\tau}-\hat{y}_{\tau})^2}{\mathrm{MSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau})}
# $$
pd_vs_pl(
    msse(series, models, 7, series),
    msse(series_pl, models, 7, series_pl),
    models,
)
### Root Mean Squared Scaled Error

# $$
# \mathrm{RMSSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau}) =
# \sqrt{\frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{(y_{\tau}-\hat{y}_{\tau})^2}{\mathrm{MSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau})}}
# $$
pd_vs_pl(
    rmsse(series, models, 7, series),
    rmsse(series_pl, models, 7, series_pl),
    models,
)
## 4. Probabilistic Errors
### Quantile Loss

# $$
# \mathrm{QL}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{(q)}_{\tau}) =
# \frac{1}{H} \sum^{t+H}_{\tau=t+1}
# \Big( (1-q)\,( \hat{y}^{(q)}_{\tau} - y_{\tau} )_{+}
# + q\,( y_{\tau} - \hat{y}^{(q)}_{\tau} )_{+} \Big)
# $$
df = pd.DataFrame(
    {
        'unique_id': [0, 1, 2],
        'y': [1.0, 2.0, 3.0],
        'overestimation': [2.0, 3.0, 4.0],  # y + 1.
        'underestimation': [0.0, 1.0, 2.0],  # y - 1.
    }
)
df['unique_id'] = df['unique_id'].astype('category')
df = pd.concat([df, df.assign(unique_id=2)]).reset_index(drop=True)

ql_models_test = ['overestimation', 'underestimation']
quantiles = np.array([0.1, 0.9])

for q in quantiles:
    ql_df = quantile_loss(df, models=dict(zip(ql_models_test, ql_models_test)), q=q)
    # for overestimation, delta_y = y - y_hat = -1 so ql = max(-q, -(q-1))
    assert all(max(-q, -(q - 1)) == ql for ql in ql_df['overestimation'])
    # for underestimation, delta_y = y - y_hat = 1, so ql = max(q, q-1)
    assert all(max(q, q - 1) == ql for ql in ql_df['underestimation'])
q_models = {
    0.1: {
        'model0': 'model0-lo-80',
        'model1': 'model1-lo-80',
    },
    0.9: {
        'model0': 'model0-hi-80',
        'model1': 'model1-hi-80',
    },
}

for q in quantiles:
    pd_vs_pl(
        quantile_loss(series, q_models[q], q=q),
        quantile_loss(series_pl, q_models[q], q=q),
        models,
    )
### Scaled Quantile Loss

# $$
# \mathrm{SQL}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{(q)}_{\tau}) =
# \frac{1}{H} \sum^{t+H}_{\tau=t+1}
# \frac{(1-q)\,( \hat{y}^{(q)}_{\tau} - y_{\tau} )_{+}
# + q\,( y_{\tau} - \hat{y}^{(q)}_{\tau} )_{+}}{\mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau})}
# $$
q_models = {
    0.1: {
        'model0': 'model0-lo-80',
        'model1': 'model1-lo-80',
    },
    0.9: {
        'model0': 'model0-hi-80',
        'model1': 'model1-hi-80',
    },
}

for q in quantiles:
    pd_vs_pl(
        scaled_quantile_loss(series, q_models[q], seasonality=1, train_df=series, q=q),
        scaled_quantile_loss(
            series_pl, q_models[q], seasonality=1, train_df=series_pl, q=q
        ),
        models,
    )
### Multi-Quantile Loss

# $$
# \mathrm{MQL}(\mathbf{y}_{\tau},
# [\mathbf{\hat{y}}^{(q_{1})}_{\tau}, ... ,\hat{y}^{(q_{n})}_{\tau}]) =
# \frac{1}{n} \sum_{q_{i}} \mathrm{QL}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{(q_{i})}_{\tau})
# $$
mq_models = {
    'model0': ['model0-lo-80', 'model0-hi-80'],
    'model1': ['model1-lo-80', 'model1-hi-80'],
}

expected = (
    pd.concat(
        [
            quantile_loss(series, models=q_models[q], q=q)
            for i, q in enumerate(quantiles)
        ]
    )
    .groupby('unique_id', observed=True, as_index=False)
    .mean()
)
actual = mqloss(
    series,
    models=mq_models,
    quantiles=quantiles,
)
pd.testing.assert_frame_equal(actual, expected)
pd_vs_pl(
    mqloss(series, mq_models, quantiles=quantiles),
    mqloss(series_pl, mq_models, quantiles=quantiles),
    models,
)
for series_df in [series, series_pl]:
    if isinstance(series_df, pd.DataFrame):
        df_test = series_df.assign(unique_id=lambda df: df['unique_id'].astype(str))
    else:
        df_test = series_df.with_columns(pl.col('unique_id').cast(pl.Utf8))
    mql_df = mqloss(
        df_test,
        mq_models,
        quantiles=quantiles,
    )
    assert mql_df.shape == (series['unique_id'].nunique(), 1 + len(models))
    if isinstance(mql_df, pd.DataFrame):
        null_vals = mql_df.isna().sum().sum()
    else:
        null_vals = series_df.select(pl.all().is_null().sum()).sum_horizontal()
    assert null_vals.item() == 0
### Scaled Multi-Quantile Loss

# $$
# \mathrm{MQL}(\mathbf{y}_{\tau},
# [\mathbf{\hat{y}}^{(q_{1})}_{\tau}, ... ,\hat{y}^{(q_{n})}_{\tau}]) =
# \frac{1}{n} \sum_{q_{i}} \frac{\mathrm{QL}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{(q_{i})}_{\tau})}{\mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau})}
# $$
pd_vs_pl(
    scaled_mqloss(
        series, mq_models, quantiles=quantiles, seasonality=1, train_df=series
    ),
    scaled_mqloss(
        series_pl, mq_models, quantiles=quantiles, seasonality=1, train_df=series_pl
    ),
    models,
)
### Coverage
pd_vs_pl(
    coverage(series, models, 80),
    coverage(series_pl, models, 80),
    models,
)
### Calibration
show_doc(calibration, title_level=4)
pd_vs_pl(
    calibration(series, q_models[0.1]),
    calibration(series_pl, q_models[0.1]),
    models,
)
### CRPS

# $$
# \mathrm{sCRPS}(\hat{F}_{\tau}, \mathbf{y}_{\tau}) = \frac{2}{N} \sum_{i}
# \int^{1}_{0} \frac{\mathrm{QL}(\hat{F}_{i,\tau}, y_{i,\tau})_{q}}{\sum_{i} | y_{i,\tau} |} dq
# $$

# Where $\hat{F}_{\tau}$ is the an estimated multivariate distribution, and $y_{i,\tau}$ are its realizations.

pd_vs_pl(
    scaled_crps(series, mq_models, quantiles),
    scaled_crps(series_pl, mq_models, quantiles),
    models,
)
