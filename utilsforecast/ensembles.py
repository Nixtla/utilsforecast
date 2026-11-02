"""Forecast ensemble utilities."""

__all__ = [
    "add_mean_ensemble",
    "apply_ensemble",
    "fit_ensemble",
    "fit_greedy_ensemble",
]

import inspect
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .compat import DataFrame, pl_DataFrame
from .validation import validate_format


def _to_pandas(df: DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy(deep=False)
    return df.to_pandas()


def _to_original_type(df: pd.DataFrame, original: DataFrame) -> DataFrame:
    if isinstance(original, pd.DataFrame):
        return df
    return pl_DataFrame(df)

def _get_model_cols(
    df: pd.DataFrame,
    models: Optional[List[str]],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
) -> List[str]:
    if models is None:
        reserved = {id_col, time_col, target_col, cutoff_col}
        models = [c for c in df.columns if c not in reserved]
    missing = [m for m in models if m not in df.columns]
    if missing:
        raise ValueError(f"The following model columns are missing: {missing}")
    non_numeric = [m for m in models if not pd.api.types.is_numeric_dtype(df[m])]
    if non_numeric:
        raise ValueError(
            f"The following model columns are not numeric: {non_numeric}"
        )
    if not models:
        raise ValueError("No model columns were found.")
    return models


def _prepare_metric_kwargs(
    metric: Callable,
    df: pd.DataFrame,
    models: List[str],
    id_col: str,
    target_col: str,
    train_df: Optional[DataFrame],
    metric_kwargs: Optional[Dict],
) -> Dict:
    sig = inspect.signature(metric).parameters
    metric_name = getattr(metric, "__name__", type(metric).__name__)
    kwargs = {}
    if "df" in sig:
        kwargs["df"] = df
    if "models" in sig:
        kwargs["models"] = models
    if "id_col" in sig:
        kwargs["id_col"] = id_col
    if "target_col" in sig:
        kwargs["target_col"] = target_col
    if "train_df" in sig:
        if train_df is None:
            raise ValueError(
                f"The metric `{metric_name}` requires `train_df` but it wasn't provided."
            )
        kwargs["train_df"] = train_df
    if metric_kwargs is not None:
        kwargs.update(metric_kwargs)
    return kwargs


def _validate_greedy_schema(
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
) -> None:
    expected = {
        "id_col": "unique_id",
        "time_col": "ds",
        "target_col": "y",
        "cutoff_col": "cutoff",
    }
    received = {
        "id_col": id_col,
        "time_col": time_col,
        "target_col": target_col,
        "cutoff_col": cutoff_col,
    }
    mismatches = [
        f"{name}='{value}'"
        for name, value in received.items()
        if value != expected[name]
    ]
    if mismatches:
        mismatch_str = ", ".join(mismatches)
        raise ValueError(
            "Greedy ensemble fitting currently requires the standard Nixtla schema "
            "column names: id_col='unique_id', time_col='ds', target_col='y', "
            "cutoff_col='cutoff'. "
            f"Received {mismatch_str}."
        )


def _score_candidate(
    df: pd.DataFrame,
    preds: np.ndarray,
    metric: Callable,
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
    train_df: Optional[DataFrame],
    metric_kwargs: Optional[Dict],
) -> pd.DataFrame:
    metric_df = df[[id_col, time_col, target_col]].copy()
    if cutoff_col in df.columns:
        metric_df[cutoff_col] = df[cutoff_col].to_numpy()
    metric_df["__ensemble__"] = preds
    result = metric(
        **_prepare_metric_kwargs(
            metric=metric,
            df=metric_df,
            models=["__ensemble__"],
            id_col=id_col,
            target_col=target_col,
            train_df=train_df,
            metric_kwargs=metric_kwargs,
        )
    )
    if "__ensemble__" not in result.columns or id_col not in result.columns:
        raise ValueError(
            "The metric must return a dataframe with the id column and the scored model column."
        )
    if cutoff_col in result.columns:
        # CV metrics return one row per (cutoff, id); greedy selection needs
        # one score per series, so average the per-window scores first.
        result = (
            result[[id_col, "__ensemble__"]]
            if result.shape[0] <= 1
            else result.groupby(id_col, as_index=False)["__ensemble__"].mean()
        )
    if result[id_col].duplicated().any():
        raise ValueError(
            "The metric must return a single score per id when fitting greedy ensembles."
        )
    return result[[id_col, "__ensemble__"]]


def fit_ensemble(
    df: DataFrame,
    method: Literal["mean", "greedy", "greedy_global", "greedy_local"] = "mean",
    models: Optional[List[str]] = None,
    metric: Optional[Callable] = None,
    max_iters: int = 10,
    kind: Literal["global", "local"] = "global",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    train_df: Optional[DataFrame] = None,
    metric_kwargs: Optional[Dict] = None,
) -> DataFrame:
    """Fit ensemble weights from forecast dataframes.

    Args:
        df: Dataframe containing the target and model predictions.
        method: Ensemble strategy to fit. `mean` returns equal weights.
            `greedy_global` and `greedy_local` run greedy selection with replacement.
            `greedy` uses the provided `kind`.
        models: Model columns to consider. By default all point forecast columns
            are used.
        metric: Metric callable used by greedy methods.
        max_iters: Number of greedy selection steps.
        kind: Greedy fitting scope used only when `method='greedy'`.
        id_col: Series identifier column.
        time_col: Time column.
        target_col: Target column.
        cutoff_col: Cross-validation cutoff column.
        train_df: Optional training dataframe required by metrics such as `mase`.
        metric_kwargs: Extra keyword arguments forwarded to the metric.

    Returns:
        Dataframe containing the fitted ensemble weights.
    """
    original = df
    pdf = _to_pandas(df)
    target_for_validation = target_col if method != "mean" else None
    validate_format(
        pdf,
        id_col=id_col,
        time_col=time_col,
        target_col=target_for_validation,
    )
    model_cols = _get_model_cols(
        pdf,
        models=models,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    if method == "mean":
        weights = pd.DataFrame([{model: 1.0 for model in model_cols}])
        return _to_original_type(weights, original)
    if method == "greedy":
        greedy_kind = kind
    elif method == "greedy_global":
        greedy_kind = "global"
    elif method == "greedy_local":
        greedy_kind = "local"
    else:
        raise ValueError(
            "`method` must be one of "
            "`'mean'`, `'greedy'`, `'greedy_global'`, `'greedy_local'`."
        )
    if metric is None:
        raise ValueError("Greedy ensemble fitting requires providing `metric`.")
    _validate_greedy_schema(
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return fit_greedy_ensemble(
        df=df,
        metric=metric,
        models=model_cols,
        kind=greedy_kind,
        max_iters=max_iters,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        train_df=train_df,
        metric_kwargs=metric_kwargs,
    )


def fit_greedy_ensemble(
    df: DataFrame,
    metric: Callable,
    models: Optional[List[str]] = None,
    kind: Literal["global", "local"] = "global",
    max_iters: int = 10,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    train_df: Optional[DataFrame] = None,
    metric_kwargs: Optional[Dict] = None,
) -> DataFrame:
    """Fit a greedy ensemble using out-of-fold predictions.

    The algorithm follows a forward-selection scheme with replacement. At each
    iteration it adds the model that most improves the chosen metric.

    Args:
        df: Dataframe containing the target and model predictions.
        metric: Metric callable from `utilsforecast.losses` or a compatible function.
        models: Model columns to consider. By default all point forecast columns
            are used.
        kind: Whether to fit a single global ensemble or one per series.
        max_iters: Number of greedy selection steps.
        id_col: Series identifier column.
        time_col: Time column.
        target_col: Target column.
        cutoff_col: Cross-validation cutoff column. Ignored by the metric fit but
            excluded from inferred model columns.
        train_df: Optional training dataframe required by metrics such as `mase`.
        metric_kwargs: Extra keyword arguments forwarded to the metric.

    Returns:
        Dataframe containing one row of normalized weights for `kind='global'`,
        or one row per series for `kind='local'`.
    """
    if max_iters < 1:
        raise ValueError("`max_iters` must be greater than or equal to one.")
    if kind not in {"global", "local"}:
        raise ValueError("`kind` must be either 'global' or 'local'.")
    _validate_greedy_schema(
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    original = df
    pdf = _to_pandas(df)
    validate_format(pdf, id_col=id_col, time_col=time_col, target_col=target_col)
    model_cols = _get_model_cols(
        pdf,
        models=models,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    base_cols = [c for c in [id_col, time_col, target_col, cutoff_col] if c in pdf.columns]
    pdf = pdf[base_cols + model_cols].copy()
    values = pdf[model_cols].to_numpy()

    if kind == "global":
        counts = np.zeros(len(model_cols), dtype=np.int64)
        current_sum = np.zeros(pdf.shape[0], dtype=np.float64)
        for n_selected in range(max_iters):
            best_score = None
            best_idx = None
            denom = n_selected + 1
            for model_idx, model in enumerate(model_cols):
                preds = (current_sum + values[:, model_idx]) / denom
                score_df = _score_candidate(
                    pdf,
                    preds=preds,
                    metric=metric,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    cutoff_col=cutoff_col,
                    train_df=train_df,
                    metric_kwargs=metric_kwargs,
                )
                score = float(score_df["__ensemble__"].mean())
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = model_idx
            assert best_idx is not None
            counts[best_idx] += 1
            current_sum += values[:, best_idx]
        weights = pd.DataFrame([counts / counts.sum()], columns=model_cols)
        return _to_original_type(weights, original)

    ids = pd.Index(pdf[id_col].drop_duplicates())
    id_to_pos = {uid: i for i, uid in enumerate(ids)}
    row_id_pos = pdf[id_col].map(id_to_pos).to_numpy()
    counts = np.zeros((len(ids), len(model_cols)), dtype=np.int64)
    current_sum = np.zeros(pdf.shape[0], dtype=np.float64)

    for n_selected in range(max_iters):
        denom = n_selected + 1
        score_matrix = np.empty((len(ids), len(model_cols)), dtype=np.float64)
        for model_idx, model in enumerate(model_cols):
            preds = (current_sum + values[:, model_idx]) / denom
            score_df = _score_candidate(
                pdf,
                preds=preds,
                metric=metric,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                cutoff_col=cutoff_col,
                train_df=train_df,
                metric_kwargs=metric_kwargs,
            )
            score_matrix[:, model_idx] = (
                score_df.set_index(id_col)["__ensemble__"].reindex(ids).to_numpy()
            )
        best_models = score_matrix.argmin(axis=1)
        counts[np.arange(len(ids)), best_models] += 1
        current_sum += values[np.arange(values.shape[0]), best_models[row_id_pos]]

    weights = pd.DataFrame(
        counts / counts.sum(axis=1, keepdims=True), columns=model_cols
    )
    weights.insert(0, id_col, ids.to_numpy())
    return _to_original_type(weights, original)


def apply_ensemble(
    df: DataFrame,
    weights: DataFrame,
    models: Optional[List[str]] = None,
    ensemble_name: str = "Ensemble",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DataFrame:
    """Apply ensemble weights to a forecast dataframe."""
    _ = target_col, cutoff_col
    original = df
    pdf = _to_pandas(df)
    validate_format(pdf, id_col=id_col, time_col=time_col, target_col=None)
    weights_df = _to_pandas(weights)
    if id_col in weights_df.columns:
        available_models = [c for c in weights_df.columns if c != id_col]
    else:
        available_models = list(weights_df.columns)
    if models is None:
        models = [m for m in available_models if m in pdf.columns]
    missing_models = [
        m for m in models if m not in available_models or m not in pdf.columns
    ]
    if missing_models:
        raise ValueError(
            f"The following models are missing from the forecasts or weights: {missing_models}"
        )
    if not models:
        raise ValueError("No model columns were found to ensemble.")
    if id_col in weights_df.columns:
        if weights_df[id_col].duplicated().any():
            raise ValueError("Local ensemble weights must contain a single row per id.")
        weights_df = weights_df[[id_col] + models].copy()
        weight_cols = {m: f"__weight_{m}" for m in models}
        weights_df = weights_df.rename(columns=weight_cols)
        merged = pdf[[id_col] + models].merge(weights_df, on=id_col, how="left")
        missing_ids = merged[[weight_cols[m] for m in models]].isna().all(axis=1)
        if missing_ids.any():
            missing = pdf.loc[missing_ids, id_col].drop_duplicates().tolist()
            raise ValueError(f"Missing local ensemble weights for ids: {missing}")
        weight_values = merged[[weight_cols[m] for m in models]].to_numpy()
    else:
        if weights_df.shape[0] != 1:
            raise ValueError(
                "Global ensemble weights must contain a single row without the id column."
            )
        weight_values = np.repeat(weights_df[models].to_numpy(), pdf.shape[0], axis=0)
    weight_sums = weight_values.sum(axis=1, keepdims=True)
    if np.any(np.isclose(weight_sums, 0.0)):
        raise ValueError("Ensemble weights must sum to a non-zero value.")
    preds = (pdf[models].to_numpy() * weight_values / weight_sums).sum(axis=1)
    out = pdf.copy()
    out[ensemble_name] = preds
    return _to_original_type(out, original)


def add_mean_ensemble(
    df: DataFrame,
    models: Optional[List[str]] = None,
    ensemble_name: str = "MeanEnsemble",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> DataFrame:
    """Add a simple equally-weighted mean ensemble."""
    weights = fit_ensemble(
        df=df,
        method="mean",
        models=models,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
    return apply_ensemble(
        df=df,
        weights=weights,
        models=models,
        ensemble_name=ensemble_name,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
    )
