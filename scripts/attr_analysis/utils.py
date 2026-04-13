from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def finite_rows(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=1)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def masked_array(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    masked = np.full(values.shape, np.nan, dtype=np.float32)
    masked[valid_mask] = values[valid_mask]
    return masked


def path_length(positions: np.ndarray, valid_mask: np.ndarray) -> float:
    length = 0.0
    for idx in range(1, len(positions)):
        if valid_mask[idx] and valid_mask[idx - 1]:
            length += float(np.linalg.norm(positions[idx] - positions[idx - 1]))
    return length


def heading_change_abs_sum(heading: np.ndarray, valid_mask: np.ndarray) -> float:
    finite = valid_mask & np.isfinite(heading)
    if finite.sum() < 2:
        return float("nan")
    diffs = wrap_angle(np.diff(heading[finite]))
    return float(np.abs(diffs).sum())


def describe_numeric(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def describe_boolean(series: pd.Series) -> dict[str, float | int]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "count_true": 0, "share_true": 0.0}
    clean_bool = clean.astype(bool)
    return {
        "count": int(clean_bool.count()),
        "count_true": int(clean_bool.sum()),
        "share_true": float(clean_bool.mean()),
    }


def value_counts_dict(series: pd.Series, *, dropna: bool = False) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=dropna).items()}


def has_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in frame.columns for column in columns)


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value
