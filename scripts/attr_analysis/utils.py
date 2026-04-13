from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from datasets import (
    CANONICAL_MAP_TYPES,
    StandardConfig,
    get_standardized_agent_arrays,
    get_standardized_map_arrays,
)


_VEHICLE_TYPE_INDEX = 0
_SEGMENT_SUFFIX_RE = re.compile(r"#seg\d+$")


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def describe_numeric(series: pd.Series) -> dict[str, float | int]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def build_row_prefix(scenario: Any) -> dict[str, Any]:
    return {
        "dataset": str(scenario.source),
        "split": str(scenario.split),
        "scenario_id": str(scenario.scenario_id),
    }


def standardized_vehicle_arrays(scenario: Any) -> tuple[dict[str, Any], np.ndarray]:
    agent_arrays = get_standardized_agent_arrays(scenario)
    vehicle_mask = agent_arrays["agent_types"] == _VEHICLE_TYPE_INDEX
    vehicle_indices = np.flatnonzero(vehicle_mask)
    return agent_arrays, vehicle_indices


def infer_dt_seconds(scenario: Any) -> float:
    timestamps = np.asarray(scenario.timestamps_seconds, dtype=np.float32)
    if timestamps.size >= 2:
        deltas = np.diff(timestamps)
        finite = deltas[np.isfinite(deltas)]
        if finite.size > 0:
            return float(finite[0])
    return float(StandardConfig.dt)


def compute_speed_mps(velocities: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocities, axis=-1).astype(np.float32)


def compute_acceleration_mps2(
    velocities: np.ndarray,
    valid_mask: np.ndarray,
    dt: float,
) -> np.ndarray:
    accelerations = np.full((velocities.shape[0],), np.nan, dtype=np.float32)
    if dt <= 0.0:
        return accelerations
    for timestep in range(1, velocities.shape[0]):
        if bool(valid_mask[timestep]) and bool(valid_mask[timestep - 1]):
            delta = (velocities[timestep] - velocities[timestep - 1]) / dt
            accelerations[timestep] = float(np.linalg.norm(delta))
    return accelerations


def analysis_area_m2() -> float:
    range_m = float(StandardConfig.map.range_m)
    if StandardConfig.map.crop_shape == "circle":
        return float(math.pi * range_m * range_m)
    return float((2.0 * range_m) * (2.0 * range_m))


def density_per_km2(count: int | float, area_m2: float | None = None) -> float:
    if area_m2 is None:
        area_m2 = analysis_area_m2()
    if area_m2 <= 0.0:
        return float("nan")
    return float(float(count) * 1_000_000.0 / area_m2)


def normalize_map_feature_id(feature_id: str) -> str:
    return _SEGMENT_SUFFIX_RE.sub("", feature_id)


def collect_logical_map_counts(scenario: Any) -> dict[str, int]:
    map_arrays = get_standardized_map_arrays(scenario)
    logical_features: set[tuple[str, str]] = set()
    for feature_id, type_index in zip(map_arrays["map_ids"], map_arrays["map_types"]):
        feature_type = CANONICAL_MAP_TYPES[int(type_index)]
        logical_features.add((feature_type, normalize_map_feature_id(str(feature_id))))

    counts = {feature_type: 0 for feature_type in CANONICAL_MAP_TYPES}
    for feature_type, _ in logical_features:
        counts[feature_type] += 1
    return counts
