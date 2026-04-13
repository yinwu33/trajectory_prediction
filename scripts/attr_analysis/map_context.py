from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point

from datasets.motion_dataset import MotionScenario
from datasets.standardization import (
    CANONICAL_MAP_TYPES,
    StandardizationConfig,
    get_standardized_map_arrays,
)

from .base import BaseAttrAnalysis
from .context import AnalysisContext, track_result_row
from .utils import describe_boolean, describe_numeric, finite_rows, masked_array


def build_map_geometries(sample: MotionScenario) -> dict[str, Any]:
    map_arrays = get_standardized_map_arrays(sample)
    lane_lines: list[LineString] = []
    intersection_lane_lines: list[LineString] = []
    crosswalk_lines: list[LineString] = []

    lane_ids: set[str] = set()
    intersection_lane_ids: set[str] = set()
    crosswalk_ids: set[str] = set()

    for polyline_index in range(map_arrays["map_points"].shape[0]):
        valid_mask = map_arrays["map_valid_mask"][polyline_index]
        if valid_mask.sum() < 2:
            continue

        points = map_arrays["map_points"][polyline_index, valid_mask]
        line = LineString(points)
        if line.is_empty:
            continue

        feature_type = CANONICAL_MAP_TYPES[int(map_arrays["map_types"][polyline_index])]
        base_id = map_arrays["map_ids"][polyline_index].split("#seg", 1)[0]

        if feature_type == "lane_centerline":
            lane_lines.append(line)
            lane_ids.add(base_id)
            if bool(map_arrays["map_is_intersection"][polyline_index]):
                intersection_lane_lines.append(line)
                intersection_lane_ids.add(base_id)
        elif feature_type == "crosswalk":
            crosswalk_lines.append(line)
            crosswalk_ids.add(base_id)

    lane_geometry = MultiLineString(lane_lines) if lane_lines else GeometryCollection()
    intersection_geometry = (
        MultiLineString(intersection_lane_lines)
        if intersection_lane_lines
        else GeometryCollection()
    )
    crosswalk_geometry = (
        MultiLineString(crosswalk_lines) if crosswalk_lines else GeometryCollection()
    )

    return {
        "lane_geometry": lane_geometry,
        "intersection_geometry": intersection_geometry,
        "crosswalk_geometry": crosswalk_geometry,
        "num_lane_segments": len(lane_ids),
        "num_intersection_lane_segments": len(intersection_lane_ids),
        "num_crosswalks": len(crosswalk_ids),
        "num_drivable_areas": 0,
    }


def min_track_distance_to_geometry(positions: np.ndarray, geometry: Any) -> float:
    if geometry.is_empty:
        return float("nan")
    valid_positions = positions[finite_rows(positions)]
    if valid_positions.size == 0:
        return float("nan")
    return float(min(Point(x, y).distance(geometry) for x, y in valid_positions))


class MapContextAnalysis(BaseAttrAnalysis):
    name = "map_context"

    def __init__(
        self,
        *,
        intersection_near_threshold_m: float = 20,
        crosswalk_near_threshold_m: float = 10,
    ) -> None:
        self.intersection_near_threshold_m = float(intersection_near_threshold_m)
        self.crosswalk_near_threshold_m = float(crosswalk_near_threshold_m)

    def analyze(self, context: AnalysisContext) -> dict[str, Any]:
        ego_positions = masked_array(context.ego_positions, context.ego_valid)
        focus_positions = masked_array(context.focus_positions, context.focus_valid)
        map_geometries = build_map_geometries(context.sample)

        ego_lane_distance = min_track_distance_to_geometry(
            ego_positions, map_geometries["lane_geometry"]
        )
        focus_lane_distance = min_track_distance_to_geometry(
            focus_positions, map_geometries["lane_geometry"]
        )
        ego_intersection_distance = min_track_distance_to_geometry(
            ego_positions, map_geometries["intersection_geometry"]
        )
        focus_intersection_distance = min_track_distance_to_geometry(
            focus_positions, map_geometries["intersection_geometry"]
        )
        ego_crosswalk_distance = min_track_distance_to_geometry(
            ego_positions, map_geometries["crosswalk_geometry"]
        )
        focus_crosswalk_distance = min_track_distance_to_geometry(
            focus_positions, map_geometries["crosswalk_geometry"]
        )

        row = track_result_row(context)
        row.update(
            {
                "num_lane_segments": map_geometries["num_lane_segments"],
                "num_intersection_lane_segments": map_geometries[
                    "num_intersection_lane_segments"
                ],
                "num_crosswalks": map_geometries["num_crosswalks"],
                "num_drivable_areas": map_geometries["num_drivable_areas"],
                "ego_nearest_lane_centerline_distance_m": ego_lane_distance,
                "focus_nearest_lane_centerline_distance_m": focus_lane_distance,
                "ego_nearest_intersection_lane_distance_m": ego_intersection_distance,
                "focus_nearest_intersection_lane_distance_m": focus_intersection_distance,
                "ego_nearest_crosswalk_distance_m": ego_crosswalk_distance,
                "focus_nearest_crosswalk_distance_m": focus_crosswalk_distance,
                "ego_intersection_near": bool(
                    np.isfinite(ego_intersection_distance)
                    and ego_intersection_distance <= self.intersection_near_threshold_m
                ),
                "focus_intersection_near": bool(
                    np.isfinite(focus_intersection_distance)
                    and focus_intersection_distance
                    <= self.intersection_near_threshold_m
                ),
                "ego_crosswalk_near": bool(
                    np.isfinite(ego_crosswalk_distance)
                    and ego_crosswalk_distance <= self.crosswalk_near_threshold_m
                ),
                "focus_crosswalk_near": bool(
                    np.isfinite(focus_crosswalk_distance)
                    and focus_crosswalk_distance <= self.crosswalk_near_threshold_m
                ),
            }
        )
        return row

    def build_summary(
        self,
        results: pd.DataFrame,
        *,
        dataset_name: str,
        split: str,
        config: StandardizationConfig,
    ) -> dict[str, Any]:
        numeric_columns = [
            "num_lane_segments",
            "num_intersection_lane_segments",
            "num_crosswalks",
            "num_drivable_areas",
            "ego_nearest_lane_centerline_distance_m",
            "focus_nearest_lane_centerline_distance_m",
            "ego_nearest_intersection_lane_distance_m",
            "focus_nearest_intersection_lane_distance_m",
            "ego_nearest_crosswalk_distance_m",
            "focus_nearest_crosswalk_distance_m",
        ]
        bool_columns = [
            "ego_intersection_near",
            "focus_intersection_near",
            "ego_crosswalk_near",
            "focus_crosswalk_near",
        ]
        summary: dict[str, Any] = {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_scenarios": int(len(results)),
            "intersection_near_threshold_m": self.intersection_near_threshold_m,
            "crosswalk_near_threshold_m": self.crosswalk_near_threshold_m,
            "numeric_metrics": {},
            "boolean_metrics": {},
        }
        for column in numeric_columns:
            if column in results.columns:
                summary["numeric_metrics"][column] = describe_numeric(results[column])
        for column in bool_columns:
            if column in results.columns:
                summary["boolean_metrics"][column] = describe_boolean(results[column])
        return summary
