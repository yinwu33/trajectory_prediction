from __future__ import annotations

from typing import Any

import pandas as pd

from datasets import CANONICAL_MAP_TYPES

from .base import BaseAttrAnalysis
from .utils import (
    analysis_area_m2,
    build_row_prefix,
    collect_logical_map_counts,
    density_per_km2,
    describe_numeric,
)


class MapElementsAnalysis(BaseAttrAnalysis):
    name = "map_elements"

    def collect(self, scenario: Any) -> list[dict[str, Any]]:
        counts = collect_logical_map_counts(scenario)
        area_m2 = analysis_area_m2()
        row = build_row_prefix(scenario)

        total_count = int(sum(counts.values()))
        row["map_element_count_total"] = total_count
        row["map_element_density_total_per_km2"] = density_per_km2(total_count, area_m2)

        for feature_type in CANONICAL_MAP_TYPES:
            count = int(counts.get(feature_type, 0))
            row[f"{feature_type}_count"] = count
            row[f"{feature_type}_density_per_km2"] = density_per_km2(count, area_m2)
        return [row]

    def summarize(
        self,
        results: pd.DataFrame,
        *,
        dataset_name: str,
        split: str,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_rows": int(len(results)),
            "num_scenarios": (
                int(results["scenario_id"].nunique())
                if "scenario_id" in results.columns
                else 0
            ),
            "analysis_area_m2": float(analysis_area_m2()),
            "metrics": {},
        }
        metric_columns = [
            "map_element_count_total",
            "map_element_density_total_per_km2",
        ]
        for feature_type in CANONICAL_MAP_TYPES:
            metric_columns.append(f"{feature_type}_count")
            metric_columns.append(f"{feature_type}_density_per_km2")

        for column in metric_columns:
            summary["metrics"][column] = (
                describe_numeric(results[column])
                if column in results.columns
                else {"count": 0}
            )
        return summary
