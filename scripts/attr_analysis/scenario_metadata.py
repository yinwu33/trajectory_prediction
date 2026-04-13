from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from datasets.standardization import (
    StandardizationConfig,
    get_anchor_track_id,
    get_coord_frame,
    get_future_steps,
    get_history_steps,
    get_primary_target_track_id,
)

from .base import BaseAttrAnalysis
from .context import AnalysisContext, track_result_row
from .utils import describe_numeric, value_counts_dict


class ScenarioMetadataAnalysis(BaseAttrAnalysis):
    name = "scenario_metadata"

    def analyze(self, context: AnalysisContext) -> dict[str, Any]:
        sample = context.sample
        row = track_result_row(context)
        row.update(
            {
                "focal_track_id": get_primary_target_track_id(sample),
                "coord_frame": get_coord_frame(sample),
                "history_steps": get_history_steps(sample),
                "future_steps": get_future_steps(sample),
                "anchor_track_id": get_anchor_track_id(sample),
                "primary_target_track_id": get_primary_target_track_id(sample),
                "selected_agent_count": int(sample.metadata["selected_agent_count"]),
                "selected_polyline_count": int(sample.metadata["selected_polyline_count"]),
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
        summary: dict[str, Any] = {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_scenarios": int(len(results)),
            "standardization_config": asdict(config),
            "dataset_counts": value_counts_dict(results["dataset"], dropna=False) if "dataset" in results.columns else {},
            "city_counts": value_counts_dict(results["city"]) if "city" in results.columns else {},
            "ego_type_counts": value_counts_dict(results["ego_type"], dropna=False) if "ego_type" in results.columns else {},
            "focus_type_counts": value_counts_dict(results["focus_type"], dropna=False) if "focus_type" in results.columns else {},
            "numeric_metrics": {},
        }

        numeric_columns = [
            "history_steps",
            "future_steps",
            "selected_agent_count",
            "selected_polyline_count",
        ]
        for column in numeric_columns:
            if column in results.columns:
                summary["numeric_metrics"][column] = describe_numeric(results[column])
        return summary
