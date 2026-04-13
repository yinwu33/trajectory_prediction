from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseAttrAnalysis
from .utils import (
    analysis_area_m2,
    build_row_prefix,
    density_per_km2,
    describe_numeric,
    standardized_vehicle_arrays,
)


class VehicleDensityAnalysis(BaseAttrAnalysis):
    name = "vehicle_density"

    def collect(self, scenario: Any) -> list[dict[str, Any]]:
        agent_arrays, vehicle_indices = standardized_vehicle_arrays(scenario)
        prefix = build_row_prefix(scenario)
        area_m2 = analysis_area_m2()

        if vehicle_indices.size == 0:
            num_steps = int(np.asarray(scenario.timestamps_seconds).shape[0])
            return [
                {
                    **prefix,
                    "timestep": int(timestep),
                    "vehicle_count": 0,
                    "vehicle_density_per_km2": density_per_km2(0, area_m2),
                }
                for timestep in range(num_steps)
            ]

        vehicle_valid = np.asarray(
            agent_arrays["agent_valid_mask"][vehicle_indices],
            dtype=bool,
        )
        counts = vehicle_valid.sum(axis=0)
        rows: list[dict[str, Any]] = []
        for timestep, count in enumerate(counts):
            rows.append(
                {
                    **prefix,
                    "timestep": int(timestep),
                    "vehicle_count": int(count),
                    "vehicle_density_per_km2": density_per_km2(int(count), area_m2),
                }
            )
        return rows

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
        for column in ("vehicle_count", "vehicle_density_per_km2"):
            summary["metrics"][column] = (
                describe_numeric(results[column])
                if column in results.columns
                else {"count": 0}
            )
        return summary
