from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseAttrAnalysis
from .utils import (
    build_row_prefix,
    compute_acceleration_mps2,
    compute_speed_mps,
    describe_numeric,
    infer_dt_seconds,
    standardized_vehicle_arrays,
)


class VehicleMotionAnalysis(BaseAttrAnalysis):
    name = "vehicle_motion"

    def collect(self, scenario: Any) -> list[dict[str, Any]]:
        agent_arrays, vehicle_indices = standardized_vehicle_arrays(scenario)
        dt = infer_dt_seconds(scenario)
        rows: list[dict[str, Any]] = []
        prefix = build_row_prefix(scenario)

        for agent_index in vehicle_indices:
            track_id = str(agent_arrays["agent_ids"][agent_index])
            velocities = np.asarray(
                agent_arrays["agent_velocities"][agent_index],
                dtype=np.float32,
            )
            valid_mask = np.asarray(
                agent_arrays["agent_valid_mask"][agent_index],
                dtype=bool,
            )
            speeds = compute_speed_mps(velocities)
            accelerations = compute_acceleration_mps2(
                velocities=velocities,
                valid_mask=valid_mask,
                dt=dt,
            )

            for timestep in np.flatnonzero(valid_mask):
                rows.append(
                    {
                        **prefix,
                        "track_id": track_id,
                        "timestep": int(timestep),
                        "speed_mps": float(speeds[timestep]),
                        "acceleration_mps2": float(accelerations[timestep]),
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
            "num_tracks": (
                int(results["track_id"].nunique())
                if "track_id" in results.columns
                else 0
            ),
            "metrics": {},
        }
        for column in ("speed_mps", "acceleration_mps2"):
            summary["metrics"][column] = (
                describe_numeric(results[column])
                if column in results.columns
                else {"count": 0}
            )
        return summary
