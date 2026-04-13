from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.standardization import StandardizationConfig

from .base import BaseAttrAnalysis
from .context import AnalysisContext, track_result_row
from .utils import describe_numeric, finite_rows, has_columns, masked_array

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


def compute_interaction_metrics(
    ego_positions: np.ndarray,
    ego_velocity: np.ndarray,
    focus_positions: np.ndarray,
    focus_velocity: np.ndarray,
) -> dict[str, Any]:
    valid = finite_rows(ego_positions) & finite_rows(focus_positions)
    if not valid.any():
        return {
            "distance_last_m": float("nan"),
            "min_distance_obs_m": float("nan"),
            "min_distance_timestep": float("nan"),
            "relative_speed_last_mps": float("nan"),
            "closing_speed_at_min_distance_mps": float("nan"),
            "num_overlap_steps": 0,
        }

    separation = focus_positions - ego_positions
    distance = np.linalg.norm(separation, axis=1)
    overlap_steps = np.flatnonzero(valid)
    valid_distance = distance[valid]
    min_idx_within_valid = int(np.argmin(valid_distance))
    min_timestep = int(overlap_steps[min_idx_within_valid])
    min_distance = float(valid_distance[min_idx_within_valid])

    last_overlap_timestep = int(overlap_steps[-1])
    relative_velocity = focus_velocity - ego_velocity
    relative_speed_last = (
        float(np.linalg.norm(relative_velocity[last_overlap_timestep]))
        if np.isfinite(relative_velocity[last_overlap_timestep]).all()
        else float("nan")
    )

    closing_speed = float("nan")
    rel_velocity_at_min = relative_velocity[min_timestep]
    separation_at_min = separation[min_timestep]
    if np.isfinite(rel_velocity_at_min).all() and np.isfinite(separation_at_min).all() and min_distance > 1e-6:
        line_of_sight = separation_at_min / min_distance
        closing_speed = float(-np.dot(rel_velocity_at_min, line_of_sight))

    return {
        "distance_last_m": float(distance[last_overlap_timestep]),
        "min_distance_obs_m": min_distance,
        "min_distance_timestep": min_timestep,
        "relative_speed_last_mps": relative_speed_last,
        "closing_speed_at_min_distance_mps": closing_speed,
        "num_overlap_steps": int(valid.sum()),
    }


class InteractionAnalysis(BaseAttrAnalysis):
    name = "interaction"

    def analyze(self, context: AnalysisContext) -> dict[str, Any]:
        row = track_result_row(context)
        row.update(
            compute_interaction_metrics(
                ego_positions=masked_array(context.ego_positions, context.ego_valid),
                ego_velocity=masked_array(context.ego_velocities, context.ego_valid),
                focus_positions=masked_array(context.focus_positions, context.focus_valid),
                focus_velocity=masked_array(context.focus_velocities, context.focus_valid),
            )
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
            "distance_last_m",
            "min_distance_obs_m",
            "min_distance_timestep",
            "relative_speed_last_mps",
            "closing_speed_at_min_distance_mps",
            "num_overlap_steps",
        ]
        summary: dict[str, Any] = {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_scenarios": int(len(results)),
            "numeric_metrics": {},
            "closest_scenarios": [],
        }
        for column in numeric_columns:
            if column in results.columns:
                summary["numeric_metrics"][column] = describe_numeric(results[column])

        if has_columns(results, ["scenario_id", "city", "min_distance_obs_m", "distance_last_m"]):
            closest = results.nsmallest(20, "min_distance_obs_m")[
                ["scenario_id", "city", "min_distance_obs_m", "distance_last_m"]
            ]
            summary["closest_scenarios"] = closest.to_dict(orient="records")
        return summary

    def save_plots(self, results: pd.DataFrame, *, output_dir: Path, file_prefix: str) -> list[Path]:
        if plt is None:
            print(f"matplotlib not available, skipping plots for {self.name}.")
            return []

        if "min_distance_obs_m" not in results.columns:
            return []

        values = results["min_distance_obs_m"].dropna()
        if values.empty:
            return []

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=40)
        ax.set_title("Min ego-focus distance in history (m)")
        ax.set_xlabel("min_distance_obs_m")
        ax.set_ylabel("count")
        fig.tight_layout()

        path = output_dir / f"{file_prefix}_hist_min_distance_obs_m.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return [path]
