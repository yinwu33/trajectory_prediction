from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.standardization import StandardizationConfig

from .base import BaseAttrAnalysis
from .context import AnalysisContext, track_result_row
from .utils import (
    describe_numeric,
    has_columns,
    heading_change_abs_sum,
    last_finite,
    masked_array,
    nanpercentile,
    path_length,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


def agent_motion_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    masked_positions = masked_array(positions, valid_mask)
    masked_velocities = masked_array(velocities, valid_mask)
    masked_heading = masked_array(heading, valid_mask)

    speed = np.linalg.norm(masked_velocities, axis=1)
    return {
        f"{prefix}_observed_ratio": float(valid_mask.mean()),
        f"{prefix}_speed_source": "standardized",
        f"{prefix}_speed_min_mps": float(np.nanmin(speed)) if np.isfinite(speed).any() else float("nan"),
        f"{prefix}_speed_mean_mps": float(np.nanmean(speed)) if np.isfinite(speed).any() else float("nan"),
        f"{prefix}_speed_p95_mps": nanpercentile(speed, 95),
        f"{prefix}_speed_max_mps": float(np.nanmax(speed)) if np.isfinite(speed).any() else float("nan"),
        f"{prefix}_speed_last_mps": last_finite(speed),
        f"{prefix}_path_length_obs_m": path_length(masked_positions, valid_mask),
        f"{prefix}_heading_change_abs_sum_rad": heading_change_abs_sum(masked_heading, valid_mask),
        f"{prefix}_num_observed_steps": int(valid_mask.sum()),
        f"{prefix}_last_x": last_finite(masked_positions[:, 0]),
        f"{prefix}_last_y": last_finite(masked_positions[:, 1]),
    }


class AgentMotionAnalysis(BaseAttrAnalysis):
    name = "agent_motion"

    def analyze(self, context: AnalysisContext) -> dict[str, Any]:
        row = track_result_row(context)
        row.update(
            agent_motion_metrics(
                context.ego_positions,
                context.ego_velocities,
                context.ego_headings,
                context.ego_valid,
                "ego",
            )
        )
        row.update(
            agent_motion_metrics(
                context.focus_positions,
                context.focus_velocities,
                context.focus_headings,
                context.focus_valid,
                "focus",
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
            "ego_observed_ratio",
            "ego_speed_min_mps",
            "ego_speed_mean_mps",
            "ego_speed_p95_mps",
            "ego_speed_max_mps",
            "ego_speed_last_mps",
            "ego_path_length_obs_m",
            "ego_heading_change_abs_sum_rad",
            "ego_num_observed_steps",
            "focus_observed_ratio",
            "focus_speed_min_mps",
            "focus_speed_mean_mps",
            "focus_speed_p95_mps",
            "focus_speed_max_mps",
            "focus_speed_last_mps",
            "focus_path_length_obs_m",
            "focus_heading_change_abs_sum_rad",
            "focus_num_observed_steps",
        ]
        summary: dict[str, Any] = {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_scenarios": int(len(results)),
            "numeric_metrics": {},
        }
        for column in numeric_columns:
            if column in results.columns:
                summary["numeric_metrics"][column] = describe_numeric(results[column])
        if "ego_speed_source" in results.columns:
            summary["ego_speed_source_counts"] = {
                str(key): int(value) for key, value in results["ego_speed_source"].value_counts(dropna=False).items()
            }
        if "focus_speed_source" in results.columns:
            summary["focus_speed_source_counts"] = {
                str(key): int(value) for key, value in results["focus_speed_source"].value_counts(dropna=False).items()
            }
        return summary

    def save_plots(self, results: pd.DataFrame, *, output_dir: Path, file_prefix: str) -> list[Path]:
        if plt is None:
            print(f"matplotlib not available, skipping plots for {self.name}.")
            return []

        saved_paths: list[Path] = []
        plot_specs = [
            ("ego_speed_p95_mps", "Ego speed p95 (m/s)", f"{file_prefix}_hist_ego_speed_p95_mps.png"),
            ("focus_speed_p95_mps", "Focus speed p95 (m/s)", f"{file_prefix}_hist_focus_speed_p95_mps.png"),
        ]

        for column, title, filename in plot_specs:
            if column not in results.columns:
                continue
            values = results[column].dropna()
            if values.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(values, bins=40)
            ax.set_title(title)
            ax.set_xlabel(column)
            ax.set_ylabel("count")
            fig.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved_paths.append(path)

        scatter_columns = ["ego_speed_p95_mps", "focus_speed_p95_mps"]
        if has_columns(results, scatter_columns):
            scatter_df = results[scatter_columns].dropna()
            if not scatter_df.empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(
                    scatter_df["ego_speed_p95_mps"],
                    scatter_df["focus_speed_p95_mps"],
                    s=8,
                    alpha=0.4,
                )
                ax.set_xlabel("ego_speed_p95_mps")
                ax.set_ylabel("focus_speed_p95_mps")
                ax.set_title("Ego vs focus speed p95")
                fig.tight_layout()
                path = output_dir / f"{file_prefix}_scatter_ego_vs_focus_speed_p95_mps.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                saved_paths.append(path)
        return saved_paths
