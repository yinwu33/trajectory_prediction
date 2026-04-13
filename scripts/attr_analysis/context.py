from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datasets.motion_dataset import MotionScenario
from datasets.standardization import (
    CANONICAL_AGENT_TYPES,
    get_history_steps,
    get_primary_target_index,
    get_standardized_agent_arrays,
)


@dataclass(frozen=True)
class AnalysisContext:
    sample: MotionScenario
    ego_index: int
    focus_index: int
    history_slice: slice
    ego_track_id: str
    focus_track_id: str
    ego_type: str
    focus_type: str
    ego_valid: np.ndarray
    focus_valid: np.ndarray
    ego_positions: np.ndarray
    ego_velocities: np.ndarray
    ego_headings: np.ndarray
    focus_positions: np.ndarray
    focus_velocities: np.ndarray
    focus_headings: np.ndarray


def build_analysis_context(sample: MotionScenario) -> AnalysisContext:
    agent_arrays = get_standardized_agent_arrays(sample)
    ego_indices = np.flatnonzero(agent_arrays["agent_is_ego"])
    if ego_indices.size == 0:
        raise ValueError(f"Scenario {sample.scenario_id} is missing ego track after standardization.")

    primary_target_index = get_primary_target_index(sample)
    if primary_target_index < 0:
        raise ValueError(f"Scenario {sample.scenario_id} is missing primary target after standardization.")

    ego_index = int(ego_indices[0])
    focus_index = int(primary_target_index)
    history_slice = slice(0, get_history_steps(sample))

    ego_type = CANONICAL_AGENT_TYPES[int(agent_arrays["agent_types"][ego_index])]
    focus_type = CANONICAL_AGENT_TYPES[int(agent_arrays["agent_types"][focus_index])]

    return AnalysisContext(
        sample=sample,
        ego_index=ego_index,
        focus_index=focus_index,
        history_slice=history_slice,
        ego_track_id=agent_arrays["agent_ids"][ego_index],
        focus_track_id=agent_arrays["agent_ids"][focus_index],
        ego_type=ego_type,
        focus_type=focus_type,
        ego_valid=agent_arrays["agent_observed_mask"][ego_index, history_slice],
        focus_valid=agent_arrays["agent_observed_mask"][focus_index, history_slice],
        ego_positions=agent_arrays["agent_positions"][ego_index, history_slice],
        ego_velocities=agent_arrays["agent_velocities"][ego_index, history_slice],
        ego_headings=agent_arrays["agent_headings"][ego_index, history_slice],
        focus_positions=agent_arrays["agent_positions"][focus_index, history_slice],
        focus_velocities=agent_arrays["agent_velocities"][focus_index, history_slice],
        focus_headings=agent_arrays["agent_headings"][focus_index, history_slice],
    )


def base_result_row(context: AnalysisContext) -> dict[str, Any]:
    return {
        "scenario_id": context.sample.scenario_id,
        "dataset": context.sample.source,
        "city": context.sample.city_name or "unknown",
    }


def track_result_row(context: AnalysisContext) -> dict[str, Any]:
    row = base_result_row(context)
    row.update(
        {
            "ego_track_id": context.ego_track_id,
            "target_track_id": context.focus_track_id,
            "ego_type": context.ego_type,
            "target_type": context.focus_type,
            "focus_type": context.focus_type,
        }
    )
    return row
