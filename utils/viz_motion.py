from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # safe headless backend for dataloader forks
import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    CANONICAL_AGENT_TYPES,
    CANONICAL_MAP_TYPES,
    MotionScenario,
    get_primary_target_index,
    get_standardized_agent_arrays,
    get_standardized_map_arrays,
    is_standardized_scenario,
)

_COLOR_MAP = {
    "focal": ["#ff9999", "#ff0000", "#8b0000"],
    "av": ["#a1c9f4", "#1f77b4", "#084594"],
    "score": ["#ffbb78", "#ff7f0e", "#a65628"],
    "other": ["#d3d3d3", "#7f7f7f", "#555555"],
    "lane": "#e0e0e0",
}

_POINT_SIZE = 5
_LINE_WIDTH = 1.0
_FONT_SIZE = 6
_VIEW_RADIUS = 60.0

_SCORE_TYPES = ["focal", "av", "score"]
_POLYGON_MAP_TYPES = {"crosswalk", "speed_bump", "driveway", "drivable_area"}
_MAP_STYLES = {
    "lane_centerline": {"color": _COLOR_MAP["lane"], "linewidth": _LINE_WIDTH, "zorder": 0},
    "road_line": {"color": "#dddddd", "linewidth": 0.8, "zorder": 0},
    "road_edge": {"color": "#d0d0d0", "linewidth": 0.9, "zorder": 0},
    "crosswalk": {"color": "#e4e4e4", "linewidth": 0.8, "zorder": 0},
    "speed_bump": {"color": "#dddddd", "linewidth": 0.8, "zorder": 0},
    "driveway": {"color": "#e5e5e5", "linewidth": 0.8, "zorder": 0},
    "drivable_area": {"color": "#efefef", "linewidth": 0.7, "zorder": 0},
}


@dataclass
class _PreparedAgent:
    source_index: int
    track_id: str
    agent_type: str
    positions: np.ndarray
    history_mask: np.ndarray
    future_mask: np.ndarray
    current_pos: np.ndarray
    role: str


@dataclass
class _PreparedMapFeature:
    points: np.ndarray
    feature_type: str
    is_intersection: bool = False


@dataclass
class _PreparedScenario:
    scenario_id: str
    source: str
    agents: list[_PreparedAgent]
    map_features: list[_PreparedMapFeature]
    focus_agent_idx: int


def plot_scenario(
    sample: MotionScenario,
    preds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    k: int = 1,
    target_track_id: str | None = None,
    target_agent_idx: int | None = None,
    view_radius: float = _VIEW_RADIUS,
    show_agent_ids: bool = False,
):
    """Visualize a raw or standardized motion sample."""

    prepared = _prepare_sample(
        sample=sample,
        target_track_id=target_track_id,
        target_agent_idx=target_agent_idx,
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    _plot_map(ax, prepared.map_features)

    for agent_idx, agent in enumerate(prepared.agents):
        role = agent.role
        label = None
        current_only = True
        if role == "focal":
            label = "focal"
            current_only = False
        elif role == "av":
            label = "av"
            current_only = False
        elif role == "score":
            label = "scored"
            current_only = False

        _plot_agent(
            ax=ax,
            agent_positions_np=agent.positions,
            agent_history_mask_np=agent.history_mask,
            agent_future_mask_np=agent.future_mask,
            agent_current_pos_np=agent.current_pos,
            color_map=_COLOR_MAP[role],
            label=label,
            current_only=current_only,
            agent_type=agent.agent_type,
            text=agent.track_id if show_agent_ids else None,
        )

    _plot_sample_predictions(
        ax=ax,
        preds=preds,
        probs=probs,
        max_k=k,
        agents=prepared.agents,
        focus_agent_idx=prepared.focus_agent_idx,
    )

    if 0 <= prepared.focus_agent_idx < len(prepared.agents):
        center = prepared.agents[prepared.focus_agent_idx].current_pos
    else:
        center = np.zeros(2, dtype=np.float32)

    ax.set_xlim(float(center[0] - view_radius), float(center[0] + view_radius))
    ax.set_ylim(float(center[1] - view_radius), float(center[1] + view_radius))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{prepared.scenario_id} ({prepared.source})")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_motion_sample(
    sample: MotionScenario,
    preds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    k: int = 1,
    target_track_id: str | None = None,
    target_agent_idx: int | None = None,
    view_radius: float = _VIEW_RADIUS,
    show_agent_ids: bool = False,
):
    return plot_scenario(
        sample=sample,
        preds=preds,
        probs=probs,
        k=k,
        target_track_id=target_track_id,
        target_agent_idx=target_agent_idx,
        view_radius=view_radius,
        show_agent_ids=show_agent_ids,
    )


def _prepare_sample(
    sample: MotionScenario,
    target_track_id: str | None,
    target_agent_idx: int | None,
) -> _PreparedScenario:
    if not isinstance(sample, MotionScenario):
        raise TypeError(f"plot_scenario expects MotionScenario, got {type(sample)!r}")
    if is_standardized_scenario(sample):
        return _prepare_standardized_scenario(sample, target_track_id, target_agent_idx)
    return _prepare_motion_scenario(sample, target_track_id, target_agent_idx)


def _prepare_motion_scenario(
    sample: MotionScenario,
    target_track_id: str | None,
    target_agent_idx: int | None,
) -> _PreparedScenario:
    current_time_index = _resolve_motion_current_time_index(sample)
    focus_track_id = target_track_id or _resolve_motion_focus_track_id(sample)

    agents: list[_PreparedAgent] = []
    map_features = _collect_motion_map_features(sample)
    time_index = np.arange(sample.num_steps)

    for track_idx, track in enumerate(sample.tracks):
        valid_mask = np.asarray(track.valid_mask, dtype=bool)
        if not valid_mask.any():
            continue

        positions = np.asarray(track.positions, dtype=np.float32)[..., :2]
        observed_mask = np.asarray(track.observed_mask, dtype=bool) & valid_mask
        history_mask = (
            observed_mask
            if observed_mask.any()
            else valid_mask & (time_index <= current_time_index)
        )
        future_mask = valid_mask & (time_index > current_time_index)
        reference_mask = valid_mask & (time_index <= current_time_index)
        current_pos = _last_valid_position(positions, reference_mask, fallback_mask=valid_mask)
        role = _resolve_motion_agent_role(track, focus_track_id)

        agents.append(
            _PreparedAgent(
                source_index=track_idx,
                track_id=track.track_id,
                agent_type=_normalize_agent_type_name(track.object_type),
                positions=positions,
                history_mask=history_mask,
                future_mask=future_mask,
                current_pos=current_pos,
                role=role,
            )
        )

    focus_agent_idx = _resolve_focus_agent_idx_from_track_id(agents, focus_track_id)
    if target_agent_idx is not None:
        focus_agent_idx = _resolve_focus_agent_idx_from_source_index(agents, int(target_agent_idx))
    if focus_agent_idx < 0:
        focus_agent_idx = _fallback_focus_agent_idx(agents)

    _promote_focus_agent(agents, focus_agent_idx)

    return _PreparedScenario(
        scenario_id=sample.scenario_id,
        source=sample.source,
        agents=agents,
        map_features=map_features,
        focus_agent_idx=focus_agent_idx,
    )


def _prepare_standardized_scenario(
    sample: MotionScenario,
    target_track_id: str | None,
    target_agent_idx: int | None,
) -> _PreparedScenario:
    agent_arrays = get_standardized_agent_arrays(sample)
    map_arrays = get_standardized_map_arrays(sample)
    current_time_index = int(sample.current_time_index)
    total_steps = int(agent_arrays["agent_positions"].shape[1])
    time_index = np.arange(total_steps)

    agents: list[_PreparedAgent] = []
    map_features = _collect_standardized_map_features(map_arrays)

    for agent_idx in range(agent_arrays["agent_positions"].shape[0]):
        valid_mask = np.asarray(agent_arrays["agent_valid_mask"][agent_idx], dtype=bool)
        if not valid_mask.any():
            continue

        positions = np.asarray(agent_arrays["agent_positions"][agent_idx], dtype=np.float32)
        history_mask = valid_mask & (time_index <= current_time_index)
        observed_mask = np.asarray(agent_arrays["agent_observed_mask"][agent_idx], dtype=bool) & valid_mask
        future_mask = valid_mask & (time_index > current_time_index)
        reference_mask = valid_mask & (time_index <= current_time_index)
        current_pos = _last_valid_position(positions, reference_mask, fallback_mask=valid_mask)
        role = _resolve_standardized_agent_role(sample, agent_arrays, agent_idx)

        agents.append(
            _PreparedAgent(
                source_index=agent_idx,
                track_id=agent_arrays["agent_ids"][agent_idx],
                agent_type=_canonical_agent_type_name(agent_arrays["agent_types"][agent_idx]),
                positions=positions,
                history_mask=history_mask,
                future_mask=future_mask,
                current_pos=current_pos,
                role=role,
            )
        )

    focus_agent_idx = _resolve_standardized_focus_agent_idx(
        sample=sample,
        agent_arrays=agent_arrays,
        agents=agents,
        target_track_id=target_track_id,
        target_agent_idx=target_agent_idx,
    )
    if focus_agent_idx < 0:
        focus_agent_idx = _fallback_focus_agent_idx(agents)

    _promote_focus_agent(agents, focus_agent_idx)

    return _PreparedScenario(
        scenario_id=sample.scenario_id,
        source=sample.source,
        agents=agents,
        map_features=map_features,
        focus_agent_idx=focus_agent_idx,
    )


def _resolve_motion_current_time_index(sample: MotionScenario) -> int:
    if sample.current_time_index is not None:
        return int(sample.current_time_index)

    latest_observed = []
    for track in sample.tracks:
        valid_mask = np.asarray(track.valid_mask, dtype=bool)
        observed_mask = np.asarray(track.observed_mask, dtype=bool) & valid_mask
        observed_indices = np.flatnonzero(observed_mask)
        if observed_indices.size > 0:
            latest_observed.append(int(observed_indices.max()))

    if latest_observed:
        return max(latest_observed)
    return max(int(sample.num_steps - 1), 0)


def _resolve_motion_focus_track_id(sample: MotionScenario) -> str | None:
    valid_track_ids = {
        track.track_id
        for track in sample.tracks
        if np.asarray(track.valid_mask, dtype=bool).any()
    }

    if sample.focal_track_id is not None and sample.focal_track_id in valid_track_ids:
        return sample.focal_track_id

    for track in sample.tracks:
        if track.track_id in valid_track_ids and track.is_prediction_target:
            return track.track_id

    if sample.sdc_track_id is not None and sample.sdc_track_id in valid_track_ids:
        return sample.sdc_track_id

    for track in sample.tracks:
        if track.track_id in valid_track_ids:
            return track.track_id

    return None


def _resolve_motion_agent_role(track, focus_track_id: str | None) -> str:
    if track.is_ego:
        return "av"
    if focus_track_id is not None and track.track_id == focus_track_id:
        return "focal"
    if track.is_prediction_target or track.is_object_of_interest or track.is_focal:
        return "score"
    return "other"


def _resolve_standardized_agent_role(
    sample: MotionScenario,
    agent_arrays: dict[str, np.ndarray],
    agent_idx: int,
) -> str:
    if bool(agent_arrays["agent_is_ego"][agent_idx]):
        return "av"
    if int(agent_idx) == int(get_primary_target_index(sample)):
        return "focal"
    if bool(agent_arrays["agent_is_target"][agent_idx]) or bool(agent_arrays["agent_is_interest"][agent_idx]):
        return "score"
    return "other"


def _resolve_focus_agent_idx_from_track_id(
    agents: list[_PreparedAgent],
    focus_track_id: str | None,
) -> int:
    if focus_track_id is None:
        return -1
    for idx, agent in enumerate(agents):
        if agent.track_id == focus_track_id:
            return idx
    return -1


def _resolve_focus_agent_idx_from_source_index(
    agents: list[_PreparedAgent],
    source_index: int,
) -> int:
    for idx, agent in enumerate(agents):
        if agent.source_index == source_index:
            return idx
    return -1


def _resolve_standardized_focus_agent_idx(
    sample: MotionScenario,
    agent_arrays: dict[str, np.ndarray | list[str]],
    agents: list[_PreparedAgent],
    target_track_id: str | None,
    target_agent_idx: int | None,
) -> int:
    if target_track_id is not None:
        focus_agent_idx = _resolve_focus_agent_idx_from_track_id(agents, target_track_id)
        if focus_agent_idx >= 0:
            return focus_agent_idx

    if target_agent_idx is not None:
        focus_agent_idx = _resolve_focus_agent_idx_from_source_index(agents, int(target_agent_idx))
        if focus_agent_idx >= 0:
            return focus_agent_idx

    if (
        get_primary_target_index(sample) >= 0
        and 0 <= int(get_primary_target_index(sample)) < len(agent_arrays["agent_ids"])
        and np.asarray(agent_arrays["agent_valid_mask"][int(get_primary_target_index(sample))], dtype=bool).any()
    ):
        primary_track_id = agent_arrays["agent_ids"][int(get_primary_target_index(sample))]
        focus_agent_idx = _resolve_focus_agent_idx_from_track_id(agents, primary_track_id)
        if focus_agent_idx >= 0:
            return focus_agent_idx

    ego_indices = np.flatnonzero(agent_arrays["agent_is_ego"] & agent_arrays["agent_valid_mask"].any(axis=-1))
    if ego_indices.size > 0:
        ego_track_id = agent_arrays["agent_ids"][int(ego_indices[0])]
        focus_agent_idx = _resolve_focus_agent_idx_from_track_id(agents, ego_track_id)
        if focus_agent_idx >= 0:
            return focus_agent_idx

    return -1


def _fallback_focus_agent_idx(agents: list[_PreparedAgent]) -> int:
    for idx, agent in enumerate(agents):
        if agent.role in {"focal", "av", "score"}:
            return idx
    return 0 if agents else -1


def _promote_focus_agent(agents: list[_PreparedAgent], focus_agent_idx: int) -> None:
    if not (0 <= focus_agent_idx < len(agents)):
        return
    if agents[focus_agent_idx].role == "av":
        return

    for idx, agent in enumerate(agents):
        if idx != focus_agent_idx and agent.role == "focal":
            agent.role = "score"
    agents[focus_agent_idx].role = "focal"


def _collect_motion_map_features(sample: MotionScenario) -> list[_PreparedMapFeature]:
    features: list[_PreparedMapFeature] = []

    for lane_segment in sample.lane_segments:
        points = _extract_xy(lane_segment.centerline)
        if points.shape[0] >= 2:
            features.append(
                _PreparedMapFeature(
                    points=points,
                    feature_type="lane_centerline",
                    is_intersection=bool(lane_segment.is_intersection),
                )
            )

    for road_line in sample.road_lines:
        points = _extract_xy(road_line.points)
        if points.shape[0] >= 2:
            features.append(_PreparedMapFeature(points=points, feature_type="road_line"))

    for road_edge in sample.road_edges:
        points = _extract_xy(road_edge.points)
        if points.shape[0] >= 2:
            features.append(_PreparedMapFeature(points=points, feature_type="road_edge"))

    for polygon_feature in sample.crosswalks:
        points = _close_polygon_if_needed(_extract_xy(polygon_feature.polygon))
        if points.shape[0] >= 2:
            features.append(_PreparedMapFeature(points=points, feature_type="crosswalk"))

    for polygon_feature in sample.speed_bumps:
        points = _close_polygon_if_needed(_extract_xy(polygon_feature.polygon))
        if points.shape[0] >= 2:
            features.append(_PreparedMapFeature(points=points, feature_type="speed_bump"))

    for polygon_feature in sample.driveways:
        points = _close_polygon_if_needed(_extract_xy(polygon_feature.polygon))
        if points.shape[0] >= 2:
            features.append(_PreparedMapFeature(points=points, feature_type="driveway"))

    for polygon_feature in sample.drivable_areas:
        points = _close_polygon_if_needed(_extract_xy(polygon_feature.polygon))
        if points.shape[0] >= 2:
            features.append(_PreparedMapFeature(points=points, feature_type="drivable_area"))

    return features


def _collect_standardized_map_features(
    map_arrays: dict[str, np.ndarray | list[str]],
) -> list[_PreparedMapFeature]:
    features: list[_PreparedMapFeature] = []

    for map_idx in range(map_arrays["map_points"].shape[0]):
        valid_mask = np.asarray(map_arrays["map_valid_mask"][map_idx], dtype=bool)
        if valid_mask.sum() < 2:
            continue

        points = np.asarray(map_arrays["map_points"][map_idx][valid_mask], dtype=np.float32)
        feature_type = _canonical_map_type_name(map_arrays["map_types"][map_idx])
        features.append(
            _PreparedMapFeature(
                points=points,
                feature_type=feature_type,
                is_intersection=bool(map_arrays["map_is_intersection"][map_idx]),
            )
        )

    return features


def _plot_map(ax: plt.Axes, map_features: list[_PreparedMapFeature]) -> None:
    for feature in map_features:
        style = _MAP_STYLES.get(feature.feature_type, _MAP_STYLES["lane_centerline"]).copy()
        if feature.is_intersection and feature.feature_type == "lane_centerline":
            style["color"] = "#d6d6d6"
            style["linewidth"] = 1.1

        points = feature.points
        if points.shape[0] < 2:
            continue

        if feature.feature_type in _POLYGON_MAP_TYPES and np.allclose(points[0], points[-1]):
            ax.fill(
                points[:, 0],
                points[:, 1],
                color=style["color"],
                alpha=0.08,
                zorder=style["zorder"],
            )

        ax.plot(
            points[:, 0],
            points[:, 1],
            color=style["color"],
            linewidth=style["linewidth"],
            zorder=style["zorder"],
        )


def _plot_agent(
    ax: plt.Axes,
    agent_positions_np: np.ndarray,
    agent_history_mask_np: np.ndarray,
    agent_future_mask_np: np.ndarray,
    agent_current_pos_np: np.ndarray,
    color_map: Optional[dict] = _COLOR_MAP["other"],
    label: Optional[str] = None,
    current_only: bool = True,
    agent_type: str = "default",
    text: str | None = None,
):
    if not current_only:
        traj_history = agent_positions_np[agent_history_mask_np]
        traj_future = agent_positions_np[agent_future_mask_np]

        if traj_history.shape[0] >= 2:
            ax.plot(
                traj_history[:, 0],
                traj_history[:, 1],
                color=color_map[0],
                linewidth=_LINE_WIDTH,
                label=f"{label} history" if label is not None else None,
                zorder=1,
            )
        elif traj_history.shape[0] == 1:
            ax.scatter(
                traj_history[:, 0],
                traj_history[:, 1],
                color=color_map[0],
                s=_POINT_SIZE,
                zorder=1,
            )

        if traj_future.shape[0] >= 2:
            ax.plot(
                traj_future[:, 0],
                traj_future[:, 1],
                color=color_map[1],
                linewidth=_LINE_WIDTH,
                label=f"{label} future" if label is not None else None,
                zorder=2,
            )
        elif traj_future.shape[0] == 1:
            ax.scatter(
                traj_future[:, 0],
                traj_future[:, 1],
                color=color_map[1],
                s=_POINT_SIZE,
                zorder=2,
            )

    ax.scatter(
        agent_current_pos_np[0],
        agent_current_pos_np[1],
        color=color_map[2],
        s=_POINT_SIZE,
        zorder=5,
    )

    if text:
        ax.text(
            float(agent_current_pos_np[0]),
            float(agent_current_pos_np[1]),
            text,
            color=color_map[2],
            fontsize=_FONT_SIZE,
        )


def _plot_sample_predictions(
    ax: plt.Axes,
    preds: np.ndarray | None,
    probs: np.ndarray | None,
    max_k: int,
    agents: list[_PreparedAgent],
    focus_agent_idx: int,
) -> None:
    if preds is None:
        return

    preds_np = _to_numpy(preds)
    probs_np = _to_numpy(probs) if probs is not None else None

    if preds_np.ndim == 4:
        for agent_idx in range(min(preds_np.shape[0], len(agents))):
            role = agents[agent_idx].role
            if role not in _SCORE_TYPES:
                continue
            agent_probs = None
            if probs_np is not None and probs_np.ndim >= 2 and agent_idx < probs_np.shape[0]:
                agent_probs = probs_np[agent_idx]

            _plot_predictions(
                ax=ax,
                preds_np=preds_np[agent_idx],
                probs_np=agent_probs,
                max_k=max_k,
                color_map=_COLOR_MAP[role],
                plot_text=(role in {"focal", "av"}),
            )
    elif preds_np.ndim == 3:
        role = "focal"
        if 0 <= focus_agent_idx < len(agents):
            role = agents[focus_agent_idx].role
            if role not in _SCORE_TYPES:
                role = "focal"
        _plot_predictions(
            ax=ax,
            preds_np=preds_np,
            probs_np=probs_np,
            max_k=max_k,
            color_map=_COLOR_MAP[role],
            plot_text=(role in {"focal", "av"}),
        )
    else:
        raise ValueError(
            "preds must have shape [k, t, 2] or [num_agents, k, t, 2], "
            f"got {preds_np.shape}"
        )


def _plot_predictions(
    ax: plt.Axes,
    preds_np: np.ndarray,
    probs_np: np.ndarray | None,
    max_k: int | None = None,
    color_map: Optional[dict] = _COLOR_MAP["other"],
    plot_text: bool = False,
):
    k = int(preds_np.shape[0])
    color = color_map[2]

    if probs_np is None:
        probs_np = np.ones((k,), dtype=np.float32)
    else:
        probs_np = np.asarray(probs_np, dtype=np.float32).reshape(-1)
        if probs_np.shape[0] != k:
            raise ValueError(
                f"probs shape mismatch: expected {k} values, got {probs_np.shape[0]}"
            )

    if max_k is not None and max_k < k:
        sorted_indices = np.argsort(-probs_np)
        selected_indices = sorted_indices[:max_k]
        preds_np = preds_np[selected_indices]
        probs_np = probs_np[selected_indices]

    for pred_coords, pred_prob in zip(preds_np, probs_np):
        ax.plot(
            pred_coords[:, 0],
            pred_coords[:, 1],
            color=color,
            linewidth=_LINE_WIDTH,
            linestyle="--",
            alpha=0.3,
            zorder=3,
        )
        ax.scatter(
            pred_coords[-1, 0],
            pred_coords[-1, 1],
            color=color,
            marker="o",
            s=_POINT_SIZE,
            alpha=0.3,
            zorder=5,
        )

        if plot_text:
            ax.text(
                float(pred_coords[-1, 0]),
                float(pred_coords[-1, 1]),
                f"{pred_prob:.2f}",
                color=color,
                fontsize=_FONT_SIZE,
            )


def _last_valid_position(
    positions: np.ndarray,
    primary_mask: np.ndarray,
    fallback_mask: np.ndarray,
) -> np.ndarray:
    primary_indices = np.flatnonzero(primary_mask)
    if primary_indices.size > 0:
        return np.asarray(positions[int(primary_indices[-1])], dtype=np.float32)

    fallback_indices = np.flatnonzero(fallback_mask)
    if fallback_indices.size > 0:
        return np.asarray(positions[int(fallback_indices[-1])], dtype=np.float32)

    return np.zeros(2, dtype=np.float32)


def _extract_xy(points: np.ndarray | list[list[float]]) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points_np[:, :2]


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "dtype") and str(getattr(array, "dtype")) in {"torch.bfloat16", "torch.float16"}:
        array = array.float()
    if hasattr(array, "numpy"):
        return np.asarray(array.numpy())
    return np.asarray(array)


def _close_polygon_if_needed(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    if np.allclose(points[0], points[-1]):
        return points
    return np.concatenate([points, points[:1]], axis=0)


def _normalize_agent_type_name(agent_type: str | None) -> str:
    if agent_type is None:
        return "unknown"
    return str(agent_type).lower()


def _canonical_agent_type_name(agent_type_index: int) -> str:
    agent_type_index = int(agent_type_index)
    if 0 <= agent_type_index < len(CANONICAL_AGENT_TYPES):
        return CANONICAL_AGENT_TYPES[agent_type_index]
    return "unknown"


def _canonical_map_type_name(map_type_index: int) -> str:
    map_type_index = int(map_type_index)
    if 0 <= map_type_index < len(CANONICAL_MAP_TYPES):
        return CANONICAL_MAP_TYPES[map_type_index]
    return "lane_centerline"


__all__ = ["plot_motion_sample", "plot_scenario"]
