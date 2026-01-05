from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
from shapely.geometry import LineString

# AV2 API Imports
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    TrackCategory,
    ObjectType,
)

_AGENT_TYPE_MAP = {
    ObjectType.VEHICLE: 0,
    ObjectType.PEDESTRIAN: 1,
    ObjectType.MOTORCYCLIST: 2,
    ObjectType.CYCLIST: 3,
    ObjectType.BUS: 4,
    # static objects are labeled as 6
    ObjectType.UNKNOWN: 5,
}

_LANE_TYPE_MAP = {
    LaneType.VEHICLE: 0,
    LaneType.BIKE: 1,
    LaneType.BUS: 2,
}


_AGENT_SCORE_TYPE_MAP = {
    "frag": 0,
    TrackCategory.TRACK_FRAGMENT: 0,
    "unscore": 1,
    TrackCategory.UNSCORED_TRACK: 1,
    "score": 2,
    TrackCategory.SCORED_TRACK: 2,
    "focal": 3,
    TrackCategory.FOCAL_TRACK: 3,
    "av": 4,
}

_LANE_MARK_TYPE = {
    LaneMarkType.DASH_SOLID_YELLOW: 0,
    LaneMarkType.DASH_SOLID_WHITE: 0,
    LaneMarkType.DASHED_WHITE: 0,
    LaneMarkType.DASHED_YELLOW: 0,
    LaneMarkType.DOUBLE_DASH_YELLOW: 0,
    LaneMarkType.DOUBLE_DASH_WHITE: 0,
    LaneMarkType.DOUBLE_SOLID_YELLOW: 1,
    LaneMarkType.DOUBLE_SOLID_WHITE: 1,
    LaneMarkType.SOLID_YELLOW: 1,
    LaneMarkType.SOLID_WHITE: 1,
    LaneMarkType.SOLID_DASH_WHITE: 1,
    LaneMarkType.SOLID_DASH_YELLOW: 1,
    LaneMarkType.SOLID_BLUE: 1,
}


class AV2BaseDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        hist_steps: int = 50,
        fut_steps: int = 60,
        max_agents: int = 64,
        max_lanes: int = 128,
        lane_seg_length: float = 15.0,
        num_points_per_lane: int = 10,
        radius: float = 100.0,
        min_distance_threshold: float = 20.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split

        self.hist_steps = hist_steps
        self.fut_steps = fut_steps
        self.curr_step_idx = hist_steps - 1
        self.min_dist_threshold = min_distance_threshold
        self.total_steps = hist_steps + fut_steps
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.radius = radius
        self.num_lane_seg = lane_seg_length
        self.num_lane_nodes = num_points_per_lane
        # Timestamps
        if self.split == "test":
            # Test set only has history data
            self.ts_frame_indices = np.arange(0, self.hist_steps)
        else:
            self.ts_frame_indices = np.arange(0, self.total_steps)

        self.log_dirs = sorted((self.data_root / split).glob("*"))

    def __len__(self) -> int:
        return len(self.log_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_scene_centric_data(
        self, idx: int, centric: str = "focal"
    ) -> Dict[str, Any]:
        """Return a single scene in the desired coordinate frame."""
        assert centric in {"focal", "global", "ego"}

        data = self._build_from_raw(self.log_dirs[idx].name, self.log_dirs[idx])

        if centric == "global":
            return data
        elif centric in ["focal", "ego"]:
            anchor_idx = 0 if centric == "focal" else 1

            anchor_pos = data["agent_last_positions"][anchor_idx]
            anchor_ang = data["agent_last_heading_angles"][anchor_idx]

            agent_pos_local = self.transform_global_to_local(
                data["agent_positions"], anchor_pos, anchor_ang
            )
            agent_vel_local = self.transform_global_to_local(
                data["agent_velocities"], anchor_pos, anchor_ang, translate=False
            )
            agent_ang_local = self.transform_angle_to_local(
                data["agent_heading_angles"], anchor_ang
            )
            agent_last_pos_local = self.transform_global_to_local(
                data["agent_last_positions"], anchor_pos, anchor_ang
            )
            agent_last_ang_local = self.transform_angle_to_local(
                data["agent_last_heading_angles"], anchor_ang
            )
            agent_last_vel_local = self.transform_global_to_local(
                data["agent_last_velocities"], anchor_pos, anchor_ang, translate=False
            )
            lane_points_local = self.transform_global_to_local(
                data["lane_points"], anchor_pos, anchor_ang
            )

            data.update(
                {
                    "agent_positions": agent_pos_local,
                    "agent_velocities": agent_vel_local,
                    "agent_heading_angles": agent_ang_local,
                    "agent_last_positions": agent_last_pos_local,
                    "agent_last_heading_angles": agent_last_ang_local,
                    "agent_last_velocities": agent_last_vel_local,
                    "lane_points": lane_points_local,
                }
            )
            return data
        else:
            raise ValueError(f"Unknown centric type: {centric}")

    def get_agent_centric_data(self, idx: int) -> List[Dict[str, Any]]:
        # TODO: I'm not sure if I random select one agent as anchor or return all agents
        # as there is no agent-centric model currently, skip this
        raise NotImplementedError

    def get_instance_centric_data(self, idx: int) -> Dict[str, Any]:
        """
        Return a single dict where:
        - agent trajectories/velocities/headings are expressed relative to each
          agent's own last pose, and
        - lane segments are expressed relative to their first node/orientation.

        Additional keys:
            lane_anchor_positions: [M, 2] global coordinates of lane anchors.
            lane_anchor_headings: [M] heading angles (rad) of the anchor frames.
        """
        data = self._build_from_raw(self.log_dirs[idx].name, self.log_dirs[idx])

        anchor_pos = data["agent_last_positions"]  # [N, 2]
        anchor_ang = data["agent_last_heading_angles"]  # [N, ]

        agent_pos_local = self.transform_global_to_local(
            data["agent_positions"],
            anchor_pos[:, None, :],
            anchor_ang[:, None],
        )
        agent_vel_local = self.transform_global_to_local(
            data["agent_velocities"],
            None,
            anchor_ang[:, None],
            translate=False,
        )
        agent_ang_local = self.transform_angle_to_local(
            data["agent_heading_angles"],
            anchor_ang[:, None],
        )

        lane_points = data["lane_points"]  # [M, num_points_per_lane, 2]
        # lane_anchor_pos = lane_points[:, 0, :]
        # lane_anchor_vec = lane_points[:, 1, :] - lane_points[:, 0, :]
        # above uses first segment to define anchor frame, below uses middle point
        lane_anchor_pos = lane_points[:, self.num_lane_nodes // 2, :]
        lane_anchor_vec = (
            lane_points[:, self.num_lane_nodes // 2 + 1, :]
            - lane_points[:, self.num_lane_nodes // 2 - 1, :]
        )
        lane_anchor_ang = np.arctan2(lane_anchor_vec[:, 1], lane_anchor_vec[:, 0])
        lane_points_local = self.transform_global_to_local(
            lane_points,
            lane_anchor_pos[:, None, :],
            lane_anchor_ang[:, None],
        )

        data.update(
            {
                "agent_positions": agent_pos_local,
                "agent_velocities": agent_vel_local,
                "agent_heading_angles": agent_ang_local,
                "lane_points": lane_points_local,
                "lane_anchor_positions": lane_anchor_pos,
                "lane_anchor_headings": lane_anchor_ang,
            }
        )
        return data

    def _build_from_raw(self, log_id: str, log_dir: Path) -> Optional[Dict]:
        json_file = log_dir / f"log_map_archive_{log_id}.json"
        parquet_file = log_dir / f"scenario_{log_id}.parquet"
        if not json_file.exists():
            raise FileNotFoundError(f"{json_file} not found")
        if not parquet_file.exists():
            raise FileNotFoundError(f"{parquet_file} not found")

        # * load raw AV2 objects
        static_map = ArgoverseStaticMap.from_json(json_file)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_file)

        # * extract data
        lane_data_dict = self._extract_lane_data(static_map)
        agent_data_dict = self._extract_agent_data(
            scenario,
            lane_points=lane_data_dict["lane_points"],
        )
        lane_data_dict = self._filter_lanes_topk(
            lane_data_dict,
            center=agent_data_dict["agent_last_positions"][0],
        )

        sample = {
            "scenario_id": log_id,
            "city": scenario.city_name,
            **lane_data_dict,
            **agent_data_dict,
        }
        return sample

    def _extract_agent_data(
        self,
        scenario: ArgoverseScenario,
        lane_points: Optional[np.ndarray] = None,
    ):
        """
        Extracts, filters, normalizes, and pads trajectories.
        lane_points: [num_lanes, num_points_per_lane, 2]
        """
        # Identify Track Indices
        focal_idx, ego_idx = None, None
        scored_idcs, unscored_idcs, fragment_idcs = [], [], []

        for idx, track in enumerate(scenario.tracks):
            if (
                track.track_id == scenario.focal_track_id
                and track.category == TrackCategory.FOCAL_TRACK
            ):
                focal_idx = idx
            elif track.track_id == "AV":
                ego_idx = idx
            elif track.category == TrackCategory.SCORED_TRACK:
                scored_idcs.append(idx)
            elif track.category == TrackCategory.UNSCORED_TRACK:
                unscored_idcs.append(idx)
            elif track.category == TrackCategory.TRACK_FRAGMENT:
                fragment_idcs.append(idx)

        # Enforce existence of AV and Focal
        if focal_idx is None or ego_idx is None:
            raise ValueError(
                f"Focal or ego track not found in scenario {scenario.scenario_id}"
            )

        # Sorting order: Focal -> AV -> Scored -> Unscored -> Fragments
        sorted_indices = (
            [focal_idx, ego_idx] + scored_idcs + unscored_idcs + fragment_idcs
        )
        sorted_score_types = (
            [
                _AGENT_SCORE_TYPE_MAP["focal"],
                _AGENT_SCORE_TYPE_MAP["av"],
            ]
            + [_AGENT_SCORE_TYPE_MAP["score"]] * len(scored_idcs)
            + [_AGENT_SCORE_TYPE_MAP["unscore"]] * len(unscored_idcs)
            + [_AGENT_SCORE_TYPE_MAP["frag"]] * len(fragment_idcs)
        )

        # Containers
        agent_pos_global_list = []
        agent_ang_global_list = []
        agent_vel_global_list = []
        agent_last_pos_global_list = []
        agent_last_ang_global_list = []
        agent_last_vel_global_list = []
        agent_score_type_list = []
        agent_type_list = []
        valid_mask_list = []
        # * get original scene-centric agent trajectories
        lane_points_cache = None
        if lane_points is not None:
            lane_points_cache = np.expand_dims(
                lane_points.reshape(-1, 2),
                axis=0,
            )

        for i, track_idx in enumerate(sorted_indices):
            track = scenario.tracks[track_idx]

            # Extract raw states
            i_agent_ts = np.array(
                [x.timestep for x in track.object_states], dtype=np.int16
            )
            i_agent_pos = np.array([x.position for x in track.object_states])  # [T, 2]
            i_agent_ang = np.array([x.heading for x in track.object_states])  # [T, ]
            i_agent_vel = np.array([x.velocity for x in track.object_states])  # [T, 2]

            # * Filter: Skip if strictly future or not present at observation time
            if (
                i_agent_ts[0] > self.curr_step_idx
                or self.curr_step_idx not in i_agent_ts
            ):
                continue

            if not self._agent_passes_filters(
                score_type=sorted_score_types[i],
                timestamps=i_agent_ts,
                positions=i_agent_pos,
                lane_points_cache=lane_points_cache,
            ):
                continue

            if len(agent_pos_global_list) >= self.max_agents:
                break

            idx_at_curr = np.where(i_agent_ts == self.curr_step_idx)[0][0]

            i_agent_last_pos = i_agent_pos[idx_at_curr]  # [2]
            i_agent_last_ang = i_agent_ang[idx_at_curr]  # Scalar
            i_agent_last_vel = i_agent_vel[idx_at_curr]  # [2]

            agent_last_pos_global_list.append(i_agent_last_pos)
            agent_last_ang_global_list.append(i_agent_last_ang)
            agent_last_vel_global_list.append(i_agent_last_vel)

            # --- Padding ---
            # Create full length arrays filled with nan/zeros
            # Flags (1 if present)
            valid_mask = np.zeros(self.total_steps, dtype=bool)
            has_ts_mask = np.isin(i_agent_ts, self.ts_frame_indices)
            # Map valid timestamps to indices in the fixed array
            mapped_indices = i_agent_ts[has_ts_mask]
            valid_mask[mapped_indices] = True

            type_id = _AGENT_TYPE_MAP.get(track.object_type, 6)  # default to static(6)

            # Position & Angle (Nearest Neighbor Padding)
            agent_pos_pad = np.full((self.total_steps, 2), np.nan)
            agent_pos_pad[mapped_indices] = i_agent_pos[has_ts_mask]
            agent_pos_pad = self._padding_nearest_neighbor(agent_pos_pad)

            agent_ang_pad = np.full(self.total_steps, np.nan)
            agent_ang_pad[mapped_indices] = i_agent_ang[has_ts_mask]
            agent_ang_pad = self._padding_nearest_neighbor(agent_ang_pad)

            # Velocity (Zero Padding)
            agent_vel_pad = np.zeros((self.total_steps, 2))
            agent_vel_pad[mapped_indices] = i_agent_vel[has_ts_mask]

            # Append
            agent_pos_global_list.append(agent_pos_pad)
            agent_ang_global_list.append(agent_ang_pad)
            agent_vel_global_list.append(agent_vel_pad)
            agent_type_list.append(type_id)
            valid_mask_list.append(valid_mask)
            agent_score_type_list.append(sorted_score_types[i])

        agent_pos_global_np = np.array(
            agent_pos_global_list, dtype=np.float32
        )  # [N, 110, 2]
        agent_ang_global_np = np.array(
            agent_ang_global_list, dtype=np.float32
        )  # [N, 110]

        agent_vel_global_np = np.array(
            agent_vel_global_list, dtype=np.float32
        )  # [N, 110, 2]
        agent_last_pos_global_np = np.array(
            agent_last_pos_global_list, dtype=np.float32
        )  # [N, 2]
        agent_last_ang_global_np = np.array(
            agent_last_ang_global_list, dtype=np.float32
        )  # [N, ]
        agent_last_vel_global_np = np.array(
            agent_last_vel_global_list, dtype=np.float32
        )  # [N, 2]

        agent_type_np = np.array(agent_type_list, dtype=np.int16)
        agent_score_type_np = np.array(agent_score_type_list, dtype=np.int16)
        valid_mask_np = np.array(valid_mask_list, dtype=bool)

        # Stack into arrays
        return {
            "agent_positions": agent_pos_global_np,  # [N, 110, 2]
            "agent_heading_angles": agent_ang_global_np,  # [N, 110]
            "agent_velocities": agent_vel_global_np,  # [N, 110, 2]
            "agent_valid_masks": valid_mask_np,  # [N, 110]
            "agent_last_positions": agent_last_pos_global_np,  # [N, 2]
            "agent_last_heading_angles": agent_last_ang_global_np,  # [N, ]
            "agent_last_velocities": agent_last_vel_global_np,  # [N, 2]
            "agent_types": agent_type_np,
            "agent_score_types": agent_score_type_np,
        }

    def _agent_passes_filters(
        self,
        score_type: int,
        timestamps: np.ndarray,
        positions: np.ndarray,
        lane_points_cache: Optional[np.ndarray],
    ) -> bool:
        """Apply per-agent filters before stacking into tensors."""
        if lane_points_cache is not None and score_type in {
            _AGENT_SCORE_TYPE_MAP["unscore"],
            _AGENT_SCORE_TYPE_MAP["frag"],
        }:
            history_mask = timestamps <= self.curr_step_idx
            if not history_mask.any():
                return False
            agent_history_points = np.expand_dims(
                positions[history_mask],
                axis=1,
            )
            dist = np.linalg.norm(
                agent_history_points - lane_points_cache,
                axis=-1,
            )
            if np.min(dist) > self.min_dist_threshold:
                return False
        return True

    def _padding_nearest_neighbor(self, traj):
        """Fills NaNs/Nones with the nearest valid value (forward then backward)."""
        # Note: Input assumes NaNs for float arrays.
        n = len(traj)
        # Forward pass
        buff = None
        for i in range(n):
            if not np.isnan(traj[i]).any():
                buff = traj[i]
            elif buff is not None:
                traj[i] = buff

        # Backward pass
        buff = None
        for i in reversed(range(n)):
            if not np.isnan(traj[i]).any():
                buff = traj[i]
            elif buff is not None:
                traj[i] = buff

        # Handle case where everything is NaN (shouldn't happen due to logic)
        if np.isnan(traj).any():
            traj[np.isnan(traj)] = 0.0

        return traj

    def _extract_lane_data(
        self,
        static_map: ArgoverseStaticMap,
    ):
        """
        Discretizes lane segments and extracts topological features.
        """
        # * 1. collect lane and do segment and interpolation
        lane_seg_global_list = []
        lane_type_list = []
        lane_is_intersect_list = []
        left_mark_type_list, right_mark_type_list = [], []
        has_left_nb_list, has_right_nb_list = [], []
        for lane_id, lane in static_map.vector_lane_segments.items():
            # Get Centerline
            centerline_raw = static_map.get_lane_segment_centerline(lane_id)[
                :, 0:2
            ]  # [N, 2], only 2D needed
            centerline_ls = LineString(centerline_raw)

            # Determine number of sub-segments
            num_segs = max(int(np.floor(centerline_ls.length / self.num_lane_seg)), 1)
            ds = centerline_ls.length / num_segs

            for i in range(num_segs):
                # Interpolate points for this sub-segment
                sub_lower_bound = i * ds
                sub_upper_bound = (i + 1) * ds
                num_sub_nodes = self.num_lane_nodes

                centerline_points = [
                    centerline_ls.interpolate(s)
                    for s in np.linspace(
                        sub_lower_bound, sub_upper_bound, num_sub_nodes + 1
                    )
                ]  # 11 points for 10 vectors
                centerline_points = np.array(
                    LineString(centerline_points).coords
                )  # [N+1, 2], [11, 2]

                lane_seg_global_list.append(centerline_points)

                # lane type
                lane_type_list.append(_LANE_TYPE_MAP[lane.lane_type])

                # lane intersection
                lane_is_intersect_list.append(1 if lane.is_intersection else 0)

                # lane mark type:
                left_mark_type_list.append(_LANE_MARK_TYPE.get(lane.left_mark_type, 2))
                right_mark_type_list.append(
                    _LANE_MARK_TYPE.get(lane.right_mark_type, 2)
                )

                # neighbors
                has_left_nb_list.append(1 if lane.left_neighbor_id else 0)
                has_right_nb_list.append(1 if lane.right_neighbor_id else 0)

        lane_points_np = np.array(lane_seg_global_list).astype(
            np.float32
        )  # [M, 11, 2], global coords
        lane_types_np = np.array(lane_type_list).astype(np.int16)  # [M, 3]
        lane_is_intersect_np = np.array(lane_is_intersect_list).astype(
            np.int16
        )  # [M, ]
        left_mark_np = np.array(left_mark_type_list).astype(np.int16)  # [M, ]
        right_mark_np = np.array(right_mark_type_list).astype(np.int16)  # [M, ]
        has_left_nb_np = np.array(has_left_nb_list).astype(np.int16)  # [M, ]
        has_right_nb_np = np.array(has_right_nb_list).astype(np.int16)  # [M, ]

        lane_data = {
            "lane_points": lane_points_np,
            "lane_types": lane_types_np,
            "lane_is_intersect": lane_is_intersect_np,
            "left_marks": left_mark_np,
            "right_marks": right_mark_np,
            "has_left_nb": has_left_nb_np,
            "has_right_nb": has_right_nb_np,
        }
        return lane_data

    def _filter_lanes_topk(
        self,
        lane_data: Dict[str, np.ndarray],
        center: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Filter lane segments to top-k (closest to ego_xy).

        Args:
            lane_data:
                dict that contains at least:
                - "lane_points": np.ndarray, shape [M, P, 2]
                and optionally other fields with first dim M
            ego_xy:
                np.ndarray shape [2,], ego position in the same coord as lane_points
            max_lanes:
                int, keep at most this many lane segments; if None, use self.max_lanes

        Returns:
            filtered lane_data dict with first dim <= max_lanes
        """
        # no limit
        if self.max_lanes is None:
            return lane_data

        lane_points = lane_data["lane_points"]

        M = lane_points.shape[0]
        if M <= self.max_lanes:
            return lane_data

        center = np.asarray(center, dtype=np.float32).reshape(1, 1, 2)

        # center of each lane segment: [M, 2]
        centers = lane_points.mean(axis=1, keepdims=True)  # [M,1,2]
        d2 = np.sum((centers - center) ** 2, axis=2).reshape(-1)  # [M]
        keep = np.argsort(d2)[: self.max_lanes]  # [K]

        # Filter all fields that have first dim == M
        filtered: Dict[str, Any] = {}
        for k, v in lane_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == M:
                filtered[k] = v[keep]
            else:
                # keep unchanged (or drop if你希望严格一致也行)
                filtered[k] = v

        return filtered

    def transform_global_to_local(
        self,
        arr: np.ndarray,
        anchor_pos: Optional[np.ndarray],
        anchor_ang: np.ndarray,
        translate: bool = True,
    ) -> np.ndarray:
        """Rotate (and optionally translate) global coordinates into the local frame."""
        arr = np.asarray(arr, dtype=np.float32)
        anchor_ang = np.asarray(anchor_ang, dtype=np.float32)
        cos_ang = np.cos(anchor_ang)
        sin_ang = np.sin(anchor_ang)

        x = arr[..., 0]
        y = arr[..., 1]

        if translate:
            if anchor_pos is None:
                raise ValueError("anchor_pos required when translate=True.")
            anchor_pos = np.asarray(anchor_pos, dtype=np.float32)
            dx = x - anchor_pos[..., 0]
            dy = y - anchor_pos[..., 1]
        else:
            dx = x
            dy = y

        local_x = cos_ang * dx + sin_ang * dy
        local_y = -sin_ang * dx + cos_ang * dy
        return np.stack([local_x, local_y], axis=-1).astype(np.float32)

    def transform_local_to_global(
        self,
        arr: np.ndarray,
        anchor_pos: Optional[np.ndarray],
        anchor_ang: np.ndarray,
        translate: bool = True,
    ) -> np.ndarray:
        """Transform local coordinates back to the global frame."""
        arr = np.asarray(arr, dtype=np.float32)
        anchor_ang = np.asarray(anchor_ang, dtype=np.float32)
        cos_ang = np.cos(anchor_ang)
        sin_ang = np.sin(anchor_ang)

        x = arr[..., 0]
        y = arr[..., 1]

        global_x = cos_ang * x - sin_ang * y
        global_y = sin_ang * x + cos_ang * y
        coords = np.stack([global_x, global_y], axis=-1)

        if translate:
            if anchor_pos is None:
                raise ValueError("anchor_pos required when translate=True.")
            anchor_pos = np.asarray(anchor_pos, dtype=np.float32)
            coords = coords + anchor_pos

        return coords.astype(np.float32)

    def transform_angle_to_local(
        self, heading: np.ndarray, anchor_ang: np.ndarray
    ) -> np.ndarray:
        """Express heading in the anchor frame."""
        heading = np.asarray(heading, dtype=np.float32)
        anchor_ang = np.asarray(anchor_ang, dtype=np.float32)
        rel_heading = heading - anchor_ang
        return self._wrap_angle(rel_heading)

    def transform_angle_to_global(
        self, heading: np.ndarray, anchor_ang: np.ndarray
    ) -> np.ndarray:
        """Convert a local heading back to the global frame."""
        heading = np.asarray(heading, dtype=np.float32)
        anchor_ang = np.asarray(anchor_ang, dtype=np.float32)
        global_heading = heading + anchor_ang
        return self._wrap_angle(global_heading)

    def _wrap_angle(self, angle: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi)."""
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        return wrapped.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    idx = 0

    def _plot_scene(ax, sample: Dict[str, Any], title: str) -> None:
        """Simple visualization of lanes plus agent history/future."""
        hist_steps = 50

        # * plot lanes
        lane_points = sample.get("lane_points")
        if lane_points is not None:
            for lane in lane_points:
                ax.plot(
                    lane[:, 0], lane[:, 1], color="#000000", linewidth=0.5, zorder=0
                )

        agent_pos = sample["agent_positions"]
        agent_masks = sample["agent_valid_masks"]

        cmap = plt.get_cmap("tab20")
        for idx, (traj, mask) in enumerate(zip(agent_pos, agent_masks)):
            if not mask.any():
                continue
            color = cmap(idx % 20)
            hist_mask = mask.copy()
            hist_mask[hist_steps:] = False
            fut_mask = mask.copy()
            fut_mask[:hist_steps] = False

            hist_traj = traj[hist_mask]
            fut_traj = traj[fut_mask]

            if len(hist_traj) > 0:
                ax.plot(hist_traj[:, 0], hist_traj[:, 1], color=color, linewidth=1.0)
            if len(fut_traj) > 0:
                ax.plot(
                    fut_traj[:, 0],
                    fut_traj[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                )
            ax.scatter(
                traj[hist_steps - 1 : hist_steps, 0],
                traj[hist_steps - 1 : hist_steps, 1],
                color=color,
                s=10,
                zorder=2,
            )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.4)

    dataset = AV2BaseDataset(
        data_root="data",
        split="train",
    )

    views = {}
    views["global"] = dataset.get_scene_centric_data(idx, "global")
    views["focal"] = dataset.get_scene_centric_data(idx, "focal")
    views["ego"] = dataset.get_scene_centric_data(idx, "ego")
    views["instance"] = dataset.get_instance_centric_data(idx)

    num_plots = len(views)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, sample) in zip(axes, views.items()):
        num_agents = sample["agent_positions"].shape[0]
        lane_points = sample.get("lane_points")
        num_lanes = lane_points.shape[0] if lane_points is not None else 0
        print(f"{name}: agents={num_agents}, lanes={num_lanes}")
        _plot_scene(ax, sample, title=name)

    # Hide unused axes
    for ax in axes[len(views) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("test_av2_base_dataset.png", dpi=300)
