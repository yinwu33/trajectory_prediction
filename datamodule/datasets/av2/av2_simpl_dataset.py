import os
import math
import random
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import LineString

# AV2 API Imports
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor."""
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data


class AV2SimplDataset(Dataset):
    """
    A unified dataset class for Argoverse 2 Motion Forecasting.
    Handles preprocessing from raw logs, caching, and loading for training.
    """

    def __init__(
        self,
        data_root: str,
        preprocess_dir: str,
        split: str = "train",
        history_steps: int = 50,
        future_steps: int = 60,
        min_distance_threshold: float = 20.0,
        augmentation: bool = False,
        force_preprocess: bool = False,
    ):
        """
        Args:
            raw_data_path: Path to the raw AV2 dataset (folders containing log_map_archive and scenarios).
            cached_data_path: Path where processed .pt files will be saved/loaded.
            split: 'train', 'val', or 'test'.
            history_steps: Number of history frames (default 50).
            future_steps: Number of future frames to predict (default 60).
            min_distance_threshold: Filter out actors too far from the map (default 20.0).
            augmentation: Whether to apply data augmentation (random flips).
            force_preprocess: If True, ignores existing cache and re-processes data.
        """
        self.data_root = Path(data_root) / split
        self.preprocess_dir = Path(preprocess_dir) / "av2_simpl" / split
        self.split = split
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.seq_len = history_steps + future_steps
        self.min_dist_threshold = min_distance_threshold
        self.augmentation = augmentation
        self.force_preprocess = force_preprocess

        self.truncate_steps = 2

        # Constants for Map Processing
        self.LANE_SEG_LENGTH = 15.0  # Approximated lane segment length
        self.LANE_NODES_COUNT = 10  # Number of nodes per segment

        # Prepare Dataset File List (Lazy Loading)
        # We store metadata here and process in __getitem__ for parallelization
        self.scenario_ids = []
        self._prepare_file_list()

    def _prepare_file_list(self):
        """
        Scans directories to build a list of available scenarios.
        Does not process data; processing happens in __getitem__.
        """
        self.preprocess_dir.mkdir(parents=True, exist_ok=True)

        # 1. Check for Raw Data
        if self.data_root.exists():
            # If raw data exists, we base our dataset on the raw log folders.
            # This allows us to process them if cache is missing.
            log_dirs = sorted(list(self.data_root.glob("*")))
            self.scenario_ids = [d.name for d in log_dirs]
            print(f"[{self.split}] Found {len(self.scenario_ids)} raw logs.")
            print(
                f"[{self.split}] Data will be processed/loaded lazily in __getitem__."
            )

        # 2. Fallback: Check for Cached Data only
        # (Useful if raw data was deleted to save space)
        elif self.preprocess_dir.exists():
            cached_files = sorted(list(self.preprocess_dir.glob("*.pt")))
            self.scenario_ids = [f.stem for f in cached_files]
            print(
                f"[{self.split}] Found {len(self.scenario_ids)} cached files (Raw data not found)."
            )

        else:
            print(
                f"Warning: No data found at {self.data_root} or {self.preprocess_dir}"
            )

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a scenario, processes it if necessary (and saves to cache),
        applies augmentation, and formats input.
        """
        log_id = self.scenario_ids[idx]
        cached_path = self.preprocess_dir / f"{log_id}.pt"
        raw_path = self.data_root / log_id

        data_item = None

        # 1. Try Loading from Cache
        # Skip if forcing preprocess
        if cached_path.exists() and not self.force_preprocess:
            try:
                # Weights_only=False required for some dict/list structures,
                # but explicit safe loading is preferred if possible.
                data_item = torch.load(
                    cached_path, map_location="cpu", weights_only=False
                )
            except Exception as e:
                print(
                    f"Error loading cached file {cached_path}: {e}. Retrying raw process."
                )
                # If cache is corrupted, try to delete it
                try:
                    os.remove(cached_path)
                except OSError:
                    pass

        # 2. Process from Raw if needed
        if data_item is None:
            if not raw_path.exists():
                print(f"Error: Cache missing and raw data missing for {log_id}")
                # Fallback to random sample
                return self.__getitem__(random.randint(0, len(self) - 1))

            try:
                data_item = self._process_raw_log(raw_path)

                if data_item is not None:
                    # Save to cache using a temp file + atomic rename to prevent
                    # corruption if multiple workers write or process is interrupted
                    tmp_path = cached_path.with_suffix(".tmp")
                    torch.save(data_item, tmp_path)
                    try:
                        tmp_path.rename(cached_path)
                    except OSError:
                        # Windows sometimes struggles with atomic renames of existing files
                        if not cached_path.exists():
                            tmp_path.rename(cached_path)
                        else:
                            os.remove(tmp_path)
            except Exception as e:
                print(f"Failed to process {log_id}: {e}")

        # 3. Validation
        if data_item is None:
            # If processing failed (e.g. empty log or error), skip to another
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 4. Augmentation
        data = self._apply_augmentation(data_item)

        # 5. Unpack Data
        seq_id = data["SEQ_ID"]
        city_name = data["CITY_NAME"]
        orig = data["ORIG"]
        rot = data["ROT"]
        trajs_data = data["TRAJS"]
        lane_graph = data["LANE_GRAPH"]

        # 6. Split Trajectories into History (Obs) and Future (Fut)
        # trajs_pos shape: [N_actors, 110, 2]
        trajs_pos_obs = trajs_data["trajs_pos"][:, : self.history_steps]
        trajs_ang_obs = trajs_data["trajs_ang"][:, : self.history_steps]
        trajs_vel_obs = trajs_data["trajs_vel"][:, : self.history_steps]
        pad_obs = trajs_data["has_flags"][:, : self.history_steps]
        trajs_type = trajs_data["trajs_type"][:, : self.history_steps]

        trajs_pos_fut = trajs_data["trajs_pos"][:, self.history_steps :]
        trajs_ang_fut = trajs_data["trajs_ang"][:, self.history_steps :]
        trajs_vel_fut = trajs_data["trajs_vel"][:, self.history_steps :]
        pad_fut = trajs_data["has_flags"][:, self.history_steps :]

        # 7. Construct Output Dictionary
        output_trajs = {
            # Observation
            "TRAJS_POS_OBS": trajs_pos_obs,
            "TRAJS_ANG_OBS": np.stack(
                [np.cos(trajs_ang_obs), np.sin(trajs_ang_obs)], axis=-1
            ),
            "TRAJS_VEL_OBS": trajs_vel_obs,
            "TRAJS_TYPE": trajs_type,
            "PAD_OBS": pad_obs,
            # Future Ground Truth
            "TRAJS_POS_FUT": trajs_pos_fut,
            "TRAJS_ANG_FUT": np.stack(
                [np.cos(trajs_ang_fut), np.sin(trajs_ang_fut)], axis=-1
            ),
            "TRAJS_VEL_FUT": trajs_vel_fut,
            "PAD_FUT": pad_fut,
            # Anchors / Metadata
            "TRAJS_CTRS": trajs_data["trajs_ctrs"],
            "TRAJS_VECS": trajs_data["trajs_vecs"],
            "TRAJS_TID": trajs_data["trajs_tid"],  # target id
            "TRAJS_CAT": trajs_data["trajs_cat"],
            # Training Mask (Assume all valid loaded actors are trainable)
            "TRAIN_MASK": np.ones(len(trajs_data["trajs_ctrs"]), dtype=bool),
        }

        # Yaw Loss Mask: Only apply yaw loss for vehicles/bikes (indices 0, 2, 3, 4)
        # Object Types: 0:Veh, 1:Ped, 2:Moto, 3:Cyc, 4:Bus, 5:Unknown, 6:Static
        yaw_loss_mask = np.array(
            [np.where(x)[0][0] in [0, 2, 3, 4] for x in trajs_type[:, -1]], dtype=bool
        )
        output_trajs["YAW_LOSS_MASK"] = yaw_loss_mask

        # 8. Calculate Relative Positional Embeddings (RPE)
        # Convert to torch for calculation
        scene_ctrs = torch.cat(
            [
                torch.from_numpy(output_trajs["TRAJS_CTRS"]),
                torch.from_numpy(lane_graph["lane_ctrs"]),
            ],
            dim=0,
        )

        scene_vecs = torch.cat(
            [
                torch.from_numpy(output_trajs["TRAJS_VECS"]),
                torch.from_numpy(lane_graph["lane_vecs"]),
            ],
            dim=0,
        )

        rpe = self._get_rpe(scene_ctrs, scene_vecs)

        return {
            "SEQ_ID": seq_id,
            "CITY_NAME": city_name,
            "ORIG": orig,
            "ROT": rot,
            "TRAJS": output_trajs,
            "LANE_GRAPH": lane_graph,
            "RPE": rpe,  # list
        }

    # =========================================================================
    # PREPROCESSING HELPERS
    # =========================================================================

    def _process_raw_log(self, log_dir: Path) -> Optional[Dict]:
        """
        Processes a single raw log folder into a dictionary.
        """
        log_id = log_dir.name
        json_file = log_dir / f"log_map_archive_{log_id}.json"
        parquet_file = log_dir / f"scenario_{log_id}.parquet"

        if not json_file.exists() or not parquet_file.exists():
            raise RuntimeError(f"json file or parquet file not founded: {log_id}")

        # Load raw AV2 objects
        static_map = ArgoverseStaticMap.from_json(json_file)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_file)

        # 1. Extract Trajectories
        # trajs_* shapes: [N, 110, ...]
        (
            trajs_pos,
            trajs_ang,
            trajs_vel,
            trajs_type,
            has_flags,
            trajs_tid,
            trajs_cat,
            orig_seq,
            rot_seq,
        ) = self._extract_trajectories(scenario, static_map)

        if len(trajs_pos) == 0:
            return None

        # 2. Extract Lane Graph (Vector Map)
        lane_graph = self._extract_lane_graph(orig_seq, rot_seq, static_map)

        # Package data
        # Note: We save a dict, not a list of lists, for clarity
        return {
            "SEQ_ID": log_id,
            "CITY_NAME": scenario.city_name,
            "ORIG": orig_seq,
            "ROT": rot_seq,
            "TRAJS": {
                "trajs_pos": trajs_pos,
                "trajs_ang": trajs_ang,
                "trajs_vel": trajs_vel,
                "trajs_type": trajs_type,
                "has_flags": has_flags,
                "trajs_tid": trajs_tid,
                "trajs_cat": trajs_cat,
                # These are computed in extract_trajectories relative to scene origin
                "trajs_ctrs": trajs_pos[:, self.history_steps - 1, :],
                "trajs_vecs": np.stack(
                    [
                        np.cos(trajs_ang[:, self.history_steps - 1]),
                        np.sin(trajs_ang[:, self.history_steps - 1]),
                    ],
                    axis=-1,
                ),
            },
            "LANE_GRAPH": lane_graph,
        }

    def _extract_trajectories(
        self, scenario: ArgoverseScenario, static_map: ArgoverseStaticMap
    ):
        """
        Extracts, filters, normalizes, and pads trajectories.
        """
        # Identify Track Indices
        focal_idx, av_idx = None, None
        scored_idcs, unscored_idcs, fragment_idcs = [], [], []

        for idx, track in enumerate(scenario.tracks):
            if (
                track.track_id == scenario.focal_track_id
                and track.category == TrackCategory.FOCAL_TRACK
            ):
                focal_idx = idx
            elif track.track_id == "AV":
                av_idx = idx
            elif track.category == TrackCategory.SCORED_TRACK:
                scored_idcs.append(idx)
            elif track.category == TrackCategory.UNSCORED_TRACK:
                unscored_idcs.append(idx)
            elif track.category == TrackCategory.TRACK_FRAGMENT:
                fragment_idcs.append(idx)

        # Enforce existence of AV and Focal
        if av_idx is None or focal_idx is None:
            raise ValueError("Missing AV or Focal track.")

        # Sorting order: Focal -> AV -> Scored -> Unscored -> Fragments
        sorted_idcs = [focal_idx, av_idx] + scored_idcs + unscored_idcs + fragment_idcs
        sorted_cat = (
            ["focal", "av"]
            + ["score"] * len(scored_idcs)
            + ["unscore"] * len(unscored_idcs)
            + ["frag"] * len(fragment_idcs)
        )
        sorted_tid = [scenario.tracks[idx].track_id for idx in sorted_idcs]

        # Timestamps
        if self.split == "test":
            # Test set only has obs
            ts_frame_indices = np.arange(0, self.history_steps)
        else:
            ts_frame_indices = np.arange(0, self.seq_len)

        obs_end_frame = self.history_steps - 1

        # Pre-fetch map points for distance filtering
        map_pts = []
        for lane_id, lane in static_map.vector_lane_segments.items():
            map_pts.append(static_map.get_lane_segment_centerline(lane_id)[:, 0:2])
        map_pts = np.concatenate(map_pts, axis=0)
        map_pts = np.expand_dims(map_pts, axis=0)  # [1, N_map, 2]

        # Containers
        trajs_pos_list, trajs_ang_list, trajs_vel_list = [], [], []
        trajs_type_list, has_flags_list = [], []
        final_tids, final_cats = [], []

        # Reference Frame Initialization (Based on Focal Agent at last obs frame)
        # Will be set in the first iteration (k=0 is focal)
        orig_seq, rot_seq, theta_seq = None, None, None

        for k, track_idx in enumerate(sorted_idcs):
            track = scenario.tracks[track_idx]

            # Extract raw states
            timestamps_iter = np.array(
                [x.timestep for x in track.object_states], dtype=np.int16
            )
            pos_iter = np.array([list(x.position) for x in track.object_states])
            ang_iter = np.array([x.heading for x in track.object_states])
            vel_iter = np.array([list(x.velocity) for x in track.object_states])

            # Filter: Skip if strictly future or not present at observation time
            if (
                timestamps_iter[0] > obs_end_frame
                or obs_end_frame not in timestamps_iter
            ):
                continue

            # Define Coordinate System based on Focal Agent (k=0)
            if k == 0:
                curr_orig = pos_iter[timestamps_iter == obs_end_frame][0]
                curr_theta = ang_iter[timestamps_iter == obs_end_frame][0]
                curr_rot = np.array(
                    [
                        [np.cos(curr_theta), -np.sin(curr_theta)],
                        [np.sin(curr_theta), np.cos(curr_theta)],
                    ]
                )
                orig_seq, rot_seq, theta_seq = curr_orig, curr_rot, curr_theta

            # Distance Filter (Skip irrelevant background actors)
            # Only check for non-scored/non-focal/non-av tracks
            if sorted_cat[k] in ["unscore", "frag"]:
                # Check distance of observed points to map
                obs_mask = timestamps_iter <= obs_end_frame
                traj_obs_pts = np.expand_dims(
                    pos_iter[obs_mask], axis=1
                )  # [T_obs, 1, 2]
                dist = np.linalg.norm(traj_obs_pts - map_pts, axis=-1)
                if np.min(dist) > self.min_dist_threshold:
                    continue

            # --- Normalization (To Scene Centric) ---
            # Transform position and velocity using the focal agent's frame
            pos_norm = (pos_iter - orig_seq).dot(rot_seq)
            ang_norm = ang_iter - theta_seq
            vel_norm = vel_iter.dot(rot_seq)

            # --- Padding ---
            # Create full length arrays filled with nan/zeros
            total_frames = len(ts_frame_indices)

            # Flags (1 if present)
            has_flag = np.zeros(total_frames, dtype=np.int16)
            valid_indices = np.isin(timestamps_iter, ts_frame_indices)
            # Map valid timestamps to indices in the fixed array
            mapped_indices = timestamps_iter[valid_indices]
            has_flag[mapped_indices] = 1

            # Object Type One-Hot (7 classes)
            # Vehicle, Pedestrian, Motorcyclist, Cyclist, Bus, Unknown, Static
            obj_type_vec = np.zeros(7)
            type_map = {
                ObjectType.VEHICLE: 0,
                ObjectType.PEDESTRIAN: 1,
                ObjectType.MOTORCYCLIST: 2,
                ObjectType.CYCLIST: 3,
                ObjectType.BUS: 4,
                ObjectType.UNKNOWN: 5,
            }
            type_idx = type_map.get(track.object_type, 6)  # Default to 6 (Static)
            obj_type_vec[type_idx] = 1

            traj_type = np.zeros((total_frames, 7))
            traj_type[mapped_indices] = obj_type_vec

            # Position & Angle (Nearest Neighbor Padding)
            pos_pad = np.full((total_frames, 2), np.nan)
            pos_pad[mapped_indices] = pos_norm[valid_indices]
            pos_pad = self._padding_nearest_neighbor(pos_pad)

            ang_pad = np.full(total_frames, np.nan)
            ang_pad[mapped_indices] = ang_norm[valid_indices]
            ang_pad = self._padding_nearest_neighbor(ang_pad)

            # Velocity (Zero Padding)
            vel_pad = np.zeros((total_frames, 2))
            vel_pad[mapped_indices] = vel_norm[valid_indices]

            # Append
            trajs_pos_list.append(pos_pad)
            trajs_ang_list.append(ang_pad)
            trajs_vel_list.append(vel_pad)
            trajs_type_list.append(traj_type)
            has_flags_list.append(has_flag)
            final_tids.append(sorted_tid[k])
            final_cats.append(sorted_cat[k])

        # Stack into arrays
        return (
            np.array(trajs_pos_list, dtype=np.float32),
            np.array(trajs_ang_list, dtype=np.float32),
            np.array(trajs_vel_list, dtype=np.float32),
            np.array(trajs_type_list, dtype=np.int16),
            np.array(has_flags_list, dtype=np.int16),
            final_tids,
            final_cats,
            orig_seq,
            rot_seq,
        )

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

    def _extract_lane_graph(
        self, orig: np.ndarray, rot: np.ndarray, static_map: ArgoverseStaticMap
    ):
        """
        Discretizes lane segments and extracts topological features.
        """
        node_ctrs, node_vecs = [], []
        lane_ctrs, lane_vecs = [], []
        # Features
        lane_type, intersect = [], []
        cross_left, cross_right = [], []
        left_nb, right_nb = [], []

        for lane_id, lane in static_map.vector_lane_segments.items():
            # Get Centerline
            cl_raw = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]
            cl_ls = LineString(cl_raw)

            # Determine number of sub-segments
            num_segs = max(int(np.floor(cl_ls.length / self.LANE_SEG_LENGTH)), 1)
            ds = cl_ls.length / num_segs

            for i in range(num_segs):
                # Interpolate points for this sub-segment
                s_lb = i * ds
                s_ub = (i + 1) * ds
                num_sub_nodes = self.LANE_NODES_COUNT

                cl_pts = [
                    cl_ls.interpolate(s)
                    for s in np.linspace(s_lb, s_ub, num_sub_nodes + 1)
                ]
                ctrln = np.array(LineString(cl_pts).coords)  # [N+1, 2]

                # Transform to local scene frame
                ctrln = (ctrln - orig).dot(rot)

                # Calculate Anchor for this segment
                anch_pos = np.mean(ctrln, axis=0)
                anch_vec = ctrln[-1] - ctrln[0]
                norm = np.linalg.norm(anch_vec)
                anch_vec = anch_vec / (norm + 1e-6)
                anch_rot = np.array(
                    [[anch_vec[0], -anch_vec[1]], [anch_vec[1], anch_vec[0]]]
                )

                lane_ctrs.append(anch_pos)
                lane_vecs.append(anch_vec)

                # Transform nodes to instance frame (relative to segment anchor)
                ctrln_local = (ctrln - anch_pos).dot(anch_rot)

                # Nodes (midpoints) and vectors (tangents)
                ctrs = (ctrln_local[:-1] + ctrln_local[1:]) / 2.0
                vecs = ctrln_local[1:] - ctrln_local[:-1]

                node_ctrs.append(ctrs.astype(np.float32))
                node_vecs.append(vecs.astype(np.float32))

                # --- Extract Attributes ---

                # Lane Type (Vehicle, Bike, Bus)
                l_type = np.zeros(3)
                if lane.lane_type == LaneType.VEHICLE:
                    l_type[0] = 1
                elif lane.lane_type == LaneType.BIKE:
                    l_type[1] = 1
                elif lane.lane_type == LaneType.BUS:
                    l_type[2] = 1
                lane_type.append(np.tile(l_type, (num_sub_nodes, 1)))

                # Intersection
                is_inter = 1.0 if lane.is_intersection else 0.0
                intersect.append(np.full(num_sub_nodes, is_inter, dtype=np.float32))

                # Crossing Markers (Left/Right)
                def get_mark_type(mark_type):
                    # Returns [Crossable, Not-Crossable, Unknown]
                    res = np.zeros(3)
                    if mark_type in [
                        LaneMarkType.DASH_SOLID_YELLOW,
                        LaneMarkType.DASH_SOLID_WHITE,
                        LaneMarkType.DASHED_WHITE,
                        LaneMarkType.DASHED_YELLOW,
                        LaneMarkType.DOUBLE_DASH_YELLOW,
                        LaneMarkType.DOUBLE_DASH_WHITE,
                    ]:
                        res[0] = 1
                    elif mark_type in [
                        LaneMarkType.DOUBLE_SOLID_YELLOW,
                        LaneMarkType.DOUBLE_SOLID_WHITE,
                        LaneMarkType.SOLID_YELLOW,
                        LaneMarkType.SOLID_WHITE,
                        LaneMarkType.SOLID_DASH_WHITE,
                        LaneMarkType.SOLID_DASH_YELLOW,
                        LaneMarkType.SOLID_BLUE,
                    ]:
                        res[1] = 1
                    else:
                        res[2] = 1
                    return res

                cross_left.append(
                    np.tile(get_mark_type(lane.left_mark_type), (num_sub_nodes, 1))
                )
                cross_right.append(
                    np.tile(get_mark_type(lane.right_mark_type), (num_sub_nodes, 1))
                )

                # Neighbors
                left_nb.append(
                    np.ones(num_sub_nodes)
                    if lane.left_neighbor_id
                    else np.zeros(num_sub_nodes)
                )
                right_nb.append(
                    np.ones(num_sub_nodes)
                    if lane.right_neighbor_id
                    else np.zeros(num_sub_nodes)
                )

        # Stack Dictionary
        graph = {
            "node_ctrs": np.stack(node_ctrs).astype(np.float32),
            "node_vecs": np.stack(node_vecs).astype(np.float32),
            "lane_ctrs": np.array(lane_ctrs).astype(np.float32),
            "lane_vecs": np.array(lane_vecs).astype(np.float32),
            "lane_type": np.stack(lane_type).astype(np.int16),
            "intersect": np.stack(intersect).astype(np.int16),
            "cross_left": np.stack(cross_left).astype(np.int16),
            "cross_right": np.stack(cross_right).astype(np.int16),
            "left": np.stack(left_nb).astype(np.int16),
            "right": np.stack(right_nb).astype(np.int16),
        }

        graph["num_nodes"] = graph["node_ctrs"].shape[0] * graph["node_ctrs"].shape[1]
        graph["num_lanes"] = graph["lane_ctrs"].shape[0]

        return graph

    # =========================================================================
    # RUNTIME DATA TRANSFORMATION
    # =========================================================================

    def _apply_augmentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Applies random vertical flip with 30% probability."""
        # Deep copy needed if we are modifying in place and caching is in memory
        # But here we load fresh from disk each time, so partial copy is okay.

        # Only flip if enabled and coin toss passes
        if not (self.augmentation and random.random() < 0.3):
            return data

        # Flip Y coordinates
        # Trajectories
        data["TRAJS"]["trajs_ctrs"][..., 1] *= -1
        data["TRAJS"]["trajs_vecs"][..., 1] *= -1
        data["TRAJS"]["trajs_pos"][..., 1] *= -1
        data["TRAJS"]["trajs_ang"] *= -1  # Angle flip
        data["TRAJS"]["trajs_vel"][..., 1] *= -1

        # Lane Graph
        data["LANE_GRAPH"]["lane_ctrs"][..., 1] *= -1
        data["LANE_GRAPH"]["lane_vecs"][..., 1] *= -1
        data["LANE_GRAPH"]["node_ctrs"][..., 1] *= -1
        data["LANE_GRAPH"]["node_vecs"][..., 1] *= -1

        # Swap Left/Right Neighbor attributes
        lg = data["LANE_GRAPH"]
        lg["left"], lg["right"] = lg["right"], lg["left"]

        return data

    def _get_rpe(self, ctrs, vecs, radius=100.0):
        """
        Relative Positional Encoding calculation.
        """
        # distance encoding
        d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
        d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
        pos_rpe = d_pos.unsqueeze(0)

        # angle diff
        def get_cos(v1, v2):
            return (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-10)

        def get_sin(v1, v2):
            return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (
                v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-10
            )

        v_edge = vecs.unsqueeze(0)  # [1, N, 2]
        v_edge_t = vecs.unsqueeze(1)  # [N, 1, 2]
        v_rel = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)  # [N, N, 2]

        cos_a1 = get_cos(v_edge, v_edge_t)
        sin_a1 = get_sin(v_edge, v_edge_t)
        cos_a2 = get_cos(v_edge, v_rel)
        sin_a2 = get_sin(v_edge, v_rel)

        ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
        rpe = torch.cat([ang_rpe, pos_rpe], dim=0)
        return rpe

    # =========================================================================
    # BATCH COLLATION
    # =========================================================================

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        """
        Custom collate function to handle variable numbers of actors/lanes.
        """
        batch = from_numpy(batch)
        data = dict()
        data["BATCH_SIZE"] = len(batch)

        # 1. Standard keys aggregation
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]

        # 2. Flatten Actors (for GNN processing)
        agent_feats, agent_masks = self._actor_gather(data["BATCH_SIZE"], data["TRAJS"])

        # 3. Flatten Lanes
        lane_feats, lane_masks = self._lane_gather(
            data["BATCH_SIZE"], data["LANE_GRAPH"]
        )

        return {
            "agent_feats": agent_feats,  # [B, Nmax, D, T]
            "agent_masks": agent_masks,  # [B, Nmax]
            "lane_feats": lane_feats,  # [B, Nmax, 10, 16]
            "lane_masks": lane_masks,  # [B, Nmax]
            "RPE": data["RPE"],
            "TRAJS_POS_FUT": [x["TRAJS_POS_FUT"] for x in data["TRAJS"]],
            "PAD_FUT": [x["PAD_FUT"] for x in data["TRAJS"]],
            "TRAIN_MASK": [x["TRAIN_MASK"] for x in data["TRAJS"]],
            "TRAJS_ANG_FUT": [x["TRAJS_ANG_FUT"] for x in data["TRAJS"]],
            "YAW_LOSS_MASK": [x["YAW_LOSS_MASK"] for x in data["TRAJS"]],
        }

    def _actor_gather(self, batch_size, trajs):
        """Flattens actor features from a batch into a single tensor."""
        num_agents = [len(x["TRAJS_CTRS"]) for x in trajs]
        max_agents = max(num_agents)

        agent_feats = torch.zeros(
            [batch_size, max_agents, 14, self.history_steps - self.truncate_steps]
        )  # truncate first 2 steps
        agent_masks = torch.zeros([batch_size, max_agents], dtype=torch.bool)
        for i in range(batch_size):
            traj_pos = trajs[i]["TRAJS_POS_OBS"]
            traj_disp = torch.zeros_like(traj_pos)
            traj_disp[:, 1:, :] = traj_pos[:, 1:, :] - traj_pos[:, :-1, :]

            # Concatenate features: [Disp, Ang, Vel, Type, Pad]
            feat = torch.cat(
                [
                    traj_disp,
                    trajs[i]["TRAJS_ANG_OBS"],
                    trajs[i]["TRAJS_VEL_OBS"],
                    trajs[i]["TRAJS_TYPE"],
                    trajs[i]["PAD_OBS"].unsqueeze(-1),
                ],
                dim=-1,
            )
            # [N, T, D] -> [N, D, T], truncate first 2 steps
            feat = feat.transpose(1, 2)[..., 2:]
            agent_feats[i, : num_agents[i]] = feat
            agent_masks[i][: num_agents[i]] = True

        return agent_feats, agent_masks

        # TODO
        # Create indices mapping batch items to actors
        actor_idcs = []
        count = 0
        for i in range(batch_size):
            actor_idcs.append(torch.arange(count, count + num_agents[i]))
            count += num_agents[i]

        return actors, actor_idcs

    def _lane_gather(self, batch_size, graphs):
        """Flattens lane graph features from a batch."""
        lane_idcs = []
        lane_count = 0

        num_lanes = [g["num_lanes"] for g in graphs]
        max_lanes = max(num_lanes)

        lane_feats = torch.zeros([batch_size, max_lanes, 10, 16])
        lane_masks = torch.zeros([batch_size, max_lanes], dtype=torch.bool)

        # for i in range(batch_size):

        for i in range(batch_size):
            lane_idcs.append(
                torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            )
            lane_count += graphs[i]["num_lanes"]

            feat = torch.concat(
                [
                    graphs[i]["node_ctrs"],
                    graphs[i]["node_vecs"],
                    graphs[i]["intersect"][..., None],
                    graphs[i]["lane_type"],
                    graphs[i]["cross_left"],
                    graphs[i]["cross_right"],
                    graphs[i]["left"][..., None],
                    graphs[i]["right"][..., None],
                ],
                dim=-1,
            )

            lane_feats[i, : num_lanes[i]] = feat
            lane_masks[i, : num_lanes[i]] = True

        return lane_feats, lane_masks

        # Aggregate fields
        agg_graph = {}
        for key in [
            "node_ctrs",
            "node_vecs",
            "intersect",
            "lane_type",
            "cross_left",
            "cross_right",
            "left",
            "right",
        ]:
            agg_graph[key] = torch.cat([x[key] for x in graphs], 0)

        # Concatenate into single feature vector per lane segment node
        lanes = torch.cat(
            [
                agg_graph["node_ctrs"],
                agg_graph["node_vecs"],
                agg_graph["intersect"].unsqueeze(2),
                agg_graph["lane_type"],
                agg_graph["cross_left"],
                agg_graph["cross_right"],
                agg_graph["left"].unsqueeze(2),
                agg_graph["right"].unsqueeze(2),
            ],
            dim=-1,
        )

        return lanes, lane_idcs
