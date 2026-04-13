from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except (
    Exception
):  # pragma: no cover - allows schema-only use when torch runtime is unavailable

    class Dataset:  # type: ignore[override]
        pass


DatasetSource = Literal["av2", "waymo"]


@dataclass(frozen=True)
class ScenarioReference:
    source: DatasetSource
    split: str
    path: str
    scenario_id: str | None = None
    record_index: int | None = None


@dataclass
class MotionTrack:
    track_id: str
    object_type: str
    category: str | None
    positions: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    sizes: np.ndarray
    valid_mask: np.ndarray
    observed_mask: np.ndarray
    is_ego: bool = False
    is_focal: bool = False
    is_prediction_target: bool = False
    is_object_of_interest: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class MotionLaneSegment:
    lane_id: str
    centerline: np.ndarray
    lane_type: str | None = None
    left_boundary: np.ndarray | None = None
    right_boundary: np.ndarray | None = None
    left_mark_type: str | None = None
    right_mark_type: str | None = None
    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()
    left_neighbor_ids: tuple[str, ...] = ()
    right_neighbor_ids: tuple[str, ...] = ()
    left_boundary_feature_ids: tuple[str, ...] = ()
    right_boundary_feature_ids: tuple[str, ...] = ()
    is_intersection: bool | None = None
    speed_limit_mph: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionPolylineFeature:
    feature_id: str
    feature_type: str
    points: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionPolygonFeature:
    feature_id: str
    feature_type: str
    polygon: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionPointFeature:
    feature_id: str
    feature_type: str
    point: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionScenario:
    scenario_id: str
    source: DatasetSource
    split: str
    timestamps_seconds: np.ndarray
    current_time_index: int | None
    tracks: list[MotionTrack]
    lane_segments: list[MotionLaneSegment]
    road_lines: list[MotionPolylineFeature] = field(default_factory=list)
    road_edges: list[MotionPolylineFeature] = field(default_factory=list)
    crosswalks: list[MotionPolygonFeature] = field(default_factory=list)
    speed_bumps: list[MotionPolygonFeature] = field(default_factory=list)
    driveways: list[MotionPolygonFeature] = field(default_factory=list)
    drivable_areas: list[MotionPolygonFeature] = field(default_factory=list)
    stop_signs: list[MotionPointFeature] = field(default_factory=list)
    city_name: str | None = None
    focal_track_id: str | None = None
    sdc_track_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    @property
    def num_steps(self) -> int:
        return int(self.timestamps_seconds.shape[0])


class MotionDataset(Dataset):
    """Canonical ragged motion-forecasting dataset over raw AV2 and Waymo scenarios.

    The core design decision here is to preserve variable-length scenario structure:
    - scenario time axes stay per-scenario instead of being globally padded
    - lane centerlines and boundaries keep their native polyline lengths
    - downstream models are expected to adapt this canonical representation into
      model-specific tensor layouts
    """

    def __init__(
        self,
        scenario_refs: Sequence[ScenarioReference],
        loader: Callable[[ScenarioReference], MotionScenario],
        transform: Callable[[MotionScenario], Any] | None = None,
    ) -> None:
        self.scenario_refs = list(scenario_refs)
        self.scenario_files = self.scenario_refs
        self._loader = loader
        self._transform = transform

    def __len__(self) -> int:
        return len(self.scenario_refs)

    def __getitem__(self, idx: int) -> Any:
        scenario_ref = self.scenario_refs[idx]
        scenario = self._loader(scenario_ref)
        if self._transform is not None:
            return self._transform(scenario)
        return scenario

    @staticmethod
    def collate_fn(batch: list[Any]) -> list[Any]:
        # Ragged scenarios do not have a safe default tensor stack operation.
        return batch

    @classmethod
    def create_from_av2(
        cls,
        data_root: str | Path,
        split: str,
        transform: Callable[[MotionScenario], Any] | None = None,
    ) -> MotionDataset:
        root = Path(data_root).expanduser().resolve()
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"AV2 split directory not found: {split_dir}")

        refs: list[ScenarioReference] = []
        for scenario_dir in sorted(
            path for path in split_dir.iterdir() if path.is_dir()
        ):
            scenario_id = scenario_dir.name
            refs.append(
                ScenarioReference(
                    source="av2",
                    split=split,
                    path=str(scenario_dir),
                    scenario_id=scenario_id,
                )
            )
        return cls(refs, loader=_load_av2_scenario, transform=transform)

    @classmethod
    def create_from_waymo(
        cls,
        data_root: str | Path,
        split: str | None = None,
        index_file: str | Path | None = None,
        transform: Callable[[MotionScenario], Any] | None = None,
        max_scenarios: int | None = None,
    ) -> MotionDataset:
        split = "training" if split == "train" else split
        split = "validation" if split == "val" else split
        split = "testing" if split == "test" else split

        root = Path(data_root).expanduser().resolve()
        search_root = root / split if split is not None else root
        if not search_root.exists():
            raise FileNotFoundError(f"Waymo search root not found: {search_root}")

        if index_file is None:
            index_file = root / ".motion_dataset_index" / f"waymo_{split or 'all'}.json"
        index_path = Path(index_file)

        refs: list[ScenarioReference]
        if index_path.exists():
            refs = _load_waymo_index(index_path)
        else:
            tfrecord_paths = sorted(search_root.rglob("*.tfrecord*"))
            if not tfrecord_paths:
                raise FileNotFoundError(
                    f"No Waymo TFRecord files matching {'*.tfrecord*'!r} found under {search_root}"
                )
            refs = _build_waymo_index(
                tfrecord_paths, split=split or "all", max_scenarios=max_scenarios
            )
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                json.dumps([ref.__dict__ for ref in refs], indent=2),
                encoding="utf-8",
            )

        if max_scenarios is not None:
            refs = refs[:max_scenarios]
        return cls(refs, loader=_load_waymo_scenario, transform=transform)


def _load_waymo_index(index_path: Path) -> list[ScenarioReference]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return [ScenarioReference(**item) for item in payload]


def _build_waymo_index(
    tfrecord_paths: Sequence[Path],
    split: str,
    max_scenarios: int | None = None,
) -> list[ScenarioReference]:
    tf, scenario_pb2 = _require_waymo_scenario_modules()

    refs: list[ScenarioReference] = []
    for tfrecord_path in tfrecord_paths:
        dataset = tf.data.TFRecordDataset([str(tfrecord_path)], compression_type="")
        for record_index, raw_record in enumerate(dataset.as_numpy_iterator()):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_record)
            refs.append(
                ScenarioReference(
                    source="waymo",
                    split=split,
                    path=str(tfrecord_path),
                    scenario_id=str(scenario.scenario_id),
                    record_index=record_index,
                )
            )
            if max_scenarios is not None and len(refs) >= max_scenarios:
                return refs
    return refs


def _load_av2_scenario(ref: ScenarioReference) -> MotionScenario:
    scenario_serialization, ArgoverseStaticMap, TrackCategory = _require_av2_modules()

    scenario_dir = Path(ref.path)
    scenario_id = ref.scenario_id or scenario_dir.name
    parquet_path = scenario_dir / f"scenario_{scenario_id}.parquet"
    map_path = scenario_dir / f"log_map_archive_{scenario_id}.json"
    if not parquet_path.exists():
        raise FileNotFoundError(f"AV2 scenario parquet not found: {parquet_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"AV2 map json not found: {map_path}")

    scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
    static_map = ArgoverseStaticMap.from_json(map_path)

    timestamps_ns = np.asarray(scenario.timestamps_ns, dtype=np.float64)
    timestamps_seconds = timestamps_ns / 1e9

    observed_steps = [
        int(state.timestep)
        for track in scenario.tracks
        for state in track.object_states
        if bool(getattr(state, "observed", False))
    ]
    current_time_index = max(observed_steps) if observed_steps else None

    tracks: list[MotionTrack] = []
    sdc_track_id: str | None = None
    for track in scenario.tracks:
        motion_track = _normalize_av2_track(
            track=track,
            num_steps=len(timestamps_seconds),
            focal_track_id=str(scenario.focal_track_id),
            track_category_enum=TrackCategory,
        )
        if motion_track.is_ego:
            sdc_track_id = motion_track.track_id
        tracks.append(motion_track)

    lane_segments = []
    for lane_id, lane_segment in static_map.vector_lane_segments.items():
        lane_segments.append(
            MotionLaneSegment(
                lane_id=str(lane_id),
                centerline=_to_numpy(static_map.get_lane_segment_centerline(lane_id)),
                lane_type=_enum_name(lane_segment.lane_type),
                left_boundary=_to_numpy(lane_segment.left_lane_boundary.xyz),
                right_boundary=_to_numpy(lane_segment.right_lane_boundary.xyz),
                left_mark_type=_enum_name(lane_segment.left_mark_type),
                right_mark_type=_enum_name(lane_segment.right_mark_type),
                predecessor_ids=tuple(str(item) for item in lane_segment.predecessors),
                successor_ids=tuple(str(item) for item in lane_segment.successors),
                left_neighbor_ids=tuple(
                    [str(lane_segment.left_neighbor_id)]
                    if lane_segment.left_neighbor_id is not None
                    else []
                ),
                right_neighbor_ids=tuple(
                    [str(lane_segment.right_neighbor_id)]
                    if lane_segment.right_neighbor_id is not None
                    else []
                ),
                is_intersection=bool(lane_segment.is_intersection),
            )
        )

    crosswalks = []
    for crossing_id, crossing in static_map.vector_pedestrian_crossings.items():
        polygon = np.concatenate(
            [crossing.edge1.xyz, np.flip(crossing.edge2.xyz, axis=0)],
            axis=0,
        )
        crosswalks.append(
            MotionPolygonFeature(
                feature_id=str(crossing_id),
                feature_type="crosswalk",
                polygon=_to_numpy(polygon),
            )
        )

    drivable_areas = [
        MotionPolygonFeature(
            feature_id=str(area_id),
            feature_type="drivable_area",
            polygon=_to_numpy(drivable_area.xyz),
        )
        for area_id, drivable_area in static_map.vector_drivable_areas.items()
    ]

    return MotionScenario(
        scenario_id=str(scenario.scenario_id),
        source="av2",
        split=ref.split,
        timestamps_seconds=timestamps_seconds.astype(np.float32),
        current_time_index=current_time_index,
        tracks=tracks,
        lane_segments=lane_segments,
        crosswalks=crosswalks,
        drivable_areas=drivable_areas,
        city_name=str(getattr(scenario, "city_name", "")) or None,
        focal_track_id=str(scenario.focal_track_id),
        sdc_track_id=sdc_track_id,
        metadata={
            "scenario_dir": str(scenario_dir),
            "raw_parquet_path": str(parquet_path),
            "raw_map_path": str(map_path),
        },
    )


def _normalize_av2_track(
    track: Any,
    num_steps: int,
    focal_track_id: str,
    track_category_enum: Any,
) -> MotionTrack:
    positions, headings, velocities, sizes, valid_mask, observed_mask = (
        _allocate_track_arrays(num_steps)
    )
    for state in track.object_states:
        timestep = int(state.timestep)
        if timestep < 0 or timestep >= num_steps:
            continue
        positions[timestep, :2] = np.asarray(state.position, dtype=np.float32)
        positions[timestep, 2] = 0.0
        headings[timestep] = float(state.heading)
        velocities[timestep, :2] = np.asarray(state.velocity, dtype=np.float32)
        velocities[timestep, 2] = 0.0
        valid_mask[timestep] = True
        observed_mask[timestep] = bool(state.observed)

    category = _enum_name(track.category)
    is_focal = (
        str(track.track_id) == focal_track_id
        and category == track_category_enum.FOCAL_TRACK.name.lower()
    )
    is_prediction_target = category in {
        track_category_enum.FOCAL_TRACK.name.lower(),
        track_category_enum.SCORED_TRACK.name.lower(),
    }

    return MotionTrack(
        track_id=str(track.track_id),
        object_type=_enum_value_or_name(track.object_type),
        category=category,
        positions=positions,
        headings=headings,
        velocities=velocities,
        sizes=sizes,
        valid_mask=valid_mask,
        observed_mask=observed_mask,
        is_ego=str(track.track_id) == "AV",
        is_focal=is_focal,
        is_prediction_target=is_prediction_target,
        metadata={"source_dataset": "av2"},
    )


def _load_waymo_scenario(ref: ScenarioReference) -> MotionScenario:
    tf, scenario_pb2 = _require_waymo_scenario_modules()
    map_pb2 = _require_waymo_map_module()

    if ref.record_index is None:
        raise ValueError("Waymo ScenarioReference.record_index is required")

    dataset = tf.data.TFRecordDataset([ref.path], compression_type="")
    raw_record = next(dataset.skip(ref.record_index).take(1).as_numpy_iterator(), None)
    if raw_record is None:
        raise IndexError(f"Waymo record {ref.record_index} not found in {ref.path}")

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw_record)

    timestamps_seconds = np.asarray(scenario.timestamps_seconds, dtype=np.float32)
    current_time_index = int(scenario.current_time_index)
    sdc_track_id = None
    if 0 <= scenario.sdc_track_index < len(scenario.tracks):
        sdc_track_id = str(scenario.tracks[scenario.sdc_track_index].id)

    prediction_indices = {
        int(item.track_index): int(item.difficulty)
        for item in scenario.tracks_to_predict
    }
    objects_of_interest = {str(item) for item in scenario.objects_of_interest}

    tracks: list[MotionTrack] = []
    for track_index, track in enumerate(scenario.tracks):
        tracks.append(
            _normalize_waymo_track(
                track=track,
                track_index=track_index,
                current_time_index=current_time_index,
                prediction_indices=prediction_indices,
                objects_of_interest=objects_of_interest,
                is_sdc=track_index == int(scenario.sdc_track_index),
            )
        )

    lane_segments: list[MotionLaneSegment] = []
    road_lines: list[MotionPolylineFeature] = []
    road_edges: list[MotionPolylineFeature] = []
    crosswalks: list[MotionPolygonFeature] = []
    speed_bumps: list[MotionPolygonFeature] = []
    driveways: list[MotionPolygonFeature] = []
    stop_signs: list[MotionPointFeature] = []

    for feature in scenario.map_features:
        feature_id = str(feature.id)
        feature_kind = feature.WhichOneof("feature_data")
        if feature_kind == "lane":
            lane = feature.lane
            lane_segments.append(
                MotionLaneSegment(
                    lane_id=feature_id,
                    centerline=_waymo_map_points_to_numpy(lane.polyline),
                    lane_type=_proto_field_enum_name(lane, "type"),
                    predecessor_ids=tuple(str(item) for item in lane.entry_lanes),
                    successor_ids=tuple(str(item) for item in lane.exit_lanes),
                    left_neighbor_ids=tuple(
                        str(item.feature_id) for item in lane.left_neighbors
                    ),
                    right_neighbor_ids=tuple(
                        str(item.feature_id) for item in lane.right_neighbors
                    ),
                    left_boundary_feature_ids=tuple(
                        str(item.boundary_feature_id) for item in lane.left_boundaries
                    ),
                    right_boundary_feature_ids=tuple(
                        str(item.boundary_feature_id) for item in lane.right_boundaries
                    ),
                    is_intersection=None,
                    speed_limit_mph=(
                        float(lane.speed_limit_mph)
                        if float(lane.speed_limit_mph) > 0
                        else None
                    ),
                    metadata={
                        "interpolating": bool(lane.interpolating),
                        "left_boundary_types": [
                            _proto_field_enum_name(item, "boundary_type")
                            for item in lane.left_boundaries
                        ],
                        "right_boundary_types": [
                            _proto_field_enum_name(item, "boundary_type")
                            for item in lane.right_boundaries
                        ],
                    },
                )
            )
        elif feature_kind == "road_line":
            road_lines.append(
                MotionPolylineFeature(
                    feature_id=feature_id,
                    feature_type="road_line",
                    points=_waymo_map_points_to_numpy(feature.road_line.polyline),
                    metadata={
                        "road_line_type": _proto_field_enum_name(
                            feature.road_line, "type"
                        )
                    },
                )
            )
        elif feature_kind == "road_edge":
            road_edges.append(
                MotionPolylineFeature(
                    feature_id=feature_id,
                    feature_type="road_edge",
                    points=_waymo_map_points_to_numpy(feature.road_edge.polyline),
                    metadata={
                        "road_edge_type": _proto_field_enum_name(
                            feature.road_edge, "type"
                        )
                    },
                )
            )
        elif feature_kind == "crosswalk":
            crosswalks.append(
                MotionPolygonFeature(
                    feature_id=feature_id,
                    feature_type="crosswalk",
                    polygon=_waymo_map_points_to_numpy(feature.crosswalk.polygon),
                )
            )
        elif feature_kind == "speed_bump":
            speed_bumps.append(
                MotionPolygonFeature(
                    feature_id=feature_id,
                    feature_type="speed_bump",
                    polygon=_waymo_map_points_to_numpy(feature.speed_bump.polygon),
                )
            )
        elif feature_kind == "driveway":
            driveways.append(
                MotionPolygonFeature(
                    feature_id=feature_id,
                    feature_type="driveway",
                    polygon=_waymo_map_points_to_numpy(feature.driveway.polygon),
                )
            )
        elif feature_kind == "stop_sign":
            point = feature.stop_sign.position
            stop_signs.append(
                MotionPointFeature(
                    feature_id=feature_id,
                    feature_type="stop_sign",
                    point=np.asarray([point.x, point.y, point.z], dtype=np.float32),
                    metadata={
                        "lane_ids": [str(item) for item in feature.stop_sign.lane]
                    },
                )
            )

    return MotionScenario(
        scenario_id=str(scenario.scenario_id),
        source="waymo",
        split=ref.split,
        timestamps_seconds=timestamps_seconds,
        current_time_index=current_time_index,
        tracks=tracks,
        lane_segments=lane_segments,
        road_lines=road_lines,
        road_edges=road_edges,
        crosswalks=crosswalks,
        speed_bumps=speed_bumps,
        driveways=driveways,
        stop_signs=stop_signs,
        focal_track_id=None,
        sdc_track_id=sdc_track_id,
        metadata={
            "raw_tfrecord_path": ref.path,
            "record_index": ref.record_index,
            "objects_of_interest": sorted(objects_of_interest),
            "tracks_to_predict": {str(k): v for k, v in prediction_indices.items()},
        },
    )


def _normalize_waymo_track(
    track: Any,
    track_index: int,
    current_time_index: int,
    prediction_indices: dict[int, int],
    objects_of_interest: set[str],
    is_sdc: bool,
) -> MotionTrack:
    num_steps = len(track.states)
    positions, headings, velocities, sizes, valid_mask, observed_mask = (
        _allocate_track_arrays(num_steps)
    )

    for timestep, state in enumerate(track.states):
        if not bool(state.valid):
            continue
        positions[timestep] = np.asarray(
            [state.center_x, state.center_y, state.center_z],
            dtype=np.float32,
        )
        headings[timestep] = float(state.heading)
        velocities[timestep, :2] = np.asarray(
            [state.velocity_x, state.velocity_y],
            dtype=np.float32,
        )
        velocities[timestep, 2] = 0.0
        sizes[timestep] = np.asarray(
            [state.length, state.width, state.height],
            dtype=np.float32,
        )
        valid_mask[timestep] = True
        observed_mask[timestep] = timestep <= current_time_index

    difficulty = prediction_indices.get(track_index)
    track_id = str(track.id)
    return MotionTrack(
        track_id=track_id,
        object_type=_proto_field_enum_name(track, "object_type"),
        category=None,
        positions=positions,
        headings=headings,
        velocities=velocities,
        sizes=sizes,
        valid_mask=valid_mask,
        observed_mask=observed_mask & valid_mask,
        is_ego=is_sdc,
        is_focal=False,
        is_prediction_target=track_index in prediction_indices,
        is_object_of_interest=track_id in objects_of_interest,
        metadata={
            "source_dataset": "waymo",
            "track_index": track_index,
            "prediction_difficulty": difficulty,
        },
    )


def _allocate_track_arrays(
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.full((num_steps, 3), np.nan, dtype=np.float32)
    headings = np.full((num_steps,), np.nan, dtype=np.float32)
    velocities = np.full((num_steps, 3), np.nan, dtype=np.float32)
    sizes = np.full((num_steps, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros((num_steps,), dtype=bool)
    observed_mask = np.zeros((num_steps,), dtype=bool)
    return positions, headings, velocities, sizes, valid_mask, observed_mask


def _to_numpy(array_like: Any) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float32)


def _waymo_map_points_to_numpy(points: Any) -> np.ndarray:
    return np.asarray(
        [[point.x, point.y, point.z] for point in points], dtype=np.float32
    )


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "name"):
        return str(value.name).lower()
    if hasattr(value, "value"):
        return str(value.value).lower()
    return str(value).lower()


def _enum_value_or_name(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value).lower()
    if hasattr(value, "name"):
        return str(value.name).lower()
    return str(value).lower()


def _proto_field_enum_name(message: Any, field_name: str) -> str:
    field = message.DESCRIPTOR.fields_by_name[field_name]
    value = int(getattr(message, field_name))
    enum_descriptor = field.enum_type
    if enum_descriptor is None:
        return str(value).lower()
    enum_value = enum_descriptor.values_by_number.get(value)
    return enum_value.name.lower() if enum_value is not None else str(value).lower()


def _require_av2_modules() -> tuple[Any, Any, Any]:
    try:
        from av2.datasets.motion_forecasting import scenario_serialization
        from av2.datasets.motion_forecasting.data_schema import TrackCategory
        from av2.map.map_api import ArgoverseStaticMap
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "AV2 support requires the `av2` package to be installed in the project environment."
        ) from exc
    return scenario_serialization, ArgoverseStaticMap, TrackCategory


def _require_waymo_scenario_modules() -> tuple[Any, Any]:
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Waymo support requires `tensorflow` and `waymo_open_dataset` in the project environment."
        ) from exc
    return tf, scenario_pb2


def _require_waymo_map_module() -> Any:
    try:
        from waymo_open_dataset.protos import map_pb2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Waymo map parsing requires `waymo_open_dataset` in the project environment."
        ) from exc
    return map_pb2


__all__ = [
    "MotionDataset",
    "MotionLaneSegment",
    "MotionPointFeature",
    "MotionPolygonFeature",
    "MotionPolylineFeature",
    "MotionScenario",
    "MotionTrack",
    "ScenarioReference",
]
