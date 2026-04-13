from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.lane_segment import LaneType

_AGENT_TYPE_MAP = {
    ObjectType.VEHICLE: 0,
    ObjectType.PEDESTRIAN: 1,
    ObjectType.MOTORCYCLIST: 2,
    ObjectType.CYCLIST: 3,
    ObjectType.BUS: 4,
    ObjectType.UNKNOWN: 5,
}

_LANE_TYPE_MAP = {
    LaneType.VEHICLE: 0,
    LaneType.BIKE: 1,
    LaneType.BUS: 2,
}
