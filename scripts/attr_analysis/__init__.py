from __future__ import annotations

from .base import AnalysisSaveResult, BaseAttrAnalysis
from .map_elements import MapElementsAnalysis
from .vehicle_density import VehicleDensityAnalysis
from .vehicle_motion import VehicleMotionAnalysis


def build_analyses() -> list[BaseAttrAnalysis]:
    return [
        VehicleMotionAnalysis(),
        VehicleDensityAnalysis(),
        MapElementsAnalysis(),
    ]


__all__ = [
    "AnalysisSaveResult",
    "BaseAttrAnalysis",
    "MapElementsAnalysis",
    "VehicleDensityAnalysis",
    "VehicleMotionAnalysis",
    "build_analyses",
]
