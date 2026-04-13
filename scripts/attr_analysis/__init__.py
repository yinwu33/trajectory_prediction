from __future__ import annotations

from .agent_motion import AgentMotionAnalysis
from .base import AnalysisSaveResult, BaseAttrAnalysis
from .interaction import InteractionAnalysis
from .map_context import MapContextAnalysis
from .runner import AnalysisRunResult, run_standard_dataset_analyses
from .scenario_metadata import ScenarioMetadataAnalysis


def build_default_analyses() -> list[BaseAttrAnalysis]:
    return [
        ScenarioMetadataAnalysis(),
        AgentMotionAnalysis(),
        InteractionAnalysis(),
        MapContextAnalysis(),
    ]


__all__ = [
    "AgentMotionAnalysis",
    "AnalysisRunResult",
    "AnalysisSaveResult",
    "BaseAttrAnalysis",
    "InteractionAnalysis",
    "MapContextAnalysis",
    "ScenarioMetadataAnalysis",
    "build_default_analyses",
    "run_standard_dataset_analyses",
]
