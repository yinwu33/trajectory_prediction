from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from datasets.standardization import StandardizationConfig

from .context import AnalysisContext
from .utils import json_ready


@dataclass
class AnalysisSaveResult:
    csv_path: Path
    parquet_path: Path
    summary_path: Path
    extra_paths: list[Path] = field(default_factory=list)


class BaseAttrAnalysis(ABC):
    name: str

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> dict[str, Any]:
        raise NotImplementedError

    def build_summary(
        self,
        results: pd.DataFrame,
        *,
        dataset_name: str,
        split: str,
        config: StandardizationConfig,
    ) -> dict[str, Any]:
        return {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_scenarios": int(len(results)),
            "columns": list(results.columns),
        }

    def save_plots(self, results: pd.DataFrame, *, output_dir: Path, file_prefix: str) -> list[Path]:
        return []

    def save(
        self,
        results: Sequence[dict[str, Any]],
        output_dir: Path,
        dataset_name: str,
        split: str,
        *,
        config: StandardizationConfig,
        include_plots: bool,
    ) -> AnalysisSaveResult:
        results_df = pd.DataFrame(results)
        if "scenario_id" in results_df.columns:
            results_df = results_df.sort_values("scenario_id").reset_index(drop=True)

        file_prefix = f"{dataset_name}_{split}_{self.name}"
        csv_path = output_dir / f"{file_prefix}.csv"
        parquet_path = output_dir / f"{file_prefix}.parquet"
        summary_path = output_dir / f"{file_prefix}_summary.json"

        results_df.to_csv(csv_path, index=False)
        results_df.to_parquet(parquet_path, index=False)

        summary = self.build_summary(
            results_df,
            dataset_name=dataset_name,
            split=split,
            config=config,
        )
        summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

        extra_paths = self.save_plots(results_df, output_dir=output_dir, file_prefix=file_prefix) if include_plots else []
        return AnalysisSaveResult(
            csv_path=csv_path,
            parquet_path=parquet_path,
            summary_path=summary_path,
            extra_paths=extra_paths,
        )
