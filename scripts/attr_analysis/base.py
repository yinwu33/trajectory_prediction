from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import json_ready


@dataclass(frozen=True)
class AnalysisSaveResult:
    csv_path: Path
    parquet_path: Path
    summary_path: Path


class BaseAttrAnalysis(ABC):
    name: str

    @abstractmethod
    def collect(self, scenario: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def summarize(
        self,
        results: pd.DataFrame,
        *,
        dataset_name: str,
        split: str,
    ) -> dict[str, Any]:
        return {
            "analysis": self.name,
            "dataset": dataset_name,
            "split": split,
            "num_rows": int(len(results)),
            "columns": list(results.columns),
        }

    def save(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        output_dir: Path,
        dataset_name: str,
        split: str,
    ) -> AnalysisSaveResult:
        results = pd.DataFrame(list(rows))
        sort_columns = [
            column
            for column in ("scenario_id", "track_id", "timestep")
            if column in results.columns
        ]
        if sort_columns:
            results = results.sort_values(sort_columns).reset_index(drop=True)

        file_prefix = f"{dataset_name}_{split}_{self.name}"
        csv_path = output_dir / f"{file_prefix}.csv"
        parquet_path = output_dir / f"{file_prefix}.parquet"
        summary_path = output_dir / f"{file_prefix}_summary.json"

        results.to_csv(csv_path, index=False)
        results.to_parquet(parquet_path, index=False)

        summary = self.summarize(
            results,
            dataset_name=dataset_name,
            split=split,
        )
        summary_path.write_text(
            json.dumps(json_ready(summary), indent=2),
            encoding="utf-8",
        )
        return AnalysisSaveResult(
            csv_path=csv_path,
            parquet_path=parquet_path,
            summary_path=summary_path,
        )
