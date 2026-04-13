from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from datasets.standardization import StandardizationConfig

from .base import AnalysisSaveResult, BaseAttrAnalysis
from .context import build_analysis_context


@dataclass
class AnalysisRunResult:
    analysis_outputs: dict[str, AnalysisSaveResult]
    failures_path: Path
    config_path: Path
    total_scenarios: int
    failure_count: int


def flush_analysis_outputs(
    analyses: Sequence[BaseAttrAnalysis],
    results_by_analysis: dict[str, list[dict[str, object]]],
    failures: list[dict[str, str]],
    output_dir: Path,
    dataset_name: str,
    split: str,
    *,
    config: StandardizationConfig,
    include_plots: bool,
) -> AnalysisRunResult:
    analysis_outputs: dict[str, AnalysisSaveResult] = {}
    for analysis in analyses:
        analysis_outputs[analysis.name] = analysis.save(
            results_by_analysis[analysis.name],
            output_dir,
            dataset_name,
            split,
            config=config,
            include_plots=include_plots,
        )

    failures_path = output_dir / f"{dataset_name}_{split}_failures.json"
    config_path = output_dir / f"{dataset_name}_{split}_standardization_config.json"
    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    return AnalysisRunResult(
        analysis_outputs=analysis_outputs,
        failures_path=failures_path,
        config_path=config_path,
        total_scenarios=0,
        failure_count=len(failures),
    )


def run_standard_dataset_analyses(
    dataset,
    analyses: Sequence[BaseAttrAnalysis],
    *,
    output_dir: Path,
    dataset_name: str,
    split: str,
    config: StandardizationConfig,
    limit: int | None = None,
    progress_every: int = 500,
    save_every: int = 500,
    include_plots: bool = True,
) -> AnalysisRunResult:
    total = len(dataset) if limit is None else min(len(dataset), limit)
    results_by_analysis: dict[str, list[dict[str, object]]] = {analysis.name: [] for analysis in analyses}
    failures: list[dict[str, str]] = []

    print(
        f"Analyzing dataset={dataset_name} split={split} scenarios={total} history_steps={config.history_steps} "
        f"future_steps={config.future_steps} coord_frame={config.coord_frame} "
        f"map_range={config.map.range_m}m map_precision={config.map.precision_m}m "
        f"output_dir={output_dir}"
    )

    for index in range(total):
        scenario_ref = dataset.scenario_refs[index]
        scenario_id = scenario_ref.scenario_id or str(index)

        try:
            sample = dataset[index]
            context = build_analysis_context(sample)
        except Exception as exc:  # pragma: no cover - depends on data quality/runtime
            failures.append({"scenario_id": scenario_id, "analysis": "context", "error": str(exc)})
            context = None

        if context is not None:
            for analysis in analyses:
                try:
                    results_by_analysis[analysis.name].append(analysis.analyze(context))
                except Exception as exc:  # pragma: no cover - depends on data quality/runtime
                    failures.append({"scenario_id": scenario_id, "analysis": analysis.name, "error": str(exc)})

        processed = index + 1
        if processed % max(progress_every, 1) == 0 or processed == total:
            row_counts = " ".join(
                f"{analysis.name}={len(results_by_analysis[analysis.name])}" for analysis in analyses
            )
            print(f"processed={processed}/{total} failures={len(failures)} rows[{row_counts}]")

        if (
            processed % max(save_every, 1) == 0
            or processed == total
            or (failures and len(failures) % max(save_every, 1) == 0)
        ):
            checkpoint = flush_analysis_outputs(
                analyses,
                results_by_analysis,
                failures,
                output_dir,
                dataset_name,
                split,
                config=config,
                include_plots=False,
            )
            checkpoint.total_scenarios = total
            print(
                "checkpoint saved: "
                f"processed={processed}/{total} "
                + " ".join(
                    f"{name}={output.csv_path.name}"
                    for name, output in checkpoint.analysis_outputs.items()
                )
                + f" failures={checkpoint.failures_path.name} config={checkpoint.config_path.name}"
            )

    if not any(results_by_analysis.values()):
        failures_path = output_dir / f"{dataset_name}_{split}_failures.json"
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        config_path = output_dir / f"{dataset_name}_{split}_standardization_config.json"
        config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        print("No scenarios were successfully analyzed.")
        print(f"Saved failures to {failures_path}")
        return AnalysisRunResult(
            analysis_outputs={},
            failures_path=failures_path,
            config_path=config_path,
            total_scenarios=total,
            failure_count=len(failures),
        )

    final_result = flush_analysis_outputs(
        analyses,
        results_by_analysis,
        failures,
        output_dir,
        dataset_name,
        split,
        config=config,
        include_plots=include_plots,
    )
    final_result.total_scenarios = total
    return final_result
