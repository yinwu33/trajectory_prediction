from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional runtime helper
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import MotionDataset, standardize
from scripts.attr_analysis import AnalysisSaveResult, build_analyses
from scripts.attr_analysis.utils import json_ready


_WORKER_DATASET = None
_WORKER_ANALYSES = None


def parse_args() -> argparse.Namespace:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Analyze standardized AV2 or Waymo attribute distributions.",
    )
    parser.add_argument(
        "--dataset",
        choices=("av2", "waymo"),
        required=True,
        help="Dataset backend to analyze.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenarios to analyze.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max((os.cpu_count() or 1) - 1, 1),
        help="Number of worker processes for per-scenario collection.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N processed scenarios.",
    )
    return parser.parse_args()


def _normalize_split(dataset_name: str, split: str | None) -> str:
    if split is None:
        return "train" if dataset_name == "av2" else "training"

    if dataset_name == "av2":
        aliases = {
            "train": "train",
            "val": "val",
            "test": "test",
            "mini_train": "mini_train",
            "mini_val": "mini_val",
        }
        return aliases.get(split, split)

    aliases = {
        "train": "training",
        "training": "training",
        "val": "validation",
        "validation": "validation",
        "test": "testing",
        "testing": "testing",
    }
    return aliases.get(split, split)


def _resolve_data_root(dataset_name: str) -> Path:
    env_var = "AV2_DATA_ROOT" if dataset_name == "av2" else "WAYMO_DATA_ROOT"
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(f"Missing dataset root env var: {env_var}")
    return Path(value).expanduser().resolve()


def create_standardized_dataset(
    *,
    dataset_name: str,
    data_root: Path,
    split: str,
):
    if dataset_name == "av2":
        dataset = MotionDataset.create_from_av2(
            data_root=data_root,
            split=split,
        )
    elif dataset_name == "waymo":
        dataset = MotionDataset.create_from_waymo(
            data_root=data_root,
            split=split,
        )
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return standardize(dataset)


def _init_worker(dataset_name: str, data_root: str, split: str) -> None:
    global _WORKER_DATASET, _WORKER_ANALYSES
    _WORKER_DATASET = create_standardized_dataset(
        dataset_name=dataset_name,
        data_root=Path(data_root),
        split=split,
    )
    _WORKER_ANALYSES = build_analyses()


def _collect_one(index: int) -> dict[str, Any]:
    global _WORKER_DATASET, _WORKER_ANALYSES
    scenario_ref = _WORKER_DATASET.scenario_refs[index]
    scenario_id = scenario_ref.scenario_id or str(index)
    payload = {
        "index": int(index),
        "scenario_id": str(scenario_id),
        "rows_by_analysis": {analysis.name: [] for analysis in _WORKER_ANALYSES},
        "failures": [],
    }

    try:
        scenario = _WORKER_DATASET[index]
    except Exception as exc:  # pragma: no cover - dataset/runtime dependent
        payload["failures"].append(
            {
                "index": int(index),
                "scenario_id": str(scenario_id),
                "analysis": "scenario_load",
                "error": str(exc),
            }
        )
        return payload

    for analysis in _WORKER_ANALYSES:
        try:
            payload["rows_by_analysis"][analysis.name] = analysis.collect(scenario)
        except Exception as exc:  # pragma: no cover - dataset/runtime dependent
            payload["failures"].append(
                {
                    "index": int(index),
                    "scenario_id": str(scenario_id),
                    "analysis": analysis.name,
                    "error": str(exc),
                }
            )
    return payload


def _process_results(
    *,
    results_iterable,
    analyses,
    total: int,
    progress_every: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows_by_analysis = {analysis.name: [] for analysis in analyses}
    failures: list[dict[str, Any]] = []

    for processed, payload in enumerate(results_iterable, start=1):
        for analysis_name, rows in payload["rows_by_analysis"].items():
            rows_by_analysis[analysis_name].extend(rows)
        failures.extend(payload["failures"])

        if processed % max(progress_every, 1) == 0 or processed == total:
            row_counts = " ".join(
                f"{analysis.name}={len(rows_by_analysis[analysis.name])}"
                for analysis in analyses
            )
            print(
                f"processed={processed}/{total} failures={len(failures)} rows[{row_counts}]"
            )
    return rows_by_analysis, failures


def _run_serial(dataset, analyses, indices: list[int], progress_every: int):
    global _WORKER_DATASET, _WORKER_ANALYSES
    _WORKER_DATASET = dataset
    _WORKER_ANALYSES = analyses
    return _process_results(
        results_iterable=(_collect_one(index) for index in indices),
        analyses=analyses,
        total=len(indices),
        progress_every=progress_every,
    )


def _run_parallel(
    *,
    dataset_name: str,
    data_root: Path,
    split: str,
    analyses,
    indices: list[int],
    num_workers: int,
    progress_every: int,
):
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(dataset_name, str(data_root), split),
    ) as executor:
        results_iterable = executor.map(_collect_one, indices, chunksize=1)
        return _process_results(
            results_iterable=results_iterable,
            analyses=analyses,
            total=len(indices),
            progress_every=progress_every,
        )


def main() -> int:
    args = parse_args()
    dataset_name = args.dataset
    split = _normalize_split(dataset_name, args.split)
    data_root = _resolve_data_root(dataset_name)
    output_dir = (
        args.output_dir or ROOT / "outputs" / "data_analysis" / dataset_name / split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_standardized_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
    )
    analyses = build_analyses()

    if len(dataset) == 0:
        raise RuntimeError("Loaded dataset is empty")

    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    indices = list(range(total))
    print(
        f"dataset={dataset_name} split={split} scenarios={total} "
        f"num_workers={args.num_workers} output_dir={output_dir}"
    )

    if args.num_workers <= 1:
        rows_by_analysis, failures = _run_serial(
            dataset,
            analyses,
            indices,
            args.progress_every,
        )
    else:
        rows_by_analysis, failures = _run_parallel(
            dataset_name=dataset_name,
            data_root=data_root,
            split=split,
            analyses=analyses,
            indices=indices,
            num_workers=args.num_workers,
            progress_every=args.progress_every,
        )

    outputs: dict[str, AnalysisSaveResult] = {}
    for analysis in analyses:
        outputs[analysis.name] = analysis.save(
            rows_by_analysis[analysis.name],
            output_dir=output_dir,
            dataset_name=dataset_name,
            split=split,
        )

    failures_path = output_dir / f"{dataset_name}_{split}_failures.json"
    failures_path.write_text(
        json.dumps(json_ready(failures), indent=2),
        encoding="utf-8",
    )

    for name, output in outputs.items():
        print(f"{name}: csv={output.csv_path}")
        print(f"{name}: parquet={output.parquet_path}")
        print(f"{name}: summary={output.summary_path}")
    print(f"failures: {failures_path}")
    print(f"completed: scenarios={total} failures={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
