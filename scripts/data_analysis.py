"""Entry point for standardized motion attribute analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional runtime helper
    load_dotenv = None

from datasets.standardization import (
    StandardConfig,
    StandardAgentConfig,
    StandardMapConfig,
    StandardizationConfig,
    standardize,
)
from datasets.motion_dataset import MotionDataset
from scripts.attr_analysis import build_default_analyses, run_standard_dataset_analyses


def parse_args() -> argparse.Namespace:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Analyze standardized AV2 or Waymo motion attributes with modular analyzers."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("av2", "waymo"),
        default="av2",
        help="Dataset backend to analyze.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. Defaults to AV2_DATA_ROOT or WAYMO_DATA_ROOT depending on --dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to analyze. Defaults to train for AV2 and training for Waymo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs. Defaults to outputs/motion_analysis_standard/<dataset>/<split>.",
    )

    # parser.add_argument(
    #     "--progress-every",
    #     type=int,
    #     default=500,
    #     help="Log progress every N processed scenarios.",
    # )
    # parser.add_argument(
    #     "--save-every",
    #     type=int,
    #     default=500,
    #     help="Persist intermediate outputs every N processed scenarios.",
    # )
    # parser.add_argument(
    #     "--skip-plots",
    #     action="store_true",
    #     help="Skip matplotlib plot generation.",
    # )
    return parser.parse_args()


def _resolve_default_data_root(dataset_name: str) -> Path:
    env_var = "AV2_DATA_ROOT" if dataset_name == "av2" else "WAYMO_DATA_ROOT"
    return Path(os.environ.get(env_var, ""))


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
    else:
        aliases = {
            "train": "training",
            "training": "training",
            "val": "validation",
            "validation": "validation",
            "test": "testing",
            "testing": "testing",
        }
    return aliases.get(split, split)


def create_dataset(
    *,
    dataset_name: str,
    data_root: Path,
    split: str,
):
    config = StandardConfig
    if dataset_name == "av2":
        return standardize(
            MotionDataset.create_from_av2(
                data_root=data_root,
                split=split,
            ),
            config=config,
        )
    if dataset_name == "waymo":
        return standardize(
            MotionDataset.create_from_waymo(
                data_root=data_root,
                split=split,
            ),
            config=config,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> int:
    args = parse_args()
    dataset_name = args.dataset
    data_root = args.data_root or _resolve_default_data_root(dataset_name)
    split = _normalize_split(dataset_name, args.split)
    output_dir = (
        args.output_dir or Path("outputs") / "motion_analysis" / dataset_name / split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
    )
    analyses = build_default_analyses()

    result = run_standard_dataset_analyses(
        dataset,
        analyses,
        output_dir=output_dir,
        dataset_name=dataset_name,
        split=split,
        limit=args.limit,
        progress_every=args.progress_every,
        save_every=args.save_every,
        include_plots=not args.skip_plots,
    )

    if not result.analysis_outputs:
        return 1

    for name, output in result.analysis_outputs.items():
        print(f"{name}: csv={output.csv_path}")
        print(f"{name}: parquet={output.parquet_path}")
        print(f"{name}: summary={output.summary_path}")
        for extra_path in output.extra_paths:
            print(f"{name}: plot={extra_path}")
    print(f"failures: {result.failures_path}")
    print(f"config: {result.config_path}")
    print(
        f"completed: scenarios={result.total_scenarios} failures={result.failure_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
