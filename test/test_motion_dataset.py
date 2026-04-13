from __future__ import annotations

import argparse
import os
import re
import sys
from dotenv import load_dotenv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import MotionDataset, MotionScenario, standardize
from utils.viz_motion import plot_scenario

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load MotionDataset and standardized MotionDataset samples, visualize them, "
            "and save PNGs to the repository root."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("av2", "waymo"),
        default="av2",
        help="Dataset to load.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split. For Waymo, this is optional in the underlying API but defaults to train here.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[0],
        help="List of sample indices to visualize. Example: --sample-indices 0 5 10",
    )
    parser.add_argument(
        "--view-radius",
        type=float,
        default=60.0,
        help="Visualization window radius in meters.",
    )
    parser.add_argument(
        "--show-agent-ids",
        action="store_true",
        help="Overlay agent track ids next to the current position.",
    )
    return parser.parse_args()


def save_visualization(
    sample: MotionScenario,
    *,
    dataset_name: str,
    sample_index: int,
    repo_root: Path,
    view_radius: float,
    show_agent_ids: bool,
) -> Path:
    fig = plot_scenario(
        sample,
        view_radius=view_radius,
        show_agent_ids=show_agent_ids,
    )

    output_path = (
        repo_root / f"{dataset_name}_sample_{sample_index}_{sample.scenario_id}.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    md = None
    smd = None

    args = parse_args()
    if args.dataset == "waymo":
        data_root = os.getenv("WAYMO_DATA_ROOT")
        md = MotionDataset.create_from_waymo(
            data_root=data_root,
            split=args.split,
        )
        smd = standardize(md)
    elif args.dataset == "av2":
        data_root = os.getenv("AV2_DATA_ROOT")
        md = MotionDataset.create_from_av2(
            data_root=data_root,
            split=args.split,
        )
        smd = standardize(md)

    dataset_len = len(md)
    if dataset_len == 0:
        raise RuntimeError("Loaded dataset is empty")

    print(
        f"Loaded {args.dataset} split={args.split!r}: "
        f"MotionDataset={len(md)} samples, "
        f"standardized MotionDataset={len(smd)} samples"
    )

    for idx in args.sample_indices:
        sample = md[idx]
        standard_motion_sample = smd[idx]

        raw_output_path = save_visualization(
            sample,
            dataset_name=f"{args.dataset}_raw",
            sample_index=idx,
            repo_root=REPO_ROOT,
            view_radius=args.view_radius,
            show_agent_ids=args.show_agent_ids,
        )
        standard_output_path = save_visualization(
            standard_motion_sample,
            dataset_name=f"{args.dataset}_standardized",
            sample_index=idx,
            repo_root=REPO_ROOT,
            view_radius=args.view_radius,
            show_agent_ids=args.show_agent_ids,
        )

        print(f"[idx={idx}] raw        -> {raw_output_path}")
        print(f"[idx={idx}] standardized -> {standard_output_path}")


if __name__ == "__main__":
    main()
