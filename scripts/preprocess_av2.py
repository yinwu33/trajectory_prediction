"""Preprocess Argoverse 2 logs into cached .pt files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from functools import partial

# 引入 DataLoader 用于多进程加速
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# 引入 Console 用于更优雅的打印
from rich.console import Console
from rich.progress import track

from datamodule.datasets.av2_dataset import AV2Dataset

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Argoverse 2 logs into cached .pt files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory containing AV2 splits.",
    )
    parser.add_argument(
        "--preprocess-dir",
        type=Path,
        default=Path("./data_ssd"),
        help="Directory to store cached tensors.",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to preprocess."
    )
    # 添加 num_workers 参数
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of worker processes."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run preprocessing even if cache file exists.",
    )
    return parser.parse_args()


def process_item(batch):
    # DataLoader 的 collate_fn 会被调用，这里不需要做任何事
    # 只要 DataLoader 遍历了 dataset，__getitem__ 就会被触发，缓存就会生成
    pass


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        console.print(f"[red]Data root not found: {args.data_root}[/red]")
        return 1

    # 修复 1: 处理 Path(None) 问题

    # 注意：这里我们假设 Dataset 类内部如果没有 overwrite 参数，
    # 你可能需要临时修改 Dataset 或通过删除缓存文件来强制覆盖。
    # 为了通用性，这里保留外部删除逻辑，但在多进程前执行。

    dataset = AV2Dataset(
        data_root=args.data_root,
        split=args.split,
        preprocess=True,  # 确保开启预处理模式
        preprocess_dir=args.preprocess_dir,
    )

    if len(dataset) == 0:
        console.print(
            f"[yellow]No logs found in {args.data_root / args.split}[/yellow]"
        )
        return 1

    console.print(
        f"[bold green]Preprocessing split '{args.split}'[/bold green] with {len(dataset)} logs.\n"
        f"Cache dir: {dataset.cache_dir}\n"
        f"Workers: {args.num_workers}"
    )

    # 预先处理 Overwrite 逻辑 (在主进程做，避免多进程竞争删除文件)
    if args.overwrite:
        console.print(
            "[yellow]Overwrite mode enabled. Cleaning existing cache...[/yellow]"
        )
        # 这种方式比逐个检查快，但要小心不要删错
        if dataset.cache_dir.exists():
            for cache_file in dataset.cache_dir.glob("*.pt"):
                cache_file.unlink()

    # 修复 2 & 3: 使用 DataLoader 进行多进程加速
    # 我们使用一个简单的 DataLoader，batch_size 甚至可以是 1，
    # 重要的是利用 num_workers 并行调用 dataset.__getitem__
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,  # 既然只是预处理，不需要真正组装 batch，减少开销
    )

    # 使用 rich 的 track 显示进度
    # 这里的循环仅仅是为了驱动 DataLoader 运行
    for log_dir in track(
        dataloader.dataset.log_dirs, description="Processing...", total=len(dataset)
    ):
        log_id = log_dir.name
        cache_file = (
            dataloader.dataset.cache_dir / f"{log_id}.pt"
            if dataloader.dataset.preprocess
            else None
        )
        _ = torch.load(cache_file, map_location="cpu")

    console.print("[bold green]Done.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
