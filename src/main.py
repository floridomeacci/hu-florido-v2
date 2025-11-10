#!/usr/bin/env python3
"""Entry point for running preprocessing and generating figures."""

from pathlib import Path
from typing import Callable, Iterable, Tuple

import pandas as pd

from create_plots import (
    comparing_categories,
    distribution,
    relationship,
    tSNE,
    time_series,
)
from data_cleaning import run_preprocessing

REPO_ROOT = Path(__file__).resolve().parent.parent

PlotFunc = Callable[[pd.DataFrame, Path], Path | None]
FIGURE_PIPELINE: Tuple[Tuple[str, PlotFunc], ...] = (
    ("distribution", distribution),
    ("relationship", relationship),
    ("tSNE", tSNE),
    ("time_series", time_series),
    ("comparing_categories", comparing_categories),
)


def generate_figures(df: pd.DataFrame, output_dir: Path, figures: Iterable[Tuple[str, PlotFunc]]) -> list[Path]:
    """Run the registered plot functions and capture successful outputs."""
    generated: list[Path] = []
    for name, func in figures:
        try:
            result = func(df, output_dir)
            if result:
                generated.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ ERROR generating {name}: {exc}")
            import traceback

            traceback.print_exc()
    return generated


def main() -> None:
    """Main orchestration function for preprocessing and plotting."""
    print("\n" + "=" * 70)
    print("5 KEY FIGURES GENERATION - STANDALONE")
    print("=" * 70)

    csv_path = run_preprocessing()

    print("\n" + "=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    if not csv_path.exists():
        print(f"❌ ERROR: {csv_path} does not exist!")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} messages from {csv_path}")

    print("\n" + "=" * 70)
    print("GENERATING 5 KEY FIGURES")
    print("=" * 70 + "\n")

    img_dir = REPO_ROOT / "img"
    img_dir.mkdir(exist_ok=True)

    generated = generate_figures(df, img_dir, FIGURE_PIPELINE)

    print("\n" + "=" * 70)
    print(f"✅ COMPLETED: Generated {len(generated)}/{len(FIGURE_PIPELINE)} figures")
    print("=" * 70)
    print("\nGenerated files:")
    for path in generated:
        print(f"  📈 {path}")
    print()


if __name__ == "__main__":
    main()
