"""Command-line entry point for the EDA framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.eda import load_csv
from src.report import create_figures, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a CSV dataset and generate EDA summaries and visualizations."
    )
    parser.add_argument(
        "--input",
        default="data/sample_data.csv",
        help="Path to input CSV (default: data/sample_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Directory for report and CSV summaries (default: reports)",
    )
    parser.add_argument(
        "--figures",
        default=None,
        help="Directory for figures (default: figures under the project root)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    figures_dir = Path(args.figures) if args.figures else Path("figures")

    try:
        df = load_csv(input_path)
        if df.empty:
            print("Warning: the input CSV has column headers but no data rows.")
        print(f"Loaded {input_path}: {df.shape[0]} rows x {df.shape[1]} columns")

        figure_names = create_figures(df, figures_dir)
        report_path = write_report(df, output_dir, figures_dir, figure_names)

        print(f"Report: {report_path}")
        print(f"Tables: {output_dir.resolve()}")
        print(f"Figures: {figures_dir.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
