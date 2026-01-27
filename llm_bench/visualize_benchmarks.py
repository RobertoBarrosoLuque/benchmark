import argparse
from pathlib import Path

import altair as alt
import pandas as pd


def create_benchmark_visualization(df: pd.DataFrame) -> alt.VConcatChart:
    """
    Create a combined visualization with two charts:
    1. Concurrency vs P95 Latency
    2. Concurrency vs P50 Latency

    Both charts are colored by model and faceted by workload.
    """
    df = df[df["Concurrency"] >= 5]

    base = alt.Chart(df).encode(
        color=alt.Color("Model:N", legend=alt.Legend(title="Model")),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Workload:N"),
            alt.Tooltip("Concurrency:Q"),
            alt.Tooltip("Latency p50 (ms):Q", format=".0f"),
            alt.Tooltip("Latency p95 (ms):Q", format=".0f"),
        ],
    )

    # Chart 1: Concurrency vs P95 Latency
    concurrency_chart = (
        base.mark_line(point=True)
        .encode(
            x=alt.X("Concurrency:Q", title="Concurrency"),
            y=alt.Y("Latency p95 (ms):Q", title="P95 Latency (ms)"),
        )
        .properties(width=300, height=200, title="Concurrency vs P95 Latency")
        .facet(column=alt.Column("Workload:N", title="Workload"))
        .resolve_scale(y="independent")
    )

    # Chart 2: Concurrency vs P50 Latency
    p50_chart = (
        base.mark_line(point=True)
        .encode(
            x=alt.X("Concurrency:Q", title="Concurrency"),
            y=alt.Y("Latency p50 (ms):Q", title="P50 Latency (ms)"),
        )
        .properties(width=300, height=200, title="Concurrency vs P50 Latency")
        .facet(column=alt.Column("Workload:N", title="Workload"))
        .resolve_scale(y="independent")
    )

    return alt.vconcat(concurrency_chart, p50_chart).resolve_scale(color="shared")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results using Altair"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the benchmark comparison Excel file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML file path (default: input_file with .html extension)",
    )
    parser.add_argument(
        "--exclude-model",
        type=str,
        nargs="+",
        default=[],
        help="Model names (or substrings) to exclude from the visualization",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input_file.with_suffix(".html")

    df = pd.read_excel(args.input_file)
    for pattern in args.exclude_model:
        df = df[~df["Model"].str.contains(pattern, case=False)]
    chart = create_benchmark_visualization(df)
    chart.save(str(args.output))
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()