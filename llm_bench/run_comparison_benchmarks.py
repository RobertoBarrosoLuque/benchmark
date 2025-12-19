import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configuration - Edit these to match your deployments
DEPLOYMENTS = [
    {
        "name": "Qwen3-8B-VL-Instruct",
        "deployment_id": "accounts/pyroworks/deployedModels/qwen3-vl-8b-instruct-wtocxydw",
        "model_name": "qwen3-vl-8b-instruct",
    },
    {
        "name": "Qwen3-8B",
        "deployment_id": "accounts/pyroworks/deployedModels/qwen3-8b-lb2y987a",
        "model_name": "qwen3-8b",
        "reasoning_effort": "none",
    },
    {
        "name": "Ministral-3-8B",
        "deployment_id": "accounts/pyroworks/deployedModels/ministral-3-8b-instruct-2512-m001qgzp",
        "model_name": "ministral-3-8b",
    },
    {
        "name": "Ministral-3-14B",
        "deployment_id": "accounts/pyroworks/deployedModels/ministral-3-14b-instruct-2512-re3bbuh2",
        "model_name": "ministral-3-14b",
    },
    {
        "name": "Gemma3-12B",
        "deployment_id": "accounts/pyroworks/deployedModels/gemma-3-12b-it-n3df9f5k",
        "model_name": "gemma-3-12b-it",
    },
]

WORKLOADS = [
    {"name": "Long Context", "input_tokens": 3000, "output_tokens": 140},
    {"name": "Short Context", "input_tokens": 400, "output_tokens": 20},
]

# Benchmark settings
CONCURRENCY_LEVELS = [1, 2]
DURATION = "3min"


def run_benchmark(deployment: dict, workload: dict) -> bool:
    """Run a single benchmark and return success status."""
    script_dir = Path(__file__).parent

    cmd = [
        sys.executable,
        str(script_dir / "run_benchmark.py"),
        "--deployment-id", deployment["deployment_id"],
        "--model-name", deployment["model_name"],
        "--prompt-length", str(workload["input_tokens"]),
        "--output-length", str(workload["output_tokens"]),
        "--duration", DURATION,
        "--concurrency", *[str(c) for c in CONCURRENCY_LEVELS],
    ]

    if "reasoning_effort" in deployment:
        cmd.extend(["--reasoning-effort", deployment["reasoning_effort"]])

    print(f"\n{'='*60}")
    print(f"Running benchmark:")
    print(f"  Model: {deployment['name']} ({deployment['model_name']})")
    print(f"  Workload: {workload['name']} ({workload['input_tokens']} in / {workload['output_tokens']} out)")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def collect_results() -> pd.DataFrame:
    """Collect all benchmark results into a single DataFrame."""
    results_dir = Path(__file__).parent / "results"
    all_results = []

    for deployment in DEPLOYMENTS:
        model_name = deployment["model_name"]

        for workload in WORKLOADS:
            input_tokens = workload["input_tokens"]
            output_tokens = workload["output_tokens"]

            csv_file = results_dir / f"{model_name}_input{input_tokens}_output{output_tokens}_latency_stats.csv"

            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df["Model"] = deployment["name"]
                df["Workload"] = workload["name"]
                df["Input Tokens"] = input_tokens
                df["Output Tokens"] = output_tokens
                all_results.append(df)
                print(f"Loaded: {csv_file.name}")
            else:
                print(f"WARNING: Not found: {csv_file.name}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        cols = ["Model", "Workload", "Input Tokens", "Output Tokens", "Concurrency"] + \
               [c for c in combined_df.columns if c not in ["Model", "Workload", "Input Tokens", "Output Tokens", "Concurrency"]]
        return combined_df[cols]

    return pd.DataFrame()


def save_to_xlsx(df: pd.DataFrame, output_path: Path):
    """Save DataFrame to xlsx with separate sheets for analysis."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Results', index=False)

        for workload in WORKLOADS:
            workload_df = df[df["Workload"] == workload["name"]]
            if not workload_df.empty:
                sheet_name = workload["name"].replace(" ", "_")[:31]
                workload_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for deployment in DEPLOYMENTS:
            model_df = df[df["Model"] == deployment["name"]]
            if not model_df.empty:
                sheet_name = deployment["name"][:31]
                model_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nResults saved to: {output_path}")


def main():
    print("="*60)
    print("LLM Benchmark Comparison Script")
    print("="*60)
    print(f"\nDeployments: {len(DEPLOYMENTS)}")
    for d in DEPLOYMENTS:
        extra = f" (reasoning_effort={d['reasoning_effort']})" if "reasoning_effort" in d else ""
        print(f"  - {d['name']}{extra}")
    print(f"\nWorkloads: {len(WORKLOADS)}")
    for w in WORKLOADS:
        print(f"  - {w['name']}: {w['input_tokens']} in / {w['output_tokens']} out")
    print(f"\nConcurrency: {CONCURRENCY_LEVELS}")
    print(f"Duration: {DURATION}")
    print(f"\nTotal runs: {len(DEPLOYMENTS) * len(WORKLOADS)}")

    failed = []
    for deployment in DEPLOYMENTS:
        for workload in WORKLOADS:
            if not run_benchmark(deployment, workload):
                failed.append(f"{deployment['name']} - {workload['name']}")

    print("\n" + "="*60)
    print("Collecting results...")
    print("="*60)

    results_df = collect_results()

    if not results_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / f"benchmark_comparison_{timestamp}.xlsx"
        save_to_xlsx(results_df, output_file)

        print("\n" + "="*60)
        print("Summary: Latency p50 (ms) @ Concurrency=1")
        print("="*60)
        if 1 in results_df["Concurrency"].values:
            summary = results_df[results_df["Concurrency"] == 1][
                ["Model", "Workload", "Latency p50 (ms)", "TTFT p50 (ms)", "LPT p50 (ms)"]
            ]
            print(summary.to_string(index=False))
        else:
            print(results_df[["Model", "Workload", "Concurrency", "Latency p50 (ms)"]].to_string(index=False))
    else:
        print("No results collected!")

    if failed:
        print(f"\nWARNING: Failed benchmarks: {failed}")

    print("\nBenchmark comparison complete!")


if __name__ == "__main__":
    main()