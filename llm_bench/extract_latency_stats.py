import os
import pandas as pd
import re
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'


def extract_concurrency(dirname):
    """Extract concurrency from directory name like 'model_in3000_out140_5u'"""
    match = re.search(r'_(\d+)u$', dirname)
    return int(match.group(1)) if match else 0


def process_stats(model_name: str, input_len: int, output_len: int) -> pd.DataFrame:
    """Process stats for a specific model and workload, return DataFrame."""
    results = []

    # Pattern: {model_name}_in{input}_out{output}_{concurrency}u
    pattern = f"{model_name}_in{input_len}_out{output_len}_"

    print(f"Looking for directories matching: {pattern}*")

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return pd.DataFrame()

    for dirname in os.listdir(RESULTS_DIR):
        dir_path = RESULTS_DIR / dirname
        if not dir_path.is_dir():
            continue

        if not dirname.startswith(pattern):
            continue

        stats_file = dir_path / 'stats_stats.csv'
        if not stats_file.exists():
            print(f"  No stats file in {dirname}")
            continue

        df = pd.read_csv(stats_file)
        if df.empty or df.shape[0] <= 1:
            print(f"  Empty stats in {dirname}")
            continue

        def get_metric_row(name):
            rows = df[df['Name'] == name]
            return rows.iloc[0] if len(rows) > 0 else None

        total_latency_row = get_metric_row('total_latency')
        lpt_row = get_metric_row('latency_per_token')
        ttft_row = get_metric_row('time_to_first_token')

        if total_latency_row is None:
            print(f"  Missing total_latency in {dirname}")
            continue

        concurrency = extract_concurrency(dirname)

        result = {
            'Concurrency': concurrency,
            'Requests/s': total_latency_row['Requests/s'],
            'Latency Average': total_latency_row['Average Response Time'],
            'Latency p50 (ms)': total_latency_row['50%'],
            'Latency p90 (ms)': total_latency_row['90%'],
            'Latency p95 (ms)': total_latency_row['95%'],
            'Latency p99 (ms)': total_latency_row['99%'],
            'Latency p99.9 (ms)': total_latency_row['99.9%'],
        }

        # Add LPT metrics if available
        if lpt_row is not None:
            result.update({
                'LPT Average (ms)': lpt_row['Average Response Time'],
                'LPT p50 (ms)': lpt_row['50%'],
                'LPT p90 (ms)': lpt_row['90%'],
                'LPT p95 (ms)': lpt_row['95%'],
                'LPT p99 (ms)': lpt_row['99%'],
                'LPT p99.9 (ms)': lpt_row['99.9%'],
            })

        # Add TTFT metrics if available
        if ttft_row is not None:
            result.update({
                'TTFT Average (ms)': ttft_row['Average Response Time'],
                'TTFT p50 (ms)': ttft_row['50%'],
                'TTFT p90 (ms)': ttft_row['90%'],
                'TTFT p95 (ms)': ttft_row['95%'],
                'TTFT p99 (ms)': ttft_row['99%'],
                'TTFT p99.9 (ms)': ttft_row['99.9%'],
            })
        results.append(result)
        print(f"  Loaded: {dirname} (concurrency={concurrency})")

    if not results:
        print(f"No results found for {model_name} in{input_len} out{output_len}")
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values('Concurrency')

    # Save aggregated CSV
    output_file = RESULTS_DIR / f'{model_name}_input{input_len}_output{output_len}_latency_stats.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Created: {output_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Extract latency statistics from benchmark results')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Short model name (e.g., qwen3-8b)')
    parser.add_argument('--input-length', type=int, required=True,
                        help='Input token length')
    parser.add_argument('--output-length', type=int, required=True,
                        help='Output token length')

    args = parser.parse_args()
    process_stats(args.model_name, args.input_length, args.output_length)


if __name__ == "__main__":
    main()
