import os
import pandas as pd
import re


def extract_concurrency(dirname):
    match = re.search(r'(\d+)u-', dirname)
    return int(match.group(1)) if match else 0


def process_stats(base_dir, output_size):
    results = []
    pattern = f'r1-output-{output_size}-'

    # Find all relevant directories
    for dirname in os.listdir(base_dir):
        if pattern in dirname and dirname.endswith('5min'):
            stats_file = os.path.join(base_dir, dirname, 'stats_stats.csv')
            if os.path.exists(stats_file):
                # Read the stats file
                df = pd.read_csv(stats_file)
                # Get both metric rows
                lpt_row = df[df['Name'] == 'latency_per_token'].iloc[0]
                ttft_row = df[df['Name'] == 'time_to_first_token'].iloc[0]

                # Extract concurrency level
                concurrency = extract_concurrency(dirname)

                # Create result row
                result = {
                    'Concurrency': concurrency,
                    'Requests/s': lpt_row['Requests/s'],
                    'LPT Median (ms)': lpt_row['Median Response Time'],
                    'LPT Average (ms)': lpt_row['Average Response Time'],
                    'LPT p50 (ms)': lpt_row['50%'],
                    'LPT p90 (ms)': lpt_row['90%'],
                    'LPT p95 (ms)': lpt_row['95%'],
                    'LPT p99 (ms)': lpt_row['99%'],
                    'LPT p99.9 (ms)': lpt_row['99.9%'],
                    'TTFT Median (ms)': ttft_row['Median Response Time'],
                    'TTFT Average (ms)': ttft_row['Average Response Time'],
                    'TTFT p50 (ms)': ttft_row['50%'],
                    'TTFT p90 (ms)': ttft_row['90%'],
                    'TTFT p95 (ms)': ttft_row['95%'],
                    'TTFT p99 (ms)': ttft_row['99%'],
                    'TTFT p99.9 (ms)': ttft_row['99.9%']
                }
                results.append(result)

    # Convert to DataFrame and sort by concurrency
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Concurrency')

    # Save to CSV
    output_file = f'latency_stats_output_{output_size}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Created {output_file}")


def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

    # Process output size of 480
    process_stats(base_dir, 480)


if __name__ == "__main__":
    main()