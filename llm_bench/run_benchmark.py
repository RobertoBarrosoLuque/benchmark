import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n=== {description} ===")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Run complete benchmark: collect data and extract latency stats')
    
    # Required arguments
    parser.add_argument('--deployment-id', required=True, help='Deployment ID for the model')
    parser.add_argument('--prompt-length', type=int, required=True, help='Input prompt length in tokens')
    parser.add_argument('--output-length', type=int, required=True, help='Expected output token length')
    
    # Optional arguments with defaults
    parser.add_argument('--model', default="accounts/fireworks/models/deepseek-v3-0324", help='Base model name')
    parser.add_argument('--account', default="fireworks", help='Account name for extract script')
    parser.add_argument('--concurrency', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='List of concurrent workers')
    ## Add QPS:
    parser.add_argument(
        "--qps",
        nargs='+',
        type=int,
        default=None,
        help="Enabled 'fixed QPS' mode where requests are issues at the specified rate regardless of how long the processing takes. In this case --users and --spawn-rate need to be set to a sufficiently high value (e.g. 100)",
    )
    parser.add_argument('--spawn-rate', type=int, default=100, help='Rate of spawning new workers')
    parser.add_argument('--prompt-cache-max-len', type=int, default=0, help='Token count for caching')
    parser.add_argument('--duration', default="5min", help='Duration for each test')
    parser.add_argument('--api-key', help='Fireworks API key')
    parser.add_argument('--host', default="https://api.fireworks.ai/inference", help='Host URL for the API')

    ## Add boolean  --embeddings
    parser.add_argument('--embeddings', action='store_true', help='Run with embeddings endpoint')

    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Step 1: Run collect_data.py
    collect_cmd = [
        sys.executable, str(script_dir / "collect_data.py"),
        "--deployment-id", args.deployment_id,
        "--prompt-length", str(args.prompt_length),
        "--output-length", str(args.output_length),
        "--model", args.model,
        "--spawn-rate", str(args.spawn_rate),
        "--prompt-cache-max-len", str(args.prompt_cache_max_len),
        "--duration", args.duration,
        "--host", args.host
    ]
    
    # Add concurrency arguments
    collect_cmd.extend(["--concurrency"] + [str(c) for c in args.concurrency])
    
    # Add API key if provided
    if args.api_key:
        collect_cmd.extend(["--api-key", args.api_key])

    # Add embeddings flag if provided
    if args.embeddings:
        collect_cmd.append("--embeddings")

    # Add QPS if provided
    if args.qps is not None:
        collect_cmd.extend(["--qps"] + [str(q) for q in args.qps])

    if not run_command(collect_cmd, "Data Collection"):
        sys.exit(1)
    
    # Step 2: Run extract_latency_stats.py
    model_name = args.model.split('/')[-1]  # Extract model name from full path
    extract_cmd = [
        sys.executable, str(script_dir / "extract_latency_stats.py"),
        "--output-length", str(args.output_length),
        "--input-length", str(args.prompt_length),
        "--model-name", model_name,
        "--account", args.account
    ]
    
    if not run_command(extract_cmd, "Latency Stats Extraction"):
        sys.exit(1)
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()