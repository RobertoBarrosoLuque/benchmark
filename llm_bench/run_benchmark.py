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
    parser = argparse.ArgumentParser(description='Run benchmark and extract latency stats')

    # Required arguments
    parser.add_argument('--deployment-id', required=True, help='Deployment ID for the model')
    parser.add_argument('--model-name', required=True, help='Short model name (e.g., qwen3-8b)')
    parser.add_argument('--prompt-length', type=int, required=True, help='Input prompt length in tokens')
    parser.add_argument('--output-length', type=int, required=True, help='Expected output token length')

    # Optional arguments
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 2, 3, 4, 5, 10],
                        help='List of concurrent workers')
    parser.add_argument('--qps', nargs='+', type=int, default=None, help='Fixed QPS mode')
    parser.add_argument('--spawn-rate', type=int, default=100, help='Worker spawn rate')
    parser.add_argument('--prompt-cache-max-len', type=int, default=0, help='Prompt cache tokens')
    parser.add_argument('--duration', default="3min", help='Duration per test')
    parser.add_argument('--api-key', help='Fireworks API key')
    parser.add_argument('--host', default="https://api.fireworks.ai/inference", help='API host')
    parser.add_argument('--embeddings', action='store_true', help='Use embeddings endpoint')
    parser.add_argument('--tokenizer', help='HF tokenizer for validation')
    parser.add_argument('--reasoning-effort', type=str, choices=['none', 'low', 'medium', 'high'],
                        help='Reasoning effort for thinking models')

    args = parser.parse_args()
    script_dir = Path(__file__).parent

    # Step 1: Run collect_data.py
    collect_cmd = [
        sys.executable, str(script_dir / "collect_data.py"),
        "--deployment-id", args.deployment_id,
        "--model-name", args.model_name,
        "--prompt-length", str(args.prompt_length),
        "--output-length", str(args.output_length),
        "--spawn-rate", str(args.spawn_rate),
        "--prompt-cache-max-len", str(args.prompt_cache_max_len),
        "--duration", args.duration,
        "--host", args.host,
        "--concurrency", *[str(c) for c in args.concurrency],
    ]

    if args.api_key:
        collect_cmd.extend(["--api-key", args.api_key])
    if args.embeddings:
        collect_cmd.append("--embeddings")
    if args.qps is not None:
        collect_cmd.extend(["--qps"] + [str(q) for q in args.qps])
    if args.tokenizer:
        collect_cmd.extend(["--tokenizer", args.tokenizer])
    if args.reasoning_effort:
        collect_cmd.extend(["--reasoning-effort", args.reasoning_effort])

    if not run_command(collect_cmd, "Data Collection"):
        sys.exit(1)

    # Step 2: Run extract_latency_stats.py
    extract_cmd = [
        sys.executable, str(script_dir / "extract_latency_stats.py"),
        "--model-name", args.model_name,
        "--input-length", str(args.prompt_length),
        "--output-length", str(args.output_length),
    ]

    if not run_command(extract_cmd, "Latency Stats Extraction"):
        sys.exit(1)

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()