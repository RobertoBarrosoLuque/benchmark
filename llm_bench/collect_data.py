from dotenv import load_dotenv

load_dotenv()
import os
import subprocess
import sys
import time
import argparse

# Get the locust binary from the same environment as this script
LOCUST_BIN = os.path.join(os.path.dirname(sys.executable), "locust")


def main():
    parser = argparse.ArgumentParser(description='Run LLM benchmarking with configurable parameters')
    parser.add_argument('--deployment-id', required=True, help='Deployment ID for the model')
    parser.add_argument('--model-name', required=True, help='Short model name for results directory (e.g., qwen3-8b)')
    parser.add_argument('--concurrency', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
                        help='List of concurrent workers')
    parser.add_argument('--qps', nargs='+', type=int, default=None,
                        help='Fixed QPS mode (mutually exclusive with concurrency iteration)')
    parser.add_argument('--spawn-rate', type=int, default=100,
                        help='Rate of spawning workers (workers/second)')
    parser.add_argument('--prompt-length', type=int, required=True,
                        help='Input prompt length in tokens')
    parser.add_argument('--output-length', type=int, required=True,
                        help='Expected output token length')
    parser.add_argument('--prompt-cache-max-len', type=int, default=0,
                        help='Token count for caching (0 disables)')
    parser.add_argument('--duration', default="3min",
                        help='Duration for each test (e.g., 3min, 5min)')
    parser.add_argument('--api-key', help='Fireworks API key (overrides .env)')
    parser.add_argument('--host', default="https://api.fireworks.ai/inference",
                        help='Host URL for the API')
    parser.add_argument('--embeddings', action='store_true', help='Use embeddings endpoint')
    parser.add_argument('--tokenizer', help='HF tokenizer for output validation')
    parser.add_argument('--reasoning-effort', type=str, choices=['none', 'low', 'medium', 'high'],
                        help='Reasoning effort for thinking models (e.g., Qwen3)')

    args = parser.parse_args()

    api_key = args.api_key or os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("No API key. Set FIREWORKS_API_KEY in .env or use --api-key")

    model_name = args.model_name
    input_len = args.prompt_length
    output_len = args.output_length
    duration = args.duration
    duration_short = duration.replace('min', '')

    os.makedirs("results", exist_ok=True)

    # Determine iteration mode
    if args.qps is not None:
        iteration_values = args.qps
        iteration_mode = "qps"
        fixed_users = 100
    else:
        iteration_values = args.concurrency
        iteration_mode = "concurrency"

    for value in iteration_values:
        if iteration_mode == "qps":
            results_dir = f"results/{model_name}_in{input_len}_out{output_len}_{value}qps"
            users = fixed_users
        else:
            results_dir = f"results/{model_name}_in{input_len}_out{output_len}_{value}u"
            users = value

        os.makedirs(results_dir, exist_ok=True)

        cmd = [
            LOCUST_BIN,
            "--headless",
            "--only-summary",
            "-H", args.host,
            "--provider", "fireworks",
            "--model", args.deployment_id,
            "--api-key", api_key,
            "-t", duration,
            "--html", f"{results_dir}/report.html",
            "--csv", f"{results_dir}/stats",
            "-u", str(users),
            "-r", str(args.spawn_rate),
            "-p", str(input_len),
            "--prompt-cache-max-len", str(args.prompt_cache_max_len),
            "-o", str(output_len),
            "--stream"
        ]

        if args.embeddings:
            cmd.append("--embeddings")
        if iteration_mode == "qps":
            cmd.extend(["--qps", str(value)])
        if args.tokenizer:
            cmd.extend(["--tokenizer", args.tokenizer])
        if args.reasoning_effort:
            cmd.extend(["--reasoning-effort", args.reasoning_effort])

        locust_file = os.path.join(os.path.dirname(__file__), "load_test.py")
        cmd.extend(["-f", locust_file])

        success = execute_subprocess(cmd)
        if success:
            time.sleep(1)
        time.sleep(25)


def execute_subprocess(cmd):
    print(f"\nExecuting: {' '.join(str(arg) for arg in cmd)}\n")
    process = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True
    )
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return_code = process.poll()
    if return_code != 0:
        print(f"Benchmark failed with return code: {return_code}")
        return False
    return True


if __name__ == "__main__":
    main()