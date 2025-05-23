from dotenv import load_dotenv

load_dotenv()
import os
import subprocess
import time
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLM benchmarking with configurable parameters')
    parser.add_argument('--deployment-id', required=True, help='Deployment ID for the model')
    parser.add_argument('--concurrency', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='List of concurrent workers (e.g., 1 10 20 30)')
    parser.add_argument('--spawn-rate', type=int, default=100,
                        help='Rate of spawning new workers (workers/second)')
    parser.add_argument('--prompt-length', type=int, required=True,
                        help='Input prompt length in tokens')
    parser.add_argument('--prompt-cache-max-len', type=int, default=0,
                        help='Token count for caching (0 disables caching)')
    parser.add_argument('--output-length', type=int, required=True,
                        help='Expected output token length')
    parser.add_argument('--duration', default="5min",
                        help='Duration for each test (e.g., 5min, 10min)')
    parser.add_argument('--model', default="accounts/fireworks/models/deepseek-v3-0324",
                        help='Base model name (deployment ID will be appended)')
    parser.add_argument('--api-key', help='Fireworks API key (overrides .env file)')
    parser.add_argument('--host', default="https://api.fireworks.ai/inference",
                        help='Host URL for the API')

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("No API key provided. Set FIREWORKS_API_KEY in .env file or use --api-key")

    deployment_id = args.deployment_id
    us = args.concurrency
    r = args.spawn_rate
    prompt_length = args.prompt_length
    prompt_cache_max_len = args.prompt_cache_max_len
    output_length = args.output_length
    t = args.duration

    provider_name = "fireworks"
    model_name = f"{args.model}#accounts/pyroworks/deployments/{deployment_id}"
    h = args.host

    # Create base results directory
    os.makedirs("results", exist_ok=True)

    # Rest of the function remains the same as original code
    for u in us:
        # Create results directory name
        results_dir = f"results/{args.model}{output_length}-{u}u-{t.replace('min', '')}"
        os.makedirs(results_dir, exist_ok=True)

        # Construct the command
        cmd = [
            "locust",
            "--headless",  # Run without web UI
            "--only-summary",  # Only show summary stats
            "-H", h,  # Host URL
            "--provider", provider_name,
            "--model", model_name,
            "--api-key", api_key,
            "-t", t,  # Test duration
            "--html", f"{results_dir}/report.html",  # Generate HTML report
            "--csv", f"{results_dir}/stats",  # Generate CSV stats
            "-u", str(u),  # Number of users
            "-r", str(r),  # Spawn rate
            "-p", str(prompt_length),
            "--prompt-cache-max-len", str(prompt_cache_max_len),
            "-o", str(output_length),
            "--stream"
        ]

        # Add load_test.py as the locust file
        locust_file = os.path.join(os.path.dirname(__file__), "load_test.py")
        cmd.extend(["-f", locust_file])

        # Execute the command
        success = execute_subprocess(cmd)

        if success:
            time.sleep(1)

        time.sleep(25)


# Function to utilize subprocess to run the locust script
def execute_subprocess(cmd):
    print(f"\nExecuting benchmark: {' '.join(str(arg) for arg in cmd)}\n")
    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    # Display output in real-time
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