from dotenv import load_dotenv
load_dotenv()
import os
import datetime
import subprocess
import time

"""
Change constants here
"""
api_key = os.getenv('FIREWORKS_API_KEY')
deployment_id = "33aa85b1"
# us = [1,2,3,4,5,6,7,8,9,10]
us = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]     # Number of concurrent workers
r = 100      # Rate of spawning new workers (workers/second). Look through README.md for more details on spawn rate
prompt_length = 63360
prompt_cache_max_len = 0
output_length = 640
t = "5min" #test duration, set to 1 minute for now

#Function to utilize subprocess to run the locust script
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


'''
Make sure to create a .env file in the root directory and add your API keys.
For this example, we will use the Fireworks API key.

Add the following to your .env file:

FIREWORKS_API_KEY=<your_fireworks_api_key>.

Alternatively you can edit the following script flags for custom configurations.
'''


provider_name = "fireworks"
model_name = "accounts/fireworks/models/deepseek-v3-0324" + f"#accounts/pyroworks/deployments/{deployment_id}"
h = "https://api.fireworks.ai/inference" #host url

# Create base results directory
os.makedirs("results", exist_ok=True)

for u in us:
    # Create results directory of name single_model_provider_analysis_{TIMESTAMP}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    edited_model_name = model_name.replace("/", "_") if provider_name != "fireworks" else model_name.replace("accounts/fireworks/models/", "").replace("/", "_")

    results_dir = f"results/r1-output-{output_length}-{u}u-5min"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the command
    cmd = [
        "locust",
        "--headless",       # Run without web UI
        "--only-summary",   # Only show summary stats
        "-H", h,           # Host URL
        "--provider", provider_name,
        "--model", model_name,
        "--api-key", api_key,
        "-t", t,           # Test duration
        "--html", f"{results_dir}/report.html",  # Generate HTML report
        "--csv", f"{results_dir}/stats",        # Generate CSV stats
    ]

    # Add Mode 1 (Fixed QPS) parameters if uncommented, remember to remove --qps below if using fixed concurrency mode
    cmd.extend([
    # "--qps", str(qps),  # Target QPS
        "-u", str(u),      # Number of users
        "-r", str(r),      # Spawn rate
        "-p", str(prompt_length),
        "--prompt-cache-max-len", str(prompt_cache_max_len),
        "-o", str(output_length),
        "--stream"
    ])

    # Add load_test.py as the locust file
    locust_file = os.path.join(os.path.dirname(__file__), "load_test.py")
    cmd.extend(["-f", locust_file])

    #call our helper function to execute the command
    success = execute_subprocess(cmd)

    #Visualize the results
    if success:
        time.sleep(1)
        stat_result_paths = [{"path": f'{results_dir}/stats_stats.csv', "config": {"provider": provider_name, "model": model_name}}]
        #visualize_comparative_results(stat_result_paths, results_dir)

    time.sleep(25)