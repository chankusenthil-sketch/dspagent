"""
example.py — Programmatic usage of the DSP Agent.

Shows how to call run_agent() with different CSV files and goals,
inspect the returned state, and iterate over tool results.
Output is printed to the console and saved to a log file.
"""

import uuid
import sys
import os
from datetime import datetime
from dsp_agent import run_agent

# ── Set up logging to file + console ───────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"dsp_agent_{datetime.now():%Y%m%d_%H%M%S}.log")
log_file = open(log_filename, "w")

class TeeWriter:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

original_stdout = sys.stdout
sys.stdout = TeeWriter(original_stdout, log_file)

print(f"Logging to {log_filename}\n")

# ── 1. Basic usage ──────────────────────────────────────────────────────────

result = run_agent(
    csv_path="data/samples/imu_jogging_subject1.csv",
    user_goal="What is the orientation of this device? Where this device might be located on the person's body? Is the person moving or stationary?",
    fs=50.0,                        # sampling rate in Hz
    max_iterations=3,               # max plan→execute loops
    thread_id=str(uuid.uuid4()),    # unique thread for checkpointing
)

# ── 2. Inspect results ─────────────────────────────────────────────────────

print("\n── Agent Results ──")
print("Final answer:", result.get("final_answer", "N/A"))
print("Iterations used:", result.get("iteration", "?"))

# List every tool the agent chose to run
tool_results = result.get("tool_results", [])
print(f"Tools executed ({len(tool_results)}):")
for tr in tool_results:
    print(f"  - {tr['tool']}  args={tr.get('args', {})}")

'''
# ── 3. Run on a different file with a different goal ────────────────────────

result2 = run_agent(
    csv_path="data/samples/imu_sitting_subject1.csv",
    user_goal="Compute the signal energy and statistics for accelerometer axes. Is the person moving?",
    fs=50.0,
    max_iterations=2,
    thread_id=str(uuid.uuid4()),
)

print("\n── Second Run ──")
print("Answer:", result2.get("final_answer", "N/A"))
'''
# ── Cleanup ─────────────────────────────────────────────────────────────────

sys.stdout = original_stdout
log_file.close()
print(f"\nLog saved to {log_filename}")

