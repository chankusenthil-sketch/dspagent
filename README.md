# DSP Agent

AI-powered signal processing agent that analyzes IMU sensor signals using DSP tools and AHRS sensor fusion — planned and orchestrated by a local LLM.

## Project Goal

The LLM acts as a **thinking/planning agent**: it reads the user's goal, examines signal metadata, decides which DSP or sensor-fusion operations to apply, executes them, and interprets the results.

**Key capabilities:**
- Read IMU signals (accelerometer, gyroscope) from CSV files
- LLM-driven planning — the model decides what analysis to perform
- Iterative analysis: plan → execute → observe → plan again (or finish)
- **17 DSP/IMU tools** + **5 descriptor tools**: 12 SciPy DSP tools + 5 IMU Fusion (AHRS) tools + 5 output descriptors
- Stateful agent with memory via LangGraph `StateGraph` + `MemorySaver`
- Prompt templates loaded from external `.md` files (`prompts/`)
- Local LLM inference — no cloud API needed

---

## Architecture

The agent is a **5-node LangGraph StateGraph** that loops through plan-execute-observe
cycles until the LLM decides it has enough information (or the iteration cap is hit).

```
                          User Goal + CSV File
                                  │
                                  ▼
                        ┌─────────────────┐
                        │   load_signal    │  Parse CSV, detect fs, build signal summary
                        └────────┬────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
            ┌────►│            plan               │  LLM chooses tools + descriptors
            │     │  (prompt: prompts/plan.md)     │  Parses TOOLS & DESCRIPTORS sections
            │     └──────────────┬───────────────┘
            │                    │
            │          ┌────────▼────────┐
            │          │  tools queued?   │
            │          └──┬───────────┬──┘
            │             │ yes       │ no (DONE / max iter)
            │             ▼           ▼
            │     ┌──────────────┐  ┌──────────────┐
            │     │ execute_tool │  │   summarize   │  LLM generates final answer
            │     │              │  │  (prompt:      │  (prompt: prompts/summarize.md)
            │     └──────┬───────┘  │  summarize.md) │
            │            │          └──────┬───────┘
            │            ▼                 │
            │     ┌──────────────┐         ▼
            │     │   observe     │      Final Answer
            │     │  (prompt:     │
            │     │  observe.md)  │
            │     └──────┬───────┘
            │            │
            │   ┌────────▼────────┐
            │   │ more iterations? │
            │   └──┬───────────┬──┘
            │      │ yes       │ no (max iter)
            └──────┘           │
                               ▼
                        ┌──────────────┐
                        │   summarize   │
                        └──────┬───────┘
                               ▼
                          Final Answer
```

### Nodes (5 total)

| Node | Type | Description |
|------|------|-------------|
| `load_signal` | Data ingestion | Parse CSV into numpy arrays (`acc`, `gyro`, `timestamps`), auto-detect sampling rate, build `signal_info` text summary |
| `plan` | LLM call | LLM reads goal + signal metadata + prior observations, outputs structured THINK/PIPELINE/TOOLS/DESCRIPTORS sections. Agent parses tool calls and descriptor calls from LLM text. |
| `execute_tool` | Tool execution | Runs all queued DSP/IMU tools, stores raw results and intermediate arrays for tool-output chaining |
| `observe` | Descriptor execution | Runs descriptor tools on raw results to produce concise, LLM-readable observation text. Falls back to default summaries for tools without a descriptor. |
| `summarize` | LLM call | LLM interprets all accumulated tool results and answers the user's question |

### Graph Edges

| From | To | Condition |
|------|----|-----------|
| `START` | `load_signal` | Always (entry point) |
| `load_signal` | `plan` | Always (fixed edge) |
| `plan` | `execute_tool` | `tool_queue` is non-empty |
| `plan` | `summarize` | LLM says DONE, or `max_iterations` reached |
| `execute_tool` | `observe` | Always (fixed edge) |
| `observe` | `plan` | `iteration < max_iterations` (loop back) |
| `observe` | `summarize` | `iteration >= max_iterations` |
| `summarize` | `END` | Always (terminal) |

### One Full Iteration

A single plan-execute-observe cycle works like this:

1. **Plan** -- The LLM receives `signal_info`, `user_goal`, previous `observations`, and the full tool/descriptor registries via the `plan.md` template. It outputs structured text with THINK, PIPELINE, TOOLS, and DESCRIPTORS sections.
2. **Parse** -- `parse_tool_calls()` extracts tool names, args, and data sources from the TOOLS section. `parse_descriptor_calls()` extracts descriptor names, tool references, and args from the DESCRIPTORS section. Both use fuzzy matching to handle LLM misspellings.
3. **Execute** -- `execute_tool` iterates through `tool_queue`. For each tool: resolve the data source (acc/gyro/intermediate), dispatch to SciPy or IMU Fusion, store raw results in `tool_results`, and save any array outputs in `intermediate_results` for chaining.
4. **Observe** -- For each descriptor in `descriptor_queue`, find the matching tool result by `tool_ref`, run the descriptor method (e.g. `top_n_peaks`, `per_axis_summary`), and collect formatted observation strings. Tools without a descriptor get a default summary via `_summarize_result()`.
5. **Route** -- If iterations remain, go back to step 1. The plan node now sees the new observations. Otherwise, go to `summarize`.

---

### Agent State (`AgentState`)

All nodes read from and write to a shared `TypedDict` state. Key fields:

| Field | Type | Written by | Read by | Purpose |
|-------|------|-----------|---------|---------|
| `user_goal` | `str` | caller | `plan`, `summarize` | The user's analysis question |
| `csv_path` | `str` | caller | `load_signal` | Path to IMU CSV file |
| `fs` | `float` | `load_signal` | `execute_tool`, `plan` | Sampling frequency (Hz) |
| `acc`, `gyro` | `np.ndarray` | `load_signal` | `execute_tool` | Raw sensor arrays (N,3) |
| `signal_info` | `str` | `load_signal` | `plan`, `summarize` | Human-readable signal summary |
| `tool_queue` | `list[dict]` | `plan` | `execute_tool` | Parsed tool calls to execute |
| `descriptor_queue` | `list[dict]` | `plan` | `observe` | Parsed descriptor calls to apply |
| `tool_results` | `list[dict]` | `execute_tool` | `observe`, `summarize` | Accumulated results from all tools across all iterations |
| `results_per_iteration` | `list[int]` | `execute_tool` | `observe` | Count of results added per iteration (for slicing recent vs. old) |
| `intermediate_results` | `dict[str, ndarray]` | `execute_tool` | `execute_tool` | Named arrays from tool outputs, enabling tool-output chaining (e.g. `lowpass` output fed to `dominant_frequency`) |
| `observations` | `list[str]` | `observe` | `plan` | Formatted observation text from each iteration, fed back to the LLM |
| `plan_history` | `list[str]` | `plan` | `plan` | Previous raw LLM plan texts (prevents redundant tool calls) |
| `iteration` | `int` | `plan` | routing | Current iteration counter |
| `max_iterations` | `int` | `load_signal` | routing | Safety limit for plan-execute loops |
| `messages` | `list` | all nodes | LangGraph | LangChain message trace (`Annotated[list, add_messages]`) |
| `final_answer` | `str` | `summarize` | caller | LLM's final response |

---

### The Plan Node (detail)

**Input:** `signal_info`, `user_goal`, `observations` (from prior iterations), `intermediate_results` (available data sources), tool/descriptor registries.

**Prompt assembly:** The `plan.md` template is filled with `str.format()`:
- `{tool_list}` -- formatted table of all 17 tools from `TOOL_REGISTRY` (includes short aliases)
- `{descriptor_list}` -- formatted table of all 5 descriptors from `DESCRIPTOR_REGISTRY`
- `{observations}` -- cumulative observation text from all prior iterations (escaped braces)
- `{intermediates_text}` -- list of available intermediate data arrays with shapes
- `{valid_intermediates}` -- explicit list of valid DATA source names

**LLM output format** (instructed by `plan.md`):
```
OBSERVE:   (interpret prior results -- only on iteration 1+)
THINK:     (reasoning about the goal)
PIPELINE:  (numbered steps: input -> tool(args) -> output)
TOOLS:     (one TOOL/ARGS/DATA block per step)
DESCRIPTORS: (one DESCRIPTOR/TOOL_REF/ARGS block per tool)
DONE       (when analysis is complete)
```

**Parsing pipeline:**
1. `parse_tool_calls(llm_text)` -- Skips THINK/PIPELINE preamble, extracts TOOL/ARGS/DATA blocks via regex. Tool names are fuzzy-matched against `TOOL_REGISTRY` + `_TOOL_ALIASES` using: exact match, normalized match (strip underscores), containment match (substring), and Levenshtein distance (<=2 edits). Args are parsed from `key=value` pairs. Data sources are normalized. Results are deduplicated.
2. `parse_descriptor_calls(llm_text)` -- Finds the DESCRIPTORS section, extracts DESCRIPTOR/TOOL_REF/ARGS blocks. Descriptor names are fuzzy-matched against `DESCRIPTOR_REGISTRY`. Tool refs are fuzzy-matched against `TOOL_REGISTRY`.

**Output to state:** `tool_queue` (list of tool call dicts), `descriptor_queue` (list of descriptor call dicts), `plan_history` updated.

---

### The Execute Node (detail)

**Input:** `tool_queue`, `acc`, `gyro`, `timestamps`, `fs`, `intermediate_results`.

**For each tool call in the queue:**

1. **Resolve data source** -- The `data_source` field names where input comes from:
   - `"acc"` or `"gyro"` -- raw sensor arrays
   - An intermediate name (e.g. `"linear_acceleration"`, `"lowpass_acc"`) -- looked up in `intermediate_results`
   - If the name is garbled, `_fuzzy_match_data_source()` tries to recover it (same multi-stage strategy as tool matching)
   - If the source requires a prerequisite tool (e.g. `"linear_acceleration"` needs `imu_linear_acceleration`), the prerequisite is **auto-resolved** and run first

2. **Dispatch** -- IMU Fusion tools are called with `(acc=, gyro=, timestamps=)`. SciPy tools are called with `(data=, fs=, ...)` based on the tool's registered args.

3. **Store results:**
   - Raw result (dict or scalar) goes into `tool_results[]` with a text `summary`
   - Any `np.ndarray` outputs are stored in `intermediate_results` under descriptive names (e.g. `"lowpass_acc"`, `"linear_acceleration"`) for downstream tool chaining

**Output to state:** updated `tool_results`, `intermediate_results`, `results_per_iteration`.

---

### The Observe Node (detail)

**Input:** `descriptor_queue`, `tool_results`, `results_per_iteration`.

The observe node is the bridge between raw DSP output and the LLM's next planning step.
It transforms numerical tool outputs into concise text the LLM can reason about.

**Process:**

1. **Slice recent results** -- Uses `results_per_iteration[-1]` to get only the current iteration's tool results (not all accumulated results).

2. **Build lookup** -- `result_by_tool: dict` maps tool name to its most recent result entry.

3. **Run descriptors** -- For each descriptor in `descriptor_queue`:
   - Find the matching tool result by `tool_ref` name (with fuzzy fallback)
   - If found, call the descriptor method (e.g. `descriptor.top_n_peaks(raw_result, n=3)`) with the raw result dict/array
   - If the descriptor method doesn't exist, fall back to `_summarize_result()`
   - If no matching tool result is found, log a skip message

4. **Fallback summaries** -- Any tool results that weren't covered by a descriptor get a default summary (truncated to 500 chars).

5. **Format output** -- All observation lines are joined and stored. The `observe.md` template wraps them with iteration metadata.

**Output to state:** `observations` list (appended), `descriptor_queue` cleared.

**Known limitation:** Descriptors can only reference tool results from the **current** iteration. If the LLM requests a descriptor for a tool that ran in a previous iteration, it will be skipped with a "no matching tool result" message.

---

### The Descriptor System

Descriptors are a **formatting layer** between raw DSP tool outputs and LLM-readable text.
The LLM specifies in its plan which descriptor to apply to each tool's output, controlling
what information it receives in the next planning iteration.

Defined in `tools/descriptor_tool.py`. The `DescriptorTool` class contains methods that
accept a raw tool result (dict or ndarray) and return a concise text string.

**Available Descriptors (5):**

| Descriptor | Works with | Description | Args |
|------------|-----------|-------------|------|
| `top_n_peaks` | `fft_magnitude`, `dominant_frequency`, `peak_detection` | Extract top N frequencies by magnitude from spectral results | `n` (default 3) |
| `per_axis_summary` | `compute_statistics`, raw arrays | Per-axis summary with selectable metrics | `metrics` (list, default: mean,std,rms) |
| `compare_axes` | `compute_statistics`, `signal_energy`, raw arrays | Compare axes to find which has highest variance/energy | (none) |
| `value_at_frequency` | `fft_magnitude` | Report magnitude/energy at a specific frequency band | `target_freq` (Hz), `bandwidth` (Hz, default 0.5) |
| `signal_shape` | Any tool returning ndarray | Basic signal metadata: shape, range, mean, std per axis | (none) |

**How descriptors flow through the system:**

```
plan node                    execute_tool node           observe node
─────────                    ─────────────────           ────────────
LLM outputs:                 Runs tools, stores          Matches descriptors to
 DESCRIPTORS:                raw results in              tool results by tool_ref,
  DESCRIPTOR: top_n_peaks    tool_results[].raw          calls descriptor methods,
  TOOL_REF: fft_magnitude  ─────────────────────────►   produces formatted text
  ARGS: n=3                                              ──► observations[] ──► plan
                                                              (next iteration)
```

---

## Project Structure

```
dspagent/
├── dsp_agent.py              Main LangGraph agent (entry point)
├── example.py                Programmatic usage examples
├── llm_client.py             HTTP client for local LLM server
├── requirements.txt          Python dependencies
├── README.md                 This file
│
├── prompts/
│   ├── plan.md               Prompt template for the planning node
│   ├── observe.md            Prompt template for the observe node
│   └── summarize.md          Prompt template for the summarization node
│
├── tools/
│   ├── scipy_tool.py         SciPy DSP tools (12 operations + TOOL_REGISTRY)
│   ├── imufusion_tool.py     IMU Fusion AHRS tools (5 operations + IMU_TOOL_REGISTRY)
│   └── descriptor_tool.py    Descriptor tools (5 formatters + DESCRIPTOR_REGISTRY)
│
├── scripts/
│   ├── start_llm_server.py   FastAPI server for local LLM inference
│   ├── download_model.py     Download HuggingFace models
│
├── data/
│   ├── download_har_dataset.py  Download UCI HAR dataset + create CSVs
│   ├── samples/                 Sample CSV files (one per activity)
│   └── UCI HAR Dataset/         Raw dataset (created by download script)
│
│
└── models/
    ├── Mistral-7B-Instruct-v0.3/   Primary LLM (instruction-tuned)
    └── opt-6.7b/                    Fallback base model
```

---

## Available Tools (17 DSP/IMU + 5 Descriptors)

### SciPy DSP Tools (`tools/scipy_tool.py`)

**Filtering:**

| Tool | Description |
|------|-------------|
| `lowpass` | Butterworth lowpass filter — remove high-frequency noise |
| `highpass` | Butterworth highpass filter — remove DC/gravity drift |
| `bandpass` | Butterworth bandpass filter — isolate frequency range |

**Spectral Analysis:**

| Tool | Description |
|------|-------------|
| `fft_magnitude` | FFT magnitude spectrum — frequency content |
| `dominant_frequency` | Find strongest frequency per axis |
| `spectrogram` | Time-frequency spectrogram |

**Statistical Features:**

| Tool | Description |
|------|-------------|
| `compute_statistics` | Mean, std, min, max, rms, median per axis |
| `signal_magnitude_area` | SMA — overall activity level indicator |
| `zero_crossing_rate` | Periodicity indicator per axis |

**Signal Metrics:**

| Tool | Description |
|------|-------------|
| `compute_magnitude` | Vector magnitude from 3-axis → single axis |
| `signal_energy` | Energy in frequency domain per axis |
| `peak_detection` | Find peaks — step counting, event detection |

### IMU Fusion Tools (`tools/imufusion_tool.py`)

All IMU fusion tools use the [imufusion](https://github.com/xioTechnologies/Fusion) AHRS library to fuse accelerometer + gyroscope data. No magnetometer required (uses `update_no_magnetometer` when mag data is absent).

| Tool | Description |
|------|-------------|
| `imu_orientation` | Device orientation as quaternions (w, x, y, z) per sample |
| `imu_euler_angles` | Roll, pitch, yaw in degrees with statistics |
| `imu_earth_acceleration` | Acceleration rotated into earth frame (removes tilt effect) |
| `imu_linear_acceleration` | Pure motion acceleration — gravity removed via AHRS |
| `imu_gravity` | Estimated gravity vector per sample — tilt/orientation indicator |

---

## Prompt Templates

LLM prompts are stored as external Markdown files in `prompts/` for easy editing.
All templates use Python `str.format()` placeholders.

| Template | Used by node | Placeholders | Purpose |
|----------|-------------|--------------|---------|
| `prompts/plan.md` | `plan` | `{signal_info}`, `{user_goal}`, `{tool_list}`, `{descriptor_list}`, `{observations}`, `{intermediates_text}`, `{valid_intermediates}`, `{results_text}` | Full planning prompt: instructs the LLM to output THINK/PIPELINE/TOOLS/DESCRIPTORS sections. Includes the complete tool and descriptor registries, valid data sources, and prior observations. |
| `prompts/observe.md` | `observe` | `{iteration}`, `{iteration_observations}` | Wraps the formatted observation text with iteration metadata. Used to structure the observe node's output before it is fed back to the plan node. |
| `prompts/summarize.md` | `summarize` | `{signal_info}`, `{user_goal}`, `{results_text}` | Final answer prompt: instructs the LLM to base conclusions only on numerical tool results, not filenames or labels. |

---

## LLM Models

| Model | Type | Size | Notes |
|-------|------|------|-------|
| **Mistral-7B-Instruct-v0.3** | Instruction-tuned | 14.5 GB (fp16) | Primary — follows structured tool-call format, coherent analysis |
| facebook/opt-6.7b | Base completion | 13 GB (fp16) | Fallback — less structured, requires keyword-based parsing |

**Context window:** Mistral-7B supports **32,768 tokens**. Full tool data is fed to the LLM without truncation.

---

## Supported NVIDIA Hardware

Mistral-7B-Instruct in fp16 requires ~14.5 GB VRAM. Any NVIDIA GPU with sufficient memory can run the DSP Agent.

### Recommended GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| **NVIDIA DGX A100** | 40/80 GB per GPU | Data-center grade; can run multiple models simultaneously |
| **NVIDIA DGX H100** | 80 GB per GPU | Latest DGX; highest throughput for inference |
| **NVIDIA DGX B200** | 192 GB per GPU | Next-gen Blackwell DGX |
| **NVIDIA DGX Spark** | 128 GB unified | Desktop AI supercomputer; large unified memory pool |
| **NVIDIA RTX 4090** | 24 GB | Consumer flagship; comfortably fits Mistral-7B fp16 |
| **NVIDIA RTX 4080** | 16 GB | Fits Mistral-7B fp16 with limited headroom |
| **NVIDIA RTX 6000 Ada** | 48 GB | Professional workstation GPU |
| **NVIDIA A100 (standalone)** | 40/80 GB | Cloud and on-prem inference standard |
| **NVIDIA H100 (standalone)** | 80 GB | High-throughput inference |
| **NVIDIA L40S** | 48 GB | Cloud inference GPU |

### Minimum Requirements

- **VRAM:** ≥ 16 GB (fp16) or ≥ 8 GB (int8/int4 quantization)
- **CUDA:** ≥ 11.8
- **Driver:** ≥ 525.x

---

## Setup

### 1. Install Dependencies

```bash
conda activate base
pip install -r requirements.txt
```

### 2. Download the LLM Model

The `download_model.py` script supports presets and custom repo IDs.

```bash
# List available presets
python scripts/download_model.py --list

# Download Mistral-7B-Instruct (default)
python scripts/download_model.py

# Download using a named preset
python scripts/download_model.py --preset mistral
python scripts/download_model.py --preset opt
python scripts/download_model.py --preset llama2

# Download any HuggingFace model by repo ID
python scripts/download_model.py mistralai/Mistral-7B-Instruct-v0.3

# Custom output directory
python scripts/download_model.py --preset opt --out /tmp/my-model
```

| Preset | Repo ID | Notes |
|--------|---------|-------|
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | **Default** — recommended, instruction-tuned |
| `opt` | `facebook/opt-6.7b` | No HF token needed |
| `llama2` | `meta-llama/Llama-2-13b-hf` | Requires HF token with Llama access |

The script reads a HuggingFace token from `hf_token.txt` if present.

### 3. Start the LLM Server

```bash
# Default (auto-detects Mistral-7B-Instruct)
python scripts/start_llm_server.py

# Or specify a different model
MODEL_DIR="models/opt-6.7b" python scripts/start_llm_server.py
```

The server starts on `http://0.0.0.0:8080`.

### 4. Verify the Server

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "ok", "model_dir": "...", "instruct_mode": true}
```

### Server API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/v1/generate` | Raw prompt completion (works with any model) |
| `POST` | `/v1/chat` | Chat messages with template (best for instruct models) |

**`/v1/generate` payload:**
```json
{"prompt": "...", "max_tokens": 256, "temperature": 0.3}
```

**`/v1/chat` payload:**
```json
{"messages": [{"role": "user", "content": "..."}], "max_tokens": 256}
```

---

## Dataset Setup

Download and prepare the UCI HAR (Human Activity Recognition) dataset:

```bash
python data/download_har_dataset.py
```

This creates sample CSV files in `data/samples/`:

| File | Activity |
|------|----------|
| `imu_walking_subject1.csv` | Walking |
| `imu_sitting_subject1.csv` | Sitting |
| `imu_standing_subject1.csv` | Standing |
| `imu_laying_subject1.csv` | Laying |
| `imu_walking_upstairs_subject1.csv` | Walking upstairs |
| `imu_walking_downstairs_subject1.csv` | Walking downstairs |
| `imu_multi_activity.csv` | Multiple activities |

**CSV format:** `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`
**Sampling rate:** 50 Hz, 128 samples per window (2.56 seconds)

---

## Usage

### Command Line

```bash
python dsp_agent.py <csv_file> "<goal>"
```

**Examples:**

```bash
# Analyze walking data — detect activity and step frequency
python dsp_agent.py data/samples/imu_walking_subject1.csv \
    "What activity is the person doing? What is the step frequency?"

# Estimate device orientation using sensor fusion
python dsp_agent.py data/samples/imu_walking_subject1.csv \
    "Estimate the device orientation and extract linear acceleration."

# Check if person is stationary
python dsp_agent.py data/samples/imu_sitting_subject1.csv \
    "Is this person moving or stationary? Analyze the signal energy."

# Analyze multi-activity recording
python dsp_agent.py data/samples/imu_multi_activity.csv \
    "How many different activities are in this recording? Describe them."
```

### Programmatic

See `example.py` for a full working script. Quick snippet:

```python
import uuid
from dsp_agent import run_agent

result = run_agent(
    csv_path="data/samples/imu_walking_subject1.csv",
    user_goal="What activity is the person performing? Estimate the step frequency.",
    fs=50.0,                        # sampling frequency (0 = auto-detect)
    max_iterations=3,               # max plan-execute loops
    thread_id=str(uuid.uuid4()),    # unique thread for checkpointing
)

# Inspect results
print(result["final_answer"])     # LLM's analysis
print(result["iteration"])        # number of plan-execute loops used

# Iterate over tool results
for tr in result.get("tool_results", []):
    print(tr["tool"], tr.get("args", {}))
```

Run the full example:

```bash
python example.py
```

---

## Quick Start

```bash
# 1. Install dependencies
conda activate base
pip install -r requirements.txt

# 2. Start LLM server (ensure model is downloaded first)
python scripts/start_llm_server.py &

# 3. Download dataset
python data/download_har_dataset.py

# 4. Run the agent
python dsp_agent.py data/samples/imu_walking_subject1.csv \
    "Analyze this signal. What activity? What is the step frequency?"
```
