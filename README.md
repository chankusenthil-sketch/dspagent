# DSP Agent

AI-powered signal processing agent that analyzes IMU sensor signals using DSP tools and AHRS sensor fusion — planned and orchestrated by a local LLM.

## Project Goal

The LLM acts as a **thinking/planning agent**: it reads the user's goal, examines signal metadata, decides which DSP or sensor-fusion operations to apply, executes them, and interprets the results.

**Key capabilities:**
- Read IMU signals (accelerometer, gyroscope) from CSV files
- LLM-driven planning — the model decides what analysis to perform
- Iterative analysis: plan → execute → review → plan again (or finish)
- **17 tools**: 12 SciPy DSP tools + 5 IMU Fusion (AHRS) tools
- Stateful agent with memory via LangGraph `StateGraph` + `MemorySaver`
- Prompt templates loaded from external `.md` files (`prompts/`)
- Local LLM inference — no cloud API needed

---

## Architecture

```
User Goal + CSV File
      │
      ▼
┌─────────────┐
│ load_signal  │  Read CSV, auto-detect sampling rate, build signal summary
└──────┬──────┘
       ▼
┌─────────────┐
│    plan      │  LLM reads goal + signal info + prior results → picks tools
└──────┬──────┘    (prompt template: prompts/plan.md)
       │
  ┌────▼─────┐    yes    ┌──────────────┐
  │ tools    ├──────────►│ execute_tool  │  Run DSP / IMU Fusion tools
  │ queued?  │           └──────┬───────┘
  └────┬─────┘                  │
       │ no                     ▼
       │               ┌───────────────┐
       │               │ Back to plan?  │  If max iterations not reached
       │               └──────┬────────┘
       ▼                      │
┌─────────────┐               │
│  summarize   │◄─────────────┘  LLM generates final answer
└──────┬──────┘    (prompt template: prompts/summarize.md)
       ▼
  Final Answer
```

**LangGraph StateGraph nodes:**

| Node | Description |
|------|-------------|
| `load_signal` | Parse CSV into numpy arrays, compute signal metadata |
| `plan` | LLM selects DSP/fusion tools based on goal and current results |
| `execute_tool` | Run selected tools on signal data (SciPy or IMU Fusion) |
| `summarize` | LLM interprets all results and answers the user's question |

**Conditional edges:**
- `plan` → `execute_tool` (if tools are queued)
- `plan` → `summarize` (if LLM says DONE or max iterations reached)
- `execute_tool` → `plan` (loop back for more analysis)
- `execute_tool` → `summarize` (if max iterations reached)

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
│   └── summarize.md          Prompt template for the summarization node
│
├── tools/
│   ├── scipy_tool.py         SciPy DSP tools (12 operations + TOOL_REGISTRY)
│   └── imufusion_tool.py     IMU Fusion AHRS tools (5 operations + IMU_TOOL_REGISTRY)
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

## Available Tools (17 total)

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

LLM prompts are stored as external Markdown files in `prompts/` for easy editing:

- **`prompts/plan.md`** — System prompt + planning instructions. Tells the LLM to:
  - Use domain knowledge (IMU concepts, human motion patterns)
  - Ignore filename and ground-truth columns
  - Combine DSP + fusion tools for thorough analysis
  - Output structured `TOOL: / ARGS: / DATA:` format

- **`prompts/summarize.md`** — System prompt + summarization instructions. Tells the LLM to:
  - Base conclusions only on numerical tool results
  - Not reference filenames or label columns
  - Be specific with numbers and units

Templates use Python `str.format()` placeholders: `{signal_info}`, `{user_goal}`, `{results_text}`, `{tool_list}`.

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
