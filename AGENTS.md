# AGENTS.md — DSP Agent Codebase Guide

## Project Overview

AI-powered signal processing agent that analyzes IMU sensor signals using DSP tools
and AHRS sensor fusion, orchestrated by a local LLM via LangGraph. Python 3.10+.

## Project Structure

```
dsp_agent.py          # Main agent — LangGraph StateGraph (load_signal → plan → execute_tool → summarize)
llm_client.py         # HTTP client wrapper for local LLM server
example.py            # Programmatic usage examples
tools/
  scipy_tool.py       # 12 SciPy DSP tools (filtering, FFT, statistics, etc.)
  imufusion_tool.py   # 5 IMU Fusion AHRS tools (orientation, euler, gravity, etc.)
scripts/
  start_llm_server.py # FastAPI server serving HuggingFace model (Mistral-7B / OPT-6.7b)
  download_model.py   # CLI tool to download models from HuggingFace
prompts/
  plan.md             # LLM planning prompt template (str.format placeholders)
  summarize.md        # LLM summarization prompt template
data/samples/         # UCI HAR IMU CSV files (50 Hz, accelerometer + gyroscope)
models/               # Local model weights (Mistral-7B-Instruct-v0.3, opt-6.7b)
logs/                 # Runtime conversation and debug logs
```

Flat module layout — no `__init__.py` files, no Python package. Imports are direct
(e.g., `from tools.scipy_tool import SciPyDSPTool`).

## Build & Run Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download model
```bash
python scripts/download_model.py              # default: Mistral-7B-Instruct-v0.3
python scripts/download_model.py --model opt  # alternative: OPT-6.7b
```

### Start the LLM server
```bash
python scripts/start_llm_server.py
# Or with a custom model directory:
MODEL_DIR=/path/to/model python scripts/start_llm_server.py
```

### Run the agent
```bash
# CLI usage
python dsp_agent.py <csv_path> "<goal>"
python dsp_agent.py data/samples/imu_walking_subject1.csv "What activity is the person doing?"

# Programmatic usage
python example.py
```

### Environment variables
- `LLM_HOST` — LLM server host (default: `localhost`)
- `LLM_PORT` — LLM server port (default: `8080`)
- `MODEL_DIR` — Path to HuggingFace model directory for the server

## Tests

**There are no tests in this codebase.** No test framework is configured, no test files
exist, and no CI/CD pipeline is set up. If adding tests:

- Use `pytest` as the test framework (consistent with the Python ecosystem used here).
- Place tests in a `tests/` directory at the project root.
- Name test files `test_<module>.py` (e.g., `test_scipy_tool.py`).
- Run a single test: `pytest tests/test_scipy_tool.py::test_lowpass -v`
- Run all tests: `pytest tests/ -v`

## Linting & Formatting

**No linter or formatter is configured.** No `.flake8`, `ruff.toml`, `.pylintrc`,
or `pyproject.toml` exists. If adding tooling, follow the existing code style below.

## Code Style Guidelines

### Imports
- **Order**: Standard library, then third-party, then local modules. No blank line
  separators are used between groups (but adding them is acceptable).
- **Style**: Direct imports preferred. Use `from module import Class, CONSTANT`.
- **Example**:
  ```python
  import os
  import json
  import re
  import numpy as np
  from typing import Any, Dict, List, TypedDict, Annotated
  from langgraph.graph import StateGraph, START, END
  from llm_client import LLHTTPClient
  from tools.scipy_tool import SciPyDSPTool, TOOL_REGISTRY
  ```

### Naming Conventions
- **Functions/methods**: `snake_case` (e.g., `load_signal`, `parse_tool_calls`)
- **Private helpers**: Prefix with `_` (e.g., `_fuzzy_match_tool`, `_load_prompt`, `_log_message`)
- **Classes**: `PascalCase` (e.g., `SciPyDSPTool`, `IMUFusionTool`, `AgentState`)
- **Constants/registries**: `UPPER_CASE` (e.g., `TOOL_REGISTRY`, `IMU_TOOL_REGISTRY`)
- **Variables**: `snake_case`

### Type Hints
- Use type hints on all function parameters and return types.
- Use `typing` module types: `Dict[str, Any]`, `List`, `TypedDict`, `Annotated`.
- Python 3.10+ union syntax is used in newer code: `str | None` instead of `Optional[str]`.
- NumPy arrays are typed as `np.ndarray`. Generic array state fields use `Any`.
- Example:
  ```python
  def lowpass(self, data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
  ```

### Documentation
- Every module has a module-level docstring explaining its purpose.
- Public functions and classes have docstrings (single-line or multi-line).
- Section separators use banner comments: `# ---------------------------------------------------------------------------`
- Inline section labels use `# --- Section Name ---` within classes.

### Error Handling
- Use `try/except Exception as e` blocks that return error info in result dicts
  (e.g., `{"error": str(e)}`), rather than raising exceptions up to the caller.
- Use `ImportError` handling for optional dependencies (e.g., `imufusion`).
- HTTP client uses `resp.raise_for_status()` to propagate `requests.HTTPError`.
- No custom exception classes — keep it simple.

### Tool Pattern
Tools follow a consistent pattern:
1. A class containing related DSP methods (e.g., `SciPyDSPTool`, `IMUFusionTool`).
2. Each method accepts `np.ndarray` data and parameters, returns `np.ndarray` or `Dict[str, Any]`.
3. A module-level `TOOL_REGISTRY` dict maps tool names to descriptions and argument specs
   for the LLM planner. When adding new tools, register them in the appropriate registry.

### State Management
- Agent state is defined as a `TypedDict` (`AgentState`).
- LangGraph `StateGraph` nodes are plain functions that accept and return state dicts.
- `plan_history` (list of strings) accumulates previous LLM plan texts so the planner
  can review its own prior reasoning and avoid redundant tool calls across iterations.
- Conversation history uses `Annotated[list, add_messages]` for LangChain message merging.
- Memory checkpointing is via `MemorySaver`.

### Prompt Templates
- Stored as Markdown files in `prompts/` directory.
- Use `str.format()` placeholders (e.g., `{user_goal}`, `{tool_list}`, `{signal_info}`).
- Loaded via `_load_prompt(name)` helper.

### General Patterns
- Flat module structure — no nested packages. Keep it that way unless complexity demands it.
- Module-level singleton instances for tools: `dsp = SciPyDSPTool()`.
- Logging uses Python's `logging` module with dedicated loggers (e.g., `dsp_agent.conversation`).
- f-strings for all string formatting in code (except prompt templates which use `.format()`).
- Line length is not strictly enforced but generally stays under ~120 characters.
- No trailing whitespace. Files end with a newline.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` / `langchain_core` | Agent graph orchestration and message types |
| `scipy` | Signal processing (filtering, FFT, spectrogram) |
| `numpy` | Array operations |
| `imufusion` | AHRS sensor fusion (optional) |
| `transformers` / `torch` | Local LLM inference |
| `fastapi` / `uvicorn` | LLM HTTP server |
| `requests` | HTTP client for LLM server |
| `pydantic` | Request/response models for FastAPI |

## Hardware Requirements

NVIDIA GPU with >= 16 GB VRAM (fp16) for running the local LLM server.
