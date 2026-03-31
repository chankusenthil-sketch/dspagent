"""DSP Agent — LangGraph-based signal processing agent.

The LLM (OPT-6.7B served locally) acts as a planner:
  1. Reads the user's goal and signal metadata
  2. Plans which DSP tools to apply (and which descriptors to use for output formatting)
  3. Executes the tools
  4. Observe node runs descriptors on raw results to produce LLM-readable observations
  5. Summarizes and interprets results
  6. Decides if more analysis is needed (loop) or done

LangGraph StateGraph with:
  - load_signal: Read CSV into state
  - plan: LLM decides next DSP step and specifies output descriptors
  - execute_tool: Run the chosen DSP tool
  - observe: Run descriptor tools on raw results, produce formatted observations
  - summarize: LLM interprets all results and answers the user's question
"""

import os
import json
import re
import csv
import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from llm_client import LLHTTPClient
from tools.scipy_tool import SciPyDSPTool, TOOL_REGISTRY
from tools.imufusion_tool import IMUFusionTool, IMU_TOOL_REGISTRY
from tools.descriptor_tool import DescriptorTool, DESCRIPTOR_REGISTRY

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # User input
    user_goal: str
    csv_path: str
    fs: float  # sampling frequency

    # Loaded signal data
    timestamps: Any  # np.ndarray
    acc: Any          # np.ndarray (N,3)
    gyro: Any         # np.ndarray (N,3)
    activity_labels: Any  # list of strings
    signal_info: str  # text summary of loaded data

    # Planning & execution loop
    plan: str                    # current LLM plan text
    plan_history: list           # previous plan texts for LLM context across iterations
    tool_queue: list             # list of tool calls to execute
    current_tool: dict           # the tool being executed now
    tool_results: list           # accumulated results from all tools
    results_per_iteration: list  # number of tool results added per iteration (for slicing)
    intermediate_results: dict   # name → np.ndarray from previous tool outputs (for chaining)
    iteration: int               # current planning iteration
    max_iterations: int          # safety limit

    # Descriptor & Observe
    descriptor_queue: list       # parsed descriptor calls from plan
    observations: list           # formatted observation strings from observe node

    # Conversation trace (LangChain messages between nodes)
    messages: Annotated[list, add_messages]

    # Final output
    final_answer: str
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

dsp = SciPyDSPTool()
descriptor = DescriptorTool()
try:
    imu_fusion = IMUFusionTool()
except ImportError:
    imu_fusion = None

# Merge IMU Fusion tools into the unified registry
TOOL_REGISTRY.update(IMU_TOOL_REGISTRY)

# ---------------------------------------------------------------------------
# Conversation logger (separate from debug/print log)
# ---------------------------------------------------------------------------

_CONV_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_CONV_LOG_DIR, exist_ok=True)

conv_logger = logging.getLogger("dsp_agent.conversation")
conv_logger.setLevel(logging.INFO)
conv_logger.propagate = False  # don't pollute root logger

def _init_conv_log(tag: str = "") -> None:
    """Create a fresh file handler for each agent run."""
    # Remove old handlers
    for h in conv_logger.handlers[:]:
        conv_logger.removeHandler(h)
        h.close()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(_CONV_LOG_DIR, f"dsp_conversation_{ts}{suffix}.log")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    conv_logger.addHandler(fh)
    conv_logger.info("=== Conversation trace started ===")
    return path

def _log_message(msg) -> None:
    """Log a single LangChain message to the conversation log."""
    role = msg.__class__.__name__.replace("Message", "").upper()
    content = msg.content
    # Truncate very long tool outputs for readability
    if len(content) > 2000:
        content = content[:2000] + f"\n... [truncated, {len(msg.content)} chars total]"
    conv_logger.info(f"[{role}] {content}")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(name: str) -> str:
    """Load a prompt template from prompts/<name>.md."""
    path = os.path.join(_PROMPT_DIR, f"{name}.md")
    with open(path, "r") as f:
        return f.read()


def get_llm_client() -> LLHTTPClient:
    host = os.environ.get("LLM_HOST", "localhost")
    port = int(os.environ.get("LLM_PORT", "8081"))
    return LLHTTPClient(host=host, port=port, timeout=300)


def call_llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.3,
             system: str = None) -> str:
    """Call local LLM via chat endpoint and return response text."""
    client = get_llm_client()
    messages = []
    if system:
        messages.append({"role": "user", "content": system + "\n\n" + prompt})
    else:
        messages.append({"role": "user", "content": prompt})
    return client.chat(messages, max_tokens=max_tokens, temperature=temperature)


def format_tool_list() -> str:
    """Format available tools as a tabular list with exact names for the LLM prompt.
    
    Long IMU tool names also show a short alias that the system accepts.
    """
    # Build reverse alias map: canonical name -> shortest alias
    _shortest_alias: Dict[str, str] = {}
    for alias, canonical in _TOOL_ALIASES.items():
        if canonical not in _shortest_alias or len(alias) < len(_shortest_alias[canonical]):
            _shortest_alias[canonical] = alias

    # Find the longest tool name for alignment
    max_name_len = max(len(name) for name in TOOL_REGISTRY)
    lines = []
    for name, info in TOOL_REGISTRY.items():
        # Extract args list, excluding 'data' (always implicit)
        args = [a for a in info.get("args", []) if a != "data"]
        desc = info["description"].split(".")[0]  # First sentence only
        if args:
            args_str = f"  ARGS: {', '.join(args)}"
        else:
            args_str = ""
        # Show short alias for long names
        alias_hint = ""
        if name in _shortest_alias:
            alias_hint = f"  (or: {_shortest_alias[name]})"
        lines.append(f"  {name:<{max_name_len}}  — {desc}.{args_str}{alias_hint}")
    return "\n".join(lines)


def parse_tool_calls(llm_text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM output.
    
    Expected format from LLM (with THINK/PIPELINE/TOOLS sections):
      THINK:
      (reasoning text — skipped by parser)
      PIPELINE:
      Step 1: input -> tool_name(params) -> output
      (pipeline text — skipped by parser)
      TOOLS:
      TOOL: tool_name
      ARGS: arg1=value1, arg2=value2
      DATA: acc|gyro|intermediate_name
    or
      DONE
    """
    tools = []

    # Normalize LLM text: remove markdown escapes, extra whitespace
    cleaned = llm_text.replace("\\", "").replace("`", "").replace("*", "")

    def _normalise_data_source(raw: str) -> str:
        """Clean a raw DATA: capture into a normalised source name."""
        # Remove trailing keywords/phrases like "Data Source", "specific axis"
        s = re.split(r'\b(?:data\s*source|specific|axis|for|from|the|and|of)\b', raw, maxsplit=1, flags=re.IGNORECASE)[0]
        # Remove non-alpha chars, collapse whitespace/underscores
        s = re.sub(r'[^a-z0-9 _]', '', s.lower())
        s = re.sub(r'[\s_]+', '_', s).strip('_')
        return s if s else "acc"

    # Try to find TOOL: lines (handles "Tool:", "TOOL:", "ToOL:" etc.)
    # Capture everything after the colon up to a newline so we can normalise
    # names that the LLM splits with spaces (e.g. "im u_linear_acceleratoin").
    tool_pattern = re.compile(
        r'tool\s*:\s*([^\n]+)',
        re.IGNORECASE
    )
    data_pattern = re.compile(
        r'data\s*(?:source)?\s*:\s*([^\n]+)',
        re.IGNORECASE
    )
    args_pattern = re.compile(
        r'args?\s*[:\s]+(.+?)(?=\n|tool|data|done|$)',
        re.IGNORECASE
    )

    # Section header patterns for OBSERVE/THINK/PIPELINE/TOOLS/DESCRIPTORS
    section_header = re.compile(r'^(observe|think|pipeline|tools|descriptors)\s*:', re.IGNORECASE)

    # Split into lines and parse line by line
    lines = cleaned.split("\n")
    current_tool = None
    current_args = {}
    current_data = "acc"

    # Skip THINK and PIPELINE sections — only parse lines after reaching
    # the TOOLS: header (or the first TOOL: match if no header is found).
    in_preamble = True
    for line in lines:
        stripped = line.strip()

        # Detect section headers to track where we are
        header_match = section_header.match(stripped)
        if header_match:
            section_name = header_match.group(1).lower()
            if section_name == "tools":
                in_preamble = False
            else:
                # Inside THINK or PIPELINE — keep skipping
                in_preamble = True
            continue

        # While in THINK/PIPELINE preamble, skip lines unless we see a
        # direct TOOL: match (handles cases where LLM omits the TOOLS: header)
        if in_preamble:
            if tool_pattern.search(stripped):
                in_preamble = False
            else:
                continue

        tool_match = tool_pattern.search(line)
        if tool_match:
            # Save previous tool if exists
            if current_tool:
                tools.append({"tool": current_tool, "args": current_args, "data_source": current_data})

            # Extract and normalise the tool name.  The capture group may
            # contain the full rest-of-line, e.g.
            #   "im u_linear_acceleratoin ARGS: cutoff=5"
            # Strategy: strip trailing keywords, remove non-alpha chars,
            # collapse whitespace/underscores into single underscores.
            raw_tail = tool_match.group(1).strip().lower()
            # Chop off anything from the first ARGS/DATA/DONE keyword onward
            raw_tail = re.split(r'\b(?:args?|data|done)\b', raw_tail, maxsplit=1, flags=re.IGNORECASE)[0]
            # Remove markdown cruft, punctuation, special chars — keep letters, digits, spaces, underscores
            raw_tail = re.sub(r'[^a-z0-9 _]', '', raw_tail)
            # Collapse whitespace + underscores into single underscores
            raw_name = re.sub(r'[\s_]+', '_', raw_tail).strip('_')

            # Fuzzy match against registry
            matched = _fuzzy_match_tool(raw_name)
            if matched is None:
                print(f"  [Skip] Ignoring unknown tool '{raw_name}'")
                current_tool = None
            else:
                current_tool = matched
            current_args = {}
            current_data = "acc"

            # Check if ARGS and DATA are on the same line
            args_match = args_pattern.search(line[tool_match.end():])
            if args_match:
                current_args = _parse_args(args_match.group(1))
                # If data=<name> was embedded in ARGS, use it as data source
                if "_data_source" in current_args:
                    current_data = current_args.pop("_data_source")
            data_match = data_pattern.search(line[tool_match.end():])
            if data_match:
                current_data = _normalise_data_source(data_match.group(1))
            continue

        # Check for standalone ARGS/DATA lines
        if current_tool:
            args_match = args_pattern.search(line)
            if args_match and not tool_match:
                current_args = _parse_args(args_match.group(1))
                if "_data_source" in current_args:
                    current_data = current_args.pop("_data_source")
            data_match = data_pattern.search(line)
            if data_match:
                current_data = _normalise_data_source(data_match.group(1))

    # Don't forget the last tool
    if current_tool:
        tools.append({"tool": current_tool, "args": current_args, "data_source": current_data})

    # Deduplicate tools — allow same tool name with different args/data_source
    seen_keys = set()
    unique_tools = []
    for t in tools:
        if not t["tool"]:
            continue
        # Build a dedup key from tool name + data source + args
        dedup_key = (t["tool"], t.get("data_source", "acc"), tuple(sorted(t.get("args", {}).items())))
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            unique_tools.append(t)
    tools = unique_tools

    # If no structured format found, try to infer from keywords
    if not tools:
        text_lower = llm_text.lower()
        if "statistic" in text_lower or "mean" in text_lower or "std" in text_lower:
            tools.append({"tool": "compute_statistics", "args": {}, "data_source": "acc"})
        if "peak" in text_lower or "step" in text_lower:
            tools.append({"tool": "peak_detection", "args": {}, "data_source": "acc"})
        if "frequency" in text_lower or "fft" in text_lower or "spectrum" in text_lower:
            tools.append({"tool": "fft_magnitude", "args": {}, "data_source": "acc"})
        if "dominant" in text_lower:
            tools.append({"tool": "dominant_frequency", "args": {}, "data_source": "acc"})
        if "lowpass" in text_lower or "filter" in text_lower or "noise" in text_lower:
            tools.append({"tool": "lowpass", "args": {"cutoff": 10.0}, "data_source": "acc"})
        if "energy" in text_lower:
            tools.append({"tool": "signal_energy", "args": {}, "data_source": "acc"})
        if "magnitude" in text_lower and "signal_magnitude" not in text_lower:
            tools.append({"tool": "compute_magnitude", "args": {}, "data_source": "acc"})
        if "sma" in text_lower or "signal magnitude area" in text_lower or "activity level" in text_lower:
            tools.append({"tool": "signal_magnitude_area", "args": {}, "data_source": "acc"})
        if "zero crossing" in text_lower or "zero_crossing" in text_lower:
            tools.append({"tool": "zero_crossing_rate", "args": {}, "data_source": "acc"})
        if "highpass" in text_lower or "high pass" in text_lower:
            tools.append({"tool": "highpass", "args": {"cutoff": 1.0}, "data_source": "acc"})
        if "orientation" in text_lower or "fusion" in text_lower or "quaternion" in text_lower or "ahrs" in text_lower:
            tools.append({"tool": "imu_orientation", "args": {}, "data_source": "acc"})
        if "euler" in text_lower or "roll" in text_lower or "pitch" in text_lower or "yaw" in text_lower or "tilt" in text_lower:
            tools.append({"tool": "imu_euler_angles", "args": {}, "data_source": "acc"})
        if "earth" in text_lower and "accel" in text_lower:
            tools.append({"tool": "imu_earth_acceleration", "args": {}, "data_source": "acc"})
        if "linear" in text_lower and "accel" in text_lower:
            tools.append({"tool": "imu_linear_acceleration", "args": {}, "data_source": "acc"})
        if "gravity" in text_lower:
            tools.append({"tool": "imu_gravity", "args": {}, "data_source": "acc"})

    # Fallback: if still nothing, do basic analysis
    if not tools:
        tools.append({"tool": "compute_statistics", "args": {}, "data_source": "acc"})
        tools.append({"tool": "dominant_frequency", "args": {}, "data_source": "acc"})

    return tools


def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s) < len(t):
        return _levenshtein(t, s)
    if len(t) == 0:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, sc in enumerate(s):
        curr = [i + 1]
        for j, tc in enumerate(t):
            cost = 0 if sc == tc else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# Short aliases for long tool names that the LLM consistently misspells.
# Checked before fuzzy matching.
_TOOL_ALIASES: Dict[str, str] = {
    "linear_acc":       "imu_linear_acceleration",
    "linear_accel":     "imu_linear_acceleration",
    "linear_acceleration": "imu_linear_acceleration",
    "earth_acc":        "imu_earth_acceleration",
    "earth_accel":      "imu_earth_acceleration",
    "earth_acceleration": "imu_earth_acceleration",
    "orientation":      "imu_orientation",
    "euler":            "imu_euler_angles",
    "euler_angles":     "imu_euler_angles",
    "gravity":          "imu_gravity",
    "stats":            "compute_statistics",
    "statistics":       "compute_statistics",
    "magnitude":        "compute_magnitude",
    "mag":              "compute_magnitude",
    "sma":              "signal_magnitude_area",
    "zcr":              "zero_crossing_rate",
    "fft":              "fft_magnitude",
    "peaks":            "peak_detection",
}


def _fuzzy_match_tool(raw_name: str) -> str | None:
    """Fuzzy match a (possibly misspelled) tool name to the registry.

    Match strategy (first match wins):
    0. Alias match — check against _TOOL_ALIASES.
    1. Exact match against TOOL_REGISTRY keys.
    2. Normalized match — strip underscores/hyphens/spaces from both sides.
    3. Containment match — check if a registered tool name (normalized) is a
       substring of the raw name (normalized), or vice-versa.  Pick the
       longest matching registered name to avoid short-name false positives.
       Both raw name and registered name must be >= 6 chars (normalized).
    4. Typo match — allow up to 2 character edits (Levenshtein) between
       normalized names.  Only considered when names are similar length.
    """
    # 0. Alias match — check short aliases first
    if raw_name in _TOOL_ALIASES:
        matched = _TOOL_ALIASES[raw_name]
        print(f"  [Alias] '{raw_name}' -> '{matched}'")
        return matched
    # Also try normalised alias lookup
    normalized_alias = raw_name.replace("_", "").replace("-", "").replace(" ", "")
    for alias, target in _TOOL_ALIASES.items():
        if alias.replace("_", "") == normalized_alias:
            print(f"  [Alias] '{raw_name}' -> '{target}'")
            return target

    # 1. Exact match
    if raw_name in TOOL_REGISTRY:
        return raw_name

    # 2. Normalized match: remove underscores/hyphens/spaces, lowercase
    normalized = raw_name.replace("_", "").replace("-", "").replace(" ", "")
    for registered in TOOL_REGISTRY:
        if registered.replace("_", "") == normalized:
            print(f"  [Fuzzy] '{raw_name}' matched to '{registered}'")
            return registered

    # 3. Containment match — longest registered name wins
    if len(normalized) >= 6:
        best: str | None = None
        best_len = 0
        for registered in TOOL_REGISTRY:
            reg_norm = registered.replace("_", "")
            if len(reg_norm) < 6:
                continue
            if reg_norm in normalized or normalized in reg_norm:
                if len(reg_norm) > best_len:
                    best = registered
                    best_len = len(reg_norm)
        if best is not None:
            print(f"  [Fuzzy] '{raw_name}' matched to '{best}' (containment)")
            return best

    # 4. Typo match — Levenshtein distance <= 2 for similar-length names
    if len(normalized) >= 6:
        best_typo: str | None = None
        best_dist = 3  # threshold: accept dist <= 2
        for registered in TOOL_REGISTRY:
            reg_norm = registered.replace("_", "")
            # Only consider names of similar length (within 2 chars)
            if abs(len(reg_norm) - len(normalized)) > 2:
                continue
            dist = _levenshtein(normalized, reg_norm)
            if dist < best_dist:
                best_dist = dist
                best_typo = registered
        if best_typo is not None:
            print(f"  [Fuzzy] '{raw_name}' matched to '{best_typo}' (typo, dist={best_dist})")
            return best_typo

    print(f"  [Skip] No match for tool '{raw_name}'")
    return None


def _fuzzy_match_data_source(raw_source: str, valid_sources: List[str]) -> str | None:
    """Fuzzy match a (possibly garbled) data source name to a valid source.

    The LLM often produces garbled data source names like 'filtered_',
    'linear_accuracy', 'linear_acc'.  This function tries to recover the
    intended source using the same multi-stage strategy as _fuzzy_match_tool.

    *valid_sources* is the list of currently valid names — typically
    ["acc", "gyro"] plus the keys of intermediate_results plus the
    keys of _PREREQUISITE_TOOLS (canonical intermediate names).

    Returns the matched source name, or None if no match is found.
    """
    raw = raw_source.strip().lower()

    # 1. Exact match
    if raw in valid_sources:
        return raw

    # 2. Normalised match — strip underscores/hyphens/spaces
    normalized = raw.replace("_", "").replace("-", "").replace(" ", "")
    for src in valid_sources:
        if src.replace("_", "") == normalized:
            print(f"  [FuzzyData] '{raw_source}' matched to '{src}'")
            return src

    # 3. Containment match — pick longest matching source name.
    #    Require the shorter string to be at least 50% of the longer one
    #    to avoid false positives like "acc" matching "linear_accuracy".
    if len(normalized) >= 3:
        best: str | None = None
        best_len = 0
        for src in valid_sources:
            src_norm = src.replace("_", "")
            if len(src_norm) < 3:
                continue
            shorter = min(len(src_norm), len(normalized))
            longer = max(len(src_norm), len(normalized))
            if shorter < longer * 0.5:
                continue
            if src_norm in normalized or normalized in src_norm:
                if len(src_norm) > best_len:
                    best = src
                    best_len = len(src_norm)
        if best is not None:
            print(f"  [FuzzyData] '{raw_source}' matched to '{best}' (containment)")
            return best

    # 4. Typo match — Levenshtein distance <= 3 for similar-length names.
    #    We allow a slightly larger distance than for tool names because
    #    the LLM garbles intermediate data names more aggressively.
    if len(normalized) >= 4:
        best_typo: str | None = None
        best_dist = 4  # threshold: accept dist <= 3
        for src in valid_sources:
            src_norm = src.replace("_", "")
            if abs(len(src_norm) - len(normalized)) > 3:
                continue
            dist = _levenshtein(normalized, src_norm)
            if dist < best_dist:
                best_dist = dist
                best_typo = src
        if best_typo is not None:
            print(f"  [FuzzyData] '{raw_source}' matched to '{best_typo}' (typo, dist={best_dist})")
            return best_typo

    return None


def _parse_args(args_str: str) -> dict:
    """Parse argument string like 'fs=50, cutoff=10' into a dict.

    If the args string contains 'data=<name>', it is extracted and stored
    under the special key '_data_source' so the caller can use it for
    tool-output chaining.
    """
    args: Dict[str, Any] = {}
    # Extract data=<name> before removing it
    data_match = re.search(r'data\s*=\s*(\w+)', args_str, flags=re.IGNORECASE)
    if data_match:
        args["_data_source"] = data_match.group(1).strip().lower()
    args_str = re.sub(r'data\s*=\s*\w+', '', args_str, flags=re.IGNORECASE)
    for part in re.split(r'[,;]', args_str):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower()
            v = v.strip().strip("'\"")
            # Only keep numeric args we care about
            if k in ("fs", "cutoff", "low", "high", "order", "nperseg", "height", "distance"):
                try:
                    val = float(v)
                    # SciPy expects these as int
                    if k in ("order", "nperseg", "distance"):
                        val = int(val)
                    args[k] = val
                except ValueError:
                    pass
    return args


def is_done(llm_text: str) -> bool:
    """Check if LLM says analysis is complete."""
    text_lower = llm_text.lower()
    return "done" in text_lower and "tool:" not in text_lower


def parse_descriptor_calls(llm_text: str) -> List[Dict[str, Any]]:
    """Parse descriptor calls from LLM output.

    Expected format in the DESCRIPTORS: section:
      DESCRIPTOR: descriptor_name
      TOOL_REF: tool_name
      ARGS: param=value, param2=value2

    Returns a list of dicts:
      [{"descriptor": "top_n_peaks", "tool_ref": "fft_magnitude", "args": {"n": 3}}, ...]
    """
    descriptors = []

    # Normalize LLM text
    cleaned = llm_text.replace("\\", "").replace("`", "").replace("*", "")

    # Find the DESCRIPTORS: section
    desc_section_match = re.search(r'descriptors\s*:', cleaned, re.IGNORECASE)
    if not desc_section_match:
        return []

    # Extract text from DESCRIPTORS: header to end (or next major section)
    section_text = cleaned[desc_section_match.end():]
    # Stop at DONE or end of text
    done_match = re.search(r'\bDONE\b', section_text, re.IGNORECASE)
    if done_match:
        section_text = section_text[:done_match.start()]

    descriptor_pattern = re.compile(r'descriptor\s*:\s*([^\n]+)', re.IGNORECASE)
    tool_ref_pattern = re.compile(r'tool_ref\s*:\s*([^\n]+)', re.IGNORECASE)
    args_pattern = re.compile(r'args?\s*:\s*([^\n]+)', re.IGNORECASE)

    current_desc = None
    current_tool_ref = None
    current_args = {}

    for line in section_text.split("\n"):
        stripped = line.strip()

        desc_match = descriptor_pattern.search(stripped)
        if desc_match:
            # Save previous descriptor if exists
            if current_desc and current_tool_ref:
                descriptors.append({
                    "descriptor": current_desc,
                    "tool_ref": current_tool_ref,
                    "args": current_args,
                })

            raw_name = desc_match.group(1).strip().lower()
            raw_name = re.sub(r'[^a-z0-9 _]', '', raw_name)
            raw_name = re.sub(r'[\s_]+', '_', raw_name).strip('_')
            current_desc = _fuzzy_match_descriptor(raw_name)
            current_tool_ref = None
            current_args = {}
            continue

        ref_match = tool_ref_pattern.search(stripped)
        if ref_match and current_desc:
            raw_ref = ref_match.group(1).strip().lower()
            raw_ref = re.sub(r'[^a-z0-9 _]', '', raw_ref)
            raw_ref = re.sub(r'[\s_]+', '_', raw_ref).strip('_')
            # Fuzzy match against tool registry
            matched = _fuzzy_match_tool(raw_ref)
            current_tool_ref = matched if matched else raw_ref
            continue

        args_match = args_pattern.search(stripped)
        if args_match and current_desc:
            current_args = _parse_descriptor_args(args_match.group(1))
            continue

    # Don't forget the last descriptor
    if current_desc and current_tool_ref:
        descriptors.append({
            "descriptor": current_desc,
            "tool_ref": current_tool_ref,
            "args": current_args,
        })

    return descriptors


def _fuzzy_match_descriptor(raw_name: str) -> str | None:
    """Fuzzy match a descriptor name against DESCRIPTOR_REGISTRY.

    Uses the same multi-stage strategy as _fuzzy_match_tool.
    """
    # 1. Exact match
    if raw_name in DESCRIPTOR_REGISTRY:
        return raw_name

    # 2. Normalized match
    normalized = raw_name.replace("_", "").replace("-", "").replace(" ", "")
    for registered in DESCRIPTOR_REGISTRY:
        if registered.replace("_", "") == normalized:
            return registered

    # 3. Containment match
    if len(normalized) >= 4:
        best: str | None = None
        best_len = 0
        for registered in DESCRIPTOR_REGISTRY:
            reg_norm = registered.replace("_", "")
            if reg_norm in normalized or normalized in reg_norm:
                if len(reg_norm) > best_len:
                    best = registered
                    best_len = len(reg_norm)
        if best is not None:
            return best

    # 4. Typo match
    if len(normalized) >= 4:
        best_typo: str | None = None
        best_dist = 3
        for registered in DESCRIPTOR_REGISTRY:
            reg_norm = registered.replace("_", "")
            if abs(len(reg_norm) - len(normalized)) > 2:
                continue
            dist = _levenshtein(normalized, reg_norm)
            if dist < best_dist:
                best_dist = dist
                best_typo = registered
        if best_typo is not None:
            return best_typo

    print(f"  [Skip] No match for descriptor '{raw_name}'")
    return None


def _parse_descriptor_args(args_str: str) -> Dict[str, Any]:
    """Parse descriptor argument string into a dict.

    Handles: n=3, metrics=mean,std,rms, target_freq=1.5, bandwidth=0.5
    """
    args: Dict[str, Any] = {}
    for part in re.split(r'[;]', args_str):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower()
            v = v.strip().strip("'\"")
            if k == "metrics":
                # Parse comma-separated metric names
                args[k] = [m.strip() for m in v.split(",")]
            elif k in ("n", "target_freq", "bandwidth"):
                try:
                    val = float(v)
                    if k == "n":
                        val = int(val)
                    args[k] = val
                except ValueError:
                    pass
    return args


def format_descriptor_list() -> str:
    """Format available descriptors as a list for the LLM prompt."""
    lines = []
    for name, info in DESCRIPTOR_REGISTRY.items():
        args = info.get("args", [])
        desc = info["description"]
        args_str = f"  ARGS: {', '.join(args)}" if args else ""
        lines.append(f"  {name} — {desc}{args_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def load_signal(state: AgentState) -> dict:
    """Load IMU signal from CSV file."""
    csv_path = state["csv_path"]
    if not os.path.exists(csv_path):
        return {"error": f"File not found: {csv_path}"}

    timestamps, acc_x, acc_y, acc_z = [], [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    activities = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            acc_x.append(float(row["acc_x"]))
            acc_y.append(float(row["acc_y"]))
            acc_z.append(float(row["acc_z"]))
            gyro_x.append(float(row["gyro_x"]))
            gyro_y.append(float(row["gyro_y"]))
            gyro_z.append(float(row["gyro_z"]))
            if "activity" in row:
                activities.append(row["activity"])

    timestamps_arr = np.array(timestamps)
    acc = np.column_stack([acc_x, acc_y, acc_z])
    gyro = np.column_stack([gyro_x, gyro_y, gyro_z])

    # Infer sampling frequency
    fs = state.get("fs", 0.0)
    if fs <= 0 and len(timestamps) > 1:
        dt = np.median(np.diff(timestamps_arr))
        fs = 1.0 / dt if dt > 0 else 50.0

    n_samples = len(timestamps)
    duration = timestamps_arr[-1] - timestamps_arr[0] if n_samples > 1 else 0
    unique_activities = list(set(activities)) if activities else ["unknown"]

    signal_info = (
        f"Loaded {n_samples} samples from {os.path.basename(csv_path)}. "
        f"Duration: {duration:.2f}s, Sampling rate: {fs:.1f} Hz. "
        f"Channels: acc (x,y,z), gyro (x,y,z). "
        f"Acc range: [{acc.min():.3f}, {acc.max():.3f}] g. "
        f"Gyro range: [{gyro.min():.3f}, {gyro.max():.3f}] rad/s. "
        f"Activities: {', '.join(unique_activities)}."
    )

    return {
        "timestamps": timestamps_arr,
        "acc": acc,
        "gyro": gyro,
        "activity_labels": activities,
        "fs": fs,
        "signal_info": signal_info,
        "tool_results": [],
        "results_per_iteration": [],
        "intermediate_results": {},
        "plan_history": [],
        "descriptor_queue": [],
        "observations": [],
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 5),
        "error": "",
        "messages": [
            HumanMessage(content=f"Goal: {state['user_goal']}\nFile: {state['csv_path']}"),
            AIMessage(content=f"[load_signal] {signal_info}"),
        ],
    }


def plan(state: AgentState) -> dict:
    """LLM plans what DSP tools to use next based on the goal and current results."""
    iteration = state.get("iteration", 0)
    tool_results = state.get("tool_results", [])
    plan_history = state.get("plan_history", [])

    # Build intermediate results listing so LLM knows what chained data is available
    intermediate_results = state.get("intermediate_results", {})
    intermediates_text = ""
    if intermediate_results:
        intermediates_text = "\nAVAILABLE INTERMEDIATE DATA (created by previous tools):\n"
        for name, arr in intermediate_results.items():
            if isinstance(arr, np.ndarray):
                intermediates_text += f"- \"{name}\"  shape={arr.shape}\n"
            else:
                intermediates_text += f"- \"{name}\"\n"

    # Build explicit list of valid DATA sources
    valid_intermediates = ""
    if intermediate_results:
        names = [f'"{name}"' for name in intermediate_results.keys()]
        valid_intermediates = "- Intermediate data from previous tools: " + ", ".join(names) + "\n"

    # --- Build observations text from the observe node ---
    # On iteration 0 (first plan), there are no observations yet.
    # On iteration 1+, observations come from the observe node's output.
    observations_text = ""
    observations_list = state.get("observations", [])
    if observations_list:
        # Build a cumulative observations section showing all iterations
        obs_parts = []
        for obs_idx, obs in enumerate(observations_list):
            obs_parts.append(f"--- Iteration {obs_idx + 1} Results ---\n{obs}")
        observations_text = "\n\n".join(obs_parts)
        # Escape braces so str.format() doesn't choke on JSON in observations
        observations_text = observations_text.replace("{", "{{").replace("}", "}}")

    # On first iteration (no results yet), results_text is empty
    results_text = ""

    # Load prompt template from prompts/plan.md
    template = _load_prompt("plan")
    prompt = template.format(
        signal_info=state.get('signal_info', ''),
        user_goal=state['user_goal'],
        results_text=results_text,
        tool_list=format_tool_list(),
        descriptor_list=format_descriptor_list(),
        intermediates_text=intermediates_text,
        valid_intermediates=valid_intermediates,
        observations=observations_text,
    )

    llm_response = call_llm(prompt, max_tokens=1024, temperature=0.2)
    print(f"\n[Plan iteration {iteration}] LLM response:\n{llm_response}\n")

    # Check if LLM says done
    if is_done(llm_response) and len(tool_results) > 0:
        return {
            "plan": llm_response,
            "plan_history": plan_history + [llm_response],
            "tool_queue": [],
            "descriptor_queue": [],
            "iteration": iteration + 1,
            "messages": [
                AIMessage(content=f"[plan iter={iteration}] LLM says DONE.\n{llm_response}"),
            ],
        }

    # Parse tool calls and descriptor calls
    tools = parse_tool_calls(llm_response)
    descriptors = parse_descriptor_calls(llm_response)
    tool_names = [t["tool"] for t in tools]

    return {
        "plan": llm_response,
        "plan_history": plan_history + [llm_response],
        "tool_queue": tools,
        "descriptor_queue": descriptors,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(content=f"[plan iter={iteration}] Tools chosen: {tool_names}\n{llm_response}"),
        ],
    }


def execute_tool(state: AgentState) -> dict:
    """Execute all queued DSP tools and collect results."""
    # Mapping from tool name to the dict key holding its primary array output.
    # Used to auto-store intermediate results for tool-output chaining.
    _INTERMEDIATE_KEYS: Dict[str, str] = {
        "imu_linear_acceleration": "linear_acceleration",
        "imu_earth_acceleration": "earth_acceleration",
        "imu_orientation": "quaternions",
        "imu_euler_angles": "euler_angles",
        "imu_gravity": "gravity",
    }

    # Reverse mapping: intermediate name -> tool that produces it.
    # Used for auto-resolving missing prerequisites (one level deep).
    # Only parameter-free IMU fusion tools are auto-resolvable.
    _PREREQUISITE_TOOLS: Dict[str, str] = {
        "linear_acceleration": "imu_linear_acceleration",
        "earth_acceleration": "imu_earth_acceleration",
        "quaternions": "imu_orientation",
        "euler_angles": "imu_euler_angles",
        "gravity": "imu_gravity",
    }

    tool_queue = state.get("tool_queue", [])
    tool_results = list(state.get("tool_results", []))
    acc = state.get("acc")
    gyro = state.get("gyro")
    fs = state.get("fs", 50.0)
    timestamps = state.get("timestamps")
    intermediate_results: Dict[str, Any] = dict(state.get("intermediate_results", {}))
    results_before = len(tool_results)

    for tool_call in tool_queue:
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})
        data_source = tool_call.get("data_source", "acc")

        # --- Normalise garbled DATA source names --------------------------
        # The LLM often garbles data source names (e.g. "filtered_",
        # "linear_accuracy", "low pass_filter").  Apply the same
        # normalisation we do for tool names, then fuzzy-match.
        if data_source not in ("acc", "gyro") and data_source not in intermediate_results and data_source not in _PREREQUISITE_TOOLS:
            # Normalise: strip markdown cruft, collapse spaces/underscores
            ds_clean = re.sub(r'[^a-z0-9 _]', '', data_source.lower())
            ds_clean = re.sub(r'[\s_]+', '_', ds_clean).strip('_')
            # Build list of all currently valid sources
            valid_sources = ["acc", "gyro"] + list(intermediate_results.keys()) + list(_PREREQUISITE_TOOLS.keys())
            matched_ds = _fuzzy_match_data_source(ds_clean, valid_sources)
            if matched_ds is not None:
                data_source = matched_ds
            # else: leave as-is; will hit the fallback-to-acc path below

        # Select data — check intermediate results first, then fall back to raw
        if data_source in intermediate_results:
            data = intermediate_results[data_source]
        elif data_source == "gyro":
            data = gyro
        elif data_source == "acc":
            data = acc
        elif data_source in _PREREQUISITE_TOOLS:
            # Auto-resolve: run the prerequisite IMU fusion tool to produce the data
            prereq_name = _PREREQUISITE_TOOLS[data_source]
            print(f"  [Auto] Running prerequisite '{prereq_name}' to produce '{data_source}'")
            if imu_fusion is None:
                tool_results.append({
                    "tool": tool_name,
                    "summary": f"Error: need '{prereq_name}' for '{data_source}' but imufusion not installed",
                    "raw": None,
                })
                continue
            prereq_method = getattr(imu_fusion, prereq_name, None)
            if prereq_method is None:
                tool_results.append({
                    "tool": tool_name,
                    "summary": f"Error: prerequisite tool '{prereq_name}' not found",
                    "raw": None,
                })
                continue
            try:
                prereq_result = prereq_method(acc=acc, gyro=gyro, timestamps=timestamps)
                # Store all arrays from the prerequisite result
                if isinstance(prereq_result, dict):
                    key = _INTERMEDIATE_KEYS.get(prereq_name)
                    if key and key in prereq_result and isinstance(prereq_result[key], np.ndarray):
                        intermediate_results[key] = prereq_result[key]
                    for rk, rv in prereq_result.items():
                        if isinstance(rv, np.ndarray):
                            intermediate_results[f"{prereq_name}_{rk}"] = rv
                elif isinstance(prereq_result, np.ndarray):
                    intermediate_results[data_source] = prereq_result
                # Log the auto-resolved prerequisite
                prereq_summary = _summarize_result(prereq_name, prereq_result, "acc")
                tool_results.append({
                    "tool": prereq_name,
                    "summary": f"(auto) {prereq_summary}",
                    "raw": prereq_result if not isinstance(prereq_result, np.ndarray) else None,
                    "data_source": "acc",
                })
                print(f"  [Auto] {prereq_name}(acc): {prereq_summary}")
            except Exception as e:
                tool_results.append({
                    "tool": tool_name,
                    "summary": f"Error: auto-prerequisite '{prereq_name}' failed: {str(e)}",
                    "raw": None,
                })
                continue
            # Now fetch the data we needed
            if data_source in intermediate_results:
                data = intermediate_results[data_source]
            else:
                tool_results.append({
                    "tool": tool_name,
                    "summary": f"Error: prerequisite '{prereq_name}' did not produce '{data_source}'",
                    "raw": None,
                })
                continue
        else:
            # Unknown data source — fall back to acc with a warning
            print(f"  [Warn] Unknown data source '{data_source}', falling back to acc")
            data = acc

        if data is None:
            tool_results.append({
                "tool": tool_name,
                "summary": "Error: no signal data loaded",
                "raw": None,
            })
            continue

        try:
            # --- IMU Fusion tools (need acc + gyro + timestamps) ---
            if tool_name in IMU_TOOL_REGISTRY:
                if imu_fusion is None:
                    tool_results.append({
                        "tool": tool_name,
                        "summary": "Error: imufusion library not installed",
                        "raw": None,
                    })
                    continue
                method = getattr(imu_fusion, tool_name, None)
                if method is None:
                    tool_results.append({
                        "tool": tool_name,
                        "summary": f"Error: unknown IMU fusion tool '{tool_name}'",
                        "raw": None,
                    })
                    continue
                result = method(acc=acc, gyro=gyro, timestamps=timestamps)

            else:
                # --- SciPy DSP tools ---
                method = getattr(dsp, tool_name, None)
                if method is None:
                    tool_results.append({
                        "tool": tool_name,
                        "summary": f"Error: unknown tool '{tool_name}'",
                        "raw": None,
                    })
                    continue

                # Build kwargs based on what the method needs
                required_args = TOOL_REGISTRY.get(tool_name, {}).get("args", [])
                kwargs = {}
                if "data" in required_args:
                    kwargs["data"] = data
                if "fs" in required_args:
                    kwargs["fs"] = args.get("fs", fs)
                if "cutoff" in required_args:
                    kwargs["cutoff"] = args.get("cutoff", fs / 5)
                if "low" in required_args:
                    kwargs["low"] = args.get("low", 0.5)
                if "high" in required_args:
                    kwargs["high"] = args.get("high", fs / 4)

                result = method(**kwargs)

            # Store intermediate results for tool-output chaining
            if isinstance(result, np.ndarray):
                inter_name = f"{tool_name}_{data_source}"
                intermediate_results[inter_name] = result
            elif isinstance(result, dict):
                if tool_name in _INTERMEDIATE_KEYS:
                    key = _INTERMEDIATE_KEYS[tool_name]
                    if key in result and isinstance(result[key], np.ndarray):
                        intermediate_results[key] = result[key]
                # Also store any array values with descriptive names
                for rk, rv in result.items():
                    if isinstance(rv, np.ndarray):
                        intermediate_results[f"{tool_name}_{rk}"] = rv

            # Summarize result for LLM consumption
            summary = _summarize_result(tool_name, result, data_source)
            tool_results.append({
                "tool": tool_name,
                "summary": summary,
                "raw": result if not isinstance(result, np.ndarray) else None,
                "data_source": data_source,
            })
            print(f"  [Tool] {tool_name}({data_source}): {summary}")

        except Exception as e:
            tool_results.append({
                "tool": tool_name,
                "summary": f"Error: {str(e)}",
                "raw": None,
            })

    # Track how many results were added in this iteration
    results_this_iteration = len(tool_results) - results_before
    results_per_iteration = list(state.get("results_per_iteration", []))
    results_per_iteration.append(results_this_iteration)

    # Build ToolMessages for conversation trace
    tool_messages = []
    for tr in tool_results[results_before:]:
        tool_messages.append(
            ToolMessage(
                content=f"[{tr['tool']}] {tr['summary']}",
                tool_call_id=tr["tool"],
            )
        )

    return {
        "tool_results": tool_results,
        "results_per_iteration": results_per_iteration,
        "tool_queue": [],
        "messages": tool_messages,
        "intermediate_results": intermediate_results,
    }


def _summarize_result(tool_name: str, result: Any, data_source: str) -> str:
    """Create a concise text summary of a tool result for LLM consumption."""
    if isinstance(result, dict):
        # Include full data — Mistral-7B supports 32K context
        summary_dict = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                summary_dict[k] = np.round(v, 6).tolist()
            elif isinstance(v, dict):
                summary_dict[k] = v
            elif isinstance(v, list):
                summary_dict[k] = v
            else:
                summary_dict[k] = v
        return json.dumps(summary_dict, default=str)
    elif isinstance(result, np.ndarray):
        return f"{data_source} filtered: shape={result.shape}, mean={result.mean():.6f}, std={result.std():.6f}, min={result.min():.6f}, max={result.max():.6f}"
    elif isinstance(result, (int, float)):
        return f"{tool_name}={result:.4f}"
    else:
        return str(result)


def observe(state: AgentState) -> dict:
    """Run descriptor tools on raw results to produce LLM-readable observations.

    For each descriptor in the queue, finds the matching tool result by
    tool_ref name, runs the descriptor method on the raw result, and
    collects the formatted output string.  Tools without a matching
    descriptor get a fallback summary via _summarize_result().
    """
    descriptor_queue = state.get("descriptor_queue", [])
    tool_results = state.get("tool_results", [])
    results_per_iteration = state.get("results_per_iteration", [])
    plan_history = state.get("plan_history", [])
    iteration = state.get("iteration", 0)

    # Get the most recent iteration's results
    if results_per_iteration:
        last_count = results_per_iteration[-1]
        recent_results = tool_results[-last_count:] if last_count > 0 else []
    else:
        recent_results = tool_results

    # Build a lookup: tool_name -> most recent result entry
    result_by_tool: Dict[str, dict] = {}
    for r in recent_results:
        result_by_tool[r["tool"]] = r

    # Track which tools have been described by a descriptor
    described_tools = set()

    observation_lines = []

    # --- Run descriptors on matching tool results ---
    for desc_call in descriptor_queue:
        desc_name = desc_call.get("descriptor")
        tool_ref = desc_call.get("tool_ref")
        args = desc_call.get("args", {})

        if not desc_name or not tool_ref:
            continue

        # Find the matching tool result
        matched_result = result_by_tool.get(tool_ref)
        if matched_result is None:
            # Try fuzzy match against available tool result names
            for t_name, t_entry in result_by_tool.items():
                if t_name.replace("_", "") == tool_ref.replace("_", ""):
                    matched_result = t_entry
                    break

        if matched_result is None:
            observation_lines.append(
                f"- {tool_ref}: [descriptor {desc_name} skipped — no matching tool result]"
            )
            continue

        raw = matched_result.get("raw")
        data_source = matched_result.get("data_source", "acc")

        if raw is None:
            # Fallback to summary
            observation_lines.append(
                f"- {tool_ref}({data_source}): {matched_result.get('summary', 'no result')}"
            )
            described_tools.add(tool_ref)
            continue

        # Run the descriptor method
        try:
            desc_method = getattr(descriptor, desc_name, None)
            if desc_method is None:
                observation_lines.append(
                    f"- {tool_ref}({data_source}): [unknown descriptor '{desc_name}', using default]"
                )
                observation_lines.append(
                    f"  {_summarize_result(tool_ref, raw, data_source)}"
                )
            else:
                desc_output = desc_method(raw, **args)
                observation_lines.append(f"- {tool_ref}({data_source}) [{desc_name}]:\n  {desc_output}")
        except Exception as e:
            observation_lines.append(
                f"- {tool_ref}({data_source}): [descriptor error: {str(e)}]"
            )
            # Fallback to default summary
            observation_lines.append(
                f"  Default: {matched_result.get('summary', 'no result')}"
            )

        described_tools.add(tool_ref)

    # --- Fallback: summarize tools that had no descriptor ---
    for r in recent_results:
        t_name = r["tool"]
        if t_name not in described_tools:
            data_source = r.get("data_source", "acc")
            summary = r.get("summary", "no result")
            # Truncate very long summaries
            if len(summary) > 500:
                summary = summary[:500] + "... [truncated]"
            observation_lines.append(f"- {t_name}({data_source}): {summary}")

    # --- Build the full observations text using the observe template ---
    observations_text = "\n".join(observation_lines)

    # Also include previous iterations' observations for context
    previous_observations = list(state.get("observations", []))
    previous_observations.append(observations_text)

    # Build the full observe section using the template
    observe_template = _load_prompt("observe")
    full_observe_text = observe_template.format(
        iteration=iteration,
        iteration_observations=observations_text,
    )

    print(f"\n[Observe iteration {iteration}] Observations:\n{observations_text}\n")

    return {
        "observations": previous_observations,
        "descriptor_queue": [],  # Clear after use
        "messages": [
            AIMessage(content=f"[observe iter={iteration}] Observations:\n{observations_text}"),
        ],
    }


def summarize(state: AgentState) -> dict:
    """LLM generates final answer based on all tool results."""
    tool_results = state.get("tool_results", [])

    # Deduplicate results (keep last result per tool name)
    seen = {}
    for r in tool_results:
        seen[r["tool"]] = r
    unique_results = list(seen.values())

    # Feed full results — Mistral-7B supports 32K context
    results_lines = []
    for r in unique_results:
        results_lines.append(f"- {r['tool']}: {r['summary']}")
    results_text = "\n".join(results_lines)

    # Load prompt template from prompts/summarize.md
    template = _load_prompt("summarize")
    prompt = template.format(
        signal_info=state.get('signal_info', ''),
        user_goal=state['user_goal'],
        results_text=results_text,
    )

    answer = call_llm(prompt, max_tokens=2048, temperature=0.2)
    print(f"\n[Final Answer] LLM:\n{answer}\n")

    return {
        "final_answer": answer,
        "messages": [
            AIMessage(content=f"[summarize] Final Answer:\n{answer}"),
        ],
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """Decide whether to execute tools, summarize, or stop."""
    if state.get("error"):
        return "summarize"

    tool_queue = state.get("tool_queue", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)

    # If tools are queued, execute them
    if tool_queue:
        return "execute_tool"

    # If we've hit max iterations or plan says done, summarize
    if iteration >= max_iter:
        return "summarize"

    return "summarize"


def after_execute(state: AgentState) -> str:
    """After executing tools, always go to observe node."""
    return "observe"


def after_observe(state: AgentState) -> str:
    """After observing results, go back to plan or summarize."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)

    if iteration >= max_iter:
        return "summarize"

    # Go back to plan for another iteration
    return "plan"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_agent() -> StateGraph:
    """Construct the DSP agent LangGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("load_signal", load_signal)
    graph.add_node("plan", plan)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("observe", observe)
    graph.add_node("summarize", summarize)

    # Edges
    graph.add_edge(START, "load_signal")
    graph.add_edge("load_signal", "plan")
    graph.add_conditional_edges("plan", should_continue, {
        "execute_tool": "execute_tool",
        "summarize": "summarize",
    })
    graph.add_edge("execute_tool", "observe")
    graph.add_conditional_edges("observe", after_observe, {
        "plan": "plan",
        "summarize": "summarize",
    })
    graph.add_edge("summarize", END)

    return graph


def create_agent(thread_id: str = "default"):
    """Create a compiled agent with memory checkpointing."""
    graph = build_agent()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory), {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(csv_path: str, user_goal: str, fs: float = 0.0, max_iterations: int = 3, thread_id: str = "default"):
    """Run the DSP agent on a CSV file with a user goal.
    
    Args:
        csv_path: Path to IMU CSV file  
        user_goal: What the user wants to know/analyze
        fs: Sampling frequency (0 = auto-detect from timestamps)
        max_iterations: Max plan-execute loops
        thread_id: Thread ID for memory/checkpointing
        
    Returns:
        Final agent state with results
    """
    # Initialize a dedicated conversation log file for this run
    conv_log_path = _init_conv_log(tag=thread_id[:8] if thread_id != "default" else "")

    agent, config = create_agent(thread_id)

    initial_state = {
        "user_goal": user_goal,
        "csv_path": csv_path,
        "fs": fs,
        "max_iterations": max_iterations,
    }

    print(f"=== DSP Agent ===")
    print(f"Goal: {user_goal}")
    print(f"File: {csv_path}")
    print(f"{'='*50}")

    result = agent.invoke(initial_state, config)

    # Flush all LangChain messages to the conversation log
    for msg in result.get("messages", []):
        _log_message(msg)
    conv_logger.info("=== Conversation trace ended ===")
    print(f"\nConversation log: {conv_log_path}")

    print(f"\n{'='*50}")
    print(f"FINAL ANSWER:\n{result.get('final_answer', 'No answer generated')}")
    print(f"{'='*50}")

    return result


if __name__ == "__main__":
    import sys

    # Default example
    csv_file = os.path.join(os.path.dirname(__file__), "data", "samples", "imu_walking_subject1.csv")
    goal = "Analyze this IMU signal. What activity is the person doing? What is the step frequency?"

    if len(sys.argv) >= 3:
        csv_file = sys.argv[1]
        goal = sys.argv[2]
    elif len(sys.argv) == 2:
        csv_file = sys.argv[1]

    run_agent(csv_file, goal, fs=50.0, max_iterations=3)
