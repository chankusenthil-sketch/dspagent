"""DSP Agent — LangGraph-based signal processing agent.

The LLM (OPT-6.7B served locally) acts as a planner:
  1. Reads the user's goal and signal metadata
  2. Plans which DSP tools to apply
  3. Executes the tools
  4. Summarizes and interprets results
  5. Decides if more analysis is needed (loop) or done

LangGraph StateGraph with:
  - load_signal: Read CSV into state
  - plan: LLM decides next DSP step
  - execute_tool: Run the chosen DSP tool
  - summarize: LLM interprets all results and answers the user's question
"""

import os
import json
import re
import csv
import numpy as np
from typing import Any, Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from llm_client import LLHTTPClient
from tools.scipy_tool import SciPyDSPTool, TOOL_REGISTRY
from tools.imufusion_tool import IMUFusionTool, IMU_TOOL_REGISTRY

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
    tool_queue: list             # list of tool calls to execute
    current_tool: dict           # the tool being executed now
    tool_results: list           # accumulated results from all tools
    iteration: int               # current planning iteration
    max_iterations: int          # safety limit

    # Final output
    final_answer: str
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

dsp = SciPyDSPTool()
try:
    imu_fusion = IMUFusionTool()
except ImportError:
    imu_fusion = None

# Merge IMU Fusion tools into the unified registry
TOOL_REGISTRY.update(IMU_TOOL_REGISTRY)

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
    port = int(os.environ.get("LLM_PORT", "8080"))
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
    """Format available tools as text for the LLM prompt."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)


def parse_tool_calls(llm_text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM output.
    
    Expected format from LLM:
      TOOL: tool_name
      ARGS: arg1=value1, arg2=value2
      DATA: acc|gyro|magnitude
    or
      DONE
    """
    tools = []

    # Normalize LLM text: remove markdown escapes, extra whitespace
    cleaned = llm_text.replace("\\", "").replace("`", "").replace("*", "")

    # Try to find TOOL: lines (handles "Tool:", "TOOL:", "ToOL:" etc.)
    tool_pattern = re.compile(
        r'tool\s*:\s*(\w+)',
        re.IGNORECASE
    )
    data_pattern = re.compile(
        r'data\s*:\s*(\w+)',
        re.IGNORECASE
    )
    args_pattern = re.compile(
        r'args?\s*[:\s]+(.+?)(?=\n|tool|data|done|$)',
        re.IGNORECASE
    )

    # Split into lines and parse line by line
    lines = cleaned.split("\n")
    current_tool = None
    current_args = {}
    current_data = "acc"

    for line in lines:
        tool_match = tool_pattern.search(line)
        if tool_match:
            # Save previous tool if exists
            if current_tool:
                tools.append({"tool": current_tool, "args": current_args, "data_source": current_data})
            raw_name = tool_match.group(1).strip().lower()
            # Fuzzy match against registry
            current_tool = _fuzzy_match_tool(raw_name)
            current_args = {}
            current_data = "acc"

            # Check if ARGS and DATA are on the same line
            args_match = args_pattern.search(line[tool_match.end():])
            if args_match:
                current_args = _parse_args(args_match.group(1))
            data_match = data_pattern.search(line[tool_match.end():])
            if data_match:
                current_data = data_match.group(1).strip().lower()
            continue

        # Check for standalone ARGS/DATA lines
        if current_tool:
            args_match = args_pattern.search(line)
            if args_match and not tool_match:
                current_args = _parse_args(args_match.group(1))
            data_match = data_pattern.search(line)
            if data_match:
                d = data_match.group(1).strip().lower()
                if d in ("acc", "gyro"):
                    current_data = d

    # Don't forget the last tool
    if current_tool:
        tools.append({"tool": current_tool, "args": current_args, "data_source": current_data})

    # Deduplicate tools (keep first occurrence)
    seen_tools = set()
    unique_tools = []
    for t in tools:
        if t["tool"] and t["tool"] not in seen_tools:
            seen_tools.add(t["tool"])
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


def _fuzzy_match_tool(raw_name: str) -> str | None:
    """Fuzzy match a (possibly misspelled) tool name to the registry."""
    # Exact match
    if raw_name in TOOL_REGISTRY:
        return raw_name
    # Normalize: remove underscores, lowercase
    normalized = raw_name.replace("_", "").replace("-", "")
    for registered in TOOL_REGISTRY:
        if registered.replace("_", "") == normalized:
            return registered
    # Prefix match
    for registered in TOOL_REGISTRY:
        if registered.startswith(raw_name[:5]):
            return registered
    # Substring match
    for registered in TOOL_REGISTRY:
        if raw_name[:4] in registered:
            return registered
    return None


def _parse_args(args_str: str) -> dict:
    """Parse argument string like 'fs=50, cutoff=10' into a dict."""
    args = {}
    # Remove non-arg text
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
                    args[k] = float(v)
                except ValueError:
                    pass
    return args


def is_done(llm_text: str) -> bool:
    """Check if LLM says analysis is complete."""
    text_lower = llm_text.lower()
    return "done" in text_lower and "tool:" not in text_lower


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
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 5),
        "error": "",
    }


def plan(state: AgentState) -> dict:
    """LLM plans what DSP tools to use next based on the goal and current results."""
    iteration = state.get("iteration", 0)
    tool_results = state.get("tool_results", [])

    # Build context for LLM — feed full tool results (Mistral supports 32K tokens)
    results_text = ""
    if tool_results:
        results_text = "\nPrevious results:\n"
        for r in tool_results:
            results_text += f"- {r['tool']}: {r['summary']}\n"

    # Load prompt template from prompts/plan.md
    template = _load_prompt("plan")
    prompt = template.format(
        signal_info=state.get('signal_info', ''),
        user_goal=state['user_goal'],
        results_text=results_text,
        tool_list=format_tool_list(),
    )

    llm_response = call_llm(prompt, max_tokens=1024, temperature=0.2)
    print(f"\n[Plan iteration {iteration}] LLM response:\n{llm_response}\n")

    # Check if LLM says done
    if is_done(llm_response) and len(tool_results) > 0:
        return {
            "plan": llm_response,
            "tool_queue": [],
            "iteration": iteration + 1,
        }

    # Parse tool calls
    tools = parse_tool_calls(llm_response)

    return {
        "plan": llm_response,
        "tool_queue": tools,
        "iteration": iteration + 1,
    }


def execute_tool(state: AgentState) -> dict:
    """Execute all queued DSP tools and collect results."""
    tool_queue = state.get("tool_queue", [])
    tool_results = list(state.get("tool_results", []))
    acc = state.get("acc")
    gyro = state.get("gyro")
    fs = state.get("fs", 50.0)
    timestamps = state.get("timestamps")

    for tool_call in tool_queue:
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})
        data_source = tool_call.get("data_source", "acc")

        # Select data
        if data_source == "gyro":
            data = gyro
        else:
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

    return {"tool_results": tool_results, "tool_queue": []}


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

    return {"final_answer": answer}


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
    """After executing tools, go back to plan or summarize."""
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
    graph.add_node("summarize", summarize)

    # Edges
    graph.add_edge(START, "load_signal")
    graph.add_edge("load_signal", "plan")
    graph.add_conditional_edges("plan", should_continue, {
        "execute_tool": "execute_tool",
        "summarize": "summarize",
    })
    graph.add_conditional_edges("execute_tool", after_execute, {
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
