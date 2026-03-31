"""Descriptor tool for formatting DSP tool outputs into LLM-readable summaries.

The planning LLM specifies which descriptor to apply to each tool's output,
allowing it to control what information it receives in the next planning
iteration.  Each descriptor method takes a raw tool result (dict or ndarray)
and optional parameters, then returns a concise text string the LLM can
reason about.
"""
from typing import Any, Dict, List
import numpy as np


class DescriptorTool:
    """Collection of descriptor methods that transform raw DSP tool outputs
    into concise, LLM-readable text summaries."""

    # --- Spectral / Peak descriptors ---

    def top_n_peaks(self, result: Any, n: int = 3) -> str:
        """Extract the top N frequencies by magnitude from FFT or spectral results.

        Works with outputs from: fft_magnitude, dominant_frequency, peak_detection.
        """
        if isinstance(result, dict):
            # fft_magnitude returns {"freqs": array, "magnitudes": array}
            freqs = result.get("freqs")
            mags = result.get("magnitudes")
            if freqs is not None and mags is not None:
                freqs = np.asarray(freqs)
                mags = np.asarray(mags)
                if mags.ndim == 1:
                    return self._top_n_from_spectrum(freqs, mags, n, axis_label=None)
                else:
                    # Multi-axis: report top N per axis
                    axis_names = ["x", "y", "z"]
                    parts = []
                    for col in range(mags.shape[1]):
                        label = axis_names[col] if col < len(axis_names) else f"axis_{col}"
                        parts.append(self._top_n_from_spectrum(freqs, mags[:, col], n, axis_label=label))
                    return "\n".join(parts)

            # dominant_frequency returns {"axis_0": {"freq": ..., "magnitude": ...}, ...}
            # or {"dominant_freq": ..., "magnitude": ...}
            if "dominant_freq" in result:
                return f"Dominant frequency: {result['dominant_freq']:.2f} Hz (magnitude: {result['magnitude']:.4f})"
            axis_keys = [k for k in result if k.startswith("axis_")]
            if axis_keys:
                parts = []
                for k in sorted(axis_keys):
                    v = result[k]
                    if isinstance(v, dict):
                        parts.append(f"{k}: {v.get('freq', 0):.2f} Hz (mag: {v.get('magnitude', 0):.4f})")
                return "Dominant frequencies per axis: " + ", ".join(parts)

            # peak_detection returns {"peak_count": ..., "peak_times": [...], ...}
            if "peak_count" in result:
                count = result["peak_count"]
                times = result.get("peak_times", [])
                values = result.get("peak_values", [])
                top_indices = np.argsort(values)[-n:][::-1] if len(values) > 0 else []
                parts = [f"Total peaks: {count}"]
                for idx in top_indices:
                    parts.append(f"  peak at t={times[idx]:.2f}s, value={values[idx]:.4f}")
                return "\n".join(parts)

        return f"top_n_peaks: cannot interpret result type {type(result).__name__}"

    def _top_n_from_spectrum(self, freqs: np.ndarray, mags: np.ndarray,
                             n: int, axis_label: str | None) -> str:
        """Helper: extract top N peaks from a 1D frequency-magnitude pair."""
        # Skip DC component (index 0)
        if len(mags) > 1:
            search_mags = mags[1:]
            search_freqs = freqs[1:]
        else:
            search_mags = mags
            search_freqs = freqs

        n = min(n, len(search_mags))
        top_indices = np.argsort(search_mags)[-n:][::-1]

        prefix = f"[{axis_label}] " if axis_label else ""
        parts = [f"{prefix}Top {n} frequencies:"]
        for rank, idx in enumerate(top_indices, 1):
            parts.append(f"  {rank}. {search_freqs[idx]:.2f} Hz (magnitude: {search_mags[idx]:.4f})")
        return "\n".join(parts)

    # --- Statistical descriptors ---

    def per_axis_summary(self, result: Any, metrics: List[str] | None = None) -> str:
        """Produce a per-axis summary with selectable metrics.

        Works with outputs from: compute_statistics, or raw arrays.
        Default metrics: mean, std, rms.
        """
        if metrics is None:
            metrics = ["mean", "std", "rms"]

        if isinstance(result, np.ndarray):
            # Raw array — compute stats on the fly
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            axis_names = ["x", "y", "z"] if result.shape[1] <= 3 else [f"col_{i}" for i in range(result.shape[1])]
            parts = []
            for i in range(result.shape[1]):
                col = result[:, i]
                name = axis_names[i] if i < len(axis_names) else f"col_{i}"
                vals = self._compute_metrics(col, metrics)
                parts.append(f"  {name}: {vals}")
            return "Per-axis summary:\n" + "\n".join(parts)

        if isinstance(result, dict):
            # compute_statistics output: {"x": {"mean": ..., "std": ..., ...}, ...}
            # or IMU tool output: {"x": {"mean": ..., "std": ...}, "linear_acceleration": array}
            axis_keys = [k for k in result if isinstance(result[k], dict) and "mean" in result.get(k, {})]
            if axis_keys:
                parts = []
                for k in axis_keys:
                    stat_dict = result[k]
                    selected = {m: stat_dict[m] for m in metrics if m in stat_dict}
                    vals_str = ", ".join(f"{m}={v:.4f}" for m, v in selected.items())
                    parts.append(f"  {k}: {vals_str}")
                return "Per-axis summary:\n" + "\n".join(parts)

        return f"per_axis_summary: cannot interpret result type {type(result).__name__}"

    def _compute_metrics(self, col: np.ndarray, metrics: List[str]) -> str:
        """Compute requested metrics for a 1D array and format as string."""
        computed = {}
        for m in metrics:
            if m == "mean":
                computed[m] = float(np.mean(col))
            elif m == "std":
                computed[m] = float(np.std(col))
            elif m == "rms":
                computed[m] = float(np.sqrt(np.mean(col ** 2)))
            elif m == "min":
                computed[m] = float(np.min(col))
            elif m == "max":
                computed[m] = float(np.max(col))
            elif m == "median":
                computed[m] = float(np.median(col))
        return ", ".join(f"{k}={v:.4f}" for k, v in computed.items())

    # --- Comparison descriptors ---

    def compare_axes(self, result: Any) -> str:
        """Compare axes to identify which has the highest variance/energy.

        Works with: compute_statistics outputs, signal_energy outputs, or raw arrays.
        """
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                return "Single axis data — no comparison possible."
            axis_names = ["x", "y", "z"] if result.shape[1] <= 3 else [f"col_{i}" for i in range(result.shape[1])]
            variances = np.var(result, axis=0)
            max_idx = int(np.argmax(variances))
            parts = []
            for i, (name, var) in enumerate(zip(axis_names, variances)):
                marker = " <-- highest" if i == max_idx else ""
                parts.append(f"  {name}: variance={var:.6f}{marker}")
            ratios = variances / variances[max_idx]
            ratio_str = ", ".join(f"{axis_names[i]}={r:.2f}" for i, r in enumerate(ratios))
            return f"Axis comparison (variance):\n" + "\n".join(parts) + f"\nRatios (relative to max): {ratio_str}"

        if isinstance(result, dict):
            # compute_statistics: {"x": {"std": ...}, "y": {"std": ...}, ...}
            axis_keys = [k for k in result if isinstance(result[k], dict) and "std" in result.get(k, {})]
            if axis_keys:
                stds: Dict[str, float] = {k: float(result[k]["std"]) for k in axis_keys}
                max_key = max(stds, key=lambda k: stds[k])
                parts = []
                for k, s in stds.items():
                    marker = " <-- highest" if k == max_key else ""
                    parts.append(f"  {k}: std={s:.4f}{marker}")
                return "Axis comparison (std):\n" + "\n".join(parts)

            # signal_energy: {"x": energy_value, "y": energy_value, ...}
            energy_keys = [k for k in result if isinstance(result[k], (int, float))]
            if energy_keys:
                max_key = max(energy_keys, key=lambda k: result[k])
                parts = []
                for k in energy_keys:
                    marker = " <-- highest" if k == max_key else ""
                    parts.append(f"  {k}: energy={result[k]:.4f}{marker}")
                return "Axis comparison (energy):\n" + "\n".join(parts)

        return f"compare_axes: cannot interpret result type {type(result).__name__}"

    # --- Frequency band descriptors ---

    def value_at_frequency(self, result: Any, target_freq: float = 1.0,
                           bandwidth: float = 0.5) -> str:
        """Report magnitude/energy at a specific frequency or band.

        Looks in [target_freq - bandwidth/2, target_freq + bandwidth/2].
        Works with: fft_magnitude output.
        """
        if isinstance(result, dict):
            freqs = result.get("freqs")
            mags = result.get("magnitudes")
            if freqs is not None and mags is not None:
                freqs = np.asarray(freqs)
                mags = np.asarray(mags)
                low = target_freq - bandwidth / 2
                high = target_freq + bandwidth / 2
                mask = (freqs >= low) & (freqs <= high)
                if not np.any(mask):
                    return f"No frequency content found in [{low:.2f}, {high:.2f}] Hz"

                if mags.ndim == 1:
                    band_energy = float(np.sum(mags[mask] ** 2))
                    peak_idx = np.argmax(mags[mask])
                    peak_freq = float(freqs[mask][peak_idx])
                    peak_mag = float(mags[mask][peak_idx])
                    return (f"Band [{low:.2f}-{high:.2f}] Hz: "
                            f"energy={band_energy:.6f}, peak at {peak_freq:.2f} Hz (mag={peak_mag:.4f})")
                else:
                    axis_names = ["x", "y", "z"]
                    parts = [f"Band [{low:.2f}-{high:.2f}] Hz:"]
                    for col in range(mags.shape[1]):
                        label = axis_names[col] if col < len(axis_names) else f"axis_{col}"
                        col_mags = mags[:, col]
                        band_energy = float(np.sum(col_mags[mask] ** 2))
                        peak_idx = np.argmax(col_mags[mask])
                        peak_freq = float(freqs[mask][peak_idx])
                        peak_mag = float(col_mags[mask][peak_idx])
                        parts.append(f"  {label}: energy={band_energy:.6f}, peak at {peak_freq:.2f} Hz (mag={peak_mag:.4f})")
                    return "\n".join(parts)

        return f"value_at_frequency: cannot interpret result — expected fft_magnitude output"

    # --- Shape / metadata descriptors ---

    def signal_shape(self, result: Any) -> str:
        """Describe basic shape, duration, and range of a signal array.

        Works with: any tool that returns an np.ndarray (filters, IMU outputs).
        """
        if isinstance(result, np.ndarray):
            parts = [f"Shape: {result.shape}"]
            if result.ndim == 1:
                parts.append(f"Range: [{result.min():.6f}, {result.max():.6f}]")
                parts.append(f"Mean: {np.mean(result):.6f}, Std: {np.std(result):.6f}")
            elif result.ndim == 2:
                axis_names = ["x", "y", "z"] if result.shape[1] <= 3 else [f"col_{i}" for i in range(result.shape[1])]
                for i in range(result.shape[1]):
                    name = axis_names[i] if i < len(axis_names) else f"col_{i}"
                    col = result[:, i]
                    parts.append(f"  {name}: range=[{col.min():.6f}, {col.max():.6f}], mean={np.mean(col):.6f}, std={np.std(col):.6f}")
            return "Signal metadata:\n" + "\n".join(parts)

        if isinstance(result, dict):
            # Try to find the main array in the dict (e.g., "linear_acceleration", "quaternions")
            for key, val in result.items():
                if isinstance(val, np.ndarray) and val.ndim >= 1 and val.size > 10:
                    return f"[{key}] " + self.signal_shape(val)
            # Fallback: report dict keys and types
            key_info = {k: type(v).__name__ for k, v in result.items()}
            return f"Signal metadata: dict with keys {key_info}"

        return f"signal_shape: unexpected type {type(result).__name__}"


# ---------------------------------------------------------------------------
# Registry of descriptor methods for the LLM planner
# ---------------------------------------------------------------------------

DESCRIPTOR_REGISTRY = {
    "top_n_peaks": {
        "description": "Extract top N frequencies by magnitude from spectral/FFT results. Args: n (default 3)",
        "args": ["n"],
    },
    "per_axis_summary": {
        "description": "Per-axis summary with selectable metrics (mean, std, rms, min, max, median). Args: metrics (comma-separated, default: mean,std,rms)",
        "args": ["metrics"],
    },
    "compare_axes": {
        "description": "Compare axes to find which has highest variance/energy. No args.",
        "args": [],
    },
    "value_at_frequency": {
        "description": "Report magnitude/energy at a specific frequency band. Args: target_freq (Hz), bandwidth (Hz, default 0.5)",
        "args": ["target_freq", "bandwidth"],
    },
    "signal_shape": {
        "description": "Basic signal metadata: shape, range, mean, std per axis. No args.",
        "args": [],
    },
}


if __name__ == "__main__":
    print("DescriptorTool: available descriptors:", list(DESCRIPTOR_REGISTRY.keys()))
