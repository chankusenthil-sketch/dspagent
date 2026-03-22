"""DSP utilities wrapper exposing common SciPy signal functions as tools.

Each method is a standalone DSP operation the agent can invoke.
All methods work on numpy arrays and return numpy arrays or dicts.
"""
from typing import Dict, Any, List
import numpy as np
from scipy import signal, stats


class SciPyDSPTool:
    """Collection of DSP operations usable as agent tools."""

    # --- Filtering ---

    def lowpass(self, data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth lowpass filter. data shape (N,) or (N,3)."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data, axis=0)

    def highpass(self, data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth highpass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, data, axis=0)

    def bandpass(self, data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low_n = low / nyq
        high_n = high / nyq
        b, a = signal.butter(order, [low_n, high_n], btype='band')
        return signal.filtfilt(b, a, data, axis=0)

    # --- Spectral Analysis ---

    def spectrogram(self, data: np.ndarray, fs: float, nperseg: int = 256) -> Dict[str, Any]:
        """Compute spectrogram. Returns freqs, times, and power matrix."""
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg)
        return {"freqs": f, "times": t, "Sxx": Sxx}

    def fft_magnitude(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """Compute FFT magnitude spectrum. Returns freqs and magnitudes."""
        n = len(data)
        if data.ndim == 1:
            fft_vals = np.fft.rfft(data)
            magnitudes = np.abs(fft_vals) * 2.0 / n
        else:
            magnitudes = []
            for col in range(data.shape[1]):
                fft_vals = np.fft.rfft(data[:, col])
                magnitudes.append(np.abs(fft_vals) * 2.0 / n)
            magnitudes = np.column_stack(magnitudes)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        return {"freqs": freqs, "magnitudes": magnitudes}

    def dominant_frequency(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """Find dominant frequency per axis. Returns dict with freq values."""
        result = self.fft_magnitude(data, fs)
        freqs = result["freqs"]
        mags = result["magnitudes"]
        if mags.ndim == 1:
            idx = np.argmax(mags[1:]) + 1  # skip DC
            return {"dominant_freq": float(freqs[idx]), "magnitude": float(mags[idx])}
        else:
            dom = {}
            for col in range(mags.shape[1]):
                idx = np.argmax(mags[1:, col]) + 1
                dom[f"axis_{col}"] = {"freq": float(freqs[idx]), "magnitude": float(mags[idx, col])}
            return dom

    # --- Statistical Features ---

    def compute_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistics: mean, std, min, max, rms, median per axis."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result = {}
        axis_names = ["x", "y", "z"] if data.shape[1] <= 3 else [f"col_{i}" for i in range(data.shape[1])]
        for i in range(data.shape[1]):
            col = data[:, i]
            name = axis_names[i] if i < len(axis_names) else f"col_{i}"
            result[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "rms": float(np.sqrt(np.mean(col**2))),
                "median": float(np.median(col)),
            }
        return result

    def signal_magnitude_area(self, data: np.ndarray) -> float:
        """Compute Signal Magnitude Area (SMA) - sum of absolute values across axes.
        Common feature for activity recognition."""
        if data.ndim == 1:
            return float(np.mean(np.abs(data)))
        return float(np.mean(np.sum(np.abs(data), axis=1)))

    def zero_crossing_rate(self, data: np.ndarray) -> Dict[str, float]:
        """Count zero crossings per axis - useful for detecting periodic motion."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result = {}
        axis_names = ["x", "y", "z"]
        for i in range(data.shape[1]):
            col = data[:, i]
            zc = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
            name = axis_names[i] if i < len(axis_names) else f"col_{i}"
            result[name] = float(zc) / len(col)
        return result

    # --- Signal Metrics ---

    def compute_magnitude(self, data: np.ndarray) -> np.ndarray:
        """Compute vector magnitude from multi-axis data. shape (N,3) -> (N,)."""
        return np.sqrt(np.sum(data**2, axis=1))

    def signal_energy(self, data: np.ndarray, fs: float) -> Dict[str, float]:
        """Compute signal energy in frequency domain per axis."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result = {}
        axis_names = ["x", "y", "z"]
        for i in range(data.shape[1]):
            fft_vals = np.fft.rfft(data[:, i])
            energy = float(np.sum(np.abs(fft_vals)**2) / len(data[:, i]))
            name = axis_names[i] if i < len(axis_names) else f"col_{i}"
            result[name] = energy
        return result

    def peak_detection(self, data: np.ndarray, fs: float, height: float = None, distance: int = None) -> Dict[str, Any]:
        """Detect peaks in signal. Useful for step counting, etc."""
        if data.ndim > 1:
            data = self.compute_magnitude(data)
        if distance is None:
            distance = int(fs * 0.3)  # minimum 0.3s between peaks
        peaks, properties = signal.find_peaks(data, height=height, distance=distance)
        return {
            "peak_indices": peaks.tolist(),
            "peak_count": len(peaks),
            "peak_times": (peaks / fs).tolist(),
            "peak_values": data[peaks].tolist() if len(peaks) > 0 else [],
        }


# Registry of available tool functions with descriptions for the LLM planner
TOOL_REGISTRY = {
    "lowpass": {
        "description": "Apply lowpass filter to remove high-frequency noise. Args: data, fs (sampling freq), cutoff (Hz)",
        "args": ["data", "fs", "cutoff"],
    },
    "highpass": {
        "description": "Apply highpass filter to remove low-frequency drift/gravity. Args: data, fs, cutoff (Hz)",
        "args": ["data", "fs", "cutoff"],
    },
    "bandpass": {
        "description": "Apply bandpass filter to isolate frequency range. Args: data, fs, low (Hz), high (Hz)",
        "args": ["data", "fs", "low", "high"],
    },
    "fft_magnitude": {
        "description": "Compute FFT magnitude spectrum to see frequency content. Args: data, fs",
        "args": ["data", "fs"],
    },
    "dominant_frequency": {
        "description": "Find the dominant (strongest) frequency per axis. Args: data, fs",
        "args": ["data", "fs"],
    },
    "compute_statistics": {
        "description": "Compute statistics (mean, std, min, max, rms, median) per axis. Args: data",
        "args": ["data"],
    },
    "signal_magnitude_area": {
        "description": "Compute Signal Magnitude Area (SMA) - activity level indicator. Args: data",
        "args": ["data"],
    },
    "zero_crossing_rate": {
        "description": "Compute zero crossing rate per axis - periodicity indicator. Args: data",
        "args": ["data"],
    },
    "compute_magnitude": {
        "description": "Compute vector magnitude from 3-axis data to single axis. Args: data",
        "args": ["data"],
    },
    "signal_energy": {
        "description": "Compute signal energy in frequency domain per axis. Args: data, fs",
        "args": ["data", "fs"],
    },
    "peak_detection": {
        "description": "Detect peaks in signal - useful for step counting. Args: data, fs",
        "args": ["data", "fs"],
    },
    "spectrogram": {
        "description": "Compute time-frequency spectrogram. Args: data, fs",
        "args": ["data", "fs"],
    },
}


if __name__ == "__main__":
    print("SciPyDSPTool: import and use from your app")
    print(f"Available tools: {list(TOOL_REGISTRY.keys())}")
