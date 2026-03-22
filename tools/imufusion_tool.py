"""Wrapper for the IMU Fusion (imufusion) AHRS library.

Exposes multiple tools that the DSP agent can invoke:
  - imu_orientation:       Quaternion orientation from acc+gyro fusion
  - imu_euler_angles:      Roll, pitch, yaw (Euler angles) from fusion
  - imu_earth_acceleration: Acceleration rotated into earth frame
  - imu_linear_acceleration: Acceleration with gravity removed (pure motion)
  - imu_gravity:           Estimated gravity vector per sample

All tools accept acc (N,3), gyro (N,3), and timestamps (N,) arrays.
Magnetometer is optional — zeros are accepted (uses update_no_magnetometer).
"""
from typing import Dict, Any
import numpy as np

try:
    import imufusion
except ImportError:
    imufusion = None


class IMUFusionTool:
    """Collection of IMU sensor fusion tools using the imufusion AHRS library."""

    def __init__(self):
        if imufusion is None:
            raise ImportError("imufusion library not available. Install via: pip install imufusion")

    def _run_fusion(self, acc: np.ndarray, gyro: np.ndarray, timestamps: np.ndarray,
                    mag: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Run AHRS fusion over all samples and collect per-sample outputs.

        Returns dict with keys: quaternions, euler, earth_acceleration,
        linear_acceleration, gravity — each shape (N, 3) or (N, 4).
        """
        ahrs = imufusion.Ahrs()
        n = len(timestamps)

        quaternions = np.zeros((n, 4), dtype=np.float32)
        euler_angles = np.zeros((n, 3), dtype=np.float32)
        earth_acc = np.zeros((n, 3), dtype=np.float32)
        linear_acc = np.zeros((n, 3), dtype=np.float32)
        gravity = np.zeros((n, 3), dtype=np.float32)

        use_mag = mag is not None and not (np.all(mag == 0) or np.any(np.isnan(mag)))

        prev_t = None
        for i in range(n):
            dt = 0.0 if prev_t is None else float(timestamps[i] - prev_t)
            prev_t = timestamps[i]

            g = gyro[i].tolist()
            a = acc[i].tolist()

            if use_mag:
                ahrs.update(g, a, mag[i].tolist(), dt)
            else:
                ahrs.update_no_magnetometer(g, a, dt)

            q = ahrs.quaternion.copy()
            quaternions[i] = q
            euler_angles[i] = imufusion.quaternion_to_euler(q)
            earth_acc[i] = ahrs.earth_acceleration.copy()
            linear_acc[i] = ahrs.linear_acceleration.copy()
            gravity[i] = ahrs.gravity.copy()

        return {
            "quaternions": quaternions,
            "euler": euler_angles,
            "earth_acceleration": earth_acc,
            "linear_acceleration": linear_acc,
            "gravity": gravity,
        }

    # --- Individual tools (called by the agent) ---

    def imu_orientation(self, acc: np.ndarray, gyro: np.ndarray,
                        timestamps: np.ndarray) -> Dict[str, Any]:
        """Estimate device orientation as quaternions (w, x, y, z) per sample."""
        fusion = self._run_fusion(acc, gyro, timestamps)
        quats = fusion["quaternions"]
        return {
            "quaternions": quats,
            "summary": {
                "mean_w": float(np.mean(quats[:, 0])),
                "mean_x": float(np.mean(quats[:, 1])),
                "mean_y": float(np.mean(quats[:, 2])),
                "mean_z": float(np.mean(quats[:, 3])),
            },
        }

    def imu_euler_angles(self, acc: np.ndarray, gyro: np.ndarray,
                         timestamps: np.ndarray) -> Dict[str, Any]:
        """Compute roll, pitch, yaw (Euler angles in degrees) per sample."""
        fusion = self._run_fusion(acc, gyro, timestamps)
        euler = fusion["euler"]
        return {
            "euler_angles": euler,
            "roll": {"mean": float(np.mean(euler[:, 0])), "std": float(np.std(euler[:, 0])),
                     "min": float(np.min(euler[:, 0])), "max": float(np.max(euler[:, 0]))},
            "pitch": {"mean": float(np.mean(euler[:, 1])), "std": float(np.std(euler[:, 1])),
                      "min": float(np.min(euler[:, 1])), "max": float(np.max(euler[:, 1]))},
            "yaw": {"mean": float(np.mean(euler[:, 2])), "std": float(np.std(euler[:, 2])),
                    "min": float(np.min(euler[:, 2])), "max": float(np.max(euler[:, 2]))},
        }

    def imu_earth_acceleration(self, acc: np.ndarray, gyro: np.ndarray,
                               timestamps: np.ndarray) -> Dict[str, Any]:
        """Rotate accelerometer readings into earth frame (removes device tilt effect)."""
        fusion = self._run_fusion(acc, gyro, timestamps)
        ea = fusion["earth_acceleration"]
        return {
            "earth_acceleration": ea,
            "x": {"mean": float(np.mean(ea[:, 0])), "std": float(np.std(ea[:, 0]))},
            "y": {"mean": float(np.mean(ea[:, 1])), "std": float(np.std(ea[:, 1]))},
            "z": {"mean": float(np.mean(ea[:, 2])), "std": float(np.std(ea[:, 2]))},
        }

    def imu_linear_acceleration(self, acc: np.ndarray, gyro: np.ndarray,
                                timestamps: np.ndarray) -> Dict[str, Any]:
        """Extract linear (motion-only) acceleration by removing gravity component."""
        fusion = self._run_fusion(acc, gyro, timestamps)
        la = fusion["linear_acceleration"]
        return {
            "linear_acceleration": la,
            "x": {"mean": float(np.mean(la[:, 0])), "std": float(np.std(la[:, 0]))},
            "y": {"mean": float(np.mean(la[:, 1])), "std": float(np.std(la[:, 1]))},
            "z": {"mean": float(np.mean(la[:, 2])), "std": float(np.std(la[:, 2]))},
        }

    def imu_gravity(self, acc: np.ndarray, gyro: np.ndarray,
                    timestamps: np.ndarray) -> Dict[str, Any]:
        """Estimate gravity vector per sample (useful for tilt/orientation analysis)."""
        fusion = self._run_fusion(acc, gyro, timestamps)
        grav = fusion["gravity"]
        return {
            "gravity": grav,
            "x": {"mean": float(np.mean(grav[:, 0])), "std": float(np.std(grav[:, 0]))},
            "y": {"mean": float(np.mean(grav[:, 1])), "std": float(np.std(grav[:, 1]))},
            "z": {"mean": float(np.mean(grav[:, 2])), "std": float(np.std(grav[:, 2]))},
        }


# Registry entries for IMU Fusion tools (merged into TOOL_REGISTRY by dsp_agent.py)
IMU_TOOL_REGISTRY = {
    "imu_orientation": {
        "description": "Estimate device orientation (quaternions w,x,y,z) using AHRS sensor fusion of acc+gyro. Useful for understanding device pose.",
        "args": [],
    },
    "imu_euler_angles": {
        "description": "Compute roll, pitch, yaw (Euler angles in degrees) from acc+gyro fusion. Useful for tilt and heading analysis.",
        "args": [],
    },
    "imu_earth_acceleration": {
        "description": "Rotate acceleration into earth frame using AHRS fusion. Removes tilt effect, giving true north/east/down acceleration.",
        "args": [],
    },
    "imu_linear_acceleration": {
        "description": "Extract linear (motion-only) acceleration by removing gravity via AHRS fusion. Useful for detecting pure movement.",
        "args": [],
    },
    "imu_gravity": {
        "description": "Estimate gravity vector per sample using AHRS fusion. Useful for determining device tilt/orientation relative to ground.",
        "args": [],
    },
}


if __name__ == "__main__":
    print("IMUFusionTool: available tools:", list(IMU_TOOL_REGISTRY.keys()))

