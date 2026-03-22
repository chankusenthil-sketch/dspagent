"""Download UCI HAR Dataset and convert to CSV files for the DSP agent.

The UCI HAR Dataset contains IMU data (accelerometer + gyroscope) from 30 subjects
performing 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
Sampling rate: 50 Hz, 3-axis accelerometer + 3-axis gyroscope.

Source: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""
import os
import zipfile
import urllib.request
import numpy as np
import csv

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH = os.path.join(DATA_DIR, "uci_har.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "UCI HAR Dataset")


def download():
    if os.path.isdir(EXTRACT_DIR):
        print(f"Dataset already extracted at {EXTRACT_DIR}")
        return
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print(f"Downloaded to {ZIP_PATH}")
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    print(f"Extracted to {EXTRACT_DIR}")


def load_inertial_signals(dataset_type="train"):
    """Load raw inertial signals from the UCI HAR dataset.
    
    Returns:
        signals: dict of signal_name -> np.array of shape (num_samples, 128)
        labels: np.array of activity labels (1-6)
        subject_ids: np.array of subject IDs
    """
    base = os.path.join(EXTRACT_DIR, dataset_type, "Inertial Signals")
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    signals = {}
    for name in signal_names:
        fpath = os.path.join(base, f"{name}_{dataset_type}.txt")
        if os.path.exists(fpath):
            signals[name] = np.loadtxt(fpath)
    
    labels = np.loadtxt(os.path.join(EXTRACT_DIR, dataset_type, f"y_{dataset_type}.txt"), dtype=int)
    subjects = np.loadtxt(os.path.join(EXTRACT_DIR, dataset_type, f"subject_{dataset_type}.txt"), dtype=int)
    return signals, labels, subjects


def create_sample_csvs(num_samples=5):
    """Create sample CSV files from the dataset for easy loading by the DSP agent.
    
    Each CSV has columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity
    """
    activity_map = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
                    4: "SITTING", 5: "STANDING", 6: "LAYING"}
    
    signals, labels, subjects = load_inertial_signals("train")
    
    # Pick one sample per activity
    samples_dir = os.path.join(DATA_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    activities_found = set()
    count = 0
    
    for idx in range(len(labels)):
        activity = labels[idx]
        if activity in activities_found:
            continue
        activities_found.add(activity)
        
        # 128 timesteps at 50 Hz = 2.56 seconds per window
        fs = 50.0
        n_steps = signals["body_acc_x"].shape[1]  # 128
        timestamps = np.arange(n_steps) / fs
        
        acc_x = signals["total_acc_x"][idx]
        acc_y = signals["total_acc_y"][idx]
        acc_z = signals["total_acc_z"][idx]
        gyro_x = signals["body_gyro_x"][idx]
        gyro_y = signals["body_gyro_y"][idx]
        gyro_z = signals["body_gyro_z"][idx]
        
        activity_name = activity_map[activity]
        subject_id = subjects[idx]
        
        csv_path = os.path.join(samples_dir, f"imu_{activity_name.lower()}_subject{subject_id}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "activity"])
            for t in range(n_steps):
                writer.writerow([
                    f"{timestamps[t]:.4f}",
                    f"{acc_x[t]:.6f}", f"{acc_y[t]:.6f}", f"{acc_z[t]:.6f}",
                    f"{gyro_x[t]:.6f}", f"{gyro_y[t]:.6f}", f"{gyro_z[t]:.6f}",
                    activity_name
                ])
        print(f"  Created: {csv_path}")
        count += 1
        if count >= num_samples:
            break
    
    # Also create a multi-activity CSV with several windows concatenated
    multi_path = os.path.join(samples_dir, "imu_multi_activity.csv")
    with open(multi_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "activity"])
        
        time_offset = 0.0
        fs = 50.0
        # Pick 10 consecutive windows from the dataset
        for idx in range(min(10, len(labels))):
            n_steps = signals["body_acc_x"].shape[1]
            timestamps = np.arange(n_steps) / fs + time_offset
            
            activity_name = activity_map[labels[idx]]
            for t in range(n_steps):
                writer.writerow([
                    f"{timestamps[t]:.4f}",
                    f"{signals['total_acc_x'][idx][t]:.6f}",
                    f"{signals['total_acc_y'][idx][t]:.6f}",
                    f"{signals['total_acc_z'][idx][t]:.6f}",
                    f"{signals['body_gyro_x'][idx][t]:.6f}",
                    f"{signals['body_gyro_y'][idx][t]:.6f}",
                    f"{signals['body_gyro_z'][idx][t]:.6f}",
                    activity_name
                ])
            time_offset += n_steps / fs
    print(f"  Created: {multi_path}")
    print(f"\nDone! {count + 1} CSV files created in {samples_dir}")


if __name__ == "__main__":
    download()
    create_sample_csvs(num_samples=6)  # one per activity
