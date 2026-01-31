#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code implements comparative analysis between DOA (Direction of Arrival) estimation results and ground truth for passive radar. Key features include:
# 1. Automatic GPU/CPU computation adaptation: Prioritizes CuPy (GPU acceleration), falls back to NumPy (CPU) if CuPy is unavailable;
# 2. Loads IQ data files and ground truth CSV files, parses total frame count of data, and supports processing specified frame ranges;
# 3. Loads full-channel array data frame by frame, calculates and implements DOA estimation;
# 4. Converts azimuth angles from mathematical coordinate system to ENU geographic coordinate system;
# 5. Generates polar coordinate comparison plots: Visualizes true and estimated azimuth angles, and marks key frames;
# 6. Calculates and outputs error statistics.
# Note: This code is based on the Uniform Circular Array (UCA) model, and the steering vector calculation uses wavelength-normalized radius (0.33λ).
# Note: Direct path interference is NOT considered in this implementation.
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

# Try to import CuPy, fall back to NumPy if unavailable
try:
    import cupy as cp
    USE_GPU = True
    print("✅ Using CuPy (GPU acceleration)")
except ImportError:
    cp = np
    USE_GPU = False
    print("⚠️  CuPy not installed, using NumPy (CPU)")

# ==============================
# Configuration Parameters
# ==============================
SAMPLE_COUNT = 600000
CHANNEL_COUNT = 9
RX_FILE = "SAMPLESTREAM" # TODO: Modify this path to your actual IQ data file (e.g., xxx.cf32)
TRUTH_CSV = "TRUTHCSV" # TODO: Replace this with your actual ground truth CSV file path (set in previous configuration)
FRAME_START = 0
FRAME_END = None  # None means process all frames

# ==============================
# Core Functions
# ==============================

def load_full_frame(filename, total_channels, samples_per_frame, frame_index):
    all_data = np.fromfile(filename, dtype=np.complex64)
    samples_per_full_frame = total_channels * samples_per_frame
    n_frames = len(all_data) // samples_per_full_frame
    if not (0 <= frame_index < n_frames):
        raise ValueError(f"Frame index {frame_index} out of range (total {n_frames} frames)")
    frame_start = frame_index * samples_per_full_frame
    frame_raw = all_data[frame_start : frame_start + samples_per_full_frame]
    data = np.zeros((total_channels, samples_per_frame), dtype=np.complex64)
    for ch in range(total_channels):
        start = ch * samples_per_frame
        end = start + samples_per_frame
        data[ch, :] = frame_raw[start:end]
    return data, n_frames

def calculate_steering_vector(math_azimuth_rad, channel_count=8, radius_lambda=0.33):
    sv = np.zeros(channel_count, dtype=np.complex64)
    sv[0] = 1.0
    if channel_count > 1:
        for ch in range(1, channel_count):
            phi_m = 2.0 * np.pi * (ch - 1) / (channel_count - 1)
            x = radius_lambda * np.cos(phi_m)
            y = radius_lambda * np.sin(phi_m)
            phase = 2.0 * np.pi * (x * np.cos(math_azimuth_rad) + y * np.sin(math_azimuth_rad))
            sv[ch] = np.exp(1j * phase)
    return sv

def math_to_enu(math_deg):
    return (90.0 - math_deg) % 360.0

def bartlett_doa_gpu(x_matrix, steering_vectors):
    M, N = x_matrix.shape
    R = (x_matrix @ x_matrix.conj().T) / N
    A = steering_vectors[:, :, None]
    AH = cp.conj(cp.transpose(A, (0, 2, 1)))
    temp = AH @ R
    power = cp.real(temp @ A).squeeze()
    return power

# ==============================
# Main Program
# ==============================

def main():
    print("📥 Loading ground truth file...")
    truth_df = pd.read_csv(TRUTH_CSV)
    truth_azis = truth_df['azimuth_deg'].values

    dummy_data = np.fromfile(RX_FILE, dtype=np.complex64)
    total_samples = len(dummy_data)
    samples_per_full_frame = CHANNEL_COUNT * SAMPLE_COUNT
    total_frames = total_samples // samples_per_full_frame
    print(f"✅ Total frames in data: {total_frames}")

    start = FRAME_START
    end = FRAME_END if FRAME_END is not None else total_frames
    end = min(end, total_frames)
    frame_indices = list(range(start, end))
    num_frames = len(frame_indices)
    print(f"⚙️  Will process frames {start} to {end - 1} (total {num_frames} frames)")

    AZIMUTHS_MATH_DEG = np.linspace(0, 360, 721)
    steering_vectors_cpu = np.array([
        calculate_steering_vector(np.deg2rad(az), CHANNEL_COUNT, 0.33)
        for az in AZIMUTHS_MATH_DEG
    ], dtype=np.complex64)

    estimated_azis = []
    ground_truth_azis = []

    if USE_GPU:
        steering_vectors_gpu = cp.asarray(steering_vectors_cpu)

    for idx in tqdm(frame_indices, desc="📡 Processing DOA frames"):
        if idx >= len(truth_azis):
            continue
        gt_azi = float(truth_azis[idx])
        ground_truth_azis.append(gt_azi)

        array_data, _ = load_full_frame(RX_FILE, CHANNEL_COUNT, SAMPLE_COUNT, idx)
        array_data = array_data - np.mean(array_data, axis=1, keepdims=True)

        if USE_GPU:
            x_gpu = cp.asarray(array_data)
            power_gpu = bartlett_doa_gpu(x_gpu, steering_vectors_gpu)
            power = cp.asnumpy(power_gpu)
        else:
            R = (array_data @ array_data.conj().T) / array_data.shape[1]
            power = np.zeros(len(steering_vectors_cpu))
            for i, a in enumerate(steering_vectors_cpu):
                a = a.reshape(-1, 1)
                power[i] = np.real(np.conj(a.T) @ R @ a)[0, 0]

        peak_idx = int(np.argmax(power))
        est_math = AZIMUTHS_MATH_DEG[peak_idx]
        est_enu = math_to_enu(est_math)
        estimated_azis.append(round(est_enu, 2))

    # Convert to radians
    gt_radians = np.deg2rad(ground_truth_azis)
    est_radians = np.deg2rad(estimated_azis)

    # === Plotting ===
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Plot scatter points
    ax.scatter(gt_radians, np.ones_like(gt_radians), color='red', s=45, label='True Azimuth', alpha=0.85)
    ax.scatter(est_radians, np.ones_like(est_radians), color='blue', s=45, label='Estimated Azimuth', alpha=0.85)

    ax.set_ylim(0, 1.35)

    # === Mark key frames: 0th frame, every 30 frames, last frame ===
    valid_frame_count = len(gt_radians)  # Number of valid processed frames
    key_frame_local_indices = set()
    if valid_frame_count > 0:
        key_frame_local_indices.add(0)  # First frame
        if valid_frame_count > 1:
            key_frame_local_indices.add(valid_frame_count - 1)  # Last frame
        # Every 30 frames (starting from 20th frame, i.e., local index 19)
        for i in range(29, valid_frame_count, 30):
            key_frame_local_indices.add(i)
        key_frame_local_indices = sorted(key_frame_local_indices)

    # Add text annotations
    for local_idx in key_frame_local_indices:
        global_frame_num = frame_indices[local_idx]  # Map back to original frame number
        # True value annotation (slightly inner)
        ax.text(gt_radians[local_idx], 0.9, f'Frame {global_frame_num}',
                fontsize=9, ha='center', va='center', color='darkred', fontweight='bold')
        # Estimated value annotation (slightly outer)
        ax.text(est_radians[local_idx], 1.1, f'Frame {global_frame_num}',
                fontsize=9, ha='center', va='center', color='darkblue', fontweight='bold')

    # Title
    ax.set_title(
        "DOA Estimation Results vs Ground Truth (Polar Plot)\n(Unit Circle, ENU Coordinate System)",
        fontsize=17,
        fontweight='bold',
        pad=30
    )

    # Legend
    legend = ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.28, 1.02),
        fontsize=14,
        frameon=True,
        shadow=True
    )
    for text in legend.get_texts():
        text.set_fontsize(14)

    # Direction labels
    ax.set_thetagrids([0, 90, 180, 270], labels=['North (0°)', '       East (90°)', 'South (180°)', 'West (270°)        '])
    ax.tick_params(axis='both', which='major', labelsize=13)

    plt.tight_layout()
    plt.savefig("doa_polar_comparison.png", dpi=220, bbox_inches='tight')
    print("\n💾 Polar plot saved: doa_polar_comparison.png")
    plt.show()

    # Print error statistics
    errors = [abs((e - g + 180) % 360 - 180) for e, g in zip(estimated_azis, ground_truth_azis)]
    print(f"\n📊 Mean Absolute Error (MAE): {np.mean(errors):.2f}°")
    print(f"   Maximum Error: {np.max(errors):.2f}°")
    print(f"   Minimum Error: {np.min(errors):.2f}°")
    print(f"   Total Valid Frames: {len(errors)}")

if __name__ == "__main__":
    main()

