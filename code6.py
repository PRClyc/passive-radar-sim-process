#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code Function Description:
# This code implements spatial-domain direct wave suppression and joint target DOA/CAF estimation for passive radar based on an 8-element Uniform Circular Array (UCA).
# It core solves the problem that traditional single-channel direct wave cancellation does not consider the spatial incident angle, with key features including:
# 1. Heterogeneous computing adaptation: Automatically detect CuPy (GPU acceleration)/NumPy (CPU) to be compatible with different hardware environments;
# 2. Array data processing: Load 9-channel UCA array data (1 center channel + 8 ring channels), discard the center channel and only use the ring array;
# 3. Spatial estimation of direct wave: Calculate the steering vector of 8 elements based on the azimuth angle in ENU coordinate system, and estimate the spatial incident angle (DOA) of direct wave;
# 4. Array-domain direct wave suppression: Construct a null space through orthogonal projection matrix, and suppress direct wave with specific incident angle from array received data;
# 5. CAF target detection: Sum the suppressed array data, calculate the Cross-Ambiguity Function (CAF) to extract target delay/Doppler characteristics;
# 6. Target DOA estimation: Re-estimate the spatial azimuth angle of the target combined with CAF target characteristics in the projected null space;
# 7. Visual verification: Plot CAF comparison charts before and after direct wave suppression, and DOA spectrum comparison charts of direct wave/target to quantitatively demonstrate suppression effect;
# 8. Result output: Save the direct wave azimuth, target azimuth, delay/Doppler data of each frame into CSV file, supporting batch processing of multiple frames.
# Note: The core difference of this code is that direct wave suppression adopts array spatial processing (instead of single-channel time/frequency domain cancellation), and fully considers the spatial characteristics of the incident angle of direct wave.

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from tqdm import tqdm
from typing import Tuple

# ==============================
# Auto-select backend: CuPy (GPU) or NumPy (CPU)
# ==============================
try:
    import cupy as cp
    xp = cp
    print("✅ Using CuPy for GPU acceleration")
except ImportError:
    xp = np
    print("⚠️ CuPy not installed, using NumPy (CPU)")

# ==============================
# Configuration Parameters
# ==============================
SAMPLE_RATE = 2.4e6
SAMPLE_COUNT = 600000

TOTAL_CHANNEL_COUNT = 9
CENTER_CH_INDEX = 0
RING_CHANNEL_COUNT = 8

RX_FILE = "SAMPLESTREAM" # TODO: Modify this path to your actual IQ data file (e.g., xxx.cf32)
REF_FILE = "ref.cf32"

RADIUS_LAMBDA = 0.33

AZIMUTHS_ENU_DEG = np.linspace(0, 360, 1441)
AZIMUTHS_ENU_RAD = np.deg2rad(AZIMUTHS_ENU_DEG)

MAX_DELAY = 300
MAX_DOPPLER = 200
DOPPLER_STEP = 0.5

SAVE_PLOTS = True
MAX_PROCESS_FRAMES = 1

CAF_DELAY_HALF_WIN = 1
CAF_DOPPLER_HALF_BINS = 1

# ==============================
# Array/Signal Processing Utility Functions
# ==============================
def steering_uca8_from_enu(theta_enu_rad: float, radius_lambda: float = RADIUS_LAMBDA) -> np.ndarray:
    theta_math = (np.pi / 2.0 - theta_enu_rad) % (2.0 * np.pi)
    M = 8
    alpha = 2.0 * np.pi / M
    a = np.zeros(M, dtype=np.complex64)
    for m in range(M):
        phi_m = m * alpha
        x = radius_lambda * np.cos(phi_m)
        y = radius_lambda * np.sin(phi_m)
        phase = 2.0 * np.pi * (x * np.cos(theta_math) + y * np.sin(theta_math))
        a[m] = np.exp(1j * phase).astype(np.complex64)
    return a

def projection_matrix_perp(a_d: np.ndarray) -> np.ndarray:
    a = a_d.reshape(-1, 1).astype(np.complex64)
    denom = (np.vdot(a, a).real + 1e-12)
    P = np.eye(a.shape[0], dtype=np.complex64) - (a @ a.conj().T) / denom
    return P

# ==============================
# GPU/CPU Compatible Core Calculation Functions
# ==============================
def orthogonal_projection_nulling(X: np.ndarray, P_perp: np.ndarray) -> np.ndarray:
    X_gpu = xp.asarray(X)
    P_gpu = xp.asarray(P_perp)
    result = P_gpu @ X_gpu
    return xp.asnumpy(result) if xp is cp else result

def estimate_doa_uca8(ref: np.ndarray, X_ring: np.ndarray, azimuths_enu_rad: np.ndarray) -> Tuple[int, float, np.ndarray]:
    ref = xp.asarray(ref, dtype=xp.complex64)
    X_ring = xp.asarray(X_ring, dtype=xp.complex64)
    ref_conj = xp.conj(ref)

    M = len(azimuths_enu_rad)
    A = xp.zeros((8, M), dtype=xp.complex64)
    for i, az in enumerate(azimuths_enu_rad):
        A[:, i] = xp.asarray(steering_uca8_from_enu(az, RADIUS_LAMBDA))

    Y = xp.conj(A.T) @ X_ring
    J = xp.abs(xp.sum(ref_conj[None, :] * Y, axis=1)).astype(xp.float64)

    J_np = xp.asnumpy(J) if xp is cp else J
    best_idx = int(np.argmax(J_np))
    best_enu_deg = float((np.rad2deg(azimuths_enu_rad[best_idx])) % 360.0)
    return best_idx, best_enu_deg, J_np

def compute_caf_single_frame(ref: np.ndarray, obs: np.ndarray):
    delays = np.arange(-MAX_DELAY, MAX_DELAY + 1, dtype=np.int32)
    dopplers = np.arange(-MAX_DOPPLER, MAX_DOPPLER + DOPPLER_STEP, DOPPLER_STEP, dtype=np.float64)
    N = len(ref)
    t = xp.asarray(np.arange(N, dtype=np.float64) / SAMPLE_RATE)

    ref_gpu = xp.asarray(ref)
    obs_gpu = xp.asarray(obs)

    ref_fft = xp.fft.fft(ref_gpu)
    ref_fft_conj = xp.conj(ref_fft)

    caf = xp.zeros((len(dopplers), len(delays)), dtype=xp.float64)
    center = N // 2
    delay_slice = slice(center - MAX_DELAY, center + MAX_DELAY + 1)

    for i, fd in enumerate(dopplers):
        comp = xp.exp(-1j * 2.0 * np.pi * fd * t).astype(xp.complex64)
        obs_d = obs_gpu * comp
        corr = xp.fft.ifft(ref_fft_conj * xp.fft.fft(obs_d))
        corr_abs = xp.abs(xp.fft.fftshift(corr))
        caf[i, :] = corr_abs[delay_slice]

    caf /= (xp.max(caf) + 1e-12)
    return xp.asnumpy(caf) if xp is cp else caf, delays, dopplers

def find_peak(caf: np.ndarray, delays: np.ndarray, dopplers: np.ndarray):
    idx = np.unravel_index(np.argmax(caf), caf.shape)
    peak_delay = int(delays[idx[1]])
    peak_doppler = float(dopplers[idx[0]])
    return peak_delay, peak_doppler, idx

def caf_energy_at_bin(ref: np.ndarray, y: np.ndarray, delay_samp: int, doppler_hz: float,
                     delay_half_win: int = CAF_DELAY_HALF_WIN,
                     doppler_half_bins: int = CAF_DOPPLER_HALF_BINS) -> float:
    N = len(ref)
    t = xp.asarray(np.arange(N, dtype=np.float64) / SAMPLE_RATE)

    dopplers_all = np.arange(-MAX_DOPPLER, MAX_DOPPLER + DOPPLER_STEP, DOPPLER_STEP, dtype=np.float64)
    i0 = int(np.argmin(np.abs(dopplers_all - doppler_hz)))
    i1 = max(0, i0 - doppler_half_bins)
    i2 = min(len(dopplers_all), i0 + doppler_half_bins + 1)
    dopplers_local = dopplers_all[i1:i2]

    ref_gpu = xp.asarray(ref)
    y_gpu = xp.asarray(y)
    ref_fft_conj = xp.conj(xp.fft.fft(ref_gpu))
    center = N // 2
    k0 = center + int(delay_samp)
    k1 = max(0, k0 - delay_half_win)
    k2 = min(N, k0 + delay_half_win + 1)

    energy = 0.0
    for fd in dopplers_local:
        comp = xp.exp(-1j * 2.0 * np.pi * fd * t).astype(xp.complex64)
        yd = y_gpu * comp
        corr = xp.fft.ifft(ref_fft_conj * xp.fft.fft(yd))
        corr = xp.fft.fftshift(corr)
        win = corr[k1:k2]
        energy += float(xp.sum(xp.abs(win) ** 2).real)

    return energy

def estimate_target_doa_angle_caf_projected(ref: np.ndarray,
                                           X_ring_clean: np.ndarray,
                                           P_perp: np.ndarray,
                                           azimuths_enu_rad: np.ndarray,
                                           target_delay: int,
                                           target_doppler: float) -> Tuple[int, float, np.ndarray]:
    ref = np.asarray(ref, dtype=np.complex64)
    X_ring_clean = np.asarray(X_ring_clean, dtype=np.complex64)

    J = np.zeros(len(azimuths_enu_rad), dtype=np.float64)

    for i, az in enumerate(azimuths_enu_rad):
        a = steering_uca8_from_enu(az, RADIUS_LAMBDA)
        a_eff = P_perp @ a
        a_eff = a_eff / (np.linalg.norm(a_eff) + 1e-12)
        y = np.conj(a_eff).T @ X_ring_clean
        J[i] = caf_energy_at_bin(ref, y, target_delay, target_doppler)

    best_idx = int(np.argmax(J))
    best_enu_deg = float((np.rad2deg(azimuths_enu_rad[best_idx])) % 360.0)
    return best_idx, best_enu_deg, J

# ==============================
# Data Loading
# ==============================
def load_full_frame(filename, total_channels, samples_per_frame, frame_index, center_ch_index=0):
    all_data = np.fromfile(filename, dtype=np.complex64)
    samples_per_full_frame = total_channels * samples_per_frame
    n_frames = len(all_data) // samples_per_full_frame
    if n_frames <= 0:
        raise ValueError("Array data file length is insufficient to form a frame, please check parameters.")
    if not (0 <= frame_index < n_frames):
        raise ValueError(f"Frame index {frame_index} out of range (total {n_frames} frames)")

    frame_start = frame_index * samples_per_full_frame
    frame_raw = all_data[frame_start: frame_start + samples_per_full_frame]
    X_all = frame_raw.reshape(total_channels, samples_per_frame)

    x_center = X_all[center_ch_index, :]
    ring_idx = [i for i in range(total_channels) if i != center_ch_index]
    X_ring = X_all[ring_idx, :]

    return X_all, X_ring, x_center, n_frames

def load_reference_frame(filename, sample_count, frame_index):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Reference signal file does not exist: {filename}")
    all_ref = np.fromfile(filename, dtype=np.complex64)
    n_frames = len(all_ref) // sample_count
    if n_frames <= 0:
        raise ValueError("ref.cf32 length is insufficient to form a frame.")
    if frame_index >= n_frames:
        frame_index = 0
    start = frame_index * sample_count
    ref = all_ref[start:start + sample_count]
    if len(ref) < sample_count:
        raise ValueError("Reference signal frame length is insufficient.")
    return ref, n_frames

# ==============================
# Plotting (Optional)
# ==============================
def plot_caf(caf, delays, dopplers, peak_idx, title, out_png):
    plt.figure(figsize=(12, 8))
    plt.imshow(20 * np.log10(caf + 1e-12),
               extent=[delays[0], delays[-1], dopplers[-1], dopplers[0]],
               aspect='auto')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Delay (samples)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.grid(alpha=0.25, linestyle='--')
    plt.scatter(delays[peak_idx[1]], dopplers[peak_idx[0]],
                c='red', s=90, marker='x', linewidth=2, label='Peak')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_doa_spectrum(az_enu_deg, J, frame_index, out_png, title_suffix=""):
    J_db = 20 * np.log10(J / (np.max(J) + 1e-12) + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.plot(az_enu_deg, J_db, linewidth=2)
    plt.xlabel("Azimuth (ENU degrees) 0°=North")
    plt.ylabel("Matching Degree (dB)")
    plt.title(f"DOA Spectrum {title_suffix} (Frame {frame_index})")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.ylim(-40, 0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ==============================
# Main Process
# ==============================
def main():
    print("🚀 Pure UCA(8) Processing: Discard center ch0 → Direct wave DOA → P_perp nulling → CAF find target → Estimate target DOA with projected steering")

    all_rx = np.fromfile(RX_FILE, dtype=np.complex64)
    all_ref = np.fromfile(REF_FILE, dtype=np.complex64)

    samples_per_full_frame = TOTAL_CHANNEL_COUNT * SAMPLE_COUNT
    n_frames_rx = len(all_rx) // samples_per_full_frame
    n_frames_ref = len(all_ref) // SAMPLE_COUNT

    if n_frames_ref < 1:
        raise ValueError("❌ Insufficient reference signal (less than 1 frame), cannot process!")
    if n_frames_rx < 1:
        raise ValueError("❌ Insufficient received signal (less than 1 frame), check RX_FILE or parameters!")

    if MAX_PROCESS_FRAMES == -1:
        TOTAL_FRAMES = n_frames_rx
        print(f"📊 Will process all {TOTAL_FRAMES} frames")
    else:
        TOTAL_FRAMES = min(MAX_PROCESS_FRAMES, n_frames_rx)
        print(f"📊 Limiting to first {TOTAL_FRAMES} frames (configured MAX_PROCESS_FRAMES={MAX_PROCESS_FRAMES})")

    output_dir = os.path.expanduser("~/passive-radar-sim-process/predict")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "estimates.csv")
    output_dir = os.path.expanduser("~/passive-radar-sim-process")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_index',
            'direct_azimuth_enu_deg',
            'target_azimuth_enu_deg',
            'target_delay_samp',
            'target_doppler_hz'
        ])

    for frame_idx in tqdm(range(TOTAL_FRAMES), desc="📡 Processing frames", unit="frame"):
        _, X_ring, _, _ = load_full_frame(
            RX_FILE, TOTAL_CHANNEL_COUNT, SAMPLE_COUNT, frame_idx, center_ch_index=CENTER_CH_INDEX
        )
        ref, _ = load_reference_frame(REF_FILE, SAMPLE_COUNT, frame_idx)

        # Remove mean
        X0 = X_ring - np.mean(X_ring, axis=1, keepdims=True)

        # Direct wave DOA
        _, direct_enu_deg, J_dp = estimate_doa_uca8(ref, X0, AZIMUTHS_ENU_RAD)

        # Projection nulling
        a_d = steering_uca8_from_enu(np.deg2rad(direct_enu_deg), RADIUS_LAMBDA)
        P_perp = projection_matrix_perp(a_d)
        X_clean = orthogonal_projection_nulling(X0, P_perp)

        # CAF find target
        obs_after = np.sum(X_clean, axis=0)
        caf1, delays, dopplers = compute_caf_single_frame(ref, obs_after)
        peak_delay, peak_doppler, peak_idx = find_peak(caf1, delays, dopplers)

        # Target DOA (projected space)
        _, target_enu_deg, J_tgt = estimate_target_doa_angle_caf_projected(
            ref=ref,
            X_ring_clean=X_clean,
            P_perp=P_perp,
            azimuths_enu_rad=AZIMUTHS_ENU_RAD,
            target_delay=peak_delay,
            target_doppler=peak_doppler
        )

        # Save results
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_idx,
                f"{direct_enu_deg:.2f}",
                f"{target_enu_deg:.2f}",
                peak_delay,
                f"{peak_doppler:.2f}"
            ])

        tqdm.write(
            f"✅ Frame {frame_idx}: Direct wave={direct_enu_deg:.2f}°, "
            f"Target={target_enu_deg:.2f}°, delay={peak_delay}, doppler={peak_doppler:.2f} Hz"
        )

        # === New: Plot direct wave suppression comparison ===
        if SAVE_PLOTS:
            # 1. CAF of original signal (before suppression)
            obs_before = np.sum(X0, axis=0)  # X0 is mean-removed but non-nulled ring array
            caf_before, delays, dopplers = compute_caf_single_frame(ref, obs_before)
            peak_before = find_peak(caf_before, delays, dopplers)[2]

            # 2. CAF after suppression (already available)
            caf_after = caf1
            peak_after = peak_idx

            # Plot comparison
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            for ax, caf, pk, title in zip(
                axes,
                [caf_before, caf_after],
                [peak_before, peak_after],
                ["Before Direct Wave Suppression", "After Direct Wave Suppression"]
            ):
                im = ax.imshow(
                    20 * np.log10(caf + 1e-12),
                    extent=[delays[0], delays[-1], dopplers[-1], dopplers[0]],
                    aspect='auto',
                    cmap='viridis'
                )
                ax.scatter(delays[pk[1]], dopplers[pk[0]], c='red', s=100, marker='x', linewidth=3)
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Delay (samples)')
                ax.set_ylabel('Doppler Frequency (Hz)')
                ax.grid(alpha=0.2)
                plt.colorbar(im, ax=ax, label='Amplitude (dB)')

            plt.suptitle(f"Direct Wave Suppression Comparison (Frame {frame_idx})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"suppression_comparison_frame{frame_idx}.png"), dpi=150)
            plt.close()

            # 3. DOA spectrum comparison (original vs projected target)
            J_direct_db = 20 * np.log10(J_dp / (np.max(J_dp) + 1e-12) + 1e-12)
            J_target_db = 20 * np.log10(J_tgt / (np.max(J_tgt) + 1e-12) + 1e-12)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(AZIMUTHS_ENU_DEG, J_direct_db, 'b-', linewidth=2)
            ax1.axvline(direct_enu_deg, color='r', linestyle='--', label=f'Direct Wave {direct_enu_deg:.1f}°')
            ax1.set_ylabel("DOA Spectrum (Direct Wave)\nMatching Degree (dB)", fontsize=12)
            ax1.set_ylim(-40, 0)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(AZIMUTHS_ENU_DEG, J_target_db, 'g-', linewidth=2)
            ax2.axvline(target_enu_deg, color='m', linestyle='--', label=f'Target {target_enu_deg:.1f}°')
            ax2.set_xlabel("Azimuth (ENU degrees) 0°=North")
            ax2.set_ylabel("DOA Spectrum (Target)\nEnergy (dB)", fontsize=12)
            ax2.set_ylim(-40, 0)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.suptitle(f"DOA Spectrum Comparison: Direct Wave vs Suppressed Target (Frame {frame_idx})")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir, f"doa_comparison_frame{frame_idx}.png"), dpi=150)
            plt.close()

    print(f"\n🎉 Multi-frame processing completed! Results saved to: {output_csv}")

if __name__ == "__main__":
    main()

