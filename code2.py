# This code implements GPU-accelerated computation of multi-frame CAF (Cross Ambiguity Function) for passive radar, 
# Saves the peak data of all frames to a CSV file.
# Note: Direct path interference is NOT considered in this implementation.
import numpy as np
import matplotlib as mpl
import time
import os
import pandas as pd

# ===== Configuration Parameters =====
SAMPLE_RATE = 2.4e6          # 2.4 MHz
SAMPLE_COUNT = 600000
CHANNEL_COUNT = 9
USE_CHANNEL = 0              # 0 represents the center channel (circle center)
MAX_DELAY = 300
MAX_DOPPLER = 200
DOPPLER_STEP = 0.01          # High precision: 0.01 Hz
REF_FILE = "ref.cf32"
RX_FILE = "SAMPLESTREAM" # TODO: Modify this path to your actual IQ data file (e.g., xxx.cf32)

# ===== User Optional Interface: Specify End Frame (Set to None for processing all frames) =====
USER_END_FRAME = None        # ←←← Modify here! Example: USER_END_FRAME = 10

# ===== Utility Functions: Load Data & Get Total Frame Count =====
def get_total_frames(rx_file, channel_count, sample_count_per_frame):
    file_size = os.path.getsize(rx_file)
    complex64_size = 8  # np.complex64 = 8 bytes
    total_samples = file_size // complex64_size
    samples_per_frame = channel_count * sample_count_per_frame
    total_frames = total_samples // samples_per_frame
    return total_frames

def load_array_channel(filename, channel_idx, total_channels, sample_count_per_frame, frame_index):
    all_data = np.fromfile(filename, dtype=np.complex64)
    samples_per_frame = total_channels * sample_count_per_frame
    n_frames = len(all_data) // samples_per_frame
    if not (0 <= frame_index < n_frames):
        raise ValueError(f"Invalid frame index: {frame_index} (total {n_frames} frames)")
    frame_start = frame_index * samples_per_frame
    channel_start = frame_start + channel_idx * sample_count_per_frame
    return all_data[channel_start:channel_start + sample_count_per_frame]

def load_reference_signal(filename, sample_count):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Reference signal file not found: {filename}")
    data = np.fromfile(filename, dtype=np.complex64, count=sample_count)
    if len(data) < sample_count:
        raise ValueError(f"Insufficient reference signal samples: {len(data)} < {sample_count}")
    return data

# ===== Frequency Domain CAF (GPU) — Return Peak Values Only =====
def compute_caf_peak_gpu(ref, obs, max_delay, fs, max_doppler, doppler_step):
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("❌ CuPy not installed!")

    ref_gpu = cp.asarray(ref, dtype=cp.complex64)
    obs_gpu = cp.asarray(obs, dtype=cp.complex64)
    ref_fft_gpu = cp.fft.fft(ref_gpu).conj()

    delays = np.arange(-max_delay, max_delay + 1)
    dopplers = np.arange(-max_doppler, max_doppler + doppler_step, doppler_step)
    t_gpu = cp.arange(len(ref), dtype=cp.float32) / fs

    caf_gpu = cp.zeros((len(dopplers), len(delays)), dtype=cp.float32)
    N = len(ref)
    center = N // 2

    for i, fd in enumerate(dopplers):
        phase = cp.exp(-1j * 2 * cp.pi * fd * t_gpu)
        obs_shifted = obs_gpu * phase
        corr_freq = ref_fft_gpu * cp.fft.fft(obs_shifted)
        corr_time = cp.fft.ifft(corr_freq)
        corr_mag = cp.abs(corr_time)

        corr_mag_shifted = cp.concatenate([corr_mag[center:], corr_mag[:center]])
        caf_gpu[i, :] = corr_mag_shifted[center - max_delay : center + max_delay + 1]

    idx_flat = cp.argmax(caf_gpu)
    i_doppler, j_delay = cp.unravel_index(idx_flat, caf_gpu.shape)
    peak_delay = delays[j_delay.get()]
    peak_doppler = dopplers[i_doppler.get()]

    return peak_delay, peak_doppler

# ===== Main Program =====
if __name__ == "__main__":
    print("🚀 Passive Radar Multi-Frame CAF Analysis (GPU Version)")
    print(f"🔧 Sample Rate: {SAMPLE_RATE/1e6:.1f} MHz | Samples per frame: {SAMPLE_COUNT}")
    print(f"🔍 Channel in use: {USE_CHANNEL}")

    # === 1. Get Total Frame Count ===
    try:
        total_frames = get_total_frames(RX_FILE, CHANNEL_COUNT, SAMPLE_COUNT)
        print(f"📊 Observation file contains {total_frames} frames in total")
    except Exception as e:
        print(f"❌ Failed to get total frame count: {e}")
        exit(1)

    # === 2. Determine Frame Range ===
    START_FRAME = 0
    if USER_END_FRAME is not None:
        END_FRAME = min(USER_END_FRAME, total_frames - 1)
        print(f"📌 User specified: Process only frames [{START_FRAME}, {END_FRAME}]")
    else:
        END_FRAME = total_frames - 1
        print(f"📌 Default mode: Process all frames [{START_FRAME}, {END_FRAME}]")

    # === 3. Load Reference Signal ===
    try:
        ref_signal = load_reference_signal(REF_FILE, SAMPLE_COUNT)
        print(f"✅ Reference signal loaded successfully: {len(ref_signal)} samples")
    except Exception as e:
        print(f"❌ Failed to load reference signal: {e}")
        exit(1)

    # === 4. Start Batch Processing ===
    results = []
    total_start = time.time()

    for frame_idx in range(START_FRAME, END_FRAME + 1):
        print(f"\n⏳ Processing frame {frame_idx}/{END_FRAME}...")
        try:
            rx_signal = load_array_channel(
                RX_FILE, USE_CHANNEL, CHANNEL_COUNT, SAMPLE_COUNT, frame_idx
            )
            peak_delay, peak_doppler = compute_caf_peak_gpu(
                ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP
            )

            delay_us = peak_delay / SAMPLE_RATE * 1e6
            print(f"   ✅ Peak Value → Delay: {peak_delay:.0f} samples ({delay_us:.2f} μs), Doppler: {peak_doppler:.2f} Hz")

            results.append({
                "Frame": frame_idx,
                "Channel": USE_CHANNEL,
                "Peak_Delay_Samples": int(peak_delay),
                "Peak_Delay_us": round(delay_us, 2),
                "Peak_Doppler_Hz": round(peak_doppler, 2)
            })

        except Exception as e:
            print(f"   ❌ Failed to process frame {frame_idx}: {e}")
            results.append({
                "Frame": frame_idx,
                "Channel": USE_CHANNEL,
                "Peak_Delay_Samples": np.nan,
                "Peak_Delay_us": np.nan,
                "Peak_Doppler_Hz": np.nan
            })

    # === 5. Save Results ===
    df = pd.DataFrame(results)
    # Create 'predict' directory to store output results (automatically created if not exists)
    predict_dir = "predict"
    os.makedirs(predict_dir, exist_ok=True)  # Ensure directory exists (no error if already present)
    # Define path for CSV output file (stores peak delay/Doppler results for all processed frames)
    csv_path = os.path.join(predict_dir, "out.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")

    # === 6. Output Elapsed Time ===
    total_elapsed = time.time() - total_start
    print(f"\n🎉 Processed {len(results)} frames in total")
    print(f"⏱️  Total computation time: {total_elapsed:.2f} seconds")
    print(f"💾 Results saved to: {os.path.abspath(csv_path)}")

