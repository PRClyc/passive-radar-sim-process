# Passive Radar CAF (Cross Ambiguity Function) Performance Comparison
# Purpose: This code is specifically used to discuss/analyze the GPU acceleration process for CAF computation
# Note: Direct path interference is NOT considered in this implementation (focus solely on CAF acceleration analysis)
# Compare time differences between CPU time-domain, CPU frequency-domain, and GPU frequency-domain implementations.
import numpy as np
import matplotlib as mpl
import time
import os
import matplotlib.pyplot as plt

# ===== Configuration Parameters =====
SAMPLE_RATE = 2.4e6
SAMPLE_COUNT = 600000
CHANNEL_COUNT = 9
FRAME_INDEX = 0
USE_CHANNEL = 0   # 0 represents the center channel (circle center)
MAX_DELAY = 300
MAX_DOPPLER = 200
DOPPLER_STEP = 0.5
FFT_UPSAMPLE_FACTOR = 8
REF_FILE = "ref.cf32"
RX_FILE = "SAMPLESTREAM" # TODO: Modify this path to your actual IQ data file (e.g., xxx.cf32)

# ===== Utility Function: Load Data =====
def load_array_channel(filename, channel_idx, total_channels, sample_count_per_frame, frame_index):
    all_data = np.fromfile(filename, dtype=np.complex64)
    samples_per_frame = total_channels * sample_count_per_frame
    n_frames = len(all_data) // samples_per_frame
    if not (0 <= frame_index < n_frames):
        raise ValueError(f"Invalid frame index: {frame_index} (total {n_frames} frames)")
    frame_start = frame_index * samples_per_frame
    channel_start = frame_start + channel_idx * sample_count_per_frame
    return all_data[channel_start:channel_start + sample_count_per_frame], n_frames

def load_reference_signal(filename, sample_count):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Reference signal file not found: {filename}")
    data = np.fromfile(filename, dtype=np.complex64, count=sample_count)
    if len(data) < sample_count:
        raise ValueError(f"Insufficient reference signal samples: {len(data)} < {sample_count}")
    return data

# ===== Frequency Domain CAF (CPU) =====
def compute_caf_freq_domain(ref, obs, max_delay, fs, max_doppler, doppler_step):
    print("\n⏳ [Frequency Domain - CPU] Computing CAF...")
    start = time.time()
    delays = np.arange(-max_delay, max_delay + 1)
    dopplers = np.arange(-max_doppler, max_doppler + doppler_step, doppler_step)
    ref_fft = np.fft.fft(ref)
    caf = np.zeros((len(dopplers), len(delays)))
    for i, fd in enumerate(dopplers):
        t = np.arange(len(ref)) / fs
        compensation = np.exp(-1j * 2 * np.pi * fd * t)
        obs_shifted = obs * compensation
        corr = np.fft.ifft(ref_fft.conj() * np.fft.fft(obs_shifted))
        corr = np.abs(np.fft.fftshift(corr))
        center = len(corr) // 2
        caf[i, :] = corr[center - max_delay : center + max_delay + 1]
    elapsed = time.time() - start
    print(f"✅ [Frequency Domain - CPU] Completed! Time elapsed: {elapsed:.2f} seconds")
    return caf / (np.max(caf) + 1e-12), delays, dopplers

# ===== Pure Time Domain CAF =====
def compute_caf_time_domain_high_res(ref, obs, max_delay, fs, max_doppler, doppler_step,
                                     fft_upsample_factor=FFT_UPSAMPLE_FACTOR):
    print("\n⏳ [Time Domain] Computing CAF (pure time domain, high frequency sampling)...")
    start = time.time()
    N = len(ref)
    lags = np.arange(-max_delay, max_delay + 1)
    num_lags = len(lags)
    doppler_axis = np.arange(-max_doppler, max_doppler + doppler_step, doppler_step)
    num_doppler = len(doppler_axis)

    L_min = N - max_delay
    N_fft_target = int(np.ceil(fs / doppler_step))
    N_fft_min = max(L_min, N_fft_target)
    N_fft = 1
    while N_fft < N_fft_min:
        N_fft <<= 1
    if N_fft > int(L_min * fft_upsample_factor):
        N_fft = int(2 ** np.ceil(np.log2(L_min * fft_upsample_factor)))

    df = fs / N_fft
    print(f"   Using FFT length N_fft = {N_fft}, theoretical frequency resolution df ≈ {df:.3f} Hz")

    freqs_shift = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1/fs))
    caf = np.zeros((num_doppler, num_lags), dtype=np.float32)

    for j, tau in enumerate(lags):
        L = N - abs(tau)
        if L <= 0:
            continue
        if tau >= 0:
            ref_seg = ref[:L]
            obs_seg = obs[tau:tau + L]
        else:
            shift = -tau
            ref_seg = ref[shift:shift + L]
            obs_seg = obs[:L]
        product = obs_seg * np.conj(ref_seg)
        spectrum = np.fft.fft(product, n=N_fft)
        spectrum_shift = np.fft.fftshift(spectrum)
        mag = np.abs(spectrum_shift)
        caf[:, j] = np.interp(doppler_axis, freqs_shift, mag, left=0.0, right=0.0)

    caf /= (np.max(caf) + 1e-12)
    elapsed = time.time() - start
    print(f"✅ [Time Domain] Completed! Time elapsed: {elapsed:.2f} seconds")
    return caf, lags, doppler_axis

# ===== Frequency Domain CAF (GPU) =====
def compute_caf_freq_domain_gpu(ref, obs, max_delay, fs, max_doppler, doppler_step):
    try:
        import cupy as cp
    except ImportError:
        print("⚠️  CuPy not installed, skipping GPU version.")
        return None, None, None

    print("\n⏳ [Frequency Domain - GPU] Computing CAF...")
    start = time.time()

    # Transfer to GPU
    ref_gpu = cp.asarray(ref, dtype=cp.complex64)
    obs_gpu = cp.asarray(obs, dtype=cp.complex64)
    ref_fft_gpu = cp.fft.fft(ref_gpu).conj()

    delays = np.arange(-max_delay, max_delay + 1)
    dopplers = np.arange(-max_doppler, max_doppler + doppler_step, doppler_step)
    t_gpu = cp.arange(len(ref), dtype=cp.float32) / fs

    caf_gpu = cp.zeros((len(dopplers), len(delays)), dtype=cp.float32)
    center = len(ref) // 2

    for i, fd in enumerate(dopplers):
        phase = cp.exp(-1j * 2 * cp.pi * fd * t_gpu)
        obs_shifted = obs_gpu * phase
        corr_freq = ref_fft_gpu * cp.fft.fft(obs_shifted)
        corr_time = cp.fft.ifft(corr_freq)
        corr_mag = cp.abs(corr_time)

        # Manual fftshift
        corr_mag_shifted = cp.concatenate([corr_mag[center:], corr_mag[:center]])
        caf_gpu[i, :] = corr_mag_shifted[center - max_delay : center + max_delay + 1]

    caf_cpu = (caf_gpu / cp.max(caf_gpu)).get()
    elapsed = time.time() - start
    print(f"✅ [Frequency Domain - GPU] Completed! Time elapsed: {elapsed:.2f} seconds")
    return caf_cpu, delays, dopplers

# ===== Peak Detection and Plotting =====
def find_and_plot_peak(caf, delays, dopplers, method_name, frame_idx, ch_idx):
    idx = np.unravel_index(np.argmax(caf), caf.shape)
    peak_doppler = dopplers[idx[0]]
    peak_delay = delays[idx[1]]
    print(f"\n🎯 {method_name} Peak:")
    print(f"   Delay: {peak_delay} samples ({peak_delay/SAMPLE_RATE*1e6:.2f} μs)")
    print(f"   Doppler: {peak_doppler:.2f} Hz")
    
    plt.figure(figsize=(10, 6))
    extent = [delays[0], delays[-1], dopplers[-1], dopplers[0]]
    plt.imshow(20 * np.log10(caf + 1e-12), extent=extent, aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Delay (samples)')
    plt.ylabel('Doppler (Hz)')
    plt.title(f"{method_name} — Frame {frame_idx}, Channel {ch_idx}")
    plt.scatter(peak_delay, peak_doppler, c='red', s=100, marker='x', linewidth=2, label='Peak')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===== Main Program =====
if __name__ == "__main__":
    print("🚀 Passive Radar CAF Analysis (Time Domain vs Frequency Domain vs Frequency Domain-GPU)")
    print(f"🔧 Sample Rate: {SAMPLE_RATE/1e6:.1f} MHz | Samples per frame: {SAMPLE_COUNT}")
    print(f"🔍 Analysis: Frame {FRAME_INDEX}, Channel {USE_CHANNEL}, Delay ±{MAX_DELAY}, Doppler ±{MAX_DOPPLER} Hz")

    try:
        rx_signal, total_frames = load_array_channel(
            RX_FILE, USE_CHANNEL, CHANNEL_COUNT, SAMPLE_COUNT, FRAME_INDEX
        )
        ref_signal = load_reference_signal(REF_FILE, SAMPLE_COUNT)
        print(f"✅ Data loaded successfully: Reference signal {len(ref_signal)} samples, Observation signal {len(rx_signal)} samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        exit(1)

    # --- 1. Frequency Domain CPU ---
    caf_freq, delays_f, dopplers_f = compute_caf_freq_domain(
        ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP
    )
    find_and_plot_peak(caf_freq, delays_f, dopplers_f, "Frequency Domain CAF (CPU)", FRAME_INDEX, USE_CHANNEL)

    # --- 2. Pure Time Domain ---
    caf_time, delays_t, dopplers_t = compute_caf_time_domain_high_res(
        ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP
    )
    find_and_plot_peak(caf_time, delays_t, dopplers_t, "Time Domain CAF (High Frequency Sampling)", FRAME_INDEX, USE_CHANNEL)

    # --- 3. Frequency Domain GPU ---
    caf_gpu, delays_g, dopplers_g = compute_caf_freq_domain_gpu(
        ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP
    )
    if caf_gpu is not None:
        find_and_plot_peak(caf_gpu, delays_g, dopplers_g, "Frequency Domain CAF (GPU)", FRAME_INDEX, USE_CHANNEL)

    # --- Performance Comparison ---
    print("\n" + "="*60)
    print("⏱️  Precise Timing Comparison (computation only, no plotting)")

    # CPU Frequency Domain
    t0 = time.time()
    _ = compute_caf_freq_domain(ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP)
    t_freq = time.time() - t0

    # Time Domain
    t0 = time.time()
    _ = compute_caf_time_domain_high_res(ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP)
    t_time = time.time() - t0

    # GPU Frequency Domain
    t_gpu = float('inf')
    if caf_gpu is not None:
        t0 = time.time()
        _ = compute_caf_freq_domain_gpu(ref_signal, rx_signal, MAX_DELAY, SAMPLE_RATE, MAX_DOPPLER, DOPPLER_STEP)
        t_gpu = time.time() - t0

    print(f"\n📊 Computation Time:")
    print(f"   Frequency Domain (CPU):     {t_freq:.2f} seconds")
    print(f"   Time Domain (High Res):     {t_time:.2f} seconds")
    if t_gpu != float('inf'):
        print(f"   Frequency Domain (GPU):     {t_gpu:.2f} seconds")
        print(f"   GPU Speedup Ratio:          {t_freq/t_gpu:.1f}x")
    else:
        print("   Frequency Domain (GPU):     ❌ Not run (CuPy not installed or unavailable)")

    print("\n🎉 All three comparisons completed!")

