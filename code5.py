# Code Function Description:
# This code implements the core signal processing workflow for passive radar - target detection and parameter estimation based on Cross-Ambiguity Function (CAF).
# It focuses on solving the critical problem of target signal masking by direct-path interference in passive radar. Core features include:
# 1. Data processing constraint: Analysis is performed using data from the center channel of the array, without introducing spatial information from other array channels.
# 2. Direct-path suppression: Direct-path cancellation is realized using the frequency-domain least squares algorithm (without considering the incident angle/spatial orientation of the direct path).
#    Cancellation is achieved solely through single-channel time-/frequency-domain matching, and the direct-path suppression ratio (dB) is quantified and output.
# 3. CAF analysis: Compute the Cross-Ambiguity Function within the specified delay/Doppler search range, extract peaks after normalization.
# 4. Feature extraction: Locate target delay (samples/microseconds) and Doppler frequency (Hz) corresponding to the CAF peak.
# 5. Visualization output: Plot CAF heatmap, mark and annotate peak coordinates, save high-resolution result images.
# Note: This code only processes single-channel (center) data, does not utilize the spatial resolution capability of the array, and direct-path cancellation does not incorporate incident angle compensation.
# It is suitable for basic CAF analysis and direct-path suppression effect verification of passive radar single-channel signals.
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 16,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})

# ===== Try Importing CuPy (GPU Acceleration) =====
try:
    import cupy as cp
    HAS_CUPY = True
    print("🟢 CuPy detected, enabling GPU acceleration")
except ImportError:
    HAS_CUPY = False
    cp = np  # Fall back to NumPy
    print("🟠 CuPy not detected, using CPU mode (slower speed)")

# ===== Configuration Parameters =====
SAMPLE_RATE = 2.4e6      # 2.4 MHz
SAMPLE_COUNT = 600000    # Samples per frame
CHANNEL_COUNT = 9        # Number of array channels
FRAME_INDEX = 208        # Frame index (0 = first frame)
USE_CHANNEL = 0          # Channel to use (0 = center channel)
MAX_DELAY = 300          # Search range ±300 samples
MAX_DOPPLER = 200        # Search range ±200 Hz
DOPPLER_STEP = 0.01      # Doppler step size (Hz)

# File paths
REF_FILE = "ref.cf32"
RX_FILE = "SAMPLESTREAM" # TODO: Modify this path to your actual IQ data file (e.g., xxx.cf32)

# ===== Core Functions =====
def load_array_channel(filename, channel_idx, total_channels, sample_count_per_frame, frame_index):
    all_data = np.fromfile(filename, dtype=np.complex64)
    samples_per_frame = total_channels * sample_count_per_frame
    n_frames = len(all_data) // samples_per_frame
    
    if not (0 <= frame_index < n_frames):
        raise ValueError(f"Invalid frame index: {frame_index} (total {n_frames} frames)")
    
    frame_start = frame_index * samples_per_frame
    channel_start = frame_start + channel_idx * sample_count_per_frame
    channel_data = all_data[channel_start : channel_start + sample_count_per_frame]
    return channel_data, n_frames

def load_reference_signal(filename, sample_count):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Reference signal file not found: {filename}")
    data = np.fromfile(filename, dtype=np.complex64, count=sample_count)
    if len(data) < sample_count:
        raise ValueError(f"Insufficient reference signal: {len(data)} < {sample_count}")
    return data

def cancel_direct_path(ref, obs, channel_length=128):
    """
    Frequency-domain least squares direct path cancellation (auto uses GPU or CPU)
    """
    N = len(ref)
    if channel_length > N:
        channel_length = N // 2

    L = N + channel_length - 1
    nfft = 2**int(np.ceil(np.log2(L)))

    # Transfer to computation device (GPU or CPU)
    ref_xp = cp.asarray(ref)
    obs_xp = cp.asarray(obs)

    REF = cp.fft.fft(ref_xp, n=nfft)
    OBS = cp.fft.fft(obs_xp, n=nfft)

    epsilon = 1e-6 * cp.mean(cp.abs(REF)**2)
    H_est = (OBS * cp.conj(REF)) / (cp.abs(REF)**2 + epsilon)

    h_est = cp.fft.ifft(H_est)[:channel_length]

    # Manual FFT convolution (avoid circular convolution)
    pad_len = nfft - N
    ref_padded = cp.concatenate([ref_xp, cp.zeros(pad_len, dtype=ref_xp.dtype)])
    h_padded = cp.concatenate([h_est, cp.zeros(nfft - channel_length, dtype=h_est.dtype)])
    dp_reconstructed = cp.fft.ifft(cp.fft.fft(ref_padded) * cp.fft.fft(h_padded))[:N]

    obs_clean_xp = obs_xp - dp_reconstructed

    # Transfer back to NumPy (for subsequent plotting, etc.)
    obs_clean = cp.asnumpy(obs_clean_xp) if HAS_CUPY else obs_clean_xp

    p_before = np.mean(np.abs(obs)**2)
    p_after = np.mean(np.abs(obs_clean)**2)
    ratio_db = 10 * np.log10(p_before / p_after) if p_after > 0 else np.inf
    print(f"📡 Direct path suppression ratio: {ratio_db:.1f} dB")

    return obs_clean

def compute_caf(ref, obs):
    print("\n⏳ Computing CAF...")
    start = time.time()
    
    delays = np.arange(-MAX_DELAY, MAX_DELAY + 1)
    dopplers = np.arange(-MAX_DOPPLER, MAX_DOPPLER + DOPPLER_STEP, DOPPLER_STEP)
    
    N = len(ref)
    t = np.arange(N) / SAMPLE_RATE

    # Transfer to computation device
    t_xp = cp.asarray(t)
    ref_xp = cp.asarray(ref)
    obs_xp = cp.asarray(obs)

    ref_fft_xp = cp.fft.fft(ref_xp)
    caf_xp = cp.zeros((len(dopplers), len(delays)), dtype=cp.float32)

    for i, doppler in enumerate(dopplers):
        compensation = cp.exp(-1j * 2 * cp.pi * doppler * t_xp)
        obs_shifted = obs_xp * compensation
        correlation = cp.fft.ifft(cp.conj(ref_fft_xp) * cp.fft.fft(obs_shifted))
        correlation = cp.abs(cp.fft.fftshift(correlation))
        center = len(correlation) // 2
        caf_xp[i, :] = correlation[center - MAX_DELAY : center + MAX_DELAY + 1]

    caf = cp.asnumpy(caf_xp) if HAS_CUPY else caf_xp
    elapsed = time.time() - start
    print(f"✅ CAF computation completed! Time elapsed: {elapsed:.2f} seconds")
    
    max_val = np.max(caf)
    if max_val > 0:
        caf /= max_val
    return caf, delays, dopplers

def find_peak(caf, delays, dopplers):
    idx = np.unravel_index(np.argmax(caf), caf.shape)
    peak_delay = delays[idx[1]]
    peak_doppler = dopplers[idx[0]]
    peak_value = caf[idx]
    
    print(f"\n🎯 CAF peak detection results:")
    print(f"   Delay: {peak_delay} samples ({peak_delay/SAMPLE_RATE*1e6:.2f} μs)")
    print(f"   Doppler: {peak_doppler:.1f} Hz")
    print(f"   Peak amplitude: {peak_value:.4f} (normalized)")
    return peak_delay, peak_doppler, idx

def plot_caf_results(caf, delays, dopplers, peak_idx, frame_idx, channel_idx):
    plt.figure(figsize=(12, 8))
    im = plt.imshow(20 * np.log10(caf + 1e-10),
                    extent=[delays[0], delays[-1], dopplers[-1], dopplers[0]],
                    aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Amplitude (dB)')
    plt.xlabel('Delay (samples)', fontsize=12)
    plt.ylabel('Doppler (Hz)', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')

    # Get peak coordinates
    peak_delay_val = delays[peak_idx[1]]
    peak_doppler_val = dopplers[peak_idx[0]]

    # Mark peak with red cross
    plt.scatter(peak_delay_val, peak_doppler_val,
                c='red', s=120, marker='x', linewidth=3, label='Detected Peak')

    # Annotate peak coordinates on plot
    plt.text(peak_delay_val, peak_doppler_val,
             f'({int(peak_delay_val)}, {peak_doppler_val:.3f})',
             color='white', fontsize=16, fontweight='bold',
             ha='left', va='bottom',
             bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.3'))

    title = f"CAF Analysis (Least Squares Direct Path Cancellation)"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(loc='upper right')

    output_file = f"caf_frame{frame_idx}_ls_ch{channel_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 CAF results saved to: {output_file}")

    plt.tight_layout()
    plt.show()
    return output_file

# ===== Main Program =====
if __name__ == "__main__":
    print("🚀 Passive Radar CAF Analysis (with Direct Path Cancellation + GPU Acceleration)")
    print(f"🔧 Sample rate: {SAMPLE_RATE/1e6:.1f} MHz | Samples per frame: {SAMPLE_COUNT}")
    print(f"🔧 Array: {CHANNEL_COUNT} channels | Analyzing: Frame {FRAME_INDEX}, Channel {USE_CHANNEL}")

    try:
        # 1. Load data
        print("\n⏳ Loading data...")
        rx_signal, total_frames = load_array_channel(RX_FILE, USE_CHANNEL, CHANNEL_COUNT, SAMPLE_COUNT, FRAME_INDEX)
        ref_signal = load_reference_signal(REF_FILE, SAMPLE_COUNT)
        
        print(f"✅ Array data: {total_frames} frames")
        print(f"✅ Reference signal: {len(ref_signal)} samples")
        print(f"📊 Power comparison: Reference={10*np.log10(np.mean(np.abs(ref_signal)**2)):.1f} dB, "
              f"Received={10*np.log10(np.mean(np.abs(rx_signal)**2)):.1f} dB")

        # 2. [Key Step] Direct path cancellation
        print("\n🧹 Performing direct path cancellation...")
        rx_clean = cancel_direct_path(ref_signal, rx_signal, channel_length=128)

        # 3. Compute CAF (using canceled signal!)
        caf_matrix, delay_axis, doppler_axis = compute_caf(ref_signal, rx_clean)

        # 4. Detect peak
        peak_delay, peak_doppler, peak_idx = find_peak(caf_matrix, delay_axis, doppler_axis)

        # 5. Plot results (using your specified style)
        print("\n🖼️  Generating CAF visualization...")
        plot_caf_results(caf_matrix, delay_axis, doppler_axis, peak_idx, FRAME_INDEX, USE_CHANNEL)

        print("\n🎉 Processing completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

