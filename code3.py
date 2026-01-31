# This code is used for physical quantity comparison analysis between predicted results and ground truth of CAF (Cross Ambiguity Function) for passive radar. 
# It converts predicted delay/Doppler values into physical quantities (bistatic range difference/speed), calculates error metrics, and generates comparison plots.
# Note: Direct path interference is NOT considered in this implementation.
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== Configuration =====
PREDICT_CSV = "predict/out.csv"
TRUTH_CSV = "TRUTHCSV" # TODO: Replace this with your actual ground truth CSV file path (set in previous configuration)
OUTPUT_PLOT = "./xxx.png"
SKIP_FRAMES = 1

# ===== Physical Constants (Strictly Consistent with C Code) =====
LIGHTSPEED_IN_AIR = 2.99709e8   # m/s
CENTER_FREQ       = 105e6       # Hz
SAMPLE_RATE       = 2.4e6       # Hz

LAMBDA = LIGHTSPEED_IN_AIR / CENTER_FREQ  # Wavelength (meters)

# ===== Data Loading =====
pred_df = pd.read_csv(PREDICT_CSV)
truth_df = pd.read_csv(TRUTH_CSV)

min_len = min(len(pred_df), len(truth_df))
pred_df = pred_df.iloc[:min_len].reset_index(drop=True)
truth_df = truth_df.iloc[:min_len].reset_index(drop=True)

# --- Predicted Values (Model Output) ---
pred_delay = pred_df['Peak_Delay_Samples'].values
pred_doppler = pred_df['Peak_Doppler_Hz'].values

# --- Ground Truth Values (Directly from CSV) ---
truth_route_diff = truth_df['route_diff_m'].values          # meters
truth_bistatic_speed = truth_df['bistatic_speed_mps'].values  # m/s

# --- Convert Predicted Values to Same Physical Quantities ---
pred_route_diff = pred_delay * (LIGHTSPEED_IN_AIR / SAMPLE_RATE)      # meters
pred_bistatic_speed = -pred_doppler * LAMBDA                         # m/s

# ===== Error Calculation (Physical Quantities) =====
route_errors = pred_route_diff - truth_route_diff
speed_errors = pred_bistatic_speed - truth_bistatic_speed

mae_route = np.mean(np.abs(route_errors))
rmse_route = np.sqrt(np.mean(route_errors**2))
mae_speed = np.mean(np.abs(speed_errors))
rmse_speed = np.sqrt(np.mean(speed_errors**2))

# ===== Downsampling for Plotting =====
indices = np.arange(0, min_len, SKIP_FRAMES)
x_pred = pred_bistatic_speed[indices]    # X-axis: Speed (m/s)
y_pred = pred_route_diff[indices]        # Y-axis: Path difference (meters)
x_truth = truth_bistatic_speed[indices]
y_truth = truth_route_diff[indices]

# ===== Error Text =====
error_text = (
    f"Bistatic Range Difference Error\n"
    f"  MAE: {mae_route:.4f} meters\n"
    f"  RMSE: {rmse_route:.4f} meters\n"
    f"\n"
    f"Bistatic Speed Error\n"
    f"  MAE: {mae_speed:.4f} m/s\n"
    f"  RMSE: {rmse_speed:.4f} m/s"
)

# ===== Plotting =====
plt.figure(figsize=(14, 9))

plt.plot(x_truth, y_truth, color='tab:blue', marker='o', linestyle='-', linewidth=2, markersize=6, label='Ground Truth')
plt.plot(x_pred, y_pred, color='tab:red', marker='s', linestyle='--', linewidth=2, markersize=6, label='Predicted Values')

# Mark Key Points
mark_step = 20
if len(indices) > 1:
    mark_local_indices = [0] + list(range(mark_step, len(indices), mark_step)) + [len(indices) - 1]
    mark_local_indices = sorted(set(mark_local_indices))
else:
    mark_local_indices = [0]

for i in mark_local_indices:
    if i < len(x_truth):
        plt.text(x_truth[i], y_truth[i], f' {i+1}', fontsize=14, ha='right', va='bottom', color='tab:blue', fontweight='bold')
        #plt.text(x_pred[i], y_pred[i], f' {i+1}', fontsize=11, ha='left', va='bottom', color='tab:red', fontweight='bold')
        plt.scatter([x_truth[i]], [y_truth[i]], color='tab:blue', s=80, zorder=5)
        plt.scatter([x_pred[i]], [y_pred[i]], color='tab:red', s=80, zorder=5)

# Legend
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fancybox=True, shadow=True, fontsize=12)

# Error Box
ax = plt.gca()
props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='darkgray')
plt.text(0.15, 0.98, error_text,
         transform=ax.transAxes,
         fontsize=11,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=props)

# Axis Labels (Physical Units!)
plt.xlabel('Bistatic Speed (m/s)', fontsize=14)
plt.ylabel('Bistatic Range Difference (meters)', fontsize=14)
plt.title('Predicted vs Ground Truth (Physical Quantity Space)', fontsize=16, pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save Plot
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"📊 Physical quantity comparison plot saved to: {os.path.abspath(OUTPUT_PLOT)}")
plt.show()

