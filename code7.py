#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code Function Description:
# This code implements visual analysis of passive radar target localization results, with the core function of completing the conversion from geographic coordinate system (latitude/longitude) to ENU local coordinate system,
# and comparing and displaying the differences between the true target trajectory and the passive radar estimated trajectory. Key features include:
# 1. Coordinate conversion: Based on the WGS84 ellipsoid model, implement the complete conversion from geodetic coordinate system (latitude/longitude/altitude) → ECEF Earth-centered coordinate system → ENU local coordinate system;
# 2. Data preprocessing: Load passive radar estimation result CSV and true trajectory CSV, align data lengths, and filter invalid (empty/non-numeric) estimation values;
# 3. ENU coordinate system construction: Take the receiver (Rx) as the origin, and uniformly convert the true/estimated positions of the receiver, transmitter, and target to the ENU coordinate system;
# 4. Trajectory visualization: Plot the true target trajectory (blue solid line) and estimated trajectory (orange dashed line) in the ENU plane, and mark the positions of the transmitter and receiver;
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# === Configuration: Your System Parameters ===
# TODO: Replace the placeholder values below with your actual receiver/transmitter coordinates
RX_LAT, RX_LON, RX_ALT = 0.0, 0.0, 0.0   
# Receiver (ENU origin) - Latitude (deg), Longitude (deg), Altitude (m)
TX_LAT, TX_LON, TX_ALT = 0.0, 0.0, 0.0  
# Transmitter - Latitude (deg), Longitude (deg), Altitude (m)

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F**2

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
    x = (N + alt_m) * cos_lat * np.cos(lon)
    y = (N + alt_m) * cos_lat * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z])

def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, h0_m):
    ref_ecef = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    dx, dy, dz = x - ref_ecef[0], y - ref_ecef[1], z - ref_ecef[2]
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    east  = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = -np.cos(lon0)*np.sin(lat0)*dx - np.sin(lon0)*np.sin(lat0)*dy + np.cos(lat0)*dz
    return east, north

def safe_float(x):
    try:
        return float(x) if str(x).strip() != "" else np.nan
    except:
        return np.nan

def main():
    # === File Paths ===
    pred_file = os.path.expanduser("~/passim-main-new/predict/estimates.csv")
    truth_file = "TRUTHCSV" # TODO: Replace this with your actual ground truth CSV file path (set in previous configuration)

    # === Load Data ===
    pred_df = pd.read_csv(pred_file)
    truth_df = pd.read_csv(truth_file)

    min_len = min(len(pred_df), len(truth_df))
    pred_df = pred_df.iloc[:min_len].reset_index(drop=True)
    truth_df = truth_df.iloc[:min_len].reset_index(drop=True)

    # Extract latitude and longitude
    pred_lat = pred_df['target_lat'].apply(safe_float).values
    pred_lon = pred_df['target_lon'].apply(safe_float).values
    truth_lat = truth_df['target_lat'].astype(float).values
    truth_lon = truth_df['target_lon'].astype(float).values

    # Filter valid estimations
    valid = ~np.isnan(pred_lat) & ~np.isnan(pred_lon)
    pred_lat, pred_lon = pred_lat[valid], pred_lon[valid]
    truth_lat, truth_lon = truth_lat[valid], truth_lon[valid]

    # === Convert all points to ENU (Rx as origin) ===
    def to_enu(lat, lon):
        x, y, z = geodetic_to_ecef(lat, lon, 0.0)
        e, n = ecef_to_enu(x, y, z, RX_LAT, RX_LON, RX_ALT)
        return e, n

    # Receiver station (origin)
    rx_e, rx_n = 0.0, 0.0

    # Transmitter station
    tx_e, tx_n = to_enu(TX_LAT, TX_LON)

    # True trajectory
    truth_points = [to_enu(lat, lon) for lat, lon in zip(truth_lat, truth_lon)]
    truth_e = [p[0] for p in truth_points]
    truth_n = [p[1] for p in truth_points]

    # Estimated trajectory
    pred_points = [to_enu(lat, lon) for lat, lon in zip(pred_lat, pred_lon)]
    pred_e = [p[0] for p in pred_points]
    pred_n = [p[1] for p in pred_points]

    # === Plotting ===
    plt.figure(figsize=(10, 8))

    # True trajectory
    plt.plot(truth_e, truth_n, 'b-', linewidth=2, label='True Trajectory', alpha=0.8)
    # Estimated trajectory
    plt.plot(pred_e, pred_n, 'orange', linestyle='--', linewidth=2, label='Estimated Trajectory', alpha=0.8)

    # Station markers
    plt.scatter(rx_e, rx_n, c='green', s=200, marker='^', edgecolor='k', linewidth=1, label='Receiver (Rx)')
    plt.scatter(tx_e, tx_n, c='red', s=200, marker='v', edgecolor='k', linewidth=1, label='Transmitter (Tx)')

    # Add legend and labels
    plt.xlabel('East (m)', fontsize=12)
    plt.ylabel('North (m)', fontsize=12)
    plt.title('Passive Radar Localization Results (ENU Coordinate System)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # Maintain equal scale

    # Save plot
    output_dir = os.path.expanduser("~/passive-radar-sim-process")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "simple_trajectory.png"), dpi=200, bbox_inches='tight')
    plt.show()

    print(f"✅ Plot saved to: {os.path.join(output_dir, 'simple_trajectory.png')}")

if __name__ == "__main__":
    main()

