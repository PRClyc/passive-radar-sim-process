#!/bin/bash

# =====================
# Output/Input Path Configuration (replace with your actual file paths)
# =====================
# IQ signal stream file (CF32 format - typical for IQ signal data)
# Replace: Your actual signal file path (e.g., your_signal_105M.cf32)
SAMPLESTREAM=~/passim-main-new/samplein/iq_signal_sample.cf32

# Time stream metadata file (TXT format - contains time sequence information)
# Replace: Your actual time stream file path (e.g., your_timestream_105M.txt)
TIMESTREAM=~/passim-main-new/tsin/timestream_metadata.txt

# Truth validation file (CSV format - reference data for verification)
# Replace: Your actual truth data file path (e.g., your_truth_data.csv)
TRUTHCSV=~/passim-main-new/truth/ground_truth_data.csv

# =====================
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.

# =====================
# Run PASSIM (Your Currently Finalized Parameters)
# =====================
./passim.out \
  -s 2.4e6 \
  -n 600000  \
  -m 105.5e6 \
  -T "YYYY-MM-DDTHH:MM:SS" \
  -e "YYYY-MM-DDTHH:MM:SS" \
  -d 1 \
  -t "LAT1,LON1,ALT1" \
  -r "LAT2,LON2,ALT2" \
  -f tracks/your_file.kml,1000 \
  -o $SAMPLESTREAM \
  -p $TIMESTREAM \
  -q $TRUTHCSV \
  -x 0 \
  -v 1 \
  -a uca.so


echo "==============================="
echo " PASSIM RUN COMPLETE "
echo " Samples  => $SAMPLESTREAM"
echo " Time log => $TIMESTREAM"
echo " Truth    => $TRUTHCSV"
echo "==============================="
