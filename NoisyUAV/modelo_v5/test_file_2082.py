import os, sys, torch, logging
# Add root to sys.path
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v5.pipeline import segment_file

logging.basicConfig(level=logging.INFO)
fe = {'path': r"C:\TFM_data\NoisyUAV\drone_RF_data\IQdata_sample12476_target4_snr20.pt", 'class': 0, 'snr': 20}
print("Processing file 2082...")
res = segment_file(fe)
print(f"Done! Result: {res}")
