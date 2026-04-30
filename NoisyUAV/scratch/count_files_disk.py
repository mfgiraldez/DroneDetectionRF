import glob
import os

path = r"C:\TFM_data\NoisyUAV\drone_RF_data"
pattern = "*_target2_snr-10.pt"
files = glob.glob(os.path.join(path, pattern))
print(f"Total files on disk for Target 2, SNR -10: {len(files)}")
