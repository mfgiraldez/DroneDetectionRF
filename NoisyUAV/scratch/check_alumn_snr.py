import pandas as pd
import numpy as np

csv_path = r'c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv'
df = pd.read_csv(csv_path)
snrs = np.sort(df['snr'].unique())
print(f"Unique SNRs in dataset: {snrs}")
print(f"Min SNR: {snrs.min()} dB")
print(f"Max SNR: {snrs.max()} dB")

print("\nValue counts for SNR:")
print(df['snr'].value_counts().sort_index())
