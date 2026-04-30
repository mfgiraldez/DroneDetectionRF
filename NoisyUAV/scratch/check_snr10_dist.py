import pandas as pd

csv_path = r'c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv'
df = pd.read_csv(csv_path)

# Filter for drones at SNR -10
drones_snr10 = df[(df['label'] == 1) & (df['snr'] == -10)]

print(f"Total ráfagas Dron SNR -10: {len(drones_snr10)}")
print("\nDistribución por Target y Split:")
dist = drones_snr10.groupby(['target_multiclass', 'split'])['file_path'].nunique().unstack(fill_value=0)
print(dist)
