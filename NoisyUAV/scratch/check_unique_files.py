import pandas as pd

csv_path = r'c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv'
df = pd.read_csv(csv_path)

target = 2
snr = -10

subset = df[(df['target_multiclass'] == target) & (df['snr'] == snr)]
unique_files = subset.groupby('split')['file_path'].nunique()
print(f"Archivos únicos con ráfagas detectadas para Target {target}, SNR {snr}:")
print(unique_files)
print(f"Total: {subset['file_path'].nunique()}")
