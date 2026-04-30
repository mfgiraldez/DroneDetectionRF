import pandas as pd

csv_path = r'c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv'
df = pd.read_csv(csv_path)

target = 2
snr = -10

subset = df[(df['target_multiclass'] == target) & (df['snr'] == snr)]
print(f"Total ráfagas para Target {target}, SNR {snr}: {len(subset)}")
print(f"Distribución de splits:\n{subset['split'].value_counts()}")

print(f"\nArchivos únicos en TEST para este caso:")
test_files = subset[subset['split'] == 'test']['file_path'].unique()
print(len(test_files))
print(test_files)
