import pandas as pd

csv_path = r'c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv'
df = pd.read_csv(csv_path)
counts = df['split'].value_counts()
print(f"Split counts:\n{counts}")
print(f"\nTotal test samples: {counts.get('test', 0)}")
