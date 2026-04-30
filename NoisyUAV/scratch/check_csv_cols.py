import pandas as pd

csv_path = r'c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv'
df = pd.read_csv(csv_path)
print(df.columns.tolist())
print(df.head(2))
