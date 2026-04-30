import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = r"C:\TFM_data\NoisyUAV\stage2\metadata_stage2.csv"
df = pd.read_csv(csv_path)

# Logic from get_dataloaders in dataset_stage2.py
test_size = 0.15
val_size = 0.15

df_train, df_temp = train_test_split(
    df, 
    test_size=(test_size + val_size), 
    stratify=df["label"], 
    random_state=42
)

val_ratio = val_size / (test_size + val_size)
df_val, df_test = train_test_split(
    df_temp, 
    test_size=(1.0 - val_ratio), 
    stratify=df_temp["label"], 
    random_state=42
)

print(f"Stage 2 Metadata Total: {len(df)}")
print(f"Stage 2 Train: {len(df_train)}")
print(f"Stage 2 Val  : {len(df_val)}")
print(f"Stage 2 Test : {len(df_test)}")
