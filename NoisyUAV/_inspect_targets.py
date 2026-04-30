import sys
sys.path.insert(0, 'c:/repos/DroneDetectionRF')
from NoisyUAV.funciones.dataset import obtener_splits_dataset
import pandas as pd

# Ver distribución completa por target_multiclass y SNR en el test set
_, _, df_test = obtener_splits_dataset()

print("=== TOTAL TEST SET ===")
print(f"  Total muestras: {len(df_test)}")
print(f"  Noise (label=0): {(df_test['label']==0).sum()}")
print(f"  Drone (label=1): {(df_test['label']==1).sum()}")

print("\n=== MUESTRAS POR TARGET Y SNR (drones) ===")
drones = df_test[df_test['label']==1]
pivot = drones.pivot_table(index='target_multiclass', columns='snr', aggfunc='size', fill_value=0)
print(pivot.to_string())

print("\n=== TOTAL POR TARGET ===")
print(drones.groupby('target_multiclass').size())

print("\n=== TOTAL DATASET (train+val+test) ===")
df_train, df_val, df_test2 = obtener_splits_dataset()
df_all = pd.concat([df_train, df_val, df_test2])
print(f"  Total: {len(df_all)}")
print(f"  Drone (label=1): {(df_all['label']==1).sum()}")
print(f"  Noise (label=0): {(df_all['label']==0).sum()}")
print(df_all.groupby(['label','target_multiclass']).size().reset_index(name='n').to_string())
