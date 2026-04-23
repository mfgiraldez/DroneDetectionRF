"""
diagnostico_distribucion_training.py
Diagnostica cuántos bursts de entrenamiento existen para cada (target, SNR).
Ejecutar desde c:\\repos\\DroneDetectionRF\\NoisyUAV\\
"""
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_NORM  = r"C:\TFM_data\NoisyUAV\stage2_norm\metadata_stage2.csv"
CSV_PLAIN = r"C:\TFM_data\NoisyUAV\stage2\metadata_stage2.csv"
CSV_PATH  = CSV_NORM if os.path.exists(CSV_NORM) else CSV_PLAIN

NOMBRES = {0:'DJI', 1:'FutabaT14', 2:'FutabaT7', 3:'Graupner',
           4:'Noise', 5:'Taranis', 6:'Turnigy'}

df = pd.read_csv(CSV_PATH)

# Mismo split que en entrenamiento (random_state=42)
df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
df_val,   df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp["label"], random_state=42)

print(f"Total bursts en TRAINING: {len(df_train)}")
print(f"  → Drones : {len(df_train[df_train['label']==1])}")
print(f"  → Noise  : {len(df_train[df_train['label']==0])}")

# ── Distribución por (target × SNR) solo en training ──────────────────────────
drones_train = df_train[df_train["label"] == 1].copy()
drones_train["nombre"] = drones_train["target"].map(NOMBRES)

pivot = (drones_train
         .groupby(["nombre", "snr"])
         .size()
         .unstack(fill_value=0))

print("\n=== BURSTS DE DRON EN TRAINING POR (TARGET × SNR) ===")
print(pivot.to_string())

# ── Resumen: mínimos por target ────────────────────────────────────────────────
print("\n=== MÍNIMO DE BURSTS POR TARGET (peor SNR) ===")
for t_id, t_name in NOMBRES.items():
    if t_id == 4:  # Noise, skip
        continue
    subset = drones_train[drones_train["target"] == t_id]
    if subset.empty:
        print(f"  {t_name:12s}: ❌ SIN DATOS en training")
        continue
    por_snr = subset.groupby("snr").size()
    snr_min_count = por_snr.idxmin()
    count_min = por_snr.min()
    snr_max_count = por_snr.idxmax()
    count_max = por_snr.max()
    total = len(subset)
    print(f"  {t_name:12s}: total={total:5d} | "
          f"SNR más pobre = {snr_min_count:+4.0f}dB ({count_min:3d} bursts) | "
          f"SNR más rica  = {snr_max_count:+4.0f}dB ({count_max:3d} bursts)")
