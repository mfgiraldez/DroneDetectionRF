"""
prepare_ht_dataset.py — Generador del Dataset Hard Test
=========================================================
Lee el CSV original del V2.1 y genera un CSV nuevo donde:
  - Train y Val: se eliminan TODAS las filas con target_multiclass == TARGET_HELD_OUT
  - Test:        se mantiene intacto (el Golden Set es el que evalúa)

El dron excluido (TARGET_HELD_OUT=5, Taranis) NUNCA se verá en entrenamiento,
pero sí aparecerá en el Golden Set para medir la generalización zero-shot.

Uso:
    python prepare_ht_dataset.py
"""
import os
import pandas as pd

TARGET_HELD_OUT = 5   # Taranis (T5) — el dron que NUNCA verá el modelo

SRC_CSV = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\dataset_v2_1_clean_pointers.csv"
OUT_CSV = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual_hard_test\dataset_ht.csv"

def main():
    df = pd.read_csv(SRC_CSV)
    print(f"CSV original: {len(df)} instancias")
    print(f"Distribución por split:\n{df['split'].value_counts()}")
    print(f"\nDistribución target_multiclass en TRAIN/VAL:")
    tv = df[df['split'].isin(['train', 'val'])]
    print(tv['target_multiclass'].value_counts().sort_index())

    # Eliminar T5 de train y val SOLAMENTE
    mask_remove = (df['split'].isin(['train', 'val'])) & (df['target_multiclass'] == TARGET_HELD_OUT)
    n_removed = mask_remove.sum()
    df_ht = df[~mask_remove].copy()

    print(f"\n[HT] Filas eliminadas de train/val (Target={TARGET_HELD_OUT}): {n_removed}")
    print(f"[HT] CSV resultante: {len(df_ht)} instancias")
    print(f"[HT] Distribución por split:\n{df_ht['split'].value_counts()}")
    print(f"\n[HT] Target_multiclass en TRAIN/VAL tras filtrado:")
    tv_ht = df_ht[df_ht['split'].isin(['train', 'val'])]
    print(tv_ht['target_multiclass'].value_counts().sort_index())

    # Verificar: target_multiclass=5 NO debe aparecer en train ni val
    leakage = df_ht[(df_ht['split'].isin(['train', 'val'])) &
                    (df_ht['target_multiclass'] == TARGET_HELD_OUT)]
    assert len(leakage) == 0, f"ERROR: {len(leakage)} filas de T{TARGET_HELD_OUT} encontradas en train/val!"
    print(f"\n[OK] Auditoría de leakage: 0 filas de Target={TARGET_HELD_OUT} en train/val.")

    df_ht.to_csv(OUT_CSV, index=False)
    print(f"[OK] CSV guardado en: {OUT_CSV}")

if __name__ == "__main__":
    main()
