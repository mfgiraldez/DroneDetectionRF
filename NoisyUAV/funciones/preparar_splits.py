import sys
import os
import glob
import re
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('..'))
from NoisyUAV.funciones import DATA_DIR, TARGET_NOISE

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "dataset_splits.csv")

def is_drone(target_idx: int) -> int:
    # 0 = Ruido, 1 = Dron
    return 0 if target_idx == TARGET_NOISE else 1

def assign_curriculum_group(snr: int) -> str:
    """
    Grupo A (Fácil): +10 a +30 dB
    Grupo B (Medio): -6 a +8 dB
    Grupo C (Extremo): -20 a -8 dB
    """
    if snr >= 10:
        return 'A'
    elif snr >= -6:
        return 'B'
    else:
        return 'C'

def main():
    print("Calculando splits del dataset...")
    
    # 1. Recopilar todos los archivos
    pattern = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")
    archivos = glob.glob(os.path.join(DATA_DIR, "IQdata_*.pt"))
    
    data = []
    for f in archivos:
        filename = os.path.basename(f)
        m = pattern.match(filename)
        if m:
            sample_id = int(m.group(1))
            target = int(m.group(2))
            snr = int(m.group(3))
            
            data.append({
                'filename': filename,
                'filepath': f,
                'target': target,
                'is_drone': is_drone(target),
                'snr': snr,
                'curriculum_group': assign_curriculum_group(snr)
            })
            
    df = pd.DataFrame(data)
    print(f"Total muestras encontradas: {len(df)}")
    
    # Columna combinada para estratificación perfecta
    # Queremos que la proporción de Target (0-6) y SNR (-20 a 30) sea exacta en Train/Val/Test
    df['stratify_key'] = df['target'].astype(str) + "_" + df['snr'].astype(str)
    
    # 2. División Train (70%), Val (15%), Test (15%)
    # Primero separamos Test (15%)
    df_temp, df_test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['stratify_key']
    )
    
    # Luego de Temp (85%) separamos Val (15% del total -> 15/85 del resto)
    df_train, df_val = train_test_split(
        df_temp, test_size=(0.15 / 0.85), random_state=42, stratify=df_temp['stratify_key']
    )
    
    # 3. Asignar etiquetas de split
    df.loc[df_train.index, 'split'] = 'train'
    df.loc[df_val.index, 'split'] = 'val'
    df.loc[df_test.index, 'split'] = 'test'
    
    # Quitar columna temporal
    df = df.drop(columns=['stratify_key'])
    
    # 4. Chequeo de cordura (Sanity Check)
    print("\n--- Distribución por Split ---")
    print(df['split'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    print("\n--- Distribución de Grupos Curriculum en Train ---")
    print(df[df['split'] == 'train']['curriculum_group'].value_counts().sort_index())
    
    # 5. Guardar CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Splits guardados con éxito en: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
