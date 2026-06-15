import os
import pandas as pd
import ast
from pathlib import Path

# Configuración
INPUT_CSV = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\figuras_golden_filtrado_umbral75\golden_results_v2_filt.csv"
OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\figuras_golden_filtrado_umbral75"
OUT_CSV = os.path.join(OUT_DIR, "golden_results_v2_filt.csv")

THRESHOLD = 0.75
MIN_CONSECUTIVE = 2

def apply_temporal_filter(probs_list):
    consecutive_count = 0
    for p in probs_list:
        if p >= THRESHOLD:
            consecutive_count += 1
            if consecutive_count >= MIN_CONSECUTIVE:
                return 1
        else:
            consecutive_count = 0
    return 0

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Leyendo datos originales desde {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Aplicando filtro temporal (Umbral={THRESHOLD}, Consecutivas={MIN_CONSECUTIVE})...")
    # Asegurar que las listas son listas
    df['window_probs_list'] = df['window_probs'].apply(ast.literal_eval)
    
    # Recalcular la predicción final
    df['pred_bin'] = df['window_probs_list'].apply(apply_temporal_filter)
    
    # Limpiar columnas temporales si queremos
    df = df.drop(columns=['window_probs_list'])
    
    df.to_csv(OUT_CSV, index=False)
    print(f"Nuevos resultados generados al instante en: {OUT_CSV}")

if __name__ == "__main__":
    main()
