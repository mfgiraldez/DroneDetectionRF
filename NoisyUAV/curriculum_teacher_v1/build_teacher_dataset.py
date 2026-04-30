import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.funciones.detector_entropia import detectar_bursts

OUTPUT_DIR = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "teacher_dataset_high_snr.csv"

# Parámetros CFAR relajados (Etapa 1: Maximizar Propuestas / Recall)
FS = 14e6
NPERSEG = 2048
Z_THRESH = 2.0       
MIN_BURST_MS = 0.25  
MERGE_GAP_MS = 1.0   
BG_MULT = 4.0
MAX_BINS_FRAC = 0.25    # Tolerancia generosa para bins activos

# Parámetros Curriculum (Umbral Dinámico Relativo)
Z_RELAX_RATIO = 0.80   # max_z * 0.80

def process_row(row_data):
    idx, file_path, target_mc, snr, label, split_name = row_data
    records = []
    
    # ---------------------------------------------------------
    # CURRICULUM FILTER: Sólo usamos SNR >= 0 para el Teacher
    # ---------------------------------------------------------
    if snr < 0:
        return []
    
    try:
        d  = torch.load(file_path, map_location='cpu', weights_only=False)
        iq = d['x_iq'].float()
    except Exception as e:
        return []
        
    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
        iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC
    )
    
    global_nf     = float(np.median(nf_v))
    global_ns     = float(np.clip(ns, 0, 5))
    global_H      = float(np.mean(H_smooth))
    global_p75_ac = float(np.percentile(n_active, 75))
    
    filename = Path(file_path).name
    
    if len(bursts) == 0:
        # En high-SNR esto rara vez pasa salvo para ruido puro sin impulsos.
        # Guardamos un dummy para que el fichero no se pierda si es de ruido.
        record = {
            'file_path': filename, 'split': split_name, 'label': label,
            'target_multiclass': target_mc, 'snr': snr, 'burst_id': 0,
            't_start': 35.0, 't_end': 35.5, 't_center': 35.25, 'dur_ms': 0.5,
            'z_peak': 0.0, 'drop_b': 0.0, 'n_act_burst': 0.0,
            'global_nf': global_nf, 'global_ns': global_ns, 
            'global_H_mean': global_H, 'global_p75_act': global_p75_ac,
            'is_dummy': True
        }
        records.append(record)
    else:
        # Lógica de Curriculum Dinámico Relativo (Z-Max * 80%)
        max_z_abs = max(abs(b['z_peak']) for b in bursts)
        umbral_dinamico = max_z_abs * Z_RELAX_RATIO
        
        # Filtramos (purificación)
        valid_bursts = [b for b in bursts if abs(b['z_peak']) >= umbral_dinamico]
        
        if len(valid_bursts) == 0:
            record = {
                'file_path': filename, 'split': split_name, 'label': label,
                'target_multiclass': target_mc, 'snr': snr, 'burst_id': 0,
                't_start': 35.0, 't_end': 35.5, 't_center': 35.25, 'dur_ms': 0.5,
                'z_peak': 0.0, 'drop_b': 0.0, 'n_act_burst': 0.0,
                'global_nf': global_nf, 'global_ns': global_ns, 
                'global_H_mean': global_H, 'global_p75_act': global_p75_ac,
                'is_dummy': True
            }
            records.append(record)
        else:
            # Aquí NO descartamos el resto. Si hay 5 bursts válidos (por ej. 5 saltos de dron),
            # creamos 5 filas. Maximizamos el aprovechamiento de datos.
            for b_idx, b in enumerate(valid_bursts):
                record = {
                    'file_path': filename, 'split': split_name, 'label': label,
                    'target_multiclass': target_mc, 'snr': snr, 'burst_id': b_idx+1,
                    't_start': b['t0'], 't_end': b['t1'], 
                    't_center': (b['t0'] + b['t1']) / 2.0, 
                    'dur_ms': np.clip(b['dur_ms'], 0, 75),
                    'z_peak': np.clip(abs(b['z_peak']), 0, 30), 
                    'drop_b': np.clip(b['drop_b'], 0, 10), 
                    'n_act_burst': np.clip(b['n_act'], 0, 2048),
                    'global_nf': global_nf, 'global_ns': global_ns, 
                    'global_H_mean': global_H, 'global_p75_act': global_p75_ac,
                    'is_dummy': False
                }
                records.append(record)
            
    return records

def process_dataframe(df, split_name):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    print(f"\nProcesando split '{split_name}' (Solo SNR >= 0) con Filtro Dinámico 80%...")
    
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((idx, row['filepath'], row['target_multiclass'], row['snr'], row['label'], split_name))
        
    all_records = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_row, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            all_records.extend(future.result())
            
    return all_records

def main():
    print("Obteniendo particiones del dataset original...")
    df_train, df_val, df_test = obtener_splits_dataset()
    
    all_records = []
    
    all_records.extend(process_dataframe(df_train, 'train'))
    all_records.extend(process_dataframe(df_val, 'val'))
    all_records.extend(process_dataframe(df_test, 'test'))
    
    final_df = pd.DataFrame(all_records)
    final_df.to_csv(OUT_CSV, index=False)
    
    print("\n=======================================================")
    print(f"Dataset Oráculo (Teacher) exportado a: {OUT_CSV}")
    print(f"Total bursts puros encontrados (SNR>=0): {len(final_df)}")
    print(f"Balance de Clases (Bursts):")
    print(final_df['label'].value_counts())
    print("=======================================================")
    
if __name__ == '__main__':
    main()
