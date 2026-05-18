"""
Generador Dataset V5b — INSTANCIAS INDIVIDUALES para Dual-Stream V2
===================================================================
Cada detección del CFAR = 1 fila en el CSV (nivel de instancia).
Si el CFAR no detecta nada en un fichero, se genera UNA sola fila
con el centro de la señal (fallback mínimo).
Los splits se asignan a NIVEL DE FICHERO para evitar data leakage.
"""
import os, sys, glob, re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_CSV  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v2_dual\dataset_v5_pointers.csv"

# CFAR config
FS = 14e6; NPERSEG = 2048; Z_THRESH = 4.0      
TARGET_NOISE = 4

def get_splits():
    pattern = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")
    archivos = glob.glob(os.path.join(DATA_DIR, "IQdata_*.pt"))
    data = []
    for f in archivos:
        m = pattern.match(os.path.basename(f))
        if m:
            target = int(m.group(2)); snr = int(m.group(3))
            data.append({
                'filepath': f, 'filename': os.path.basename(f),
                'target_multiclass': target,
                'label': 0 if target == TARGET_NOISE else 1,
                'snr': snr,
            })
    df = pd.DataFrame(data)
    
    # Estratificación FINA por target_multiclass × snr
    df['stratify_key'] = df['target_multiclass'].astype(str) + '_' + df['snr'].astype(str)
    
    # Celdas con menos de 3 muestras: reparto manual
    counts = df['stratify_key'].value_counts()
    tiny_keys = counts[counts < 3].index.tolist()
    
    if len(tiny_keys) > 0:
        print(f"  [WARN] {len(tiny_keys)} celdas target x snr con <3 muestras. Reparto manual.")
        df_normal = df[~df['stratify_key'].isin(tiny_keys)].copy()
        df_tiny = df[df['stratify_key'].isin(tiny_keys)].copy()
        
        df_temp, df_test = train_test_split(df_normal, test_size=0.15, random_state=42, stratify=df_normal['stratify_key'])
        df_train, df_val = train_test_split(df_temp, test_size=0.15/0.85, random_state=42, stratify=df_temp['stratify_key'])
        
        df_train['split'] = 'train'
        df_val['split'] = 'val'
        df_test['split'] = 'test'
        
        tiny_parts = []
        for key in tiny_keys:
            sub = df_tiny[df_tiny['stratify_key'] == key].copy()
            n = len(sub)
            sub = sub.sample(frac=1, random_state=42).reset_index(drop=True)
            if n == 1:
                sub['split'] = 'train'
            elif n == 2:
                sub.loc[sub.index[0], 'split'] = 'train'
                sub.loc[sub.index[1], 'split'] = 'test'
            else:
                sub.loc[sub.index[0], 'split'] = 'train'
                sub.loc[sub.index[1], 'split'] = 'val'
                sub.loc[sub.index[2], 'split'] = 'test'
            tiny_parts.append(sub)
        
        return pd.concat([df_train, df_val, df_test] + tiny_parts)
    else:
        df_temp, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['stratify_key'])
        df_train, df_val = train_test_split(df_temp, test_size=0.15/0.85, random_state=42, stratify=df_temp['stratify_key'])
        
        df_train['split'] = 'train'
        df_val['split'] = 'val'
        df_test['split'] = 'test'
        return pd.concat([df_train, df_val, df_test])

def main():
    print("==================================================")
    print(" GENERADOR DATASET V5b (INSTANCIAS PARA DUAL-V2)  ")
    print("==================================================")
    
    df_all = get_splits()
    records = []
    n_cfar = 0
    n_fallback = 0
    n_skipped = 0
    
    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Escaneando archivos"):
        try:
            d = torch.load(row['filepath'], map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
        except Exception:
            continue
            
        t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
            iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
            min_burst_ms=0.5, merge_gap_ms=0.75, min_z_abs=3.5,
            bg_mult=4, max_bins_frac=1.0, smooth_ms=0.3, adaptive_window_ms=10)
            
        global_nf = float(np.median(nf_v))
        global_H_mean = float(np.mean(H_smooth))
        
        if len(bursts) > 0:
            # Una fila por cada detección del CFAR
            for b in bursts:
                t_center = (b['t0'] + b['t1']) / 2.0
                records.append({
                    'filename': row['filename'],
                    'split': row['split'],
                    'label': row['label'],
                    'target_multiclass': row['target_multiclass'],
                    'snr': row['snr'],
                    't_center': t_center,
                    'z_peak': abs(b['z_peak']),
                    'global_nf': global_nf,
                    'global_H_mean': global_H_mean,
                    'is_fallback': False,
                })
                n_cfar += 1
        else:
            # Fallback SOLO para ficheros de RUIDO (label=0).
            # Para drones sin detección CFAR, NO añadimos nada:
            # la augmentación AWGN en el DataLoader cubrirá las SNRs bajas.
            if row['label'] == 0:
                L_ms = (iq.shape[1] / FS) * 1000
                records.append({
                    'filename': row['filename'],
                    'split': row['split'],
                    'label': row['label'],
                    'target_multiclass': row['target_multiclass'],
                    'snr': row['snr'],
                    't_center': L_ms / 2.0,
                    'z_peak': 0.0,
                    'global_nf': global_nf,
                    'global_H_mean': global_H_mean,
                    'is_fallback': True,
                })
                n_fallback += 1
            else:
                n_skipped += 1
                
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"Total instancias: {len(df_out)}")
    print(f"  - CFAR detections: {n_cfar}")
    print(f"  - Fallback ruido (1 ventana): {n_fallback}")
    print(f"  - Drones sin CFAR (descartados): {n_skipped}")
    print(f"Splits: Train={len(df_out[df_out['split']=='train'])} | "
          f"Val={len(df_out[df_out['split']=='val'])} | "
          f"Test={len(df_out[df_out['split']=='test'])}")
    print(f"Guardado en: {OUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
