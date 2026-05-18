import os, sys, glob, re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Configuración
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
GOLDEN_CSV = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
OUT_CSV = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\dataset_v4_train_val.csv"

FS = 14e6
NPERSEG = 2048
Z_THRESH = 3.5 # Un poco más estricto para asegurar pureza a SNR >= 8dB
TARGET_NOISE = 4
WINDOW_MS = 9.4
WINDOW_SAMPLES = int((WINDOW_MS / 1000.0) * FS)

def main():
    print("=== GENERADOR DATASET V4: HARD NEGATIVE MINING & INTERFERENCE EXTR === ")
    
    # 1. Cargar Golden Set para excluirlo
    golden_df = pd.read_csv(GOLDEN_CSV)
    test_filenames = set(golden_df['filename'].tolist())
    
    pattern = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")
    archivos = glob.glob(os.path.join(DATA_DIR, "IQdata_*.pt"))
    
    # Filtrar archivos que no estén en el test set
    archivos_train_val = [f for f in archivos if os.path.basename(f) not in test_filenames]
    print(f"Archivos disponibles para Train/Val: {len(archivos_train_val)}")
    
    records = []
    
    for fpath in tqdm(archivos_train_val, desc="Procesando ficheros"):
        fname = os.path.basename(fpath)
        m = pattern.match(fname)
        if not m: continue
        
        target = int(m.group(2))
        snr = int(m.group(3))
        label = 0 if target == TARGET_NOISE else 1
        
        # Cargar IQ
        try:
            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
            L_samples = iq.shape[1]
            L_ms = (L_samples / FS) * 1000
        except: continue
        
        # Extraer ráfagas
        t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
            iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
            min_burst_ms=0.4, merge_gap_ms=0.5, min_z_abs=3.0,
            bg_mult=4, max_bins_frac=1.0, smooth_ms=0.2)
            
        global_nf = float(np.median(nf_v))
        global_H_mean = float(np.mean(H_smooth))
        
        burst_intervals = []
        
        # A. CASO DRON (SNR >= 8dB)
        if label == 1 and snr >= 8:
            if len(bursts) > 0:
                for b in bursts:
                    t_center = (b['t0'] + b['t1']) / 2.0
                    records.append({
                        'filename': fname, 'label': 1, 'type': 'drone_real',
                        'target': target, 'snr': snr, 't_center': t_center,
                        'z_peak': abs(b['z_peak']), 'n_bins': b.get('n_act', 0),
                        'global_nf': global_nf, 'global_H_mean': global_H_mean
                    })
                    burst_intervals.append((b['t0'], b['t1']))
                
                # Extraer 1 VENTANA DE SILENCIO (donde no hay bursts)
                # Buscamos huecos de al menos WINDOW_MS
                gaps = []
                last_t = 0
                for t0, t1 in sorted(burst_intervals):
                    if t0 - last_t > WINDOW_MS:
                        gaps.append((last_t, t0))
                    last_t = t1
                if L_ms - last_t > WINDOW_MS:
                    gaps.append((last_t, L_ms))
                
                if gaps:
                    g, g_end = random.choice(gaps)
                    t_silence = g + WINDOW_MS/2.0
                    records.append({
                        'filename': fname, 'label': 0, 'type': 'silence_in_drone',
                        'target': target, 'snr': snr, 't_center': t_silence,
                        'z_peak': 0.0, 'n_bins': 0,
                        'global_nf': global_nf, 'global_H_mean': global_H_mean
                    })
        
        # B. CASO RUIDO (Cualquier SNR)
        elif label == 0:
            if len(bursts) > 0:
                # Interferencias (WiFi/BT)
                for b in bursts:
                    t_center = (b['t0'] + b['t1']) / 2.0
                    records.append({
                        'filename': fname, 'label': 0, 'type': 'interference',
                        'target': target, 'snr': snr, 't_center': t_center,
                        'z_peak': abs(b['z_peak']), 'n_bins': b.get('n_act', 0),
                        'global_nf': global_nf, 'global_H_mean': global_H_mean
                    })
            else:
                # Ruido puro (fallback centrado)
                records.append({
                    'filename': fname, 'label': 0, 'type': 'noise_fallback',
                    'target': target, 'snr': snr, 't_center': L_ms / 2.0,
                    'z_peak': 0.0, 'n_bins': 0,
                    'global_nf': global_nf, 'global_H_mean': global_H_mean
                })

    df = pd.DataFrame(records)
    
    # Equilibrar: Queremos que drone_real no esté enterrado
    # Hacemos un split por archivo para evitar leakage antes de guardar
    unique_files = df['filename'].unique()
    random.seed(42)
    random.shuffle(unique_files)
    train_files = unique_files[:int(len(unique_files)*0.85)]
    df['split'] = df['filename'].apply(lambda x: 'train' if x in train_files else 'val')
    
    df.to_csv(OUT_CSV, index=False)
    print(f"Dataset V4 guardado: {OUT_CSV}")
    print(df['type'].value_counts())

if __name__ == "__main__":
    main()
