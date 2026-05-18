"""
create_dataset_mil_v4.py
=========================
Generador de dataset puro para MIL-Alumn-Xin. 
Elimina los hacks heurísticos (como la plantilla de duración a baja SNR)
y confía plenamente en la capacidad del Multiple Instance Learning (MIL) 
para encontrar el dron cuando el CFAR falla.

Lógica simplificada y robusta:
1. Usa un Z_THRESH estricto constante (4.0) para asegurar que lo que recorta es real.
2. Si encuentra ráfagas: las recorta y guarda su prob_ia.
3. Si NO encuentra ráfagas: guarda un Fallback de 75ms completo para el MIL.
"""
import sys, os, glob, re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as TeacherCVCNN

DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TARGET_NOISE = 4
TEACHER_CKPT  = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_teacher_v1\checkpoints\teacher_model_best.pt")
OUT_CSV       = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_MIL_alumn_xin\alumn_dataset_pseudo_v4.csv")
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FS = 14e6; NPERSEG = 2048
MIN_BURST_MS = 0.5; MERGE_GAP_MS = 1.5; MIN_Z_ABS = 3.5
BG_MULT = 4; MAX_BINS_FRAC = 0.40; SMOOTH_MS = 0.2; ADAPTIVE_WINDOW_MS = 15
Z_THRESH = 4.0  # Fijo y estricto. Si no lo supera, es trabajo para el MIL.

def obtener_splits_dataset(data_dir=DATA_DIR, test_size=0.15, val_size=0.15, random_state=42):
    pattern  = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")
    archivos = glob.glob(os.path.join(data_dir, "IQdata_*.pt"))
    data = []
    for f in archivos:
        m = pattern.match(os.path.basename(f))
        if m:
            target = int(m.group(2)); snr = int(m.group(3))
            data.append({
                'filepath': f, 'target_multiclass': target,
                'label': 0 if target == TARGET_NOISE else 1,
                'snr': snr,
                'stratify_key': f"{'0' if target == TARGET_NOISE else '1'}_{snr}",
            })
    df = pd.DataFrame(data)
    df_temp, df_test  = train_test_split(df, test_size=test_size,
                                          random_state=random_state, stratify=df['stratify_key'])
    df_train, df_val  = train_test_split(df_temp, test_size=val_size/(1-test_size),
                                          random_state=random_state, stratify=df_temp['stratify_key'])
    return df_train, df_val, df_test

def run_teacher(iq, b, global_nf, global_ns, global_H_mean, global_p75_act,
                teacher_model, phys_mean, phys_std):
    idx_i  = int(b['t0'] * 1e-3 * FS)
    idx_f  = int(b['t1'] * 1e-3 * FS)
    p_raw  = iq[:, idx_i:idx_f]
    p_norm = p_raw / p_raw.pow(2).mean().clamp(min=1e-12).sqrt()
    T_LEN  = int(9.4e-3 * FS)
    p_pad  = (torch.cat([p_norm, torch.zeros(2, T_LEN - p_norm.shape[1])], dim=1)
              if p_norm.shape[1] < T_LEN else p_norm[:, :T_LEN])

    feat = torch.from_numpy(np.array([
        np.clip(b['dur_ms'], 0, 75),
        np.clip(abs(b['z_peak']), 0, 30),
        np.clip(b['drop_b'], 0, 10),
        np.clip(b['n_act'], 0, 2048),
        global_nf, global_ns, global_H_mean, global_p75_act,
    ], dtype=np.float32)).to(DEVICE)
    feat_norm = torch.clamp((feat - phys_mean) / (phys_std + 1e-8), -5., 5.).unsqueeze(0)

    return torch.sigmoid(teacher_model(p_pad.unsqueeze(0).to(DEVICE), feat_norm)).item()

def process_file(file_path, target_mc, snr, label, split_name, teacher_model, phys_mean, phys_std):
    records = []
    filename = Path(file_path).name
    try:
        d  = torch.load(file_path, map_location='cpu', weights_only=False)
        iq = d['x_iq'].float()
    except Exception: return []

    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
        iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    global_nf      = float(np.median(nf_v))
    global_ns      = float(np.clip(ns, 0, 5))
    global_H_mean  = float(np.mean(H_smooth))
    global_p75_act = float(np.percentile(n_active, 75))

    # CASO 1: CFAR Ciego -> FALLBACK MIL
    if len(bursts) == 0:
        dur_total_ms = iq.shape[1] / FS * 1000.0
        records.append({
            'file_path': filename, 'split': split_name, 'label': label,
            'target_multiclass': target_mc, 'snr': snr, 'burst_id': 0,
            't_start': 0.0, 't_end': dur_total_ms, 'dur_ms': dur_total_ms,
            'z_peak': 0.0, 'drop_b': 0.0, 'n_act_burst': 0.0,
            'global_nf': global_nf, 'global_ns': global_ns,
            'global_H_mean': global_H_mean, 'global_p75_act': global_p75_act,
            'prob_ia': 1.0, # MIL confía en la etiqueta del fichero
            'pseudo_label': label,
            'is_pseudo': False, 'fallback': True,
        })
        return records

    # CASO 2: CFAR detecta -> Extraer ráfagas limpias
    for b_idx, b in enumerate(bursts):
        prob = run_teacher(iq, b, global_nf, global_ns, global_H_mean, global_p75_act, teacher_model, phys_mean, phys_std)
        # Asignar pseudo-label según la probabilidad del Teacher (0.5 threshold)
        pseudo_label = 1 if prob >= 0.5 else 0
        
        # En caso de ruido (label == 0), no queremos que el modelo aprenda a buscar drones.
        if label == 0: pseudo_label = 0 
        
        records.append({
            'file_path': filename, 'split': split_name, 'label': label,
            'target_multiclass': target_mc, 'snr': snr, 'burst_id': b_idx + 1,
            't_start': b['t0'], 't_end': b['t1'],
            'dur_ms': np.clip(b['dur_ms'], 0, 75),
            'z_peak': np.clip(abs(b['z_peak']), 0, 30),
            'drop_b': np.clip(b['drop_b'], 0, 10),
            'n_act_burst': np.clip(b['n_act'], 0, 2048),
            'global_nf': global_nf, 'global_ns': global_ns,
            'global_H_mean': global_H_mean, 'global_p75_act': global_p75_act,
            'prob_ia': prob, 'pseudo_label': pseudo_label,
            'is_pseudo': True, 'fallback': False,
        })
    return records

def main():
    print("=" * 65)
    print("  BUILD DATASET V4 (MIL PURE) - Cero Hacks")
    print("=" * 65)

    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    teacher_model = TeacherCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    teacher_model.load_state_dict(ckpt['model_state'])
    teacher_model.eval()
    phys_mean = torch.from_numpy(ckpt['phys_mean']).to(DEVICE)
    phys_std  = torch.from_numpy(ckpt['phys_std']).to(DEVICE)
    
    df_train, df_val, df_test = obtener_splits_dataset()
    all_records = []
    
    for split_name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        print(f"Procesando split '{split_name}'...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            res = process_file(
                row['filepath'], row['target_multiclass'], row['snr'],
                row['label'], split_name, teacher_model, phys_mean, phys_std
            )
            all_records.extend(res)

    out_df = pd.DataFrame(all_records)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nDataset guardado en {OUT_CSV}")
    print(f"Total muestras: {len(out_df)} | Fallbacks (MIL): {out_df['fallback'].sum()}")

if __name__ == "__main__":
    main()
