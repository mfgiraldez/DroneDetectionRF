"""
build_alumn_dataset_v2.py
=========================
Versión mejorada de build_alumn_dataset.py con dos correcciones para
eliminar la poda excesiva a SNR bajo que dejaba ficheros sin representación:

  MEJORA A — Z_THRESH adaptativo por grupo SNR:
      SNR >= 0  : Z_THRESH = 4.0  (igual que v1, pureza máxima)
      SNR < 0   : Z_THRESH = 2.5  (más permisivo, el Oráculo V9 filtra después)
      SNR <= -10: Z_THRESH = 1.8  (mínimo razonable para señales muy débiles)

  MEJORA B — Fallback de señal completa cuando bursts == 0:
      Si el detector no encuentra NINGÚN burst tras relajar el umbral,
      se incluye la muestra completa (t_start=0, t_end=75ms) con la
      etiqueta binaria real del fichero (sin pseudo-labeling, is_pseudo=False).
      Esto garantiza al menos 1 muestra por fichero en el dataset.

Salida: modelo_alumn_v1/alumn_dataset_pseudo_v2.csv
"""
import sys, os, json, itertools
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as TeacherCVCNN

# obtener_splits_dataset importado directamente del modulo para evitar
# dependencia del DATA_DIR del __init__.py (que esta comentado tras la refactorizacion)
import glob, re
from sklearn.model_selection import train_test_split

DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TARGET_NOISE = 4

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


os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── CONFIGURACION ─────────────────────────────────────────────────────────────
TEACHER_CKPT  = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_teacher_v1\checkpoints\teacher_model_best.pt")
DURATION_JSON = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\drone_duration_ref.json")
OUT_CSV       = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\alumn_dataset_pseudo_v2.csv")

# Parámetros del detector — los no-SNR-dependientes se fijan aqui
FS = 14e6; NPERSEG = 2048
MIN_BURST_MS = 0.5; MERGE_GAP_MS = 1.5; MIN_Z_ABS = 3.5
BG_MULT = 4; MAX_BINS_FRAC = 0.25; SMOOTH_MS = 0.2; ADAPTIVE_WINDOW_MS = 15

# MEJORA A: Z_THRESH adaptativo por grupo SNR
def get_z_thresh(snr: float) -> float:
    if snr <= -10:
        return 1.8   # Muy bajo SNR: máxima permisividad
    elif snr < 0:
        return 2.5   # Bajo SNR: permisivo (el Oráculo V9 filtra)
    else:
        return 4.0   # Alto SNR: estricto (igual que v1)

TOL_DUR = 0.20   # ±20% tolerancia de duración FHSS (igual que v1)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def get_suelo_ms(h_seg, u_seg, dt):
    drop_max = np.max(u_seg - h_seg) + 1e-8
    mask = h_seg < (u_seg - 0.7 * drop_max)
    if not np.any(mask): return 0.0
    return max(len(list(g)) for k, g in itertools.groupby(mask) if k) * dt


def run_teacher(iq, b, global_nf, global_ns, global_H_mean, global_p75_act,
                teacher_model, phys_mean, phys_std):
    """Calcula prob_ia del Teacher para un burst."""
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

    return torch.sigmoid(
        teacher_model(p_pad.unsqueeze(0).to(DEVICE), feat_norm)
    ).item()


def process_file(file_path, target_mc, snr, label, split_name,
                 teacher_model, phys_mean, phys_std, duration_ref):
    records = []
    filename = Path(file_path).name

    try:
        d  = torch.load(file_path, map_location='cpu', weights_only=False)
        iq = d['x_iq'].float()
    except Exception:
        return []

    # MEJORA A: umbral adaptativo
    z_thresh = get_z_thresh(snr)

    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
        iq, fs=FS, nperseg=NPERSEG, z_thresh=z_thresh,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    global_nf      = float(np.median(nf_v))
    global_ns      = float(np.clip(ns, 0, 5))
    global_H_mean  = float(np.mean(H_smooth))
    global_p75_act = float(np.percentile(n_active, 75))
    dt             = float(t_ms[1] - t_ms[0])

    # MEJORA B: fallback si bursts == 0
    if len(bursts) == 0:
        N = iq.shape[1]
        dur_total_ms = N / FS * 1000.0
        records.append({
            'file_path':         filename,
            'split':             split_name,
            'label':             label,
            'target_multiclass': target_mc,
            'snr':               snr,
            'burst_id':          0,
            't_start':           0.0,
            't_end':             dur_total_ms,
            'dur_ms':            dur_total_ms,
            'z_peak':            0.0,
            'drop_b':            0.0,
            'n_act_burst':       0.0,
            'global_nf':         global_nf,
            'global_ns':         global_ns,
            'global_H_mean':     global_H_mean,
            'global_p75_act':    global_p75_act,
            'prob_ia':           -1.0,   # -1 indica que es fallback (sin inferencia)
            'pseudo_label':      label,  # etiqueta real del fichero
            'is_pseudo':         False,  # NO es pseudo-etiquetado, es GT
            'fallback':          True,
        })
        return records

    def make_record(b, b_idx, pseudo_label, is_pseudo):
        return {
            'file_path':         filename,
            'split':             split_name,
            'label':             label,
            'target_multiclass': target_mc,
            'snr':               snr,
            'burst_id':          b_idx + 1,
            't_start':           b['t0'],
            't_end':             b['t1'],
            'dur_ms':            np.clip(b['dur_ms'], 0, 75),
            'z_peak':            np.clip(abs(b['z_peak']), 0, 30),
            'drop_b':            np.clip(b['drop_b'], 0, 10),
            'n_act_burst':       np.clip(b['n_act'], 0, 2048),
            'global_nf':         global_nf,
            'global_ns':         global_ns,
            'global_H_mean':     global_H_mean,
            'global_p75_act':    global_p75_act,
            'prob_ia':           b['prob_ia'],
            'pseudo_label':      pseudo_label,
            'is_pseudo':         is_pseudo,
            'fallback':          False,
        }

    # ── PASE 1: Inferencia IA en todos los bursts ─────────────────────────────
    teacher_model.eval()
    with torch.no_grad():
        for b in bursts:
            h_seg = H_smooth[b['i0']:b['i1'] + 1]
            u_seg = umbral_v[b['i0']:b['i1'] + 1]
            b['suelo_ms']  = get_suelo_ms(h_seg, u_seg, dt)
            b['rugosidad'] = float(np.std(np.diff(h_seg))) if len(h_seg) > 1 else 0.5
            b['prob_ia']   = run_teacher(iq, b, global_nf, global_ns, global_H_mean,
                                         global_p75_act, teacher_model, phys_mean, phys_std)

    # ── ALTA SNR (>= 0dB): purificación Z-ratio 80% ───────────────────────────
    if snr >= 0:
        max_z_abs  = max(abs(b['z_peak']) for b in bursts)
        umbral_pur = max_z_abs * 0.80
        valid = [b for b in bursts if abs(b['z_peak']) >= umbral_pur]
        for b_idx, b in enumerate(valid):
            records.append(make_record(b, b_idx, label, False))
        return records

    # ── BAJA SNR (< 0dB): Oráculo V9 ─────────────────────────────────────────
    if label == 0:
        for b_idx, b in enumerate(bursts):
            records.append(make_record(b, b_idx, 0, True))
        return records

    # Dron a baja SNR: Champion-First + plantilla de duración FHSS
    target_key = str(target_mc)
    if target_key in duration_ref:
        dur_ref_mean = duration_ref[target_key]['dur_ref']
        campeon = min(
            bursts,
            key=lambda b: (
                abs(b['dur_ms'] - dur_ref_mean) / (dur_ref_mean + 1e-8),
                -b['prob_ia']
            )
        )
    else:
        campeon = max(bursts, key=lambda b: b['prob_ia'])

    ref_dur = campeon['dur_ms']
    for b_idx, b in enumerate(bursts):
        desv = abs(b['dur_ms'] - ref_dur) / (ref_dur + 1e-8)
        if b is campeon or desv <= TOL_DUR:
            records.append(make_record(b, b_idx, 1, True))
        else:
            records.append(make_record(b, b_idx, 0, True))

    return records


def main():
    print("=" * 65)
    print("  BUILD ALUMN DATASET V2 - Z_THRESH adaptativo + Fallback")
    print("=" * 65)
    print("  Z_THRESH: SNR<=-10 -> 1.8 | SNR<0 -> 2.5 | SNR>=0 -> 4.0")
    print("  Fallback: ficheros sin bursts incluidos con etiqueta real")
    print("=" * 65)

    if not TEACHER_CKPT.exists():
        print(f"ERROR: No se encuentra el checkpoint Teacher: {TEACHER_CKPT}")
        return

    duration_ref = {}
    if DURATION_JSON.exists():
        with open(DURATION_JSON) as f:
            duration_ref = json.load(f)
        print(f"Referencia de duraciones: targets {list(duration_ref.keys())}")
    else:
        print("AVISO: drone_duration_ref.json no encontrado. Orculo sin restriccion de duracion.")

    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    teacher_model = TeacherCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    teacher_model.load_state_dict(ckpt['model_state'])
    teacher_model.eval()
    phys_mean = torch.from_numpy(ckpt['phys_mean']).to(DEVICE)
    phys_std  = torch.from_numpy(ckpt['phys_std']).to(DEVICE)
    print(f"Teacher cargado (Val F1={ckpt.get('val_f1', 0):.4f}) - Device: {DEVICE}\n")

    df_train, df_val, df_test = obtener_splits_dataset()
    all_records = []
    n_fallback  = 0
    n_empty_v1  = 0  # cuantos ficheros habrian quedado vacios con v1

    for split_name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        print(f"Procesando split '{split_name}' ({len(df)} ficheros)...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            res = process_file(
                row['filepath'], row['target_multiclass'], row['snr'],
                row['label'], split_name,
                teacher_model, phys_mean, phys_std, duration_ref
            )
            if len(res) == 1 and res[0].get('fallback'):
                n_fallback += 1
            if len(res) == 0:
                n_empty_v1 += 1
            all_records.extend(res)

    final_df = pd.DataFrame(all_records)
    final_df.to_csv(OUT_CSV, index=False)

    print(f"\nDataset V2 guardado: {OUT_CSV}")
    print(f"Total rafagas: {len(final_df):,}")
    print(f"  - Con fallback (sin bursts detectados): {n_fallback:,}")
    print(f"  - Ficheros que v1 habria descartado:    {n_empty_v1 + n_fallback:,}")

    print(f"\nDistribucion pseudo_label:")
    print(final_df['pseudo_label'].value_counts())

    print(f"\nDistribucion SNR x pseudo_label:")
    print(final_df.groupby('snr')['pseudo_label'].value_counts().unstack(fill_value=0).to_string())

    print(f"\nFallbacks por SNR (ficheros sin bursts detectados):")
    fb = final_df[final_df['fallback'] == True]
    if len(fb) > 0:
        print(fb.groupby(['snr', 'label'])['file_path'].count().to_string())
    else:
        print("  Ninguno - todos los ficheros tuvieron al menos 1 burst detectado.")


if __name__ == '__main__':
    main()
