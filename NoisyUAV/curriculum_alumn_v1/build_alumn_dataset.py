"""
build_alumn_dataset.py
======================
Genera el dataset pseudo-etiquetado para el modelo Alumno usando el
Oráculo V9 validado:
  - Para ficheros SNR >= 0: purificación con Z-ratio 80% (igual que el Teacher)
  - Para ficheros SNR < 0:
      * Dron  → Campeón elegido por proximidad de duración a la referencia
                calibrada (drone_duration_ref.json) + desempate por prob_ia.
                El resto se clasifica por plantilla ±TOL_DUR respecto al campeón.
      * Ruido → todos los bursts etiquetados como 0 directamente.

Salida: curriculum_alumn_v1/alumn_dataset_pseudo.csv
"""
import sys, os, json, itertools
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.funciones.detector_entropia import detectar_bursts
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as TeacherCVCNN
from NoisyUAV.funciones import TARGET_NOISE

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── CONFIGURACION ─────────────────────────────────────────────────────────────
TEACHER_CKPT  = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1\checkpoints\teacher_model_best.pt")
DURATION_JSON = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\drone_duration_ref.json")
OUTPUT_DIR    = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "alumn_dataset_pseudo.csv"

# Parámetros del detector validados en el notebook para SNR bajo
FS = 14e6; NPERSEG = 2048; Z_THRESH = 4.0
MIN_BURST_MS = 0.5; MERGE_GAP_MS = 1.5; MIN_Z_ABS = 4.0
BG_MULT = 4; MAX_BINS_FRAC = 0.25; SMOOTH_MS = 0.2; ADAPTIVE_WINDOW_MS = 15

TOL_DUR = 0.20   # ±20% tolerancia de duración FHSS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def get_suelo_ms(h_seg, u_seg, dt):
    drop_max = np.max(u_seg - h_seg) + 1e-8
    mask = h_seg < (u_seg - 0.7 * drop_max)
    if not np.any(mask): return 0.0
    return max(len(list(g)) for k, g in itertools.groupby(mask) if k) * dt


def process_file(file_path, target_mc, snr, label, split_name,
                 teacher_model, phys_mean, phys_std, duration_ref):
    records = []

    try:
        d  = torch.load(file_path, map_location='cpu', weights_only=False)
        iq = d['x_iq'].float()
    except Exception:
        return []

    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
        iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    if len(bursts) == 0:
        return []

    global_nf      = float(np.median(nf_v))
    global_ns      = float(np.clip(ns, 0, 5))
    global_H_mean  = float(np.mean(H_smooth))
    global_p75_act = float(np.percentile(n_active, 75))
    dt             = float(t_ms[1] - t_ms[0])
    filename       = Path(file_path).name

    # ── PASE 1: Inferencia IA en todos los bursts ─────────────────────────────
    teacher_model.eval()
    with torch.no_grad():
        for b in bursts:
            h_seg = H_smooth[b['i0']:b['i1']+1]
            u_seg = umbral_v[b['i0']:b['i1']+1]
            b['suelo_ms']  = get_suelo_ms(h_seg, u_seg, dt)
            b['rugosidad'] = float(np.std(np.diff(h_seg))) if len(h_seg) > 1 else 0.5

            idx_i = int(b['t0']*1e-3*FS)
            idx_f = int(b['t1']*1e-3*FS)
            p_raw  = iq[:, idx_i:idx_f]
            p_norm = p_raw / p_raw.pow(2).mean().clamp(min=1e-12).sqrt()
            T_LEN  = int(9.4e-3*FS)
            p_pad  = (torch.cat([p_norm, torch.zeros(2, T_LEN-p_norm.shape[1])], dim=1)
                      if p_norm.shape[1] < T_LEN else p_norm[:, :T_LEN])

            feat = torch.from_numpy(np.array([
                np.clip(b['dur_ms'], 0, 75),
                np.clip(abs(b['z_peak']), 0, 30),
                np.clip(b['drop_b'], 0, 10),
                np.clip(b['n_act'], 0, 2048),
                global_nf, global_ns, global_H_mean, global_p75_act
            ], dtype=np.float32)).to(DEVICE)
            feat_norm = torch.clamp((feat - phys_mean) / (phys_std + 1e-8), -5., 5.).unsqueeze(0)

            b['prob_ia'] = torch.sigmoid(
                teacher_model(p_pad.unsqueeze(0).to(DEVICE), feat_norm)
            ).item()

    def make_record(b, b_idx, pseudo_label, is_pseudo):
        return {
            'file_path':       filename,
            'split':           split_name,
            'label':           label,
            'target_multiclass': target_mc,
            'snr':             snr,
            'burst_id':        b_idx + 1,
            't_start':         b['t0'],
            't_end':           b['t1'],
            'dur_ms':          np.clip(b['dur_ms'], 0, 75),
            'z_peak':          np.clip(abs(b['z_peak']), 0, 30),
            'drop_b':          np.clip(b['drop_b'], 0, 10),
            'n_act_burst':     np.clip(b['n_act'], 0, 2048),
            'global_nf':       global_nf,
            'global_ns':       global_ns,
            'global_H_mean':   global_H_mean,
            'global_p75_act':  global_p75_act,
            'prob_ia':         b['prob_ia'],
            'pseudo_label':    pseudo_label,
            'is_pseudo':       is_pseudo,
        }

    # ── ALTA SNR (>= 0dB): purificación Z-ratio 80% igual que el Teacher ─────
    if snr >= 0:
        max_z_abs = max(abs(b['z_peak']) for b in bursts)
        umbral_pur = max_z_abs * 0.80
        valid = [b for b in bursts if abs(b['z_peak']) >= umbral_pur]
        for b_idx, b in enumerate(valid):
            records.append(make_record(b, b_idx, label, False))
        return records

    # ── BAJA SNR (< 0dB): Oráculo V9 ─────────────────────────────────────────
    if label == 0:
        # Ruido puro: cualquier burst detectado → etiqueta 0
        for b_idx, b in enumerate(bursts):
            records.append(make_record(b, b_idx, 0, True))
        return records

    # Dron a baja SNR: Champion-First + plantilla de duración FHSS
    target_key = str(target_mc)
    if target_key in duration_ref:
        dur_ref_mean = duration_ref[target_key]['dur_ref']
        # Pase 2: Campeón = burst más cercano a la referencia calibrada.
        # Desempate por mayor prob_ia.
        campeon = min(
            bursts,
            key=lambda b: (
                abs(b['dur_ms'] - dur_ref_mean) / (dur_ref_mean + 1e-8),
                -b['prob_ia']
            )
        )
    else:
        # Sin referencia: campeón por mayor prob_ia (V9 original)
        campeon = max(bursts, key=lambda b: b['prob_ia'])

    ref_dur = campeon['dur_ms']

    # Pase 3: clasificar por plantilla ±TOL_DUR respecto al campeón
    for b_idx, b in enumerate(bursts):
        desv = abs(b['dur_ms'] - ref_dur) / (ref_dur + 1e-8)
        if b is campeon or desv <= TOL_DUR:
            records.append(make_record(b, b_idx, 1, True))
        else:
            records.append(make_record(b, b_idx, 0, True))

    return records


def main():
    print("=" * 60)
    print("  BUILD ALUMN DATASET — Oráculo V9 Validado")
    print("=" * 60)

    if not TEACHER_CKPT.exists():
        print(f"❌ No se encuentra el checkpoint Teacher: {TEACHER_CKPT}")
        return

    # Cargar referencia de duraciones
    duration_ref = {}
    if DURATION_JSON.exists():
        with open(DURATION_JSON) as f:
            duration_ref = json.load(f)
        print(f"✅ Referencia de duraciones cargada: targets {list(duration_ref.keys())}")
    else:
        print("⚠️  drone_duration_ref.json no encontrado. Oráculo sin restricción de duración.")
        print("   Ejecuta primero: python calibrate_drone_durations.py")

    # Cargar Teacher
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    teacher_model = TeacherCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    teacher_model.load_state_dict(ckpt['model_state'])
    teacher_model.eval()
    phys_mean = torch.from_numpy(ckpt['phys_mean']).to(DEVICE)
    phys_std  = torch.from_numpy(ckpt['phys_std']).to(DEVICE)
    print(f"✅ Teacher cargado (Val F1={ckpt.get('val_f1', '?'):.4f}) — Device: {DEVICE}\n")

    df_train, df_val, df_test = obtener_splits_dataset()
    all_records = []

    for split_name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        print(f"Procesando split '{split_name}' ({len(df)} ficheros)...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            file_path = Path(row['filepath'])
            res = process_file(
                file_path, row['target_multiclass'], row['snr'], row['label'], split_name,
                teacher_model, phys_mean, phys_std, duration_ref
            )
            all_records.extend(res)

    final_df = pd.DataFrame(all_records)
    final_df.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Dataset Alumno guardado en: {OUT_CSV}")
    print(f"   Total ráfagas: {len(final_df)}")
    print(f"\nDistribución pseudo_label:")
    print(final_df['pseudo_label'].value_counts())
    print(f"\nDistribución SNR:")
    print(final_df.groupby('snr')['pseudo_label'].value_counts().unstack(fill_value=0))


if __name__ == '__main__':
    main()
