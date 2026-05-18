"""
Generador Dataset V6 — Dual-Stream V3a (n_bins como feature adicional)
======================================================================
Enriquece el CSV V5 existente añadiendo la columna n_bins_peak a cada
instancia (1 fila = 1 detección CFAR). Los splits se preservan del CSV V5
original para garantizar comparabilidad directa con el modelo V2b.

Cambio respecto a V5:
  + n_bins_peak : número máximo de bins activos durante el burst CFAR.
                  Discriminador físico entre FHSS estrecho (~186 bins)
                  y WiFi broadband (~2048 bins).

Uso:
    conda activate IAIAVv3
    cd c:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_alumn_v3_dual_pro\\modelo_alumn_v3a
    python build_dataset_v6_pointers.py
"""
import os, sys, re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# ── Rutas ──────────────────────────────────────────────────────────────────────
V5_CSV   = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v3_dual_pro\dataset_v5_pointers.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_CSV  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v3_dual_pro\modelo_alumn_v3a\dataset_v6_pointers.csv"

# ── CFAR config (idéntico al builder V5) ──────────────────────────────────────
FS        = 14e6
NPERSEG   = 2048
Z_THRESH  = 4.0
T_TOL_MS  = 2.0  # tolerancia de matching de t_center (ms)

def main():
    print("=" * 60)
    print("  GENERADOR DATASET V6 (V5 + n_bins_peak)  ")
    print("=" * 60)

    df = pd.read_csv(V5_CSV)
    print(f"V5 cargado: {len(df)} instancias de {df['filename'].nunique()} ficheros")

    df['n_bins_peak'] = 0.0  # columna nueva, por defecto 0

    ficheros = df['filename'].unique()
    n_matched = 0
    n_fallback = 0
    n_error = 0

    for fname in tqdm(ficheros, desc="Añadiendo n_bins_peak"):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            n_error += 1
            continue

        # Índices del CSV para este fichero
        idx_file = df.index[df['filename'] == fname].tolist()

        try:
            d  = torch.load(fpath, map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
        except Exception:
            n_error += 1
            continue

        try:
            t_ms, _, _, _, _, _, n_active, bursts = detectar_bursts(
                iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
                min_burst_ms=0.5, merge_gap_ms=0.75, min_z_abs=3.5,
                bg_mult=4, max_bins_frac=1.0, smooth_ms=0.3,
                adaptive_window_ms=10
            )
        except Exception:
            n_error += 1
            continue

        if len(bursts) == 0:
            # Filas fallback → n_bins_peak = 0 (ya está por defecto)
            n_fallback += len(idx_file)
            continue

        # Construir lookup: t_center → n_bins_peak
        burst_lookup = {}
        for b in bursts:
            t_c = (b['t0'] + b['t1']) / 2.0
            mask = (t_ms >= b['t0']) & (t_ms <= b['t1'])
            nbins = float(np.max(n_active[mask])) if np.any(mask) else 0.0
            burst_lookup[t_c] = nbins

        burst_centers = sorted(burst_lookup.keys())

        # Asignar n_bins_peak a cada fila del fichero por nearest t_center
        for i in idx_file:
            t_row = float(df.loc[i, 't_center'])
            is_fb = bool(df.loc[i, 'is_fallback'])
            if is_fb or not burst_centers:
                df.loc[i, 'n_bins_peak'] = 0.0
                n_fallback += 1
            else:
                nearest = min(burst_centers, key=lambda t: abs(t - t_row))
                if abs(nearest - t_row) <= T_TOL_MS:
                    df.loc[i, 'n_bins_peak'] = burst_lookup[nearest]
                    n_matched += 1
                else:
                    df.loc[i, 'n_bins_peak'] = 0.0
                    n_fallback += 1

    # Guardar CSV V6
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Total instancias: {len(df)}")
    print(f"  - n_bins asignado por burst CFAR : {n_matched}")
    print(f"  - n_bins = 0 (fallback/sin burst) : {n_fallback}")
    print(f"  - Ficheros con error              : {n_error}")
    print(f"\nEstadísticas n_bins_peak:")
    print(df[df['n_bins_peak'] > 0]['n_bins_peak'].describe().round(1))
    print(f"\nGuardado en: {OUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
