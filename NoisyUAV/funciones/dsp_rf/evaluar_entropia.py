"""
evaluar_entropia.py
===================
Evaluación masiva del Detector de Entropía Espectral de Shannon sobre NoisyUAV.
Calcula Pd y Pfa por nivel de SNR y por clase de dron.

Ejecución:
    conda run -n IAIAVv3 python evaluar_entropia.py

Salida:
    - Tabla en consola: SNR × {Pd, Pfa, Accuracy}
    - Gráfica Pd/Pfa vs SNR guardada en PLOTS_DIR
    - CSV con resultados completos
"""

import os
import glob
import re
import random
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from detector_entropia import detectar_hay_senal

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR      = r"C:\TFM_data\NoisyUAV\drone_RF_data"
PLOTS_DIR     = r"C:\TFM_data\plots"
NOISE_CLASS   = 4          # target id de la clase "solo ruido"

# Parámetros del detector
FS            = 14e6
NPERSEG       = 4096
Z_THRESH      = 4.0
MIN_BURST_MS  = 0.3
MERGE_GAP_MS  = 2.0
MIN_Z_ABS     = 10.0

# Evaluación
FILES_PER_SNR = 50         # archivos por nivel de SNR (señal y ruido por separado)
SEED          = 42
# ══════════════════════════════════════════════════════════════════════════════

_PATTERN = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")
random.seed(SEED);  np.random.seed(SEED)


def _cargar_iq(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x_iq"].float()


def _catalogar(data_dir: str):
    """
    Cataloga todos los .pt del dataset en dos diccionarios:
      by_drone[snr]  = [path, ...]   → archivos con dron
      by_noise[snr]  = [path, ...]   → archivos sin dron (solo ruido)
    """
    all_files = glob.glob(os.path.join(data_dir, "IQdata_*.pt"))
    by_drone  = defaultdict(list)
    by_noise  = defaultdict(list)

    for f in all_files:
        m = _PATTERN.match(Path(f).name)
        if not m:
            continue
        target = int(m.group(2))
        snr    = int(m.group(3))
        if target == NOISE_CLASS:
            by_noise[snr].append(f)
        else:
            by_drone[snr].append(f)

    return by_drone, by_noise


def evaluar(
    files_per_snr: int   = FILES_PER_SNR,
    z_thresh: float      = Z_THRESH,
    min_burst_ms: float  = MIN_BURST_MS,
    merge_gap_ms: float  = MERGE_GAP_MS,
    min_z_abs: float     = MIN_Z_ABS,
    nperseg: int         = NPERSEG,
    verbose: bool        = True,
) -> pd.DataFrame:
    """
    Evaluación masiva: itera sobre todos los SNR del dataset.

    Returns
    -------
    df : pd.DataFrame  — columnas: snr, n_drone, n_noise, tp, fn, fp, tn, Pd, Pfa, Acc
    """
    by_drone, by_noise = _catalogar(DATA_DIR)
    snr_levels = sorted(set(by_drone) | set(by_noise))

    if verbose:
        print("=" * 70)
        print(f"  Detector Entropía Shannon  |  nperseg={nperseg}  z={z_thresh}  "
              f"minBurst={min_burst_ms}ms  minZ={min_z_abs}")
        print(f"  Dataset: {DATA_DIR}")
        print("=" * 70)
        print(f"  {'SNR':>6}  {'N_dr':>5}  {'N_ru':>5}  "
              f"{'Pd':>7}  {'Pfa':>7}  {'Acc':>7}")
        print(f"  {'─'*52}")

    rows = []
    for snr in snr_levels:
        drone_files = random.sample(by_drone[snr], min(files_per_snr, len(by_drone[snr])))
        noise_files = random.sample(by_noise[snr], min(files_per_snr, len(by_noise[snr])))

        tp = fn = fp = tn = 0

        for f in tqdm(drone_files, desc=f"SNR={snr:+d} dB [drone]", leave=False, disable=not verbose):
            iq   = _cargar_iq(f)
            pred = detectar_hay_senal(iq, fs=FS, nperseg=nperseg,
                                      z_thresh=z_thresh, min_burst_ms=min_burst_ms,
                                      merge_gap_ms=merge_gap_ms, min_z_abs=min_z_abs)
            if pred: tp += 1
            else:    fn += 1

        for f in tqdm(noise_files, desc=f"SNR={snr:+d} dB [noise]", leave=False, disable=not verbose):
            iq   = _cargar_iq(f)
            pred = detectar_hay_senal(iq, fs=FS, nperseg=nperseg,
                                      z_thresh=z_thresh, min_burst_ms=min_burst_ms,
                                      merge_gap_ms=merge_gap_ms, min_z_abs=min_z_abs)
            if pred: fp += 1
            else:    tn += 1

        n_dr = tp + fn
        n_ru = fp + tn
        Pd   = tp / n_dr if n_dr > 0 else float("nan")
        Pfa  = fp / n_ru if n_ru > 0 else float("nan")
        Acc  = (tp + tn) / (n_dr + n_ru) if (n_dr + n_ru) > 0 else float("nan")

        rows.append({"snr": snr, "n_drone": n_dr, "n_noise": n_ru,
                     "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                     "Pd": Pd, "Pfa": Pfa, "Acc": Acc})

        if verbose:
            bar = "█" * int(Pd * 15) if not np.isnan(Pd) else ""
            print(f"  {snr:+4d} dB  {n_dr:5d}  {n_ru:5d}  "
                  f"{Pd:7.4f}  {Pfa:7.4f}  {Acc:7.4f}  {bar}")

    df = pd.DataFrame(rows)

    if verbose:
        total_tp = df["tp"].sum(); total_fn = df["fn"].sum()
        total_fp = df["fp"].sum(); total_tn = df["tn"].sum()
        Pd_g  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        Pfa_g = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
        Acc_g = (total_tp + total_tn) / (total_tp + total_fn + total_fp + total_tn)
        print(f"  {'─'*52}")
        print(f"  {'GLOBAL':>6}  {'':>5}  {'':>5}  "
              f"{Pd_g:7.4f}  {Pfa_g:7.4f}  {Acc_g:7.4f}")
        print("=" * 70)

        # Objetivo en -12 dB
        r12 = df[df["snr"] == -12]
        if not r12.empty:
            pd12  = float(r12["Pd"].iloc[0])
            pfa12 = float(r12["Pfa"].iloc[0])
            ok    = pd12 >= 0.9 and pfa12 <= 0.1
            print(f"\n  En -12 dB → Pd={pd12:.4f}  Pfa={pfa12:.4f}  "
                  f"{'✓ OBJETIVO' if ok else '✗ objetivo: Pd≥0.9, Pfa≤0.1'}")

    return df


def plot_resultados(df: pd.DataFrame, out_dir: str, tag: str = ""):
    """
    Guarda la curva Pd/Pfa vs SNR en un PNG.

    Parameters
    ----------
    df      : DataFrame con columnas snr, Pd, Pfa, Acc
    out_dir : directorio de salida
    tag     : sufijo para el nombre del fichero
    """
    os.makedirs(out_dir, exist_ok=True)
    snrs = df["snr"].values
    Pd   = df["Pd"].values
    Pfa  = df["Pfa"].values
    Acc  = df["Acc"].values

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#e0e0e0')
    for sp in ax.spines.values(): sp.set_color('#444')

    ax.plot(snrs, Pd,  'o-',  color='#3498db', lw=2, ms=6, label='Pd  (Prob. Detección)')
    ax.plot(snrs, Pfa, 's--', color='#e74c3c', lw=2, ms=6, label='Pfa (Prob. Falsa Alarma)')
    ax.plot(snrs, Acc, '^:',  color='#2ecc71', lw=1.5, ms=5, label='Accuracy')
    ax.axvline(-12, color='#9b59b6', ls='--', lw=1.5, alpha=0.8, label='SOTA Glüge (-12 dB)')
    ax.axhline(0.9, color='#3498db',  ls=':', lw=1, alpha=0.4, label='Objetivo Pd=0.9')
    ax.axhline(0.1, color='#e74c3c',  ls=':', lw=1, alpha=0.4, label='Objetivo Pfa=0.1')

    ax.set_xlabel('SNR (dB)', color='white', fontsize=12)
    ax.set_ylabel('Probabilidad', color='white', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f'Detector Entropía Shannon  z={Z_THRESH}  nperseg={NPERSEG}  '
        f'minBurst={MIN_BURST_MS}ms  minZ={MIN_Z_ABS}',
        color='white', fontsize=12)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.grid(color='#2c3e6b', alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"entropia_pd_pfa{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica: {fname}")
    return fname


def main():
    df = evaluar()
    plot_resultados(df, PLOTS_DIR, tag=f"_z{Z_THRESH}_nperseg{NPERSEG}")

    # Guardar CSV
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = os.path.join(PLOTS_DIR, "entropia_resultados.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV:     {csv_path}")


if __name__ == "__main__":
    main()
