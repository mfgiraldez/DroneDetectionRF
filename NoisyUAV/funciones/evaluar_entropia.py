"""
Evaluación masiva del Detector de Entropía Espectral de Shannon.
Calcula Pd y Pfa por nivel de SNR sobre una muestra representativa del dataset.

Ejecución: python evaluar_entropia.py
"""

import os
import glob
import re
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR         = r"C:\TFM_data\NoisyUAV\drone_RF_data"
PLOTS_DIR        = r"C:\TFM_data\plots"
NOISE_CLASS      = 4
FS               = 14e6       # Frecuencia de muestreo
NPERSEG          = 1024       # Resolución temporal del STFT (~73 µs por bin)
FILES_PER_SNR    = 30         # Archivos evaluados por nivel de SNR
SEED             = 42

# Umbral: cuántas desviaciones típicas por debajo de la mediana se dispara el trigger
# Valores más altos → más sensible (Pd↑, Pfa↑), valores más bajos → más conservador
THRESHOLD_SIGMA  = 1.0

_PATTERN = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTOR DE ENTROPÍA (extraído del notebook)
# ─────────────────────────────────────────────────────────────────────────────

def entropia_espectral(iq_tensor: torch.Tensor,
                       fs: float = FS,
                       nperseg: int = NPERSEG):
    """
    Calcula la Entropía Espectral de Shannon frame a frame.
    Retorna (tiempo_ms, entropia) arrays.
    """
    signal = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
    _, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, return_onesided=False)
    Pxx = np.abs(Zxx) ** 2
    suma = Pxx.sum(axis=0)
    suma[suma == 0] = 1e-12
    prob = Pxx / suma + 1e-12
    entropia = -(prob * np.log2(prob)).sum(axis=0)
    return t * 1000, entropia


def detectar(iq_tensor: torch.Tensor, sigma: float = THRESHOLD_SIGMA) -> bool:
    """
    Devuelve True si se detecta al menos una ráfaga de transmisión.
    Umbral adaptativo: mediana - sigma * std(entropia)
    """
    _, entropia = entropia_espectral(iq_tensor)
    umbral = np.median(entropia) - sigma * np.std(entropia)
    return bool(np.any(entropia < umbral))


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUACIÓN MASIVA
# ─────────────────────────────────────────────────────────────────────────────

def cargar_iq(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x_iq"].float()


def evaluar(files_per_snr: int = FILES_PER_SNR, sigma: float = THRESHOLD_SIGMA):
    # Catalogar archivos por (target, snr)
    all_files = glob.glob(os.path.join(DATA_DIR, "IQdata_*.pt"))
    by_snr_drone = defaultdict(list)
    by_snr_noise = defaultdict(list)

    for f in all_files:
        m = _PATTERN.match(Path(f).name)
        if not m:
            continue
        target = int(m.group(2))
        snr    = int(m.group(3))
        if target == NOISE_CLASS:
            by_snr_noise[snr].append(f)
        else:
            by_snr_drone[snr].append(f)

    snr_levels = sorted(set(by_snr_drone) | set(by_snr_noise))
    print(f"  Niveles de SNR encontrados : {len(snr_levels)}")
    print(f"  Archivos por nivel (máx)   : {files_per_snr}")
    print(f"  Sigma del umbral           : {sigma}")
    print()
    print(f"  {'SNR':>6}  {'N_dr':>5}  {'N_ru':>5}  {'Pd':>7}  {'Pfa':>7}  {'Acc':>7}")
    print(f"  {'─'*46}")

    results = {}
    total_tp = total_fn = total_fp = total_tn = 0

    for snr in snr_levels:
        drone_files = random.sample(by_snr_drone[snr],
                                    min(files_per_snr, len(by_snr_drone[snr])))
        noise_files = random.sample(by_snr_noise[snr],
                                    min(files_per_snr, len(by_snr_noise[snr])))

        tp = fn = fp = tn = 0
        for f in drone_files:
            pred = detectar(cargar_iq(f), sigma)
            if pred: tp += 1
            else:    fn += 1

        for f in noise_files:
            pred = detectar(cargar_iq(f), sigma)
            if pred: fp += 1
            else:    tn += 1

        total_tp += tp; total_fn += fn
        total_fp += fp; total_tn += tn

        n_dr   = tp + fn
        n_ru   = fp + tn
        pd     = tp / n_dr if n_dr > 0 else float("nan")
        pfa    = fp / n_ru if n_ru > 0 else float("nan")
        acc    = (tp + tn) / (n_dr + n_ru) if (n_dr + n_ru) > 0 else float("nan")
        results[snr] = {"Pd": pd, "Pfa": pfa, "accuracy": acc,
                        "n_drone": n_dr, "n_noise": n_ru}

        bar = "█" * int(pd * 15) if not np.isnan(pd) else ""
        print(f"  {snr:+4d} dB  {n_dr:5d}  {n_ru:5d}  {pd:7.4f}  {pfa:7.4f}  {acc:7.4f}  {bar}")

    # Resumen global
    pd_global  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    pfa_global = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    acc_global = (total_tp + total_tn) / (total_tp + total_fn + total_fp + total_tn)
    print(f"  {'─'*46}")
    print(f"  {'GLOBAL':>6}  {'':>5}  {'':>5}  {pd_global:7.4f}  {pfa_global:7.4f}  {acc_global:7.4f}")

    return results, pd_global, pfa_global


# ─────────────────────────────────────────────────────────────────────────────
#  GRÁFICA
# ─────────────────────────────────────────────────────────────────────────────

def plot_resultados(results: dict, sigma: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    snrs = sorted(results.keys())
    pd   = [results[s]["Pd"]       for s in snrs]
    pfa  = [results[s]["Pfa"]      for s in snrs]
    acc  = [results[s]["accuracy"] for s in snrs]

    plt.figure(figsize=(13, 5))
    plt.plot(snrs, pd,  "b-o",  markersize=5, label="Pd  (Prob. Detección)")
    plt.plot(snrs, pfa, "r-s",  markersize=5, label="Pfa (Prob. Falsa Alarma)")
    plt.plot(snrs, acc, "g--^", markersize=5, label="Accuracy")
    plt.axvline(x=-12, color="purple", linestyle="--", alpha=0.8,
                label="Límite SOTA Glüge (-12 dB)")
    plt.axhline(y=0.9,  color="blue", linestyle=":", alpha=0.4, label="Pd objetivo = 0.9")
    plt.axhline(y=0.1,  color="red",  linestyle=":", alpha=0.4, label="Pfa objetivo = 0.1")
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Probabilidad", fontsize=12)
    plt.title(f"Detector Entropía Shannon — σ={sigma}  |  Pd y Pfa vs SNR", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    out = os.path.join(out_dir, f"entropia_pd_pfa_sigma{sigma}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Gráfica guardada: {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Evaluación: Detector de Entropía Espectral Shannon")
    print("=" * 55)
    print()

    results, pd_g, pfa_g = evaluar(FILES_PER_SNR, THRESHOLD_SIGMA)
    plot_resultados(results, THRESHOLD_SIGMA, PLOTS_DIR)

    print(f"\n  Pd global : {pd_g:.4f}")
    print(f"  Pfa global: {pfa_g:.4f}")

    # Verificar el objetivo -12 dB
    r12 = results.get(-12, {})
    pd12  = r12.get("Pd",  float("nan"))
    pfa12 = r12.get("Pfa", float("nan"))
    print(f"\n  En -12 dB → Pd={pd12:.4f}  Pfa={pfa12:.4f}", end="  ")
    if pd12 >= 0.9 and pfa12 <= 0.1:
        print("✓ OBJETIVO CONSEGUIDO")
    else:
        print(f"  (objetivo: Pd≥0.9, Pfa≤0.1)")


if __name__ == "__main__":
    main()
