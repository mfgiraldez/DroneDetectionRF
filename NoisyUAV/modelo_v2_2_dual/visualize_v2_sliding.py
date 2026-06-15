import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

import argparse
from tqdm import tqdm

# Config
GOLDEN_RESULTS = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\figuras_golden\golden_results_v2_1.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\checkpoints\best_model.pth"
FS = 14e6
WIN_LEN = 131072
N_STEPS = 16

def calculate_phys_features(win):
    try:
        _, _, H_smooth, _, nf_v, _, _, bursts = detectar_bursts(
            win, fs=FS, nperseg=2048, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        z_peak    = max([abs(b['z_peak']) for b in bursts]) if bursts else 0.0
        return [global_nf, global_H, min(z_peak, 30.0)]
    except: return [0.0, 0.0, 0.0]

@torch.no_grad()
def visualize(target=None, snr=None, forced_filename=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if forced_filename:
        filename = forced_filename
        # Intentar sacar target/snr del nombre si no vienen
        target = target if target is not None else 4
        snr = snr if snr is not None else 0
        prob_max_val = 0.0
        predicted_val = -1
    else:
        # 1. Buscar fichero en el Golden Set
        df = pd.read_csv(GOLDEN_RESULTS)
        matches = df[(df['target_multiclass'] == target) & (df['snr'] == snr)]
        
        if matches.empty:
            print(f"Error: No se encontró ningún fichero con Target {target} y SNR {snr} en el Golden Set.")
            return

        sample = matches.iloc[0]
        filename = sample['filename']
        prob_max_val = sample['prob_max']
        predicted_val = sample['predicted']

    print(f"Visualizando: {filename}")

    # 2. Cargar modelo y datos
    model = DualStreamCVCNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    fpath = os.path.join(DATA_DIR, filename)
    d = torch.load(fpath, map_location='cpu', weights_only=False)
    iq_full = d['x_iq'].float()
    iq_complex = (iq_full[0] + 1j * iq_full[1]).numpy()

    # 3. Sliding Window
    max_idx = iq_full.shape[1]
    step = (max_idx - WIN_LEN) // (N_STEPS - 1)
    probs = []
    times = []

    for i in tqdm(range(N_STEPS), desc="Procesando ventanas"):
        start = i * step
        end = start + WIN_LEN
        win = iq_full[:, start:end]
        
        # Inferencia
        phys = torch.tensor([calculate_phys_features(win)], dtype=torch.float32).to(device)
        power = win.pow(2).mean().sqrt()
        win_norm = (win / power).unsqueeze(0).to(device)
        
        logits, _ = model(win_norm, phys)
        prob = torch.sigmoid(logits).item()
        probs.append(prob)
        times.append((start + WIN_LEN/2) / FS * 1000)

    # 4. PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Espectrograma
    f, t, Sxx = spectrogram(iq_complex, fs=FS, nperseg=1024, noverlap=512, return_onesided=False)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10*np.log10(Sxx+1e-12)
    ax1.pcolormesh(t*1000, f/1e6, Sxx_db, shading='gouraud', cmap='viridis', vmin=np.percentile(Sxx_db, 5), vmax=np.percentile(Sxx_db, 95))
    ax1.set_ylabel("Frecuencia (MHz)", fontsize=12)
    ax1.set_title(f"Diagnóstico V2.1 (Golden Set) | Fichero: {filename}\nTarget: {target} | SNR: {snr}dB | Pred: {predicted_val}", fontsize=14, fontweight='bold')

    # Probabilidad
    ax2.plot(times, probs, marker='o', color='#0077B6', linewidth=3, markersize=8, label="Salida Sigmoid V2.1")
    ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Umbral Decisión")
    ax2.set_xlabel("Tiempo (ms)", fontsize=12)
    ax2.set_ylabel("Probabilidad", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out_name = f"figuras_eval_visual/diagnose_v2_T{target}_SNR{snr}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Diagnóstico guardado en: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza la detección V2.1 sobre el Golden Set.")
    parser.add_argument("--target", type=int, help="ID de la clase (0-6)")
    parser.add_argument("--snr", type=int, help="SNR en dB")
    parser.add_argument("--filename", type=str, help="Nombre del fichero específico")
    args = parser.parse_args()
    
    if args.filename:
        # Buscar en el CSV para obtener target y snr originales
        df = pd.read_csv(GOLDEN_RESULTS)
        row = df[df['filename'] == args.filename]
        
        t, s = None, None
        if not row.empty:
            t = row.iloc[0]['target_multiclass']
            s = row.iloc[0]['snr']
            
        visualize(target=t, snr=s, forced_filename=args.filename)
    elif args.target is not None and args.snr is not None:
        visualize(target=args.target, snr=args.snr)
    elif args.target is not None and args.snr is not None:
        visualize(args.target, args.snr)
    else:
        print("Error: Debes proporcionar (--target y --snr) o (--filename).")
