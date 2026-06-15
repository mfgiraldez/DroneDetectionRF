import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import argparse
from tqdm import tqdm
from pathlib import Path

# Rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Config
GOLDEN_RESULTS = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\checkpoints\best_model.pth"
OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\figuras_visualizador_v2_filtrado"

FS = 14e6
WIN_LEN = 131072
N_STEPS = 16

def calculate_phys_features(iq_full):
    try:
        _, _, H_smooth, _, nf_v, _, _, bursts = detectar_bursts(
            iq_full, fs=FS, nperseg=2048, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        z_peak    = max([abs(b['z_peak']) for b in bursts]) if bursts else 0.0
        return [global_nf, global_H, min(z_peak, 30.0)]
    except: return [0.0, 0.0, 0.0]

@torch.no_grad()
def visualize(target=None, snr=None, forced_filename=None, threshold=0.75, min_consecutive=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Buscar fichero y metadatos
    df = pd.read_csv(GOLDEN_RESULTS)
    if forced_filename:
        filename = forced_filename
        row = df[df['filename'] == filename]
        if not row.empty:
            target = row.iloc[0]['target']
            snr = row.iloc[0]['snr']
        else:
            target, snr = target or "Unknown", snr or "Unknown"
    else:
        matches = df[(df['target'] == target) & (df['snr'] == snr)]
        if matches.empty:
            print(f"Error: No se encontró Target {target} SNR {snr}")
            return
        filename = matches.sample(1).iloc[0]['filename']

    print(f"Visualizando V2 (Física Global): {filename}")

    # 2. Cargar modelo y datos
    model = DualStreamCVCNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    fpath = os.path.join(DATA_DIR, filename)
    d = torch.load(fpath, map_location='cpu', weights_only=False)
    iq_full = d['x_iq'].float()
    iq_complex = (iq_full[0] + 1j * iq_full[1]).numpy()

    # 3. [CLAVE] Calcular Features Físicas GLOBALES (sobre los 75ms)
    # Esto es lo que el modelo espera según su diseño Dual-Stream
    phys_vals = calculate_phys_features(iq_full)
    phys_tensor = torch.tensor([phys_vals], dtype=torch.float32).to(device)

    # 4. Sliding Window (Inferencia CNN local + Física Global)
    max_idx = iq_full.shape[1]
    step = (max_idx - WIN_LEN) // (N_STEPS - 1)
    probs = []
    times = []
    window_spans = []

    for i in tqdm(range(N_STEPS), desc="Procesando ventanas"):
        start = i * step
        end = start + WIN_LEN
        win = iq_full[:, start:end]
        
        # Normalización RMS local
        power = win.pow(2).mean().clamp(min=1e-12).sqrt()
        win_norm = (win / power).unsqueeze(0).to(device)
        
        # Inferencia (CNN ve el trozo, pero phys_tensor es el contexto global)
        logits, _ = model(win_norm, phys_tensor)
        prob = torch.sigmoid(logits).item()
        probs.append(prob)
        times.append((start + WIN_LEN/2) / FS * 1000)
        window_spans.append((start / FS * 1000, (start + WIN_LEN) / FS * 1000))

    # LÓGICA DE FILTRADO TEMPORAL
    consecutive_count = 0
    confirmed_detection = False
    detection_zones = []
    
    for i in range(len(probs)):
        if probs[i] >= threshold:
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                confirmed_detection = True
                z_start = times[max(0, i - min_consecutive + 1)]
                z_end = times[i]
                detection_zones.append((z_start, z_end))
        else:
            consecutive_count = 0

    # PLOT (Estructura de 2 paneles elegante y minimalista - Opción 2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    f, t, Sxx = spectrogram(iq_complex, fs=FS, nperseg=1024, noverlap=512, return_onesided=False)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10*np.log10(Sxx+1e-12)
    ax1.pcolormesh(t*1000, f/1e6, Sxx_db, shading='gouraud', cmap='viridis', 
                   vmin=np.percentile(Sxx_db, 5), vmax=np.percentile(Sxx_db, 95))
    ax1.set_ylabel("Frecuencia (MHz)", fontsize=12)
    
    status_str = "DRON DETECTADO" if confirmed_detection else "RUIDO AISLADO (SIN DETECCIÓN)"
    color_status = "darkgreen" if confirmed_detection else "darkred"
    
    # Título principal hiper-profesional
    titulo = (f"Arquitectura Dual-Stream V2.1 - Evaluación \n"
              f"Archivo: {filename} \n"
              f"Clase real: {target} - SNR: {snr} dB - Detección: {status_str}")
    ax1.set_title(titulo, fontsize=13, fontweight='bold', color=color_status)

    # ================= PANEL 2: CURVA DE PROBABILIDAD CON SEGMENTOS DE VENTANA =================
    # Línea de tendencia muy sutil de fondo
    ax2.plot(times, probs, color='#BDC3C7', linestyle=':', alpha=0.5, zorder=1)
    
    # Dibujar cada ventana como un segmento horizontal de 9.4 ms a la altura de su probabilidad
    for idx, (w_start, w_end) in enumerate(window_spans):
        prob = probs[idx]
        
        # Color azul para detecciones por encima del umbral, gris para las de abajo
        w_color = '#0077B6' if prob >= threshold else '#7F8C8D'
        w_alpha = 0.85 if prob >= threshold else 0.35
        w_lw = 3.0 if prob >= threshold else 1.5
        
        # Segmento de ventana temporal horizontal
        ax2.plot([w_start, w_end], [prob, prob], color=w_color, alpha=w_alpha, linewidth=w_lw, zorder=2)
        
        # Punto en el centro de la ventana
        ax2.plot([times[idx]], [prob], marker='o', color=w_color, alpha=w_alpha, markersize=4, zorder=3)
        
        # Valor de probabilidad para picos sutilmente marcados
        if prob >= threshold:
            ax2.text(times[idx], prob + 0.04, f"{prob:.2f}", color='#0077B6', fontsize=8, 
                     ha='center', va='bottom', fontweight='bold', zorder=4)

    ax2.axhline(threshold, color='orange', linestyle='--', linewidth=2, label=f"Umbral Óptimo Operativo ({threshold})")
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label="Umbral Base Teórico (0.5)")
    ax2.set_xlabel("Evolución Temporal (ms)", fontsize=12)
    ax2.set_ylabel("Probabilidad", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # ================= SUPERPOSICIÓN DE LA ZONA DE DETECCIÓN CONFIRMADA (VERDE) =================
    for z_s, z_e in detection_zones:
        w_half = (WIN_LEN / FS * 1000) / 2
        # Sombrear la zona real abarcada por las ventanas en el espectrograma
        ax1.axvspan(z_s - w_half, z_e + w_half, color='green', alpha=0.12)
        # Sombrear en la probabilidad
        ax2.axvspan(z_s, z_e, color='green', alpha=0.15)

    plt.tight_layout()
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    out_name = os.path.join(OUT_DIR, f"diagnose_v2_filt_T{target}_SNR{snr}.png")
    plt.savefig(out_name, dpi=150)
    print(f"Diagnóstico V2 (Global) guardado en: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza la detección V2.1 con lógica de post-procesado.")
    parser.add_argument("--target", type=int, help="ID de la clase (0-6)")
    parser.add_argument("--snr", type=int, help="SNR en dB")
    parser.add_argument("--filename", type=str, help="Nombre del fichero")
    parser.add_argument("--threshold", type=float, default=0.75, help="Umbral de decisión")
    parser.add_argument("--min_consecutive", type=int, default=2)
    args = parser.parse_args()
    
    visualize(target=args.target, snr=args.snr, forced_filename=args.filename, 
              threshold=args.threshold, min_consecutive=args.min_consecutive)
