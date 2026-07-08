import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl

# Rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Configuración
GOLDEN_RESULTS = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\checkpoints\best_model.pth"
BASE_OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas"

FS = 14e6
WIN_LEN = 131072
N_STEPS = 16

# Configuración de estilo profesional para TFM
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 16

def calculate_phys_features_local(iq_tensor):
    """ Calcula las features físicas localmente (por ventana) para V2.2 """
    try:
        _, _, H_smooth, _, nf_v, ns, _, bursts = detectar_bursts(
            iq_tensor, fs=FS, nperseg=2048,
            adaptive_window_ms=0.0, min_burst_ms=0.3, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        
        if bursts:
            z_peak = max([abs(b['z_peak']) for b in bursts])
        else:
            z_peak = (np.min(H_smooth) - global_nf) / (ns + 1e-10)
            
        z_peak = float(np.clip(abs(z_peak), 0, 30))
        return [global_nf, global_H, z_peak]
    except Exception:
        return [0.0, 0.0, 0.0]

@torch.no_grad()
def generate_figure(filename, mode_name, threshold, min_consecutive=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Buscar fichero y metadatos
    df = pd.read_csv(GOLDEN_RESULTS)
    row = df[df['filename'] == filename]
    if not row.empty:
        target = row.iloc[0]['target']
        snr = row.iloc[0]['snr']
    else:
        print(f"Advertencia: {filename} no encontrado en el ground truth.")
        target = "Unknown"
        snr = "Unknown"

    print(f"Procesando: {filename} en modo {mode_name}")

    # 2. Cargar modelo
    model = DualStreamCVCNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    fpath = os.path.join(DATA_DIR, filename)
    d = torch.load(fpath, map_location='cpu', weights_only=False)
    iq_full = d['x_iq'].float()
    iq_complex = (iq_full[0] + 1j * iq_full[1]).numpy()

    # 3. Sliding Window (Inferencia local V2.2)
    max_idx = iq_full.shape[1]
    step = (max_idx - WIN_LEN) // (N_STEPS - 1)
    probs = []
    times = []
    window_spans = []

    for i in tqdm(range(N_STEPS), desc="Procesando ventanas V2.2"):
        start = i * step
        end = start + WIN_LEN
        win = iq_full[:, start:end]
        
        # Features Locales V2.2
        phys_vals = calculate_phys_features_local(win)
        phys_tensor = torch.tensor([phys_vals], dtype=torch.float32).to(device)

        # Normalización RMS local
        power = win.pow(2).mean().clamp(min=1e-12).sqrt()
        win_norm = (win / power).unsqueeze(0).to(device)
        
        # Inferencia
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

    # PLOT
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 2.5, 1], hspace=0.1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    
    # Eje de tiempo para señal en el dominio del tiempo
    t_ms = np.linspace(0, iq_full.shape[1]/FS*1000, iq_full.shape[1])
    
    # Panel 0: Señal IQ en el tiempo
    # Se usa rasterized=True para evitar un PDF excesivamente pesado con 1 Millón de puntos
    ax0.plot(t_ms, iq_full[0].numpy(), color='#3498DB', alpha=0.7, linewidth=0.5, label='In-Phase (I)', rasterized=True)
    ax0.plot(t_ms, iq_full[1].numpy(), color='#E67E22', alpha=0.7, linewidth=0.5, label='Quadrature (Q)', rasterized=True)
    ax0.set_ylabel("Amplitud", fontweight='semibold')
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Espectrograma
    f, t, Sxx = spectrogram(iq_complex, fs=FS, nperseg=1024, noverlap=512, return_onesided=False)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10*np.log10(Sxx+1e-12)
    im = ax1.pcolormesh(t*1000, f/1e6, Sxx_db, shading='gouraud', cmap='viridis', 
                   vmin=np.percentile(Sxx_db, 5), vmax=np.percentile(Sxx_db, 95), rasterized=True)
    
    ax1.set_ylabel("Frecuencia (MHz)", fontweight='semibold')
    # Ocultar ticks X en el panel superior
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.grid(False) # Quitar grid al espectrograma
    
    status_str = "DRON DETECTADO" if confirmed_detection else "RUIDO AISLADO (SIN DETECCIÓN)"
    color_status = "#27AE60" if confirmed_detection else "#E74C3C"
    
    # Título elegante
    titulo = (f"Modelo: Dual-Stream V2.2 (CFAR Local) | Modo: {mode_name}\n"
              f"Archivo: {filename} | SNR: {snr} dB\n"
              f"Detección: {status_str}")
    ax0.set_title(titulo, fontsize=14, fontweight='bold', color='#2C3E50', pad=15)

    # Añadir un discreto borde a la imagen del espectrograma
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('#BDC3C7')
        spine.set_linewidth(1.5)
        
    for spine in ax0.spines.values():
        spine.set_visible(True)
        spine.set_color('#BDC3C7')
        spine.set_linewidth(1.5)

    # Panel Inferior: Probabilidades
    ax2.plot(times, probs, color='#95A5A6', linestyle='--', alpha=0.6, zorder=1)
    
    for idx, (w_start, w_end) in enumerate(window_spans):
        prob = probs[idx]
        w_color = '#2980B9' if prob >= threshold else '#7F8C8D'
        w_alpha = 0.9 if prob >= threshold else 0.4
        w_lw = 3.5 if prob >= threshold else 2.0
        
        # Segmento horizontal
        ax2.plot([w_start, w_end], [prob, prob], color=w_color, alpha=w_alpha, linewidth=w_lw, zorder=2)
        # Punto central
        ax2.plot([times[idx]], [prob], marker='o', color=w_color, alpha=w_alpha, markersize=5, zorder=3)
        
        # Texto del valor
        if prob >= threshold:
            ax2.text(times[idx], prob + 0.05, f"{prob:.2f}", color='#2980B9', fontsize=9, 
                     ha='center', va='bottom', fontweight='bold', zorder=4)
                     
        # Colorear ventanas en la señal temporal superior
        if prob >= threshold:
            ax0.axvspan(w_start, w_end, color='#AED6F1', alpha=0.3, zorder=0)
            
        # Dibujar bordes verticales de la ventana para que se vea exactamente dónde empieza y acaba
        ax0.axvline(w_start, color='#7F8C8D', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)
        ax0.axvline(w_end, color='#7F8C8D', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)

    # Líneas de umbral
    ax2.axhline(threshold, color='#E67E22', linestyle='-', linewidth=2, label=f"Umbral de Decisión ({threshold:.2f})")
    if threshold != 0.5:
        ax2.axhline(0.5, color='#95A5A6', linestyle=':', linewidth=1.5, alpha=0.7, label="Umbral Teórico (0.50)")
    
    # Estilizado del panel inferior
    ax2.set_xlabel("Evolución Temporal (ms)", fontweight='semibold')
    ax2.set_ylabel("Probabilidad", fontweight='semibold')
    ax2.set_ylim(-0.05, 1.1)
    
    # Zonas verdes de confirmación
    for z_s, z_e in detection_zones:
        w_half = (WIN_LEN / FS * 1000) / 2
        ax1.axvspan(z_s - w_half, z_e + w_half, color='#2ECC71', alpha=0.15)
        ax2.axvspan(z_s, z_e, color='#2ECC71', alpha=0.15)

    ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, edgecolor='#BDC3C7')
    
    # Ajustes finales de layout
    out_dir_mode = os.path.join(BASE_OUT_DIR, f"DualStream_V2_2_{mode_name.replace(' ', '_')}")
    Path(out_dir_mode).mkdir(parents=True, exist_ok=True)
    out_name = os.path.join(out_dir_mode, f"eval_v2_2_{filename.replace('.pt', '')}.pdf")
    
    plt.savefig(out_name, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Figura guardada: {out_name}")

if __name__ == "__main__":
    df_golden = pd.read_csv(GOLDEN_RESULTS)
    
    # Seleccionar un pool representativo
    # Targets representativos (ej. DJI=0, Futaba=1, Noise=4, Turnigy=6)
    # Configuraciones específicas
    tasks = [
        ("IQdata_sample6428_target5_snr-14.pt", "Alerta Temprana", 0.50)
    ]
    
    print(f"Iniciando generacion de {len(tasks)} archivos...")
    
    for f, mode_name, threshold in tasks:
        generate_figure(f, mode_name, threshold)
