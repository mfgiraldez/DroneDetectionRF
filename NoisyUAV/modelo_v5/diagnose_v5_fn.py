import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

# Configuración de rutas para importar desde el proyecto
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v5.pipeline import (
    BurstEncoder, GatedAttentionMIL, discover_files,
    OUTPUT_DIR, DATA_ROOT, TEST_SPLIT_FILE, FS, IQ_INPUT_LEN, BAG_MAX_INSTANCES
)

def plot_diagnosis(filename, snr, true_class, prob, attn_weights, bursts, output_path):
    # Cargar IQ
    fpath = os.path.join(DATA_ROOT, filename)
    d = torch.load(fpath, map_location='cpu', weights_only=False)
    iq_tensor = d['x_iq']
    iq_complex = (iq_tensor[0] + 1j * iq_tensor[1]).numpy()
    
    # Espectrograma
    f, t, Sxx = spectrogram(iq_complex, fs=FS, nperseg=512, noverlap=256, return_onesided=False)
    
    # Reordenar para que las frecuencias negativas queden abajo
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Panel 1: Espectrograma + Bursts
    im = ax1.pcolormesh(t*1000, f/1e6, Sxx_db, shading='gouraud', cmap='viridis', vmin=Sxx_db.mean()-10, vmax=Sxx_db.max())
    ax1.set_ylabel("Frecuencia (MHz)")
    ax1.set_title(f"DIAGNÓSTICO FALSO NEGATIVO V5\nFichero: {filename} | SNR: {snr}dB | Clase: {true_class}\nProbabilidad Dron: {prob:.4f}")
    
    # Dibujar ráfagas
    for i, b in enumerate(bursts[:BAG_MAX_INSTANCES]):
        t0, t1 = b['t0'], b['t1']
        weight = attn_weights[i] if i < len(attn_weights) else 0
        # Normalizar color por peso
        color = 'red' if weight > (1.2/len(bursts)) else 'cyan'
        ax1.axvspan(t0, t1, color=color, alpha=0.2)
        ax1.text((t0+t1)/2, (f.max()/1e6)*0.9, f"{weight:.2f}", color='white', fontsize=8, ha='center', fontweight='bold')
        
        # Panel 2: Atencion
        ax2.bar((t0+t1)/2, weight, width=(t1-t0), color=color, alpha=0.7)

    ax2.set_xlabel("Tiempo (ms)")
    ax2.set_ylabel("Atención")
    ax2.set_ylim(0, 1.1 * max(attn_weights) if len(attn_weights)>0 else 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar Modelos
    model_path = f"{OUTPUT_DIR}/abmil_model.pt"
    encoder = BurstEncoder().to(device)
    mil = GatedAttentionMIL().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    mil.load_state_dict(checkpoint['mil'])
    encoder.eval(); mil.eval()
    
    # 2. Obtener lista de test
    _, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)
    
    # 3. Buscar Falsos Negativos (Drones a SNR >= 0 clasificados como ruido)
    print("Buscando 10 Falsos Negativos (SNR 0 a 10 dB)...")
    found_count = 0
    diag_dir = f"{OUTPUT_DIR}/results/diagnosis_fn"
    os.makedirs(diag_dir, exist_ok=True)
    
    from NoisyUAV.modelo_v5.pipeline import segment_file

    for fe in test_files:
        if fe['class'] == 4: continue # Solo drones
        if not (0 <= fe['snr'] <= 10): continue 
        
        fname = Path(fe['path']).name
        stem = Path(fe['path']).stem
        
        # Segmentar al vuelo para asegurar que tenemos t0, t1
        storage_path = segment_file(fe)
        with open(storage_path, 'rb') as f:
            bursts = pickle.load(f)
        
        if not bursts: continue
        
        # Preparar batch
        iq_list, psd_list = [], []
        for b in bursts[:BAG_MAX_INSTANCES]:
            iq_t, psd_t = encoder.preprocess_burst(b['iq'])
            iq_list.append(iq_t)
            psd_list.append(psd_t)
        
        iq_batch = torch.stack(iq_list).to(device)
        psd_batch = torch.stack(psd_list).to(device)
        
        with torch.no_grad():
            H = encoder(iq_batch, psd_batch)
            logits, attn = mil(H.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
            attn_weights = attn.squeeze(0).cpu().numpy()
            
        if prob < 0.2: # Falso Negativo claro
            found_count += 1
            out_img = f"{diag_dir}/FN_{found_count}_{stem}.png"
            print(f"[{found_count}/10] Generando diagnóstico para: {fname} (Prob: {prob:.4f})")
            plot_diagnosis(fname, fe['snr'], fe['class'], prob, attn_weights, bursts, out_img)
            
            if found_count >= 10: break

    print(f"\nDiagnóstico completado. Imágenes guardadas en: {diag_dir}")

if __name__ == "__main__":
    main()
