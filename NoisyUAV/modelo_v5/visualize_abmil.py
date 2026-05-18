import os, sys, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

# Configuración de rutas
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v5.pipeline import (
    segment_file, BurstEncoder, GatedAttentionMIL, 
    OUTPUT_DIR, FS, IQ_INPUT_LEN, EMBED_DIM, PSD_N_BINS, BAG_MAX_INSTANCES
)

FILE_PATH = r"C:\TFM_data\NoisyUAV\drone_RF_data\IQdata_sample1604_target1_snr-10.pt"
MODEL_PATH = f"{OUTPUT_DIR}/abmil_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar Modelos
    encoder = BurstEncoder().to(device)
    mil = GatedAttentionMIL().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    mil.load_state_dict(checkpoint['mil'])
    encoder.eval(); mil.eval()
    
    # 2. Segmentar el fichero (Propuestas de ráfagas)
    # Nota: segment_file ahora guarda en disco y devuelve el path
    storage_path = segment_file({'path': FILE_PATH, 'class': 1, 'snr': -10})
    with open(storage_path, 'rb') as f:
        bursts = pickle.load(f)
    
    print(f"Propuestas de ráfagas encontradas: {len(bursts)}")
    
    # 3. Preparar entrada para el modelo
    iq_list, psd_list = [], []
    for b in bursts[:BAG_MAX_INSTANCES]:
        iq_t, psd_t = encoder.preprocess_burst(b['iq'])
        iq_list.append(iq_t)
        psd_list.append(psd_t)
        
    iq_batch = torch.stack(iq_list).to(device)
    psd_batch = torch.stack(psd_list).to(device)
    
    # 4. Inferencia y Atención
    with torch.no_grad():
        H = encoder(iq_batch, psd_batch) # [K, EMBED_DIM]
        logits, attn = mil(H.unsqueeze(0)) # [1, K]
        prob = torch.sigmoid(logits).item()
        attn_weights = attn.squeeze(0).cpu().numpy()
        
    print(f"Probabilidad de Dron: {prob:.4f}")
    
    # 5. Visualización
    # Cargar IQ original para el espectrograma
    d = torch.load(FILE_PATH, map_location='cpu', weights_only=False)
    iq_tensor = d['x_iq']
    iq_complex = (iq_tensor[0] + 1j * iq_tensor[1]).numpy()
    
    f, t, Sxx = spectrogram(iq_complex, fs=FS, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Espectrograma
    im = ax1.pcolormesh(t*1000, f/1e6, Sxx_db, shading='gouraud', cmap='viridis')
    ax1.set_ylabel("Frecuencia (MHz)")
    ax1.set_title(f"Inferencia V5: {Path(FILE_PATH).name} (SNR: -10 dB)\nProbabilidad Dron: {prob:.4f}")
    
    # Dibujar ráfagas y su atención
    for i, b in enumerate(bursts[:BAG_MAX_INSTANCES]):
        t0, t1 = b['t0'], b['t1']
        weight = attn_weights[i]
        alpha = 0.2 + 0.8 * (weight / (max(attn_weights) + 1e-8))
        color = 'red' if weight > (1.0/len(bursts)) else 'gray'
        
        ax1.axvspan(t0, t1, color=color, alpha=alpha * 0.3)
        # Etiqueta de atención
        ax2.bar((t0+t1)/2, weight, width=(t1-t0), color=color, alpha=0.7)

    ax2.set_xlabel("Tiempo (ms)")
    ax2.set_ylabel("Atención")
    ax2.set_ylim(0, 1.1 * max(attn_weights))
    
    plt.tight_layout()
    output_img = f"{OUTPUT_DIR}/results/inference_visualization.png"
    plt.savefig(output_img)
    print(f"Visualización guardada en: {output_img}")

if __name__ == "__main__":
    main()
