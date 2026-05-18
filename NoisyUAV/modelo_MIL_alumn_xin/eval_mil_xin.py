import os, sys, torch, random
import pandas as pd, numpy as np
import plotly.graph_objects as go

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dataset.cargador import cargar_muestra
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts, plot_muestra
from NoisyUAV.modelo_MIL_alumn_xin.model import MILAlumnXinCVCNN
from NoisyUAV.modelo_MIL_alumn_xin.dataset_mil import batched_burst_iq_to_xin_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CARGAR CSV V4 Y ESTADÍSTICAS
csv_path = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_MIL_alumn_xin\alumn_dataset_pseudo_v4.csv"
df = pd.read_csv(csv_path)
df_train = df[df['split'] == 'train']
feats = df_train[['dur_ms', 'z_peak', 'drop_b', 'n_act_burst', 'global_nf', 'global_ns', 'global_H_mean', 'global_p75_act']].values
phys_mean = torch.tensor(feats.mean(axis=0), dtype=torch.float32).to(device)
phys_std  = torch.tensor(feats.std(axis=0), dtype=torch.float32).to(device)

# 2. CARGAR MODELO MIL
ruta_pesos = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_MIL_alumn_xin\checkpoints\best_model.pt"
model = MILAlumnXinCVCNN().to(device)
model.load_state_dict(torch.load(ruta_pesos, map_location=device, weights_only=False))
model.eval()

# ===== PARÁMETROS =====
TARGET_DESEADO = 3     
SNR_DESEADA    = -16   
RUTA_RAW_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
FS = 14e6; NPERSEG = 2048; Z_THRESH = 4.0      

# Elegir muestra
filtro = (df['target_multiclass'] == TARGET_DESEADO) & (df['snr'] == SNR_DESEADA) & (df['split'] == 'test')
df_cand = df[filtro]
if df_cand.empty: raise ValueError("Sin ráfagas de TEST.")
filename = random.choice(df_cand['file_path'].unique())
iq_tensor, _, _, _ = cargar_muestra(os.path.join(RUTA_RAW_DIR, filename))

# Detección CFAR estricta
t_ms, H, H_sm, umb, nf, ns, n_act, bursts = detectar_bursts(
    iq_tensor, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
    min_burst_ms=0.5, merge_gap_ms=0.75, min_z_abs=3.5, 
    bg_mult=4, max_bins_frac=1.0, smooth_ms=0.3, adaptive_window_ms=10)

fig = plot_muestra(iq_tensor, t_ms, H, H_sm, umb, nf, ns, n_act, bursts, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH, bg_mult=4, max_bins_frac=1.0, adaptive_window_ms=10, titulo=f"MIL TEST: {filename}")
fig.show()

print("="*50 + "\n  INFERENCIA MIL-ALUMN-XIN\n" + "="*50)
global_nf = np.median(nf)
global_ns_val = np.clip(ns, 0, 5)
global_H_mean = np.mean(H_sm)
global_p75 = np.percentile(n_act, 75)

with torch.no_grad():
    if len(bursts) == 0:
        print("  ⚠️ CFAR Ciego. Iniciando escaneo MIL en 16 ventanas...")
        L = iq_tensor.shape[1]
        TARGET_LEN = 131072
        step = TARGET_LEN // 2
        
        windows = []
        for start in range(0, L - TARGET_LEN + 1, step):
            windows.append(iq_tensor[:, start:start+TARGET_LEN])
        
        w_tensor = torch.stack(windows, dim=0) # [16, 2, L]
        spec = batched_burst_iq_to_xin_tensor(w_tensor).to(device) # [16, 2, 256, 256]
        
        feat_arr = np.array([75.0, 0.0, 0.0, 0.0, global_nf, global_ns_val, global_H_mean, global_p75], dtype=np.float32)
        f_t = torch.from_numpy(feat_arr).to(device)
        f_norm = torch.clamp((f_t - phys_mean) / (phys_std + 1e-8), -5., 5.).unsqueeze(0).repeat(spec.shape[0], 1)
        
        logits = model(spec, f_norm)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy() * 100
        
        max_idx = np.argmax(probs)
        print(f"  🧠 Ventana ganadora MIL: #{max_idx+1} con {probs[max_idx]:.2f}% de probabilidad dron")
        for i, p in enumerate(probs):
            print(f"     Win {i+1:02d}: {p:5.2f}%")
    else:
        for i, b in enumerate(bursts):
            # Recorte normal...
            idx_inicio, idx_fin = int(b['t0'] * 1e-3 * FS), int(b['t1'] * 1e-3 * FS)
            pulso = iq_tensor[:, idx_inicio:idx_fin]
            p_norm = pulso / pulso.pow(2).mean().clamp(min=1e-12).sqrt()
            p_pad = torch.cat([p_norm, torch.zeros(2, 131072 - p_norm.shape[1])], dim=1)[:, :131072] if p_norm.shape[1] < 131072 else p_norm[:, :131072]
            
            spec = batched_burst_iq_to_xin_tensor(p_pad.unsqueeze(0)).to(device)
            feat_arr = np.array([b['dur_ms'], abs(b['z_peak']), b['drop_b'], b['n_act'], global_nf, global_ns_val, global_H_mean, global_p75], dtype=np.float32)
            f_norm = torch.clamp((torch.from_numpy(feat_arr).to(device) - phys_mean) / (phys_std + 1e-8), -5., 5.).unsqueeze(0)
            
            logit = model(spec, f_norm)
            prob = torch.sigmoid(logit).item() * 100
            print(f"  [B{i+1:02d}] t={b['t0']:6.2f} ms | IA: {prob:6.2f}%")
