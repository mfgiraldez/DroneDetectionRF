import os, sys, torch, random, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dataset.cargador import cargar_muestra
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts, plot_muestra, print_diagnostico
from NoisyUAV.modelo_v4.model_v4 import DualStreamCVCNN_V4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ruta_pesos = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\checkpoints\best_model.pth"

model = DualStreamCVCNN_V4().to(device)
ckpt = torch.load(ruta_pesos, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

print(f"✅ Modelo Dual-Stream V4 cargado en {device.type.upper()}")
print(f"   Época: {ckpt.get('epoch','N/A')} | Val F1: {ckpt.get('val_f1',0.0):.4f}")

# ===== PARÁMETROS =====
TARGET_DESEADO = 1      # 0=DJI, 1=FutabaT14, 2=FutabaT7, 3=Graupner, 4=Ruido, 5=Taranis, 6=Turnigy
SNR_DESEADA    = -4
RUTA_RAW_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TARGET_NOISE   = 4      # Target 4 = ficheros de ruido puro
GOLDEN_CSV     = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
# ======================

df_golden = pd.read_csv(GOLDEN_CSV)

filtro = (df_golden['target'] == TARGET_DESEADO) & (df_golden['snr'] == SNR_DESEADA)
df_candidatos = df_golden[filtro]

if df_candidatos.empty:
    raise ValueError(f"❌ No existen ficheros en el GOLDEN SET para Target={TARGET_DESEADO} y SNR={SNR_DESEADA} dB.")

filename_random = random.choice(df_candidatos['filename'].unique())

ruta_completa = os.path.join(RUTA_RAW_DIR, filename_random)
print("=" * 60)
print(f"🎬 Muestra Golden Set: {filename_random}")
print(f"   Target: {TARGET_DESEADO} | SNR: {SNR_DESEADA} dB")
print("=" * 60)
iq_tensor, _, original_target, original_snr = cargar_muestra(ruta_completa)

# Parámetros IDÉNTICOS a build_dataset_v4.py
FS      = 14e6
NPERSEG = 2048
Z_THRESH = 3.5

t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
    iq_tensor, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
    min_burst_ms=0.4, merge_gap_ms=0.5, min_z_abs=3.0,
    bg_mult=4, max_bins_frac=1.0, smooth_ms=0.2, adaptive_window_ms=15,
)
print_diagnostico(
    t_ms, nf_v, ns, umbral_v, n_active, bursts,
    nperseg=NPERSEG, fs=FS, z_thresh=Z_THRESH,
    bg_mult=4, max_bins_frac=1.0, min_burst_ms=0.4, merge_gap_ms=0.5,
    target=f"Target {original_target}", snr=f"{original_snr}", index=0,
)
fig_2d = plot_muestra(
    iq_tensor, t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts,
    fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
    bg_mult=4, max_bins_frac=1.0, adaptive_window_ms=15,
    titulo=f"CFAR V4: {filename_random}"
)
fig_2d.show()

# Ground truth del fichero (determinista desde el nombre)
es_dron_real = (original_target != TARGET_NOISE)
gt_str = "DRON" if es_dron_real else "RUIDO"

# Parámetros
PASO_MS = 2.0
window_len = 131072
half_win   = window_len // 2
max_idx    = iq_tensor.shape[1]

global_nf     = float(np.median(nf_v))
global_H_mean = float(np.mean(H_smooth))

# --- FASE 1: Veredicto CFAR ---
print("=" * 60)
print(f"  VEREDICTO DUAL-STREAM V4 | GT: {gt_str} (Target {original_target}, {original_snr} dB)")
print("=" * 60)

centers, z_peaks, n_bins_list = [], [], []
is_blind = False

if len(bursts) > 0:
    for b in bursts:
        centers.append((b['t0'] + b['t1']) / 2.0)
        z_peaks.append(abs(b['z_peak']))
        n_bins_list.append(b.get('n_act', 0))
else:
    print("  CFAR ciego. Activando escaner ciego (cada 10ms)...")
    is_blind = True
    L_ms = (iq_tensor.shape[1] / FS) * 1000
    t_s = 5.0
    while t_s < L_ms - 5.0:
        centers.append(t_s); z_peaks.append(0.0); n_bins_list.append(0); t_s += 10.0

drones_cfar, ruidos_cfar = 0, 0

with torch.no_grad():
    for i, (t_c, z, n_b) in enumerate(zip(centers, z_peaks, n_bins_list)):
        c_idx = int((t_c / 1000.0) * FS)
        s = c_idx - half_win; e = c_idx + half_win
        if s < 0:         win = iq_tensor[:, 0:window_len]
        elif e > max_idx: win = iq_tensor[:, max_idx-window_len:max_idx]
        else:             win = iq_tensor[:, s:e]
        pwr = win.pow(2).mean().clamp(min=1e-12).sqrt()
        win = (win / pwr).unsqueeze(0).to(device)
        
        # FEATURE VECTOR V4: [global_nf, global_H_mean, z_peak, n_bins]
        feat = torch.tensor([[global_nf/10.0, global_H_mean/10.0, float(z)/50.0, float(n_b)/2048.0]], dtype=torch.float32).to(device)
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logit, attn = model(win, feat)
            prob = torch.sigmoid(logit).item() * 100
            aw   = attn[0].cpu().numpy() * 100
        etiq = "B" if not is_blind else "W"
        if prob > 50.0:
            drones_cfar += 1
            print(f"  [{etiq}{i+1:02d}] t={t_c:6.2f}ms | DRON   [{prob:5.1f}%] OK | IQ={aw[0]:.0f}% PSD={aw[1]:.0f}%")
        else:
            ruidos_cfar += 1
            print(f"  [{etiq}{i+1:02d}] t={t_c:6.2f}ms | RUIDO  [{prob:5.1f}%]    | IQ={aw[0]:.0f}% PSD={aw[1]:.0f}%")

veredicto_cfar = drones_cfar > 0

# --- FASE 2: Sliding Window ---
paso_idx = int((PASO_MS / 1000.0) * FS)
L_total  = iq_tensor.shape[1]
z_global = max(z_peaks) if z_peaks else 0.0
n_bins_global = max(n_bins_list) if n_bins_list else 0

t_sw, prob_sw, attn_iq_sw, attn_psd_sw = [], [], [], []

with torch.no_grad():
    for v in range((L_total - window_len) // paso_idx + 1):
        s_v = v * paso_idx; e_v = s_v + window_len
        if e_v > L_total: break
        win_v = iq_tensor[:, s_v:e_v].clone()
        pwr   = win_v.pow(2).mean().clamp(min=1e-12).sqrt()
        win_v = (win_v / pwr).unsqueeze(0).to(device)
        
        feat_v = torch.tensor([[global_nf/10.0, global_H_mean/10.0, z_global/50.0, n_bins_global/2048.0]], dtype=torch.float32).to(device)
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logit_v, attn_v = model(win_v, feat_v)
            p_v  = torch.sigmoid(logit_v).item() * 100
            aw_v = attn_v[0].cpu().numpy() * 100
        t_sw.append(((s_v + window_len / 2) / FS) * 1000)
        prob_sw.append(p_v); attn_iq_sw.append(aw_v[0]); attn_psd_sw.append(aw_v[1])

prob_arr = np.array(prob_sw)
t_arr    = np.array(t_sw)

# --- Métricas de consenso ---
# Umbral duro (>50%): para DRON confirmado
THRESH_HARD_AREA = 0.05   # >5% de ventanas
THRESH_HARD_RUN  = 2      # >=2 ventanas contiguas (>=4ms)

area_score = np.mean(prob_arr > 50)
run, max_run = 0, 0
for p in prob_arr:
    run = run + 1 if p > 50 else 0
    max_run = max(max_run, run)

# Umbral suave (>38%): para SOSPECHOSO — requiere también coherencia temporal
THRESH_SOFT_MAX = 38.0
THRESH_SOFT_RUN = 3       # >=3 ventanas contiguas (>=6ms) por encima del umbral suave

run_soft, max_run_soft = 0, 0
for p in prob_arr:
    run_soft = run_soft + 1 if p > THRESH_SOFT_MAX else 0
    max_run_soft = max(max_run_soft, run_soft)

p_max_sw  = float(np.max(prob_arr))
p_mean_sw = float(np.mean(prob_arr))

# --- FASE 3: Consenso con tres niveles ---
# DRON: CFAR + SW ambos confirman activación fuerte sostenida
if veredicto_cfar and (area_score >= THRESH_HARD_AREA) and (max_run >= THRESH_HARD_RUN):
    nivel    = "DRON"
    correcto = es_dron_real

# SOSPECHOSO: SW ve activación SOSTENIDA por encima del umbral suave
# (un único pico puntual no es suficiente: max_run_soft debe ser >=3)
elif max_run_soft >= THRESH_SOFT_RUN:
    nivel    = "SOSPECHOSO"
    correcto = es_dron_real   # correcto si hay dron real detrás

# RUIDO: sin activación sostenida en ningún umbral
else:
    nivel    = "RUIDO"
    correcto = not es_dron_real

icono = "CORRECTO" if correcto else "INCORRECTO"

print("-" * 60)
print(f"  CFAR solo:    {'DRON' if veredicto_cfar else 'RUIDO'} ({drones_cfar}/{drones_cfar+ruidos_cfar} positivos)")
print(f"  SW duro:      area={area_score:.2f} | racha>{50:.0f}%={max_run}v ({max_run*PASO_MS:.0f}ms)")
print(f"  SW suave:     racha>{THRESH_SOFT_MAX:.0f}%={max_run_soft}v ({max_run_soft*PASO_MS:.0f}ms) | P_max={p_max_sw:.1f}% | P_mean={p_mean_sw:.1f}%")
print(f"  CONSENSO:     [{nivel}]")
print(f"  Ground Truth: {gt_str} (Target {original_target})")
print(f"  RESULTADO:    {icono}")
print("=" * 60)

# --- Gráfica ---
nivel_to_color = {"DRON": "red", "SOSPECHOSO": "orange", "RUIDO": "green"}
title_color = nivel_to_color[nivel]
gt_label = "DRON" if es_dron_real else "RUIDO"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.fill_between(t_arr, prob_arr, alpha=0.3, color=title_color)
ax1.plot(t_arr, prob_arr, color=title_color, linewidth=1.5, label='P(Dron)')
ax1.axhline(50,             color='gray',   linestyle='--', alpha=0.7, linewidth=1.5,
            label=f'Umbral DRON (50%) — racha>={THRESH_HARD_RUN}v')
ax1.axhline(THRESH_SOFT_MAX, color='orange', linestyle=':', alpha=0.9, linewidth=1.5,
            label=f'Umbral SOSPECHOSO ({THRESH_SOFT_MAX:.0f}%) — racha>={THRESH_SOFT_RUN}v')
ax1.axhline(p_mean_sw,      color='purple', linestyle='-.', alpha=0.5, linewidth=1.0,
            label=f'P_media={p_mean_sw:.1f}%')
ax1.set_ylabel('Probabilidad Dron (%)')
ax1.set_ylim(0, 100)
ax1.set_title(
    f'Sliding Window V4 — {filename_random} | GT: {gt_label} | [{nivel}] [{icono}]',
    color=title_color, fontweight='bold'
)
ax1.legend(fontsize=8, loc='upper right'); ax1.grid(True, alpha=0.3)

# Zonas rojas: activacion dura (>50%)
above_hard = prob_arr > 50
if np.any(above_hard):
    ups   = np.where(np.diff(above_hard.astype(int)) == 1)[0]
    downs = np.where(np.diff(above_hard.astype(int)) == -1)[0]
    for su in ups:
        cands = downs[downs > su]
        if len(cands): ax1.axvspan(t_arr[su], t_arr[cands[0]], alpha=0.25, color='red')

# Zonas naranjas: activacion suave (>THRESH_SOFT_MAX pero <50%)
above_soft = (prob_arr > THRESH_SOFT_MAX) & ~above_hard
if np.any(above_soft):
    ups   = np.where(np.diff(above_soft.astype(int)) == 1)[0]
    downs = np.where(np.diff(above_soft.astype(int)) == -1)[0]
    for su in ups:
        cands = downs[downs > su]
        if len(cands): ax1.axvspan(t_arr[su], t_arr[cands[0]], alpha=0.12, color='orange')

ax2.plot(t_arr, attn_iq_sw,  color='blue',   linewidth=1.2, label='Rama IQ')
ax2.plot(t_arr, attn_psd_sw, color='orange', linewidth=1.2, label='Rama PSD')
ax2.set_ylabel('Peso Atencion (%)'); ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

print(f"\nSW Stats: {len(prob_arr)} ventanas | P_media={p_mean_sw:.1f}% | P_max={p_max_sw:.1f}% (t={t_arr[np.argmax(prob_arr)]:.1f}ms)")
print(f"   Racha >{50:.0f}%:          {max_run}v  = {max_run*PASO_MS:.0f}ms  [umbral DRON: >={THRESH_HARD_RUN}v]")
print(f"   Racha >{THRESH_SOFT_MAX:.0f}%: {max_run_soft}v  = {max_run_soft*PASO_MS:.0f}ms  [umbral SOSP: >={THRESH_SOFT_RUN}v]")