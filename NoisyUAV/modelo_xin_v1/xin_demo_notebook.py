# %% [markdown]
# # Xin CV-CNN 2D — Evaluación Manual por Muestra
#
# Notebook interactivo para inspeccionar visualmente:
# 1. La señal I/Q cruda en el dominio temporal
# 2. El espectrograma STFT (log-PSD) con las "cajitas" de los bursts FHSS
# 3. El mapa de bordes Sobel que detecta los contornos de las cajitas
# 4. El resultado de la inferencia del modelo
#
# Configura las celdas de CONFIGURACION y ejecuta secuencialmente.

# %% [markdown]
# ## Configuracion

# %%
# ==============================================================================
# PARAMETROS A CAMBIAR
# ==============================================================================

SPLIT   = "test"     # "train" | "val" | "test"
TARGET  = 1          # 0, 1, 2, 3, 5, 6 = drones  |  4 = ruido
SNR     = 10         # Nivel de SNR exacto en dB (ver valores disponibles abajo)
SEED    = 42         # None = aleatorio en cada ejecucion | int = reproducible

# Rutas (ajustar solo si cambiaste las rutas por defecto)
CKPT_PATH  = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_xin_v1\resultados\checkpoints\xin_model_best.pt"
CSV_PATH   = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_xin_v1\resultados\xin_splits.csv"
CACHE_DIR  = r"C:\TFM_data\NoisyUAV\xin_cache_256x256"   # None si no tienes cache
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"

TARGET_NAMES = {0: "Target 0", 1: "Target 1", 2: "Target 2",
                3: "Target 3", 4: "Ruido (T4)", 5: "Target 5", 6: "Target 6"}
FS_HZ    = 14_000_000   # Frecuencia de muestreo [Hz]
NFFT     = 1024
HOP      = 512

# ==============================================================================


# %% [markdown]
# ## Setup — Imports y carga del modelo

# %%
import sys, os, random
sys.path.insert(0, r"C:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import ndimage

from NoisyUAV.modelo_xin_v1.xin_cvcnn import XinCVCNN
from NoisyUAV.modelo_xin_v1.xin_dataset import (
    iq_to_xin_tensor, NFFT, HOP_LENGTH, SPEC_H, SPEC_W, DB_CLIP
)

# ── Dispositivo ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ── Modelo ───────────────────────────────────────────────────────────────────
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model = XinCVCNN(num_classes=1, kernel_size=5).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Modelo cargado — Epoch {ckpt['epoch']} | Best Val F1: {ckpt['best_val_f1']:.4f}")


# %% [markdown]
# ## Seleccion de muestra

# %%
df = pd.read_csv(CSV_PATH)

# Filtrar por split / target / snr
mask = (df["split"] == SPLIT) & (df["target"] == TARGET) & (df["snr"] == SNR)
subset = df[mask].reset_index(drop=True)

if len(subset) == 0:
    # Informar SNRs y targets disponibles para el split elegido
    print(f"[!] No hay muestras con SPLIT={SPLIT}, TARGET={TARGET}, SNR={SNR}")
    print(f"\nSNRs disponibles para TARGET={TARGET} en SPLIT={SPLIT}:")
    avail = df[(df["split"] == SPLIT) & (df["target"] == TARGET)]["snr"].unique()
    print(sorted(avail))
    print(f"\nTargets disponibles en SPLIT={SPLIT}, SNR={SNR}:")
    avail_t = df[(df["split"] == SPLIT) & (df["snr"] == SNR)]["target"].unique()
    print(sorted(avail_t))
    raise ValueError("Modifica TARGET / SNR / SPLIT en la celda de Configuracion.")

# Seleccion aleatoria (o reproducible con SEED)
rng = random.Random(SEED)
row = subset.iloc[rng.randint(0, len(subset) - 1)]

print(f"\nMuestra seleccionada:")
print(f"  Fichero : {row['filename']}")
print(f"  Split   : {row['split']}")
print(f"  Target  : {row['target']} ({TARGET_NAMES.get(row['target'], '?')})")
print(f"  SNR     : {row['snr']} dB")
print(f"  Is drone: {bool(row['is_drone'])}")

# Cargar IQ
raw = torch.load(row["filepath"], map_location="cpu", weights_only=False)
iq  = raw["x_iq"].clone()   # [2, N_SAMPLES]
N   = iq.shape[1]
t_axis_ms = np.linspace(0, N / FS_HZ * 1000, N)   # eje temporal en ms

print(f"\n  Muestras IQ : {N:,}  (~{N/FS_HZ*1000:.1f} ms)")
print(f"  RMS I       : {iq[0].pow(2).mean().sqrt():.4f}")
print(f"  RMS Q       : {iq[1].pow(2).mean().sqrt():.4f}")


# %% [markdown]
# ## Figura 1 — Señal I/Q en el dominio temporal

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
fig.suptitle(
    f"Señal I/Q — {TARGET_NAMES.get(TARGET,'?')} | SNR={SNR} dB | {SPLIT.upper()}",
    fontsize=13, fontweight="bold"
)

# Submuestreo para la visualizacion (1 de cada 16 muestras)
step = 16
t_plot = t_axis_ms[::step]

axes[0].plot(t_plot, iq[0][::step].numpy(), linewidth=0.5, color="#0077B6")
axes[0].set_ylabel("Amplitud I", fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-4, 4)

axes[1].plot(t_plot, iq[1][::step].numpy(), linewidth=0.5, color="#D62828")
axes[1].set_ylabel("Amplitud Q", fontsize=11)
axes[1].set_xlabel("Tiempo (ms)", fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-4, 4)

# Anotar potencia instantanea (envolvente)
envelope = (iq[0].pow(2) + iq[1].pow(2)).sqrt()
env_plot  = envelope[::step].numpy()
ax_env    = axes[0].twinx()
ax_env.fill_between(t_plot, env_plot, alpha=0.15, color="#2D6A4F", label="Envolvente |A|")
ax_env.set_ylabel("|A| (Envolvente)", fontsize=9, color="#2D6A4F")
ax_env.tick_params(axis="y", colors="#2D6A4F")
ax_env.set_ylim(0, 8)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Pipeline de transformacion — Paso a paso

# %%
# ── Paso 1: Normalizacion RMS ────────────────────────────────────────────────
rms  = iq.pow(2).mean().clamp(min=1e-12).sqrt()
iq_n = iq / rms

# ── Paso 2: STFT compleja ────────────────────────────────────────────────────
sig_complex = torch.complex(iq_n[0], iq_n[1])
window      = torch.hann_window(NFFT)
stft        = torch.stft(
    sig_complex, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
    window=window, center=False, return_complex=True, onesided=False
)   # [F, T]

# ── Paso 3-5: log-PSD normalizado ───────────────────────────────────────────
psd_lin  = stft.abs().pow(2)
psd_db   = 10.0 * torch.log10(psd_lin + 1e-12)
psd_max  = psd_db.max()
psd_clip = psd_db.clamp(min=psd_max - DB_CLIP)
psd_min  = psd_clip.min()
psd_norm = (psd_clip - psd_min) / (psd_max - psd_min).clamp(min=1e-8)   # [0,1]

# ── Paso 6: Resize bilineal a [SPEC_H, SPEC_W] ──────────────────────────────
psd_4d   = psd_norm.unsqueeze(0).unsqueeze(0)
psd_res  = F.interpolate(psd_4d, size=(SPEC_H, SPEC_W),
                         mode="bilinear", align_corners=False).squeeze()

# ── Paso 7-8: Sobel ──────────────────────────────────────────────────────────
Kx = torch.tensor([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]).view(1,1,3,3)
Ky = torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]).view(1,1,3,3)
img4 = psd_res.unsqueeze(0).unsqueeze(0)
pad  = F.pad(img4, (1,1,1,1), mode="reflect")
Gx   = F.conv2d(pad, Kx).squeeze()
Gy   = F.conv2d(pad, Ky).squeeze()
sobel_mag  = torch.sqrt(Gx**2 + Gy**2 + 1e-8)
sobel_norm = (sobel_mag - sobel_mag.min()) / (sobel_mag.max() - sobel_mag.min() + 1e-8)

# ── Ejes en unidades reales ──────────────────────────────────────────────────
T_frames    = stft.shape[1]
freq_bins   = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1/FS_HZ)) / 1e6  # MHz
time_frames = np.linspace(0, N / FS_HZ * 1000, T_frames)               # ms

print(f"STFT shape: {stft.shape}  (F={NFFT} bins, T={T_frames} frames)")
print(f"Rango temporal: {time_frames[0]:.1f} – {time_frames[-1]:.1f} ms")
print(f"Rango frecuencial: {freq_bins[0]:.1f} – {freq_bins[-1]:.1f} MHz")
print(f"Tensor modelo [2,{SPEC_H},{SPEC_W}]  listo.")


# %% [markdown]
# ## Figura 2 — Espectrograma STFT: las "cajitas" FHSS
# La log-PSD muestra los saltos en frecuencia del protocolo FHSS como
# bloques rectangulares brillantes. Cada "cajita" corresponde a una
# trama de transmision del drone.

# %%
# Espectrograma completo (full resolution, antes del resize)
psd_plot = np.fft.fftshift(psd_norm.numpy(), axes=0)  # centrar frecuencia en 0

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(
    psd_plot,
    aspect="auto",
    origin="lower",
    cmap="inferno",
    extent=[time_frames[0], time_frames[-1],
            freq_bins[0],   freq_bins[-1]],
)
plt.colorbar(im, ax=ax, label="Potencia normalizada [0, 1]")
ax.set_title(
    f"STFT log-PSD — {TARGET_NAMES.get(TARGET,'?')} | SNR={SNR} dB\n"
    f"Cajitas FHSS: bloques de energia en el espectrograma",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Tiempo (ms)", fontsize=11)
ax.set_ylabel("Frecuencia (MHz)", fontsize=11)
ax.axhline(0, color="white", linewidth=0.5, alpha=0.4, linestyle="--")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Figura 3 — Deteccion de bordes Sobel: contornos de las cajitas
# El operador Sobel 3x3 calcula el gradiente espacial del espectrograma.
# Los valores altos (blanco) marcan los BORDES de las cajitas FHSS,
# es decir, los contornos que delimitan cada bloque de transmision.

# %%
# Sobel sobre imagen redimensionada (256x256, igual que ve el modelo)
psd_res_plot   = psd_res.numpy()
sobel_res_plot = sobel_norm.numpy()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    f"Pipeline de transformacion — {TARGET_NAMES.get(TARGET,'?')} | SNR={SNR} dB",
    fontsize=13, fontweight="bold"
)

# Panel 1: log-PSD redimensionado (entrada Canal 0 del modelo)
axes[0].imshow(psd_res_plot, aspect="auto", origin="lower", cmap="inferno", vmin=0, vmax=1)
axes[0].set_title("Canal 0: log-PSD [0,1]\n(entrada real al modelo)", fontsize=11)
axes[0].set_xlabel("Frames temporales (256)")
axes[0].set_ylabel("Bins frecuenciales (256)")
plt.colorbar(
    plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1)),
    ax=axes[0], fraction=0.04
)

# Panel 2: Mapa de bordes Sobel (entrada Canal 1 del modelo)
axes[1].imshow(sobel_res_plot, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=1)
axes[1].set_title("Canal 1: Gradiente Sobel [0,1]\n(bordes de las cajitas)", fontsize=11)
axes[1].set_xlabel("Frames temporales (256)")
axes[1].set_ylabel("")
plt.colorbar(
    plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1)),
    ax=axes[1], fraction=0.04
)

# Panel 3: Superposicion — PSD + bordes Sobel en rojo
axes[2].imshow(psd_res_plot, aspect="auto", origin="lower", cmap="inferno", vmin=0, vmax=1)
# Umbral adaptativo: bordes fuertes (top 15%)
th_val    = float(np.percentile(sobel_res_plot, 85))
edge_mask = sobel_res_plot > th_val
overlay   = np.zeros((*psd_res_plot.shape, 4))
overlay[edge_mask] = [1.0, 0.0, 0.0, 0.75]   # rojo semitransparente
axes[2].imshow(overlay, aspect="auto", origin="lower")
axes[2].set_title("Superposicion: PSD + contornos Sobel\n(bordes del 15% mas alto en rojo)", fontsize=11)
axes[2].set_xlabel("Frames temporales (256)")
axes[2].set_ylabel("")
red_patch = mpatches.Patch(color="red", alpha=0.75, label=f"Bordes Sobel > {th_val:.2f}")
axes[2].legend(handles=[red_patch], fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Figura 4 — Inferencia del modelo: prediccion y probabilidad

# %%
# Construir el tensor [1, 2, H, W] para el modelo
xin_tensor = iq_to_xin_tensor(iq, nfft=NFFT, hop_length=HOP_LENGTH,
                               spec_h=SPEC_H, spec_w=SPEC_W, db_clip=DB_CLIP)
batch = xin_tensor.unsqueeze(0).to(device)   # [1, 2, 256, 256]

with torch.no_grad():
    logit = model(batch)
    prob  = logit.sigmoid().item()
    pred  = int(prob >= 0.5)

label_true = int(row["is_drone"])
label_name = {0: "Ruido", 1: "Drone"}

print("=" * 50)
print(f"  Etiqueta real   : {label_name[label_true]} ({label_true})")
print(f"  Prediccion      : {label_name[pred]} ({pred})")
print(f"  P(drone)        : {prob:.4f}  ({prob*100:.1f}%)")
print(f"  Resultado       : {'CORRECTO' if pred == label_true else 'INCORRECTO'}")
print("=" * 50)

# ── Visualizacion del resultado ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                         gridspec_kw={"width_ratios": [3, 1]})
fig.suptitle(
    f"Inferencia — {TARGET_NAMES.get(TARGET,'?')} | SNR={SNR} dB | {SPLIT.upper()}",
    fontsize=13, fontweight="bold"
)

# Izquierda: espectrograma con resultado superpuesto
axes[0].imshow(psd_res_plot, aspect="auto", origin="lower", cmap="inferno")
axes[0].set_title("Espectrograma (entrada al modelo)", fontsize=11)
axes[0].set_xlabel("Frames temporales")
axes[0].set_ylabel("Bins frecuenciales")

# Borde de color segun resultado
correct   = (pred == label_true)
box_color = "#2D6A4F" if correct else "#D62828"
for spine in axes[0].spines.values():
    spine.set_edgecolor(box_color)
    spine.set_linewidth(4)

# Texto de resultado
result_text = f"P(Drone) = {prob:.3f}\nPrediccion: {label_name[pred]}\nReal: {label_name[label_true]}"
result_color = "#2D6A4F" if correct else "#D62828"
axes[0].text(
    0.02, 0.97, result_text,
    transform=axes[0].transAxes,
    fontsize=10, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
              edgecolor=result_color, linewidth=2),
    color="#1A1A1A"
)

# Derecha: barra de probabilidad
bar_colors = ["#D62828", "#2D6A4F"]
bars = axes[1].barh(
    ["Ruido", "Drone"],
    [1 - prob, prob],
    color=[bar_colors[0], bar_colors[1]],
    edgecolor="black", linewidth=0.8,
    height=0.5
)
axes[1].set_xlim(0, 1)
axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
axes[1].set_xlabel("Probabilidad", fontsize=11)
axes[1].set_title("Salida del modelo", fontsize=11)

# Etiquetas en las barras
for bar, val in zip(bars, [1-prob, prob]):
    axes[1].text(
        min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", fontsize=10, fontweight="bold"
    )

# Estrella sobre la prediccion real
pred_y = ["Ruido", "Drone"][pred]
axes[1].text(
    prob if pred == 1 else (1 - prob),
    pred_y,
    " ◄ Pred",
    va="center", ha="left", fontsize=9, color=result_color, fontweight="bold"
)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Exploracion rapida de SNRs y targets disponibles

# %%
print("SNRs disponibles por target y split:\n")
pivot = df[df["split"] == SPLIT].groupby(["target", "snr"]).size().unstack(fill_value=0)
print(pivot.to_string())
print(f"\nTotal muestras en '{SPLIT}': {(df['split']==SPLIT).sum():,}")
