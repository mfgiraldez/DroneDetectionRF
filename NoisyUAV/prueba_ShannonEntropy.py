"""
Detector de Bursts por Entropía Espectral de Shannon
=====================================================
Solución definitiva — nperseg=4096 + umbral robusto (percentil+MAD)

Parámetros de cara al usuario  (solo 4):
  TARGET, SNR, INDEX  — muestra a inspeccionar
  Z_THRESH            — sensibilidad (z=3 → Pfa ~0.13%/frame)

Parámetros de diseño fijos (no tocar salvo experimento):
  NPERSEG      = 4096   — resolución espectral óptima para FHSS
  MIN_BURST_MS = 0.5    — duración mínima en ms (filtra spikes de 1 frame)
  MERGE_GAP_MS = 3.0    — fusiona bursts contiguos separados < X ms
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.ndimage import binary_erosion, binary_dilation, label
from scipy.stats import norm

from funciones.cargador import obtener_una_muestra, DATA_DIR

# ══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS — solo cambia estos
# ══════════════════════════════════════════════════════════════════════════════
TARGET  = 1
SNR     = -4
INDEX   = 5

Z_THRESH = 3.0    # Bajar → más sensible, más Pfa. Subir → más restrictivo.
                  # z=2.5 → Pfa=0.62%/frame | z=3.0 → Pfa=0.13% | z=3.5 → Pfa=0.02%

# Parámetros de diseño (razonados físicamente, no tocar):
NPERSEG      = 4096   # Δf=3.42 kHz/bin  Δt≈0.15 ms/frame  ← óptimo para FHSS
MIN_BURST_MS = 0.5    # duración mínima de burst aceptable (ms)
MERGE_GAP_MS = 3.0    # fusionar bursts si el hueco entre ellos es < X ms
FS           = 14e6
# ══════════════════════════════════════════════════════════════════════════════


def _calcular_entropia(signal, fs, nperseg):
    """Calcula la curva de entropía espectral de Shannon H(m) sobre la STFT."""
    _, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, return_onesided=False)
    Pxx  = np.abs(Zxx) ** 2
    suma = Pxx.sum(axis=0); suma[suma == 0] = 1e-12
    prob = Pxx / suma + 1e-12
    H    = -(prob * np.log2(prob)).sum(axis=0)
    return t * 1000, H


def _umbral_robusto(H, z_thresh):
    """
    Estima el umbral de detección usando estadísticas robustas globales.

    Por qué NO usamos media/std normales:
      - La media se contamina si hay mucha señal en el archivo.
    Por qué usamos percentil alto + MAD:
      - Las transmisiones BAJAN la entropía → el percentil 90 captura el nivel
        de ruido puro (el 90% de los frames son ruido).
      - El MAD (Median Absolute Deviation) es el estimador de dispersión más
        robusto que existe: inmune a hasta el 50% de valores atípicos.
      - noise_sigma = 1.4826 × MAD  es equivalente a σ para distribuc. Gaussiana.

    El umbral equivale a: "detecto si H(m) < nivel_ruido - z × dispersión_ruido"
    """
    noise_floor = np.percentile(H, 90)                        # piso de ruido
    mad         = np.median(np.abs(H - np.median(H)))         # dispersión robusta
    noise_sigma = 1.4826 * mad                                  # equivalente a σ
    umbral      = noise_floor - z_thresh * noise_sigma
    return umbral, noise_floor, noise_sigma


def detectar_bursts(iq_tensor, fs=14e6, nperseg=4096,
                    z_thresh=3.0, min_burst_ms=0.5, merge_gap_ms=3.0):
    """
    Función principal de detección. Devuelve la curva H y la lista de bursts.
    """
    signal = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()

    # 1. Entropía
    t_ms, H = _calcular_entropia(signal, fs, nperseg)
    dt = float(t_ms[1] - t_ms[0])

    # 2. Umbral robusto (adaptativo al nivel de ruido del archivo, sin contaminar)
    umbral, noise_floor, noise_sigma = _umbral_robusto(H, z_thresh)

    # 3. Máscara de detección
    mascara_raw = H < umbral

    # 4. Post-procesado en unidades físicas (ms), no en frames
    min_f   = max(1, round(min_burst_ms  / dt))
    merge_f = max(1, round(merge_gap_ms  / dt))

    mascara_filt = binary_erosion(mascara_raw,  structure=np.ones(min_f))
    merged       = binary_dilation(mascara_filt, structure=np.ones(merge_f))
    labeled_arr, n_bursts = label(merged)

    # 5. Extraer segmentos — recorte preciso: solo donde H < umbral realmente
    bursts = []
    for i in range(1, n_bursts + 1):
        idx_region = np.where(labeled_arr == i)[0]
        # Dentro de la región, quedarnos con el núcleo que SÍ está bajo el umbral
        inside = idx_region[H[idx_region] < umbral]
        if len(inside) == 0:
            continue
        i0, i1 = inside[0], inside[-1]
        bursts.append({
            "t0":     t_ms[i0],
            "t1":     t_ms[i1],
            "i0":     i0,
            "i1":     i1,
            "dur_ms": t_ms[i1] - t_ms[i0],
            "drop_b": noise_floor - H[i0:i1+1].min(),
            "z_peak": (H[i0:i1+1].min() - noise_floor) / (noise_sigma + 1e-10),
        })

    return t_ms, H, bursts, umbral, noise_floor, noise_sigma


# ══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════
iq, _, _, _ = obtener_una_muestra(DATA_DIR, target=TARGET, snr=SNR, index=INDEX)

t_ms, H, bursts, umbral, noise_floor, noise_sigma = detectar_bursts(
    iq,
    fs           = FS,
    nperseg      = NPERSEG,
    z_thresh     = Z_THRESH,
    min_burst_ms = MIN_BURST_MS,
    merge_gap_ms = MERGE_GAP_MS,
)
dt = float(t_ms[1] - t_ms[0])
n_bursts = len(bursts)

# Diagnóstico en consola
pfa = norm.sf(Z_THRESH)
print("=" * 60)
print(f"  target={TARGET}  SNR={SNR}dB  index={INDEX}")
print("=" * 60)
print(f"  Resolución   : Δt={dt:.3f}ms/frame  Δf={FS/NPERSEG/1e3:.2f}kHz/bin")
print(f"  Frames total : {len(H)}")
print(f"  Noise floor  : {noise_floor:.4f} bits  (percentil 90)")
print(f"  Noise sigma  : {noise_sigma:.4f} bits  (MAD×1.4826)")
print(f"  Umbral       : {umbral:.4f} bits  (z={Z_THRESH} → Pfa/frame={pfa*100:.4f}%)")
print(f"  Min burst    : {MIN_BURST_MS}ms = {max(1,round(MIN_BURST_MS/dt))} frames")
print(f"  Merge gap    : {MERGE_GAP_MS}ms = {max(1,round(MERGE_GAP_MS/dt))} frames")
print()
if n_bursts == 0:
    print("  ✗ Sin transmisión detectada")
else:
    print(f"  ✓ {n_bursts} burst{'s' if n_bursts != 1 else ''} detectados:")
    for k, b in enumerate(bursts):
        print(f"    B{k+1:02d}: {b['t0']:6.2f}–{b['t1']:6.2f} ms  "
              f"dur={b['dur_ms']:.1f}ms  ↓{b['drop_b']:.3f}b  z={b['z_peak']:.1f}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════
signal_np = iq[0].numpy() + 1j * iq[1].numpy()
amp  = np.abs(signal_np)
tamp = np.arange(len(amp)) / FS * 1000

# Espectrograma con la misma resolución que el detector
f_st, t_st, Zxx_sp = stft(signal_np, fs=FS, nperseg=NPERSEG, return_onesided=False)
f_sh = np.fft.fftshift(f_st)
Pdb  = 10 * np.log10(np.abs(np.fft.fftshift(Zxx_sp, axes=0))**2 + 1e-12)

COLS = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c','#ff69b4']

def _marcar(ax, alpha=0.2, yref=None):
    for k, b in enumerate(bursts):
        col = COLS[k % len(COLS)]
        ax.axvspan(b["t0"], b["t1"], color=col, alpha=alpha)
        if yref is not None:
            ax.text((b["t0"]+b["t1"])/2, yref, f'B{k+1}',
                    color=col, ha='center', fontsize=8, fontweight='bold')

plt.style.use('dark_background')
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                         gridspec_kw={'height_ratios': [1, 2, 1.8]})
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#e0e0e0')
    for sp in ax.spines.values(): sp.set_color('#444')

# Panel 1: Amplitud
axes[0].plot(tamp[::8], amp[::8], color='#9b59b6', lw=0.4)
_marcar(axes[0], alpha=0.3, yref=amp.max() * 0.85)
axes[0].set_title('Amplitud  |I+jQ|', color='white', fontsize=12)
axes[0].set_ylabel('Magnitud', color='white')
axes[0].grid(color='#2c3e6b', alpha=0.3)

# Panel 2: Espectrograma
axes[1].pcolormesh(t_st*1000, f_sh/1e6, Pdb, shading='gouraud', cmap='viridis')
_marcar(axes[1], alpha=0.25)
axes[1].set_title(
    f'Espectrograma  (nperseg={NPERSEG} | Δf={FS/NPERSEG/1e3:.2f} kHz/bin | Δt={dt:.2f} ms/frame)',
    color='white', fontsize=12)
axes[1].set_ylabel('Frecuencia (MHz)', color='white')

# Panel 3: Entropía + umbral + bursts sombreados
axes[2].fill_between(t_ms, umbral, noise_floor,
                     alpha=0.10, color='cyan', label='Zona ruido puro')
axes[2].axhline(noise_floor, color='white', ls=':', lw=0.8, alpha=0.5,
                label=f'Nivel ruido  P90={noise_floor:.3f}b')
axes[2].axhline(umbral, color='#e74c3c', ls='--', lw=1.5,
                label=f'Umbral  z={Z_THRESH}  ({umbral:.3f}b)  Pfa/frame={pfa*100:.3f}%')
axes[2].plot(t_ms, H, color='#2ecc71', lw=1.5, label='H(m)  nperseg=4096')
for k, b in enumerate(bursts):
    col = COLS[k % len(COLS)]
    seg = slice(b["i0"], b["i1"] + 1)
    axes[2].fill_between(t_ms[seg], H[seg], umbral,
                         where=H[seg] < umbral,
                         color=col, alpha=0.55,
                         label=f"B{k+1} {b['dur_ms']:.1f}ms  ↓{b['drop_b']:.2f}b  z={b['z_peak']:.1f}")
axes[2].set_title('Entropía de Shannon  +  Umbral robusto (P90 + MAD)', color='white', fontsize=12)
axes[2].set_xlabel('Tiempo (ms)', color='white', fontsize=11)
axes[2].set_ylabel('Bits', color='white')
axes[2].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8,
               ncol=min(3, 1 + n_bursts), loc='lower left')
axes[2].grid(color='#2c3e6b', alpha=0.3)

estado = f"✓ {n_bursts} burst{'s' if n_bursts != 1 else ''}" if n_bursts else "✗ Sin transmisión"
fig.suptitle(
    f'target={TARGET}  SNR={SNR}dB  index={INDEX}  |  {estado}  |  '
    f'nperseg={NPERSEG}  z={Z_THRESH}  minBurst={MIN_BURST_MS}ms  merge={MERGE_GAP_MS}ms',
    color='white', fontsize=12, y=1.005)
plt.tight_layout()
plt.show()