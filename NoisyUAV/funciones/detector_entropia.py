"""
detector_entropia.py
====================
Detector robusto de bursts FHSS basado en Entropía Espectral de Shannon.

Pipeline de detección (3 capas de discriminación):
  1. Entropía espectral whitened  → detecta concentración espectral
  2. Umbral CFAR adaptativo       → se adapta al piso de ruido local
  3. Filtro de ancho espectral    → rechaza interferencias de banda ancha (WiFi/BT)

API pública
-----------
  detectar_bursts(iq_tensor, ...)  → (t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts)
  print_diagnostico(...)           → imprime resumen en consola
  plot_muestra(...)                → Figure de matplotlib (4 paneles)

Parámetros principales
----------------------
  fs              : frecuencia de muestreo (Hz)           [default: 14e6]
  nperseg         : tamaño de ventana STFT                [default: 2048]
  z_thresh        : z-score del umbral P90–z·MAD          [default: 3.0]
  min_burst_ms    : duración mínima de burst              [default: 0.5 ms]
  merge_gap_ms    : gap máximo para fusionar hops FHSS    [default: 1.0 ms]
  min_z_abs       : significancia mínima del pico         [default: 4.0]
  bg_mult         : múltiplo sobre fondo para bin activo  [default: 4.0]
                    P(bin_ruido > bg_mult×fondo) = 2^(-bg_mult)
  max_bins_frac   : fracción máxima de bins activos       [default: 0.25]
                    FHSS: ~pocos bins | WiFi OFDM: ~todos
  smooth_ms       : sigma del suavizado gaussiano de H    [default: 0.2 ms]
                    0 desactiva el suavizado
  adaptive_window_ms: ventana CFAR (0 = umbral global)   [default: 15.0 ms]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.ndimage import (binary_erosion, binary_dilation,
                            label, gaussian_filter1d, percentile_filter)

__all__ = ["detectar_bursts", "print_diagnostico", "plot_muestra"]

# ──────────────────────────────────────────────────────────────────────────────
#  INTERNALS
# ──────────────────────────────────────────────────────────────────────────────

def _calcular_features(signal, fs, nperseg, bg_mult):
    """
    Calcula H(m) y n_active(m) sobre la STFT con whitening espectral.

    Whitening: divide cada bin por su mediana temporal → el AWGN se normaliza
    a ~1 en todos los bins (entropía máxima). Los hops FHSS hacen que uno o
    pocos bins superen el fondo → la entropía CAE localmente.

    n_active: número de bins con Pxx_w > bg_mult por frame.
    - Ruido puro  : E[n_active] = N_bins × 2^(-bg_mult)   (análitico)
    - FHSS -12dB  : ruido + ~58 bins de señal
    - WiFi OFDM   : casi todos los bins (>>N_bins/4)
    """
    _, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, return_onesided=False)
    Pxx = np.abs(Zxx) ** 2                          # (N_freq, N_frames)

    bg = np.median(Pxx, axis=1, keepdims=True)
    bg[bg < 1e-12] = 1e-12
    Pxx_w = Pxx / bg                                # espectro whitened

    suma = Pxx_w.sum(axis=0)
    suma[suma == 0] = 1e-12
    prob = Pxx_w / suma + 1e-12
    H = -(prob * np.log2(prob)).sum(axis=0)

    n_active = (Pxx_w > bg_mult).sum(axis=0).astype(float)
    return t * 1000, H, n_active


def _umbral_robusto(H, z_thresh):
    """
    Umbral global: P90(H) − z × (1.4826 × MAD).
    P90 como estimador del piso de ruido (resistente a ~10% de frames con señal).
    MAD como estimador robusto de dispersión (resistente al 50% de outliers).
    Devuelve: (umbral_escalar, noise_floor_escalar, noise_sigma_escalar)
    """
    nf  = np.percentile(H, 90)
    mad = np.median(np.abs(H - np.median(H)))
    ns  = 1.4826 * mad
    return nf - z_thresh * ns, nf, ns


def _umbral_adaptativo(H, z_thresh, window_ms, dt):
    """
    Umbral CFAR (Constant False Alarm Rate):
      - noise_floor: P90 en ventana deslizante de window_ms ms
      - noise_sigma: MAD global (más estable que MAD local)

    Ventajas:
      - Se adapta a cambios lentos del piso de ruido (AGC, interferencia sostenida)
      - En regiones WiFi el umbral baja con el WiFi → el dron debe superar
        una caída EXTRA sobre el WiFi local → más discriminativo

    Devuelve: (umbral_vector, noise_floor_vector, noise_sigma_escalar)
    """
    W = max(3, round(window_ms / dt))
    nf_local = percentile_filter(H, percentile=90, size=W, mode='nearest')
    mad_global = np.median(np.abs(H - np.median(H)))
    ns = 1.4826 * mad_global
    return nf_local - z_thresh * ns, nf_local, ns


# ──────────────────────────────────────────────────────────────────────────────
#  API PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def detectar_bursts(iq_tensor, fs=14e6, nperseg=2048,
                    z_thresh=3.0, min_burst_ms=0.5, merge_gap_ms=1.0,
                    min_z_abs=4.0, bg_mult=4.0, max_bins_frac=0.25,
                    smooth_ms=0.2, adaptive_window_ms=15.0):
    """
    Detecta bursts FHSS en una muestra IQ.

    Parámetros
    ----------
    iq_tensor : Tensor [2, N]  (canal 0=I, canal 1=Q)
    fs        : frecuencia de muestreo (Hz)
    nperseg   : ventana STFT
    z_thresh  : umbral en unidades de sigma  (z=3 → Pfa/frame~0.13%)
    min_burst_ms : duración mínima de burst válido (ms)
    merge_gap_ms : fusiona bursts separados por menos de este gap (ms)
    min_z_abs    : z-score mínimo del pico de entropía para aceptar burst
    bg_mult      : multiplicador sobre fondo para contar bin como "activo"
    max_bins_frac: fracción máxima de bins activos  (discrimina WiFi vs FHSS)
    smooth_ms    : sigma del suavizado gaussiano sobre H  (0 = sin suavizado)
    adaptive_window_ms : ventana CFAR en ms  (0 = umbral global P90+MAD)

    Retorna
    -------
    t_ms      : array (N_frames,) — eje temporal en ms
    H         : array (N_frames,) — entropía raw (whitened)
    H_smooth  : array (N_frames,) — entropía suavizada (usada para detección)
    umbral_v  : array (N_frames,) — umbral CFAR (o constante si adaptive=0)
    nf_v      : array (N_frames,) — piso de ruido (P90 local o global)
    ns        : float             — noise_sigma (MAD global escalado)
    n_active  : array (N_frames,) — bins activos por frame
    bursts    : list[dict]        — lista de bursts detectados:
                  t0, t1, i0, i1 → delimitadores (ms e índice)
                  dur_ms          → duración
                  drop_b          → caída de entropía respecto al piso local
                  z_peak          → significancia estadística
                  n_act           → bins activos medianos en el burst
    """
    signal = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
    t_ms, H, n_active = _calcular_features(signal, fs, nperseg, bg_mult)
    dt = float(t_ms[1] - t_ms[0])

    # Suavizado gaussiano de H
    if smooth_ms > 0:
        H_smooth = gaussian_filter1d(H, sigma=max(0.5, smooth_ms / dt))
    else:
        H_smooth = H.copy()

    # Umbral adaptativo o global
    if adaptive_window_ms > 0:
        umbral, noise_floor, ns = _umbral_adaptativo(
            H_smooth, z_thresh, adaptive_window_ms, dt)
        umbral_v = umbral
        nf_v     = noise_floor
    else:
        umbral_s, nf_s, ns = _umbral_robusto(H_smooth, z_thresh)
        umbral_v = np.full_like(H_smooth, umbral_s)
        nf_v     = np.full_like(H_smooth, nf_s)

    max_bins = nperseg * max_bins_frac
    min_f    = max(1, round(min_burst_ms / dt))
    merge_f  = max(1, round(merge_gap_ms  / dt))

    mask = binary_dilation(
               binary_erosion(H_smooth < umbral_v, structure=np.ones(min_f)),
               structure=np.ones(merge_f))
    labeled_arr, n_regions = label(mask)

    bursts = []
    for i in range(1, n_regions + 1):
        idx    = np.where(labeled_arr == i)[0]
        inside = idx[H_smooth[idx] < umbral_v[idx]]
        if len(inside) == 0:
            continue
        i0, i1 = inside[0], inside[-1]

        h_seg  = H_smooth[i0:i1+1]
        nf_seg = nf_v[i0:i1+1]
        pk     = np.argmin(h_seg)
        h_min  = h_seg[pk]
        z_peak = (h_min - nf_seg[pk]) / (ns + 1e-10)
        n_act  = float(np.median(n_active[i0:i1+1]))

        if abs(z_peak) < min_z_abs:
            continue
        if n_act > max_bins:
            continue

        bursts.append({
            "t0":     t_ms[i0],
            "t1":     t_ms[i1],
            "i0":     i0,
            "i1":     i1,
            "dur_ms": t_ms[i1] - t_ms[i0],
            "drop_b": nf_seg[pk] - h_min,
            "z_peak": z_peak,
            "n_act":  n_act,
        })

    return t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts


def print_diagnostico(t_ms, nf_v, ns, umbral_v, n_active, bursts,
                      nperseg=2048, fs=14e6,
                      z_thresh=3.0, bg_mult=4.0, max_bins_frac=0.25,
                      min_burst_ms=0.5, merge_gap_ms=1.0,
                      target=None, snr=None, index=None):
    """
    Imprime en consola el resumen de detección.

    Parámetros
    ----------
    Los arrays t_ms, nf_v, ns, umbral_v, n_active, bursts son la salida
    directa de detectar_bursts().
    target, snr, index : etiquetas opcionales para el encabezado.
    """
    from scipy.stats import norm as _norm

    dt           = float(t_ms[1] - t_ms[0])
    nf_scalar    = float(np.median(nf_v))
    umbral_scalar = float(np.median(umbral_v))
    pfa          = _norm.sf(z_thresh)
    max_bins_abs      = int(nperseg * max_bins_frac)
    n_ruido_esperado  = int(nperseg * 2**(-bg_mult))

    header = ""
    if target is not None:
        header += f"target={target}"
    if snr is not None:
        header += f"  SNR={snr}dB"
    if index is not None:
        header += f"  index={index}"

    sep = "=" * 65
    print(sep)
    if header:
        print(f"  {header.strip()}")
        print(sep)
    print(f"  Resolución   : Δt={dt:.3f}ms/frame  Δf={fs/nperseg/1e3:.2f}kHz/bin")
    print(f"  Frames total : {len(t_ms)}")
    print(f"  Noise floor  : {nf_scalar:.4f} bits  (P90 whitened, mediana CFAR)")
    print(f"  Noise sigma  : {ns:.4f} bits  (MAD×1.4826)")
    print(f"  Umbral       : {umbral_scalar:.4f} bits  "
          f"(z={z_thresh} → Pfa/frame={pfa*100:.4f}%)")
    print(f"  Min burst    : {min_burst_ms}ms = {max(1,round(min_burst_ms/dt))} frames")
    print(f"  Merge gap    : {merge_gap_ms}ms = {max(1,round(merge_gap_ms/dt))} frames")
    print(f"  Bins activos : ruido~{n_ruido_esperado}  límite WiFi={max_bins_abs}  "
          f"(bg_mult={bg_mult}, {max_bins_frac*100:.0f}% de {nperseg})")
    print()
    if not bursts:
        print("  ✗ Sin transmisión detectada")
    else:
        print(f"  ✓ {len(bursts)} burst{'s' if len(bursts) != 1 else ''} detectados:")
        for k, b in enumerate(bursts):
            print(f"    B{k+1:02d}: {b['t0']:7.2f}–{b['t1']:7.2f} ms  "
                  f"dur={b['dur_ms']:.2f}ms  "
                  f"↓{b['drop_b']:.3f}b  "
                  f"z={b['z_peak']:.1f}  "
                  f"bins_act={b['n_act']:.0f}")
    print(sep)


def plot_muestra(iq_tensor, t_ms, H, H_smooth, umbral_v, nf_v, ns,
                 n_active, bursts, fs=14e6, nperseg=2048,
                 z_thresh=3.0, bg_mult=4.0, max_bins_frac=0.25,
                 adaptive_window_ms=15.0, titulo=""):
    """
    Genera figura de 4 paneles para inspección visual de la detección.

    Paneles
    -------
    1. Amplitud |I+jQ|
    2. Espectrograma (STFT)
    3. H(m) raw + H_smooth + umbral CFAR + bursts sombreados
    4. Bins activos por frame + límite WiFi
    """
    signal = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
    amp    = np.abs(signal)
    tamp   = np.arange(len(amp)) / fs * 1000

    f_st, t_st, Zxx_sp = stft(signal, fs=fs, nperseg=nperseg, return_onesided=False)
    f_sh = np.fft.fftshift(f_st)
    Pdb  = 10 * np.log10(np.abs(np.fft.fftshift(Zxx_sp, axes=0))**2 + 1e-12)

    dt          = float(t_ms[1] - t_ms[0])
    max_bins_abs     = int(nperseg * max_bins_frac)
    n_ruido_esperado = int(nperseg * 2**(-bg_mult))

    COLS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
            '#3498db', '#9b59b6', '#1abc9c', '#ff69b4']

    def _marcar(ax, alpha=0.2, yref=None):
        for k, b in enumerate(bursts):
            ax.axvspan(b["t0"], b["t1"], color=COLS[k % len(COLS)], alpha=alpha)
            if yref is not None:
                ax.text((b["t0"]+b["t1"])/2, yref, f'B{k+1}',
                        color=COLS[k % len(COLS)], ha='center',
                        fontsize=8, fontweight='bold', clip_on=True)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={'height_ratios': [1, 2, 1.8, 1.2]})
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#e0e0e0')
        for sp in ax.spines.values():
            sp.set_color('#444')

    # P1 — Amplitud
    axes[0].plot(tamp[::8], amp[::8], color='#9b59b6', lw=0.4)
    _marcar(axes[0], alpha=0.3, yref=amp.max() * 0.85)
    axes[0].set_title('Amplitud  |I+jQ|', color='white', fontsize=11)
    axes[0].set_ylabel('|IQ|', color='white')
    axes[0].grid(color='#2c3e6b', alpha=0.3)

    # P2 — Espectrograma
    axes[1].pcolormesh(t_st * 1000, f_sh / 1e6, Pdb,
                       shading='gouraud', cmap='viridis')
    _marcar(axes[1], alpha=0.25)
    axes[1].set_title(
        f'Espectrograma  Δf={fs/nperseg/1e3:.2f}kHz  Δt={dt:.2f}ms',
        color='white', fontsize=11)
    axes[1].set_ylabel('Freq (MHz)', color='white')

    # P3 — Entropía + umbral CFAR
    axes[2].fill_between(t_ms, umbral_v, nf_v,
                         alpha=0.08, color='cyan', label='Zona ruido CFAR')
    axes[2].plot(t_ms, nf_v,     color='white',   lw=0.8, ls=':', alpha=0.5,
                 label='P90 local (CFAR)')
    axes[2].plot(t_ms, umbral_v, color='#e74c3c', lw=1.5, ls='--',
                 label=f'Umbral CFAR  z={z_thresh}  W={adaptive_window_ms}ms')
    axes[2].plot(t_ms, H,        color='#2ecc71', lw=0.6, alpha=0.35, label='H(m) raw')
    axes[2].plot(t_ms, H_smooth, color='#2ecc71', lw=1.8,             label='H(m) smooth')
    for k, b in enumerate(bursts):
        seg = slice(b["i0"], b["i1"] + 1)
        axes[2].fill_between(
            t_ms[seg], H_smooth[seg], umbral_v[seg],
            where=H_smooth[seg] < umbral_v[seg],
            color=COLS[k % len(COLS)], alpha=0.55,
            label=f"B{k+1} {b['dur_ms']:.2f}ms  z={b['z_peak']:.1f}")
    axes[2].set_title('H(m) whitened + Umbral CFAR (P90+MAD)', color='white', fontsize=11)
    axes[2].set_ylabel('Bits', color='white')
    axes[2].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=7,
                   ncol=min(4, 2 + len(bursts)), loc='lower left')
    axes[2].grid(color='#2c3e6b', alpha=0.3)

    # P4 — Bins activos
    axes[3].fill_between(t_ms, 0, n_active, color='#f39c12', alpha=0.7, lw=0,
                          label='Bins activos (Pxx_w > bg_mult×fondo)')
    axes[3].axhline(max_bins_abs, color='#e74c3c', ls='--', lw=1.5,
                    label=f'Límite WiFi = {max_bins_abs} bins ({max_bins_frac*100:.0f}%)')
    axes[3].axhline(n_ruido_esperado, color='white', ls=':', lw=1, alpha=0.5,
                    label=f'Ruido esperado = {n_ruido_esperado} bins (2^-{bg_mult:.0f}×{nperseg})')
    _marcar(axes[3], alpha=0.3)
    axes[3].set_title(
        f'Bins activos/frame  →  FHSS≈{n_ruido_esperado+58} | '
        f'WiFi≈{nperseg}  |  Límite={max_bins_abs}',
        color='white', fontsize=11)
    axes[3].set_xlabel('Tiempo (ms)', color='white', fontsize=11)
    axes[3].set_ylabel('N bins', color='white')
    axes[3].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8, loc='upper right')
    axes[3].grid(color='#2c3e6b', alpha=0.3)

    estado = f"✓ {len(bursts)} bursts" if bursts else "✗ Sin detección"
    fig.suptitle(
        f'{titulo}  |  {estado}  |  '
        f'nperseg={nperseg}  z={z_thresh}  bgMult={bg_mult}  '
        f'maxBins={max_bins_frac*100:.0f}%',
        color='white', fontsize=11, y=1.005)
    plt.tight_layout()
    return fig
