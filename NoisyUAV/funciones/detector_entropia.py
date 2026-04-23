"""
detector_entropia.py
====================
Detector robusto de bursts FHSS basado en Entropía Espectral de Shannon.

Pipeline de detección (3 capas de discriminación):
  1. Entropía espectral whitened  → detecta concentración espectral
  2. Umbral CFAR adaptativo       → se adapta al piso de ruido local
  3. Filtro de ancho espectral    → rechaza interferencias de banda ancha (WiFi/BT)
                                    (criterio P75 de bins activos, más estricto que mediana)

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
                    Criterio: MAX del burst (cualquier frame WiFi = rechazar)
  smooth_ms       : sigma del suavizado gaussiano de H    [default: 0.2 ms]
                    0 desactiva el suavizado
  adaptive_window_ms: ventana CFAR (0 = umbral global)   [default: 15.0 ms]
  max_merge_gap_ms: RETIRADO - usar merge_gap_ms mayor si se necesita
"""

import numpy as np
from scipy.signal import stft
from scipy.ndimage import (binary_erosion, binary_dilation,
                            label, gaussian_filter1d, percentile_filter,
                            gaussian_filter)

__all__ = ["detectar_bursts", "print_diagnostico", "plot_muestra",
           "plot_espectrograma_3d"]

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
    # Pxx_w = Pxx                                     # Sin whitening
    
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
                    smooth_ms=0.2, adaptive_window_ms=15.0,
                    max_merge_gap_ms=5.0):
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
                   Criterio: P75 del burst → más estricto que mediana
    smooth_ms    : sigma del suavizado gaussiano sobre H  (0 = sin suavizado)
    adaptive_window_ms : ventana CFAR en ms  (0 = umbral global P90+MAD)
    max_merge_gap_ms   : gap máximo para fusionar hops del mismo dron (ms)
                         Si > merge_gap_ms, hace segunda pasada de fusión

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
    i_chan = iq_tensor[0].numpy() if hasattr(iq_tensor[0], 'numpy') else iq_tensor[0]
    q_chan = iq_tensor[1].numpy() if hasattr(iq_tensor[1], 'numpy') else iq_tensor[1]
    signal = i_chan + 1j * q_chan
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

        # FIX 1 (v3 - DEFINITIVO): medir n_active SOLO en la ventana estrecha
        # alrededor del mínimo de entropía (±1ms = el instante del hop FHSS).
        #
        # Razonamiento físico:
        #   - En el frame del mínimo de H, si es FHSS: pocos bins activos (~186)
        #   - Si es WiFi en ese mismo instante: muchos bins activos (>>512)
        #   - El WiFi que ocurre en OTROS instantes del burst (fusionado por
        #     dilatación) NO contamina esta medida → no se rechazan bursts legítimos
        core_half  = max(1, round(1.0 / dt))          # ±1 ms alrededor del pico
        core_start = max(i0, i0 + pk - core_half)
        core_end   = min(i1, i0 + pk + core_half)
        n_act      = float(np.median(n_active[core_start:core_end + 1]))

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
    Figura interactiva Plotly con 5 paneles.
    Paneles: Amplitud | Espectrograma | Espectro medio | H(m)+CFAR | Bins activos
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    signal = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
    amp    = np.abs(signal)
    tamp   = np.arange(len(amp)) / fs * 1000
    
    f_st, t_st, Zxx_sp = stft(signal, fs=fs, nperseg=nperseg, return_onesided=False)
    f_sh  = np.fft.fftshift(f_st)
    Pdb   = 10 * np.log10(np.abs(np.fft.fftshift(Zxx_sp, axes=0))**2 + 1e-12)
    Pdb_mean = Pdb.mean(axis=1)
    freq_mhz = f_sh / 1e6

    dt               = float(t_ms[1] - t_ms[0])
    max_bins_abs     = int(nperseg * max_bins_frac)
    n_ruido_esperado = int(nperseg * 2**(-bg_mult))

    # Para un layout profesional Q1 de revista, usamos colores más sobrios
    COLS = ['#e74c3c','#e67e22','#16a085','#27ae60','#2980b9','#8e44ad','#2c3e50','#f39c12']
    RGBA = ['rgba(231,76,60,{a})','rgba(230,126,34,{a})','rgba(22,160,133,{a})',
            'rgba(39,174,96,{a})','rgba(41,128,185,{a})','rgba(142,68,173,{a})',
            'rgba(44,62,80,{a})','rgba(243,156,18,{a})']

    subtitles = [
        'Amplitud  |I+jQ|',
        f'Espectrograma  Δf={fs/nperseg/1e3:.2f} kHz  Δt={dt:.2f} ms',
        'PSD media',
        'H(m) whitened + Umbral CFAR (P90+MAD)',
        f'Bins activos  →  FHSS≈{n_ruido_esperado+58} | WiFi≈{nperseg} | Límite={max_bins_abs}'
    ]
    
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.85, 0.15],
        row_heights=[0.12, 0.35, 0.30, 0.23],
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.015,
        specs=[
            [{}, None],   # Row 1: Amplitud solo col 1
            [{}, {}],     # Row 2: Espectrograma col 1, PSD col 2
            [{}, None],   # Row 3: H(m) solo col 1
            [{}, None]    # Row 4: Bins solo col 1
        ],
        subplot_titles=subtitles
    )

    # P1: Amplitud (downsampled for performance)
    fig.add_trace(go.Scatter(x=tamp[::8], y=amp[::8],
                             line=dict(color='#8e44ad', width=0.6),
                             name='|IQ|', showlegend=False), row=1, col=1)

    # P2: Espectrograma (Row 2, Col 1)
    # Colorbar is moved completely to the far right of the figure to not overlap the middle
    fig.add_trace(go.Heatmap(x=t_st*1000, y=freq_mhz, z=Pdb,
                             colorscale='Viridis', showscale=True,
                             zmin=np.percentile(Pdb, 5), zmax=np.percentile(Pdb, 99.5),
                             colorbar=dict(len=0.33, y=0.75, x=1.015, thickness=14,
                                          title=dict(text='dB', side='right', font=dict(size=13, color='black')),
                                          tickfont=dict(size=12, color='black'))),
                  row=2, col=1)

    # P3: Espectro medio (Row 2, Col 2) - Swap X/Y para que Y sea frecuencia
    noise_ref = float(np.median(Pdb_mean))
    fig.add_trace(go.Scatter(y=freq_mhz,
                             x=np.full(len(freq_mhz), noise_ref),
                             line=dict(width=0), showlegend=False,
                             hoverinfo='skip'), row=2, col=2)
    fig.add_trace(go.Scatter(y=freq_mhz,
                             x=np.clip(Pdb_mean, noise_ref, None),
                             fill='tonextx',
                             fillcolor='rgba(231,76,60,0.28)',
                             line=dict(width=0), showlegend=False,
                             hoverinfo='skip'), row=2, col=2)
    fig.add_trace(go.Scatter(y=freq_mhz, x=Pdb_mean,
                             line=dict(color='#2980b9', width=1.2),
                             name='PSD media', showlegend=False), row=2, col=2)
    fig.add_vline(x=noise_ref, line=dict(color='#555555', width=1, dash='dot'),
                  annotation_text=f'Med. {noise_ref:.0f}dB',
                  annotation_position='top right',
                  annotation_textangle=90,
                  annotation_font=dict(size=11, color='#333333'), row=2, col=2)

    # P4: H(m) + CFAR (Row 3, Col 1)
    fig.add_trace(go.Scatter(x=t_ms, y=nf_v,
                             line=dict(width=0), showlegend=False,
                             hoverinfo='skip'), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=umbral_v, fill='tonexty',
                             fillcolor='rgba(231,76,60,0.08)',
                             line=dict(color='#e74c3c', width=1.5, dash='dash'),
                             name=f'Umbral CFAR z={z_thresh} W={adaptive_window_ms}ms'),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=nf_v,
                             line=dict(color='#7f8c8d',
                                       width=0.8, dash='dot'),
                             name='P90 local'), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=H,
                             line=dict(color='rgba(39,174,96,0.35)', width=0.8),
                             name='H(m) raw'), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=H_smooth,
                             line=dict(color='#27ae60', width=2.0),
                             name='H(m) smooth'), row=3, col=1)
    
    for k, b in enumerate(bursts):
        seg = slice(b['i0'], b['i1']+1)
        t_seg = t_ms[seg]; h_seg = H_smooth[seg]; u_seg = umbral_v[seg]
        xp = np.concatenate([t_seg, t_seg[::-1]])
        yp = np.concatenate([np.minimum(h_seg, u_seg), u_seg[::-1]])
        fig.add_trace(go.Scatter(x=xp, y=yp, fill='toself',
                                 fillcolor=RGBA[k%len(RGBA)].format(a=0.5),
                                 line=dict(width=0),
                                 name=f"B{k+1} {b['dur_ms']:.2f}ms z={b['z_peak']:.1f}"),
                      row=3, col=1)

    # P5: Bins activos (Row 4, Col 1)
    fig.add_trace(go.Scatter(x=t_ms, y=n_active,
                             fill='tozeroy',
                             fillcolor='rgba(243,156,18,0.65)',
                             line=dict(color='rgba(243,156,18,0.9)', width=0.5),
                             name='Bins activos'), row=4, col=1)
    fig.add_hline(y=max_bins_abs,
                  line=dict(color='#e74c3c', width=1.5, dash='dash'),
                  annotation_text=f'Límite WiFi {max_bins_abs}',
                  annotation_position='top right',
                  annotation_font=dict(size=11, color='#e74c3c'), row=4, col=1)
    fig.add_hline(y=n_ruido_esperado,
                  line=dict(color='#7f8c8d', width=1, dash='dot'),
                  annotation_text=f'Ruido≈{n_ruido_esperado}',
                  annotation_position='bottom right',
                  annotation_font=dict(size=11, color='#333333'), row=4, col=1)

    # Burst vrects (paneles de tiempo) + etiquetas en P1
    for k, b in enumerate(bursts):
        for r in [1, 2, 3, 4]:
            fig.add_vrect(x0=b['t0'], x1=b['t1'],
                          fillcolor=COLS[k%len(COLS)], opacity=0.18,
                          layer='below', line_width=0, row=r, col=1)
        fig.add_annotation(x=(b['t0']+b['t1'])/2, y=float(amp.max())*0.88,
                           text=f'<b>B{k+1}</b>',
                           xref='x', yref='y',
                           font=dict(color=COLS[k%len(COLS)], size=12),
                           showarrow=False)

    estado = f"✓ {len(bursts)} bursts" if bursts else "✗ Sin detección"
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=950,
        font=dict(size=12, color='#111111'),
        title=dict(text=f'<b>{titulo}</b>  |  {estado}  |  '
                        f'nperseg={nperseg}  z={z_thresh}  '
                        f'bgMult={bg_mult}  maxBins={max_bins_frac*100:.0f}%',
                   font=dict(size=14, color='black'), x=0.5, xanchor='center'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#cccccc', borderwidth=1,
                    font=dict(size=11, color='black'),
                    x=0.865, y=0.48, xanchor='left', yanchor='top'),
        margin=dict(l=60, r=40, t=80, b=40),
        
        # Axis mappings based on grid specifications
        xaxis=dict(showticklabels=False, title='', linecolor='#cccccc', linewidth=1, mirror=True),
        xaxis2=dict(matches='x', showticklabels=False, title='', linecolor='#cccccc', linewidth=1, mirror=True),
        xaxis3=dict(title='dB', title_font=dict(size=12), linecolor='#cccccc', linewidth=1, mirror=True),
        xaxis4=dict(matches='x', showticklabels=False, title='', linecolor='#cccccc', linewidth=1, mirror=True),
        xaxis5=dict(matches='x', title='Tiempo (ms)', title_font=dict(size=12), linecolor='#cccccc', linewidth=1, mirror=True),

        yaxis=dict(title='|IQ|', title_font=dict(size=12), linecolor='#cccccc', linewidth=1, mirror=True),
        yaxis2=dict(title='Freq (MHz)', title_font=dict(size=12), linecolor='#cccccc', linewidth=1, mirror=True),
        yaxis3=dict(matches='y2', showticklabels=False, title='', linecolor='#cccccc', linewidth=1, mirror=True),
        yaxis4=dict(title='Bits', title_font=dict(size=12), linecolor='#cccccc', linewidth=1, mirror=True),
        yaxis5=dict(title='N bins', title_font=dict(size=12), linecolor='#cccccc', linewidth=1, mirror=True),
    )
    return fig



def plot_espectrograma_3d(iq_tensor, fs=14e6, nperseg=2048,
                          t_stride=None, f_stride=None,
                          smooth_sigma=2.0, floor_pct=5,
                          cmap='Turbo', titulo="", t_lim_ms=None):
    """
    Espectrograma 3D interactivo (Plotly).
    Soporta visualización del intervalo temporal completo gracias a t_stride/f_stride dinámicos
    cuando se dejan en None.

    Parámetros
    ----------
    t_stride  : submuestreo temporal  (None → automático: ≈400 puntos)
    f_stride  : submuestreo espectral (None → automático: ≈300 puntos)
    t_lim_ms  : (t0,t1) ms para recortar  [None = todo el intervalo, p.ej. los 75ms completos]
    smooth_sigma : suavizado gaussiano 2D  [2.0 → hace visibles los hops tapados por el ruido]
    floor_pct : percentil inferior para recortar el suelo de ruido [5]
    cmap      : colorscale Plotly          ['Turbo']
    """
    import plotly.graph_objects as go

    signal  = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
    f_st, t_st, Zxx = stft(signal, fs=fs, nperseg=nperseg, return_onesided=False)
    t_ms_3d = t_st * 1000
    f_sh    = np.fft.fftshift(f_st) / 1e6
    Pdb_3d  = 10 * np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))**2 + 1e-12)

    if t_lim_ms is not None:
        mask    = (t_ms_3d >= t_lim_ms[0]) & (t_ms_3d <= t_lim_ms[1])
        t_ms_3d = t_ms_3d[mask]
        Pdb_3d  = Pdb_3d[:, mask]

    if smooth_sigma > 0:
        Pdb_3d = gaussian_filter(Pdb_3d, sigma=smooth_sigma)

    # Compute valid automatic strides to handle 75ms files interactively (~1500 frames defaults to stride 4)
    if t_stride is None:
        t_stride = max(1, len(t_ms_3d) // 400)
    if f_stride is None:
        f_stride = max(1, len(f_sh) // 300)

    T = t_ms_3d[::t_stride]
    F = f_sh[::f_stride]
    Z = Pdb_3d[::f_stride, ::t_stride]

    vmin = float(np.percentile(Z, floor_pct))
    vmax = float(np.percentile(Z, 99))
    Z    = np.clip(Z, vmin, vmax)

    info = (f't_stride={t_stride}  f_stride={f_stride}  '
            f'smooth={smooth_sigma}  →  {len(T)}×{len(F)} pts')

    fig = go.Figure(data=[go.Surface(
        x=T, y=F, z=Z,
        colorscale=cmap,
        cmin=vmin, cmax=vmax,
        colorbar=dict(thickness=15,
                      title=dict(text='dB', side='right', font=dict(size=13, color='black')),
                      tickfont=dict(size=12, color='white')),
        hovertemplate='Tiempo: %{x:.2f} ms<br>Frec: %{y:.2f} MHz<br>Potencia: %{z:.1f} dB<extra></extra>'
    )])

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d0d1a',
        height=700,
        font=dict(size=12),
        title=dict(text=f'<b>{titulo}</b>  |  Espectrograma 3D  |  {info}',
                   font=dict(size=14, color='white'), x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='Tiempo (ms)', title_font=dict(size=13), tickfont=dict(size=11), gridcolor='#2a2a3a'),
            yaxis=dict(title='Frecuencia (MHz)', title_font=dict(size=13), tickfont=dict(size=11), gridcolor='#2a2a3a'),
            zaxis=dict(title='Potencia (dB)', title_font=dict(size=13), tickfont=dict(size=11), gridcolor='#2a2a3a',
                       range=[vmin, vmax]),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.6, y=-1.6, z=0.9)
            ),
            bgcolor='#0d0d1a'
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig
