"""
Módulo de visualización para el dataset NoisyUAV v2.
Genera paneles multi-dominio (I/Q temporal, envolvente, espectrograma STFT,
PSD Welch) y gráficas comparativas por SNR y por clase.

Optimizado para una visualización profesional y clara en notebooks Jupyter.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import scipy.signal as signal
from typing import Optional, List, Tuple

import sys
import os

# Añadir la carpeta padre al path para poder importar 'funciones'
sys.path.insert(0, r"c:\repos\DroneDetectionRF")

from NoisyUAV.funciones.dataset.cargador import (
    NOMBRES_CLASES,
    FREQ_MUESTREO,
    TARGET_NOISE,
    obtener_una_muestra,
    cargar_muestra,
    obtener_muestras_por_clase,
    DATA_DIR,
)

# ============================================================================
# CONFIGURACIÓN ESTÉTICA GLOBAL
# ============================================================================

_COLORES_CLASES = {
    0: "#e74c3c",  # DJI — Rojo
    1: "#3498db",  # FutabaT14 — Azul
    2: "#2ecc71",  # FutabaT7 — Verde
    3: "#f39c12",  # Graupner — Naranja
    4: "#7f8c8d",  # Noise — Gris
    5: "#9b59b6",  # Taranis — Púrpura
    6: "#1abc9c",  # Turnigy — Turquesa
}

_ESTILO_PANEL = {
    "figure.facecolor": "#ffffff",        # Fondo blanco para la tesis
    "axes.facecolor": "#ffffff",          # Fondo blanco en las gráficas
    "axes.edgecolor": "#333333",          # Bordes grises oscuros
    "axes.labelcolor": "#2c3e50",         # Textos de ejes en azul muy oscuro/gris
    "text.color": "#2c3e50",              # Textos generales
    "xtick.color": "#333333",             # Marcas X
    "ytick.color": "#333333",             # Marcas Y
    "grid.color": "#bdc3c7",              # Rejilla gris claro
    "grid.alpha": 0.7,                    # Rejilla más suave
    "font.family": "serif",               # Fuente con serifas (clásico de tesis/paper)
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Nimbus Roman", "Georgia", "serif"],
    "font.size": 11,                      # Tamaño de letra ligeramente mayor
    "axes.titlesize": 13,                 # Títulos de gráficas
    "axes.titleweight": "bold",           # Títulos en negrita
    "axes.spines.top": False,             # Eliminar borde superior (moderno)
    "axes.spines.right": False,           # Eliminar borde derecho
    "legend.framealpha": 0.9,             # Leyenda semi-opaca
    "legend.edgecolor": "#bdc3c7",        # Borde de leyenda sutil
    "figure.dpi": 150,                    # Alta resolución en Jupyter (evita borrosidad)
    "savefig.dpi": 300,                   # Calidad de imprenta al exportar imagen
}

def _tensor_a_numpy(iq_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte tensor [2, N] a arrays separados I y Q."""
    if isinstance(iq_tensor, torch.Tensor):
        iq = iq_tensor.numpy()
    else:
        iq = iq_tensor
    return iq[0], iq[1]


def _eje_tiempo(n_muestras: int, fs: float = FREQ_MUESTREO) -> np.ndarray:
    """Genera eje temporal en segundos."""
    return np.arange(n_muestras) / fs  # segundos


def _fmt_ms(x, _):
    """Formateador de ejes: convierte segundos a milisegundos para display."""
    return f"{x * 1000:.1f}"


def _fmt_mhz(y, _):
    """Formateador de ejes: convierte Hz a MHz para display."""
    return f"{y / 1e6:.1f}"


# ============================================================================
# PANEL COMPLETO DE VISUALIZACIÓN
# ============================================================================

def panel_completo(
    iq_tensor: torch.Tensor,
    target: int,
    snr: int,
    fs: float = FREQ_MUESTREO,
    sample_id: Optional[int] = None,
    nfft: int = 1024,
    figsize: Tuple[float, float] = (16, 13),
) -> plt.Figure:
    """
    Genera un panel de visualización con 4 subplots:
        1. Señal I/Q (componentes Re e Im en el dominio del tiempo)
        2. Envolvente / Magnitud |I + jQ| (búsqueda de ráfagas)
        3. Espectrograma STFT 2D (tiempo-frecuencia)
        4. Densidad Espectral de Potencia (PSD - Welch)

    Args:
        iq_tensor: Tensor [2, N] con canales I y Q.
        target: Índice de clase (0-6).
        snr: SNR en dB.
        fs: Frecuencia de muestreo (default 14 MHz).
        sample_id: ID numérico de la muestra (opcional, para el título).
        nfft: Tamaño de la FFT para espectrograma y PSD (default 1024).
        figsize: Tamaño de la figura.

    Returns:
        Objeto matplotlib Figure con los 4 subplots.
    """
    I, Q = _tensor_a_numpy(iq_tensor)
    n = len(I)
    t_sec = _eje_tiempo(n, fs)  # Eje temporal en SEGUNDOS (unidad nativa del specgram)
    signal_complex = I + 1j * Q

    nombre_clase = NOMBRES_CLASES.get(target, f"Clase {target}")
    if target == TARGET_NOISE:
        titulo = f"Ruido de Fondo — SNR = {snr} dB"
    else:
        titulo = f"Señal de Dron ({nombre_clase}) — SNR = {snr} dB"

    with plt.rc_context(_ESTILO_PANEL):
        fig = plt.figure(figsize=figsize)
        # 3 columnas: Columna 0 (Time plots, width=40), Columna 1 (PSD, width=8), Columna 2 (Colorbar, width=1)
        gs = gridspec.GridSpec(
            3, 3,
            width_ratios=[40, 8, 1],
            height_ratios=[1, 1, 1.8],
            wspace=0.06,
            hspace=0.35,
        )
        fig.suptitle(titulo, fontsize=15, fontweight="bold", y=0.96)

        # ============================
        # 1. SEÑAL I/Q (TIEMPO)
        # ============================
        ax_iq = fig.add_subplot(gs[0, 0])

        # Submuestreo para visualización rápida (~10k puntos)
        step = max(1, n // 10_000)
        t_sub = t_sec[::step]
        I_sub = I[::step]
        Q_sub = Q[::step]

        ax_iq.plot(t_sub, I_sub, color="#3498db", linewidth=0.4, alpha=0.85, label="I (Re)")
        ax_iq.plot(t_sub, Q_sub, color="#e74c3c", linewidth=0.4, alpha=0.85, label="Q (Im)")
        ax_iq.set_ylabel("Amplitud")
        ax_iq.set_title("1. Señal I/Q — Componentes en Fase y Cuadratura", fontsize=11)
        ax_iq.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax_iq.grid(True)
        ax_iq.set_xlim([0, t_sec[-1]])
        ax_iq.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_ms))
        ax_iq.tick_params(labelbottom=False)

        # ============================
        # 2. ENVOLVENTE / MAGNITUD
        # ============================
        ax_env = fig.add_subplot(gs[1, 0], sharex=ax_iq)
        magnitud = np.abs(signal_complex)
        mag_sub = magnitud[::step]

        # Media móvil para suavizar la envolvente
        ventana = min(256, len(mag_sub) // 10)
        if ventana > 1:
            kernel = np.ones(ventana) / ventana
            mag_suavizada = np.convolve(mag_sub, kernel, mode="same")
            ax_env.plot(t_sub, mag_suavizada, color="#f39c12", linewidth=1.2, 
                       label=f"Envolvente suavizada (ventana={ventana} muestras)", zorder=3)

        ax_env.plot(t_sub, mag_sub, color="#2ecc71", linewidth=0.3, label="|I+jQ|")

        ax_env.set_ylabel("Magnitud")
        ax_env.set_title("2. Envolvente Temporal", fontsize=11)
        ax_env.legend(loc="upper right", fontsize=8)
        ax_env.grid(True)
        ax_env.tick_params(labelbottom=False)

        # ============================
        # 3. ESPECTROGRAMA STFT
        # ============================
        # specgram con Fs genera eje x en SEGUNDOS → misma unidad que ax_iq/ax_env
        ax_spec = fig.add_subplot(gs[2, 0], sharex=ax_iq)

        Pxx, freqs, bins_t, im = ax_spec.specgram(
            signal_complex,
            NFFT=nfft,
            Fs=fs,
            noverlap=nfft // 2,
            cmap="viridis",
            scale="dB",
        )

        # Formatear etiquetas a ms y MHz
        ax_spec.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_ms))
        ax_spec.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mhz))
        ax_spec.set_xlabel("Tiempo (ms)")
        ax_spec.set_ylabel("Frecuencia (MHz)")
        ax_spec.set_title("3. Espectrograma STFT — Evolución Tiempo-Frecuencia", fontsize=11)

        # ============================
        # 4. PSD (Welch) — ALINEADA POR FRECUENCIA
        # ============================
        ax_psd = fig.add_subplot(gs[2, 1], sharey=ax_spec)
        
        # Calcular PSD usando Welch para poder rotar los ejes y alinear con el espectrograma
        f_welch, Pxx_den = signal.welch(signal_complex, fs=fs, nperseg=nfft, return_onesided=False)
        f_welch = np.fft.fftshift(f_welch)
        Pxx_den = np.fft.fftshift(Pxx_den)
        Pxx_db = 10 * np.log10(Pxx_den + 1e-12)

        # Pintar la PSD: Potencia en eje X y Frecuencia en eje Y (evitando el color morado '#9b59b6' y usando carbón '#333333' neutro)
        ax_psd.plot(Pxx_db, f_welch, color="#333333", linewidth=1.2)
        ax_psd.set_xlabel("PSD (dB/Hz)")
        ax_psd.set_title("4. PSD (Welch)", fontsize=11)
        ax_psd.grid(True)

        # Configurar límites y ticks compartidos de frecuencia
        ax_psd.set_ylim([-fs/2, fs/2])
        ax_psd.set_xlim([Pxx_db.min() - 5, Pxx_db.max() + 5])
        plt.setp(ax_psd.get_yticklabels(), visible=False) # Ocultar etiquetas de frecuencia redundantes

        # ============================
        # 5. COLORBAR
        # ============================
        cax = fig.add_subplot(gs[2, 2])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Potencia (dB/Hz)", fontsize=9)

        # Primero empaquetamos todo para que las gráficas no colisionen entre sí
        fig.tight_layout()
        # Luego bajamos el techo del layout un 8% para que el suptitle encaje perfecto
        fig.subplots_adjust(top=0.92)

    return fig


# ============================================================================
# COMPARACIÓN POR NIVELES DE SNR
# ============================================================================

def comparar_snr(
    data_dir: str = DATA_DIR,
    target: int = 4,
    snr_list: Optional[List[int]] = None,
    nfft: int = 1024,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Genera una cuadrícula comparativa mostrando la misma clase de dron
    a distintos niveles de SNR. Cada fila muestra: espectrograma + envolvente.

    Args:
        data_dir: Directorio del dataset.
        target: Índice de clase a comparar (0-6).
        snr_list: Lista de SNRs a mostrar. Default: [-20, -10, 0, 10, 20, 30].
        nfft: Tamaño de FFT.
        figsize: Tamaño de la figura.

    Returns:
        Objeto matplotlib Figure.
    """
    if snr_list is None:
        snr_list = [-20, -10, 0, 10, 20, 30]

    n_rows = len(snr_list)
    if figsize is None:
        figsize = (16, 3.2 * n_rows)

    nombre_clase = "Ruido de Fondo" if target == TARGET_NOISE else NOMBRES_CLASES.get(target, f"Clase {target}")

    with plt.rc_context(_ESTILO_PANEL):
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        fig.suptitle(
            f"Comparación por SNR — {nombre_clase}",
            fontsize=15,
            fontweight="bold",
            y=0.94
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, snr_val in enumerate(snr_list):
            try:
                iq, sid, tgt, snr_real = obtener_una_muestra(data_dir, target, snr_val)
                I, Q = _tensor_a_numpy(iq)
                signal_complex = I + 1j * Q
                t_sec = _eje_tiempo(len(I))

                # Columna 0: Espectrograma
                ax_spec = axes[i, 0]
                ax_spec.specgram(
                    signal_complex, NFFT=nfft, Fs=FREQ_MUESTREO,
                    noverlap=nfft // 2, cmap="viridis", scale="dB",
                )
                ax_spec.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_ms))
                ax_spec.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mhz))
                ax_spec.set_ylabel(f"SNR={snr_val}dB\nFreq (MHz)", fontsize=9)
                if i == n_rows - 1:
                    ax_spec.set_xlabel("Tiempo (ms)")
                if i == 0:
                    ax_spec.set_title("Espectrograma STFT", fontsize=11)

                # Columna 1: Envolvente
                ax_env = axes[i, 1]
                magnitud = np.abs(signal_complex)
                step = max(1, len(magnitud) // 5_000)
                ax_env.plot(
                    t_sec[::step], magnitud[::step],
                    color="#2ecc71", linewidth=0.4,
                )
                # Envolvente suavizada
                ventana = min(128, len(magnitud[::step]) // 10)
                if ventana > 1:
                    kernel = np.ones(ventana) / ventana
                    suavizada = np.convolve(magnitud[::step], kernel, mode="same")
                    ax_env.plot(t_sec[::step], suavizada, color="#f39c12", linewidth=1.0)
                ax_env.set_xlim([0, t_sec[-1]])
                ax_env.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_ms))
                ax_env.grid(True)
                if i == n_rows - 1:
                    ax_env.set_xlabel("Tiempo (ms)")
                if i == 0:
                    ax_env.set_title("Envolvente |I+jQ|", fontsize=11)

            except (FileNotFoundError, IndexError):
                axes[i, 0].text(
                    0.5, 0.5, f"No disponible\nSNR={snr_val} dB",
                    ha="center", va="center", fontsize=12, transform=axes[i, 0].transAxes,
                )
                axes[i, 1].text(
                    0.5, 0.5, f"No disponible",
                    ha="center", va="center", fontsize=12, transform=axes[i, 1].transAxes,
                )

        fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.5)

    return fig


# ============================================================================
# COMPARACIÓN POR CLASES
# ============================================================================

def comparar_clases(
    data_dir: str = DATA_DIR,
    snr: int = 10,
    targets: Optional[List[int]] = None,
    nfft: int = 1024,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Genera una cuadrícula comparativa mostrando distintas clases al mismo SNR.
    Cada fila: una clase. Columnas: espectrograma + envolvente.

    Args:
        data_dir: Directorio del dataset.
        snr: Nivel de SNR fijo para la comparación.
        targets: Lista de targets a mostrar. Default: todas las clases (0-6).
        nfft: Tamaño de FFT.
        figsize: Tamaño de la figura.

    Returns:
        Objeto matplotlib Figure.
    """
    if targets is None:
        targets = list(range(7))

    n_rows = len(targets)
    if figsize is None:
        figsize = (16, 3.2 * n_rows)

    with plt.rc_context(_ESTILO_PANEL):
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        fig.suptitle(
            f"Comparación por Clase — SNR = {snr} dB",
            fontsize=15,
            fontweight="bold",
            y=0.92
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, tgt in enumerate(targets):
            nombre = NOMBRES_CLASES.get(tgt, f"Clase {tgt}")
            label_y = "Ruido de Fondo" if tgt == TARGET_NOISE else f"Dron: {nombre}"
            color = _COLORES_CLASES.get(tgt, "#ffffff")

            try:
                iq, sid, _, snr_real = obtener_una_muestra(data_dir, tgt, snr)
                I, Q = _tensor_a_numpy(iq)
                signal_complex = I + 1j * Q
                t_sec = _eje_tiempo(len(I))

                # Columna 0: Espectrograma
                ax_spec = axes[i, 0]
                ax_spec.specgram(
                    signal_complex, NFFT=nfft, Fs=FREQ_MUESTREO,
                    noverlap=nfft // 2, cmap="viridis", scale="dB",
                )
                ax_spec.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_ms))
                ax_spec.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mhz))
                ax_spec.set_ylabel(f"{label_y}\nFreq (MHz)", fontsize=9)
                if i == n_rows - 1:
                    ax_spec.set_xlabel("Tiempo (ms)")
                if i == 0:
                    ax_spec.set_title("Espectrograma STFT", fontsize=11)

                # Columna 1: Envolvente
                ax_env = axes[i, 1]
                magnitud = np.abs(signal_complex)
                step = max(1, len(magnitud) // 5_000)
                ax_env.plot(
                    t_sec[::step], magnitud[::step],
                    color=color, linewidth=0.4
                )
                ventana = min(128, len(magnitud[::step]) // 10)
                if ventana > 1:
                    kernel = np.ones(ventana) / ventana
                    suavizada = np.convolve(magnitud[::step], kernel, mode="same")
                    ax_env.plot(t_sec[::step], suavizada, color="#f39c12", linewidth=1.0)
                ax_env.set_xlim([0, t_sec[-1]])
                ax_env.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_ms))
                ax_env.grid(True)
                if i == n_rows - 1:
                    ax_env.set_xlabel("Tiempo (ms)")
                if i == 0:
                    ax_env.set_title("Envolvente |I+jQ|", fontsize=11)

            except (FileNotFoundError, IndexError):
                axes[i, 0].text(
                    0.5, 0.5, f"No disponible\n{nombre}",
                    ha="center", va="center", fontsize=12, transform=axes[i, 0].transAxes,
                )
                axes[i, 1].text(
                    0.5, 0.5, f"No disponible",
                    ha="center", va="center", fontsize=12, transform=axes[i, 1].transAxes,
                )

        # constrained_layout lo gestiona automáticamente

    return fig
