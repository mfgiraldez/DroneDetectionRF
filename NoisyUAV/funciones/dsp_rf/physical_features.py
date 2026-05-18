"""
physical_features.py
====================
Extracción de 12 features físicas del detector de entropía Shannon
para alimentar el brazo de features del HybridCVCNN.

API pública:
    extract_features(iq_tensor, fs, nperseg) -> np.array [12]
    build_features_cache(filepaths, cache_path, ...)
    load_features_cache(cache_path) -> dict

Dimensiones del vector de features (FEATURES_DIM = 12):
    Global (siempre disponibles, incluso a SNR muy bajo):
        0: noise_floor    — piso de ruido CFAR mediano (bits)
        1: noise_sigma    — dispersión MAD escalada del piso de ruido
        2: mean_n_active  — bins espectrales activos medios (discrimina energía total)
        3: p75_n_active   — P75 de bins activos (discrimina WiFi OFDM. WiFi >> 512)
        4: H_min          — mínimo de entropía whitened (concentración espectral máxima)
        5: H_mean         — entropía media de la muestra (proxy del piso de ruido)

    Burst (0.0 si no se detecta ningún burst):
        6:  n_bursts       — número de bursts FHSS detectados
        7:  dur_ms_main    — duración del burst más significativo (ms)
        8:  z_peak_main    — significancia estadística del burst (z-score, clamp 30)
        9:  drop_b_main    — caída de entropía en el burst (bits, clamp 10)
        10: n_act_main     — bins activos en el núcleo del burst (FHSS ~58-200; WiFi ~512)
        11: dur_total      — duración total de toda la actividad detectada (ms, clamp 75)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict

from NoisyUAV.funciones.detector_entropia import detectar_bursts

# ─────────────────────────────────────────────────────────────────────────────
FEATURES_DIM = 12
FS_DEFAULT   = 14e6
NPERSEG_DEFAULT = 2048

FEATURE_NAMES = [
    "noise_floor",
    "noise_sigma",
    "mean_n_active",
    "p75_n_active",
    "H_min",
    "H_mean",
    "n_bursts",
    "dur_ms_main",
    "z_peak_main",
    "drop_b_main",
    "n_act_main",
    "dur_total",
]

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    iq_tensor,
    fs: float = FS_DEFAULT,
    nperseg: int = NPERSEG_DEFAULT,
) -> np.ndarray:
    """
    Extrae FEATURES_DIM=12 features físicas de un tensor IQ completo.

    Parámetros
    ----------
    iq_tensor : Tensor o array [2, N]  — canal 0=I, canal 1=Q
    fs        : frecuencia de muestreo (Hz)
    nperseg   : ventana STFT para el detector de entropía

    Retorna
    -------
    features : np.array [12] dtype=float32
        Valores normalizados a escala plausible; NaN/Inf reemplazados por 0.
    """
    try:
        _, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
            iq_tensor,
            fs=fs,
            nperseg=nperseg,
            adaptive_window_ms=15.0,
            min_burst_ms=0.3,       # ligeramente más sensible que el default
            z_thresh=2.5,           # ligeramente más sensible para SNR bajo
            merge_gap_ms=1.5,       # fusiona hops cercanos como un burst
            bg_mult=4.0,
            max_bins_frac=0.25,
        )

        # ── Features globales ──────────────────────────────────────────────
        f = [
            float(np.median(nf_v)),              # 0: noise_floor
            float(np.clip(ns, 0, 5)),            # 1: noise_sigma (bit)
            float(np.mean(n_active)),             # 2: mean_n_active
            float(np.percentile(n_active, 75)),   # 3: p75_n_active
            float(np.min(H_smooth)),              # 4: H_min
            float(np.mean(H_smooth)),             # 5: H_mean
        ]

        # ── Features de burst ─────────────────────────────────────────────
        if bursts:
            b0 = max(bursts, key=lambda b: abs(b['z_peak']))
            f += [
                float(min(len(bursts), 50)),                           # 6: n_bursts
                float(np.clip(b0['dur_ms'], 0, 75)),                   # 7: dur_ms_main
                float(np.clip(abs(b0['z_peak']), 0, 30)),             # 8: z_peak_main
                float(np.clip(b0['drop_b'], 0, 10)),                  # 9: drop_b_main
                float(np.clip(b0['n_act'], 0, 2048)),                 # 10: n_act_main
                float(np.clip(sum(b['dur_ms'] for b in bursts), 0, 75)),  # 11: dur_total
            ]
        else:
            f += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    except Exception:
        f = [0.0] * FEATURES_DIM

    result = np.array(f, dtype=np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE CACHÉ
# ─────────────────────────────────────────────────────────────────────────────

def build_features_cache(
    filepaths: List[str],
    cache_path: str,
    fs: float = FS_DEFAULT,
    nperseg: int = NPERSEG_DEFAULT,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Construye y persiste la caché de features físicas para un conjunto de archivos.

    La caché es un archivo .npz donde cada clave es el nombre base del archivo
    (sin ruta, para ser agnóstico al OS) y el valor es el array [12] de features.

    Si la caché ya existe se carga y se extiende incrementalmente.
    Los archivos ya procesados se saltan (no se recomputar).

    Parámetros
    ----------
    filepaths  : lista de rutas absolutas a archivos .pt del dataset
    cache_path : ruta de salida del archivo .npz
    verbose    : imprimir progreso

    Retorna
    -------
    cache : dict {filename_key: np.array [12]}
    """
    cache_path = Path(cache_path)

    # Cargar caché existente si hay
    cache: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            cache = {str(k): data[k] for k in data.files}
            if verbose:
                print(f"  [Cache] Loaded {len(cache)} existing entries from {cache_path.name}")
        except Exception as e:
            if verbose:
                print(f"  [Cache] Could not load existing cache: {e}. Starting fresh.")

    new_count = 0
    for i, fp in enumerate(filepaths):
        key = Path(fp).name
        if key in cache:
            continue  # ya procesado

        try:
            d = torch.load(fp, map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
            feats = extract_features(iq, fs=fs, nperseg=nperseg)
        except Exception as e:
            feats = np.zeros(FEATURES_DIM, dtype=np.float32)

        cache[key] = feats
        new_count += 1

        if verbose and (new_count % 200 == 0 or i == len(filepaths) - 1):
            print(f"  [Cache] {i+1}/{len(filepaths)} | new: {new_count} | "
                  f"elapsed: {new_count} files")

    # Guardar
    np.savez_compressed(cache_path, **cache)
    if verbose:
        print(f"  [Cache] Saved {len(cache)} entries -> {cache_path}")

    return cache


def load_features_cache(cache_path: str) -> Dict[str, np.ndarray]:
    """
    Carga la caché de features desde disco.

    Retorna
    -------
    cache : dict {filename_key: np.array [12]}

    Raises
    ------
    FileNotFoundError si el archivo no existe.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {cache_path}\n"
            "Run build_features_cache() first."
        )
    data = np.load(cache_path, allow_pickle=False)
    return {str(k): data[k] for k in data.files}


def compute_normalization_stats(
    cache: Dict[str, np.ndarray],
    filepaths: List[str],
) -> tuple:
    """
    Calcula media y desviación estándar por feature sobre el conjunto de entrenamiento.
    Estas estadísticas se usan para normalizar las features antes de pasarlas al modelo.

    Retorna
    -------
    (mean, std) : tuple de np.array [12]
    """
    keys = [Path(fp).name for fp in filepaths if Path(fp).name in cache]
    if not keys:
        return np.zeros(FEATURES_DIM), np.ones(FEATURES_DIM)

    matrix = np.stack([cache[k] for k in keys])  # [N, 12]
    mean = matrix.mean(axis=0).astype(np.float32)
    std  = matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0   # evitar división por cero en features constantes
    return mean, std
