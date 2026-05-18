"""
xin_dataset.py — Preprocesamiento y División Estratificada del Dataset
=======================================================================
Implementa el pipeline de transformación de señales I/Q a espectrogramas
complejos artificiales siguiendo la metodología de Xin et al. (2026):

    Canal Real      → log-PSD normalizado (STFT + rango dinámico 60 dB)
    Canal Imaginario → magnitud del gradiente Sobel (información estructural)

Normalización:
    1. IQ RMS: Señal normalizada a potencia unitaria antes de la STFT.
       (Lección aprendida Run #1 HybridCVCNN: crítico para evitar inestabilidad)
    2. log-PSD: Recorte a 60 dB de rango dinámico + normalización [0, 1].
    3. Sobel:   Normalización [0, 1] per-muestra.

División del dataset:
    - Estratificada por (target_multiclase × SNR_exacto), garantizando que
      TODOS los subgrupos (clase × SNR) tengan la misma proporción en
      train / val / test. Idéntica estrategia a preparar_splits.py.
    - Proporciones: 70% train | 15% val | 15% test.

Referencia de tamaño de ventana:
    Xin et al. usan ventanas de 50 ms a 100 MS/s (5 M muestras).
    Adaptación a nuestro dataset: 75 ms a 14 MS/s (1 048 576 muestras,
    señal completa, sin recorte).
"""

import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple

# ---------------------------------------------------------------------------- #
# Constantes del dataset NoisyUAV                                              #
# ---------------------------------------------------------------------------- #
FS_HZ       = 14_000_000      # Frecuencia de muestreo operativa [Hz]
N_SAMPLES   = 1_048_576       # Muestras por señal (~75 ms)
TARGET_NOISE = 4              # Clase ruido (AWGN/WiFi/BT)

# ---------------------------------------------------------------------------- #
# Parámetros STFT (fieles a Xin et al., adaptados a 14 MHz)                   #
# ---------------------------------------------------------------------------- #
NFFT        = 1024            # Puntos de la FFT (resolución frecuencial)
HOP_FRAC    = 0.5             # Solapamiento 50 % → hop_length = NFFT * 0.5
HOP_LENGTH  = int(NFFT * HOP_FRAC)   # = 512 muestras
WINDOW_TYPE = "hann"

# Rango dinámico: Xin et al. recortan 60 dB tras la conversión logarítmica.
DB_CLIP     = 60.0

# Resolución espacial de salida del espectrograma [freq_bins × time_bins].
# La red usa AdaptiveAvgPool2d en el último bloque; este resize explícito
# reduce la carga de memoria de la GPU a un nivel manejable.
# Raw STFT: [1024, ~2048] → resize → [SPEC_H, SPEC_W]
SPEC_H      = 256             # Eje frecuencial (bins de frecuencia)
SPEC_W      = 256             # Eje temporal   (frames STFT)

# ---------------------------------------------------------------------------- #
# Utilidades de parseo de nombres de fichero                                   #
# ---------------------------------------------------------------------------- #
_FNAME_RE = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")


def parse_filename(filename: str) -> Tuple[int, int, int]:
    """Extrae (sample_id, target, snr) del nombre de fichero NoisyUAV."""
    m = _FNAME_RE.match(os.path.basename(filename))
    if m is None:
        raise ValueError(f"Nombre de fichero no reconocido: {filename}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def snr_group(snr: int) -> str:
    """Asigna grupo SNR (A/B/C) según la convención del TFM."""
    if snr >= 10:
        return "A"
    elif snr >= -6:
        return "B"
    else:
        return "C"


# ---------------------------------------------------------------------------- #
# Construcción y validación del CSV de splits                                  #
# ---------------------------------------------------------------------------- #

def build_stratified_split_csv(
    data_dir: str,
    out_csv: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Escanea data_dir, construye metadatos por fichero y genera una división
    estratificada Train/Val/Test guardada en out_csv.

    La clave de estratificación es (target_multiclase × SNR_exacto), lo que
    garantiza que cada subgrupo esté representado proporcionalmente en los
    tres conjuntos. Esta es la estrategia más rigurosa disponible para el
    dataset NoisyUAV (Glüge et al., 2024).

    Parámetros
    ----------
    data_dir  : Directorio con los ficheros .pt del dataset.
    out_csv   : Ruta de salida para el CSV de metadatos.
    val_ratio : Fracción global para validación.
    test_ratio: Fracción global para test.
    seed      : Semilla de aleatoriedad reproducible.

    Retorna
    -------
    pd.DataFrame con columnas: filename, filepath, sample_id, target,
        snr, is_drone, snr_group, split.
    """
    logging.info(f"Escaneando dataset en: {data_dir}")
    archivos = sorted(glob.glob(os.path.join(data_dir, "IQdata_*.pt")))

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron ficheros .pt en {data_dir}")

    rows = []
    skipped = 0
    for fpath in archivos:
        fname = os.path.basename(fpath)
        try:
            sid, target, snr = parse_filename(fname)
        except ValueError:
            skipped += 1
            continue
        rows.append({
            "filename":   fname,
            "filepath":   fpath,
            "sample_id":  sid,
            "target":     target,
            "snr":        snr,
            "is_drone":   0 if target == TARGET_NOISE else 1,
            "snr_group":  snr_group(snr),
        })

    if skipped > 0:
        logging.warning(f"{skipped} fichero(s) omitidos por nombre no reconocido.")

    df = pd.DataFrame(rows)
    logging.info(f"Total ficheros válidos: {len(df)}")

    # Clave de estratificación exacta: (target_multiclase, SNR)
    df["_strat"] = df["target"].astype(str) + "_" + df["snr"].astype(str)

    # División en dos pasos para respetar proporciones exactas
    df_trainval, df_test = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["_strat"]
    )
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adj,
        random_state=seed,
        stratify=df_trainval["_strat"],
    )

    df.loc[df_train.index, "split"] = "train"
    df.loc[df_val.index,   "split"] = "val"
    df.loc[df_test.index,  "split"] = "test"
    df = df.drop(columns=["_strat"])

    # ── Sanity checks ───────────────────────────────────────────────────────
    _print_dataset_stats(df)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    logging.info(f"CSV de splits guardado en: {out_csv}")
    return df


def _print_dataset_stats(df: pd.DataFrame) -> None:
    """Imprime estadísticas de distribución para verificación científica."""
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE DISTRIBUCIÓN DEL DATASET")
    print("=" * 60)

    # Distribución global por split
    split_counts = df["split"].value_counts().sort_index()
    print(f"\n{'Split':<10} {'N':>8} {'%':>8}")
    print("-" * 30)
    for s, n in split_counts.items():
        print(f"{s:<10} {n:>8,} {100*n/len(df):>7.1f}%")

    # Distribución de clase binaria por split
    print(f"\n{'Split':<10} {'Drones':>10} {'Ruido':>10} {'% Dron':>10}")
    print("-" * 44)
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        n_drone = (sub["is_drone"] == 1).sum()
        n_noise = (sub["is_drone"] == 0).sum()
        print(f"{s:<10} {n_drone:>10,} {n_noise:>10,} {100*n_drone/len(sub):>9.1f}%")

    # Distribución por grupo SNR y split
    print(f"\nDistribución por Grupo SNR:")
    pivot = df.groupby(["split", "snr_group"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    # Distribución de targets de dron por split
    print(f"\nDistribución de targets de dron (target ≠ {TARGET_NOISE}) por split:")
    drones = df[df["is_drone"] == 1]
    pivot2 = drones.groupby(["split", "target"]).size().unstack(fill_value=0)
    print(pivot2.to_string())
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------- #
# Transformadas de señal: IQ → tensor complejo 2D                             #
# ---------------------------------------------------------------------------- #

def iq_to_xin_tensor(
    iq: torch.Tensor,
    nfft: int = NFFT,
    hop_length: int = HOP_LENGTH,
    spec_h: int = SPEC_H,
    spec_w: int = SPEC_W,
    db_clip: float = DB_CLIP,
) -> torch.Tensor:
    """
    Convierte una señal I/Q completa [2, N] al tensor de entrada de la red:
    [2, SPEC_H, SPEC_W] donde:
        Canal 0 (Real)      = log-PSD normalizado [0, 1]
        Canal 1 (Imaginario) = magnitud gradiente Sobel normalizada [0, 1]

    Pipeline:
        1. Normalización RMS del vector I/Q (potencia unitaria).
        2. STFT compleja con ventana Hann (nfft puntos, 50 % solapamiento).
        3. Densidad espectral de potencia logarítmica (10·log10).
        4. Recorte de rango dinámico a `db_clip` dB (Xin et al.: 60 dB).
        5. Normalización min-max del espectrograma a [0, 1].
        6. Resize bilineal a [spec_h, spec_w].
        7. Detección de bordes Sobel 3×3 sobre el espectrograma normalizado.
        8. Normalización min-max del mapa de bordes a [0, 1].
        9. Ensamble del tensor complejo [2, H, W].

    Parámetros
    ----------
    iq        : Tensor [2, N] con componentes I (canal 0) y Q (canal 1).
    nfft      : Número de puntos de la FFT.
    hop_length: Desplazamiento entre tramas STFT.
    spec_h    : Alto de salida del espectrograma (eje frecuencial).
    spec_w    : Ancho de salida del espectrograma (eje temporal).
    db_clip   : Rango dinámico máximo en dB.

    Retorna
    -------
    torch.Tensor de forma [2, spec_h, spec_w] y dtype float32.
    """
    # ── 1. Normalización RMS ────────────────────────────────────────────────
    rms = iq.pow(2).mean().clamp(min=1e-12).sqrt()
    iq = iq / rms

    # ── 2. STFT compleja ────────────────────────────────────────────────────
    # Se forma la señal compleja combinando I+jQ para obtener el espectro
    # bilateral completo (1024 bins), no el espectro unilateral de señal real.
    sig_complex = torch.complex(iq[0], iq[1])   # [N]
    window = torch.hann_window(nfft, device=iq.device)
    stft = torch.stft(
        sig_complex,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=nfft,
        window=window,
        center=False,       # Evita padding implícito que distorsiona bordes
        return_complex=True,
        onesided=False,     # Espectro bilateral completo para señal compleja
    )   # shape: [nfft, T_frames] complejo

    # ── 3. PSD logarítmica ──────────────────────────────────────────────────
    psd_linear = stft.abs().pow(2)                      # [F, T]
    psd_db = 10.0 * torch.log10(psd_linear + 1e-12)    # [F, T] en dB

    # ── 4. Recorte de rango dinámico (Xin et al.: 60 dB) ───────────────────
    psd_max = psd_db.max()
    psd_db = psd_db.clamp(min=psd_max - db_clip)       # [psd_max-60, psd_max]

    # ── 5. Normalización min-max a [0, 1] ───────────────────────────────────
    psd_min = psd_db.min()
    psd_range = (psd_max - psd_min).clamp(min=1e-8)
    log_psd_norm = (psd_db - psd_min) / psd_range      # [0, 1]

    # ── 6. Resize bilineal a [spec_h, spec_w] ──────────────────────────────
    # F.interpolate requiere [B, C, H, W]
    log_psd_4d = log_psd_norm.unsqueeze(0).unsqueeze(0)   # [1, 1, F, T]
    log_psd_resized = F.interpolate(
        log_psd_4d,
        size=(spec_h, spec_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)   # [spec_h, spec_w]

    # ── 7. Operador Sobel 3×3 ───────────────────────────────────────────────
    sobel_img = _apply_sobel(log_psd_resized)   # [spec_h, spec_w]

    # ── 8. Normalización Sobel a [0, 1] ─────────────────────────────────────
    s_min = sobel_img.min()
    s_max = sobel_img.max()
    sobel_norm = (sobel_img - s_min) / (s_max - s_min + 1e-8)

    # ── 9. Tensor complejo [2, H, W] ────────────────────────────────────────
    return torch.stack([log_psd_resized, sobel_norm], dim=0)   # [2, H, W]


def _apply_sobel(img: torch.Tensor) -> torch.Tensor:
    """
    Aplica los kernels de Sobel 3×3 a una imagen 2D [H, W] y retorna
    la magnitud del gradiente espacial [H, W].

    Los kernels detectan discontinuidades en las dos direcciones
    espaciales del espectrograma: frecuencia (Gx) y tiempo (Gy).
    """
    Kx = torch.tensor(
        [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=img.device
    ).view(1, 1, 3, 3)
    Ky = torch.tensor(
        [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=img.device
    ).view(1, 1, 3, 3)

    img_4d = img.unsqueeze(0).unsqueeze(0)              # [1, 1, H, W]
    img_pad = F.pad(img_4d, (1, 1, 1, 1), mode="reflect")

    Gx = F.conv2d(img_pad, Kx)
    Gy = F.conv2d(img_pad, Ky)
    mag = torch.sqrt(Gx.pow(2) + Gy.pow(2) + 1e-8)
    return mag.squeeze(0).squeeze(0)                    # [H, W]


# ---------------------------------------------------------------------------- #
# Clase Dataset de PyTorch                                                     #
# ---------------------------------------------------------------------------- #

class XinSpectrogramDataset(Dataset):
    """
    Dataset para la arquitectura CV-CNN 2D (Xin et al., 2026).

    Soporta dos modos de operación:

    Modo on-the-fly (cache_dir=None):
        Carga la señal I/Q desde el .pt original y calcula la transformada
        STFT+Sobel en cada acceso. Flexible pero lento (~50 ms/muestra CPU).

    Modo caché (cache_dir=<ruta>):
        Carga directamente el tensor [2, H, W] pre-computado por
        xin_precompute_cache.py. Reduce la latencia de acceso a ~5 ms
        y satura correctamente la GPU durante el entrenamiento.
        PREREQUISITO: ejecutar xin_precompute_cache.py previamente.

    Parámetros
    ----------
    csv_path  : Ruta al CSV generado por build_stratified_split_csv().
    split     : "train" | "val" | "test".
    cache_dir : Directorio de la caché pre-computada. None = on-the-fly.
    nfft, hop_length, spec_h, spec_w, db_clip : Parámetros de transformación
        (solo relevantes en modo on-the-fly; ignorados en modo caché).
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        cache_dir: str = None,
        nfft: int = NFFT,
        hop_length: int = HOP_LENGTH,
        spec_h: int = SPEC_H,
        spec_w: int = SPEC_W,
        db_clip: float = DB_CLIP,
    ):
        df = pd.read_csv(csv_path)
        self.data       = df[df["split"] == split].reset_index(drop=True)
        self.cache_dir  = cache_dir
        self.nfft       = nfft
        self.hop_length = hop_length
        self.spec_h     = spec_h
        self.spec_w     = spec_w
        self.db_clip    = db_clip

        if len(self.data) == 0:
            raise ValueError(f"El CSV no contiene muestras para split='{split}'.")

        modo = f"CACHE ({cache_dir})" if cache_dir else "on-the-fly"
        logging.info(
            f"XinSpectrogramDataset [{split}] | modo={modo} | "
            f"{len(self.data):,} muestras "
            f"({self.data['is_drone'].sum()} drones / "
            f"{(self.data['is_drone']==0).sum()} ruido)"
        )

        # Verificación rápida de integridad de caché
        if cache_dir is not None:
            self._verify_cache()

    def _verify_cache(self) -> None:
        """Comprueba que todos los ficheros de este split existen en la caché."""
        missing = [
            row["filename"]
            for _, row in self.data.iterrows()
            if not os.path.exists(os.path.join(self.cache_dir, row["filename"]))
        ]
        if missing:
            raise FileNotFoundError(
                f"Faltan {len(missing)} ficheros en la cache '{self.cache_dir}'.\n"
                f"Ejecuta xin_precompute_cache.py antes de entrenar.\n"
                f"Primer fichero ausente: {missing[0]}"
            )
        logging.info(f"  Cache verificada: {len(self.data):,} ficheros presentes.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        label = torch.tensor(row["is_drone"], dtype=torch.float32).unsqueeze(0)

        # ── Modo caché: carga directa del tensor pre-computado ──────────────
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, row["filename"])
            xin_tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
            return xin_tensor, label

        # ── Modo on-the-fly: cálculo completo desde la señal I/Q ───────────
        d = torch.load(row["filepath"], map_location="cpu", weights_only=False)
        iq = d["x_iq"].clone()   # clone() libera referencia al tensor original

        xin_tensor = iq_to_xin_tensor(
            iq,
            nfft=self.nfft,
            hop_length=self.hop_length,
            spec_h=self.spec_h,
            spec_w=self.spec_w,
            db_clip=self.db_clip,
        )   # [2, spec_h, spec_w]

        return xin_tensor, label
