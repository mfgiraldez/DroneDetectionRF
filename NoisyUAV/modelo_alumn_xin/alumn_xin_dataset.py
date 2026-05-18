"""
alumn_xin_dataset.py — Dataset hibrido para AlumnXin (Espectrograma 2D + Features Fisicas)
============================================================================================
Carga el CSV de pseudo-labels del Alumno V1 (alumn_dataset_pseudo_v3.csv) y para
cada muestra genera:
  - El espectrograma complejo 2D [2, 256, 256] (log-PSD + Sobel) sobre el burst
    recortado segun t_start/t_end del CSV.
  - Las 8 features fisicas del burst (dur_ms, z_peak, ...) normalizadas por z-score.

A diferencia del XinSpectrogramDataset que calcula la STFT sobre la señal completa
de 75ms, aqui la STFT se calcula sobre el CROP del burst (t_start -> t_end, paddeado
a TARGET_LEN = 131072 muestras). Esto preserva la granularidad del burst y la
coherencia con el pipeline del Alumno.

Modos de operacion:
  - on-the-fly: calcula STFT+Sobel en cada acceso (lento, ~150ms/muestra).
  - cache (cache_dir): carga tensor [2, 256, 256] pre-computado (rapido, ~5ms).
    Usar alumn_xin_precompute_cache.py para generar la cache previamente.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------- #
# Constantes                                                                    #
# ---------------------------------------------------------------------------- #
FS          = 14_000_000        # Hz
TARGET_LEN  = 131_072           # ~9.4 ms @ 14 MHz  (igual que BurstCVCNN)
NFFT        = 1024
HOP_LENGTH  = 512               # 50% overlap
DB_CLIP     = 60.0
SPEC_H      = 256
SPEC_W      = 256

PHYS_COLS = [
    "dur_ms", "z_peak", "drop_b", "n_act_burst",
    "global_nf", "global_ns", "global_H_mean", "global_p75_act",
]


# ---------------------------------------------------------------------------- #
# Transformada IQ burst -> tensor [2, H, W]                                    #
# ---------------------------------------------------------------------------- #

def burst_iq_to_xin_tensor(
    iq_crop: torch.Tensor,
    nfft: int = NFFT,
    hop_length: int = HOP_LENGTH,
    spec_h: int = SPEC_H,
    spec_w: int = SPEC_W,
    db_clip: float = DB_CLIP,
) -> torch.Tensor:
    """
    Convierte un crop de burst IQ [2, L] al tensor de entrada [2, spec_h, spec_w].

    Canal 0: log-PSD normalizado [0, 1]
    Canal 1: magnitud gradiente Sobel [0, 1]
    """
    # 1. Normalización RMS
    rms = iq_crop.pow(2).mean().clamp(min=1e-12).sqrt()
    iq  = iq_crop / rms

    # 2. STFT compleja (bilateral, señal I+jQ)
    sig_complex = torch.complex(iq[0], iq[1])
    window = torch.hann_window(nfft, device=iq.device)
    stft = torch.stft(
        sig_complex, n_fft=nfft, hop_length=hop_length,
        win_length=nfft, window=window,
        center=False, return_complex=True, onesided=False,
    )   # [F, T]

    # 3. log-PSD
    psd_db = 10.0 * torch.log10(stft.abs().pow(2) + 1e-12)

    # 4. Recorte rango dinamico
    psd_max = psd_db.max()
    psd_db  = psd_db.clamp(min=psd_max - db_clip)

    # 5. Normalización min-max [0, 1]
    psd_min   = psd_db.min()
    log_psd   = (psd_db - psd_min) / (psd_max - psd_min).clamp(min=1e-8)

    # 6. Resize a [spec_h, spec_w]
    log_psd_r = F.interpolate(
        log_psd.unsqueeze(0).unsqueeze(0).float(),
        size=(spec_h, spec_w), mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)

    # 7. Sobel 3x3
    Kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                       device=iq.device).view(1, 1, 3, 3)
    Ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
                       device=iq.device).view(1, 1, 3, 3)
    img_4d  = log_psd_r.unsqueeze(0).unsqueeze(0)
    img_pad = F.pad(img_4d, (1, 1, 1, 1), mode="reflect")
    sobel   = torch.sqrt(F.conv2d(img_pad, Kx).pow(2) + F.conv2d(img_pad, Ky).pow(2) + 1e-8)
    sobel   = sobel.squeeze(0).squeeze(0)

    # 8. Normalización Sobel [0, 1]
    s_min, s_max = sobel.min(), sobel.max()
    sobel_norm   = (sobel - s_min) / (s_max - s_min + 1e-8)

    return torch.stack([log_psd_r, sobel_norm], dim=0)   # [2, H, W]


# ---------------------------------------------------------------------------- #
# Dataset                                                                       #
# ---------------------------------------------------------------------------- #

class AlumnXinDataset(Dataset):
    """
    Dataset hibrido Burst-STFT + Features Fisicas para AlumnXinCVCNN.

    Parametros
    ----------
    csv_df      : DataFrame ya filtrado por split (no el CSV entero).
    data_dir    : Directorio raiz de los ficheros .pt (IQdata_*.pt).
    cache_dir   : Si se especifica, carga tensores pre-computados [2, 256, 256].
                  Nombre de fichero: <burst_uid>.pt  (ver precompute_cache).
    phys_mean   : Tensor [8] para normalizar features. None = calcula desde datos.
    phys_std    : Tensor [8] para normalizar features. None = calcula desde datos.
    """

    def __init__(
        self,
        csv_df: pd.DataFrame,
        data_dir: str,
        cache_dir: Optional[str] = None,
        phys_mean: Optional[torch.Tensor] = None,
        phys_std:  Optional[torch.Tensor] = None,
    ):
        self.df       = csv_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Features fisicas
        phys_raw = torch.tensor(self.df[PHYS_COLS].values, dtype=torch.float32)
        if phys_mean is not None and phys_std is not None:
            self.phys_mean = phys_mean
            self.phys_std  = phys_std
        else:
            self.phys_mean = phys_raw.mean(dim=0)
            self.phys_std  = phys_raw.std(dim=0).clamp(min=1e-8)
        self.phys_feats = ((phys_raw - self.phys_mean) / self.phys_std).clamp(-5., 5.)

        if len(self.df) == 0:
            raise ValueError("El dataset esta vacio.")
        logging.info(
            f"AlumnXinDataset | {len(self.df):,} muestras | "
            f"cache={'SI' if cache_dir else 'NO (on-the-fly)'}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _burst_uid(self, row) -> str:
        """Identificador unico de burst para nombrar el fichero de cache."""
        fname = Path(row["file_path"]).stem
        return f"{fname}_burst{int(row['burst_id'])}"

    def __getitem__(self, idx: int) -> dict:
        row   = self.df.iloc[idx]
        label = torch.tensor(float(row["pseudo_label"]), dtype=torch.float32).unsqueeze(0)
        feats = self.phys_feats[idx]

        # ── Modo cache ──────────────────────────────────────────────────────
        if self.cache_dir is not None:
            uid        = self._burst_uid(row)
            cache_path = self.cache_dir / f"{uid}.pt"
            spec = torch.load(cache_path, map_location="cpu", weights_only=True)
            return {
                "spec":      spec,
                "feats":     feats,
                "label":     label,
                "snr":       float(row["snr"]),
                "target_mc": int(row["target_multiclass"]),
            }

        # ── Modo on-the-fly ──────────────────────────────────────────────────
        full_path = self.data_dir / row["file_path"]
        d   = torch.load(full_path, map_location="cpu", weights_only=False)
        iq  = d["x_iq"].float()
        L   = iq.shape[1]

        is_dummy = bool(row.get("fallback", False))
        if is_dummy:
            start_idx, end_idx = 0, L
        else:
            start_idx = max(0, int(float(row["t_start"]) / 1000.0 * FS))
            end_idx   = min(L, int(float(row["t_end"])   / 1000.0 * FS))
            if start_idx >= end_idx:
                end_idx = min(L, start_idx + 1024)

        iq_crop = iq[:, start_idx:end_idx]

        # Padding / truncado a TARGET_LEN
        cl = iq_crop.shape[1]
        if cl < TARGET_LEN:
            iq_crop = torch.cat([iq_crop, torch.zeros(2, TARGET_LEN - cl)], dim=1)
        else:
            iq_crop = iq_crop[:, :TARGET_LEN]

        iq_crop = iq_crop.clone()   # evita memory leak de vistas
        spec    = burst_iq_to_xin_tensor(iq_crop)

        return {
            "spec":      spec,
            "feats":     feats,
            "label":     label,
            "snr":       float(row["snr"]),
            "target_mc": int(row["target_multiclass"]),
        }


def collate_alumn_xin(batch):
    """Collate que conserva metadatos para evaluacion."""
    return {
        "spec":      torch.stack([b["spec"]  for b in batch]),
        "feats":     torch.stack([b["feats"] for b in batch]),
        "label":     torch.stack([b["label"] for b in batch]),
        "snr":       [b["snr"]       for b in batch],
        "target_mc": [b["target_mc"] for b in batch],
    }
