"""
alumn_xin_precompute_cache.py — Pre-computo de espectrogramas STFT+Sobel por burst
====================================================================================
Genera un fichero .pt por cada fila del CSV (por burst), conteniendo el tensor
[2, 256, 256] listo para cargar directamente en AlumnXinDataset (modo cache).

Esto reduce la latencia de acceso de ~150ms (STFT on-the-fly por burst) a ~5ms
(lectura de disco), permitiendo saturar la GPU durante el entrenamiento.

Uso:
    conda activate IAIAVv3
    python -m NoisyUAV.modelo_alumn_xin.alumn_xin_precompute_cache ^
        --csv    C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_alumn_xin\\alumn_xin_dataset.csv ^
        --data_dir C:\\TFM_data\\NoisyUAV\\drone_RF_data ^
        --cache_dir C:\\TFM_data\\NoisyUAV\\alumn_xin_cache_256x256

El proceso es reanudable: los ficheros ya existentes se saltan automaticamente.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import pandas as pd
from tqdm import tqdm
from NoisyUAV.modelo_alumn_xin.alumn_xin_dataset import (
    burst_iq_to_xin_tensor, FS, TARGET_LEN, NFFT, HOP_LENGTH, DB_CLIP, SPEC_H, SPEC_W
)

CSV_DEFAULT      = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_xin\alumn_xin_dataset.csv"
DATA_DIR_DEFAULT = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CACHE_DEFAULT    = r"C:\TFM_data\NoisyUAV\alumn_xin_cache_256x256"


def precompute(csv_path: str, data_dir: str, cache_dir: str) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    data_path  = Path(data_dir)

    df = pd.read_csv(csv_path)
    print(f"Total bursts en CSV: {len(df):,}")

    n_skipped = 0
    n_done    = 0
    n_error   = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-compute"):
        fname_stem = Path(row["file_path"]).stem
        uid        = f"{fname_stem}_burst{int(row['burst_id'])}"
        out_path   = cache_path / f"{uid}.pt"

        if out_path.exists():
            n_skipped += 1
            continue

        try:
            d  = torch.load(data_path / row["file_path"], map_location="cpu", weights_only=False)
            iq = d["x_iq"].float()
            L  = iq.shape[1]

            is_fallback = bool(row.get("fallback", False))
            if is_fallback:
                start_idx, end_idx = 0, L
            else:
                start_idx = max(0, int(float(row["t_start"]) / 1000.0 * FS))
                end_idx   = min(L, int(float(row["t_end"])   / 1000.0 * FS))
                if start_idx >= end_idx:
                    end_idx = min(L, start_idx + 1024)

            iq_crop = iq[:, start_idx:end_idx].clone()
            cl = iq_crop.shape[1]
            if cl < TARGET_LEN:
                iq_crop = torch.cat([iq_crop, torch.zeros(2, TARGET_LEN - cl)], dim=1)
            else:
                iq_crop = iq_crop[:, :TARGET_LEN]

            spec = burst_iq_to_xin_tensor(iq_crop)
            torch.save(spec, out_path)
            n_done += 1

        except Exception as e:
            print(f"\n  ERROR en {row['file_path']} burst {row['burst_id']}: {e}")
            n_error += 1

    print(f"\nCompletado: {n_done:,} nuevos | {n_skipped:,} ya existian | {n_error} errores")
    print(f"Cache en: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-computo STFT+Sobel para AlumnXin")
    parser.add_argument("--csv",       type=str, default=CSV_DEFAULT)
    parser.add_argument("--data_dir",  type=str, default=DATA_DIR_DEFAULT)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DEFAULT)
    args = parser.parse_args()
    precompute(args.csv, args.data_dir, args.cache_dir)


if __name__ == "__main__":
    main()
