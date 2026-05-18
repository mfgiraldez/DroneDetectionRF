"""
xin_precompute_cache.py — Pre-cómputo de Espectrogramas para Caché en Disco
=============================================================================
Convierte todos los ficheros .pt del dataset NoisyUAV a tensores
[2, SPEC_H, SPEC_W] (log-PSD + Sobel) y los guarda en un directorio de caché.

Ventajas:
    - El DataLoader pasa de ~40-60 ms/muestra (CPU-bound STFT) a ~5 ms (lectura)
    - La GPU se satura correctamente durante el entrenamiento
    - La caché es reutilizable entre experimentos (el tensor es idéntico)
    - El proceso es REANUDABLE: los ficheros ya computados se omiten

Espacio estimado:
    17 744 muestras × [2, 256, 256] × float32 = ~8.9 GB

Uso:
    python -m NoisyUAV.modelo_xin_v1.xin_precompute_cache \\
        --data_dir  C:\\TFM_data\\NoisyUAV\\drone_RF_data \\
        --cache_dir C:\\TFM_data\\NoisyUAV\\xin_cache_256x256
"""

import os
import sys
import glob
import time
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from tqdm import tqdm

from NoisyUAV.modelo_xin_v1.xin_dataset import (
    iq_to_xin_tensor,
    NFFT, HOP_LENGTH, SPEC_H, SPEC_W, DB_CLIP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)


# ---------------------------------------------------------------------------- #
# Función principal de pre-cómputo                                             #
# ---------------------------------------------------------------------------- #

def precompute_spectrograms(
    data_dir: str,
    cache_dir: str,
    nfft: int = NFFT,
    hop_length: int = HOP_LENGTH,
    spec_h: int = SPEC_H,
    spec_w: int = SPEC_W,
    db_clip: float = DB_CLIP,
    overwrite: bool = False,
) -> None:
    """
    Itera sobre todos los ficheros .pt de `data_dir`, aplica el pipeline
    IQ -> STFT -> log-PSD -> Sobel y guarda el tensor resultante en `cache_dir`.

    Nomenclatura de ficheros en caché:
        data_dir/IQdata_sampleN_targetT_snrS.pt
            -> cache_dir/IQdata_sampleN_targetT_snrS.pt  (tensor [2, H, W])

    El proceso es atómico por fichero: se escribe a un fichero temporal
    y sólo se renombra si la escritura termina correctamente. Así, una
    interrupción a mitad del proceso no deja ficheros corruptos.

    Parámetros
    ----------
    data_dir   : Directorio con los ficheros IQdata_*.pt originales.
    cache_dir  : Directorio destino de la caché.
    overwrite  : Si True, recalcula incluso si el fichero ya existe.
    """
    os.makedirs(cache_dir, exist_ok=True)

    archivos = sorted(glob.glob(os.path.join(data_dir, "IQdata_*.pt")))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron ficheros .pt en: {data_dir}")

    total = len(archivos)
    logging.info(f"Total de ficheros a procesar: {total:,}")
    logging.info(f"Directorio de cache:          {cache_dir}")
    logging.info(f"Resolucion espectrograma:     [{spec_h} x {spec_w}]")
    logging.info(f"NFFT={nfft} | hop={hop_length} | db_clip={db_clip} dB")

    # Contar cuántos ya están computados
    ya_computados = sum(
        1 for f in archivos
        if os.path.exists(os.path.join(cache_dir, os.path.basename(f)))
    )
    if ya_computados > 0 and not overwrite:
        logging.info(
            f"Ficheros ya en cache: {ya_computados:,} / {total:,} "
            f"(se omitiran, usa --overwrite para recalcular)"
        )

    # Contadores de progreso
    n_ok = 0
    n_skip = 0
    n_err = 0
    t_start = time.time()
    tiempos = []   # Para estimar ETA

    pbar = tqdm(archivos, desc="Pre-computando espectrogramas", unit="muestras",
                dynamic_ncols=True)

    for fpath in pbar:
        fname = os.path.basename(fpath)
        out_path = os.path.join(cache_dir, fname)
        tmp_path = out_path + ".tmp"

        # Saltar si ya existe y no se fuerza overwrite
        if os.path.exists(out_path) and not overwrite:
            n_skip += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, refresh=False)
            continue

        t0 = time.time()
        try:
            # Carga del tensor IQ original
            d = torch.load(fpath, map_location="cpu", weights_only=False)
            iq = d["x_iq"].clone()   # [2, N_SAMPLES] — clone libera referencia

            # Transformada completa
            tensor = iq_to_xin_tensor(
                iq,
                nfft=nfft,
                hop_length=hop_length,
                spec_h=spec_h,
                spec_w=spec_w,
                db_clip=db_clip,
            )   # [2, spec_h, spec_w] float32

            # Escritura atómica (tmp -> final)
            torch.save(tensor, tmp_path)
            os.replace(tmp_path, out_path)

            n_ok += 1
            tiempos.append(time.time() - t0)

        except Exception as exc:
            # Limpiar fichero temporal si quedó a medias
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logging.warning(f"ERROR procesando {fname}: {exc}")
            n_err += 1

        # ETA estimada
        if tiempos:
            media = sum(tiempos) / len(tiempos)
            pendientes = total - n_ok - n_skip - n_err
            eta_s = media * pendientes
            eta_str = f"{eta_s/60:.1f} min"
        else:
            eta_str = "?"

        pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, eta=eta_str, refresh=False)

    pbar.close()

    # Informe final
    elapsed = time.time() - t_start
    logging.info("=" * 60)
    logging.info("PRE-COMPUTO FINALIZADO")
    logging.info(f"  Computados:  {n_ok:>6,}")
    logging.info(f"  Omitidos:    {n_skip:>6,}")
    logging.info(f"  Errores:     {n_err:>6,}")
    logging.info(f"  Tiempo total: {elapsed/60:.1f} min")

    if n_ok > 0:
        media_ms = (sum(tiempos) / len(tiempos)) * 1000
        logging.info(f"  Tiempo medio por muestra: {media_ms:.1f} ms")

    # Verificacion de integridad: tamanyo del directorio de cache
    cache_files = glob.glob(os.path.join(cache_dir, "IQdata_*.pt"))
    cache_size_gb = sum(os.path.getsize(f) for f in cache_files) / 1e9
    logging.info(f"  Ficheros en cache:  {len(cache_files):,} / {total:,}")
    logging.info(f"  Tamanyo total cache: {cache_size_gb:.2f} GB")
    logging.info("=" * 60)

    if n_err > 0:
        logging.warning(
            f"Se produjeron {n_err} error(es). Revisa los mensajes anteriores "
            f"y vuelve a ejecutar el script (los errores se recalcularan)."
        )


# ---------------------------------------------------------------------------- #
# Entry point                                                                  #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-computa espectrogramas STFT+Sobel y los guarda en cache."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\TFM_data\NoisyUAV\drone_RF_data",
        help="Directorio con los ficheros IQdata_*.pt originales.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=r"C:\TFM_data\NoisyUAV\xin_cache_256x256",
        help="Directorio de salida de la cache.",
    )
    parser.add_argument("--nfft",       type=int,   default=NFFT)
    parser.add_argument("--hop_length", type=int,   default=HOP_LENGTH)
    parser.add_argument("--spec_h",     type=int,   default=SPEC_H)
    parser.add_argument("--spec_w",     type=int,   default=SPEC_W)
    parser.add_argument("--db_clip",    type=float, default=DB_CLIP)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalcular ficheros ya existentes en la cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    precompute_spectrograms(
        data_dir   = args.data_dir,
        cache_dir  = args.cache_dir,
        nfft       = args.nfft,
        hop_length = args.hop_length,
        spec_h     = args.spec_h,
        spec_w     = args.spec_w,
        db_clip    = args.db_clip,
        overwrite  = args.overwrite,
    )


if __name__ == "__main__":
    main()
