"""
╔══════════════════════════════════════════════════════════════════╗
║   STAGE 1 — PREPROCESSING                                       ║
║   Convierte los 17.744 archivos .pt de NoisyUAV en un único      ║
║   dataset HDF5 compacto para entrenamiento rápido.               ║
║                                                                  ║
║   EJECUCIÓN: python preprocess_to_h5.py                         ║
║   TIEMPO ESTIMADO: 1-3 horas (depende de la velocidad del disco) ║
║   OUTPUT: detector_dataset.h5 (~9 GB en float16)                ║
╚══════════════════════════════════════════════════════════════════╝

Estrategia:
  - De cada archivo .pt [2, 1.048.576] extraemos WINDOWS_PER_FILE
    sub-ventanas de tamaño WINDOW_SIZE (1.17 ms @ 14 MHz) 
    distribuidas uniformemente + desplazamiento aleatorio.
  - Clase 4 (Noise) → etiqueta 0  |  Cualquier dron → etiqueta 1
  - Se guarda en float16 para reducir espacio a la mitad.
"""

import os
import glob
import random
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
#  CONFIG — modifica estas rutas
# ═══════════════════════════════════════════════════════════
DATA_DIR       = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUTPUT_FILE    = r"C:\TFM_data\detector_dataset.h5"

WINDOW_SIZE    = 16384   # muestras por sub-ventana (1.17 ms @ 14 MHz)
WINDOWS_PER_FILE = 8     # sub-ventanas a extraer por archivo .pt
NOISE_CLASS    = 4       # target=4 es la clase Ruido
SEED           = 42
# ═══════════════════════════════════════════════════════════

random.seed(SEED)
np.random.seed(SEED)


def parse_filename(path: str):
    """
    Extrae (target, snr) del nombre de archivo.
    Formato esperado: IQdata_sampleXXX_targetY_snrZ.pt
    """
    stem = Path(path).stem
    target, snr = None, None
    for part in stem.split('_'):
        if part.startswith('target'):
            target = int(part[6:])
        elif part.startswith('snr'):
            snr = int(part[3:])
    if target is None or snr is None:
        raise ValueError(f"No se pudo parsear target/snr de: {stem}")
    return target, snr


def extract_windows_uniform(x_iq: torch.Tensor,
                            n_windows: int,
                            window_size: int,
                            rng: random.Random) -> np.ndarray:
    """
    Extrae n_windows sub-ventanas distribuidas uniformemente
    por la señal, con un pequeño jitter aleatorio para evitar
    que siempre cojamos exactamente los mismos offsets.
    """
    total = x_iq.shape[1]
    max_start = total - window_size

    # Posiciones base: distribución uniforme
    step = max_start // n_windows
    jitter = step // 4  # ±25% del paso como jitter
    starts = []
    for i in range(n_windows):
        base = i * step
        offset = rng.randint(-jitter, jitter) if jitter > 0 else 0
        start = max(0, min(base + offset, max_start))
        starts.append(start)

    windows = []
    for s in starts:
        w = x_iq[:, s:s + window_size].numpy().astype(np.float16)
        windows.append(w)
    return np.stack(windows)   # [n_windows, 2, window_size]


def main():
    # ── Buscar archivos .pt ──────────────────────────────────
    print("Buscando archivos .pt...")
    pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.pt"), recursive=True))
    if not pt_files:
        pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))

    if not pt_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .pt en {DATA_DIR}\n"
            f"Comprueba que DATA_DIR apunta al directorio correcto."
        )

    print(f"  ✓ Encontrados {len(pt_files)} archivos .pt")

    total_windows = len(pt_files) * WINDOWS_PER_FILE
    est_gb = total_windows * 2 * WINDOW_SIZE * 2 / 1e9   # float16 = 2 bytes
    print(f"  → Se crearán {total_windows:,} ventanas")
    print(f"  → Tamaño estimado del HDF5: {est_gb:.1f} GB (float16)")

    rng = random.Random(SEED)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ── Crear HDF5 con datasets redimensionables ─────────────
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Guardamos atributos de configuración para referencia futura
        f.attrs['window_size']      = WINDOW_SIZE
        f.attrs['windows_per_file'] = WINDOWS_PER_FILE
        f.attrs['noise_class']      = NOISE_CLASS
        f.attrs['sample_rate_mhz']  = 14.0

        ds_x   = f.create_dataset('x',
                                  shape=(total_windows, 2, WINDOW_SIZE),
                                  maxshape=(None, 2, WINDOW_SIZE),
                                  dtype=np.float16,
                                  chunks=(128, 2, WINDOW_SIZE),
                                  compression='lzf')   # lzf: rápido y compacto

        ds_y   = f.create_dataset('y',
                                  shape=(total_windows,),
                                  maxshape=(None,),
                                  dtype=np.int8)

        ds_snr = f.create_dataset('snr',
                                  shape=(total_windows,),
                                  maxshape=(None,),
                                  dtype=np.int8)

        ds_cls = f.create_dataset('original_class',
                                  shape=(total_windows,),
                                  maxshape=(None,),
                                  dtype=np.int8)   # clase original (0-6)

        idx     = 0
        errors  = 0
        noise_w = 0
        signal_w = 0

        for fpath in tqdm(pt_files, desc="Procesando archivos", unit="archivo"):
            try:
                target, snr = parse_filename(fpath)
                label = 0 if target == NOISE_CLASS else 1

                data  = torch.load(fpath, map_location='cpu', weights_only=False)
                x_iq  = data['x_iq'].float()   # [2, 1048576]

                windows = extract_windows_uniform(x_iq, WINDOWS_PER_FILE,
                                                  WINDOW_SIZE, rng)
                n = len(windows)
                end = idx + n

                ds_x[idx:end]   = windows
                ds_y[idx:end]   = label
                ds_snr[idx:end] = snr
                ds_cls[idx:end] = target

                if label == 0:
                    noise_w  += n
                else:
                    signal_w += n

                idx = end

            except Exception as e:
                errors += 1
                tqdm.write(f"  ✗ Error en {Path(fpath).name}: {e}")
                continue

        # Recortar si hubo errores
        if idx < total_windows:
            ds_x.resize(idx, axis=0)
            ds_y.resize(idx, axis=0)
            ds_snr.resize(idx, axis=0)
            ds_cls.resize(idx, axis=0)

    # ── Resumen ──────────────────────────────────────────────
    size_gb = os.path.getsize(OUTPUT_FILE) / 1e9
    print(f"\n{'─'*55}")
    print(f"  ✓ Preprocesado completado")
    print(f"  Total ventanas guardadas : {idx:,}")
    print(f"  Ventanas de ruido        : {noise_w:,} ({noise_w/idx*100:.1f}%)")
    print(f"  Ventanas de señal/dron   : {signal_w:,} ({signal_w/idx*100:.1f}%)")
    print(f"  Errores                  : {errors}")
    print(f"  Tamaño del archivo HDF5  : {size_gb:.2f} GB")
    print(f"  Guardado en: {OUTPUT_FILE}")
    print(f"{'─'*55}")
    print("\nSiguiente paso: python train_detector.py")


if __name__ == '__main__':
    main()
