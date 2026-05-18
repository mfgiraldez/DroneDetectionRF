"""
run_pipeline.py — Punto de Entrada Único del Experimento Xin CV-CNN 2D
=======================================================================
Orquesta el pipeline completo en tres fases secuenciales:

    Fase 1: Construcción del dataset (CSV de splits estratificado)
    Fase 2: Carga y validación de datos
    Fase 3: Entrenamiento con trazabilidad completa

Si el CSV ya existe, la Fase 1 se omite automáticamente (idempotente).
Si existen checkpoints previos, el entrenamiento se reanuda desde el
último epoch completado.

Uso:
    conda activate IAIAVv3
    python -m NoisyUAV.modelo_xin_v1.run_pipeline [opciones]

Opciones principales:
    --data_dir   DIR   Directorio con los .pt del dataset  [requerido]
    --output_dir DIR   Directorio de salida del experimento [requerido]
    --csv_path   PATH  Ruta al CSV (si ya existe, se omite Fase 1)
    --epochs     N     Número de epochs          [default: 30]
    --batch_size N     Tamaño de batch            [default: 4]
    --lr         F     Tasa de aprendizaje        [default: 0.001]

Ejemplo:
    python -m NoisyUAV.modelo_xin_v1.run_pipeline ^
        --data_dir   C:\\TFM_data\\NoisyUAV\\drone_RF_data ^
        --output_dir C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_xin_v1\\resultados ^
        --epochs 30 --batch_size 4
"""

import os
import sys
import argparse
import logging
import time

# Asegurar que el paquete raíz es importable desde cualquier CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

os.environ["PYTHONIOENCODING"] = "utf-8"

from NoisyUAV.modelo_xin_v1.xin_dataset import build_stratified_split_csv
from NoisyUAV.modelo_xin_v1.xin_train import run_training
from NoisyUAV.modelo_xin_v1.xin_precompute_cache import precompute_spectrograms


# ─────────────────────────────────────────────────────────────────────────────
# Argumentos de línea de comandos
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline completo: Xin CV-CNN 2D sobre NoisyUAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas principales
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\TFM_data\NoisyUAV\drone_RF_data",
        help="Directorio con los ficheros IQdata_*.pt del dataset NoisyUAV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_xin_v1\resultados",
        help="Directorio de salida (checkpoints, figuras, métricas, logs).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "Ruta al CSV de splits. Si no se especifica, se genera automáticamente "
            "en output_dir/xin_splits.csv. Si ya existe el fichero, se reutiliza."
        ),
    )

    # Hiperparámetros de entrenamiento
    parser.add_argument("--epochs",       type=int,   default=30,    help="Número de epochs de entrenamiento.")
    parser.add_argument("--batch_size",   type=int,   default=4,     help="Tamaño de batch. Reducir si hay OOM.")
    parser.add_argument("--lr",           type=float, default=1e-3,  help="Tasa de aprendizaje (Adam).")
    parser.add_argument("--weight_decay", type=float, default=1e-5,  help="Penalización L2 (Adam).")
    parser.add_argument("--num_workers",  type=int,   default=0,     help="Workers DataLoader. Mantener 0 en Windows.")

    # Parámetros STFT / espectrograma
    parser.add_argument("--nfft",         type=int,   default=1024,  help="Puntos de la STFT.")
    parser.add_argument("--spec_h",       type=int,   default=256,   help="Alto del espectrograma de salida.")
    parser.add_argument("--spec_w",       type=int,   default=256,   help="Ancho del espectrograma de salida.")

    # Dataset
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Directorio con los tensores STFT+Sobel pre-computados. "
            "Si se especifica, se ejecuta el pre-computo automaticamente "
            "antes del entrenamiento (reanudable). "
            "Si es None, se calcula on-the-fly (mas lento)."
        ),
    )
    parser.add_argument(
        "--skip_precompute",
        action="store_true",
        help="Omitir la fase de pre-computo aunque --cache_dir este definido "
             "(util si la cache ya fue generada previamente).",
    )

    # Splits del dataset
    parser.add_argument("--val_ratio",  type=float, default=0.15, help="Fraccion de validacion.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraccion de test.")
    parser.add_argument("--seed",       type=int,   default=42,   help="Semilla de aleatoriedad.")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Logging global
    log_path = os.path.join(args.output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    t_total = time.time()

    logging.info("╔══════════════════════════════════════════════════════════════╗")
    logging.info("║  PIPELINE — Xin CV-CNN 2D sobre NoisyUAV                    ║")
    logging.info("║  Referencia: Xin et al. (2026) — NoisyUAV (Glüge, 2024)    ║")
    logging.info("╚══════════════════════════════════════════════════════════════╝")

    # ── Resolución de ruta del CSV ───────────────────────────────────────────
    csv_path = args.csv_path or os.path.join(args.output_dir, "xin_splits.csv")

    # ── FASE 0: Pre-cómputo de caché (si se solicita) ───────────────────────
    if args.cache_dir and not args.skip_precompute:
        logging.info("[FASE 0] Pre-computando espectrogramas STFT+Sobel...")
        t0 = time.time()
        precompute_spectrograms(
            data_dir   = args.data_dir,
            cache_dir  = args.cache_dir,
            spec_h     = args.spec_h,
            spec_w     = args.spec_w,
            nfft       = args.nfft,
        )
        logging.info(f"[FASE 0] Completada en {(time.time()-t0)/60:.1f} min")
    elif args.cache_dir and args.skip_precompute:
        logging.info("[FASE 0] Omitida (--skip_precompute activo). Usando cache existente.")
    else:
        logging.info("[FASE 0] Sin cache (modo on-the-fly).")

    # ── FASE 1: Construcción del dataset ────────────────────────────────────
    if os.path.exists(csv_path):
        logging.info(f"[FASE 1] CSV ya existe, se reutiliza: {csv_path}")
    else:
        logging.info("[FASE 1] Construyendo CSV de splits estratificados...")
        t1 = time.time()
        build_stratified_split_csv(
            data_dir   = args.data_dir,
            out_csv    = csv_path,
            val_ratio  = args.val_ratio,
            test_ratio = args.test_ratio,
            seed       = args.seed,
        )
        logging.info(f"[FASE 1] Completada en {time.time()-t1:.1f}s → {csv_path}")

    # ── FASE 2: Validación rápida del dataset ───────────────────────────────
    logging.info("[FASE 2] Validando acceso al dataset...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    for split in ["train", "val", "test"]:
        n = (df["split"] == split).sum()
        n_drone = ((df["split"] == split) & (df["is_drone"] == 1)).sum()
        logging.info(f"  {split:>5}: {n:>6,} muestras | {n_drone:>5,} drones ({100*n_drone/n:.1f}%)")
    logging.info("[FASE 2] Validación superada.")

    # ── FASE 3: Entrenamiento ────────────────────────────────────────────────
    logging.info("[FASE 3] Iniciando entrenamiento...")
    cfg = {
        "csv_path":    csv_path,
        "data_dir":    args.data_dir,
        "output_dir":  args.output_dir,
        "cache_dir":   args.cache_dir,   # None = on-the-fly
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "weight_decay":args.weight_decay,
        "num_workers": args.num_workers,
        "nfft":        args.nfft,
        "spec_h":      args.spec_h,
        "spec_w":      args.spec_w,
    }
    run_training(cfg)

    logging.info(f"Pipeline completado en {(time.time()-t_total)/60:.1f} minutos.")


if __name__ == "__main__":
    main()
