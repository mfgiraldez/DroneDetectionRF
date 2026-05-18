"""
xin_build_dataset.py — Generación y Validación del CSV de Splits
=================================================================
Escanea el directorio de datos NoisyUAV, construye la partición
Train/Val/Test estratificada e imprime un informe detallado de
distribución para verificación científica.

Uso:
    python -m NoisyUAV.modelo_xin_v1.xin_build_dataset \\
        --data_dir  C:\\TFM_data\\NoisyUAV\\drone_RF_data \\
        --out_csv   C:\\TFM_data\\NoisyUAV\\xin_splits.csv
"""

import os
import sys
import argparse
import logging

# Asegurar que el paquete raíz es importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from NoisyUAV.modelo_xin_v1.xin_dataset import build_stratified_split_csv

os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el CSV de splits estratificados para el experimento Xin CV-CNN 2D."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\TFM_data\NoisyUAV\drone_RF_data",
        help="Directorio con los ficheros IQdata_*.pt del dataset NoisyUAV.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=r"C:\TFM_data\NoisyUAV\xin_splits.csv",
        help="Ruta de salida del CSV de splits.",
    )
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.info("=" * 60)
    logging.info("CONSTRUCCIÓN DEL DATASET — Xin CV-CNN 2D")
    logging.info("=" * 60)
    logging.info(f"  data_dir  : {args.data_dir}")
    logging.info(f"  out_csv   : {args.out_csv}")
    logging.info(f"  val_ratio : {args.val_ratio}")
    logging.info(f"  test_ratio: {args.test_ratio}")
    logging.info(f"  seed      : {args.seed}")

    build_stratified_split_csv(
        data_dir   = args.data_dir,
        out_csv    = args.out_csv,
        val_ratio  = args.val_ratio,
        test_ratio = args.test_ratio,
        seed       = args.seed,
    )

    logging.info("Dataset construido correctamente.")


if __name__ == "__main__":
    main()
