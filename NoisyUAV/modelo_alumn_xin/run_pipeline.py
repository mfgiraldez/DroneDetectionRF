"""
run_pipeline.py — Pipeline Completo AlumnXin Hybrid (lanzable desde terminal)
===============================================================================
Orquesta el experimento completo en fases secuenciales:

    Fase 0 (opcional): Pre-computo de cache STFT+Sobel por burst
    Fase 1:            Validacion del CSV y datos
    Fase 2:            Entrenamiento con reanudacion automatica
    Fase 3:            Evaluacion sobre test set (5 figuras + JSON)

Si los checkpoints ya existen, el entrenamiento se reanuda desde el ultimo epoch.
Si la cache ya esta generada, se puede saltar Fase 0 con --skip_precompute.

Uso minimo (on-the-fly, sin cache):
    conda activate IAIAVv3
    python -m NoisyUAV.modelo_alumn_xin.run_pipeline ^
        --data_dir C:\\TFM_data\\NoisyUAV\\drone_RF_data ^
        --epochs 60 --batch_size 4

Uso con cache (recomendado para GPU):
    python -m NoisyUAV.modelo_alumn_xin.run_pipeline ^
        --data_dir   C:\\TFM_data\\NoisyUAV\\drone_RF_data ^
        --cache_dir  C:\\TFM_data\\NoisyUAV\\alumn_xin_cache_256x256 ^
        --epochs 60 --batch_size 8
"""

import os
import sys
import argparse
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["PYTHONIOENCODING"] = "utf-8"

# ---------------------------------------------------------------------------- #
# Argumentos                                                                    #
# ---------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline completo: AlumnXin Hybrid CV-CNN 2D sobre NoisyUAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    parser.add_argument(
        "--data_dir",
        type=str, default=r"C:\TFM_data\NoisyUAV\drone_RF_data",
        help="Directorio con los ficheros IQdata_*.pt del dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str, default=r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_xin\resultados",
        help="Directorio de salida (checkpoints, figuras, metricas, logs).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_xin\alumn_xin_dataset.csv",
        help="Ruta al CSV de pseudo-labels. Debe existir previamente.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directorio de cache STFT+Sobel por burst. None = on-the-fly (lento).",
    )
    parser.add_argument(
        "--skip_precompute", action="store_true",
        help="Omitir fase de pre-computo aunque --cache_dir este definido.",
    )

    # Hiperparametros
    parser.add_argument("--epochs",          type=int,   default=60)
    parser.add_argument("--batch_size",      type=int,   default=4,
                        help="Reducir a 2-4 si hay OOM (modelo grande: ~20M params).")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--fusion_hidden",   type=int,   default=512,
                        help="Tamano de la capa oculta del MLP de fusion.")
    parser.add_argument("--dropout_fusion",  type=float, default=0.4)

    # Evaluacion
    parser.add_argument("--eval_split",      type=str,   default="test",
                        choices=["test", "val"],
                        help="Split a usar en la fase de evaluacion final.")
    parser.add_argument("--eval_batch_size", type=int,   default=8)
    parser.add_argument("--skip_eval",       action="store_true",
                        help="Omitir la fase de evaluacion final.")

    return parser.parse_args()


# ---------------------------------------------------------------------------- #
# Main                                                                          #
# ---------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
    logging.info("║  PIPELINE — AlumnXin Hybrid CV-CNN 2D sobre NoisyUAV        ║")
    logging.info("║  Xin et al. (2026) + Alumno V1 (Pseudo-Labels FHSS)        ║")
    logging.info("╚══════════════════════════════════════════════════════════════╝")
    logging.info(f"  CSV       : {args.csv_path}")
    logging.info(f"  Data      : {args.data_dir}")
    logging.info(f"  Cache     : {args.cache_dir or 'None (on-the-fly)'}")
    logging.info(f"  Output    : {args.output_dir}")
    logging.info(f"  Epochs    : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    # ── FASE 0: Pre-computo de cache ─────────────────────────────────────────
    if args.cache_dir and not args.skip_precompute:
        logging.info("[FASE 0] Pre-computando espectrogramas STFT+Sobel por burst...")
        t0 = time.time()
        from NoisyUAV.modelo_alumn_xin.alumn_xin_precompute_cache import precompute
        precompute(args.csv_path, args.data_dir, args.cache_dir)
        logging.info(f"[FASE 0] Completada en {(time.time()-t0)/60:.1f} min")
    elif args.cache_dir and args.skip_precompute:
        logging.info("[FASE 0] Omitida (--skip_precompute). Usando cache existente.")
    else:
        logging.info("[FASE 0] Sin cache — modo on-the-fly (lento, util para pruebas rapidas).")

    # ── FASE 1: Validacion del CSV ───────────────────────────────────────────
    logging.info("[FASE 1] Validando CSV y dataset...")
    import pandas as pd
    if not os.path.exists(args.csv_path):
        logging.error(f"ERROR: CSV no encontrado: {args.csv_path}")
        logging.error("Asegurate de que alumn_xin_dataset.csv existe en modelo_alumn_xin/")
        sys.exit(1)

    df = pd.read_csv(args.csv_path)
    for split in ["train", "val", "test"]:
        sub      = df[df["split"] == split]
        n        = len(sub)
        n_drone  = (sub["pseudo_label"] == 1).sum()
        logging.info(f"  {split:>5}: {n:>7,} bursts | {n_drone:>6,} drones ({100*n_drone/max(n,1):.1f}%)")
    logging.info("[FASE 1] Validacion OK.")

    # ── FASE 2: Entrenamiento ────────────────────────────────────────────────
    logging.info("[FASE 2] Iniciando entrenamiento...")
    t2 = time.time()
    from NoisyUAV.modelo_alumn_xin.alumn_xin_train import run_training
    cfg = {
        "csv_path":       args.csv_path,
        "data_dir":       args.data_dir,
        "output_dir":     args.output_dir,
        "cache_dir":      args.cache_dir,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "fusion_hidden":  args.fusion_hidden,
        "dropout_fusion": args.dropout_fusion,
    }
    run_training(cfg)
    logging.info(f"[FASE 2] Entrenamiento completado en {(time.time()-t2)/60:.1f} min.")

    # ── FASE 3: Evaluacion ───────────────────────────────────────────────────
    if not args.skip_eval:
        logging.info("[FASE 3] Evaluando sobre el test set...")
        t3 = time.time()

        figures_test_dir = os.path.join(args.output_dir, "figures_test")
        ckpt_best = os.path.join(args.output_dir, "checkpoints", "alumn_xin_model_best.pt")

        if not os.path.exists(ckpt_best):
            logging.warning(f"  No se encontro checkpoint BEST: {ckpt_best}")
            logging.warning("  Omitiendo evaluacion.")
        else:
            import torch
            from NoisyUAV.modelo_alumn_xin.alumn_xin_eval import (
                load_model, run_inference, print_summary_and_save,
                plot_heatmap_drones, plot_heatmap_noise,
                plot_recall_snr_lines, plot_accuracy_per_snr, plot_pr_curve,
            )
            os.makedirs(figures_test_dir, exist_ok=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model, phys_mean, phys_std = load_model(ckpt_best, device)
            df_result = run_inference(
                model, args.csv_path, args.data_dir, args.eval_split,
                args.eval_batch_size, device, phys_mean, phys_std,
                cache_dir=args.cache_dir,
            )
            print_summary_and_save(df_result, figures_test_dir)
            plot_heatmap_drones(df_result, figures_test_dir)
            plot_heatmap_noise(df_result, figures_test_dir)
            plot_recall_snr_lines(df_result, figures_test_dir)
            plot_accuracy_per_snr(df_result, figures_test_dir)
            ap, auc_pr = plot_pr_curve(df_result, figures_test_dir)
            logging.info(f"  AP={ap:.4f} | AUC-PR={auc_pr:.4f}")
            logging.info(f"[FASE 3] Evaluacion completada en {(time.time()-t3)/60:.1f} min.")
            logging.info(f"  Figuras en: {figures_test_dir}")
    else:
        logging.info("[FASE 3] Omitida (--skip_eval).")

    elapsed_total = (time.time() - t_total) / 60
    logging.info(f"\nPipeline completo en {elapsed_total:.1f} minutos.")
    logging.info(f"Resultados en: {args.output_dir}")


if __name__ == "__main__":
    main()
