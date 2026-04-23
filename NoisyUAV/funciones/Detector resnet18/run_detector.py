"""
Inferencia del detector ResNet18-IQ sobre archivos .pt de NoisyUAV.

Evalúa el modelo entrenado en train_detector.py sobre el Test Set oficial
y genera la gráfica de Pd/Pfa vs SNR para el TFM.

Ejecución: python run_detector.py
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from train_detector import (
    ResNet18_IQ, IQWindowDataset, build_file_catalog,
    evaluate, plot_snr,
    MODEL_PATH, SPLIT_PATH, DATA_DIR, WINDOW_SIZE,
    PLOTS_DIR, DEVICE, BATCH_SIZE, NUM_WORKERS,
)
from torch.utils.data import DataLoader

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_model(model_path: str) -> ResNet18_IQ:
    ckpt  = torch.load(model_path, map_location=DEVICE, weights_only=False)
    ws    = ckpt["config"].get("window_size", WINDOW_SIZE)
    model = ResNet18_IQ(num_classes=2).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Modelo cargado: epoch {ckpt['epoch']}")
    print(f"  AUC-ROC (val) : {ckpt['val_metrics']['auc']:.4f}")
    print(f"  F1 (val)      : {ckpt['val_metrics']['f1']:.4f}")
    return model, ws


def main():
    print("=" * 60)
    print("  ResNet18-IQ: Evaluación sobre Test Set")
    print("=" * 60)
    print(f"  Dispositivo: {DEVICE}\n")

    # Cargar modelo
    model, ws = load_model(MODEL_PATH)
    print()

    # Reconstruir catálogo y cargar splits guardados
    catalog = build_file_catalog(DATA_DIR)
    if not catalog:
        raise FileNotFoundError(f"No se encontraron archivos .pt en: {DATA_DIR}")

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"No se encontró el fichero de splits: {SPLIT_PATH}\n"
            "Ejecuta train_detector.py primero."
        )
    with open(SPLIT_PATH) as fh:
        splits = json.load(fh)
    idx_test = splits["test"]

    test_ds = IQWindowDataset(catalog, idx_test, ws, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    criterion = torch.nn.CrossEntropyLoss()
    metrics   = evaluate(model, test_loader, criterion, DEVICE)

    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall/Pd : {metrics['recall']:.4f}")

    # Tabla SNR
    snr_m = metrics["snr_metrics"]
    print(f"\n{'─'*55}")
    print("  RENDIMIENTO POR SNR  (Test Set)")
    print(f"  {'SNR':>6}  {'Pd':>7}  {'Pfa':>7}  {'Acc':>7}")
    print(f"  {'─'*35}")
    for sv in sorted(snr_m.keys()):
        m   = snr_m[sv]
        pd  = m.get("Pd", float("nan"))
        bar = ("█" * int(pd * 20)) if not (pd != pd) else ""
        print(f"  {sv:+4d} dB  {pd:7.4f}  "
              f"{m.get('Pfa', float('nan')):7.4f}  "
              f"{m['accuracy']:7.4f}  {bar}")
    print(f"{'─'*55}")

    # Gráfica final
    plot_snr(snr_m, PLOTS_DIR)
    print(f"\n  Gráfica guardada en: {PLOTS_DIR}/snr_performance.png")


if __name__ == "__main__":
    main()
