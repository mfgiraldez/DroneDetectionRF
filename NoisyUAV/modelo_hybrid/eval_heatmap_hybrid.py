"""
eval_heatmap_hybrid.py
======================
Genera el heatmap de accuracy por Emisor RF (drone model) x SNR
para el HybridCVCNN Run #2, equivalente a la figura del baseline CV-CNN.

La figura muestra la tasa de acierto media para CADA drone por nivel de SNR,
permitiendo identificar qué modelos de drone son más difíciles de detectar
en condiciones de bajo SNR.

Uso:
    cmd /c "python eval_heatmap_hybrid.py"
    cmd /c "python eval_heatmap_hybrid.py --ckpt <ruta_checkpoint>"

Salida:
    resultados_hybrid_run2/figures/heatmap_target_snr.png
    resultados_hybrid_run2/figures/heatmap_noise_snr.png   (falsos positivos)
"""

import sys, os
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.funciones.physical_features import load_features_cache
from NoisyUAV.modelos.hybrid_cvcnn import HybridCVCNN, HybridDataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH  = r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid_run2\checkpoints\best_model.pt"
CACHE_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid\features_cache.npz"
OUT_DIR    = r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid_run2\figures"
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"

# Nombres de los emisores RF (target_multiclass)
# label=1 (drones): targets 0, 1, 2, 3, 5, 6
# label=0 (ruido):  target 4 (ruido blanco/WiFi/BT mezclado)
DRONE_TARGETS = [0, 1, 2, 3, 5, 6]
NOISE_TARGETS = [4]

TARGET_NAMES = {
    0: "Target 0",
    1: "Target 1",
    2: "Target 2",
    3: "Target 3",
    5: "Target 5",
    6: "Target 6",
    4: "Ruido (T4)",
}

FIGSAVE = dict(dpi=150, bbox_inches="tight")

# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    print(f"  Cargando checkpoint: {Path(ckpt_path).name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get('cfg', {})

    model = HybridCVCNN(
        phys_dim    = int(cfg.get('phys_dim',    12)),
        cnn_embed   = int(cfg.get('cnn_embed',   256)),
        pool_size   = int(cfg.get('pool_size',   32)),
        hidden_dim  = int(cfg.get('hidden_dim',  256)),
        dropout_cnn = float(cfg.get('dropout_cnn',  0.0)),   # eval: sin dropout
        dropout_fuse= float(cfg.get('dropout_fuse', 0.0)),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    phys_mean = torch.tensor(ckpt['phys_mean'], dtype=torch.float32)
    phys_std  = torch.tensor(ckpt['phys_std'],  dtype=torch.float32)

    print(f"  Best epoch: {ckpt['epoch']} | Val F1: {ckpt['val_f1']:.4f}")
    return model, phys_mean, phys_std, int(cfg.get('crop_len', 131072))


@torch.no_grad()
def run_inference(model, df, cache, phys_mean, phys_std, device,
                  crop_len=131072, batch_size=32):
    """
    Corre inferencia sobre un DataFrame y devuelve df con columna 'predicted'
    y 'prob_drone'.
    """
    ds = HybridDataset(df, cache, crop_len=crop_len, augment=False,
                       phys_mean=phys_mean, phys_std=phys_std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_preds = [], []
    for iq_crop, phys, _ in dl:
        iq_crop = iq_crop.to(device)
        phys    = phys.to(device)
        logit   = model(iq_crop, phys)
        prob    = torch.sigmoid(logit.squeeze(1))
        pred    = (prob > 0.5).long()
        all_probs.extend(prob.cpu().tolist())
        all_preds.extend(pred.cpu().tolist())

    df = df.copy()
    df['prob_drone'] = all_probs
    df['predicted']  = all_preds
    df['correct']    = (df['label'] == df['predicted']).astype(int)
    return df


def plot_heatmap_drones(df_result, out_dir, title_suffix=""):
    """
    Heatmap de ACIERTO (recall) por Target (drone) x SNR.
    Solo incluye muestras donde label==1 (drones reales).
    'correct' = 1 si el modelo predijo drone correctamente.
    """
    drones_df = df_result[df_result['label'] == 1].copy()
    drones_df['Emisor RF'] = drones_df['target_multiclass'].map(
        lambda x: f"Target {x}"
    )

    heatmap_data = drones_df.pivot_table(
        index='Emisor RF',
        columns='snr',
        values='correct',
        aggfunc='mean'
    )

    # Ordenar targets y SNRs
    heatmap_data = heatmap_data.sort_index()
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    n_rows = len(heatmap_data)
    n_cols = len(heatmap_data.columns)
    fig_w  = max(14, n_cols * 0.55)
    fig_h  = max(3,  n_rows * 0.85)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.4,
        linecolor='#1a1a2e',
        ax=ax,
        cbar_kws={'label': 'Tasa de Acierto'},
    )
    ax.set_title(
        f"HybridCVCNN — Detección por Emisor RF y SNR{title_suffix}",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()
    path = Path(out_dir) / "heatmap_target_snr.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
    print(f"  Guardado: {path}")
    return str(path)


def plot_heatmap_noise(df_result, out_dir, title_suffix=""):
    """
    Heatmap de ESPECIFICIDAD (1 - FPR) por tipo de ruido x SNR.
    Solo incluye muestras donde label==0 (ruido).
    Verde = pocas confusiones (bien). Rojo = muchos falsos positivos (mal).
    """
    noise_df = df_result[df_result['label'] == 0].copy()
    noise_df['Emisor RF'] = noise_df['target_multiclass'].map(
        lambda x: TARGET_NAMES.get(x, f"Ruido T{x}")
    )
    # Especificidad = correct (predijo 0 cuando era 0) = 1 - FP rate
    heatmap_data = noise_df.pivot_table(
        index='Emisor RF',
        columns='snr',
        values='correct',
        aggfunc='mean'
    )
    heatmap_data = heatmap_data.sort_index()
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    n_rows = len(heatmap_data)
    n_cols = len(heatmap_data.columns)
    fig_w  = max(14, n_cols * 0.55)
    fig_h  = max(2,  n_rows * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.4,
        linecolor='#1a1a2e',
        ax=ax,
        cbar_kws={'label': 'Especificidad (1 - FPR)'},
    )
    ax.set_title(
        f"HybridCVCNN — Especificidad por Emisor de Ruido y SNR{title_suffix}",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()
    path = Path(out_dir) / "heatmap_noise_snr.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
    print(f"  Guardado: {path}")
    return str(path)


def print_summary(df_result):
    """Imprime resumen por target y grupo SNR."""
    print("\n" + "="*65)
    print("  RESUMEN POR EMISOR RF (DRONES, label=1)")
    print("="*65)

    drones = df_result[df_result['label'] == 1]
    for target in sorted(drones['target_multiclass'].unique()):
        sub = drones[drones['target_multiclass'] == target]
        acc_global = sub['correct'].mean()
        acc_a = sub[sub['snr'] >= 10]['correct'].mean()
        sub_b = sub[(sub['snr'] >= -6) & (sub['snr'] < 10)]
        acc_b = sub_b['correct'].mean() if len(sub_b) > 0 else float('nan')
        sub_c = sub[sub['snr'] < -6]
        acc_c = sub_c['correct'].mean() if len(sub_c) > 0 else float('nan')
        print(
            f"  Target {target}: Global={acc_global:.3f} | "
            f"A(>=10)={acc_a:.3f} | B(-6..10)={acc_b:.3f} | C(<-6)={acc_c:.3f} "
            f"| n={len(sub)}"
        )

    print("\n" + "="*65)
    print("  ESPECIFICIDAD POR TIPO DE RUIDO (label=0)")
    print("="*65)
    noise = df_result[df_result['label'] == 0]
    for target in sorted(noise['target_multiclass'].unique()):
        sub  = noise[noise['target_multiclass'] == target]
        name = TARGET_NAMES.get(target, f"Ruido T{target}")
        spec_global = sub['correct'].mean()
        spec_a = sub[sub['snr'] >= 10]['correct'].mean()
        sub_c  = sub[sub['snr'] < -6]
        spec_c = sub_c['correct'].mean() if len(sub_c) > 0 else float('nan')
        fp_global = 1 - spec_global
        print(
            f"  {name}: Spec Global={spec_global:.3f} (FP={fp_global:.3f}) | "
            f"A(>=10)={spec_a:.3f} | C(<-6)={spec_c:.3f} | n={len(sub)}"
        )
    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Heatmap HybridCVCNN por target y SNR')
    parser.add_argument('--ckpt',       default=CKPT_PATH)
    parser.add_argument('--cache',      default=CACHE_PATH)
    parser.add_argument('--out',        default=OUT_DIR)
    parser.add_argument('--data_dir',   default=DATA_DIR)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split',      default='test',
                        choices=['test', 'val', 'all'],
                        help='Split a evaluar (default: test)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  HybridCVCNN — Heatmap por Emisor RF y SNR")
    print(f"  Device : {device}")
    print(f"  Split  : {args.split}")
    print(f"{'='*65}\n")

    # 1. Modelo
    model, phys_mean, phys_std, crop_len = load_model(args.ckpt, device)

    # 2. Cache
    print(f"\n  Cargando features cache...")
    cache = load_features_cache(args.cache)
    print(f"  Cache: {len(cache)} entradas")

    # 3. Dataset — necesitamos target_multiclass (no solo label binario)
    print(f"\n  Cargando splits...")
    df_train, df_val, df_test = obtener_splits_dataset(data_dir=args.data_dir)

    if args.split == 'test':
        df_eval = df_test
    elif args.split == 'val':
        df_eval = df_val
    else:   # 'all'
        df_eval = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print(f"  Muestras a evaluar: {len(df_eval):,}")
    print(f"  Targets presentes: {sorted(df_eval['target_multiclass'].unique())}")
    print(f"  SNRs presentes: {sorted(df_eval['snr'].unique())}")

    # 4. Inferencia
    print(f"\n  Ejecutando inferencia ({len(df_eval):,} muestras)...")
    df_result = run_inference(
        model, df_eval, cache, phys_mean, phys_std, device,
        crop_len=crop_len, batch_size=args.batch_size
    )

    acc_global = df_result['correct'].mean()
    print(f"  Accuracy global: {acc_global:.4f} ({acc_global*100:.2f}%)")

    # 5. Resumen textual
    print_summary(df_result)

    # 6. Figuras
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"\n  Generando heatmaps...")
    plot_heatmap_drones(df_result, args.out)
    plot_heatmap_noise(df_result, args.out)

    print(f"\n{'='*65}")
    print(f"  DONE — Figuras guardadas en {args.out}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
