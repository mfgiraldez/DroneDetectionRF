"""
evaluate_teacher.py
===================
Evaluacion standalone del Teacher desde el mejor checkpoint.
Genera: confusion matrix, curva ROC, classification report, y
re-dibuja las curvas de entrenamiento (Loss, F1, AUC) desde history.json.
No requiere re-entrenar nada.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as TeacherCVCNN, BurstCVCNNDataset as TeacherDataset, pad_seq_collate
from NoisyUAV.funciones.plot_styles import apply_ieee_style, get_color_palette, clean_spines

import pandas as pd
from tqdm import tqdm

os.environ['PYTHONIOENCODING'] = 'utf-8'

DATA_DIR    = Path(r"C:\TFM_data\NoisyUAV\drone_RF_data")
CSV_PATH    = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1\teacher_dataset_high_snr.csv")
OUT_DIR     = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1")
CHECKPOINT  = OUT_DIR / "checkpoints" / "teacher_model_best.pt"
FIG_DIR     = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def plot_training_curves(history, colors, save_path):
    """Reconstruye las curvas de entrenamiento desde history.json."""
    has_auc = 'val_auc' in history and len(history['val_auc']) > 0
    ncols = 3 if has_auc else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 4))
    x = range(1, len(history['train_loss'])+1)

    axes[0].plot(x, history['train_loss'], c=colors['blue'],   label='Train Loss')
    axes[0].plot(x, history['val_loss'],   c=colors['orange'], label='Val Loss')
    axes[0].set_title("Loss Evolution", fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].legend(); clean_spines(axes[0])

    axes[1].plot(x, history['val_f1'],  c=colors['green'],  label='Val F1')
    axes[1].plot(x, history['val_acc'], c=colors['purple'], label='Val Acc')
    axes[1].set_ylim(0, 1)
    axes[1].set_title("F1 & Accuracy", fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].legend(); clean_spines(axes[1])

    if has_auc:
        axes[2].plot(x, history['val_auc'], c=colors['red'], label='Val AUC-ROC')
        axes[2].set_ylim(0, 1)
        axes[2].set_title("AUC-ROC", fontweight='bold')
        axes[2].set_xlabel("Epoch"); axes[2].legend(); clean_spines(axes[2])

    plt.suptitle("Teacher Training History", fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Curvas de entrenamiento: {save_path.name}")


def plot_evaluation(all_lbl, all_prob, all_pred, colors, prefix, fig_dir):
    """Genera confusion matrix + curva ROC."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    cm = confusion_matrix(all_lbl, all_pred)
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix ({prefix})", fontweight='bold')
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    classes = ['Noise (0)', 'Drone (1)']
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(classes); axes[0].set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                         fontsize=14, fontweight='bold',
                         color='white' if cm[i,j] > thresh else 'black')

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_lbl, all_prob)
    auc = roc_auc_score(all_lbl, all_prob)
    axes[1].plot(fpr, tpr, c=colors['blue'], lw=2, label=f'AUC = {auc:.4f}')
    axes[1].plot([0,1],[0,1],'k--', lw=1, label='Random')
    axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.02])
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC Curve ({prefix})", fontweight='bold')
    axes[1].legend(); clean_spines(axes[1])

    plt.tight_layout()
    out = fig_dir / f"teacher_{prefix.lower()}_evaluation.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Figura evaluacion {prefix}: {out.name}")
    return auc


def run_inference(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_lbl, all_prob, all_pred = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencia", leave=False):
            iq  = batch['iq'].to(DEVICE)
            f   = batch['feats'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            logit = model(iq, f)
            total_loss += criterion(logit, lbl).item()
            prob = torch.sigmoid(logit)
            all_lbl.extend(lbl.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
            all_pred.extend((prob > 0.5).float().cpu().numpy())
    total_loss /= len(loader)
    return total_loss, np.array(all_lbl), np.array(all_prob), np.array(all_pred)


def main():
    apply_ieee_style()
    colors = get_color_palette()

    print("=" * 60)
    print("  EVALUACION TEACHER — Standalone desde best checkpoint")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print(f"❌ No se encuentra el checkpoint en:\n   {CHECKPOINT}")
        return

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = TeacherCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])

    phys_mean = torch.from_numpy(ckpt['phys_mean'])
    phys_std  = torch.from_numpy(ckpt['phys_std'])
    best_f1   = ckpt.get('val_f1', '?')
    best_ep   = ckpt.get('epoch', '?')
    print(f"  Mejor época: {best_ep+1 if isinstance(best_ep, int) else best_ep}")
    print(f"  Mejor Val F1: {best_f1:.4f}\n")

    # ── Datasets
    df = pd.read_csv(CSV_PATH)
    df_val  = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']

    ds_val  = TeacherDataset(df_val,  DATA_DIR, augment=False, phys_mean=phys_mean, phys_std=phys_std)
    ds_test = TeacherDataset(df_test, DATA_DIR, augment=False, phys_mean=phys_mean, phys_std=phys_std)
    ld_val  = DataLoader(ds_val,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pad_seq_collate)
    ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pad_seq_collate)

    n_drone = (df[df['split']=='train']['label']==1).sum()
    n_noise = (df[df['split']=='train']['label']==0).sum()
    pos_w = torch.tensor([max(n_noise,1)/max(n_drone,1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # ── Reconstruir curvas de entrenamiento
    hist_path = OUT_DIR / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_curves(history, colors, FIG_DIR / "teacher_training_curves_full.png")
    else:
        print("  ⚠️  history.json no encontrado, se omiten curvas de entrenamiento.")

    # ── Evaluación Val
    print("\n--- Val Set ---")
    _, val_lbl, val_prob, val_pred = run_inference(model, ld_val, criterion)
    val_f1  = f1_score(val_lbl, val_pred, zero_division=0)
    val_auc = roc_auc_score(val_lbl, val_prob)
    val_acc = (val_lbl == val_pred).mean()
    print(f"  F1={val_f1:.4f} | AUC={val_auc:.4f} | Acc={val_acc:.4f}")
    print(classification_report(val_lbl, val_pred, target_names=['Noise','Drone'], digits=4))
    plot_evaluation(val_lbl, val_prob, val_pred, colors, "Val", FIG_DIR)

    # ── Evaluación Test
    print("\n--- Test Set ---")
    _, test_lbl, test_prob, test_pred = run_inference(model, ld_test, criterion)
    test_f1  = f1_score(test_lbl, test_pred, zero_division=0)
    test_auc = roc_auc_score(test_lbl, test_prob)
    test_acc = (test_lbl == test_pred).mean()
    print(f"  F1={test_f1:.4f} | AUC={test_auc:.4f} | Acc={test_acc:.4f}")
    print(classification_report(test_lbl, test_pred, target_names=['Noise','Drone'], digits=4))
    plot_evaluation(test_lbl, test_prob, test_pred, colors, "Test", FIG_DIR)

    # ── Guardar métricas en JSON para la tesis
    metrics = {
        "best_epoch": int(best_ep) + 1 if isinstance(best_ep, int) else best_ep,
        "val":  {"f1": float(val_f1),  "auc": float(val_auc),  "acc": float(val_acc)},
        "test": {"f1": float(test_f1), "auc": float(test_auc), "acc": float(test_acc)},
    }
    with open(OUT_DIR / "teacher_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Métricas guardadas en teacher_eval_metrics.json")
    print(f"✅ Figuras en: {FIG_DIR}")


if __name__ == '__main__':
    main()
