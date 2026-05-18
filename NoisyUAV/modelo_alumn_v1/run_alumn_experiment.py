import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as AlumnCVCNN, BurstCVCNNDataset as AlumnDataset, pad_seq_collate
from NoisyUAV.funciones.visualizacion.plot_styles import apply_ieee_style, get_color_palette, clean_spines

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ================================
# CONFIGURACION
# ================================
DATA_DIR     = Path(r"C:\TFM_data\NoisyUAV\drone_RF_data")
TEACHER_CKPT = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_teacher_v1\checkpoints\teacher_model_best.pt")
OUT_DIR      = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1")
CSV_PATH     = OUT_DIR / "alumn_dataset_pseudo_v3.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
IMG_DIR  = OUT_DIR / "plots"
IMG_DIR.mkdir(exist_ok=True)

CHECKPOINT_BEST = CKPT_DIR / "alumn_model_best.pt"
CHECKPOINT_LAST = CKPT_DIR / "alumn_model_last.pt"

EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 1e-4       # LR bajo para fine-tuning
WEIGHT_DECAY = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# HELPERS DE PLOTTING
# ================================
def plot_live_curves(history, colors, save_path):
    """Curvas de Loss, F1 y AUC que se actualizan cada epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_x = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs_x, history['train_loss'], c=colors['blue'],   label='Train Loss')
    axes[0].plot(epochs_x, history['val_loss'],   c=colors['orange'], label='Val Loss')
    axes[0].set_title("Loss Evolution"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); clean_spines(axes[0])

    axes[1].plot(epochs_x, history['val_f1'],  c=colors['green'],  label='Val F1')
    axes[1].plot(epochs_x, history['val_acc'], c=colors['purple'], label='Val Acc')
    axes[1].set_ylim(0, 1); axes[1].set_title("F1 & Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); clean_spines(axes[1])

    axes[2].plot(epochs_x, history['val_auc'], c=colors['red'], label='Val AUC-ROC')
    axes[2].set_ylim(0, 1); axes[2].set_title("AUC-ROC")
    axes[2].set_xlabel("Epoch"); axes[2].legend(); clean_spines(axes[2])

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_final_evaluation(all_lbl, all_prob, all_pred, colors, save_dir):
    """Genera figures de evaluacion final: confusion matrix + ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- Confusion Matrix --
    cm = confusion_matrix(all_lbl, all_pred)
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Test Set)", fontweight='bold')
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    classes = ['Noise (0)', 'Drone (1)']
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(classes); axes[0].set_yticklabels(classes)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=14, fontweight='bold',
                         color='white' if cm[i, j] > cm.max()/2 else 'black')

    # -- ROC Curve --
    fpr, tpr, _ = roc_curve(all_lbl, all_prob)
    auc_val = roc_auc_score(all_lbl, all_prob)
    axes[1].plot(fpr, tpr, c=colors['blue'], lw=2, label=f'ROC (AUC = {auc_val:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.02])
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve (Test Set)", fontweight='bold')
    axes[1].legend(); clean_spines(axes[1])

    plt.tight_layout()
    fig.savefig(save_dir / "alumn_final_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura evaluacion final guardada.")


def run_validation(model, loader, criterion):
    """Ejecuta validacion y devuelve todas las metricas."""
    model.eval()
    val_loss = 0
    all_lbl, all_prob, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            iq  = batch['iq'].to(DEVICE)
            f   = batch['feats'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            logit = model(iq, f)
            val_loss += criterion(logit, lbl).item()
            prob = torch.sigmoid(logit)
            all_lbl.extend(lbl.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
            all_pred.extend((prob > 0.5).float().cpu().numpy())
    val_loss /= len(loader)
    all_lbl  = np.array(all_lbl)
    all_prob = np.array(all_prob)
    all_pred = np.array(all_pred)
    val_f1   = f1_score(all_lbl, all_pred, zero_division=0)
    val_acc  = (all_lbl == all_pred).mean()
    val_auc  = roc_auc_score(all_lbl, all_prob) if len(np.unique(all_lbl)) > 1 else 0.0
    return val_loss, val_f1, val_acc, val_auc, all_lbl, all_prob, all_pred


def main():
    apply_ieee_style()
    colors = get_color_palette()

    print(f"--- Entrenamiento ALUMNO (Fine-Tuning desde Teacher) ---")
    print(f"Device: {DEVICE}")

    if not CSV_PATH.exists():
        print(f"❌ Error: No se encuentra {CSV_PATH}. Generalo con build_alumn_dataset.py")
        return

    df = pd.read_csv(CSV_PATH)
    # El alumno aprende con las pseudo-labels del Oraculo V9
    df['label'] = df['pseudo_label'].astype(float)

    # Excluir fallbacks problematicos del split de train:
    # Son ficheros de drone a SNR muy bajo (<= -14 dB) donde el detector no encontro
    # ningun burst. El fallback mete los 75ms completos con label=1, pero a esa SNR
    # la senal es ruido casi puro -> label noise encubierto.
    # Se mantienen en val/test para evaluacion realista.
    BAD_FALLBACK_SNR_THRESHOLD = -14
    mask_bad = (
        (df['fallback'] == True) &
        (df['label'] == 1) &
        (df['snr'] <= BAD_FALLBACK_SNR_THRESHOLD)
    )
    n_excluded = mask_bad.sum()
    df = df[~mask_bad | (df['split'] != 'train')].copy()
    print(f"Fallbacks problematicos excluidos del train: {n_excluded} rafagas "
          f"(drone, SNR <= {BAD_FALLBACK_SNR_THRESHOLD} dB, sin bursts detectados)")

    df_train = df[df['split'] == 'train'].copy()
    df_val   = df[df['split'] == 'val'].copy()
    df_test  = df[df['split'] == 'test'].copy()

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"Balance Train -> Dron: {(df_train['label']==1).sum()} | Ruido: {(df_train['label']==0).sum()}")

    # Cargar estadisticas fisicas del Teacher para coherencia de normalizacion
    if not TEACHER_CKPT.exists():
        print(f"❌ Error: No se encuentra el checkpoint Teacher en {TEACHER_CKPT}")
        return
    teacher_ckpt = torch.load(TEACHER_CKPT, map_location='cpu', weights_only=False)
    phys_mean = torch.from_numpy(teacher_ckpt['phys_mean'])
    phys_std  = torch.from_numpy(teacher_ckpt['phys_std'])

    ds_train = AlumnDataset(df_train, DATA_DIR, augment=True,  phys_mean=phys_mean, phys_std=phys_std)
    ds_val   = AlumnDataset(df_val,   DATA_DIR, augment=False, phys_mean=phys_mean, phys_std=phys_std)
    ds_test  = AlumnDataset(df_test,  DATA_DIR, augment=False, phys_mean=phys_mean, phys_std=phys_std)

    # WeightedRandomSampler: balancea clases en cada batch sin perder datos
    _labels = df_train['label'].values.astype(int)
    _class_counts = np.bincount(_labels)                        # [n_noise, n_drone]
    _sample_weights = 1.0 / _class_counts[_labels]             # peso inversamente proporcional a la clase
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(_sample_weights).double(),
        num_samples=len(_sample_weights),
        replacement=True
    )
    ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, collate_fn=pad_seq_collate)
    ld_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pad_seq_collate)
    ld_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pad_seq_collate)

    # Modelo: misma arquitectura que el Teacher
    model = AlumnCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    model.load_state_dict(teacher_ckpt['model_state'])
    print("--> Pesos del Teacher cargados correctamente para Fine-Tuning.")

    # Clase imbalance
    n_drone = max((df_train['label'] == 1).sum(), 1)
    n_noise = max((df_train['label'] == 0).sum(), 1)
    pos_w = torch.tensor([n_noise / n_drone], dtype=torch.float32).to(DEVICE)
    print(f"Pos Weight: {pos_w.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_f1    = 0.0
    start_epoch = 0
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': [], 'val_auc': []}

    # === RESUME ===
    if CHECKPOINT_LAST.exists():
        print(f"--> Reanudando desde {CHECKPOINT_LAST.name}...")
        ckpt = torch.load(CHECKPOINT_LAST, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch  = ckpt['epoch'] + 1
        patience_cnt = ckpt.get('patience_cnt', 0)
        if CHECKPOINT_BEST.exists():
            best_f1 = torch.load(CHECKPOINT_BEST, map_location='cpu', weights_only=False).get('val_f1', 0)
        hist_file = OUT_DIR / "history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f)
        print(f"--> Reanudando en Epoca {start_epoch + 1}. Mejor F1 previo: {best_f1:.4f}")

    # ==============================
    # BUCLE DE ENTRENAMIENTO
    # ==============================
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0

        for batch in tqdm(ld_train, desc=f"Ep {epoch+1:02d} Train", leave=False):
            optimizer.zero_grad()
            iq  = batch['iq'].to(DEVICE)
            f   = batch['feats'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logit = model(iq, f)
                loss  = criterion(logit, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(ld_train)

        # Validacion
        val_loss, val_f1, val_acc, val_auc, _, _, _ = run_validation(model, ld_val, criterion)
        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        elapsed = time.time() - t0
        print(f"Ep {epoch+1:02d} | Tr L:{train_loss:.4f} | Val L:{val_loss:.4f} | "
              f"F1:{val_f1:.4f} | AUC:{val_auc:.4f} | Acc:{val_acc:.4f} | T:{elapsed:.1f}s")

        # Guardar BEST
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save({
                'model_state': model.state_dict(),
                'val_f1':      best_f1,
                'val_auc':     val_auc,
                'epoch':       epoch,
                'phys_mean':   phys_mean.numpy(),
                'phys_std':    phys_std.numpy(),
            }, CHECKPOINT_BEST)
            print(f"  --> Best Alumn Model Saved (F1:{best_f1:.4f} | AUC:{val_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= 12:
                print("Early Stopping!")
                break

        # Guardar LAST (siempre)
        torch.save({
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'epoch':           epoch,
            'patience_cnt':    patience_cnt,
        }, CHECKPOINT_LAST)

        # Guardar history.json (para regenerar plots a posteriori)
        with open(OUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Plot live
        plot_live_curves(history, colors, IMG_DIR / "alumn_training_curves.png")

    # ==============================
    # EVALUACION FINAL EN TEST
    # ==============================
    print("\n--- Evaluacion Final en Test Set ---")
    best_ckpt = torch.load(CHECKPOINT_BEST, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['model_state'])

    _, test_f1, test_acc, test_auc, all_lbl, all_prob, all_pred = run_validation(model, ld_test, criterion)

    print(f"Test F1: {test_f1:.4f} | AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_lbl, all_pred, target_names=['Noise', 'Drone'], digits=4))

    # Figuras finales: confusion matrix + ROC
    plot_final_evaluation(all_lbl, all_prob, all_pred, colors, IMG_DIR)

    # Guardar metricas test en JSON
    test_metrics = {
        'test_f1':  float(test_f1),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
    }
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n✅ Entrenamiento Alumno completado.")
    print(f"   Checkpoints -> {CKPT_DIR}")
    print(f"   Figuras     -> {IMG_DIR}")


if __name__ == '__main__':
    main()
