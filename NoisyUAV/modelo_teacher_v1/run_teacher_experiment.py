import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, roc_curve)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Disable GUI to avoid freezing in Windows during ML training
import matplotlib.pyplot as plt

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as TeacherCVCNN, BurstCVCNNDataset as TeacherDataset, pad_seq_collate


from NoisyUAV.funciones.plot_styles import apply_ieee_style, get_color_palette, clean_spines

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ================================
# CONFIGURACION
# ================================
DATA_DIR    = Path(r"C:\TFM_data\NoisyUAV\drone_RF_data")
CSV_PATH    = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1\teacher_dataset_high_snr.csv")
OUT_DIR     = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT      = OUT_DIR / "checkpoints" / "teacher_model_best.pt"
CHECKPOINT_LAST = OUT_DIR / "checkpoints" / "teacher_model_last.pt"  # Ultimo estado completado
CHECKPOINT.parent.mkdir(exist_ok=True)
FIG_DIR     = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPOCHS      = 60
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL   = 16 
LR          = 3e-4
WEIGHT_DECAY= 1e-3
MIXUP_ALPHA = 0.2

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    apply_ieee_style()
    colors = get_color_palette()
    
    print(f"--- Entrenamiento TEACHER (Oráculo High-SNR) ---")
    print(f"Device: {DEVICE}")
    
    if not CSV_PATH.exists():
        print(f"ERROR: No se encuentra {CSV_PATH}. Lanza primero build_teacher_dataset.py")
        return
        
    df = pd.read_csv(CSV_PATH)
    print(f"Bursts totales limpios: {len(df)}")
    
    df_train = df[df['split'] == 'train']
    df_val   = df[df['split'] == 'val']
    df_test  = df[df['split'] == 'test']
    
    # IMPORTANTE: En Train, quitamos los is_dummy 
    df_train = df_train[df_train['is_dummy'] == False].copy()
    
    print(f"Train (sin dummies): {len(df_train)}")
    print(f"Val: {len(df_val)}   Test: {len(df_test)}")
    
    # Dataset Train (computes statistics)
    ds_train = TeacherDataset(df_train, DATA_DIR, augment=True)
    mean, std = ds_train.phys_mean, ds_train.phys_std
    
    ds_val = TeacherDataset(df_val, DATA_DIR, augment=False, phys_mean=mean, phys_std=std)
    ds_test= TeacherDataset(df_test, DATA_DIR, augment=False, phys_mean=mean, phys_std=std)
    
    # IMPORTANTE WINDOWS: num_workers=0
    ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=0, collate_fn=pad_seq_collate)
    ld_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE_VAL,   shuffle=False, num_workers=0, collate_fn=pad_seq_collate)
    
    # Pos weight for Class Imbalance
    n_drone = (df_train['label'] == 1).sum()
    n_noise = (df_train['label'] == 0).sum()
    # Proteccion en caso de que n_drone sea 0 (no deberia)
    n_drone = max(n_drone, 1)
    pos_w = torch.tensor([n_noise / n_drone], dtype=torch.float32).to(DEVICE)
    print(f"Pos Weight calculado: {pos_w.item():.4f}")
    
    model = TeacherCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    best_f1 = 0
    start_epoch = 0
    patience_cnt = 0
    history = {'train_loss':[], 'val_loss':[], 'val_f1':[], 'val_acc':[], 'val_auc':[]}
    
    # === RESUME LOGIC ===
    # Prioridad: LAST (ultimo epoch completado) > BEST (mejor F1)
    ckpt_to_load = None
    if CHECKPOINT_LAST.exists():
        ckpt_to_load = CHECKPOINT_LAST
        print(f"---> Encontrado checkpoint LAST: {CHECKPOINT_LAST.name}. Reanudando desde ultimo epoch completado...")
    elif CHECKPOINT.exists():
        ckpt_to_load = CHECKPOINT
        print(f"---> No hay checkpoint LAST. Usando BEST: {CHECKPOINT.name}...")

    if ckpt_to_load is not None:
        ckpt = torch.load(ckpt_to_load, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])

        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])

        # best_f1 siempre viene del checkpoint BEST (no del LAST)
        if CHECKPOINT.exists():
            ckpt_best = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
            best_f1 = ckpt_best.get('val_f1', 0)
        else:
            best_f1 = ckpt.get('val_f1', 0)

        start_epoch = ckpt.get('epoch', 0) + 1
        patience_cnt = ckpt.get('patience_cnt', 0)
        
        hist_file = OUT_DIR / "history.json"
        if hist_file.exists():
            with open(hist_file, "r") as f:
                history = json.load(f)
                
        print(f"---> Reanudando en Época {start_epoch + 1}. (Mejor F1 anterior: {best_f1:.4f})")
    
    fig_hist, ax_hist = plt.subplots(1, 3, figsize=(15, 4))
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        
        train_loss = 0
        for batch in tqdm(ld_train, desc=f"Ep {epoch+1:02d} Train", leave=False):
            optimizer.zero_grad()
            iq = batch['iq'].to(DEVICE)
            f  = batch['feats'].to(DEVICE)
            lbl= batch['label'].to(DEVICE)
            
            # MixUp implementation
            if np.random.rand() < 0.5:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                idx = torch.randperm(iq.size(0))
                iq = lam * iq + (1-lam) * iq[idx]
                f  = lam * f  + (1-lam) * f[idx]
                lbl_b = lbl[idx]
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logit = model(iq, f)
                    loss = lam * criterion(logit, lbl) + (1-lam) * criterion(logit, lbl_b)
            else:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logit = model(iq, f)
                    loss = criterion(logit, lbl)
                    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(ld_train)
        
        # Validation
        model.eval()
        val_loss = 0
        all_lbl, all_prob, all_pred = [], [], []
        with torch.no_grad():
            for batch in tqdm(ld_val, desc=f"Ep {epoch+1:02d} Val", leave=False):
                iq = batch['iq'].to(DEVICE)
                f  = batch['feats'].to(DEVICE)
                lbl= batch['label'].to(DEVICE)
                
                logit = model(iq, f)
                loss = criterion(logit, lbl)
                val_loss += loss.item()
                
                prob = torch.sigmoid(logit)
                pred = (prob > 0.5).float()
                
                all_lbl.extend(lbl.cpu().numpy())
                all_prob.extend(prob.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                
        val_loss /= len(ld_val)
        val_acc = (np.array(all_pred) == np.array(all_lbl)).mean()
        val_f1  = f1_score(all_lbl, all_pred, zero_division=0)
        val_auc = roc_auc_score(all_lbl, all_prob) if len(np.unique(all_lbl)) > 1 else 0.0
        
        scheduler.step(val_f1)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"Ep {epoch+1:02d} | Tr L: {train_loss:.4f} | Val L: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | T: {time.time()-t0:.1f}s")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'phys_mean': mean.numpy(),
                'phys_std': std.numpy(),
                'val_f1': best_f1,
                'val_auc': val_auc,
                'epoch': epoch
            }, CHECKPOINT)
            print(f"  --> Best Teacher Model Saved (Val F1: {best_f1:.4f} | AUC: {val_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= 12:
                print("Early Stopping!")
                break

        # Guardar SIEMPRE el ultimo estado completado (independiente de si mejora F1)
        # Esto permite reanudar desde el epoch exacto en caso de interrupcion
        torch.save({
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'phys_mean':       mean.numpy(),
            'phys_std':        std.numpy(),
            'val_f1':          val_f1,
            'epoch':           epoch,
            'patience_cnt':    patience_cnt,
        }, CHECKPOINT_LAST)

        # Guardar history.json en cada epoch (no solo al final)
        with open(OUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
                
        # Live Plotting
        ax_hist[0].clear()
        ax_hist[1].clear()
        ax_hist[2].clear()
        ax_hist[0].plot(history['train_loss'], c=colors['blue'], label='Train Loss')
        ax_hist[0].plot(history['val_loss'], c=colors['orange'], label='Val Loss')
        ax_hist[0].set_title("Loss", fontweight='bold')
        ax_hist[0].legend()
        clean_spines(ax_hist[0])
        
        ax_hist[1].plot(history['val_acc'], c=colors['green'], label='Val Acc')
        ax_hist[1].plot(history['val_f1'], c=colors['purple'], label='Val F1')
        ax_hist[1].set_ylim(0, 1)
        ax_hist[1].set_title("F1 & Accuracy", fontweight='bold')
        ax_hist[1].legend()
        clean_spines(ax_hist[1])

        ax_hist[2].plot(history['val_auc'], c=colors['red'], label='Val AUC-ROC')
        ax_hist[2].set_ylim(0, 1)
        ax_hist[2].set_title("AUC-ROC", fontweight='bold')
        ax_hist[2].legend()
        clean_spines(ax_hist[2])
        
        plt.tight_layout()
        fig_hist.savefig(FIG_DIR / "teacher_training_curves_ieee.png", dpi=300, bbox_inches='tight')
        
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── EVALUACION FINAL EN TEST ──────────────────────────────────────────────
    print("\n--- Evaluacion Final en Test Set (mejor modelo) ---")
    best_ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['model_state'])
    model.eval()

    ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE_VAL, shuffle=False,
                         num_workers=0, collate_fn=pad_seq_collate)

    all_lbl_t, all_prob_t, all_pred_t = [], [], []
    with torch.no_grad():
        for batch in ld_test:
            iq  = batch['iq'].to(DEVICE)
            f   = batch['feats'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            prob = torch.sigmoid(model(iq, f))
            all_lbl_t.extend(lbl.cpu().numpy())
            all_prob_t.extend(prob.cpu().numpy())
            all_pred_t.extend((prob > 0.5).float().cpu().numpy())

    all_lbl_t  = np.array(all_lbl_t)
    all_prob_t = np.array(all_prob_t)
    all_pred_t = np.array(all_pred_t)

    test_f1  = f1_score(all_lbl_t, all_pred_t, zero_division=0)
    test_auc = roc_auc_score(all_lbl_t, all_prob_t)
    test_acc = (all_lbl_t == all_pred_t).mean()
    print(f"Test F1: {test_f1:.4f} | AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")
    print(classification_report(all_lbl_t, all_pred_t,
                                target_names=['Noise','Drone'], digits=4))

    # Confusion Matrix
    fig_eval, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(all_lbl_t, all_pred_t)
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    fig_eval.colorbar(im, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Test)", fontweight='bold')
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(['Noise','Drone']); axes[0].set_yticklabels(['Noise','Drone'])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                         fontsize=14, fontweight='bold',
                         color='white' if cm[i,j] > thresh else 'black')

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_lbl_t, all_prob_t)
    axes[1].plot(fpr, tpr, c=colors['blue'], lw=2, label=f'AUC = {test_auc:.4f}')
    axes[1].plot([0,1],[0,1],'k--', lw=1)
    axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.02])
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve (Test)", fontweight='bold')
    axes[1].legend(); clean_spines(axes[1])
    plt.tight_layout()
    fig_eval.savefig(FIG_DIR / "teacher_test_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close(fig_eval)

    # Guardar metricas finales
    import json as _json
    metrics = {
        "test": {"f1": float(test_f1), "auc": float(test_auc), "acc": float(test_acc)}
    }
    with open(OUT_DIR / "teacher_eval_metrics.json", "w") as f:
        _json.dump(metrics, f, indent=2)

    print("\nEntrenamiento Teacher completado!")
    print(f"  Figuras -> {FIG_DIR}")

if __name__ == '__main__':
    main()
