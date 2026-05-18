"""
Entrenamiento Dual-Stream V3b — GMM DivideMix Loss Weighting
=============================================================
Aplica el núcleo de DivideMix:
1. Warmup: Entrena normalmente por X épocas para que la red aprenda
   las features fáciles (muestras limpias).
2. GMM Fitting: En cada época, evalúa la loss de todas las muestras. 
   Ajusta un modelo de mezcla Gaussiana (GMM) con 2 componentes sobre
   las losses para estimar la probabilidad de que cada muestra sea "limpia".
3. Robust Training: Pondera la loss de cada muestra según su probabilidad
   de ser limpia obtenida por el GMM. Las muestras ruidosas (como WiFi)
   tendrán una probabilidad baja y apenas afectarán al gradiente.
"""
import os, sys, time, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import DualStreamCVCNN
from dataset_dual import DualDataset

# ── Rutas ──────────────────────────────────────────────────────────────────────
OUT_DIR      = os.path.dirname(__file__)
CSV_PATH     = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v3_dual_pro\modelo_alumn_v3a\dataset_v6_pointers.csv" # Reutilizamos V6
DATA_DIR     = r"C:\TFM_data\NoisyUAV\drone_RF_data"
METRICS_JSON = os.path.join(OUT_DIR, "metrics_history.json")
CKPT_DIR     = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR    = os.path.join(OUT_DIR, "plots")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

WARMUP_EPOCHS = 5

def fit_gmm_on_losses(losses):
    """Ajusta un GMM de 2 componentes sobre las losses y devuelve prob(clean)."""
    # Normalizar losses para el GMM
    losses = np.array(losses).reshape(-1, 1)
    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(losses)
    
    # La componente con menor media es la de las muestras "limpias" (low loss)
    clean_idx = gmm.means_.argmin()
    probs = gmm.predict_proba(losses)
    
    # Probabilidad de pertenecer a la componente limpia
    prob_clean = probs[:, clean_idx]
    return prob_clean

def eval_train_losses(model, dl_eval, device, criterion):
    """Evalúa la loss de todo el dataset de train SIN aumentación ni dropout."""
    model.eval()
    losses = np.zeros(len(dl_eval.dataset))
    with torch.no_grad():
        for iq_b, phys_b, lbl_b, idx_b in tqdm(dl_eval, desc="Calculando Losses (GMM)", leave=False):
            iq_b, phys_b, lbl_b = iq_b.to(device), phys_b.to(device), lbl_b.to(device).unsqueeze(1)
            with torch.amp.autocast('cuda'):
                logits, _ = model(iq_b, phys_b)
                # Loss sin reducción para obtener la loss por muestra
                loss = criterion(logits, lbl_b)
                losses[idx_b.numpy()] = loss.squeeze().cpu().numpy()
    return losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Necesitamos dos dataloaders para train: uno con augmentación y otro sin (para evaluar losses)
    ds_train      = DualDataset(CSV_PATH, DATA_DIR, split='train')
    ds_train_eval = DualDataset(CSV_PATH, DATA_DIR, split='train', eval_mode=True)
    ds_val        = DualDataset(CSV_PATH, DATA_DIR, split='val')

    dl_train      = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    dl_train_eval = DataLoader(ds_train_eval, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    dl_val        = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model     = DualStreamCVCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Loss sin reducción para aplicar los pesos del GMM
    criterion_none = nn.BCEWithLogitsLoss(reduction='none') 
    criterion_mean = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    epochs = 50
    start_epoch = 0
    best_f1 = 0.0
    history = []
    
    # Pesos iniciales: 1.0 para todas las muestras
    sample_weights = np.ones(len(ds_train))

    print(f"\nEntrenamiento V3b (GMM DivideMix) | {epochs} épocas | Warmup: {WARMUP_EPOCHS} épocas")
    print("=" * 70)

    for ep in range(start_epoch, epochs):
        t0 = time.time()
        
        # 1. Fase GMM (solo tras warmup)
        if ep >= WARMUP_EPOCHS:
            all_losses = eval_train_losses(model, dl_train_eval, device, criterion_none)
            sample_weights = fit_gmm_on_losses(all_losses)
            clean_ratio = (sample_weights > 0.5).mean()
            print(f"  [GMM] Muestras consideradas limpias (>50%): {clean_ratio*100:.1f}%")

        # 2. Fase de Entrenamiento
        train_loss = 0.0
        model.train()

        for iq_b, phys_b, lbl_b, idx_b in tqdm(dl_train, desc=f'Ep {ep+1:02d} Train', leave=False):
            iq_b, phys_b, lbl_b = iq_b.to(device), phys_b.to(device), lbl_b.to(device).unsqueeze(1)
            
            # Obtener pesos para este batch (si estamos en warmup, son 1.0)
            weights_b = torch.tensor(sample_weights[idx_b.numpy()], dtype=torch.float32).to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(iq_b, phys_b)
                # Loss ponderada por la probabilidad de ser una muestra "limpia"
                unreduced_loss = criterion_none(logits, lbl_b)
                loss = (unreduced_loss * weights_b).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(dl_train)

        # 3. Validación
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for iq_b, phys_b, lbl_b, _ in tqdm(dl_val, desc=f'Ep {ep+1:02d} Val', leave=False):
                iq_b, phys_b, lbl_b = iq_b.to(device), phys_b.to(device), lbl_b.to(device).unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    logits, _ = model(iq_b, phys_b)
                    loss = criterion_mean(logits, lbl_b)
                    val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.extend((probs > 0.5).cpu().numpy().flatten())
                all_labels.extend(lbl_b.cpu().numpy().flatten())

        val_loss /= len(dl_val)
        f1  = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(f1)

        print(f"Ep {ep+1:02d} | L_Tr={train_loss:.4f} | L_Val={val_loss:.4f} | "
              f"F1={f1:.4f} | Acc={acc:.4f} | LR={current_lr:.1e} | {elapsed:.0f}s")

        history.append({
            'epoch': ep + 1, 'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6), 'val_f1': round(f1, 6),
            'val_acc': round(acc, 6), 'lr': current_lr, 'time_s': round(elapsed, 1),
        })
        with open(METRICS_JSON, 'w') as fj: json.dump(history, fj, indent=2)

        # Gráficas
        h_tr = [h['train_loss'] for h in history]
        h_vl = [h['val_loss']   for h in history]
        h_f1 = [h['val_f1']     for h in history]
        h_ac = [h['val_acc']    for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(h_tr, label='Train Loss (Weighted)', color='blue')
        ax1.plot(h_vl, label='Val Loss',   color='red')
        ax1.set_title('Dual-Stream V3b (GMM) — Pérdida')
        ax1.legend(); ax1.grid(True)

        ax2.plot(h_f1, label='Val F1', color='green')
        ax2.plot(h_ac, label='Val Accuracy', color='purple')
        ax2.set_title('Dual-Stream V3b — Métricas')
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
        plt.close(fig)

        # Checkpoints
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'epoch': ep + 1, 'val_f1': f1}, os.path.join(CKPT_DIR, "last_model.pth"))

        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model_state': model.state_dict(), 'epoch': ep + 1, 'val_f1': f1}, 
                       os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"  → Nuevo best model guardado (F1={f1:.4f})")

if __name__ == "__main__":
    main()
