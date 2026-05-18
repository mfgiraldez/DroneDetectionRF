"""
entrenar_stage2_norm.py

Script paralelo de entrenamiento para la CV-CNN (Deep Learning Stage 2) implementando 
AGC (Normalización Min-Max) sobre la marcha para asegurar generalización Cross-Dataset.
Mantiene intactos los modelos y logs originales guardando todo en stage2_norm.
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Asegurar importe de los módulos del proyecto raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from funciones.dataset_stage2 import get_dataloaders
from modelos.cvcnn import ComplexConv1DNet

# Hiperparámetros
BATCH_SIZE = 16          
ACCUMULATION_STEPS = 8    
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# RUTAS ABSOLUTAMENTE NUEVAS PARA NO PISAR EL TRABAJO ANTERIOR
STAGE2_METADATA_DIR = r"C:\TFM_data\NoisyUAV\stage2" # El CSV está aquí
CSV_METADATA = os.path.join(STAGE2_METADATA_DIR, "metadata_stage2.csv")

# NUEVOS DIRECTORIOS DE SALIDA
STAGE2_OUT_DIR = r"C:\TFM_data\NoisyUAV\stage2_norm"
CHECKPOINT_DIR = os.path.join(STAGE2_OUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(STAGE2_OUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def apply_agc_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Aplica Control Automático de Ganancia (Min-Max) a todo el Batch de forma independiente."""
    for b in range(inputs.size(0)):
        max_val = torch.max(torch.abs(inputs[b]))
        if max_val > 0:
            inputs[b] = inputs[b] / max_val
    return inputs

def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.title("Curva de Loss (Cross Entropy) - Normalizado")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Acc", marker="o")
    plt.title("Curva de Accuracy - Normalizado")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.axhline(100, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Noise (0)", "Drone (1)"], 
                yticklabels=["Noise (0)", "Drone (1)"])
    plt.title("Test Set Confusion Matrix (Normalized Model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(".pth")[0]))
    return checkpoints[-1]

def main():
    print("=" * 60)
    print("  STAGE 2 [NORM]: ENTRENANDO LA CV-CNN BANDA BASE UNIVERSAL")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    print("\nInicializando PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=CSV_METADATA,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    model = ComplexConv1DNet(num_classes=2, pool_output_size=64, dropout=0.5)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt is not None:
        print(f"\n🔄 Reanudando desde: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        if 'history' in checkpoint:
            history = checkpoint['history']
    else:
        print("\nEmpezando entrenamiento estabilizado desde cero.")

    for epoch in range(start_epoch, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False)
        for batch_i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = apply_agc_batch(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (batch_i + 1) % ACCUMULATION_STEPS == 0 or (batch_i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = apply_agc_batch(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:05.2f}% (Loss {train_loss_avg:.4f}) | Val Acc: {val_acc:05.2f}%")
        scheduler.step(val_acc)

        plot_training_history(history, os.path.join(PLOTS_DIR, "training_curves.png"))

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:02d}.pth")
        torch.save(checkpoint_dict, ckpt_path)
        
        if is_best:
            best_model_path = os.path.join(CHECKPOINT_DIR, "cvcnn_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            
    print(f"\n✅ Entrenamiento completado. Evaluando 'cvcnn_best_model.pth' en el Test Set Ciego...")
    
    best_model_path = os.path.join(CHECKPOINT_DIR, "cvcnn_best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = apply_agc_batch(inputs)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    test_acc = 100. * test_correct / test_total
    print(f"\n🎯 Métrica final (Ciego) [Normalizado]: ACCURACY = {test_acc:.2f}%")
    
    report = classification_report(all_targets, all_preds, target_names=["NO_DRONE(0)", "DRONE(1)"])
    print("\nReporte de Clasificación:\n" + report)
    
    cm_path = os.path.join(PLOTS_DIR, "test_confusion_matrix.png")
    plot_confusion_matrix(all_targets, all_preds, cm_path)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
