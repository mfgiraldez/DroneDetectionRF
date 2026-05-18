"""
entrenar_stage2.py

Script principal de entrenamiento para la CV-CNN (Deep Learning Stage 2).
Recoge las ~65,000 ráfagas preparadas en disco, invoca a la GPU, y entrena
el detector final. Incluye Learning Rate Scheduling, Checkpointing robusto 
(permite reanudar el entrenamiento si se corta) y generación de métricas Visuales (Plotly/Matplotlib).
"""

import os
import sys
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Asegurar importe de los módulos del proyecto raíz, sin importar desde dónde se ejecute
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from funciones.dataset_stage2 import get_dataloaders
from modelos.cvcnn import ComplexConv1DNet

# Hiperparámetros base (optimizados para RTX 4060)
BATCH_SIZE = 16           # Tamaño físico en VRAM (muy bajo para evitar OOM con audios anómalos largos)
ACCUMULATION_STEPS = 8    # 16 * 8 = 128. Matemáticamente idéntico a entrenar con Batch Size de 128.
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Rutas
STAGE2_OUT_DIR = r"C:\TFM_data\NoisyUAV\stage2"
CSV_METADATA = os.path.join(STAGE2_OUT_DIR, "metadata_stage2.csv")
CHECKPOINT_DIR = os.path.join(STAGE2_OUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(STAGE2_OUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_training_history(history, save_path):
    """Grafica curvas de Loss y Accuracy y las guarda en disco."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Gráfica de Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.title("Curva de Loss (Cross Entropy)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Acc", marker="o")
    plt.title("Curva de Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.axhline(100, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Genera la matriz de confusión del Test ciego."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Noise (0)", "Drone (1)"], 
                yticklabels=["Noise (0)", "Drone (1)"])
    plt.title("Test Set Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_latest_checkpoint():
    """Encuentra el checkpoint más reciente en el directorio."""
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    # Ordenar por el número de epoch (extraerlo del string)
    checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(".pth")[0]))
    return checkpoints[-1]

def main():
    print("=" * 60)
    print("  STAGE 2: ENTRENANDO LA CV-CNN BANDA BASE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Mapeado: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    print("\nInicializando PyTorch DataLoaders (con Dynamic Padding)...")
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

    # Variables de estado
    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ─────────── RESTORE CHECKPOINT ───────────
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt is not None:
        print(f"\n🔄 Checkpoint Detectado: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"Reanudando desde Epoch {start_epoch} (Mejor Acc: {best_val_acc:.2f}%)")
    else:
        print("\nEmpezando entrenamiento desde cero.")


    print("\n🚀 [PLAY] Comenzando iteraciones de Propagation...")
    
    # Check extra por si el usuario aumentó EPOCHS tras haber acabado antes
    if start_epoch > EPOCHS:
        print(f"\nEl modelo ya estaba entrenado hasta el Epoch {start_epoch-1}.")
        print("Aumenta la variable EPOCHS si quieres seguir entrenando.")
    
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

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Gradient Accumulation: escalamos el gradiente al tamaño ficticio
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # Solo actualizamos pesos cada 8 batch_i
            if (batch_i + 1) % ACCUMULATION_STEPS == 0 or (batch_i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Recuperamos el loss real para no falsear los logs reportados
            train_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_i % 20 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*train_correct/train_total:.1f}%"})

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
        
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:05.2f}% (Loss {train_loss_avg:.4f}) | Val Acc: {val_acc:05.2f}% (Loss {val_loss_avg:.4f})")
        
        # Step LR Scheduler
        scheduler.step(val_acc)

        # Atualizar gráfico de Entrenamiento tras cada Epoch
        plot_training_history(history, os.path.join(PLOTS_DIR, "training_curves.png"))

        # --- GUARDADO CHECKPOINT ROBUSTO ---
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
        
        # Guardamos en cada epoch su archivo (ideal por si crashea Windows, conservas todo)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:02d}.pth")
        torch.save(checkpoint_dict, ckpt_path)
        
        if is_best:
            best_model_path = os.path.join(CHECKPOINT_DIR, "cvcnn_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 ¡Nuevo record! Mejor Val_Acc guardado en cvcnn_best_model.pth")
            
    # ===== INFERENCIA FINAL EN SET OCULTO =====
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
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    test_acc = 100. * test_correct / test_total
    print(f"\n🎯 Métrica final (Ciego): ACCURACY = {test_acc:.2f}%")
    
    # Reportes detallados y matriz gráfica
    report = classification_report(all_targets, all_preds, target_names=["NO_DRONE(0)", "DRONE(1)"])
    print("\nReporte de Clasificación:\n" + report)
    
    cm_path = os.path.join(PLOTS_DIR, "test_confusion_matrix.png")
    plot_confusion_matrix(all_targets, all_preds, cm_path)
    print(f"Matriz de confusión guardada en: {cm_path}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
