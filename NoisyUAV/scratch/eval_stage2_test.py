import torch
import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add path to load local modules
sys.path.insert(0, r'c:\repos\DroneDetectionRF\NoisyUAV')
from funciones.dataset_stage2 import get_dataloaders
from modelos.cvcnn import ComplexConv1DNet

def apply_agc_batch(inputs: torch.Tensor) -> torch.Tensor:
    for b in range(inputs.size(0)):
        max_val = torch.max(torch.abs(inputs[b]))
        if max_val > 0:
            inputs[b] = inputs[b] / max_val
    return inputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = r"C:\TFM_data\NoisyUAV\stage2\metadata_stage2.csv"
    model_path = r"C:\TFM_data\NoisyUAV\stage2_norm\checkpoints\cvcnn_best_model.pth"
    
    _, _, test_loader = get_dataloaders(csv_path=csv_path, batch_size=32, num_workers=0)
    
    model = ComplexConv1DNet(num_classes=2, pool_output_size=64, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = apply_agc_batch(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    print(f"\n[RESULTS] TEST RESULTS [STAGE 2 NORM]:")
    print(f"Test Total Samples: {test_total}")
    print(f"Accuracy: {100. * test_correct / test_total:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=["NO_DRONE(0)", "DRONE(1)"]))

if __name__ == "__main__":
    main()
