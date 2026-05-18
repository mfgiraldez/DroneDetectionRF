import sys, os, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v4.model_v4 import DualStreamCVCNN_V4

# Config
GOLDEN_CSV = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\figures_golden"
WINDOW_LEN = 131072
FS = 14e6

TARGET_NAMES = {0: "DJI", 1: "FutabaT14", 2: "FutabaT7", 3: "Graupner", 5: "Taranis", 6: "Turnigy", 4: "Ruido"}

def sliding_window_inference(model, iq_full, device):
    """ Escanea el fichero de 75ms con ventanas de 9.4ms solapadas 50% """
    # iq_full: [2, 1048576]
    L = iq_full.shape[1]
    step = WINDOW_LEN // 2
    windows = []
    
    for start in range(0, L - WINDOW_LEN + 1, step):
        crop = iq_full[:, start : start + WINDOW_LEN].clone()
        # Normalizar ventana
        rms = torch.sqrt(torch.mean(crop**2) + 1e-12)
        windows.append(crop / rms)
    
    batch = torch.stack(windows).to(device)
    # Features físicas ficticias para el test ciego (basadas en el suelo de ruido global)
    # En el test no tenemos el z_peak del burst, usamos 0.0 para simular búsqueda ciega
    global_nf = torch.median(torch.sqrt(torch.mean(iq_full**2, dim=0))) # Proxy nf
    phys = torch.zeros(batch.size(0), 4).to(device)
    phys[:, 0] = global_nf / 10.0
    
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits, _ = model(batch, phys)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    return np.max(probs) # El veredicto es el máximo detectado

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Cargar Modelo
    model = DualStreamCVCNN_V4().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Cargar Golden Set
    df = pd.read_csv(GOLDEN_CSV)
    results = []
    
    print(f"Evaluando Golden Set ({len(df)} archivos) por Ventana Deslizante...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fpath = os.path.join(DATA_DIR, row['filename'])
        try:
            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
            
            prob_max = sliding_window_inference(model, iq, device)
            
            results.append({
                'filename': row['filename'],
                'label': row['target'], # Original target (4=noise, others=drone)
                'snr': row['snr'],
                'prob_drone': prob_max,
                'pred': 1 if prob_max >= 0.5 else 0,
                'gt_bin': 0 if row['target'] == 4 else 1
            })
        except: continue
        
    res_df = pd.DataFrame(results)
    res_df['correct'] = (res_df['pred'] == res_df['gt_bin']).astype(int)
    
    # Generar Heatmap Drones
    drones = res_df[res_df['gt_bin'] == 1].copy()
    drones['Target'] = drones['label'].map(TARGET_NAMES)
    hm = drones.pivot_table(index='Target', columns='snr', values='correct', aggfunc='mean')
    plt.figure(figsize=(14, 6))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("V4 Sliding Window Evaluation -- Drone Recall x SNR (Golden Set)")
    plt.savefig(os.path.join(OUT_DIR, "heatmap_recall_golden.png"))
    
    # Curva P-R
    plt.figure(figsize=(8, 6))
    y_true = res_df['gt_bin'].values
    y_prob = res_df['prob_drone'].values
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(r, p, label=f"V4 Neural Scanner (AUC={auc(r, p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "pr_curve_golden.png"))
    
    # Métricas Globales
    metrics = {
        "f1": float(f1_score(y_true, res_df['pred'])),
        "acc": float(accuracy_score(y_true, res_df['pred']))
    }
    print(f"Resultado Final Golden Set: F1={metrics['f1']:.4f} | Acc={metrics['acc']:.4f}")
    with open(os.path.join(OUT_DIR, "metrics_golden.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
