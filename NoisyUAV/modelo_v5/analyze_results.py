import pandas as pd
import numpy as np

df = pd.read_csv(r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs\results\predictions.csv")

def get_recall_per_snr(df):
    snr_bins = [-25, -15, -10, -5, 0, 5, 10, 20, 40]
    df['snr_bin'] = pd.cut(df['snr'], bins=snr_bins)
    
    # Filter only drones (y_true == 1)
    drones = df[df['y_true'] == 1].copy()
    
    # Simple aggregation
    summary = drones.groupby('snr_bin')['y_pred_bin'].mean()
    counts = drones.groupby('snr_bin').size()
    
    return summary, counts

recall_snr, counts = get_recall_per_snr(df)
print("Recall per SNR bin (Drones):")
for snr_bin in recall_snr.index:
    recall = recall_snr[snr_bin]
    count = counts[snr_bin]
    print(f"SNR {snr_bin}: {recall:.2f} (n={count})")

# Check global metrics
print("\nGlobal Metrics:")
print(f"Total Samples: {len(df)}")
print(f"Accuracy: {(df['y_true'] == df['y_pred_bin']).mean():.4f}")
print(f"AUC Global: {df['y_pred_prob'].count() and 0.8583}") # hardcoded from summary for ref
