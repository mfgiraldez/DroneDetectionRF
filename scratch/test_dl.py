import sys, time
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNNDataset, pad_seq_collate

CSV_PATH = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_burst_v2\bursts_dataset.csv")
DATA_DIR = Path(r"C:\TFM_data\NoisyUAV\drone_RF_data")

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df_train = df[(df['split'] == 'train') & (df['is_dummy'] == False)].copy()
    
    print(f"Dataset Size: {len(df_train)}")
    ds = BurstCVCNNDataset(df_train, DATA_DIR, augment=True)
    
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=pad_seq_collate)
    
    print("Iterating...")
    t0 = time.time()
    for i, batch in enumerate(ld):
        t1 = time.time()
        print(f"Batch {i:03d} | Size: {batch['iq'].shape} | Time: {t1-t0:.3f}s")
        t0 = t1
        if i >= 10: break

if __name__ == '__main__':
    main()
