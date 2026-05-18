import sys, torch, os, pickle
from pathlib import Path
sys.path.append(r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v5.pipeline import discover_files, BagDataset, BurstEncoder

DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
OUTPUT_DIR       = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs"

def main():
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)
    storage_paths = pickle.load(open(f"{OUTPUT_DIR}/burst_dataset.pkl", "rb"))
    
    encoder = BurstEncoder()
    ds = BagDataset(train_files, storage_paths, encoder)
    
    print(f"Dataset length: {len(ds)}")
    
    # Check first 5 samples
    empty_count = 0
    for i in range(100):
        item = ds[i]
        if item['n_instances'] == 1 and torch.all(item['iq'] == 0):
            empty_count += 1
            
    print(f"Empty bags in first 100: {empty_count}")

if __name__ == "__main__":
    main()
