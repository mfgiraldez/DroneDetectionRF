import sys, torch, os, pickle
from pathlib import Path
sys.path.append(r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v5.pipeline import discover_files

DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
OUTPUT_DIR       = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs"

def main():
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)
    storage_paths = pickle.load(open(f"{OUTPUT_DIR}/burst_dataset.pkl", "rb"))
    
    path_map = {Path(sp).stem: sp for sp in storage_paths}
    
    print(f"Sample from train_files: {train_files[0]['path']}")
    print(f"Stem: {Path(train_files[0]['path']).stem}")
    
    print(f"Sample from storage_paths: {storage_paths[0]}")
    print(f"Stem: {Path(storage_paths[0]).stem}")
    
    match = Path(train_files[0]['path']).stem in path_map
    print(f"Match found? {match}")
    
    if not match:
        print("Keys in path_map (first 5):", list(path_map.keys())[:5])

if __name__ == "__main__":
    main()
