import sys, torch, os, pickle
from pathlib import Path
sys.path.append(r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v5.pipeline import discover_files, BagDataset, BurstEncoder, NOISE_CLASS

DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
OUTPUT_DIR       = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs"

def main():
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)
    storage_paths = pickle.load(open(f"{OUTPUT_DIR}/burst_dataset.pkl", "rb"))
    
    encoder = BurstEncoder()
    ds = BagDataset(train_files, storage_paths, encoder)
    
    empty_drones = 0
    total_drones = 0
    for i in range(len(ds)):
        item = ds[i]
        is_drone = item['class'] != NOISE_CLASS and item['class'] != 4 # target 4 is interference
        if is_drone:
            total_drones += 1
            if item['n_instances'] == 1 and torch.all(item['iq'] == 0):
                empty_drones += 1
        
        if i > 500: break # Just a sample
            
    print(f"Total drones sampled: {total_drones}")
    print(f"Empty drone bags: {empty_drones}")

if __name__ == "__main__":
    main()
