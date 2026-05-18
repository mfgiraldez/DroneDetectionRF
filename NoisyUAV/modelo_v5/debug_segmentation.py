import os, sys, torch, logging
from tqdm import tqdm
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, r"c:\repos\DroneDetectionRF")

from NoisyUAV.modelo_v5.pipeline import discover_files, segment_file

DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"

logging.basicConfig(level=logging.INFO)

def main():
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)
    
    # Try files around the crash point (4275)
    start_idx = 4270
    end_idx = 4300
    
    print(f"Testing files from {start_idx} to {end_idx}")
    
    for i in range(start_idx, min(end_idx, len(train_files))):
        fe = train_files[i]
        print(f"[{i}] Processing {fe['filename']} ...", end="", flush=True)
        try:
            bursts = segment_file(fe)
            print(f" OK ({len(bursts)} bursts)")
        except Exception as e:
            print(f" FAILED!")
            print(f"Error in file {fe['path']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
