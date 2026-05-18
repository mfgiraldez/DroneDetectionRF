import sys, os
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

from NoisyUAV.modelo_v5.pipeline import segment_file, discover_files, DATA_ROOT, TEST_SPLIT_FILE

def test_segmentation():
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)
    print(f"Found {len(train_files)} train files.")
    
    if not train_files:
        print("No files found!")
        return

    test_entry = train_files[0]
    print(f"Testing on {test_entry['path']} (SNR={test_entry['snr']}, Class={test_entry['class']})")
    
    bursts = segment_file(test_entry)
    print(f"Extracted {len(bursts)} bursts.")
    for i, b in enumerate(bursts):
        print(f"  Burst {i}: length={len(b['iq'])}")

if __name__ == "__main__":
    test_segmentation()
