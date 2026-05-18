import os
import re
import pandas as pd
import random

def create_balanced_test_set(data_dir, output_csv, samples_per_model_snr=12):
    files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    # Regex to extract target and snr
    pattern = re.compile(r'target(\d+)_snr(-?\d+)')
    
    data = []
    for f in files:
        match = pattern.search(f)
        if match:
            target = int(match.group(1))
            snr = int(match.group(2))
            data.append({'filename': f, 'target': target, 'snr': snr})
            
    df = pd.DataFrame(data)
    
    drone_targets = [0, 1, 2, 3, 5, 6]
    noise_target = 4
    
    test_filenames = []
    
    snr_levels = sorted(df['snr'].unique())
    
    random.seed(42) # For reproducibility if run again
    
    for snr in snr_levels:
        # Sample drones
        for target in drone_targets:
            subset = df[(df['target'] == target) & (df['snr'] == snr)]
            if len(subset) < samples_per_model_snr:
                print(f"Warning: Not enough samples for target {target}, snr {snr}. Have {len(subset)}, requested {samples_per_model_snr}")
                sampled = subset['filename'].tolist()
            else:
                sampled = subset.sample(n=samples_per_model_snr, random_state=42)['filename'].tolist()
            test_filenames.extend(sampled)
            
        # Sample noise (balanced with total drones for this SNR level)
        # Total drones per SNR = 6 * samples_per_model_snr = 72
        num_noise_to_sample = len(drone_targets) * samples_per_model_snr
        noise_subset = df[(df['target'] == noise_target) & (df['snr'] == snr)]
        if len(noise_subset) < num_noise_to_sample:
            print(f"Warning: Not enough noise samples for snr {snr}. Have {len(noise_subset)}, requested {num_noise_to_sample}")
            sampled_noise = noise_subset['filename'].tolist()
        else:
            sampled_noise = noise_subset.sample(n=num_noise_to_sample, random_state=42)['filename'].tolist()
        test_filenames.extend(sampled_noise)
        
    test_df = df[df['filename'].isin(test_filenames)]
    test_df.to_csv(output_csv, index=False)
    print(f"Test set saved to {output_csv} with {len(test_df)} samples.")
    
    # Create the complementary train/val set
    train_val_df = df[~df['filename'].isin(test_filenames)]
    train_val_csv = output_csv.replace('test_set.csv', 'train_val_split.csv')
    train_val_df.to_csv(train_val_csv, index=False)
    print(f"Train/Val set saved to {train_val_csv} with {len(train_val_df)} samples.")

if __name__ == '__main__':
    DATA_DIR = r'C:\TFM_data\NoisyUAV\drone_RF_data'
    OUTPUT_CSV = r'C:\TFM_data\NoisyUAV\ground_truth_test_set.csv'
    create_balanced_test_set(DATA_DIR, OUTPUT_CSV)
