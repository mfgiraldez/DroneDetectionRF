import os
import glob
import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

from NoisyUAV.funciones import DATA_DIR, TARGET_NOISE

def obtener_splits_dataset(
    data_dir: str = DATA_DIR, 
    test_size: float = 0.15, 
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna (df_train, df_val, df_test) estratificados proporcionales
    por clase binaria y grupo SNR.
    """
    pattern = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")
    archivos = glob.glob(os.path.join(data_dir, "IQdata_*.pt"))
    
    data = []
    for f in archivos:
        filename = os.path.basename(f)
        m = pattern.match(filename)
        if m:
            target = int(m.group(2))
            snr = int(m.group(3))
            
            is_drone = 0 if target == TARGET_NOISE else 1
            
            if snr >= 10:
                grupo = 'A'
            elif snr >= -6:
                grupo = 'B'
            else:
                grupo = 'C'
                
            data.append({
                'filepath': f,
                'target_multiclass': target,
                'label': is_drone,
                'snr': snr,
                'grupo': grupo
            })
            
    df = pd.DataFrame(data)
    df['stratify_key'] = df['label'].astype(str) + "_" + df['snr'].astype(str)
    
    df_temp, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['stratify_key']
    )
    
    val_ratio_temp = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_temp, test_size=val_ratio_temp, random_state=random_state, stratify=df_temp['stratify_key']
    )
    
    return df_train, df_val, df_test


class NoisyUAVDataset(Dataset):
    """
    Dataset para clasificación binaria Dron (1) vs Ruido (0).
    """
    def __init__(self, df: pd.DataFrame, grupos_activos: List[str] = ['A', 'B', 'C']):
        self.data = df[df['grupo'].isin(grupos_activos)].reset_index(drop=True)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.data.iloc[idx]
        
        d = torch.load(row['filepath'], map_location="cpu", weights_only=False)
        iq_tensor = d['x_iq']
        label = row['label']
        snr = row['snr']
        
        return iq_tensor, label, snr


def crear_dataloaders(batch_size: int = 32, fase_curriculum: int = 1) -> Dict[str, DataLoader]:
    df_train, df_val, df_test = obtener_splits_dataset()
    
    grupos_train = ['A', 'B'] if fase_curriculum == 1 else ['A', 'B', 'C']
    
    train_dataset = NoisyUAVDataset(df_train, grupos_activos=grupos_train)
    val_dataset   = NoisyUAVDataset(df_val, grupos_activos=['A', 'B', 'C'])
    test_dataset  = NoisyUAVDataset(df_test, grupos_activos=['A', 'B', 'C'])
    
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    }
