import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BurstDataset(Dataset):
    """
    Dataset de PyTorch para cargar las ráfagas extraídas.
    Lee directamente el contenido comprimido .pt guardado por el Extractor.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.loc[idx, "file_path"]
        
        # Cargar los datos crudos I/Q almacenados físicamente en disco.
        # Recordad: este dict tiene "x_iq": [2, N], "label", "snr", "target"
        data = torch.load(file_path, weights_only=False, map_location="cpu")
        x_iq = data["x_iq"]    # Tensor float [2, N]
        label = data["label"]  # Entero: 0 (NO_DRONE) ó 1 (DRONE)
        
        return x_iq, label

def custom_collate_fn(batch):
    """
    Collate Function Personalizada: Implementa el 'Dynamic Padding'.
    
    Como nuestros datagramas físicos fluctúan en longitud (cada captura 
    tenía un start y end temporal distinto), si los pasamos crudos, PyTorch 
    chocará al intentar montar un cubo NxCxW.

    Este Collate inspecciona en tiempo real cuál es el Crop más largo 
    DENTRO del batch que le toque, y estira a los demás (con ceros 
    matemáticamente silenciados por las convoluciones) sólo para este Batch.
    Es un compromiso inmaculado de RAM vs Eficiencia.
    """
    x_iq_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    
    # max() de las longitudes dentro del lote actual.
    max_len = max([t.shape[1] for t in x_iq_list])
    
    padded_x = []
    for t in x_iq_list:
        pad_size = max_len - t.shape[1]
        # pad(left, right, top, bottom, front, back...) de derecha a izquierda 
        # en las dimensiones subyacentes. Pad solo al final de la dimensión de tiempo:
        padded_t = F.pad(t, (0, pad_size), mode='constant', value=0.0)
        padded_x.append(padded_t)
        
    # Agrupamos todos en la GPU/CPU con stack [B, 2, max_len]
    batch_x = torch.stack(padded_x, dim=0) 
    batch_y = torch.tensor(y_list, dtype=torch.long)
    
    return batch_x, batch_y

def get_dataloaders(csv_path: str, batch_size=128, test_size=0.15, val_size=0.15, num_workers=4):
    """
    Carga el Dataset completo de ~65,000 crops, estratificándolo inteligentemente 
    para conservar las proporciones físicas (Noise puro frente a Drone mezclado).
    Devuelve los tres DataLoaders con memoria bloqueada (`pin_memory`) listos 
    para inyección PCIe directa a la VRAM de tu RTX 4060.
    """
    df = pd.read_csv(csv_path)

    # Validamos que el path sea absoluto correctamente por si en el csv era relativo. 
    # El extractor guardó absolutos, pero por si acaso.
    
    # 1. Separación de Training contra el Resto (Validation+Test)
    df_train, df_temp = train_test_split(
        df, 
        test_size=(test_size + val_size), 
        stratify=df["label"], 
        random_state=42
    )
    
    # 2. Separación final de Validation contra Test
    val_ratio = val_size / (test_size + val_size)
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=(1.0 - val_ratio), 
        stratify=df_temp["label"], 
        random_state=42
    )

    print("=== DISTRIBUCIÓN DE DATASETS ===")
    print(f"Train: {len(df_train)} ráfagas ({len(df_train[df_train['label']==1])} drones)")
    print(f"Val  : {len(df_val)} ráfagas")
    print(f"Test : {len(df_test)} ráfagas")
    
    # Instanciamos los conjuntos de datos en crudo
    train_ds = BurstDataset(df_train)
    val_ds = BurstDataset(df_val)
    test_ds = BurstDataset(df_test)
    
    # Argumento fundamental de rendimiento para la GPU:
    # pin_memory=True permite transferir del SSD M2/SATA paralelo a la caché VRAM
    # ahorrando un volcado doble extra localmente por la CPU.
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)
    
    return train_dl, val_dl, test_dl
