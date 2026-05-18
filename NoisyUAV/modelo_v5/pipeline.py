import os, sys, pickle, json, logging, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              ConfusionMatrixDisplay, roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"        # directorio raíz plano
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv" 
OUTPUT_DIR       = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs"

# ── SDR / señal ───────────────────────────────────────────────────────────────
FS               = 14e6          # frecuencia de muestreo (Hz)
FILE_DURATION_MS = 75            # duración de cada fichero (ms)
N_SAMPLES_FILE   = int(FS * FILE_DURATION_MS / 1000)   # muestras por fichero

# ── Segmentador ───────────────────────────────────────────────────────────────
ENTROPY_NPERSEG  = 2048
MIN_BURST_MS     = 0.4
MAX_BURST_SAMPLES = int(FS * 0.01)  # burst máximo: 10 ms
MIN_BURST_SAMPLES = int(FS * 0.0004) # 0.4 ms

# ── Filtro de interferencias ──────────────────────────────────────────────────
FILTER_SNR_THRESHOLD = 18        # SNR mínima para pseudo-etiquetado de bursts como "dron seguro"
FILTER_CONF_THRESHOLD = 0.6      # umbral de confianza para pasar un burst al MIL

# ── CV-CNN (embeddings) ───────────────────────────────────────────────────────
EMBED_DIM        = 128           # dimensión del embedding final de cada burst
IQ_INPUT_LEN     = 1024         # longitud fija de la ventana IQ por burst (resampleado si distinto)
PSD_N_BINS       = 512           # bins del espectro Welch

# ── ABMIL ────────────────────────────────────────────────────────────────────
ATTN_DIM         = 64            # dimensión interna de la red de atención
ATTN_LAMBDA      = 0.05          # peso de la penalización de entropía de atención
BAG_MAX_INSTANCES = 32           # máximo de bursts por bag (padding/truncado)
BATCH_SIZE       = 16            # bags por batch
LR               = 1e-4
EPOCHS           = 60
PATIENCE         = 10            # early stopping

# ── Evaluación ───────────────────────────────────────────────────────────────
SNR_BINS         = list(range(-20, 32, 2))   # -20, -18, …, 30
DRONE_CLASSES    = [0, 1, 2, 3, 5, 6]
NOISE_CLASS      = 4

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/results").mkdir(parents=True, exist_ok=True)
BURST_STORAGE_DIR = Path(f"{OUTPUT_DIR}/burst_cache")
BURST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MÓDULO 1 — Carga de ficheros IQ
# ==============================================================================

def load_iq_file(filepath: str) -> np.ndarray:
    """Lee el fichero .pt y devuelve array numpy complex64"""
    d = torch.load(filepath, map_location='cpu', weights_only=False)
    # d['x_iq'] es [2, N]
    iq_tensor = d['x_iq'].numpy()
    iq_complex = iq_tensor[0] + 1j * iq_tensor[1]
    return iq_complex.astype(np.complex64)


def discover_files(data_root: str, test_split_file: str):
    """Descubre los ficheros desde el CSV ground truth y completa los metadatos."""
    df_test = pd.read_csv(test_split_file)
    test_paths = set(df_test['filename'].tolist())

    train_files, test_files = [], []
    
    # Listar ficheros en el directorio root
    all_files = list(Path(data_root).glob("*.pt"))
    for fpath in all_files:
        name = fpath.name
        # Parse target y snr desde el nombre: IQdata_sample{N}_target{T}_snr{S}.pt
        m_t = re.search(r'target(\d+)', name)
        m_s = re.search(r'snr(-?\d+)', name)
        if m_t and m_s:
            t_class = int(m_t.group(1))
            snr_val = int(m_s.group(1))
            entry = {
                'path': str(fpath),
                'class': t_class,
                'snr': snr_val,
                'filename': name
            }
            if name in test_paths:
                test_files.append(entry)
            else:
                train_files.append(entry)

    log.info(f"Ficheros train/val: {len(train_files)}  |  test: {len(test_files)}")
    return train_files, test_files


# ==============================================================================
# MÓDULO 2 — Segmentador de bursts
# ==============================================================================

def cfar_spectral_detector(iq: np.ndarray, fs: float,
                            guard_cells: int = 4,
                            reference_cells: int = 16,
                            pfa: float = 1e-3) -> list[tuple[int,int]]:
    """CFAR 1D sobre la PSD para fallback cuando la SNR es muy baja."""
    freqs, psd = welch(iq, fs=fs, nperseg=1024, return_onesided=False)
    psd_db = 10 * np.log10(np.abs(psd) + 1e-12)

    detections = []
    N = len(psd_db)
    for i in range(reference_cells + guard_cells, N - reference_cells - guard_cells):
        left  = psd_db[i - reference_cells - guard_cells : i - guard_cells]
        right = psd_db[i + guard_cells + 1 : i + guard_cells + reference_cells + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        threshold = noise_est - 10 * np.log10(-np.log(pfa) / reference_cells)
        if psd_db[i] > threshold:
            detections.append((0, len(iq)))
            break 
    return detections

def segment_file(file_entry: dict) -> str:
    """Extrae bursts usando detectar_bursts y CFAR fallback."""
    d = torch.load(file_entry['path'], map_location='cpu', weights_only=False)
    iq_tensor = d['x_iq'].float()
    iq_complex = (iq_tensor[0] + 1j * iq_tensor[1]).numpy().astype(np.complex64)

    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts_info = detectar_bursts(
        iq_tensor, fs=FS, nperseg=ENTROPY_NPERSEG, z_thresh=1.5,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=0.5, min_z_abs=2.0,
        bg_mult=4, max_bins_frac=1.0, smooth_ms=0.2)

    bursts = []
    for idx, b in enumerate(bursts_info):
        s_idx = int(b['t0'] * FS / 1000.0)
        e_idx = int(b['t1'] * FS / 1000.0)
        
        try:
            burst_iq = iq_complex[s_idx:e_idx]
            if len(burst_iq) < MIN_BURST_SAMPLES:
                continue
            if len(burst_iq) > MAX_BURST_SAMPLES:
                burst_iq = burst_iq[:MAX_BURST_SAMPLES]

            bursts.append({
                "iq":     burst_iq,
                "class":  file_entry['class'],
                "snr":    file_entry['snr'],
                "file_path": file_entry['path'],
                "burst_idx": idx,
                "t0": b['t0'],
                "t1": b['t1']
            })
        except Exception as ex:
            log.warning(f"Error procesando burst {idx} en {file_entry['path']}: {ex}")
            continue

    # Guardado eficiente en disco
    file_id = Path(file_entry['path']).stem
    storage_path = BURST_STORAGE_DIR / f"{file_id}.pkl"
    with open(storage_path, 'wb') as f:
        pickle.dump(bursts, f)
    
    return str(storage_path)


def segment_files_with_checkpoint(files: list[dict], cache_file: str, desc: str = "Segmentando"):
    """Segmenta ficheros con checkpointing para permitir reanudación (MODO EFICIENTE EN RAM)."""
    checkpoint_file = cache_file + ".checkpoint"
    all_storage_paths = []
    processed_paths = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                all_storage_paths = checkpoint_data['storage_paths']
                processed_paths = checkpoint_data['processed_paths']
            log.info(f"Reanudando {desc}: {len(processed_paths)} ficheros ya procesados.")
        except Exception as e:
            log.warning(f"Error cargando checkpoint {checkpoint_file}: {e}. Empezando de cero.")

    if len(processed_paths) >= len(files) and os.path.exists(cache_file):
        log.info(f"Cargando {desc} desde cache final...")
        return pickle.load(open(cache_file, 'rb'))

    remaining_files = [f for f in files if f['path'] not in processed_paths]
    
    if not remaining_files:
        if all_storage_paths:
            pickle.dump(all_storage_paths, open(cache_file, 'wb'))
        return all_storage_paths

    pbar = tqdm(remaining_files, desc=desc)
    save_every = 100
    
    for i, fe in enumerate(pbar):
        try:
            storage_path = segment_file(fe)
            all_storage_paths.append(storage_path)
            processed_paths.add(fe['path'])
        except Exception as e:
            log.error(f"Error crítico en {fe['path']}: {e}")
            continue

        if (i + 1) % save_every == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'storage_paths': all_storage_paths, 'processed_paths': processed_paths}, f)
            
    pickle.dump(all_storage_paths, open(cache_file, 'wb'))
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return all_storage_paths


# ==============================================================================
# MÓDULO 3 — Extractor de features físicas
# ==============================================================================

def extract_physical_features(burst: dict) -> np.ndarray:
    iq = burst['iq'].astype(np.complex64)
    amp = np.abs(iq)

    _, psd = welch(iq, fs=FS, nperseg=min(256, len(iq)), return_onesided=False)
    psd = np.abs(psd)
    psd_norm = psd / (psd.sum() + 1e-12)

    psd_db = 10 * np.log10(psd + 1e-12)
    peak = psd_db.max()
    bw_mask = psd_db > (peak - 10)
    bw_inst = bw_mask.sum() * (FS / len(psd))

    duration_us = len(iq) / FS * 1e6
    duty_cycle = duration_us / (FILE_DURATION_MS * 1e3)

    freqs = np.fft.fftfreq(len(psd), d=1/FS)
    fc_est = freqs[np.argmax(psd)]

    fc_delta = 0.0

    psd_kurtosis = float(kurtosis(psd_norm))
    psd_skewness = float(skew(psd_norm))
    iq_kurtosis = float(kurtosis(amp))

    geo_mean = np.exp(np.log(psd + 1e-12).mean())
    arith_mean = psd.mean() + 1e-12
    spectral_flatness = float(geo_mean / arith_mean)

    envelope_std = float(amp.std())

    real_part = iq.real
    zcr = float(((real_part[:-1] * real_part[1:]) < 0).sum() / len(real_part))

    return np.array([bw_inst, duration_us, duty_cycle, fc_est, fc_delta,
                     psd_kurtosis, psd_skewness, iq_kurtosis,
                     spectral_flatness, envelope_std, zcr], dtype=np.float32)

def compute_fc_deltas(storage_paths: list[str]) -> list[str]:
    for spath in storage_paths:
        with open(spath, 'rb') as f:
            bursts = pickle.load(f)
        bursts.sort(key=lambda x: x['burst_idx'])
        feats = [extract_physical_features(b) for b in bursts]
        for i, (b, f_vec) in enumerate(zip(bursts, feats)):
            if i > 0:
                f_vec[4] = abs(feats[i][3] - feats[i-1][3]) 
            b['features'] = f_vec
        with open(spath, 'wb') as f:
            pickle.dump(bursts, f)
    return storage_paths


# ==============================================================================
# MÓDULO 4 y 5 — Filtro LightGBM
# ==============================================================================

def build_filter_dataset(storage_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for spath in tqdm(storage_paths, desc="Cargando bursts para filtro"):
        with open(spath, 'rb') as f:
            bursts = pickle.load(f)
        for b in bursts:
            f_feat = b['features']
            if b['class'] == NOISE_CLASS:
                X.append(f_feat); y.append(3)
            elif b['snr'] >= FILTER_SNR_THRESHOLD:
                bw = f_feat[0]
                if bw > 8e6:
                    X.append(f_feat); y.append(0)
                elif bw < 1.5e6:
                    X.append(f_feat); y.append(1)
                else:
                    X.append(f_feat); y.append(2)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 4,
    'n_estimators': 400,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1,
}

def train_interference_filter(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    log.info("Filtro de interferencias entrenado.")
    pickle.dump(clf, open(f"{OUTPUT_DIR}/filter_model.pkl", 'wb'))
    return clf

def apply_interference_filter(clf, storage_paths: list[str]) -> list[str]:
    kept_paths = []
    for spath in tqdm(storage_paths, desc="Filtrando interferencias"):
        with open(spath, 'rb') as f:
            bursts = pickle.load(f)
        
        X = np.array([b['features'] for b in bursts])
        if len(X) == 0:
            kept_paths.append(spath)
            continue
            
        probs = clf.predict_proba(X)
        p_drone = probs[:, 2]
        new_bursts = []
        for b, pd in zip(bursts, p_drone):
            if pd >= FILTER_CONF_THRESHOLD:
                new_bursts.append(b)
        
        with open(spath, 'wb') as f:
            pickle.dump(new_bursts, f)
        kept_paths.append(spath)
            
    return kept_paths


# ==============================================================================
# MÓDULO 6 — CV-CNN Encoder
# ==============================================================================

class BurstEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.iq_branch = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.psd_branch = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, 256), nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

    def preprocess_burst(self, iq_raw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        iq = iq_raw.astype(np.complex64)
        if len(iq) != IQ_INPUT_LEN:
            indices = np.linspace(0, len(iq) - 1, IQ_INPUT_LEN).astype(int)
            iq = iq[indices]

        iq = iq / (np.abs(iq).max() + 1e-8)
        iq_tensor = torch.tensor(np.stack([iq.real, iq.imag], axis=0), dtype=torch.float32)

        _, psd = welch(iq, fs=FS, nperseg=min(256, len(iq)), return_onesided=False, nfft=PSD_N_BINS * 2)
        psd = np.abs(psd[:PSD_N_BINS])
        psd = psd / (psd.max() + 1e-8)
        psd_tensor = torch.tensor(psd[np.newaxis, :], dtype=torch.float32)

        return iq_tensor, psd_tensor

    def forward(self, iq_batch: torch.Tensor, psd_batch: torch.Tensor) -> torch.Tensor:
        e_iq  = self.iq_branch(iq_batch)
        e_psd = self.psd_branch(psd_batch)
        return self.fusion(torch.cat([e_iq, e_psd], dim=1))


# ==============================================================================
# MÓDULO 7 — ABMIL
# ==============================================================================

class GatedAttentionMIL(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.V = nn.Linear(embed_dim, attn_dim, bias=False)  
        self.U = nn.Linear(embed_dim, attn_dim, bias=False)  
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None):
        attn_v = torch.tanh(self.V(H))       
        attn_u = torch.sigmoid(self.U(H))    
        attn_raw = self.w(attn_v * attn_u).squeeze(-1)  

        if mask is not None:
            attn_raw = attn_raw.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn_raw, dim=-1)  
        attn = torch.nan_to_num(attn, nan=0.0)

        z = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  
        logits = self.classifier(z).squeeze(-1)           

        return logits, attn


# ==============================================================================
# MÓDULO 8 — Dataset y Collate
# ==============================================================================

class BagDataset(Dataset):
    def __init__(self, file_list: list[dict], storage_paths: list[str], encoder: BurstEncoder, augment: bool = False):
        self.encoder = encoder
        self.augment = augment
        
        path_map = {Path(sp).stem: sp for sp in storage_paths}
        self.bags = []
        for fe in file_list:
            original_stem = Path(fe['path']).stem
            s_path = path_map.get(original_stem)
            label = 0 if fe['class'] == NOISE_CLASS else 1
            self.bags.append({
                'storage_path': s_path,
                'label': label,
                'class': fe['class'],
                'snr': fe['snr']
            })

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        bursts = []
        if bag['storage_path'] and os.path.exists(bag['storage_path']):
            with open(bag['storage_path'], 'rb') as f:
                bursts = pickle.load(f)
        
        bursts = bursts[:BAG_MAX_INSTANCES]

        iq_list, psd_list = [], []
        for b in bursts:
            iq_t, psd_t = self.encoder.preprocess_burst(b['iq'])
            
            if self.augment and np.random.rand() < 0.3:
                noise_sigma = 10 ** (-np.random.uniform(5, 20) / 20)
                iq_t += torch.randn_like(iq_t) * noise_sigma
                
            iq_list.append(iq_t)
            psd_list.append(psd_t)

        if len(iq_list) == 0:
            iq_list  = [torch.zeros(2, IQ_INPUT_LEN)]
            psd_list = [torch.zeros(1, PSD_N_BINS)]

        return {
            'iq':    torch.stack(iq_list),    
            'psd':   torch.stack(psd_list),   
            'label': torch.tensor(bag['label'], dtype=torch.float32),
            'class': bag['class'],
            'snr':   bag['snr'],
            'n_instances': len(iq_list)
        }

def collate_bags(batch):
    max_k = max(item['n_instances'] for item in batch)
    B = len(batch)

    iq_padded  = torch.zeros(B, max_k, 2, IQ_INPUT_LEN)
    psd_padded = torch.zeros(B, max_k, 1, PSD_N_BINS)
    mask       = torch.zeros(B, max_k, dtype=torch.bool)
    labels     = torch.zeros(B)
    classes    = []
    snrs       = []

    for i, item in enumerate(batch):
        k = item['n_instances']
        iq_padded[i, :k]  = item['iq']
        psd_padded[i, :k] = item['psd']
        mask[i, :k]       = True
        labels[i]         = item['label']
        classes.append(item['class'])
        snrs.append(item['snr'])

    return {
        'iq': iq_padded, 'psd': psd_padded,
        'mask': mask, 'label': labels,
        'class': classes, 'snr': snrs
    }


# ==============================================================================
# MÓDULO 9 — Loop de Entrenamiento
# ==============================================================================

def train_abmil(encoder: BurstEncoder,
                mil: GatedAttentionMIL,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device) -> dict:

    params = list(encoder.parameters()) + list(mil.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_auc = 0.0
    patience_counter = 0
    start_epoch = 0

    # Cargar checkpoint si existe para reanudar entrenamiento
    checkpoint_path = f"{OUTPUT_DIR}/abmil_model.pt"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            mil.load_state_dict(checkpoint['mil'])
            if 'history' in checkpoint:
                history = checkpoint['history']
                best_val_auc = max(history['val_auc']) if history['val_auc'] else 0.0
                start_epoch = len(history['val_auc'])
                # Adelantar el scheduler
                for _ in range(start_epoch):
                    scheduler.step()
            log.info(f"Reanudando entrenamiento desde epoch {start_epoch+1}")
        except Exception as e:
            log.warning(f"No se pudo cargar el checkpoint de entrenamiento: {e}")

    for epoch in range(start_epoch, EPOCHS):
        encoder.train(); mil.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            iq  = batch['iq'].to(device)    
            psd = batch['psd'].to(device)   
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

            B, K = iq.shape[:2]
            iq_flat  = iq.view(B*K, 2, IQ_INPUT_LEN)
            psd_flat = psd.view(B*K, 1, PSD_N_BINS)
            H_flat   = encoder(iq_flat, psd_flat)          
            H        = H_flat.view(B, K, EMBED_DIM)        

            logits, attn = mil(H, mask)

            loss_bce = criterion(logits, labels)

            attn_valid = attn * mask.float()
            attn_sum = attn_valid.sum(dim=1, keepdim=True) + 1e-8
            attn_norm = attn_valid / attn_sum
            entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum(dim=1).mean()
            loss = loss_bce + ATTN_LAMBDA * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        val_auc, val_loss = evaluate_epoch(encoder, mil, val_loader, device, criterion)
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        log.info(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f} "
                 f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'encoder': encoder.state_dict(),
                'mil': mil.state_dict(),
                'history': history
            }, f"{OUTPUT_DIR}/abmil_model.pt")
        else:
            patience_counter += 1
            
        # Guardar curvas cada época para que el usuario pueda ver el progreso
        plot_training_curves(history, OUTPUT_DIR)
        
        if patience_counter >= PATIENCE:
            log.info(f"Early stopping en epoch {epoch+1}")
            break

    return history

def evaluate_epoch(encoder, mil, loader, device, criterion):
    encoder.eval(); mil.eval()
    all_logits, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            iq  = batch['iq'].to(device)
            psd = batch['psd'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)
            B, K = iq.shape[:2]
            H = encoder(iq.view(B*K,2,IQ_INPUT_LEN),
                        psd.view(B*K,1,PSD_N_BINS)).view(B, K, EMBED_DIM)
            logits, _ = mil(H, mask)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_logits)
    all_labs  = np.concatenate(all_labels)
    try:
        val_auc = roc_auc_score(all_labs, all_preds)
    except Exception:
        val_auc = 0.5
    return val_auc, np.mean(losses)


# ==============================================================================
# MÓDULO 10 — Evaluación Test Set
# ==============================================================================

def run_inference_on_test(encoder, mil, test_loader, device):
    encoder.eval(); mil.eval()
    records = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencia test"):
            iq  = batch['iq'].to(device)
            psd = batch['psd'].to(device)
            mask = batch['mask'].to(device)
            B, K = iq.shape[:2]
            H = encoder(iq.view(B*K,2,IQ_INPUT_LEN),
                        psd.view(B*K,1,PSD_N_BINS)).view(B, K, EMBED_DIM)
            logits, attn = mil(H, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            attn_max = attn.max(dim=1).values.cpu().numpy()

            for i in range(B):
                records.append({
                    'true_class':   batch['class'][i],
                    'snr':          batch['snr'][i],
                    'y_true':       int(batch['label'][i].item()),
                    'y_pred_prob':  float(probs[i]),
                    'y_pred_bin':   int(probs[i] >= 0.5),
                    'attn_max':     float(attn_max[i])
                })
    return pd.DataFrame(records)

def plot_snr_class_heatmap(df: pd.DataFrame, output_dir: str):
    snr_vals = sorted(df['snr'].unique())
    classes  = DRONE_CLASSES

    auc_matrix = np.full((len(classes), len(snr_vals)), np.nan)

    for ci, cls in enumerate(classes):
        for si, snr in enumerate(snr_vals):
            subset = df[(df['true_class'].isin([cls, NOISE_CLASS])) &
                        (df['snr'] == snr)]
            
            if len(subset) < 2 or subset['y_true'].nunique() < 2:
                subset_fallback = df[(df['true_class'] == cls) & (df['snr'] == snr)]
                noise_fallback = df[df['true_class'] == NOISE_CLASS]
                subset = pd.concat([subset_fallback, noise_fallback])
                
            if len(subset) < 2 or subset['y_true'].nunique() < 2:
                continue
                
            try:
                auc_matrix[ci, si] = roc_auc_score(subset['y_true'], subset['y_pred_prob'])
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(auc_matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='AUC')

    ax.set_xticks(range(len(snr_vals)))
    ax.set_xticklabels([f"{s:+d}" for s in snr_vals], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"Dron clase {c}" for c in classes])
    ax.set_xlabel("SNR (dB)")
    ax.set_title("AUC por SNR y tipo de dron (test set)")

    for ci in range(len(classes)):
        for si in range(len(snr_vals)):
            val = auc_matrix[ci, si]
            if not np.isnan(val):
                ax.text(si, ci, f"{val:.2f}", ha='center', va='center',
                        fontsize=6, color='black' if val > 0.65 else 'white')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/heatmap_snr_clase.png", dpi=150)
    plt.close()

def plot_confusion_matrix_norm(df: pd.DataFrame, output_dir: str):
    cm = confusion_matrix(df['y_true'], df['y_pred_bin'], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ruido / No dron', 'Dron'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='.2%')
    ax.set_title("Matriz de confusión normalizada (test set)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/confusion_matrix.png", dpi=150)
    plt.close()

def plot_roc_curves_cls(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(DRONE_CLASSES)))

    for cls, color in zip(DRONE_CLASSES, colors):
        subset = df[df['true_class'].isin([cls, NOISE_CLASS])].copy()
        if subset['y_true'].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(subset['y_true'], subset['y_pred_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"Clase {cls} (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC por clase de dron (test set)")
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/roc_curves.png", dpi=150)
    plt.close()

def plot_training_curves(history: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train loss')
    axes[0].plot(history['val_loss'],   label='Val loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Pérdida durante entrenamiento')
    axes[0].legend()

    axes[1].plot(history['val_auc'], color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
    axes[1].set_title('AUC validación durante entrenamiento')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/training_curves.png", dpi=150)
    plt.close()

def plot_attention_examples(encoder, mil, test_loader, device, output_dir: str, n_examples: int = 4):
    encoder.eval(); mil.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    count = 0
    for batch in test_loader:
        if count >= n_examples: break
        iq  = batch['iq'].to(device)
        psd = batch['psd'].to(device)
        mask = batch['mask'].to(device)
        B, K = iq.shape[:2]
        with torch.no_grad():
            H = encoder(iq.view(B*K,2,IQ_INPUT_LEN), psd.view(B*K,1,PSD_N_BINS)).view(B, K, EMBED_DIM)
            _, attn = mil(H, mask)

        for b in range(min(B, n_examples - count)):
            k_real = mask[b].sum().item()
            attn_vals = attn[b, :k_real].cpu().numpy()
            ax = axes[count] if n_examples > 1 else axes
            if k_real > 0:
                ax.bar(range(k_real), attn_vals, color=plt.cm.hot(attn_vals / (attn_vals.max() + 1e-8)))
            label = "DRON" if batch['label'][b] == 1 else "RUIDO"
            ax.set_title(f"Bag {count+1} | {label} | clase {batch['class'][b]} | SNR {batch['snr'][b]:+d} dB")
            ax.set_xlabel("Burst #")
            ax.set_ylabel("Peso de atención")
            count += 1
            if count >= n_examples: break

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/attention_examples.png", dpi=150)
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")

    # 1. Descubrir ficheros
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)

    # 2. Segmentar bursts train/val
    burst_cache_file = f"{OUTPUT_DIR}/burst_dataset_raw.pkl"
    all_storage_paths = segment_files_with_checkpoint(train_files, burst_cache_file, desc="Segmentando train/val")
    log.info(f"Total ficheros con bursts: {len(all_storage_paths)}")

    # 3. Extraer features físicas
    log.info("Extrayendo features físicas...")
    all_storage_paths = compute_fc_deltas(all_storage_paths)

    # 4. Construir modelos
    encoder = BurstEncoder().to(device)
    mil     = GatedAttentionMIL().to(device)

    # 5. Crear DataLoaders
    train_fe, val_fe = train_test_split(
        train_files, test_size=0.15,
        stratify=[fe['class'] for fe in train_files],
        random_state=42)

    # Nota: Path(sp).stem debe coincidir con Path(fe['path']).stem
    train_burst_paths = [sp for sp in all_storage_paths if Path(sp).stem in {Path(fe['path']).stem for fe in train_fe}]
    val_burst_paths   = [sp for sp in all_storage_paths if Path(sp).stem in {Path(fe['path']).stem for fe in val_fe}]

    train_ds = BagDataset(train_fe, train_burst_paths, encoder, augment=True)
    val_ds   = BagDataset(val_fe,   val_burst_paths,   encoder, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_bags, num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_bags, num_workers=0)

    # 6. Entrenar ABMIL
    log.info("Entrenando ABMIL (Sin filtro previo)...")
    history = train_abmil(encoder, mil, train_loader, val_loader, device)
    plot_training_curves(history, OUTPUT_DIR)

    # 9. Cargar mejor modelo
    checkpoint = torch.load(f"{OUTPUT_DIR}/abmil_model.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    mil.load_state_dict(checkpoint['mil'])

    # 8. Procesar test set
    test_burst_cache = f"{OUTPUT_DIR}/test_burst_dataset_raw.pkl"
    test_burst_paths = segment_files_with_checkpoint(test_files, test_burst_cache, desc="Segmentando test set")
    
    # En esta versión V5 mejorada, realizamos la pasada de features físicas en test también
    test_burst_paths = compute_fc_deltas(test_burst_paths)

    test_ds     = BagDataset(test_files, test_burst_paths, encoder)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_bags, num_workers=4)

    # 11. Inferencia y figuras
    log.info("Generando resultados de evaluación...")
    df_results = run_inference_on_test(encoder, mil, test_loader, device)
    df_results.to_csv(f"{OUTPUT_DIR}/results/predictions.csv", index=False)

    plot_snr_class_heatmap(df_results, OUTPUT_DIR)
    plot_confusion_matrix_norm(df_results, OUTPUT_DIR)
    plot_roc_curves_cls(df_results, OUTPUT_DIR)
    plot_attention_examples(encoder, mil, test_loader, device, OUTPUT_DIR)

    # 12. Resumen numérico
    report = classification_report(df_results['y_true'], df_results['y_pred_bin'], target_names=['Ruido', 'Dron'])
    overall_auc = roc_auc_score(df_results['y_true'], df_results['y_pred_prob'])
    log.info(f"\n{report}")
    log.info(f"AUC global (test): {overall_auc:.4f}")

    with open(f"{OUTPUT_DIR}/results/summary.txt", 'w') as f:
        f.write(report + f"\nAUC global: {overall_auc:.4f}\n")

    log.info("Pipeline completado. Resultados en: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()