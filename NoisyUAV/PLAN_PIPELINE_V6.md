# Plan de actuación v6: Pipeline MIL para detección de drones en RF

## Contexto y motivación de v6

La v5 alcanzó AUC=0.858 y recall de dron del 69.3%. El análisis de falsos negativos
reveló la causa raíz: el encoder aprendió a responder a **energía instantánea** en lugar
de a **morfología espectral**. En ficheros con SNR media (0–10 dB), las interferencias
WiFi y Bluetooth tienen más energía pico que el dron, y la atención ABMIL colapsa sobre
ellas ignorando el burst de dron, que es visualmente distinguible en el espectrograma
(rectángulo compacto de banda estrecha y duración larga frente a líneas verticales anchas
y breves de WiFi/BT).

v6 introduce dos cambios que atacan exactamente ese problema:

- **Camino A:** sustituir la rama IQ crudo (Conv1d sobre señal temporal) por una rama
  espectrograma 2D (Conv2d sobre STFT), que aprende directamente la textura rectangular
  característica del dron.
- **Camino B:** añadir dos features de forma morfológica (`aspect_ratio`, `spectral_fill`)
  al vector que alimenta la red de atención, penalizando explícitamente los bursts de
  banda ancha y corta duración (WiFi/BT).

El resto de la arquitectura (ABMIL con gated attention, BagDataset con checkpointing,
loop de entrenamiento, evaluación) se mantiene igual que v5 salvo los cambios indicados.

---

## Qué reutilizar de v5 y qué regenerar

```
# REUTILIZAR (no tocar):
outputs/burst_cache/*.pkl          ← IQ raw segmentado, independiente del encoder
outputs/burst_dataset_raw.pkl      ← lista de rutas al cache, válida
outputs/test_burst_dataset_raw.pkl ← ídem para test

# BORRAR antes de lanzar v6:
outputs/abmil_model.pt             ← arquitectura cambia, pesos incompatibles
outputs/filter_model.pkl           ← las features físicas añaden 2 nuevas columnas
outputs/results/*                  ← resultados de v5, se sobreescriben
```

El agente debe cambiar `OUTPUT_DIR` a una carpeta nueva (`modelo_v6/outputs`) para no
mezclar artefactos de v5 y v6. Los pickles de `burst_cache` se copian o se referencian
con la ruta original de v5.

---

## Parámetros globales

Copiar todos los parámetros de v5 y aplicar los siguientes cambios:

```python
# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
OUTPUT_DIR       = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_v6\outputs"

# RUTA AL BURST CACHE DE V5 (se reutiliza íntegramente)
BURST_CACHE_V5   = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs\burst_cache"

# ── SDR / señal ───────────────────────────────────────────────────────────────
FS               = 14e6
FILE_DURATION_MS = 75
N_SAMPLES_FILE   = int(FS * FILE_DURATION_MS / 1000)

# ── Segmentador ───────────────────────────────────────────────────────────────
ENTROPY_NPERSEG   = 2048
MIN_BURST_MS      = 0.4
MIN_BURST_SAMPLES = int(FS * 0.0004)   # 0.4 ms
MAX_BURST_SAMPLES = int(FS * 0.01)     # 10 ms

# Parámetros del fallback de ventana deslizante (NUEVO en v6)
SLIDING_WINDOW_MS      = 2.0           # duración de cada ventana (ms)
SLIDING_WINDOW_OVERLAP = 0.5           # solapamiento entre ventanas

# ── Espectrograma STFT (NUEVO en v6, sustituye IQ_INPUT_LEN) ─────────────────
IQ_INPUT_LEN  = 1024    # longitud IQ antes de STFT (resampleo si distinto)
STFT_NPERSEG  = 64      # ventana STFT → F_bins = 64
STFT_NOVERLAP = 48      # solapamiento STFT → T_frames = 61
# Tensor resultante: (1, 64, 61)  [canales, frecuencia, tiempo]

# ── PSD (se mantiene igual que v5) ───────────────────────────────────────────
PSD_N_BINS    = 512

# ── Features de forma morfológica (NUEVO en v6) ───────────────────────────────
SHAPE_FEAT_DIM = 2      # aspect_ratio + spectral_fill
# El vector de atención pasa de dim=128 a dim=130 (128 embed + 2 shape)

# ── ABMIL ────────────────────────────────────────────────────────────────────
EMBED_DIM        = 128
ATTN_DIM         = 64
ATTN_INPUT_DIM   = EMBED_DIM + SHAPE_FEAT_DIM   # 130
ATTN_LAMBDA      = 0.05   # peso penalización varianza de atención (ver cambio en loss)
BAG_MAX_INSTANCES = 32
BATCH_SIZE       = 16
LR               = 1e-4
EPOCHS           = 60
PATIENCE         = 10

# ── Evaluación ───────────────────────────────────────────────────────────────
DRONE_CLASSES = [0, 1, 2, 3, 5, 6]
NOISE_CLASS   = 4
SNR_BINS      = list(range(-20, 32, 2))
```

---

## Módulo 0 — Importaciones y setup

Igual que v5. Añadir `from scipy.signal import stft` a las importaciones.

```python
import os, sys, pickle, logging, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from scipy.signal import welch, stft
from scipy.stats import kurtosis, skew
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              ConfusionMatrixDisplay, roc_auc_score,
                              classification_report)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/results").mkdir(parents=True, exist_ok=True)
BURST_STORAGE_DIR = Path(f"{OUTPUT_DIR}/burst_cache")
BURST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Módulo 1 — Carga y descubrimiento de ficheros

**Sin cambios respecto a v5.** Copiar `load_iq_file` y `discover_files` íntegramente.

Recordatorio de la lógica: los ficheros tienen nombre
`IQdata_sample{N}_target{T}_snr{S}.pt` y se cargan con `torch.load`. La clave `x_iq`
contiene un tensor `[2, N]` con I en la fila 0 y Q en la fila 1.

---

## Módulo 2 — Segmentador de bursts (MODIFICADO en v6)

Se añade el **fallback de ventana deslizante** cuando el detector de entropía devuelve
menos de 2 bursts. Esto mejora la cobertura a SNR baja donde la entropía no detecta nada.

```python
def segment_file(file_entry: dict) -> str:
    """
    1. Intenta segmentación por entropía de Shannon.
    2. Si obtiene < 2 bursts, aplica ventana deslizante como fallback.
    3. Guarda los bursts en burst_cache y devuelve la ruta.
    """
    d = torch.load(file_entry['path'], map_location='cpu', weights_only=False)
    iq_tensor  = d['x_iq'].float()
    iq_complex = (iq_tensor[0] + 1j * iq_tensor[1]).numpy().astype(np.complex64)

    # ── Intento 1: detector de entropía ──────────────────────────────────────
    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts_info = detectar_bursts(
        iq_tensor, fs=FS, nperseg=ENTROPY_NPERSEG, z_thresh=1.5,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=0.5, min_z_abs=2.0,
        bg_mult=4, max_bins_frac=1.0, smooth_ms=0.2)

    bursts = []
    for idx, b in enumerate(bursts_info):
        s_idx    = int(b['t0'] * FS / 1000.0)
        e_idx    = int(b['t1'] * FS / 1000.0)
        burst_iq = iq_complex[s_idx:e_idx]
        if len(burst_iq) < MIN_BURST_SAMPLES:
            continue
        burst_iq = burst_iq[:MAX_BURST_SAMPLES]
        bursts.append({
            'iq':        burst_iq,
            'class':     file_entry['class'],
            'snr':       file_entry['snr'],
            'file_path': file_entry['path'],
            'burst_idx': idx,
            't0':        b['t0'],
            't1':        b['t1'],
            'source':    'entropy'      # ← nuevo campo de procedencia
        })

    # ── Intento 2: ventana deslizante (fallback para SNR baja) ───────────────
    if len(bursts) < 2:
        win_samples  = int(SLIDING_WINDOW_MS * FS / 1000)
        step_samples = int(win_samples * (1 - SLIDING_WINDOW_OVERLAP))
        starts = range(0, len(iq_complex) - win_samples + 1, step_samples)
        for idx, s in enumerate(starts):
            chunk = iq_complex[s: s + win_samples]
            bursts.append({
                'iq':        chunk,
                'class':     file_entry['class'],
                'snr':       file_entry['snr'],
                'file_path': file_entry['path'],
                'burst_idx': 1000 + idx,
                't0':        s / FS * 1000,
                't1':        (s + win_samples) / FS * 1000,
                'source':    'sliding_window'   # ← diferencia de v5
            })

    # ── Persistencia ─────────────────────────────────────────────────────────
    file_id      = Path(file_entry['path']).stem
    storage_path = BURST_STORAGE_DIR / f"{file_id}.pkl"
    with open(storage_path, 'wb') as f:
        pickle.dump(bursts, f)
    return str(storage_path)
```

`segment_files_with_checkpoint` se copia íntegramente de v5, sin cambios.

**Nota para el agente:** con `SLIDING_WINDOW_MS=2.0` y `SLIDING_WINDOW_OVERLAP=0.5` sobre
un fichero de 75ms se generan ≈37 ventanas. Con `BAG_MAX_INSTANCES=32` el bag se trunca
a las 32 primeras. Si el agente decide cambiar el orden de bursts antes de truncar (por
ejemplo, poniendo primero los de entropía y luego los de ventana deslizante), debe
asegurarse de que los de entropía tengan `burst_idx < 1000` y los de ventana ≥ 1000,
como ya está codificado arriba.

---

## Módulo 3 — Extractor de features físicas (MODIFICADO en v6)

Se añaden las dos **features de forma morfológica** al vector de 11 features de v5.
El vector resultante tiene **13 features** por burst.

| # | Nombre | Descripción |
|---|--------|-------------|
| 0–10 | (igual que v5) | BW, duración, duty cycle, fc, fc_delta, kurtosis PSD, skewness PSD, kurtosis IQ, flatness, env_std, ZCR |
| 11 | `aspect_ratio` | BW_hz / dur_ms — alto en WiFi/BT, bajo en dron |
| 12 | `spectral_fill` | fracción del espectro con energía > pico−10 dB — alto en WiFi, bajo en dron |

```python
def extract_physical_features(burst: dict) -> np.ndarray:
    iq  = burst['iq'].astype(np.complex64)
    amp = np.abs(iq)

    _, psd     = welch(iq, fs=FS, nperseg=min(256, len(iq)), return_onesided=False)
    psd        = np.abs(psd)
    psd_norm   = psd / (psd.sum() + 1e-12)
    psd_db     = 10 * np.log10(psd + 1e-12)
    peak       = psd_db.max()
    bw_mask    = psd_db > (peak - 10)
    bw_inst    = bw_mask.sum() * (FS / len(psd))

    duration_us = len(iq) / FS * 1e6
    duty_cycle  = duration_us / (FILE_DURATION_MS * 1e3)
    freqs       = np.fft.fftfreq(len(psd), d=1/FS)
    fc_est      = freqs[np.argmax(psd)]
    fc_delta    = 0.0

    psd_kurtosis   = float(kurtosis(psd_norm))
    psd_skewness   = float(skew(psd_norm))
    iq_kurtosis    = float(kurtosis(amp))
    geo_mean       = np.exp(np.log(psd + 1e-12).mean())
    spectral_flat  = float(geo_mean / (psd.mean() + 1e-12))
    envelope_std   = float(amp.std())
    real_part      = iq.real
    zcr            = float(((real_part[:-1] * real_part[1:]) < 0).sum() / len(real_part))

    # ── NUEVAS features de forma morfológica (camino B) ──────────────────────
    dur_ms         = len(iq) / FS * 1000.0
    aspect_ratio   = float(bw_inst / (dur_ms + 1e-8))   # Hz/ms → alto=WiFi, bajo=dron
    spectral_fill  = float(bw_mask.sum() / len(psd))    # fracción espectro ocupada

    return np.array([
        bw_inst, duration_us, duty_cycle, fc_est, fc_delta,
        psd_kurtosis, psd_skewness, iq_kurtosis,
        spectral_flat, envelope_std, zcr,
        aspect_ratio, spectral_fill          # ← features 11 y 12
    ], dtype=np.float32)
```

`compute_fc_deltas` se copia íntegramente de v5, sin cambios.

---

## Módulo 4 y 5 — Filtro de interferencias LightGBM

**Sin cambios en la lógica.** El vector de features ahora tiene 13 columnas en lugar de
11, pero LightGBM las consume automáticamente.

Copiar `build_filter_dataset`, `LGBM_PARAMS`, `train_interference_filter` y
`apply_interference_filter` de v5 sin modificaciones.

**Recordatorio importante para el agente:** en v5 el filtro se construyó pero NO se
llamó desde `main()`. En v6 debe conectarse correctamente. Ver sección de `main()`.

---

## Módulo 6 — BurstEncoder2D (REESCRITO completamente en v6)

Este es el cambio central de v6. La rama IQ crudo (Conv1d sobre señal temporal) se
sustituye por una rama espectrograma (Conv2d sobre STFT). La rama PSD se mantiene igual.

### Dimensiones verificadas

Con `IQ_INPUT_LEN=1024`, `STFT_NPERSEG=64`, `STFT_NOVERLAP=48`:
- Tensor STFT: `(1, 64, 61)` → 1 canal, 64 bins de frecuencia, 61 frames temporales
- Resolución temporal: 4.6 µs/frame — suficiente para ver la duración de bursts de dron
- Resolución frecuencial: 218.8 kHz/bin — suficiente para distinguir rectángulos de 1–3 MHz

### Código completo

```python
class BurstEncoder(nn.Module):
    """
    Encoder v6: espectrograma 2D (camino A) + PSD 1D.
    Produce embedding de dimensión EMBED_DIM=128 por burst.
    """
    def __init__(self):
        super().__init__()

        # ── Rama A: espectrograma STFT 2D ────────────────────────────────────
        # Input: (B, 1, 64, 61)
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                          # → (B, 16, 32, 30)

            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                          # → (B, 32, 16, 15)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                  # → (B, 64, 4, 4)
            nn.Flatten(),                                  # → (B, 1024)

            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 64)                             # → (B, 64)
        )

        # ── Rama B: PSD 1D (igual que v5) ────────────────────────────────────
        # Input: (B, 1, PSD_N_BINS=512)
        self.psd_branch = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, 256), nn.ReLU(),
            nn.Linear(256, 64)                             # → (B, 64)
        )

        # ── Fusión ───────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )                                                  # → (B, 128)

    def preprocess_burst(self, iq_raw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convierte un burst IQ numpy (complex64) en:
          - spec_tensor: (1, 64, 61) espectrograma STFT normalizado
          - psd_tensor:  (1, 512)    PSD Welch normalizada

        NO llama a métodos de GPU. Seguro para usar en DataLoader workers.
        """
        iq = iq_raw.astype(np.complex64)

        # Resampleo a longitud fija
        if len(iq) != IQ_INPUT_LEN:
            indices = np.linspace(0, len(iq) - 1, IQ_INPUT_LEN).astype(int)
            iq = iq[indices]

        # Normalización de amplitud
        iq = iq / (np.abs(iq).max() + 1e-8)

        # ── Espectrograma STFT ────────────────────────────────────────────────
        _, _, Zxx = stft(iq, fs=FS,
                         nperseg=STFT_NPERSEG,
                         noverlap=STFT_NOVERLAP,
                         return_onesided=False)
        spec = np.abs(Zxx).astype(np.float32)              # (64, 61)
        spec = spec / (spec.max() + 1e-8)
        spec_tensor = torch.tensor(
            spec[np.newaxis], dtype=torch.float32)         # (1, 64, 61)

        # ── PSD Welch ─────────────────────────────────────────────────────────
        _, psd = welch(iq, fs=FS,
                       nperseg=min(256, len(iq)),
                       return_onesided=False,
                       nfft=PSD_N_BINS * 2)
        psd = np.abs(psd[:PSD_N_BINS]).astype(np.float32)
        psd = psd / (psd.max() + 1e-8)
        psd_tensor = torch.tensor(
            psd[np.newaxis], dtype=torch.float32)          # (1, 512)

        return spec_tensor, psd_tensor

    def forward(self, spec_batch: torch.Tensor,
                psd_batch: torch.Tensor) -> torch.Tensor:
        """
        spec_batch: (B, 1, 64, 61)
        psd_batch:  (B, 1, 512)
        Returns:    (B, 128)
        """
        e_spec = self.spec_branch(spec_batch)   # (B, 64)
        e_psd  = self.psd_branch(psd_batch)     # (B, 64)
        return self.fusion(torch.cat([e_spec, e_psd], dim=1))   # (B, 128)
```

---

## Módulo 7 — GatedAttentionMIL (MODIFICADO en v6)

La red de atención recibe ahora `ATTN_INPUT_DIM=130` en lugar de 128, porque se
concatenan las 2 features de forma morfológica (`aspect_ratio`, `spectral_fill`) al
embedding antes de calcular los scores de atención.

El clasificador de bag sigue operando sobre `z` (promedio ponderado de embeddings de
128 dims), no sobre el vector ampliado de 130 dims. Esto es deliberado: las features
de forma informan a la atención de qué bursts ignorar, pero el clasificador trabaja
sobre representaciones ricas aprendidas por el encoder.

```python
class GatedAttentionMIL(nn.Module):
    def __init__(self,
                 embed_dim: int = EMBED_DIM,           # 128
                 attn_input_dim: int = ATTN_INPUT_DIM, # 130
                 attn_dim: int = ATTN_DIM):            # 64
        super().__init__()

        # Atención con gating — recibe embed(128) + shape_feats(2) = 130
        self.V = nn.Linear(attn_input_dim, attn_dim, bias=False)
        self.U = nn.Linear(attn_input_dim, attn_dim, bias=False)
        self.w = nn.Linear(attn_dim, 1, bias=False)

        # Clasificador de bag — recibe solo el embedding de 128 dims
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self,
                H: torch.Tensor,            # (B, K, 128) embeddings
                S: torch.Tensor,            # (B, K, 2)   features de forma
                mask: torch.Tensor = None): # (B, K) bool
        """
        H:    embeddings del encoder,       (B, K, 128)
        S:    features de forma por burst,  (B, K, 2)
        mask: True = instancia real         (B, K)
        Returns:
          logits: (B,)   sin sigmoid
          attn:   (B, K) pesos de atención
        """
        # Concatenar shape features al embedding para la atención
        H_aug    = torch.cat([H, S], dim=-1)                    # (B, K, 130)

        attn_v   = torch.tanh(self.V(H_aug))                   # (B, K, 64)
        attn_u   = torch.sigmoid(self.U(H_aug))                # (B, K, 64)
        attn_raw = self.w(attn_v * attn_u).squeeze(-1)         # (B, K)

        if mask is not None:
            attn_raw = attn_raw.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn_raw, dim=-1)                  # (B, K)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Representación del bag: promedio ponderado de embeddings originales
        z      = torch.bmm(attn.unsqueeze(1), H).squeeze(1)    # (B, 128)
        logits = self.classifier(z).squeeze(-1)                 # (B,)

        return logits, attn
```

---

## Módulo 8 — BagDataset y collate_bags (MODIFICADO en v6)

El dataset ahora carga 3 elementos por instancia: spec_tensor, psd_tensor y
shape_tensor. El collate_bags añade el padding correspondiente.

```python
class BagDataset(Dataset):
    def __init__(self, file_list: list[dict], storage_paths: list[str],
                 encoder: BurstEncoder, augment: bool = False):
        self.encoder = encoder
        self.augment = augment

        path_map = {Path(sp).stem: sp for sp in storage_paths}
        self.bags = []
        for fe in file_list:
            s_path = path_map.get(Path(fe['path']).stem)
            label  = 0 if fe['class'] == NOISE_CLASS else 1
            self.bags.append({
                'storage_path': s_path,
                'label':  label,
                'class':  fe['class'],
                'snr':    fe['snr']
            })

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        bag    = self.bags[idx]
        bursts = []
        if bag['storage_path'] and os.path.exists(bag['storage_path']):
            with open(bag['storage_path'], 'rb') as f:
                bursts = pickle.load(f)

        bursts = bursts[:BAG_MAX_INSTANCES]

        spec_list, psd_list, shape_list = [], [], []

        for b in bursts:
            spec_t, psd_t = self.encoder.preprocess_burst(b['iq'])

            # Data augmentation: añadir ruido AWGN al espectrograma
            # Se añade ANTES de normalizar el espectrograma, en el dominio IQ
            # Implementación: regenerar spec con IQ ruidoso
            if self.augment and np.random.rand() < 0.3:
                iq = b['iq'].astype(np.complex64)
                if len(iq) != IQ_INPUT_LEN:
                    indices = np.linspace(0, len(iq)-1, IQ_INPUT_LEN).astype(int)
                    iq = iq[indices]
                iq = iq / (np.abs(iq).max() + 1e-8)
                noise_sigma = 10 ** (-np.random.uniform(5, 20) / 20)
                iq_noisy = iq + (np.random.randn(len(iq)) +
                                 1j * np.random.randn(len(iq))).astype(np.complex64) * noise_sigma
                _, _, Zxx = stft(iq_noisy, fs=FS,
                                 nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                                 return_onesided=False)
                spec_aug = np.abs(Zxx).astype(np.float32)
                spec_aug = spec_aug / (spec_aug.max() + 1e-8)
                spec_t   = torch.tensor(spec_aug[np.newaxis], dtype=torch.float32)

            # Features de forma morfológica (camino B)
            # Se calculan sobre el IQ original (sin augmentación)
            # ya están en b['features'][11:13] si compute_fc_deltas las incluyó
            feats = b.get('features', None)
            if feats is not None and len(feats) >= 13:
                shape_feat = torch.tensor(feats[11:13], dtype=torch.float32)
            else:
                # Fallback: calcular en el momento
                shape_feat = compute_shape_features_tensor(b['iq'])

            spec_list.append(spec_t)
            psd_list.append(psd_t)
            shape_list.append(shape_feat)

        # Bag vacío: rellenar con ceros
        if len(spec_list) == 0:
            spec_list  = [torch.zeros(1, STFT_NPERSEG, 61)]
            psd_list   = [torch.zeros(1, PSD_N_BINS)]
            shape_list = [torch.zeros(SHAPE_FEAT_DIM)]

        return {
            'spec':        torch.stack(spec_list),    # (K, 1, 64, 61)
            'psd':         torch.stack(psd_list),     # (K, 1, 512)
            'shape':       torch.stack(shape_list),   # (K, 2)
            'label':       torch.tensor(bag['label'], dtype=torch.float32),
            'class':       bag['class'],
            'snr':         bag['snr'],
            'n_instances': len(spec_list)
        }


def compute_shape_features_tensor(iq_raw: np.ndarray) -> torch.Tensor:
    """
    Calcula aspect_ratio y spectral_fill directamente desde IQ raw.
    Función auxiliar para el fallback del BagDataset.
    """
    iq  = iq_raw.astype(np.complex64)
    _, psd = welch(iq, fs=FS, nperseg=min(256, len(iq)), return_onesided=False)
    psd    = np.abs(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    peak   = psd_db.max()
    bw_mask = psd_db > (peak - 10)

    bw_inst       = bw_mask.sum() * (FS / len(psd))
    dur_ms        = len(iq) / FS * 1000.0
    aspect_ratio  = float(bw_inst / (dur_ms + 1e-8))
    spectral_fill = float(bw_mask.sum() / len(psd))
    return torch.tensor([aspect_ratio, spectral_fill], dtype=torch.float32)


def collate_bags(batch):
    max_k = max(item['n_instances'] for item in batch)
    B     = len(batch)

    # Obtener T_frames dinámicamente del primer elemento no vacío
    T_frames = 61   # valor por defecto calculado con nperseg=64, noverlap=48, IQ_LEN=1024

    spec_padded  = torch.zeros(B, max_k, 1, STFT_NPERSEG, T_frames)
    psd_padded   = torch.zeros(B, max_k, 1, PSD_N_BINS)
    shape_padded = torch.zeros(B, max_k, SHAPE_FEAT_DIM)
    mask         = torch.zeros(B, max_k, dtype=torch.bool)
    labels       = torch.zeros(B)
    classes, snrs = [], []

    for i, item in enumerate(batch):
        k = item['n_instances']
        spec_padded[i, :k]  = item['spec']
        psd_padded[i, :k]   = item['psd']
        shape_padded[i, :k] = item['shape']
        mask[i, :k]         = True
        labels[i]           = item['label']
        classes.append(item['class'])
        snrs.append(item['snr'])

    return {
        'spec':    spec_padded,
        'psd':     psd_padded,
        'shape':   shape_padded,
        'mask':    mask,
        'label':   labels,
        'class':   classes,
        'snr':     snrs
    }
```

---

## Módulo 9 — Loop de entrenamiento (MODIFICADO en v6)

Dos cambios respecto a v5:

1. Las variables de batch pasan de `iq/psd` a `spec/psd/shape`.
2. La penalización de atención cambia de **entropía** a **varianza** (más estable cuando
   todos los bursts del bag son igualmente poco informativos).

```python
def train_abmil(encoder: BurstEncoder,
                mil: GatedAttentionMIL,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device) -> dict:

    params    = list(encoder.parameters()) + list(mil.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_auc    = 0.0
    patience_counter = 0
    start_epoch     = 0

    checkpoint_path = f"{OUTPUT_DIR}/abmil_model.pt"
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt['encoder'])
            mil.load_state_dict(ckpt['mil'])
            if 'history' in ckpt:
                history      = ckpt['history']
                best_val_auc = max(history['val_auc']) if history['val_auc'] else 0.0
                start_epoch  = len(history['val_auc'])
                for _ in range(start_epoch):
                    scheduler.step()
            log.info(f"Reanudando desde epoch {start_epoch+1}")
        except Exception as e:
            log.warning(f"No se pudo cargar checkpoint: {e}")

    for epoch in range(start_epoch, EPOCHS):
        encoder.train(); mil.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            spec   = batch['spec'].to(device)    # (B, K, 1, 64, 61)
            psd    = batch['psd'].to(device)     # (B, K, 1, 512)
            shape  = batch['shape'].to(device)   # (B, K, 2)
            mask   = batch['mask'].to(device)    # (B, K)
            labels = batch['label'].to(device)   # (B,)

            B, K = spec.shape[:2]

            # Pasar instancias por el encoder (merge B y K)
            spec_flat = spec.view(B*K, 1, STFT_NPERSEG, 61)
            psd_flat  = psd.view(B*K, 1, PSD_N_BINS)
            H_flat    = encoder(spec_flat, psd_flat)          # (B*K, 128)
            H         = H_flat.view(B, K, EMBED_DIM)         # (B, K, 128)

            logits, attn = mil(H, shape, mask)

            # BCE sobre el bag
            loss_bce = criterion(logits, labels)

            # ── Penalización de varianza de atención (NUEVO en v6) ────────────
            # Maximizar la varianza de atención = forzar concentración
            # cuando hay bursts discriminativos, sin colapso artificial
            # cuando todos los bursts son similares.
            attn_valid = attn * mask.float()
            attn_sum   = attn_valid.sum(dim=1, keepdim=True) + 1e-8
            attn_norm  = attn_valid / attn_sum
            # sum(a_k^2) es máximo cuando un solo burst tiene todo el peso
            attn_var   = (attn_norm ** 2).sum(dim=1).mean()
            # Restar porque queremos MAXIMIZAR la varianza (minimizar -varianza)
            loss = loss_bce - ATTN_LAMBDA * attn_var

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
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save({
                'encoder': encoder.state_dict(),
                'mil':     mil.state_dict(),
                'history': history
            }, checkpoint_path)
        else:
            patience_counter += 1

        plot_training_curves(history, OUTPUT_DIR)

        if patience_counter >= PATIENCE:
            log.info(f"Early stopping en epoch {epoch+1}")
            break

    return history


def evaluate_epoch(encoder, mil, loader, device, criterion):
    encoder.eval(); mil.eval()
    all_preds, all_labels, losses = [], [], []

    with torch.no_grad():
        for batch in loader:
            spec   = batch['spec'].to(device)
            psd    = batch['psd'].to(device)
            shape  = batch['shape'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)
            B, K   = spec.shape[:2]

            H_flat = encoder(spec.view(B*K, 1, STFT_NPERSEG, 61),
                             psd.view(B*K, 1, PSD_N_BINS))
            H      = H_flat.view(B, K, EMBED_DIM)

            logits, _ = mil(H, shape, mask)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        val_auc = 0.5
    return val_auc, np.mean(losses)
```

---

## Módulo 10 — Inferencia y evaluación (MODIFICADO en v6)

Solo cambian las variables de batch. La lógica de evaluación y las figuras son
idénticas a v5.

```python
def run_inference_on_test(encoder, mil, test_loader, device):
    encoder.eval(); mil.eval()
    records = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencia test"):
            spec   = batch['spec'].to(device)
            psd    = batch['psd'].to(device)
            shape  = batch['shape'].to(device)
            mask   = batch['mask'].to(device)
            B, K   = spec.shape[:2]

            H_flat   = encoder(spec.view(B*K, 1, STFT_NPERSEG, 61),
                               psd.view(B*K, 1, PSD_N_BINS))
            H        = H_flat.view(B, K, EMBED_DIM)
            logits, attn = mil(H, shape, mask)
            probs    = torch.sigmoid(logits).cpu().numpy()
            attn_max = attn.max(dim=1).values.cpu().numpy()

            for i in range(B):
                records.append({
                    'true_class':  batch['class'][i],
                    'snr':         batch['snr'][i],
                    'y_true':      int(batch['label'][i].item()),
                    'y_pred_prob': float(probs[i]),
                    'y_pred_bin':  int(probs[i] >= 0.5),
                    'attn_max':    float(attn_max[i])
                })

    return pd.DataFrame(records)
```

Las funciones `plot_snr_class_heatmap`, `plot_confusion_matrix_norm`,
`plot_roc_curves_cls`, `plot_training_curves` y `plot_attention_examples` se copian
de v5 sin cambios. En `plot_attention_examples` cambiar la referencia a `batch['iq']`
por `batch['spec']` si se usa para visualización, o simplificarlo para que solo
muestre los pesos de atención por burst (que es lo que ya hacía).

---

## Módulo 11 — main() (MODIFICADO en v6)

El main conecta correctamente todas las etapas, incluyendo el filtro LightGBM que
faltaba en v5.

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")

    # ── 1. Descubrir ficheros ────────────────────────────────────────────────
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)

    # ── 2. Segmentar bursts train/val (con checkpoint y fallback deslizante) ─
    burst_cache_file = f"{OUTPUT_DIR}/burst_dataset_raw.pkl"

    # OPCIÓN A: Si se reutiliza el burst_cache de v5, apuntar a él directamente
    # y saltarse la segmentación. El agente debe verificar si el directorio
    # BURST_CACHE_V5 existe y contiene los pkl esperados.
    if Path(BURST_CACHE_V5).exists():
        log.info(f"Reutilizando burst_cache de v5: {BURST_CACHE_V5}")
        # Reconstruir la lista de rutas a partir del directorio
        all_storage_paths = [str(p) for p in Path(BURST_CACHE_V5).glob("*.pkl")]
        # Guardar la lista para el checkpoint
        pickle.dump(all_storage_paths, open(burst_cache_file, 'wb'))
    else:
        # OPCIÓN B: Segmentar desde cero (primera ejecución o v5 no disponible)
        all_storage_paths = segment_files_with_checkpoint(
            train_files, burst_cache_file, desc="Segmentando train/val")

    log.info(f"Total ficheros con bursts: {len(all_storage_paths)}")

    # ── 3. Extraer features físicas (incluyendo las 2 nuevas de forma) ───────
    log.info("Extrayendo features físicas...")
    all_storage_paths = compute_fc_deltas(all_storage_paths)

    # ── 4. Filtro de interferencias LightGBM ─────────────────────────────────
    filter_cache = f"{OUTPUT_DIR}/filter_model.pkl"
    if os.path.exists(filter_cache):
        log.info("Cargando filtro de interferencias desde cache...")
        filter_clf = pickle.load(open(filter_cache, 'rb'))
    else:
        log.info("Construyendo dataset para filtro de interferencias...")
        X_filter, y_filter = build_filter_dataset(all_storage_paths)
        log.info(f"Dataset filtro: {X_filter.shape} | clases: {np.bincount(y_filter)}")
        filter_clf = train_interference_filter(X_filter, y_filter)

    log.info("Aplicando filtro de interferencias a train/val...")
    all_storage_paths = apply_interference_filter(filter_clf, all_storage_paths)

    # ── 5. Construir encoder y modelo ────────────────────────────────────────
    encoder = BurstEncoder().to(device)
    mil     = GatedAttentionMIL().to(device)

    # ── 6. DataLoaders train / val ───────────────────────────────────────────
    train_fe, val_fe = train_test_split(
        train_files, test_size=0.15,
        stratify=[fe['class'] for fe in train_files],
        random_state=42)

    train_stems = {Path(fe['path']).stem for fe in train_fe}
    val_stems   = {Path(fe['path']).stem for fe in val_fe}

    train_paths = [sp for sp in all_storage_paths
                   if Path(sp).stem in train_stems]
    val_paths   = [sp for sp in all_storage_paths
                   if Path(sp).stem in val_stems]

    train_ds = BagDataset(train_fe, train_paths, encoder, augment=True)
    val_ds   = BagDataset(val_fe,   val_paths,   encoder, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_bags, num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_bags, num_workers=0)

    # ── 7. Entrenar ABMIL ────────────────────────────────────────────────────
    log.info("Entrenando ABMIL v6...")
    history = train_abmil(encoder, mil, train_loader, val_loader, device)
    plot_training_curves(history, OUTPUT_DIR)

    # ── 8. Cargar mejor checkpoint ───────────────────────────────────────────
    ckpt = torch.load(f"{OUTPUT_DIR}/abmil_model.pt",
                      map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    mil.load_state_dict(ckpt['mil'])

    # ── 9. Segmentar y filtrar test set ──────────────────────────────────────
    test_cache = f"{OUTPUT_DIR}/test_burst_dataset_raw.pkl"

    # Misma lógica de reutilización que en train
    test_cache_v5 = str(Path(BURST_CACHE_V5).parent / "test_burst_cache")
    if Path(BURST_CACHE_V5).exists():
        # Reutilizar pkl del burst_cache de v5 para test
        # Los ficheros de test tienen el mismo nombre que los de train
        test_stems_set = {Path(fe['path']).stem for fe in test_files}
        test_storage_paths = [
            str(Path(BURST_CACHE_V5) / f"{stem}.pkl")
            for stem in test_stems_set
            if (Path(BURST_CACHE_V5) / f"{stem}.pkl").exists()
        ]
        pickle.dump(test_storage_paths, open(test_cache, 'wb'))
    else:
        test_storage_paths = segment_files_with_checkpoint(
            test_files, test_cache, desc="Segmentando test set")

    test_storage_paths = compute_fc_deltas(test_storage_paths)
    test_storage_paths = apply_interference_filter(filter_clf, test_storage_paths)

    test_ds = BagDataset(test_files, test_storage_paths, encoder, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_bags, num_workers=0)

    # ── 10. Inferencia y figuras ─────────────────────────────────────────────
    log.info("Generando resultados de evaluación...")
    df_results = run_inference_on_test(encoder, mil, test_loader, device)
    df_results.to_csv(f"{OUTPUT_DIR}/results/predictions.csv", index=False)

    plot_snr_class_heatmap(df_results, OUTPUT_DIR)
    plot_confusion_matrix_norm(df_results, OUTPUT_DIR)
    plot_roc_curves_cls(df_results, OUTPUT_DIR)
    plot_attention_examples(encoder, mil, test_loader, device, OUTPUT_DIR)

    # ── 11. Resumen numérico ─────────────────────────────────────────────────
    report      = classification_report(df_results['y_true'], df_results['y_pred_bin'],
                                        target_names=['Ruido', 'Dron'])
    overall_auc = roc_auc_score(df_results['y_true'], df_results['y_pred_prob'])
    log.info(f"\n{report}")
    log.info(f"AUC global (test): {overall_auc:.4f}")

    with open(f"{OUTPUT_DIR}/results/summary.txt", 'w') as f:
        f.write(report + f"\nAUC global: {overall_auc:.4f}\n")

    log.info("Pipeline v6 completado. Resultados en: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()
```

---

## Instrucciones específicas para el agente

1. **Cambiar OUTPUT_DIR** a `modelo_v6/outputs` antes de cualquier otra cosa.

2. **Verificar BURST_CACHE_V5**: comprobar si existe el directorio de cache de v5. Si
   existe, reutilizarlo directamente en el main tal como está especificado. Si no existe,
   la segmentación corre desde cero con el nuevo `segment_file` que incluye el fallback
   de ventana deslizante.

3. **No reutilizar `abmil_model.pt` ni `filter_model.pkl` de v5**: las arquitecturas
   han cambiado y los pesos son incompatibles.

4. **Verificar las dimensiones del espectrograma** antes de entrenar. Añadir al inicio
   de main() esta comprobación:
   ```python
   # Verificación de dimensiones
   dummy_iq = np.random.randn(IQ_INPUT_LEN).astype(np.complex64)
   _, _, Zxx = stft(dummy_iq, fs=FS, nperseg=STFT_NPERSEG,
                    noverlap=STFT_NOVERLAP, return_onesided=False)
   T_frames = Zxx.shape[1]
   log.info(f"Espectrograma STFT: ({STFT_NPERSEG}, {T_frames}) → tensor (1, {STFT_NPERSEG}, {T_frames})")
   assert T_frames == 61, f"T_frames esperado=61, obtenido={T_frames}"
   ```
   Si el assert falla, ajustar el valor hardcodeado `61` en `collate_bags` al valor real.

5. **El filtro LightGBM ahora SÍ debe conectarse en main()**. Las llamadas a
   `build_filter_dataset`, `train_interference_filter` y `apply_interference_filter`
   están incluidas en el main() de este plan. No omitirlas.

6. **apply_interference_filter modifica los pkl en disco**: si se quiere poder
   reejecutar con umbral distinto sin re-segmentar, guardar los bursts filtrados en una
   subcarpeta `burst_cache_filtered/` separada en lugar de sobreescribir el cache
   original. El agente implementará esta separación creando `BURST_STORAGE_FILTERED_DIR`.

7. **num_workers=0** en todos los DataLoaders: Windows no soporta multiprocessing con
   pickle de objetos complejos en workers. Mantener a 0.

8. **GPU con menos de 8 GB**: reducir `BATCH_SIZE=8` y `BAG_MAX_INSTANCES=16`.

9. **Reproducibilidad**: las semillas están fijadas al inicio de main().
   No modificarlas.

10. **Resumen de cambios respecto a v5**:

| Componente | v5 | v6 |
|---|---|---|
| Rama IQ del encoder | Conv1d sobre señal temporal | Conv2d sobre espectrograma STFT |
| Input al encoder | `(B, 2, 1024)` IQ crudo | `(B, 1, 64, 61)` STFT |
| Features físicas | 11 features | 13 features (+aspect_ratio, +spectral_fill) |
| Input a la atención | `H` (128 dims) | `H‖S` (130 dims) |
| Penalización de atención | entropía | varianza (maximizar concentración) |
| Segmentador fallback | fichero completo como 1 burst | ventana deslizante de 2ms |
| Filtro LightGBM | implementado pero no conectado | conectado en main() |
| Reutilización cache | — | burst_cache de v5 reutilizable |
