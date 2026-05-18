# Plan de actuación: Pipeline MIL para detección de drones en RF

## Objetivo

Construir un único fichero `pipeline.py` ejecutable que, dado un directorio raíz con los datos
del dataset NoisyUAV y la ruta al módulo `detector_entropia.py`, realice de forma autónoma y
secuencial: creación del dataset de bursts, entrenamiento del filtro de interferencias,
entrenamiento del modelo ABMIL, y evaluación completa con figuras de salida.

El agente debe leer este plan de arriba a abajo e implementar cada módulo en el orden indicado.
Al final todos los módulos se conectan en la función `main()` del fichero `pipeline.py`.

Si alguna carpeta de salida no existe, debe ser creada por el agente.
Comprobar que existen los ficheros de datos crudos, que deben coincidir con las rutas usadas por los códigos de construcción 
del dataset hechos para otros modelos

---

## Estructura de ficheros que debe producir el agente

```
pipeline.py                  ← fichero único de ejecución
outputs/
  burst_dataset.pkl          ← dataset de bursts serializado
  filter_model.pkl           ← LightGBM entrenado
  abmil_model.pt             ← pesos del modelo ABMIL
  results/
    confusion_matrix.png
    heatmap_snr_clase.png
    roc_curves.png
    attention_examples.png
    training_curves.png
```

---

## Parámetros globales (parte superior de pipeline.py)

```python
# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_ROOT        = r"C:\TFM_data\NoisyUAV\drone_RF_data"        # directorio raíz con subcarpetas clase_1…clase_6
TEST_SPLIT_FILE  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv" # fichero con las rutas del 20% de test intocable, el mismo que para el modelo V4
DETECTOR_PATH    = r"C:\repos\DroneDetectionRF\NoisyUAV\funciones\dsp_rf\detector_entropia.py"
OUTPUT_DIR       = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\outputs"

# ── SDR / señal ───────────────────────────────────────────────────────────────
FS               = 14e6          # frecuencia de muestreo (Hz)
FILE_DURATION_MS = 75            # duración de cada fichero (ms)
N_SAMPLES_FILE   = int(FS * FILE_DURATION_MS / 1000)   # muestras por fichero

# ── Segmentador ───────────────────────────────────────────────────────────────
ENTROPY_WINDOW   = 512           # ventana de entropía (muestras)
ENTROPY_STEP     = 128           # paso de la ventana
ENTROPY_THRESHOLD_QUANTILE = 0.25  # umbral adaptativo: percentil 25 de la entropía del fichero
MIN_BURST_SAMPLES = 256          # burst mínimo válido
MAX_BURST_SAMPLES = int(FS * 0.01)  # burst máximo: 10 ms

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
DRONE_CLASSES    = [1, 2, 3, 5, 6]
NOISE_CLASS      = 4
```

---

## Módulo 0 — Importaciones y setup

```python
import importlib.util, sys, os, pickle, json, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Carga dinámica del detector de entropía del usuario
spec = importlib.util.spec_from_file_location("detector_entropia", DETECTOR_PATH)
detector_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_mod)
# El agente debe inspeccionar la API pública de detector_entropia.py y
# localizar la función principal de segmentación. Si expone una función
# llamada `segment_bursts(iq, fs, ...)` o similar, usarla directamente.
# Si no, usar la clase o función que el módulo exponga y adaptarla al
# contrato de entrada/salida definido abajo en el Módulo 1.

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/results").mkdir(parents=True, exist_ok=True)
```

---

## Módulo 1 — Carga de ficheros IQ

### Tarea
Leer los ficheros del dataset y separar train/val del test set usando el listado de ficheros
intocable. Los ficheros IQ son binarios con muestras complejas float32 intercaladas (I, Q, I, Q…).

### Implementación

```python
def load_iq_file(filepath: str) -> np.ndarray:
    """
    Lee un fichero binario IQ (float32 intercalado) y devuelve
    un array complejo numpy de forma (N,).
    Si el fichero tiene cabecera o formato distinto, el agente debe
    detectarlo (magic bytes, tamaño de fichero) y adaptarse.
    """
    raw = np.fromfile(filepath, dtype=np.float32)
    # Intercalado I/Q → complejo
    iq = raw[0::2] + 1j * raw[1::2]
    return iq.astype(np.complex64)


def parse_snr_from_filename(filepath: str) -> int:
    """
    Extrae el valor de SNR del nombre de fichero.
    Convención asumida del dataset NoisyUAV:
    el nombre contiene algo como 'snr_-10' o 'SNR10' o 'n10'.
    El agente debe inspeccionar nombres reales de fichero y adaptar
    la expresión regular. Devuelve el SNR en dB como int.
    """
    import re
    name = Path(filepath).stem
    # Intentar varios patrones comunes; el agente añadirá el correcto
    for pattern in [r'snr_?(-?\d+)', r'SNR(-?\d+)', r'_n(-?\d+)']:
        m = re.search(pattern, name, re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"No se pudo extraer SNR de: {filepath}")


def discover_files(data_root: str, test_split_file: str):
    """
    Devuelve dos listas de diccionarios:
      train_files: ficheros de entrenamiento+validación
      test_files:  ficheros de test (20% intocable)
    Cada diccionario: {'path': str, 'class': int, 'snr': int}
    """
    with open(test_split_file) as f:
        test_paths = set(line.strip() for line in f)

    all_files, test_files, train_files = [], [], []
    for class_dir in sorted(Path(data_root).iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name.split('_')[-1])  # "clase_3" → 3
        for fpath in class_dir.glob("*.bin"):           # adaptar extensión real
            entry = {'path': str(fpath),
                     'class': class_id,
                     'snr': parse_snr_from_filename(str(fpath))}
            if str(fpath) in test_paths:
                test_files.append(entry)
            else:
                train_files.append(entry)

    log.info(f"Ficheros train/val: {len(train_files)}  |  test: {len(test_files)}")
    return train_files, test_files
```

**Contrato de salida:** dos listas de dicts `{path, class, snr}`.

---

## Módulo 2 — Segmentador de bursts

### Tarea
Para cada fichero IQ extraer una lista de bursts usando el detector de entropía del usuario.
Complementar con un detector CFAR espectral para ficheros con SNR < -10 dB donde la entropía
puede no detectar nada.

### Implementación

```python
def cfar_spectral_detector(iq: np.ndarray, fs: float,
                            guard_cells: int = 4,
                            reference_cells: int = 16,
                            pfa: float = 1e-3) -> list[tuple[int,int]]:
    """
    CFAR 1D sobre la PSD del fichero completo.
    Devuelve lista de (sample_start, sample_end) de regiones con energía.
    Útil cuando la entropía no detecta bursts (SNR muy baja).
    """
    freqs, psd = welch(iq, fs=fs, nperseg=1024, return_onesided=False)
    psd_db = 10 * np.log10(np.abs(psd) + 1e-12)

    detections = []
    N = len(psd_db)
    for i in range(reference_cells + guard_cells,
                   N - reference_cells - guard_cells):
        # Celdas de referencia: excluir celdas de guarda
        left  = psd_db[i - reference_cells - guard_cells : i - guard_cells]
        right = psd_db[i + guard_cells + 1 : i + guard_cells + reference_cells + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        threshold = noise_est - 10 * np.log10(-np.log(pfa) / reference_cells)
        if psd_db[i] > threshold:
            # Convertir índice de frecuencia a rango temporal aproximado
            # (detección espectral → marca todo el fichero como burst candidato)
            detections.append((0, len(iq)))
            break  # basta con detectar energía; la posición temporal ya la gestiona la entropía
    return detections


def segment_file(file_entry: dict) -> list[dict]:
    """
    Aplica el detector de entropía del usuario al fichero.
    Si no devuelve bursts y SNR < -10, aplica CFAR como fallback
    y devuelve el fichero completo como un único burst.

    Devuelve lista de dicts:
      {'iq': np.ndarray(complex64),
       'class': int,
       'snr': int,
       'file_path': str,
       'burst_idx': int}
    """
    iq = load_iq_file(file_entry['path'])

    # Llamada al detector del usuario — adaptar según su API real
    bursts_ranges = detector_mod.segment_bursts(
        iq, fs=FS,
        window=ENTROPY_WINDOW,
        step=ENTROPY_STEP,
        threshold_quantile=ENTROPY_THRESHOLD_QUANTILE,
        min_samples=MIN_BURST_SAMPLES,
        max_samples=MAX_BURST_SAMPLES
    )

    # Fallback CFAR para SNR muy baja
    if len(bursts_ranges) == 0 and file_entry['snr'] < -10:
        bursts_ranges = cfar_spectral_detector(iq, FS)

    # Si sigue sin detectar nada, tratar fichero completo como un burst
    if len(bursts_ranges) == 0:
        bursts_ranges = [(0, len(iq))]

    bursts = []
    for idx, (s, e) in enumerate(bursts_ranges):
        burst_iq = iq[s:e]
        if len(burst_iq) < MIN_BURST_SAMPLES:
            continue
        bursts.append({
            'iq': burst_iq,
            'class': file_entry['class'],
            'snr': file_entry['snr'],
            'file_path': file_entry['path'],
            'burst_idx': idx
        })
    return bursts
```

**Contrato de salida:** lista de dicts `{iq, class, snr, file_path, burst_idx}`.

---

## Módulo 3 — Extractor de features físicas por burst

### Tarea
Computar un vector de features DSP interpretables para cada burst. Este vector se usará para
entrenar el filtro de interferencias (LightGBM) de forma que descarte WiFi, BT y ruido antes
de pasar los bursts al modelo ABMIL.

### Features a extraer (11 features en total)

| # | Nombre | Descripción | Discrimina |
|---|--------|-------------|-----------|
| 0 | `bw_inst` | Ancho de banda a -10 dB sobre el pico de la PSD normalizada (Hz) | WiFi >> Dron >> BT |
| 1 | `duration_us` | Duración del burst en microsegundos | BT ≈ 625 µs exacto |
| 2 | `duty_cycle` | Ratio duración_burst / ventana_análisis | BT ~50%, Dron bajo |
| 3 | `fc_est` | Frecuencia central estimada (Hz, relativa a fs/2) | Posición en espectro |
| 4 | `fc_delta` | Variación de fc respecto al burst anterior del mismo fichero | FHSS vs canal fijo |
| 5 | `psd_kurtosis` | Curtosis de la PSD (eje frecuencia) | OFDM WiFi muy alto |
| 6 | `psd_skewness` | Asimetría de la PSD | Dron asimétrico |
| 7 | `iq_kurtosis` | Curtosis de la amplitud instantánea | Modulación AM/FM |
| 8 | `spectral_flatness` | Planura espectral (Wiener) | FHSS plano; OFDM menos |
| 9 | `envelope_std` | Desviación estándar de la envolvente | Bursts dron irregulares |
|10 | `zero_crossing_rate` | Tasa de cruce por cero de la parte real | Indicador de FM |

```python
def extract_physical_features(burst: dict) -> np.ndarray:
    iq = burst['iq'].astype(np.complex64)
    amp = np.abs(iq)

    # PSD con Welch
    _, psd = welch(iq, fs=FS, nperseg=min(256, len(iq)), return_onesided=False)
    psd = np.abs(psd)
    psd_norm = psd / (psd.sum() + 1e-12)

    # 0: BW instantáneo a -10 dB
    psd_db = 10 * np.log10(psd + 1e-12)
    peak = psd_db.max()
    bw_mask = psd_db > (peak - 10)
    bw_inst = bw_mask.sum() * (FS / len(psd))

    # 1: Duración
    duration_us = len(iq) / FS * 1e6

    # 2: Duty cycle (aproximado: siempre 1.0 aquí, se calcula a nivel de fichero fuera)
    duty_cycle = duration_us / (FILE_DURATION_MS * 1e3)

    # 3: Frecuencia central
    freqs = np.fft.fftfreq(len(psd), d=1/FS)
    fc_est = freqs[np.argmax(psd)]

    # 4: Delta fc (se rellena a nivel de bag en el módulo de construcción del dataset)
    fc_delta = 0.0

    # 5-6: Curtosis y asimetría PSD
    psd_kurtosis = float(kurtosis(psd_norm))
    psd_skewness = float(skew(psd_norm))

    # 7: Curtosis de la amplitud
    iq_kurtosis = float(kurtosis(amp))

    # 8: Planura espectral
    geo_mean = np.exp(np.log(psd + 1e-12).mean())
    arith_mean = psd.mean() + 1e-12
    spectral_flatness = float(geo_mean / arith_mean)

    # 9: Desviación de la envolvente
    envelope_std = float(amp.std())

    # 10: Zero crossing rate (parte real)
    real_part = iq.real
    zcr = float(((real_part[:-1] * real_part[1:]) < 0).sum() / len(real_part))

    return np.array([bw_inst, duration_us, duty_cycle, fc_est, fc_delta,
                     psd_kurtosis, psd_skewness, iq_kurtosis,
                     spectral_flatness, envelope_std, zcr], dtype=np.float32)


def compute_fc_deltas(burst_list: list[dict]) -> list[dict]:
    """
    Calcula el delta de frecuencia central entre bursts consecutivos
    del mismo fichero. Modifica burst_list in-place añadiendo 'features'.
    """
    from collections import defaultdict
    by_file = defaultdict(list)
    for b in burst_list:
        by_file[b['file_path']].append(b)

    for fpath, bursts in by_file.items():
        bursts.sort(key=lambda x: x['burst_idx'])
        feats = [extract_physical_features(b) for b in bursts]
        for i, (b, f) in enumerate(zip(bursts, feats)):
            if i > 0:
                f[4] = abs(feats[i][3] - feats[i-1][3])  # fc_delta
            b['features'] = f
    return burst_list
```

**Contrato de salida:** cada dict de burst tiene ahora una clave `'features': np.ndarray(11,)`.

---

## Módulo 4 — Construcción del dataset para el filtro de interferencias

### Tarea
Generar un dataset de bursts con pseudo-etiquetas limpias para entrenar LightGBM.

### Reglas de pseudo-etiquetado

- **Clase 4 (ruido)**, cualquier SNR → bursts etiquetados como `[WiFi=0, BT=0, Dron=0, Ruido=1]`
  Los bursts de clase 4 también contienen WiFi y BT; el LightGBM aprenderá a distinguirlos
  a partir de las features físicas sin necesitar etiquetas individuales perfectas.
  **Importante:** la clase 4 proporciona ejemplos "negativos de dron" seguros.

- **Clases 1,2,3,5,6, SNR ≥ FILTER_SNR_THRESHOLD** → bursts etiquetados según BW:
  - si `bw_inst > 8e6` → WiFi (label 0)
  - si `bw_inst < 1.5e6` → BT (label 1)
  - si `1.5e6 ≤ bw_inst ≤ 8e6` → Dron (label 2)
  Esto es una heurística; el LightGBM aprenderá a refinarla con las 11 features conjuntas.

- **Clases 1,2,3,5,6, SNR < FILTER_SNR_THRESHOLD** → excluir del entrenamiento del filtro
  (SNR demasiado baja para confiar en el pseudo-etiquetado heurístico).

```python
def build_filter_dataset(burst_list: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve X (N, 11) e y (N,) con etiquetas {0=WiFi, 1=BT, 2=Dron, 3=Ruido}.
    Solo incluye bursts que cumplen los criterios de pseudo-etiquetado.
    """
    X, y = [], []
    for b in burst_list:
        f = b['features']
        if b['class'] == NOISE_CLASS:
            X.append(f); y.append(3)
        elif b['snr'] >= FILTER_SNR_THRESHOLD:
            bw = f[0]
            if bw > 8e6:
                X.append(f); y.append(0)   # WiFi
            elif bw < 1.5e6:
                X.append(f); y.append(1)   # BT
            else:
                X.append(f); y.append(2)   # Dron
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
```

**Contrato de salida:** `(X, y)` arrays numpy.

---

## Módulo 5 — Entrenamiento del filtro de interferencias (LightGBM)

### Arquitectura del modelo
LightGBM multiclase con 4 clases `{WiFi, BT, Dron, Ruido}`.

### Hiperparámetros

```python
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
```

### Implementación

```python
def train_interference_filter(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    from sklearn.model_selection import StratifiedShuffleSplit
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


def apply_interference_filter(clf, burst_list: list[dict]) -> list[dict]:
    """
    Aplica el filtro. Descarta bursts con P(Dron) < FILTER_CONF_THRESHOLD.
    Añade al dict la clave 'p_drone' (float) y 'keep' (bool).
    """
    X = np.array([b['features'] for b in burst_list])
    probs = clf.predict_proba(X)  # shape (N, 4)
    p_drone = probs[:, 2]
    for b, pd in zip(burst_list, p_drone):
        b['p_drone'] = float(pd)
        b['keep'] = pd >= FILTER_CONF_THRESHOLD
    kept = [b for b in burst_list if b['keep']]
    log.info(f"Bursts tras filtro: {len(kept)} / {len(burst_list)}")
    return kept
```

**Contrato de salida:** lista de bursts filtrada (solo los con `keep=True`).

---

## Módulo 6 — CV-CNN para embeddings de bursts

### Arquitectura detallada

La red produce un embedding de 128 dimensiones para cada burst.
Dos ramas paralelas:

#### Rama A — IQ crudo (dominio temporal, señal compleja → 2 canales reales)

```
Input: (batch, 2, IQ_INPUT_LEN)   # canal 0 = I, canal 1 = Q

Conv1d(in=2,  out=32, kernel=7, padding=3) → BN → ReLU
Conv1d(in=32, out=64, kernel=5, padding=2) → BN → ReLU → MaxPool(2)
Conv1d(in=64, out=128, kernel=3, padding=1) → BN → ReLU → MaxPool(2)
AdaptiveAvgPool1d(output_size=16)
Flatten → Linear(128*16, 256) → ReLU → Linear(256, 64)
```

#### Rama B — PSD (dominio frecuencial, señal real)

```
Input: (batch, 1, PSD_N_BINS)

Conv1d(in=1,  out=32, kernel=7, padding=3) → BN → ReLU
Conv1d(in=32, out=64, kernel=5, padding=2) → BN → ReLU → MaxPool(2)
Conv1d(in=64, out=128, kernel=3, padding=1) → BN → ReLU → MaxPool(2)
AdaptiveAvgPool1d(output_size=8)
Flatten → Linear(128*8, 256) → ReLU → Linear(256, 64)
```

#### Fusión

```
concat([embed_A (64,), embed_B (64,)]) → Linear(128, 128) → LayerNorm → ReLU
Output: (batch, EMBED_DIM=128)
```

#### Código PyTorch

```python
class BurstEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Rama IQ
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

        # Rama PSD
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

        # Fusión
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

    def preprocess_burst(self, iq_raw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocesa un burst numpy (complex64) en los dos tensores de entrada.
        Resamplea/trunca a IQ_INPUT_LEN. Normaliza amplitud.
        """
        iq = iq_raw.astype(np.complex64)

        # Resamplear a longitud fija
        if len(iq) != IQ_INPUT_LEN:
            indices = np.linspace(0, len(iq) - 1, IQ_INPUT_LEN).astype(int)
            iq = iq[indices]

        # Normalizar
        iq = iq / (np.abs(iq).max() + 1e-8)

        # Tensor IQ: (2, IQ_INPUT_LEN)
        iq_tensor = torch.tensor(
            np.stack([iq.real, iq.imag], axis=0), dtype=torch.float32)

        # PSD con Welch, normalizada
        _, psd = welch(iq, fs=FS, nperseg=min(256, len(iq)), return_onesided=False,
                       nfft=PSD_N_BINS * 2)
        psd = np.abs(psd[:PSD_N_BINS])
        psd = psd / (psd.max() + 1e-8)
        psd_tensor = torch.tensor(psd[np.newaxis, :], dtype=torch.float32)

        return iq_tensor, psd_tensor

    def forward(self, iq_batch: torch.Tensor, psd_batch: torch.Tensor) -> torch.Tensor:
        """
        iq_batch:  (B, 2, IQ_INPUT_LEN)
        psd_batch: (B, 1, PSD_N_BINS)
        Returns:   (B, EMBED_DIM)
        """
        e_iq  = self.iq_branch(iq_batch)
        e_psd = self.psd_branch(psd_batch)
        return self.fusion(torch.cat([e_iq, e_psd], dim=1))
```

---

## Módulo 7 — Modelo ABMIL (Attention-Based MIL)

### Concepto clave
- **Bag** = todos los bursts de un fichero que superaron el filtro de interferencias.
- **Label del bag** = label del fichero: 1 si es dron (clases 1,2,3,5,6), 0 si es ruido (clase 4).
- El modelo nunca necesita saber qué burst individual es dron. La pérdida se calcula a nivel de bag.

### Arquitectura: Gated Attention Pooling

La atención con gating (Ilse et al., 2018) es más estable que la atención simple:

```
Para cada instancia h_k ∈ R^{128}:
  a_k = softmax_k( w^T · (tanh(V·h_k) ⊙ sigmoid(U·h_k)) )

Representación del bag:
  z = Σ_k a_k · h_k    ∈ R^{128}

Clasificador:
  z → FC(128, 64) → ReLU → Dropout(0.3) → FC(64, 1) → Sigmoid → P(dron)
```

### Código PyTorch

```python
class GatedAttentionMIL(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, attn_dim: int = ATTN_DIM):
        super().__init__()

        # Ramas de atención con gating
        self.V = nn.Linear(embed_dim, attn_dim, bias=False)  # tanh path
        self.U = nn.Linear(embed_dim, attn_dim, bias=False)  # sigmoid path
        self.w = nn.Linear(attn_dim, 1, bias=False)

        # Clasificador a nivel de bag
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None):
        """
        H:    (B, K, embed_dim)  embeddings de instancias por bag
        mask: (B, K) bool, True = instancia real, False = padding
        Returns:
          logits: (B,)   sin sigmoid (para BCEWithLogitsLoss)
          attn:   (B, K) pesos de atención
        """
        # Gated attention scores
        attn_v = torch.tanh(self.V(H))       # (B, K, attn_dim)
        attn_u = torch.sigmoid(self.U(H))    # (B, K, attn_dim)
        attn_raw = self.w(attn_v * attn_u).squeeze(-1)  # (B, K)

        # Enmascarar padding
        if mask is not None:
            attn_raw = attn_raw.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn_raw, dim=-1)  # (B, K)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Representación del bag
        z = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # (B, embed_dim)
        logits = self.classifier(z).squeeze(-1)           # (B,)

        return logits, attn
```

---

## Módulo 8 — Dataset PyTorch y collate function

```python
class BagDataset(Dataset):
    """
    Cada muestra es un bag (fichero) con:
      - Lista de tensores IQ y PSD de sus bursts
      - Label binaria (0/1)
      - Metadatos: class_id, snr
    """
    def __init__(self, file_list: list[dict], burst_list: list[dict],
                 encoder: BurstEncoder):
        from collections import defaultdict
        self.encoder = encoder

        # Agrupar bursts por fichero
        by_file = defaultdict(list)
        for b in burst_list:
            by_file[b['file_path']].append(b)

        self.bags = []
        for fe in file_list:
            bursts = by_file.get(fe['path'], [])
            label = 0 if fe['class'] == NOISE_CLASS else 1
            self.bags.append({
                'bursts': bursts,
                'label': label,
                'class': fe['class'],
                'snr': fe['snr'],
                'path': fe['path']
            })

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        bursts = bag['bursts'][:BAG_MAX_INSTANCES]

        iq_list, psd_list = [], []
        for b in bursts:
            iq_t, psd_t = self.encoder.preprocess_burst(b['iq'])
            iq_list.append(iq_t)
            psd_list.append(psd_t)

        # Bags vacíos: insertar un burst de ceros (clase 4 puede no tener bursts candidatos)
        if len(iq_list) == 0:
            iq_list  = [torch.zeros(2, IQ_INPUT_LEN)]
            psd_list = [torch.zeros(1, PSD_N_BINS)]

        return {
            'iq':    torch.stack(iq_list),    # (K, 2, IQ_INPUT_LEN)
            'psd':   torch.stack(psd_list),   # (K, 1, PSD_N_BINS)
            'label': torch.tensor(bag['label'], dtype=torch.float32),
            'class': bag['class'],
            'snr':   bag['snr'],
            'n_instances': len(iq_list)
        }


def collate_bags(batch):
    """
    Padding de bags a la misma longitud K dentro del batch.
    Genera máscara booleana.
    """
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
```

---

## Módulo 9 — Entrenamiento del modelo ABMIL

### Función de pérdida

```
L = BCE_logits(logit, y_bag) + λ · entropy_penalty(attn)

entropy_penalty(attn) = -mean( Σ_k a_k · log(a_k + ε) )

Intuición: penalizar atención muy distribuida fuerza al modelo a
concentrarse en los bursts más informativos, facilitando interpretabilidad.
```

### Loop de entrenamiento

```python
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

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────
        encoder.train(); mil.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            iq  = batch['iq'].to(device)    # (B, K, 2, IQ_INPUT_LEN)
            psd = batch['psd'].to(device)   # (B, K, 1, PSD_N_BINS)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

            B, K = iq.shape[:2]
            # Pasar instancias por el encoder (merge B y K como batch)
            iq_flat  = iq.view(B*K, 2, IQ_INPUT_LEN)
            psd_flat = psd.view(B*K, 1, PSD_N_BINS)
            H_flat   = encoder(iq_flat, psd_flat)          # (B*K, 128)
            H        = H_flat.view(B, K, EMBED_DIM)        # (B, K, 128)

            logits, attn = mil(H, mask)

            # BCE sobre el bag
            loss_bce = criterion(logits, labels)

            # Penalización de entropía de atención
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

        # ── Validation ─────────────────────────────────────────────
        val_auc, val_loss = evaluate_epoch(encoder, mil, val_loader, device, criterion)
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        log.info(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f} "
                 f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({'encoder': encoder.state_dict(),
                        'mil': mil.state_dict()},
                       f"{OUTPUT_DIR}/abmil_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"Early stopping en epoch {epoch+1}")
                break

    return history


def evaluate_epoch(encoder, mil, loader, device, criterion):
    """Pasa por el loader en modo eval. Devuelve (auc, loss)."""
    from sklearn.metrics import roc_auc_score
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
```

---

## Módulo 10 — Evaluación completa sobre el test set

### 10.1 Inferencia sobre el test set

```python
def run_inference_on_test(encoder, mil, test_loader, device):
    """
    Devuelve un DataFrame con columnas:
      file_path, true_class, snr, y_true, y_pred_prob, y_pred_bin, attn_max
    """
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
```

### 10.2 Figura 1 — Heatmap AUC por SNR × Clase de dron

```python
def plot_snr_class_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Genera un heatmap donde:
      - Eje X: SNR en dB (de -20 a 30 en pasos de 2)
      - Eje Y: clase de dron (1,2,3,5,6) + fila 'vs ruido' (promedio)
      - Color: AUC de esa celda (clase, SNR)
    
    Para cada celda (clase c, snr s):
      - Positivos: ficheros de clase c con SNR = s
      - Negativos: todos los ficheros de clase 4 con SNR = s
    """
    from sklearn.metrics import roc_auc_score

    snr_vals = sorted(df['snr'].unique())
    classes  = DRONE_CLASSES

    auc_matrix = np.full((len(classes), len(snr_vals)), np.nan)

    for ci, cls in enumerate(classes):
        for si, snr in enumerate(snr_vals):
            subset = df[(df['true_class'].isin([cls, NOISE_CLASS])) &
                        (df['snr'] == snr)]
            if len(subset) < 2 or subset['y_true'].nunique() < 2:
                continue
            try:
                auc_matrix[ci, si] = roc_auc_score(
                    subset['y_true'], subset['y_pred_prob'])
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(auc_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0.5, vmax=1.0, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='AUC')

    ax.set_xticks(range(len(snr_vals)))
    ax.set_xticklabels([f"{s:+d}" for s in snr_vals], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"Dron clase {c}" for c in classes])
    ax.set_xlabel("SNR (dB)")
    ax.set_title("AUC por SNR y tipo de dron (test set)")

    # Anotar valores en las celdas
    for ci in range(len(classes)):
        for si in range(len(snr_vals)):
            val = auc_matrix[ci, si]
            if not np.isnan(val):
                ax.text(si, ci, f"{val:.2f}", ha='center', va='center',
                        fontsize=6, color='black' if val > 0.65 else 'white')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/heatmap_snr_clase.png", dpi=150)
    plt.close()
    log.info("Heatmap guardado.")
```

### 10.3 Figura 2 — Matriz de confusión normalizada

```python
def plot_confusion_matrix(df: pd.DataFrame, output_dir: str):
    """
    Matriz de confusión binaria (dron vs no-dron) normalizada por fila.
    Se muestra tanto en valores absolutos como en porcentaje.
    """
    cm = confusion_matrix(df['y_true'], df['y_pred_bin'], normalize='true')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Ruido / No dron', 'Dron'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='.2%')
    ax.set_title("Matriz de confusión normalizada (test set)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/confusion_matrix.png", dpi=150)
    plt.close()
    log.info("Matriz de confusión guardada.")
```

### 10.4 Figura 3 — Curvas ROC por clase de dron y por SNR

```python
def plot_roc_curves(df: pd.DataFrame, output_dir: str):
    """
    Una curva ROC por cada clase de dron (1 vs 4).
    Cada curva lleva la AUC en la leyenda.
    """
    from sklearn.metrics import roc_curve, auc as sk_auc

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(DRONE_CLASSES)))

    for cls, color in zip(DRONE_CLASSES, colors):
        subset = df[df['true_class'].isin([cls, NOISE_CLASS])].copy()
        if subset['y_true'].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(subset['y_true'], subset['y_pred_prob'])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"Clase {cls} (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC por clase de dron (test set)")
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/roc_curves.png", dpi=150)
    plt.close()
    log.info("Curvas ROC guardadas.")
```

### 10.5 Figura 4 — Curvas de entrenamiento

```python
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
    log.info("Curvas de entrenamiento guardadas.")
```

### 10.6 Figura 5 — Ejemplos de atención (opcional pero recomendada)

```python
def plot_attention_examples(encoder, mil, test_loader, device,
                            output_dir: str, n_examples: int = 4):
    """
    Muestra para N bags del test set:
      - El espectrograma del fichero completo (fondo)
      - Los bursts superpuestos, coloreados por peso de atención
    Requiere acceso al IQ raw de los bags, no solo los tensores.
    El agente puede omitir esta figura si la memoria del batch no almacena
    el IQ raw; en ese caso crear un DataLoader ad-hoc con el IQ sin preprocesar.
    """
    # Implementación simplificada: barra de atención por burst
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
            H = encoder(iq.view(B*K,2,IQ_INPUT_LEN),
                        psd.view(B*K,1,PSD_N_BINS)).view(B, K, EMBED_DIM)
            _, attn = mil(H, mask)

        for b in range(min(B, n_examples - count)):
            k_real = mask[b].sum().item()
            attn_vals = attn[b, :k_real].cpu().numpy()
            ax = axes[count] if n_examples > 1 else axes
            ax.bar(range(k_real), attn_vals,
                   color=plt.cm.hot(attn_vals / (attn_vals.max() + 1e-8)))
            label = "DRON" if batch['label'][b] == 1 else "RUIDO"
            ax.set_title(f"Bag {count+1} | {label} | clase {batch['class'][b]} "
                         f"| SNR {batch['snr'][b]:+d} dB")
            ax.set_xlabel("Burst #")
            ax.set_ylabel("Peso de atención")
            count += 1
            if count >= n_examples: break

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/attention_examples.png", dpi=150)
    plt.close()
    log.info("Ejemplos de atención guardados.")
```

---

## Módulo 11 — Función `main()` que une todo el pipeline

```python
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")

    # ── 1. Descubrir ficheros ────────────────────────────────────────────────
    train_files, test_files = discover_files(DATA_ROOT, TEST_SPLIT_FILE)

    # ── 2. Segmentar bursts (solo train+val; test se procesa aparte) ─────────
    log.info("Segmentando bursts de train/val...")
    all_bursts = []
    for fe in tqdm(train_files):
        all_bursts.extend(segment_file(fe))
    log.info(f"Total bursts extraídos: {len(all_bursts)}")

    # ── 3. Extraer features físicas ──────────────────────────────────────────
    log.info("Extrayendo features físicas...")
    all_bursts = compute_fc_deltas(all_bursts)

    # ── 4. Construir dataset y entrenar filtro de interferencias ─────────────
    X_filter, y_filter = build_filter_dataset(all_bursts)
    log.info(f"Dataset filtro: {X_filter.shape}  distribución: {np.bincount(y_filter)}")
    filter_clf = train_interference_filter(X_filter, y_filter)

    # ── 5. Filtrar bursts de train/val ───────────────────────────────────────
    log.info("Aplicando filtro de interferencias a train/val...")
    all_bursts = apply_interference_filter(filter_clf, all_bursts)

    # Guardar dataset de bursts filtrado
    pickle.dump(all_bursts, open(f"{OUTPUT_DIR}/burst_dataset.pkl", 'wb'))

    # ── 6. Construir modelos ─────────────────────────────────────────────────
    encoder = BurstEncoder().to(device)
    mil     = GatedAttentionMIL().to(device)

    # ── 7. Crear DataLoaders de train y validación ───────────────────────────
    from sklearn.model_selection import train_test_split
    train_fe, val_fe = train_test_split(
        train_files, test_size=0.15,
        stratify=[fe['class'] for fe in train_files],
        random_state=42)

    train_burst_paths = {b['file_path'] for b in all_bursts}
    train_bursts = [b for b in all_bursts if b['file_path'] in
                    {fe['path'] for fe in train_fe}]
    val_bursts   = [b for b in all_bursts if b['file_path'] in
                    {fe['path'] for fe in val_fe}]

    train_ds = BagDataset(train_fe, train_bursts, encoder)
    val_ds   = BagDataset(val_fe,   val_bursts,   encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_bags, num_workers=4,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_bags, num_workers=2)

    # ── 8. Entrenar ABMIL ────────────────────────────────────────────────────
    log.info("Entrenando ABMIL...")
    history = train_abmil(encoder, mil, train_loader, val_loader, device)
    plot_training_curves(history, OUTPUT_DIR)

    # ── 9. Cargar mejor modelo ───────────────────────────────────────────────
    checkpoint = torch.load(f"{OUTPUT_DIR}/abmil_model.pt", map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    mil.load_state_dict(checkpoint['mil'])

    # ── 10. Procesar test set ────────────────────────────────────────────────
    log.info("Procesando test set...")
    test_bursts = []
    for fe in tqdm(test_files):
        test_bursts.extend(segment_file(fe))
    test_bursts = compute_fc_deltas(test_bursts)
    test_bursts = apply_interference_filter(filter_clf, test_bursts)

    test_ds     = BagDataset(test_files, test_bursts, encoder)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_bags, num_workers=2)

    # ── 11. Inferencia y figuras ─────────────────────────────────────────────
    log.info("Generando resultados de evaluación...")
    df_results = run_inference_on_test(encoder, mil, test_loader, device)
    df_results.to_csv(f"{OUTPUT_DIR}/results/predictions.csv", index=False)

    plot_snr_class_heatmap(df_results, OUTPUT_DIR)
    plot_confusion_matrix(df_results, OUTPUT_DIR)
    plot_roc_curves(df_results, OUTPUT_DIR)
    plot_attention_examples(encoder, mil, test_loader, device, OUTPUT_DIR)

    # ── 12. Resumen numérico ─────────────────────────────────────────────────
    from sklearn.metrics import classification_report, roc_auc_score
    report = classification_report(df_results['y_true'], df_results['y_pred_bin'],
                                   target_names=['Ruido', 'Dron'])
    overall_auc = roc_auc_score(df_results['y_true'], df_results['y_pred_prob'])
    log.info(f"\n{report}")
    log.info(f"AUC global (test): {overall_auc:.4f}")

    with open(f"{OUTPUT_DIR}/results/summary.txt", 'w') as f:
        f.write(report + f"\nAUC global: {overall_auc:.4f}\n")

    log.info("Pipeline completado. Resultados en: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()
```

---

## Instrucciones específicas para el agente

1. **Leer este plan de arriba a abajo antes de escribir ningún código.**

2. **Inspeccionar `detector_entropia.py`** y localizar la función/método de segmentación.
   Adaptar la llamada en `segment_file()` a la API real que expone el módulo.

3. **Inspeccionar los ficheros del dataset** para determinar:
   - Extensión real de los ficheros IQ (`.bin`, `.dat`, `.iq`, `.npy`…)
   - Convención de nombres para extraer SNR (ajustar `parse_snr_from_filename`)
   - Convención de nombres de carpetas para extraer la clase

4. **Implementar cada módulo en el orden indicado** dentro de un único fichero `pipeline.py`.
   No dividir en múltiples ficheros salvo que sea estrictamente necesario.

5. **Gestión de dependencias**: el agente debe generar también un `requirements.txt`:
   ```
   numpy>=1.24
   scipy>=1.10
   torch>=2.0
   lightgbm>=4.0
   scikit-learn>=1.3
   pandas>=2.0
   matplotlib>=3.7
   seaborn>=0.12
   tqdm>=4.65
   ```

6. **Data augmentation opcional** (implementar si el entrenamiento muestra overfitting temprano):
   En el `__getitem__` del `BagDataset`, con probabilidad 0.3 añadir ruido AWGN a los tensores IQ:
   ```python
   if self.augment and np.random.rand() < 0.3:
       noise_sigma = 10 ** (-np.random.uniform(5, 20) / 20)
       iq_tensor += torch.randn_like(iq_tensor) * noise_sigma
   ```

7. **Si la GPU tiene memoria limitada** (< 8 GB): reducir `BATCH_SIZE` a 8 y
   `BAG_MAX_INSTANCES` a 16.

8. **Reproducibilidad**: al inicio de `main()`, fijar semillas:
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```

9. **Si la clase 4 tiene muy pocos ficheros de test**, las métricas de AUC por celda del
   heatmap pueden ser inestables. En ese caso, usar todos los ficheros de clase 4 del test
   como negativos para cada celda (clase c, SNR s) en lugar de solo los de ese SNR.

10. **El agente NO debe tocar los ficheros listados en `TEST_SPLIT_FILE`** durante ninguna
    etapa de entrenamiento o construcción del filtro.
