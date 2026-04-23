# PROJECT MEMORY — DroneDetectionRF TFM
> Actualizar este fichero cada vez que se aprenda algo nuevo, se corrija un error o cambie el estado del proyecto.
> Última actualización: 2026-04-23

---

## 1. ENTORNO DE EJECUCIÓN

| Elemento | Valor |
|---|---|
| OS | Windows |
| Shell | PowerShell — **SIEMPRE usar `cmd /c` para ejecuciones de proceso** |
| Python | 3.11.5 |
| Conda env | `C:\Users\Manuel\anaconda3\envs\IAIAVv3` |
| Ejecutable Python | `C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe` |
| PyTorch | 2.5.1+cu124 |
| Torchaudio | 2.5.1+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 4060 |
| `mamba-ssm` | ❌ No instalado → se usa BiGRU cuDNN como fallback |
| scikit-learn | ✅ Disponible |

**Regla crítica de ejecución:**
```powershell
# SIEMPRE así para evitar procesos colgados en Windows:
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe <script.py> 2>&1"
```

---

## 2. ESTRUCTURA DEL REPOSITORIO

```
C:\repos\DroneDetectionRF\
├── NoisyUAV\                        ← Proyecto principal TFM
│   ├── modelos\
│   │   ├── __init__.py              ← Exporta ambos modelos + utilidades
│   │   ├── cvcnn.py                 ← Baseline CV-CNN (~75% acc)
│   │   └── marnet_fusion.py         ← MaRNet-Fusion (SotA target >90%)
│   ├── funciones\
│   │   ├── __init__.py              ← Re-exporta constantes y utilidades
│   │   ├── cargador.py              ← Constantes del dataset + carga de .pt
│   │   ├── dataset.py               ← obtener_splits_dataset() + NoisyUAVDataset
│   │   ├── dataset_stage2.py        ← Dataset para el pipeline Stage2
│   │   ├── visualizacion.py         ← panel_completo(), comparar_snr(), comparar_clases()
│   │   ├── detector_entropia.py     ← Detector heurístico basado en entropía Shannon
│   │   ├── detector_masivo_stage1.py
│   │   ├── entrenar_stage2.py
│   │   ├── entrenar_stage2_norm.py
│   │   ├── evaluacion_analisis_modelo.py
│   │   └── evaluar_entropia.py
│   ├── resultados_marnet\           ← Resultados del último experimento MaRNet
│   │   ├── checkpoints\
│   │   │   ├── best_model.pt        ← Pesos del mejor modelo (Val F1=0.7261, época 35)
│   │   │   └── last_model.pt
│   │   ├── figures\                 ← 9 figuras PNG del experimento
│   │   ├── informe_marnet_fusion.md ← Informe completo del experimento
│   │   └── metricas_completas.json  ← Todas las métricas serializadas (historial, SNR, etc.)
│   └── run_marnet_experiment.py     ← Script autónomo de entrenamiento+evaluación+informe
└── TFM_RF_Fingerprinting\           ← Rama secundaria (RF Fingerprinting, en pausa)
```

**Importar en scripts:**
```python
import sys; sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelos import MaRNetFusion, RFDroneDataset
from NoisyUAV.funciones import DATA_DIR, NOMBRES_CLASES
from NoisyUAV.funciones.dataset import obtener_splits_dataset
```

---

## 3. DATASET: NoisyUAV v2

### 3.1 Referencia
- **Paper:** Glüge et al. (2024). *"Robust Low-Cost Drone Detection and Classification Using CNNs in Low SNR Environments"*
- **Fuente:** Kaggle — `sgluege/noisy-drone-rf-signal-classification-v2`
- **Ruta local:** `C:\TFM_data\NoisyUAV\drone_RF_data\`

### 3.2 Parámetros de Adquisición

| Parámetro | Valor |
|---|---|
| Frecuencia de muestreo | **14 MHz** (downsampled de 56 MHz originales) |
| Muestras por archivo | **1.048.576** (= 2²⁰) |
| Duración por muestra | **~74.9 ms** |
| Formato | `.pt` (PyTorch tensor) |
| Total de archivos | **17.744** |

### 3.3 Estructura de cada archivo `.pt`
```python
d = torch.load("IQdata_sample0_target0_snr-14.pt", map_location="cpu", weights_only=False)
d.keys()   # → ['x_iq', 'y', 'snr']
d['x_iq']  # → Tensor[2, 1048576], dtype=float32
            #   canal 0 = I (In-phase / parte real)
            #   canal 1 = Q (Quadrature / parte imaginaria)
d['y']     # → Tensor escalar, clase 0-6
d['snr']   # → Tensor escalar, SNR en dB (entero)
```

### 3.4 Convención de Nombres de Archivo
```
IQdata_sample{sample_id}_target{target_class}_snr{snr_dB}.pt
```
- `sample_id`: índice numérico de la muestra
- `target_class`: 0-6 (ver tabla de clases)
- `snr_dB`: puede ser negativo, p.ej. `snr-14`

### 3.5 Clases

| Index | Nombre | Tipo | Label binario | Nº archivos |
|---|---|---|---|---|
| 0 | DJI | Drone | **1 (Drone)** | 1.280 |
| 1 | FutabaT14 | Drone | **1 (Drone)** | 3.472 |
| 2 | FutabaT7 | Drone | **1 (Drone)** | 801 |
| 3 | Graupner | Drone | **1 (Drone)** | 801 |
| **4** | **Noise** | **Ruido** | **0 (No-Drone)** | **8.872** |
| 5 | Taranis | Drone | **1 (Drone)** | 1.663 |
| 6 | Turnigy | Drone | **1 (Drone)** | 855 |

> ⚠️ **`TARGET_NOISE = 4`** — El ruido es el target 4 (orden alfabético del CSV). Esto es **diferente** del orden del paper original. No confundir.

**Balance:** ~8.872 drones vs ~8.872 ruido → dataset **perfectamente balanceado** en binario.

### 3.6 Niveles de SNR

- **Rango:** -20 dB a +30 dB en pasos de 2 dB → 26 niveles
- **Muestras por nivel:** ~681-686 (muy uniforme)
- **Grupos operacionales:**

| Grupo | Rango SNR | Descripción |
|---|---|---|
| **A** | SNR ≥ 10 dB | Fácil — señal dominante |
| **B** | -6 dB ≤ SNR < 10 dB | Medio — régimen funcional |
| **C** | SNR < -6 dB | Difícil — régimen hostil crítico |

### 3.7 Splits del Dataset
Generados con `obtener_splits_dataset()` (estratificado por clase AND SNR):

| Split | Archivos | % |
|---|---|---|
| Train | 12.420 | ~70% |
| Val | 2.662 | ~15% |
| Test | 2.662 | ~15% |

```python
from NoisyUAV.funciones.dataset import obtener_splits_dataset
df_train, df_val, df_test = obtener_splits_dataset(
    data_dir=r"C:\TFM_data\NoisyUAV\drone_RF_data",
    test_size=0.15, val_size=0.15, random_state=42
)
# Columnas del DataFrame: ['filepath', 'target_multiclass', 'label', 'snr', 'grupo', 'stratify_key']
```

---

## 4. MODELOS DESARROLLADOS

### 4.1 Baseline: CV-CNN (`modelos/cvcnn.py`)

**ComplexConv1DNet** — Red neuronal convolucional de valores complejos (Complex-Valued CNN).

| Atributo | Valor |
|---|---|
| Archivo | `NoisyUAV/modelos/cvcnn.py` |
| Clase principal | `ComplexConv1DNet` |
| Tarea | Clasificación binaria: Drone vs Ruido |
| Accuracy reportada | **~75%** (sobre NoisyUAV v2) |
| Input | `[B, 2, N]` — secuencia IQ de longitud variable |
| Output | logits `[B, 2]` — dos clases |

**Arquitectura:**
```
[B, 2, N]
→ ComplexConv1d(1→32,  k=31, s=2) + CReLU
→ ComplexConv1d(32→64, k=15, s=2) + CReLU
→ ComplexConv1d(64→128,k=7,  s=2) + CReLU
→ ComplexConv1d(128→128,k=3, s=1) + CReLU
→ modulus |z| = sqrt(Re²+Im²)  [dominio real]
→ AdaptiveAvgPool1d(64)         [B, 128, 64]
→ Flatten → [B, 8192]
→ Linear(8192→512) + ReLU + Dropout(0.4)
→ Linear(512→64)   + ReLU
→ Linear(64→2)     → logits
```

**Piezas reutilizables:**
- `ComplexConv1d`: convolución compleja via diferencial de Wirtinger (`Re(W)·I − Im(W)·Q`, `Re(W)·Q + Im(W)·I`)
- `CReLU`: ReLU aplicada a Re e Im por separado
- `modulus`: convierte a dominio real tomando `sqrt(Re² + Im²)`

---

### 4.2 SotA Propuesto: MaRNet-Fusion (`modelos/marnet_fusion.py`)

**Multi-Branch RF Network with Cross-Attention Fusion** — arquitectura 2024-2026 diseñada para SNR < -10 dB.

| Atributo | Valor |
|---|---|
| Archivo | `NoisyUAV/modelos/marnet_fusion.py` |
| Clase principal | `MaRNetFusion` |
| Parámetros | **3.134.179** |
| VRAM estimada (BS=32) | ~0.40 GB |
| Backend SSM | BiGRU cuDNN (fallback de mamba-ssm) |

**Arquitectura de 3 ramas + fusión:**
```
z(t) ∈ C^L  (IQ banda-base)
│
├── RAMA 1 — ResNet-Lite 2D (espectrogramas)
│   [B,1,65,65] via STFT(n_fft=128, hop=32, crop=2048)
│   → SoftThresholdingBlock (denoising diferenciable SE-attention)
│   → ResNet: Stem(32,k7) → ResBlocks×2 ×3 étapes → AdaptivePool(4,4)
│   → e_spec [B, 256]
│
├── RAMA 2 — BiGRU cuDNN (~Bi-Mamba SSM)
│   [B, 2, 2048] normalizado z-score
│   → Linear(2→128) + BiGRU(hidden=128, layers=3, bidi=True)
│   → Mean temporal pooling → Linear(256→128) → e_iq [B, 256]
│
├── RAMA 3 — MLP para estadísticos HOS
│   [B, 5]: (Entropía PSD, Amplitud media, Varianza total, BW ocupado, Kurtosis exc.)
│   → 5→64→128→128→128 con LayerNorm+GELU+Dropout
│   → e_stat [B, 128]
│
└── PAM_FUSION (Gated Cross-Attention)
    Q = Linear([e_spec ∥ e_iq ∥ e_stat])       [B, 256]
    K, V = stack([e_spec, e_iq, e_stat])        [B, 3, 256]
    w = softmax(Q·K^T / √256)                   [B, 3] — pesos adaptativos por instancia
    e_fused = Σᵢ(wᵢ·Vᵢ) + skip + FFN + LN
    → Logit [B, 1]  (BCEWithLogitsLoss)
```

**Subsistemas exportados desde `marnet_fusion.py`:**
| Clase/Función | Descripción |
|---|---|
| `RFDroneDataset` | Dataset multi-salida: devuelve `(spectrogram, iq_sequence, stat_features, label)` |
| `SoftThresholdingBlock` | Denoising diferenciable via SE-attention |
| `ResNetBranch` | Rama CNN 2D para espectrogramas |
| `MambaBranch` | Rama BiGRU para secuencias IQ |
| `MLPBranch` | Rama MLP para estadísticos HOS |
| `PAM_Fusion` | Módulo de fusión cross-attention |
| `MaRNetFusion` | Modelo completo ensamblado |
| `train_one_epoch` | Loop de entrenamiento con MixUp+AMP |
| `evaluate` | Evaluación con métricas completas |
| `mixup_batch` | MixUp estocástico Beta(α,α) |
| `mixup_criterion` | Loss interpolada para MixUp |
| `_MAMBA_AVAILABLE` | Bool: si mamba-ssm está instalado |

**Instanciación estándar:**
```python
model = MaRNetFusion(
    latent_dim_spec=256, latent_dim_iq=256, latent_dim_stat=128,
    d_fusion=256, d_model_ssm=128, d_state_ssm=16, num_ssm_layers=3,
    dropout_cnn=0.3, dropout_ssm=0.2, dropout_mlp=0.4,
).to(device)
```

**Dataset multi-rama — uso standard:**
```python
from NoisyUAV.modelos.marnet_fusion import RFDroneDataset
ds = RFDroneDataset(df, crop_len=2048, n_fft=128, hop_length=32, augment=True)
# Devuelve: spec[1,65,65], iq[2,2048], stat[5], label(int)
```

---

## 5. DETECTORES HEURÍSTICOS

### 5.1 Detector de Entropía Shannon
- **Archivo:** `funciones/detector_entropia.py`
- **Concepto:** Umbral adaptativo sobre la entropía del PSD. Drones FHSS → entropía alta. Ruido gaussiano → entropía máxima (espectro plano).
- **Estado:** Implementado y evaluado. Supera naive baseline pero inferior a CNN.

---

## 6. RESULTADOS DEL ÚLTIMO EXPERIMENTO (MaRNet-Fusion Run #1)

### 6.1 Configuración
| Parámetro | Valor |
|---|---|
| `crop_len` | 2048 muestras (≈ 146 µs a 14 MHz) |
| `n_fft` | 128 → F=65 bins, T=65 frames |
| `hop_length` | 32 |
| `batch_size` | 48 |
| `epochs` | 50 (early stopping en época 45) |
| `lr` | 3e-4 AdamW + CosineAnnealingLR |
| `weight_decay` | 1e-4 |
| `mixup_alpha` | 0.4 |
| `mixup_prob` | 0.5 |
| `grad_clip` | 2.0 |
| `curriculum_switch_epoch` | 18 (épocas 1-18: grupos A+B; 19+: A+B+C) |
| `use_amp` | True (GradScaler CUDA) |
| `num_workers` | 0 (obligatorio en Windows para evitar errores spawn) |

### 6.2 Métricas Globales (Test Set)

| Métrica | Valor |
|---|---|
| **Accuracy** | **65.66%** |
| **F1-Score** | **0.7270** |
| Precision | 60.31% |
| Recall | 91.50% |
| Especificidad | 39.86% |
| Baseline CV-CNN | ~75.00% |
| Diferencia | **-9.34 p.p.** (el modelo tiene alto recall pero baja especificidad) |

### 6.3 Métricas por Grupo SNR

| Grupo | Rango | Accuracy Media |
|---|---|---|
| A (fácil) | SNR ≥ 10 dB | **71.59%** |
| B (medio) | -6..10 dB | **67.18%** |
| C (difícil) | SNR < -6 dB | **50.84%** |

### 6.4 Análisis del Comportamiento de la Atención

Observado durante el entrenamiento y evaluación:
- **Épocas tempranas (1-10):** La rama BiGRU domina (peso ~0.5-0.8)
- **Después del curriculum switch (época 19+):** Los estadísticos HOS ganan peso (+0.5) al incorporar muestras del grupo C
- **A SNR bajo:** Los estadísticos MLP tienen el mayor peso (kurtosis + entropía PSD son los únicos detectores estables)
- **Aparición de TrLoss=NaN:** En épocas 29-37 (intermitente) — causado por gradientes explosivos en el BiGRU con group C de bajo SNR. El grad_clip=2.0 lo contiene eventualmente y el modelo se recupera.

### 6.5 Diagnóstico de Problemas Observados

| Problema | Causa | Solución aplicada |
|---|---|---|
| `TrLoss=NaN` épocas 29-37 | Gradientes explosivos BiGRU al añadir grupo C | `grad_clip=2.0` + AMP los contiene |
| Recall alto (91%) pero Acc baja (65%) | El modelo está sesgado hacia predecir "Drone" siempre | Revisar pos_weight, threshold, o reforzar Especificidad |
| `NameError: _MAMBA_AVAILABLE` en informe | Variable no importada en script de experimento | Añadir a imports de `run_marnet_experiment.py` |
| Unicode en Windows (cp1252) | Caracteres como `►` en summary() | Reemplazados por `>` ASCII |
| `num_workers > 0` en Windows | DataLoader con fork → crash | `num_workers=0` siempre en Windows |

### 6.6 Archivos Generados
- `resultados_marnet/checkpoints/best_model.pt` — pesos época 35, Val F1=0.7261
- `resultados_marnet/checkpoints/last_model.pt` — pesos época 45
- `resultados_marnet/metricas_completas.json` — historial completo serializable
- `resultados_marnet/figures/` — 9 figuras PNG (curvas, confusion matrix, ROC, etc.)
- `resultados_marnet/informe_marnet_fusion.md` — informe académico completo

---

## 7. ISSUES CONOCIDOS Y PENDING ITEMS

### 7.1 Problemas Pendientes de Resolver
- [ ] **Recall>>Especificidad:** El modelo actual predice "Drone" demasiado frecuentemente. Posibles causas: (a) pos_weight mal calibrado (~1.0 con dataset balanceado), (b) MixUp sin ajuste de umbral, (c) falta de focal loss. Investigar.
- [ ] **Accuracy 65% < Baseline 75%:** Se esperaba superar al baseline. Posibles causas: (a) crop_len=2048 muy corto (solo 146 µs, puede no capturar estructura FHSS completa), (b) n_fft=128 con hop=32 → espectrograma 65×65 quizás demasiado pequeño, (c) necesita más épocas o LR más bajo.
- [ ] **TrLoss=NaN intermitente:** Aunque el modelo se recupera, es señal de inestabilidad. Considerar gradient clipping más agresivo (1.0) o LR más bajo en fase 2.
- [ ] **mamba-ssm no instalado:** El BiGRU es un buen fallback pero el SSM selectivo real podría mejorar la captura de correlaciones de largo alcance.

### 7.2 Próximos Experimentos Sugeridos
1. **Aumentar `crop_len`** a 8192 o 16384 (≈ 0.6-1.2 ms, más representativo de un burst FHSS)
2. **Aumentar `n_fft`** a 256 y `hop_length` a 64 → espectrograma más rico
3. **Ajustar umbral de clasificación** post-hoc: usar threshold ≠ 0.5 para equilibrar precisión/recall
4. **Focal Loss** en lugar de BCEWithLogitsLoss para penalizar más los falsos positivos
5. **Reducir `dropout_ssm`** a 0.1 para estabilizar la rama BiGRU

---

## 8. OTROS DATASETS EXPLORADOS

| Dataset | Ruta/Fuente | Estado | Notas |
|---|---|---|---|
| NoisyUAV v2 | `C:\TFM_data\NoisyUAV\drone_RF_data` | ✅ Principal | Ver sección 3 |
| DroneRF | (notebooks `prueba_DroneRF.ipynb`) | Explorado | Diferente esquema de clases y fs |
| RFUAV | (notebooks `prueba_RFUAV.ipynb`) | Explorado | Muy grande (89 MB notebook) |
| DRFF-R2 | (notebooks `prueba_DRFF-R2.ipynb`) | Explorado | 89 MB notebook |

---

## 9. DECISIONES DE DISEÑO FIJAS

1. **Clasificación BINARIA:** Drone (1) vs Ruido/No-Drone (0). No multiclase.
2. **TARGET_NOISE = 4** en NoisyUAV — es el índice del ruido en el dataset.
3. **num_workers = 0** siempre en Windows para DataLoader.
4. **cmd /c** obligatorio para ejecuciones de subprocess en Windows.
5. **BCEWithLogitsLoss** (no softmax) para clasificación binaria — más estable numéricamente.
6. **AMP activado** con GradScaler para CUDA — mejora velocidad ~2x.
7. **early_stopping** sobre Val F1 (no sobre Val Loss) por robustez a datasets desbalanceados.
8. **Estratificación por clase×SNR** en los splits para garantizar representación uniforme de todos los niveles de dificultad.
