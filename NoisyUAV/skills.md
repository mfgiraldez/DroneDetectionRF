# skills.md — Habilidades y Pipelines Reutilizables
> Guía de referencia rápida para procesos frecuentes del proyecto

---

## 1. Lanzar un Experimento Completo (Pipeline Autónomo)

```bash
# Siempre con cmd /c para garantizar EOF en Windows
# Usar -u (unbuffered) para ver output en tiempo real
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe -u c:\repos\DroneDetectionRF\NoisyUAV\run_hybrid_experiment.py 2>&1"
```

El script hace TODO de forma autónoma:
1. Construye/carga cache de features físicas (**~36 min primera vez** — 17744 ficheros)
2. Crea DataLoaders con splits estratificados
3. Instancia el modelo y muestra summary
4. Entrena con AMP + MixUp + cosine LR + early stopping
5. Carga el mejor checkpoint (por Val F1)
6. Evalúa en test global + por SNR
7. Genera 6 figuras en `resultados_hybrid/figures/`
8. Genera informe Markdown + JSON de métricas

**Tiempos reales medidos (RTX 4060, crop_len=131072, batch_size=32):**
- Cache build (primera vez): ~36 min (17744 señales * ~0.12s/señal)
- Cache load (siguientes): ~6 segundos
- Tiempo por época: ~8.5 min (12,420 muestras, num_workers=0)
- Entrenamiento completo Run #1 (25 épocas, early stopping): 3.35 horas
- Entrenamiento completo Run #2 (60 épocas, sin early stopping): **9.94 horas**
- IMPORTANTE: La cache es incremental — nunca se recomputa lo ya calculado

---

## 2. Regenerar Figuras e Informe desde Checkpoint

Útil cuando el modelo ya está entrenado pero quieres cambiar el threshold o regenerar las figuras:

```bash
cmd /c "python c:\repos\DroneDetectionRF\NoisyUAV\regenerate_from_checkpoint.py"

# Con threshold personalizado (sweep en fig_06 indica el óptimo):
cmd /c "python regenerate_from_checkpoint.py --threshold 0.45"
```

---

## 2b. Generar Heatmap por Emisor RF (Drone Model) x SNR

Genera la figura comparativa estilo `evaluacion_analisis_modelo.py` del baseline para el HybridCVCNN:

```bash
# SIEMPRE con --split all: el test set tiene ~3-8 muestras por celda (target x SNR)
# lo que produce ceros y celdas vacas estadisticamente sin sentido
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe -u c:\repos\DroneDetectionRF\NoisyUAV\eval_heatmap_hybrid.py --split all 2>&1"
```

Genera **2 figuras** en `resultados_hybrid_run2/figures/`:
- `heatmap_target_snr.png` — Recall por cada modelo de drone x SNR (6 emisores)
- `heatmap_noise_snr.png`  — Especificidad del clasificador frente al ruido x SNR

**Estructura target_multiclass en NoisyUAV `drone_RF_data`:**
- `label=1` (drones): target_multiclass = 0, 1, 2, 3, 5, 6
- `label=0` (ruido):  target_multiclass = 4 (UNICO tipo de ruido en este dataset)
- NO hay targets 2=AWGN, 3=WiFi separados — eso era `stage2` (dataset antiguo del baseline)

**Inferencia completa (17744 muestras): ~5 minutos en RTX 4060**

---

## 3. Pre-flight Check (Verificación de Imports)

Antes de lanzar un experimento largo, verificar que todo importa correctamente:

```python
import sys; sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.physical_features import extract_features, build_features_cache
from NoisyUAV.modelos.hybrid_cvcnn import HybridCVCNN, HybridDataset
from NoisyUAV.funciones.dataset import obtener_splits_dataset

# Test forward pass
import torch
model = HybridCVCNN()
iq = torch.randn(2, 2, 131072)
f  = torch.randn(2, 12)
out = model(iq, f)
print(f"Output shape: {out.shape}")  # debe ser [2, 1]
```

---

## 4. Extracción de Features Físicas (una muestra)

```python
import torch
from NoisyUAV.funciones.physical_features import extract_features

d  = torch.load("IQdata_sample1_target1_snr10.pt", weights_only=False)
iq = d['x_iq'].float()
f  = extract_features(iq, fs=14e6, nperseg=2048)
# f.shape = (12,)  dtype=float32
```

---

## 5. Construir la Cache de Features Físicas

```python
from NoisyUAV.funciones.physical_features import build_features_cache
from NoisyUAV.funciones.dataset import obtener_splits_dataset

df_train, df_val, df_test = obtener_splits_dataset()
all_fp = list(df_train['filepath']) + list(df_val['filepath']) + list(df_test['filepath'])

cache = build_features_cache(
    filepaths=all_fp,
    cache_path=r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid\features_cache.npz",
    verbose=True,
)
# La cache se guarda en .npz: {filename: np.array [12]}
# Es incremental — si ya existe, solo computa los nuevos ficheros
```

---

## 6. Evaluación por SNR (análisis de rendimiento)

```python
from NoisyUAV.modelos.hybrid_cvcnn import HybridDataset, evaluate_hybrid

for snr in sorted(df_test['snr'].unique()):
    df_s = df_test[df_test['snr'] == snr]
    ds_s = HybridDataset(df_s, cache, crop_len=131072, augment=False,
                          phys_mean=phys_mean, phys_std=phys_std)
    dl_s = DataLoader(ds_s, batch_size=32, shuffle=False, num_workers=0)
    m = evaluate_hybrid(model, dl_s, criterion, device)
    print(f"SNR={snr:+4d} | Acc={m['acc']:.4f} | F1={m['f1']:.4f}")
```

---

## 7. Cargar un Checkpoint para Inferencia

```python
import torch
from NoisyUAV.modelos.hybrid_cvcnn import HybridCVCNN

ckpt = torch.load("resultados_hybrid/checkpoints/best_model.pt",
                  map_location='cuda', weights_only=False)

cfg   = ckpt['cfg']
model = HybridCVCNN(**{k: cfg[k] for k in
                       ['phys_dim','cnn_embed','pool_size','hidden_dim','dropout_cnn','dropout_fuse']
                       if k in cfg})
model.load_state_dict(ckpt['model_state'])
model.eval()

phys_mean = torch.tensor(ckpt['phys_mean'])
phys_std  = torch.tensor(ckpt['phys_std'])
```

---

## 8. MixUp para Clasificación Binaria (BCEWithLogitsLoss)

```python
lam    = float(torch.distributions.Beta(alpha, alpha).sample())
idx    = torch.randperm(B)
x_mix  = lam * x     + (1 - lam) * x[idx]
f_mix  = lam * feats + (1 - lam) * feats[idx]
y_b    = labels[idx]

logit = model(x_mix, f_mix)
loss  = lam * criterion(logit.squeeze(1), labels) + \
        (1-lam) * criterion(logit.squeeze(1), y_b)
```

---

## 9. Diagnostic Exploratorio de Features

Antes de implementar un modelo nuevo, validar que las features discriminan:

```bash
cmd /c "python c:\repos\DroneDetectionRF\NoisyUAV\run_diagnostic.py 2>&1"
```

Genera 3 figuras + tabla de accuracy por clasificador lineal por grupo SNR.
Si Grupo C logra >65% con regresión lineal → las features son útiles.

---

## 10. Patrón de Entrenamiento Estándar (checklist)

```
[CRITICO] iq_crop = iq_crop / iq_crop.pow(2).mean().clamp(min=1e-12).sqrt()  <- NORMALIZAR SIEMPRE
[CRITICO] Reanudación Automática: Implementar siempre bloque de carga inicial `if CHECKPOINT.exists(): torch.load(...) optimizer.load_state_dict()` para reanudar.
[ ] os.environ["PYTHONIOENCODING"] = "utf-8"
[ ] matplotlib.use("Agg")
[ ] num_workers=0 en todos los DataLoaders
[ ] CLONAR vistas tensoriales (`iq_crop.clone()`) al usar slicing en `__getitem__` para evitar memory leaks de 8MB.
[ ] cmd /c "python -u ..." para lanzar desde shell (flag -u para output en tiempo real)
[ ] torch.load(..., weights_only=False) al cargar dicts de dataset
[ ] Guardar checkpoint íntegro: model, optimizer, scheduler, scaler, phys_mean, phys_std, cfg, epoch, val_f1
[ ] Usar BCEWithLogitsLoss con pos_weight para datasets desbalanceados
[ ] GradientClip = 2.0 para evitar NaN (especialmente con GRU/LSTM)
[ ] Early stopping sobre Val F1 (no sobre Val Loss — menos sensible al desbalance)
```

---

## 11. Figuras Estándar del Proyecto (dark theme)

```python
FIGSAVE = dict(dpi=140, bbox_inches="tight", facecolor="#0F1923")

def _dark(fig, ax):
    fig.patch.set_facecolor("#0F1923")
    ax.set_facecolor("#0F1923")
    ax.tick_params(colors="#A0ADB8")
    for sp in ax.spines.values(): sp.set_edgecolor("#1E2E3E")
    ax.grid(True, color="#1E2E3E", linestyle="--", alpha=0.5)
```

Paleta de colores del proyecto:
- `#2E86AB` (Azul) — Drone / Train
- `#E84855` (Rojo) — Noise / Val
- `#3BB273` (Verde) — F1 / mejora
- `#F4A261` (Ámbar) — Baseline / threshold
- `#6A0572` (Púrpura) — Specificity

---

## 12. Diagnóstico de NaN en Loss

Si aparece `TrLoss=NaN`:
1. Añadir `nn.utils.clip_grad_norm_(model.parameters(), 2.0)` ANTES del optimizer.step()
2. Reducir LR inicial (de 3e-4 a 1e-4)
3. Reducir mixup_alpha (de 0.4 a 0.2)
4. Desactivar AMP temporalmente (`use_amp=False`)
5. Verificar que el DataLoader no devuelve tensores con Inf/NaN: `torch.any(torch.isnan(x))`
