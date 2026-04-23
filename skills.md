# PROJECT SKILLS — DroneDetectionRF TFM
> Habilidades, pipelines y procesos recurrentes desarrollados a lo largo del proyecto.
> Consultar antes de implementar cualquier proceso para reutilizar código ya validado.
> Última actualización: 2026-04-23

---

## ÍNDICE

1. [Exploración rápida del dataset](#1-exploración-rápida-del-dataset)
2. [Pipeline de entrenamiento completo](#2-pipeline-de-entrenamiento-completo)
3. [Evaluación por nivel de SNR](#3-evaluación-por-nivel-de-snr)
4. [Generación de figuras y informe](#4-generación-de-figuras-e-informe)
5. [Diagnóstico de un modelo entrenado](#5-diagnóstico-de-un-modelo-entrenado)
6. [Pre-flight check (smoke test)](#6-pre-flight-check-smoke-test)
7. [Construcción de DataLoaders multi-rama](#7-construcción-de-dataloaders-multi-rama)
8. [Curriculum Learning](#8-curriculum-learning)
9. [Guardar y recargar checkpoints](#9-guardar-y-recargar-checkpoints)
10. [Serialización de métricas a JSON](#10-serialización-de-métricas-a-json)
11. [Patrones de debugging recurrentes](#11-patrones-de-debugging-recurrentes)
12. [Comandos de ejecución frecuentes](#12-comandos-de-ejecución-frecuentes)

---

## 1. EXPLORACIÓN RÁPIDA DEL DATASET

### Inspeccionar archivos y distribución
```python
import glob, re, collections, torch

files = glob.glob(r'C:\TFM_data\NoisyUAV\drone_RF_data\IQdata_*.pt')
pat = re.compile(r'target(\d+)_snr(-?\d+)')
targets, snrs = collections.Counter(), collections.Counter()
for f in files:
    m = pat.search(f)
    if m:
        targets[m.group(1)] += 1
        snrs[m.group(2)] += 1

print(f'Total archivos: {len(files)}')
print(f'Por target: {dict(sorted(targets.items()))}')
print(f'Niveles SNR: {sorted([int(k) for k in snrs.keys()])}')

# Peek a un archivo
d = torch.load(files[0], map_location='cpu', weights_only=False)
print(f"Keys: {list(d.keys())}")
print(f"x_iq shape: {d['x_iq'].shape}")  # [2, 1048576]
```

### Obtener splits y ver distribuciones
```python
from NoisyUAV.funciones.dataset import obtener_splits_dataset
df_train, df_val, df_test = obtener_splits_dataset()

# Distribución por grupo SNR y clase
print(df_train.groupby(['grupo', 'label']).size().unstack())
print(df_test['snr'].value_counts().sort_index())
```

---

## 2. PIPELINE DE ENTRENAMIENTO COMPLETO

### Script autónomo (preferido para runs largos)
```bash
# Ejecutar en background — tarda ~1-2 horas en RTX 4060
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe c:\repos\DroneDetectionRF\NoisyUAV\run_marnet_experiment.py 2>&1"
```

El script `run_marnet_experiment.py` hace:
1. Carga dataset y construye DataLoaders (fase 1 y fase 2 para curriculum)
2. Instancia `MaRNetFusion` con la config de `CFG`
3. Entrena con `train_one_epoch` (MixUp + AMP + gradient clipping)
4. Early stopping sobre Val F1
5. Carga el mejor checkpoint automáticamente
6. Evalúa en test set global + por nivel SNR
7. Genera 9 figuras PNG
8. Serializa todas las métricas a JSON
9. Genera informe Markdown completo

### Loop de entrenamiento mínimo (para experimentos rápidos)
```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from NoisyUAV.modelos.marnet_fusion import MaRNetFusion, RFDroneDataset, train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MaRNetFusion().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
crit   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
scaler = torch.amp.GradScaler() if device.type == "cuda" else None

ds_train = RFDroneDataset(df_train, crop_len=2048, n_fft=128, hop_length=32, augment=True)
dl_train = DataLoader(ds_train, batch_size=48, shuffle=True, num_workers=0)

for epoch in range(1, 51):
    tr = train_one_epoch(model, dl_train, opt, crit, device, scaler=scaler)
    vl = evaluate(model, dl_val_loader, crit, device)
    print(f"Ep {epoch} | TrAcc={tr['acc']:.4f} | VlF1={vl['f1']:.4f}")
```

---

## 3. EVALUACIÓN POR NIVEL DE SNR

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from NoisyUAV.modelos.marnet_fusion import RFDroneDataset, evaluate

@torch.no_grad()
def evaluate_by_snr(model, df_test, device, batch_size=48):
    """Devuelve dict {snr_int: {acc, precision, recall, f1, n}}"""
    criterion = nn.BCEWithLogitsLoss()
    results = {}
    for snr in sorted(df_test['snr'].unique()):
        df_s = df_test[df_test['snr'] == snr]
        ds = RFDroneDataset(df_s, crop_len=2048, n_fft=128, hop_length=32, augment=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        m = evaluate(model, dl, criterion, device)
        results[int(snr)] = {k: m[k] for k in ['acc', 'precision', 'recall', 'f1']}
        results[int(snr)]['n'] = len(df_s)
        print(f"SNR={snr:+4d} dB | Acc={m['acc']:.4f} F1={m['f1']:.4f}")
    return results
```

---

## 4. GENERACIÓN DE FIGURAS E INFORME

### Paleta de colores corporativa del proyecto
```python
import matplotlib
matplotlib.use("Agg")   # SIEMPRE backend no-interactivo en scripts de fondo
import matplotlib.pyplot as plt

# Colores fijos del proyecto
BLUE   = "#2E86AB"
RED    = "#E84855"
GREEN  = "#3BB273"
GRAY   = "#6C757D"
AMBER  = "#F4A261"
PURPLE = "#6A0572"

# Estilo oscuro homogéneo
plt.rcParams.update({
    "figure.facecolor": "#0F1923",
    "axes.facecolor":   "#0F1923",
    "text.color":       "#E8EDF2",
    "axes.labelcolor":  "#E8EDF2",
    "xtick.color":      "#A0ADB8",
    "ytick.color":      "#A0ADB8",
    "grid.color":       "#1E2E3E",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
FIGSAVE_KW = dict(dpi=150, bbox_inches="tight", facecolor="#0F1923")

# Guardar siempre así:
fig.savefig(path, **FIGSAVE_KW)
plt.close(fig)
```

### Regenerar informe desde métricas JSON guardadas
Si el script falla al generar el informe pero las métricas ya están guardadas:
```python
import json
from NoisyUAV.run_marnet_experiment import generate_report
import NoisyUAV.run_marnet_experiment as rme

with open(r"...\resultados_marnet\metricas_completas.json") as f:
    data = json.load(f)

# Patch si sklearn no puede calcular AUC (no hay probs guardados)
rme.SKLEARN_OK = False

# Preparar argumentos mínimos
CFG = {k: _convert_type(v) for k, v in data["config"].items()}
history = data["history"]
test_results = {
    "acc": data["test_acc"], "f1": data["test_f1"],
    "precision": data["test_precision"], "recall": data["test_recall"],
    "specificity": data["test_specificity"], "loss": data["test_loss"],
    "probs": [], "preds": [], "labels": []
}
snr_results = {int(k): v for k, v in data["snr_results"].items()}

class MockModel:
    def count_parameters(self): return data["model_params"]

generate_report(CFG, history, test_results, snr_results, MockModel(), t_start)
```

> ⚠️ **Lección aprendida:** Guardar siempre `probs` y `labels` en el JSON para poder calcular AUC offline. Actualmente `metricas_completas.json` NO los incluye por volumen.

---

## 5. DIAGNÓSTICO DE UN MODELO ENTRENADO

### Cargar checkpoint y evaluar
```python
import torch, torch.nn as nn
from NoisyUAV.modelos.marnet_fusion import MaRNetFusion, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaRNetFusion().to(device)

ckpt = torch.load(r"resultados_marnet\checkpoints\best_model.pt",
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Cargado: época={ckpt['epoch']}, Val F1={ckpt['val_f1']:.4f}")

# Métricas globales
crit = nn.BCEWithLogitsLoss()
metrics = evaluate(model, dl_test, crit, device)
print(metrics)  # acc, precision, recall, f1, specificity, probs, labels, preds
```

### Inspeccionar pesos de atención
```python
@torch.no_grad()
def get_attention_by_snr(model, df_test, device, batch_size=48):
    from NoisyUAV.modelos.marnet_fusion import RFDroneDataset
    from torch.utils.data import DataLoader

    model.eval()
    attn_by_snr = {}
    for snr in sorted(df_test['snr'].unique()):
        df_s = df_test[df_test['snr'] == snr]
        ds = RFDroneDataset(df_s, crop_len=2048, n_fft=128, hop_length=32, augment=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        attn_acc = torch.zeros(3)
        n = 0
        for spec, iq, stat, _ in dl:
            _, attn_w = model(spec.to(device), iq.to(device), stat.to(device))
            attn_acc += attn_w.mean(0).cpu()
            n += 1
        attn_by_snr[int(snr)] = (attn_acc / n).numpy()   # [w_spec, w_iq, w_stat]
    return attn_by_snr
```

### Ajuste post-hoc del umbral de clasificación
```python
import numpy as np
from sklearn.metrics import f1_score

# Buscar el threshold óptimo sobre el conjunto de validación
probs_val = ...  # numpy array de probabilidades sigmoid
labels_val = ...  # numpy array de etiquetas verdaderas

thresholds = np.linspace(0.1, 0.9, 81)
f1s = [f1_score(labels_val, (probs_val > t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
print(f"Mejor threshold: {best_t:.2f} → F1={max(f1s):.4f}")

# Aplicar en test
preds_test = (probs_test > best_t).astype(int)
```

---

## 6. PRE-FLIGHT CHECK (SMOKE TEST)

Ejecutar siempre antes de lanzar un entrenamiento largo:

```python
import sys, torch
sys.path.insert(0, r"c:\repos\DroneDetectionRF")

from NoisyUAV.modelos.marnet_fusion import MaRNetFusion, RFDroneDataset, mixup_batch
from NoisyUAV.funciones.dataset import obtener_splits_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  CUDA: {torch.cuda.is_available()}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Modelo forward pass
model = MaRNetFusion(d_model_ssm=64, d_state_ssm=8, num_ssm_layers=2).to(device)
model.summary()

B, F, T, L = 4, 65, 65, 2048
spec = torch.randn(B, 1, F, T, device=device)
iq   = torch.randn(B, 2,  L,   device=device)
stat = torch.randn(B, 5,      device=device)
labs = torch.randint(0, 2, (B,), device=device)

logit, attn = model(spec, iq, stat)
assert logit.shape == (B, 1), f"Shape error: {logit.shape}"
assert abs(attn.sum(dim=1).mean().item() - 1.0) < 1e-4, "Attention sums ≠ 1"

# 2. Dataset item
df_train, df_val, df_test = obtener_splits_dataset()
ds = RFDroneDataset(df_test.head(3), crop_len=2048, n_fft=128, hop_length=32)
sp, iq_, st, lbl = ds[0]
assert sp.shape == (1, 65, 65), f"Spec shape error: {sp.shape}"
assert iq_.shape == (2, 2048), f"IQ shape error: {iq_.shape}"
assert st.shape == (5,), f"Stat shape error: {st.shape}"

# 3. MixUp
spec_m, iq_m, stat_m, ya, yb, lam = mixup_batch(spec, iq, stat, labs)
assert 0 < lam < 1

print("\nPRE-FLIGHT CHECK: ALL OK")
```

---

## 7. CONSTRUCCIÓN DE DATALOADERS MULTI-RAMA

```python
from torch.utils.data import DataLoader
from NoisyUAV.modelos.marnet_fusion import RFDroneDataset
from NoisyUAV.funciones.dataset import obtener_splits_dataset

def build_dataloaders(data_dir, crop_len=2048, n_fft=128, hop_length=32,
                      batch_size=48, curriculum=True):
    df_train, df_val, df_test = obtener_splits_dataset(data_dir=data_dir)

    if curriculum:
        df_train_f1 = df_train[df_train['grupo'].isin(['A', 'B'])]  # fase 1
        df_train_f2 = df_train                                        # fase 2 (todos)
    else:
        df_train_f1 = df_train_f2 = df_train

    ds_kwargs = dict(crop_len=crop_len, n_fft=n_fft, hop_length=hop_length)
    dl_kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    # ⚠️ num_workers=0 OBLIGATORIO en Windows

    dl_train_f1 = DataLoader(RFDroneDataset(df_train_f1, augment=True,  **ds_kwargs),
                             shuffle=True,  **dl_kwargs)
    dl_train_f2 = DataLoader(RFDroneDataset(df_train_f2, augment=True,  **ds_kwargs),
                             shuffle=True,  **dl_kwargs)
    dl_val      = DataLoader(RFDroneDataset(df_val,       augment=False, **ds_kwargs),
                             shuffle=False, **dl_kwargs)
    dl_test     = DataLoader(RFDroneDataset(df_test,      augment=False, **ds_kwargs),
                             shuffle=False, **dl_kwargs)

    return dl_train_f1, dl_train_f2, dl_val, dl_test, df_train, df_val, df_test
```

---

## 8. CURRICULUM LEARNING

Estrategia implementada y validada:

```python
CFG_CURRICULUM_EPOCH = 18  # switch de fase 1 a fase 2

for epoch in range(1, max_epochs + 1):
    phase = 1 if epoch <= CFG_CURRICULUM_EPOCH else 2
    dl_train = dl_train_f1 if phase == 1 else dl_train_f2

    if epoch == CFG_CURRICULUM_EPOCH + 1:
        print(f">>> [CURRICULUM] Fase 2: incorporando grupo C (SNR < -6 dB)")

    # LR warmup lineal primeras 5 épocas
    if epoch <= 5:
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * epoch / 5
    else:
        scheduler.step()
```

**Observación:** Al hacer el switch, `TrAcc` cae ~7% porque el grupo C es mucho más difícil. Esto es normal y esperado. El modelo se recupera en 3-5 épocas.

---

## 9. GUARDAR Y RECARGAR CHECKPOINTS

### Guardar (dentro del loop de entrenamiento)
```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_f1": best_val_f1,
    "cfg": cfg,
}, ckpt_path / "best_model.pt")
```

### Recargar para evaluación o fine-tuning
```python
ckpt = torch.load("best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
# Para fine-tuning, también restaurar optimizer:
# optimizer.load_state_dict(ckpt["optimizer_state"])
print(f"Cargado desde época {ckpt['epoch']}, Val F1={ckpt['val_f1']:.4f}")
```

> ⚠️ Usar siempre `weights_only=False` con nuestros checkpoints (contienen dicts con cfg).

---

## 10. SERIALIZACIÓN DE MÉTRICAS A JSON

```python
import json, numpy as np, torch

def save_metrics_json(path, test_results, history, snr_results, cfg, model, t_start):
    def _convert(o):
        if isinstance(o, (np.integer, np.floating)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.item() if o.numel() == 1 else o.tolist()
        return o

    export = {
        "test_acc":       test_results["acc"],
        "test_f1":        test_results["f1"],
        "test_precision": test_results["precision"],
        "test_recall":    test_results["recall"],
        "test_specificity": test_results["specificity"],
        "test_loss":      test_results["loss"],
        # ⚠️ GUARDAR TAMBIÉN probs y labels para AUC offline:
        # "test_probs":   test_results["probs"],   ← recomendado en futuros runs
        # "test_labels":  test_results["labels"],  ← recomendado en futuros runs
        "best_epoch":     history.get("best_epoch", 0),
        "snr_results":    {str(k): v for k, v in snr_results.items()},
        "history":        {k: v for k, v in history.items()
                           if isinstance(v, list) and len(v) > 0
                           and not isinstance(v[0], list)},
        "config":         {k: str(v) for k, v in cfg.items()},
        "model_params":   model.count_parameters(),
        "total_time_hours": (time.time() - t_start) / 3600,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: _convert(v) for k, v in export.items()}, f, indent=2)
```

---

## 11. PATRONES DE DEBUGGING RECURRENTES

### NaN en training loss
```python
# Síntoma: TrLoss=nan, TrAcc normal (las predicciones siguen funcionando)
# Causa: gradiente explosivo en BiGRU al cambiar a grupo C
# Diagnóstico:
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN grad: {name}")
# Solución:
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # más agresivo que 2.0
```

### Recall muy alto, Specificity baja (modelo dice "Drone" siempre)
```python
# Diagnóstico:
probs = torch.sigmoid(logit.squeeze(1))
print(f"Mean prob: {probs.mean():.4f}")  # si > 0.6, el modelo está sesgado
# Soluciones:
# 1. Threshold post-hoc (ver sección 5)
# 2. Focal loss para penalizar FP:
criterion = FocalLoss(gamma=2.0, alpha=0.5)
# 3. Reducir pos_weight si el dataset está balanceado:
pos_weight = torch.tensor([0.5])  # penalizar menos los positivos
```

### UnicodeEncodeError en Windows (cp1252)
```python
# Síntoma: 'charmap' codec can't encode character '\u2588' etc.
# Causa: caracteres Unicode en print() o logging en consola Windows
# Solución: reemplazar todos los caracteres especiales por ASCII equivalentes
# ► → >   ✓ → OK   ✗ → X   § → S   etc.
# O redirigir stdout:
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```

### DataLoader cuelga / no termina en Windows
```python
# Causa: num_workers > 0 usa fork() → incompatible con Windows
# Solución ÚNICA:
DataLoader(dataset, num_workers=0, ...)  # SIEMPRE 0 en Windows
```

### weights_only=True error al cargar checkpoints
```python
# Error: "Weights only load failed"
# Causa: el checkpoint contiene dicts Python arbitrarios (cfg, etc.)
# Solución:
torch.load(path, map_location=device, weights_only=False)  # siempre en este proyecto
```

---

## 12. COMANDOS DE EJECUCIÓN FRECUENTES

```powershell
# Ejecutar script Python (SIEMPRE con cmd /c en Windows)
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe <script.py> 2>&1"

# Ejecutar el experimento MaRNet completo (background, ~1-2h)
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe c:\repos\DroneDetectionRF\NoisyUAV\run_marnet_experiment.py 2>&1"

# Pre-flight check rápido
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe c:\repos\DroneDetectionRF\_preflight.py 2>&1"

# Ver GPU disponible
cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')\" 2>&1"

# Instalar mamba-ssm (requiere CUDA toolkit y compilador C++)
# cmd /c "C:\Users\Manuel\anaconda3\envs\IAIAVv3\pip.exe install mamba-ssm causal-conv1d"
```

---

## 13. NOTAS SOBRE EL INFORME ACADÉMICO

El informe `informe_marnet_fusion.md` está diseñado para ser incluido en el TFM. Al regenerarlo:

1. **Siempre importar `_MAMBA_AVAILABLE`** en el script que llame a `generate_report()`
2. El informe incluye referencias bibliográficas formateadas
3. Las figuras se referencian con rutas relativas `figures/fig_XX_...png` — no mover las figuras
4. El campo "Mejora absoluta" puede ser negativo si el modelo no supera el baseline — esto es honesto y debe reportarse así
5. El campo AUC-ROC sale a 0.0 si `SKLEARN_OK=False` o si `probs=[]` — guardar probs en futuros experimentos

---

## 14. PIPELINE DE EXPERIMENTO COMPLETO (RESUMEN RÁPIDO)

```
1. Pre-flight check         → _preflight.py
2. Configurar CFG dict      → run_marnet_experiment.py:CFG
3. Lanzar en background     → cmd /c python run_marnet_experiment.py
4. Monitorear con           → command_status(ID, WaitDurationSeconds=300)
5. Si falla generación      → generate_report_fix.py con datos del JSON
6. Actualizar memory.md     → añadir métricas del nuevo experimento
7. Analizar resultados      → metricas_completas.json + figuras/
8. Diagnosi si Acc < base   → ver sección 11 de este fichero
```
