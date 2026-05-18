# %% [markdown]
# # Inspección de Fallbacks Sospechosos — Drone SNR >= 0 sin bursts detectados
#
# Objetivo: entender por qué el detector NO encontró ningún burst en ficheros
# de drone a SNR alto (donde debería ser trivial detectarlos).
#
# Hipótesis a verificar:
#   H1 — El filtro MAX_BINS_FRAC=0.25 rechaza los hops por "ancho de banda excesivo" (parecen WiFi)
#   H2 — MIN_Z_ABS=3.5 descarta hops legítimos con z_peak bajo
#   H3 — MIN_BURST_MS=0.5 filtra hops muy cortos de ciertos targets

# %% [markdown]
# ## Setup

# %%
import sys, os
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import torch

from NoisyUAV.funciones.dsp_rf.detector_entropia import (
    detectar_bursts, plot_muestra, print_diagnostico, plot_espectrograma_3d
)

# Parámetros usados en build_alumn_dataset_v2.py (los mismos, para reproducir)
FS              = 14e6
NPERSEG         = 2048
MIN_BURST_MS    = 0.5
MERGE_GAP_MS    = 1.5
MIN_Z_ABS       = 3.5
BG_MULT         = 4
MAX_BINS_FRAC   = 0.25
SMOOTH_MS       = 0.2
ADAPTIVE_WINDOW_MS = 15
Z_THRESH_PROD   = 4.0    # el que usó el script (SNR >= 0)

DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CSV_PATH = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\alumn_dataset_pseudo_v2.csv"

TARGET_NAMES = {0:"T0", 1:"T1", 2:"T2", 3:"T3", 4:"Ruido", 5:"T5", 6:"T6"}


# %% [markdown]
# ## Cargar fallbacks sospechosos (drone, SNR >= 0, sin bursts)

# %%
df = pd.read_csv(CSV_PATH)

# Fallbacks de drone a SNR >= 0
mask = (df["fallback"] == True) & (df["label"] == 1) & (df["snr"] >= 0)
df_suspects = df[mask][["file_path", "target_multiclass", "snr", "split"]].drop_duplicates("file_path")
df_suspects = df_suspects.sort_values(["snr", "target_multiclass"]).reset_index(drop=True)

print(f"Total ficheros sospechosos: {len(df_suspects)}")
print()
print(df_suspects.groupby(["snr", "target_multiclass"]).size().unstack(fill_value=0).to_string())


# %% [markdown]
# ## Configuración del cursor — cambia el índice para navegar de uno en uno

# %%
# ─────────────────────────────────────────────────
IDX = 0    # <-- cambia este número para ver otro fichero (0 a len-1)
# ─────────────────────────────────────────────────

row = df_suspects.iloc[IDX]
filepath = os.path.join(DATA_DIR, row["file_path"])

d  = torch.load(filepath, map_location="cpu", weights_only=False)
iq = d["x_iq"].float()

print(f"[{IDX+1}/{len(df_suspects)}]  {row['file_path']}")
print(f"  Target : {row['target_multiclass']} ({TARGET_NAMES.get(row['target_multiclass'], '?')})")
print(f"  SNR    : {row['snr']} dB")
print(f"  Split  : {row['split']}")


# %% [markdown]
# ## Detector con los parámetros de PRODUCCION (los que usó el script)
# Si no encuentra nada aquí, se confirma el motivo del fallback.

# %%
print("=" * 60)
print(f"DETECTOR — Parametros de produccion (Z_THRESH={Z_THRESH_PROD})")
print("=" * 60)

t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
    iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH_PROD,
    min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
    bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
    smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

print_diagnostico(t_ms, nf_v, ns, umbral_v, n_active, bursts)

fig1 = plot_muestra(
    iq, t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts,
    fs=FS,
    titulo=f"PRODUCCION Z={Z_THRESH_PROD} | {row['file_path']} | SNR={row['snr']}dB"
)
fig1.show()


# %% [markdown]
# ## Detector PERMISIVO — reducimos umbrales para ver qué hay realmente

# %%
# Parámetros relajados para diagnóstico
Z_THRESH_DIAG    = 1.5
MIN_Z_ABS_DIAG   = 2.0
MAX_BINS_FRAC_DIAG = 0.40   # subimos para ver si MAX_BINS era el culpable
MIN_BURST_MS_DIAG = 0.2

print("=" * 60)
print(f"DETECTOR PERMISIVO (Z={Z_THRESH_DIAG}, max_bins={MAX_BINS_FRAC_DIAG}, min_burst={MIN_BURST_MS_DIAG}ms)")
print("=" * 60)

t2, H2, H2s, u2, nf2, ns2, na2, bursts2 = detectar_bursts(
    iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH_DIAG,
    min_burst_ms=MIN_BURST_MS_DIAG, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS_DIAG,
    bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC_DIAG,
    smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

print_diagnostico(t2, nf2, ns2, u2, na2, bursts2)

fig2 = plot_muestra(
    iq, t2, H2, H2s, u2, nf2, ns2, na2, bursts2,
    fs=FS,
    titulo=f"PERMISIVO Z={Z_THRESH_DIAG} max_bins={MAX_BINS_FRAC_DIAG} | SNR={row['snr']}dB"
)
fig2.show()


# %% [markdown]
# ## Espectrograma 3D — para ver la estructura frecuencial del burst
# (confirma si el drone tiene muchos bins activos → MAX_BINS_FRAC lo filtraba)

# %%
fig3 = plot_espectrograma_3d(iq, fs=FS, nperseg=NPERSEG,
                              titulo=f"Espectrograma 3D | {row['file_path']}")
fig3.show()


# %% [markdown]
# ## Resumen comparativo — tabla de bursts detectados por cada configuracion

# %%
print(f"\nResumen comparativo para: {row['file_path']}")
print(f"{'Config':<35} {'N bursts':>8}  {'z_peak':>8}  {'n_act':>8}  {'dur_ms':>8}")
print("-" * 75)

configs = [
    ("Produccion (Z=4.0, bins<=25%)", bursts),
    ("Permisivo  (Z=1.5, bins<=40%)", bursts2),
]
for name, blist in configs:
    if blist:
        zp  = max(abs(b['z_peak']) for b in blist)
        na  = max(b['n_act'] for b in blist)
        dur = max(b['dur_ms'] for b in blist)
        print(f"  {name:<33} {len(blist):>8}  {zp:>8.2f}  {na:>8.0f}  {dur:>8.2f}ms")
    else:
        print(f"  {name:<33} {0:>8}  {'—':>8}  {'—':>8}  {'—':>8}")

print()
print("Umbral MAX_BINS_FRAC en produccion:", int(NPERSEG/2 * MAX_BINS_FRAC), "bins")
print("Umbral MAX_BINS_FRAC permisivo    :", int(NPERSEG/2 * MAX_BINS_FRAC_DIAG), "bins")


# %% [markdown]
# ## Vista global — todos los ficheros sospechosos de un vistazo

# %%
# Ejecuta el detector permisivo en TODOS los sospechosos y clasifica el motivo del fallo
print("Analizando todos los ficheros sospechosos (puede tardar ~1-2 min)...")
print()

resultados = []
for _, r in df_suspects.iterrows():
    fp = os.path.join(DATA_DIR, r["file_path"])
    try:
        iq_r = torch.load(fp, map_location="cpu", weights_only=False)["x_iq"].float()
    except Exception:
        resultados.append({"file": r["file_path"], "snr": r["snr"], "target": r["target_multiclass"],
                           "bursts_prod": 0, "bursts_perm": 0, "motivo": "ERROR_CARGA"})
        continue

    # Produccion
    _, _, _, _, _, _, _, b_prod = detectar_bursts(
        iq_r, fs=FS, nperseg=NPERSEG, z_thresh=4.0,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    # Permisivo solo en bins
    _, _, _, _, _, _, _, b_bins = detectar_bursts(
        iq_r, fs=FS, nperseg=NPERSEG, z_thresh=4.0,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
        bg_mult=BG_MULT, max_bins_frac=0.40,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    # Permisivo solo en z_abs
    _, _, _, _, _, _, _, b_zabs = detectar_bursts(
        iq_r, fs=FS, nperseg=NPERSEG, z_thresh=4.0,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=1.5,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    # Permisivo solo en z_thresh
    _, _, _, _, _, _, _, b_zt = detectar_bursts(
        iq_r, fs=FS, nperseg=NPERSEG, z_thresh=2.0,
        min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS, min_z_abs=MIN_Z_ABS,
        bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
        smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

    if   len(b_bins) > 0: motivo = "MAX_BINS_FRAC demasiado bajo"
    elif len(b_zabs) > 0: motivo = "MIN_Z_ABS demasiado alto"
    elif len(b_zt)   > 0: motivo = "Z_THRESH demasiado alto"
    else:                  motivo = "sin senal incluso permisivo"

    resultados.append({
        "file": r["file_path"], "snr": r["snr"], "target": r["target_multiclass"],
        "bursts_prod": len(b_prod), "bursts_bins": len(b_bins),
        "bursts_zabs": len(b_zabs), "bursts_zt": len(b_zt), "motivo": motivo
    })

df_res = pd.DataFrame(resultados)
print(df_res[["snr", "target", "bursts_prod", "bursts_bins", "bursts_zabs", "bursts_zt", "motivo"]].to_string())
print()
print("Resumen de motivos:")
print(df_res["motivo"].value_counts().to_string())
