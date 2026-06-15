"""
plot_architecture.py — Diagrama de Arquitectura del Modelo Dual-Stream CVCNN V2.1
===================================================================================
Genera una figura thesis-ready con el flujo completo del modelo.

Uso:
    conda activate IAIAVv3
    python plot_architecture.py
    -> Guarda 'architecture_dualstream_v2.png' en el mismo directorio.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Paleta ────────────────────────────────────────────────────────────────────
C_IQ     = "#1565C0"
C_PSD    = "#00838F"
C_FUSION = "#6A1B9A"
C_PHYS   = "#E65100"
C_CLS    = "#1B5E20"
C_OUT    = "#B71C1C"
C_POST   = "#880E4F"
C_DATA   = "#37474F"
C_ARROW  = "#424242"


def box(ax, x, y, w, h, text, color, fs=8.5, tc="white",
        sub=None, sub_fs=6.8, alpha=0.90, rad=0.016):
    """Caja redondeada con texto principal y subtexto opcional."""
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0.008,rounding_size={rad}",
                       linewidth=1.0, edgecolor="white",
                       facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(p)
    # sombra
    ps = FancyBboxPatch((x - w/2 + 0.003, y - h/2 - 0.004), w, h,
                        boxstyle=f"round,pad=0.008,rounding_size={rad}",
                        linewidth=0, facecolor="#00000020", zorder=2)
    ax.add_patch(ps)
    yo = 0.013 if sub else 0
    ax.text(x, y + yo, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=4, linespacing=1.35)
    if sub:
        ax.text(x, y - 0.026, sub, ha='center', va='center',
                fontsize=sub_fs, color=tc, alpha=0.82, style='italic', zorder=4)


def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, lbl=None,
        lbl_side='right', lbl_fs=6.5):
    """Flecha simple con etiqueta opcional."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10), zorder=5)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx = 0.025 if lbl_side == 'right' else -0.025
        ha = 'left' if lbl_side == 'right' else 'right'
        ax.text(mx+dx, my, lbl, ha=ha, va='center',
                fontsize=lbl_fs, color=C_DATA, style='italic', zorder=6)


def elbow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.3):
    """Línea en L con flecha al final."""
    ym = (y0 + y1) / 2
    ax.plot([x0, x0, x1, x1], [y0, ym, ym, y1],
            color=color, lw=lw, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x1, ym),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=9), zorder=5)


def section_rect(ax, x0, y0, x1, y1, color, label, top=True):
    p = mpatches.FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                                boxstyle="round,pad=0.005",
                                linewidth=1.3, linestyle='--',
                                edgecolor=color, facecolor='none',
                                alpha=0.40, zorder=1)
    ax.add_patch(p)
    lx = (x0+x1)/2
    ly, va = (y1+0.006, 'bottom') if top else (y0-0.010, 'top')
    ax.text(lx, ly, label, ha='center', va=va,
            fontsize=6.8, color=color, fontweight='bold', alpha=0.75)


# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

# Título
ax.text(0.5, 0.978, "Dual-Stream CVCNN — Architecture V2.1",
        ha='center', va='top', fontsize=14, fontweight='bold', color="#1A237E",
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])
ax.text(0.5, 0.957,
        "Multi-Domain Feature Extraction with Soft Attention Fusion + Physical Features",
        ha='center', va='top', fontsize=8.8, color="#546E7A")
ax.axhline(0.948, xmin=0.04, xmax=0.96, color="#CFD8DC", lw=0.8)

# ─── Coordenadas principales ───────────────────────────────────────────────
X_IN   = 0.08   # columna de entradas
Y_IQ   = 0.77   # eje Stream 1
Y_PSD2 = 0.38   # eje Stream 2
X0     = 0.185  # inicio bloques CNN
BW     = 0.088  # ancho bloque conv
BH     = 0.076  # alto bloque conv
GAP    = 0.108  # paso entre bloques

# ─── ENTRADAS ─────────────────────────────────────────────────────────────────
box(ax, X_IN, Y_IQ, 0.10, 0.065, "Raw IQ Signal", C_DATA,
    sub="[B, 2, 131 072]\n9.4 ms @ 14 MHz")

box(ax, X_IN, 0.26, 0.10, 0.065, "Physical\nFeatures", C_PHYS,
    sub="[B, 3]\nnf · H_mean · z_peak")
ax.text(X_IN, 0.208, "CFAR Entropy\nDetector", ha='center', va='top',
        fontsize=6.8, color=C_PHYS, style='italic')

# ─── FFT interta + AvgPool ────────────────────────────────────────────────────
X_FFT  = 0.215
Y_FFT  = 0.572
X_POOL = 0.335
box(ax, X_FFT, Y_FFT, 0.095, 0.055,
    "FFT + Log-PSD\n(on-device GPU)", C_PSD, fs=8,
    sub="Hann · fftshift · |z|²", alpha=0.82)

elbow(ax, X_IN+0.05, Y_IQ-0.005, X_FFT-0.047, Y_FFT+0.015, color=C_PSD)
ax.text(0.148, 0.66, "[B, 2, 131072]", fontsize=6.3, color=C_PSD,
        style='italic', ha='center')

box(ax, X_POOL, Y_FFT, 0.082, 0.047, "AvgPool1d", C_PSD, fs=8,
    sub="→ [B, 1, 2048]", alpha=0.78)
arr(ax, X_FFT+0.047, Y_FFT, X_POOL-0.041, Y_FFT, color=C_PSD,
    lbl="[B,1,131072]", lbl_fs=6.2)

# ─── STREAM 1 (IQ) ────────────────────────────────────────────────────────────
layers_iq = [
    ("Conv1d\n2→32\nk=128 s=4", "→[B,32,\n4 096]"),
    ("Conv1d\n32→64\nk=31 s=2",  "→[B,64,\n512]"),
    ("Conv1d\n64→128\nk=7",       "→[B,128,\n128]"),
    ("Conv1d\n128→256\nk=3",      "→[B,256,1]\nGAP"),
]
labels_iq = ["BN+ReLU+MaxPool(4)"]*3 + ["BN+ReLU+GAP"]

ax.text(0.525, Y_IQ+0.068,
        "── Stream 1: Temporal Domain (Raw IQ) ──",
        ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=C_IQ)

arr(ax, X_IN+0.05, Y_IQ, X0-BW/2, Y_IQ, color=C_IQ, lbl="[B,2,131072]")
x = X0
for i, (name, shape) in enumerate(layers_iq):
    box(ax, x, Y_IQ, BW, BH, name, C_IQ, fs=7.5, sub=shape, sub_fs=6.3)
    ax.text(x, Y_IQ-BH/2-0.016, labels_iq[i],
            ha='center', va='top', fontsize=5.8, color="#78909C")
    if i < len(layers_iq)-1:
        arr(ax, x+BW/2, Y_IQ, x+GAP-BW/2, Y_IQ, color=C_IQ, lw=1.2)
    x += GAP

# Proyección 256→128
X_PROJ = x - GAP + BW/2 + 0.065
box(ax, X_PROJ, Y_IQ, 0.077, 0.052, "Linear\n256→128", C_IQ, fs=7.5,
    sub="[B, 128]", alpha=0.82)
arr(ax, x-GAP+BW/2, Y_IQ, X_PROJ-0.038, Y_IQ, color=C_IQ, lw=1.2)
X_IQ_OUT = X_PROJ + 0.038

# ─── STREAM 2 (PSD) ───────────────────────────────────────────────────────────
layers_psd = [
    ("Conv1d\n1→32\nk=15 s=2", "→[B,32,512]"),
    ("Conv1d\n32→64\nk=7 s=2",  "→[B,64,128]"),
    ("Conv1d\n64→128\nk=3",     "→[B,128,1]\nGAP"),
]
labels_psd = ["BN+ReLU+MaxPool(2)"]*2 + ["BN+ReLU+GAP"]

ax.text(0.44, Y_PSD2+0.068,
        "── Stream 2: Frequency Domain (Log-PSD) ──",
        ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=C_PSD)

arr(ax, X_POOL+0.041, Y_FFT, X0-BW/2, Y_PSD2, color=C_PSD, lbl="[B,1,2048]")
x2 = X0
for i, (name, shape) in enumerate(layers_psd):
    box(ax, x2, Y_PSD2, BW, BH, name, C_PSD, fs=7.5, sub=shape, sub_fs=6.3)
    ax.text(x2, Y_PSD2-BH/2-0.016, labels_psd[i],
            ha='center', va='top', fontsize=5.8, color="#78909C")
    if i < len(layers_psd)-1:
        arr(ax, x2+BW/2, Y_PSD2, x2+GAP-BW/2, Y_PSD2, color=C_PSD, lw=1.2)
    x2 += GAP
X_PSD_OUT = x2 - GAP + BW/2

# ─── ATTENTION FUSION ─────────────────────────────────────────────────────────
X_ATT = 0.745
Y_ATT = 0.575
box(ax, X_ATT, Y_ATT, 0.115, 0.145, "Soft Attention\nFusion", C_FUSION,
    fs=9.5, alpha=0.94)
ax.text(X_ATT, Y_ATT+0.01,
        "Linear(256→128)\nReLU\nLinear(128→2)\nSoftmax(dim=1)",
        ha='center', va='center', fontsize=6.3, color='#E1BEE7', style='italic')
ax.text(X_ATT, Y_ATT-0.055,
        "f = α_IQ·e_IQ + α_PSD·e_PSD",
        ha='center', va='top', fontsize=7, color='white',
        fontweight='bold', style='italic')

elbow(ax, X_IQ_OUT, Y_IQ, X_ATT-0.057, Y_ATT+0.038, color=C_IQ, lw=1.5)
ax.text(X_IQ_OUT+0.015, Y_IQ-0.025, "[B, 128]", fontsize=6.5, color=C_IQ, style='italic')
ax.text(X_ATT-0.08, Y_ATT+0.040, "α_IQ", fontsize=8.5, color=C_IQ,
        fontweight='bold', ha='center')

elbow(ax, X_PSD_OUT, Y_PSD2, X_ATT-0.057, Y_ATT-0.038, color=C_PSD, lw=1.5)
ax.text(X_PSD_OUT+0.015, Y_PSD2+0.022, "[B, 128]", fontsize=6.5, color=C_PSD, style='italic')
ax.text(X_ATT-0.08, Y_ATT-0.040, "α_PSD", fontsize=8.5, color=C_PSD,
        fontweight='bold', ha='center')

# ─── PHYSICAL MLP ─────────────────────────────────────────────────────────────
X_PM = X_ATT
Y_PM = 0.115
box(ax, X_PM, Y_PM, 0.112, 0.072,
    "Phys. MLP\nLinear(3→16) · ReLU", C_PHYS, fs=8,
    sub="[B, 16]", alpha=0.90)
elbow(ax, X_IN+0.05, 0.26, X_PM-0.056, Y_PM, color=C_PHYS, lw=1.3)
ax.text(0.35, 0.185, "[B, 3]", fontsize=6.5, color=C_PHYS,
        style='italic', ha='center')

# ─── CONCAT + CLASSIFIER ──────────────────────────────────────────────────────
X_CAT = 0.875
Y_CAT = 0.575
# Nodo concat
circ = plt.Circle((X_CAT, Y_CAT), 0.017, color="#ECEFF1",
                  ec="#455A64", lw=1.3, zorder=6)
ax.add_patch(circ)
ax.text(X_CAT, Y_CAT, "cat", ha='center', va='center',
        fontsize=7, fontweight='bold', color="#37474F", zorder=7)
ax.text(X_CAT, Y_CAT-0.033, "[B, 144]", ha='center', va='top',
        fontsize=6.5, color=C_DATA, style='italic')

arr(ax, X_ATT+0.057, Y_ATT, X_CAT-0.017, Y_CAT,
    color=C_FUSION, lw=1.5, lbl="[B,128]")
elbow(ax, X_PM, Y_PM+0.036, X_CAT, Y_CAT-0.017, color=C_PHYS, lw=1.3)
ax.text(X_CAT+0.022, 0.37, "[B,16]", fontsize=6.5, color=C_PHYS,
        style='italic', ha='center')

# Clasificador
X_CLS = 0.945
box(ax, X_CLS, Y_CAT, 0.085, 0.145, "Classifier\nMLP", C_CLS,
    fs=9, alpha=0.93)
ax.text(X_CLS, Y_CAT+0.01,
        "Linear(144→64)\nReLU\nDropout(0.3)\nLinear(64→1)",
        ha='center', va='center', fontsize=6.5, color='#C8E6C9', style='italic')
arr(ax, X_CAT+0.017, Y_CAT, X_CLS-0.042, Y_CAT,
    color=C_CLS, lw=1.5, lbl="[B,144]")

# Salida + umbral
Y_OUT = Y_CAT - 0.195
box(ax, X_CLS, Y_OUT, 0.108, 0.075,
    "Sigmoid\nP(drone) ∈ (0,1)", C_OUT, fs=8.5, sub="[B, 1]", alpha=0.93)
arr(ax, X_CLS, Y_CAT-0.072, X_CLS, Y_OUT+0.037,
    color=C_OUT, lw=1.8, lbl="logit", lbl_side='right')

# Post-process
Y_PP = Y_OUT - 0.155
box(ax, X_CLS, Y_PP, 0.118, 0.077,
    "Temporal Filter V2", C_POST, fs=8.5,
    sub="N≥2 consec. · P>0.85", alpha=0.92)
arr(ax, X_CLS, Y_OUT-0.037, X_CLS, Y_PP+0.038,
    color=C_POST, lw=1.5, lbl="τ=0.75", lbl_side='right')

ax.text(X_CLS, Y_PP-0.058, "DRONE  ✈  /  No Drone",
        ha='center', va='top', fontsize=9.5, fontweight='bold',
        color=C_POST)

# ─── Recuadros de sección ─────────────────────────────────────────────────────
section_rect(ax, 0.14, 0.712, 0.825, 0.854, C_IQ,
             "Stream 1 — Time Domain CNN (1D)")
section_rect(ax, 0.14, 0.318, 0.730, 0.458, C_PSD,
             "Stream 2 — Frequency Domain CNN (1D)")

# ─── Cuadro de parámetros ─────────────────────────────────────────────────────
param_txt = (
    "Model Summary\n"
    "─────────────────────\n"
    "IQ Stream:   ~1.20M\n"
    "PSD Stream:  ~0.15M\n"
    "Attn Fusion: ~33K\n"
    "Phys MLP:    ~64\n"
    "Classifier:  ~9.4K\n"
    "─────────────────────\n"
    "Total: ~1.37M params\n"
    "\n"
    "Input IQ:  [B,2,131072]\n"
    "Input phy: [B,3]\n"
    "Output:    [B,1] ∈(0,1)\n"
    "Threshold: τ = 0.75\n"
    "Optimizer: AdamW lr=3e-4\n"
    "Loss:      BCEWithLogits\n"
    "AMP:       FP16 (CUDA)\n"
    "Epochs:    40"
)
ax.text(0.010, 0.51, param_txt, ha='left', va='top', fontsize=7.0,
        color="#263238", family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1',
                  edgecolor='#90A4AE', alpha=0.92), zorder=8)

# ─── Leyenda ──────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_IQ,     label="Stream IQ (Temporal)"),
    mpatches.Patch(facecolor=C_PSD,    label="Stream PSD (Frecuencial)"),
    mpatches.Patch(facecolor=C_FUSION, label="Attention Fusion"),
    mpatches.Patch(facecolor=C_PHYS,   label="Physical Features MLP"),
    mpatches.Patch(facecolor=C_CLS,    label="Final Classifier"),
    mpatches.Patch(facecolor=C_OUT,    label="Output / Sigmoid"),
    mpatches.Patch(facecolor=C_POST,   label="Temporal Post-Process V2"),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=7.5,
          framealpha=0.92, edgecolor='#90A4AE',
          bbox_to_anchor=(0.0, 0.0), ncol=4,
          title="Component Legend", title_fontsize=8)

# ─── Guardar ──────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "architecture_dualstream_v2.png")
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
print(f"[OK] Figura guardada en: {OUT_PATH}")
