import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np, os
from PIL import Image

BG="#FAFBFC"; WHT="#FFFFFF"; TXT="#1C2833"; DIM="#5D6D7E"; ARR="#2C3E50"

fig,ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16); ax.set_ylim(0, 6.5); ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

def rb(x, y, w, h, fc=WHT, ec=ARR, lw=1.8, r=0.2, z=3):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, edgecolor=ec, facecolor=fc, zorder=z))

def tx(x, y, s, fs=9, bold=False, c=TXT, z=6):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=c, fontweight="bold" if bold else "normal", zorder=z)

def ar(x0, y0, x1, y1, c=ARR, lw=2.0, z=5, arc=0.0):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=16, connectionstyle=f"arc3,rad={arc}"), zorder=z)

# Title
fig.text(0.5, 0.95, "Esquema Arquitectónico General del Sistema C-UAS", ha="center", va="top", fontsize=14, fontweight="bold", color=TXT)
fig.text(0.5, 0.90, "Detección de señales mediante filtrado físico y extracción profunda multidominio (CV-CNN)", ha="center", va="top", fontsize=10, color=DIM)

# 1. Antena / Input
IQX, IQY = 0.5, 1.5
IQW, IQH = 2.4, 3.5
rb(IQX, IQY, IQW, IQH, fc="#EBF2FB", ec="#1558A8")
tx(IQX+IQW/2, IQY+IQH-0.4, "Vigilancia RF", fs=11, bold=True, c="#1558A8")
tx(IQX+IQW/2, IQY+0.4, "Captura en Bruto\n(14 MHz, I/Q)", fs=8, c=DIM)

try:
    img_path = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM\ej_dataset_target4_0db.png"
    # Cargar y recortar la imagen para que quede muy bien
    img = Image.open(img_path)
    w, h = img.size
    img = img.crop((w*0.1, h*0.1, w*0.9, h*0.9)) # Recortar los márgenes de matplotlib si los hubiera
    im = OffsetImage(img, zoom=0.08, alpha=0.9)
    ab = AnnotationBbox(im, (IQX+IQW/2, IQY+IQH/2), frameon=True, bboxprops=dict(edgecolor="#1558A8", facecolor="white", lw=1.0))
    ax.add_artist(ab)
except Exception as e:
    tx(IQX+IQW/2, IQY+IQH/2, "[Imagen de RF]", fs=9, c=DIM)

ar(IQX+IQW, IQY+IQH/2, 3.4, IQY+IQH/2, c="#1558A8")

# 2. DualStream CV-CNN Box
ax.add_patch(FancyBboxPatch((3.4, 1.0), 6.5, 4.5, boxstyle="round,pad=0,rounding_size=0.2", lw=1.5, edgecolor="#196B3A", facecolor="#ECF5EE", linestyle="--", zorder=1))
tx(3.4+6.5/2, 5.2, "Núcleo de Extracción: Red DualStream (CV-CNN + Físico)", fs=11, bold=True, c="#196B3A")

# Top branch (Temporal IQ)
T_Y = 3.6
rb(3.8, T_Y, 2.2, 1.4, fc="#E0F5F2", ec="#0C5A52")
tx(4.9, T_Y+1.0, "Rama Temporal", fs=10, bold=True, c="#0C5A52")
tx(4.9, T_Y+0.6, "ComplexConv1D", fs=9, bold=True, c=TXT)
tx(4.9, T_Y+0.3, "Extracción de\nfase transitoria I/Q", fs=8, c=DIM)

# Bottom branch (Frecuencial)
F_Y = 1.5
rb(3.8, F_Y, 2.2, 1.4, fc="#FEF3C7", ec="#92400E")
tx(4.9, F_Y+1.0, "Rama Frecuencial", fs=10, bold=True, c="#92400E")
tx(4.9, F_Y+0.6, "Dense (MLP)", fs=9, bold=True, c=TXT)
tx(4.9, F_Y+0.3, "Compresión de\nentropía y PSD (Welch)", fs=8, c=DIM)

# Flechas internas del DualStream
ar(6.0, T_Y+0.7, 7.0, 3.8, arc=0.2)
ar(6.0, F_Y+0.7, 7.0, 2.7, arc=-0.2)

# Concat
rb(7.0, 2.0, 1.2, 2.5, fc="#FEF0E6", ec="#B44800")
tx(7.6, 3.25, "Fusión", fs=10, bold=True, c="#B44800")
tx(7.6, 2.75, "$\oplus$", fs=24, bold=True, c="#B44800")

# MLP Classifier
ar(9.9, 3.25, 10.5, 3.25)
rb(10.5, 2.3, 2.0, 1.9, fc="#FDF0F0", ec="#8B1A1A")
tx(11.5, 3.7, "Clasificador", fs=11, bold=True, c="#8B1A1A")
tx(11.5, 3.3, "Multilayer\nPerceptron", fs=10, bold=True, c=TXT)
tx(11.5, 2.7, "Decisión no lineal\nsobre latente", fs=8, c=DIM)

# Flecha a outputs
ar(12.5, 3.25, 13.5, 4.0, c="#8B1A1A", arc=0.2)
ar(12.5, 3.25, 13.5, 2.5, c="#8B1A1A", arc=-0.2)

# Outputs
rb(13.5, 3.5, 2.0, 1.1, fc="#D4EDDA", ec="#155724")
tx(14.5, 4.2, "DRON (1)", fs=12, bold=True, c="#155724")
tx(14.5, 3.8, "Señal FHSS Objetivo", fs=8, c=DIM)

rb(13.5, 1.9, 2.0, 1.1, fc="#F8D7DA", ec="#721C24")
tx(14.5, 2.6, "NO DRON (0)", fs=12, bold=True, c="#721C24")
tx(14.5, 2.2, "Ruido térmico / OFDM", fs=8, c=DIM)

plt.tight_layout()
out_dir = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas"
plt.savefig(os.path.join(out_dir, 'esquema_general.pdf'), format='pdf', bbox_inches='tight', facecolor=BG)
plt.savefig(os.path.join(out_dir, 'esquema_general.png'), format='png', bbox_inches='tight', facecolor=BG)
