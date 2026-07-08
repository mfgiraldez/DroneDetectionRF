import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis('off')

def draw_block(x, y, w, h, text, facecolor, edgecolor):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, 
            fontweight='bold', color='#333333', family='sans-serif')

def draw_arrow(x, y, dx, dy):
    ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.2, fc='k', ec='k', 
             linewidth=1.5, length_includes_head=True)

# Blocks
# 1. Antena / Captura
draw_block(0.5, 1, 1.8, 1, "Captura RF\n(Señal I/Q)", "#E3F2FD", "#1976D2")
draw_arrow(2.3, 1.5, 0.7, 0)

# 2. Detección Entropía
draw_block(3.0, 1, 2.2, 1, "Detección Frontal\n(Filtro de Entropía)", "#E8F5E9", "#388E3C")
draw_arrow(5.2, 1.5, 0.7, 0)

# 3. Red Neuronal
draw_block(5.9, 1, 2.4, 1, "Red Neuronal Compleja\n(Arquitectura DualStream)", "#FFEBEE", "#D32F2F")
draw_arrow(8.3, 1.5, 0.7, 0)

# 4. Decisión
draw_block(9.0, 1, 1.8, 1, "Clasificador\n(Dron / Ruido)", "#FFF3E0", "#F57C00")

# Titles / Backgrounds
# Procesamiento
rect_proc = patches.Rectangle((2.8, 0.7), 2.6, 1.7, linewidth=1.5, edgecolor='#388E3C', 
                              facecolor='none', linestyle='--')
ax.add_patch(rect_proc)
ax.text(4.1, 2.5, "Procesamiento de Señal", ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#388E3C', family='sans-serif')

# Inferencia
rect_inf = patches.Rectangle((5.7, 0.7), 2.8, 1.7, linewidth=1.5, edgecolor='#D32F2F', 
                             facecolor='none', linestyle='--')
ax.add_patch(rect_inf)
ax.text(7.1, 2.5, "Inferencia Deep Learning", ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#D32F2F', family='sans-serif')

plt.tight_layout()
plt.savefig('esquema_general.pdf', format='pdf', bbox_inches='tight')
