"""
plot_arch_v2.py — Diagrama tipo publicación para Dual-Stream CVCNN V2.1
Estilo: fondo oscuro, bloques CNN 3D, señal IQ, PSD, nodos MLP.
Ejecutar: python plot_arch_v2.py
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# ── Paleta ────────────────────────────────────────────────────────────────────
BG    = "#12151E"
S1    = "#1565C0"   # Stream IQ
S2    = "#00838F"   # Stream PSD
ATT   = "#7B1FA2"   # Atención
PHY   = "#E65100"   # Físicas
CLS   = "#1B5E20"   # Clasificador
OUT   = "#B71C1C"   # Salida
LIGHT = "#E0E0E0"
DIM   = "#78909C"

np.random.seed(42)
fig = plt.figure(figsize=(22, 11), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0,22); ax.set_ylim(0,11); ax.axis('off')

# ── Helpers ───────────────────────────────────────────────────────────────────
def label(x,y,t,fs=9,c=LIGHT,fw='bold',ha='center',va='center',**kw):
    ax.text(x,y,t,fontsize=fs,color=c,fontweight=fw,ha=ha,va=va,**kw)

def arr(x0,y0,x1,y1,col=LIGHT,lw=1.6):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,mutation_scale=12),zorder=10)

def rect(x,y,w,h,col,alpha=0.85,lw=1,ec="white",r=0.12):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.05,rounding_size={r}",
                     fc=col,ec=ec,lw=lw,alpha=alpha,zorder=4)
    ax.add_patch(p); return p

def section_bg(x,y,w,h,col,title):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
                     fc=col,ec=col,lw=1.5,alpha=0.12,zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2,y+h+0.12,title,ha='center',va='bottom',fontsize=9,
            color=col,fontweight='bold',alpha=0.9)

# ── Bloque CNN 3D ─────────────────────────────────────────────────────────────
def cnn3d(ax, x, y, w, h, d, col, label_t, sub="", n_lines=5):
    """Bloque CNN con efecto 3D (cara frontal + techo + lateral)."""
    dx, dy = d*0.55, d*0.35
    # Lateral derecho
    xs=[x+w, x+w+dx, x+w+dx, x+w, x+w]
    ys=[y,   y+dy,   y+h+dy, y+h, y]
    ax.fill(xs,ys,color=col,alpha=0.45,zorder=3)
    ax.plot(xs,ys,color=col,lw=0.8,alpha=0.7,zorder=3)
    # Techo
    xt=[x, x+dx, x+w+dx, x+w, x]
    yt=[y+h,y+h+dy,y+h+dy,y+h,y+h]
    ax.fill(xt,yt,color=col,alpha=0.6,zorder=3)
    ax.plot(xt,yt,color=col,lw=0.8,alpha=0.7,zorder=3)
    # Cara frontal + líneas de filtros
    rect(x,y,w,h,col,alpha=0.85,ec="white",lw=1.0)
    for i in range(1,n_lines):
        yy=y+i*(h/n_lines)
        ax.plot([x,x+w],[yy,yy],color="white",lw=0.4,alpha=0.25,zorder=5)
    ax.text(x+w/2,y+h/2+0.06,label_t,ha='center',va='center',
            fontsize=8,fontweight='bold',color='white',zorder=6,linespacing=1.3)
    if sub:
        ax.text(x+w/2,y+h/2-0.28,sub,ha='center',va='center',
                fontsize=6.5,color='white',alpha=0.78,zorder=6,style='italic')

# ── Nodos MLP ─────────────────────────────────────────────────────────────────
def mlp_nodes(ax, x, layers, y_center, col, r=0.15, spacing_x=0.55):
    prev_pos = None
    for i,n in enumerate(layers):
        xs = x + i*spacing_x
        show = min(n,6)
        ys = np.linspace(y_center - (show-1)*0.32/1, y_center + (show-1)*0.32/1, show)
        cur_pos=[]
        for yy in ys:
            c=plt.Circle((xs,yy),r,color=col,ec='white',lw=0.8,alpha=0.85,zorder=7)
            ax.add_patch(c); cur_pos.append((xs,yy))
        if prev_pos:
            for p1 in prev_pos[:4]:
                for p2 in cur_pos[:4]:
                    ax.plot([p1[0]+r,p2[0]-r],[p1[1],p2[1]],
                            color=col,lw=0.4,alpha=0.22,zorder=6)
        if n>6:
            ax.text(xs,y_center-1.05,'...',ha='center',color=col,fontsize=10,zorder=7)
        prev_pos=cur_pos
    return x+(len(layers)-1)*spacing_x

# ═══════════════════════════════════════════════════════════════════════════════
# TÍTULO
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(11,10.55,"Dual-Stream CVCNN — Architecture V2.1",ha='center',va='center',
        fontsize=16,fontweight='bold',color='white')
ax.text(11,10.12,"Multi-Domain Feature Extraction · Soft Attention Fusion · Physical Features · BCE Instance Training",
        ha='center',va='center',fontsize=8.5,color=DIM)
ax.axhline(9.85,xmin=0.02,xmax=0.98,color='#263238',lw=1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — INPUT DATA
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(0.2, 0.5, 2.4, 9.2, LIGHT, "INPUT DATA")

# IQ waveform (inset)
ax_iq = ax.inset_axes([0.022, 0.64, 0.085, 0.25])
t=np.linspace(0,1,500)
sig=np.exp(-((t-0.35)**2)/0.01)*np.cos(60*np.pi*t+0.3)
sig+=np.exp(-((t-0.65)**2)/0.008)*np.cos(60*np.pi*t+1.1)
ax_iq.plot(t,sig+np.random.randn(500)*0.08,'#42A5F5',lw=0.8,label='I')
ax_iq.plot(t,np.roll(sig,25)+np.random.randn(500)*0.08,'#EF9A9A',lw=0.8,label='Q')
ax_iq.set_facecolor('#1A2035'); ax_iq.tick_params(labelbottom=False,labelleft=False,length=0)
for sp in ax_iq.spines.values(): sp.set_color('#37474F')
ax_iq.set_title('IQ Signal (9.4 ms)',fontsize=6,color=LIGHT,pad=2)
ax_iq.legend(fontsize=5,framealpha=0,loc='upper right',labelcolor='white')

label(1.42,7.55,"[B, 2, 131 072]",fs=7,c=DIM,fw='normal')
label(1.42,7.20,"Raw IQ · 9.4 ms @ 14 MHz",fs=7,c=LIGHT,fw='normal')

# PSD (inset)
ax_psd = ax.inset_axes([0.022, 0.35, 0.085, 0.22])
f=np.linspace(-7,7,512)
psd=0.02+0.005*np.random.rand(512)
for fc in [-3.1,0.4,2.8]: psd+=0.25*np.exp(-((f-fc)**2)/0.08)
ax_psd.fill_between(f,10*np.log10(psd+1e-4),alpha=0.7,color='#00BCD4')
ax_psd.plot(f,10*np.log10(psd+1e-4),'#00E5FF',lw=0.8)
ax_psd.set_facecolor('#1A2035'); ax_psd.tick_params(labelbottom=False,labelleft=False,length=0)
for sp in ax_psd.spines.values(): sp.set_color('#37474F')
ax_psd.set_title('Log-PSD [2048 bins]',fontsize=6,color=LIGHT,pad=2)

# Physical features box
rect(0.3,0.65,2.0,1.15,PHY,alpha=0.25,ec=PHY,r=0.1)
label(1.3,1.6,"Physical Features",fs=8,c=PHY)
for i,(nm,val) in enumerate([("noise floor","nf"),("H_mean","H̄"),("z_peak","z★")]):
    bx=0.35+i*0.67
    rect(bx,0.72,0.55,0.46,PHY,alpha=0.5,ec=PHY,r=0.06)
    label(bx+0.275,1.06,val,fs=9,c='white'); label(bx+0.275,0.83,nm,fs=5.5,c=LIGHT,fw='normal')
label(1.3,0.58,"[B, 3]",fs=7,c=DIM,fw='normal')

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — STREAM 1 (IQ / Temporal)
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(2.9, 5.5, 6.8, 4.2, S1, "STREAM 1 — Temporal Domain (Raw IQ)")

arr(2.6,8.0,2.9,8.0,col=S1)
label(2.75,8.18,"[B,2,\n131K]",fs=6,c=S1,fw='normal')

configs=[
    ("Conv1d\n2→32\nk=128 s=4","BN·ReLU\nMaxPool(4)",1.05,1.4),
    ("Conv1d\n32→64\nk=31 s=2","BN·ReLU\nMaxPool(4)",0.95,1.3),
    ("Conv1d\n64→128\nk=7","BN·ReLU\nMaxPool(4)",0.85,1.2),
    ("Conv1d\n128→256\nk=3","BN·ReLU\nGAP",0.75,1.1),
]
x1=3.1; shapes=["[B,32,4K]","[B,64,512]","[B,128,32]","[B,256,1]"]
for i,(lbl,sub,w,h) in enumerate(configs):
    cnn3d(ax,x1,6.8,w,h,0.35,S1,lbl,sub)
    label(x1+w/2,6.62,shapes[i],fs=6,c=DIM,fw='normal')
    if i<3:
        nx=x1+w+0.35+0.12
        arr(x1+w+0.02,7.45,nx-0.02,7.45,col=S1,lw=1.3)
    x1+=w+0.35+0.12

# Linear projection
rect(8.55,7.0,1.1,0.9,S1,alpha=0.7,ec='white',r=0.1)
label(9.1,7.62,"Linear",fs=8.5,fw='normal')
label(9.1,7.35,"256→128",fs=7.5); label(9.1,7.18,"[B,128]",fs=6.5,c=DIM,fw='normal')
arr(x1-0.13,7.45,8.55,7.45,col=S1,lw=1.3)

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — FFT block + STREAM 2 (PSD / Frecuencial)
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(2.9, 0.5, 6.8, 4.7, S2, "STREAM 2 — Frequency Domain (Log-PSD via GPU FFT)")

# FFT block
arr(2.6,3.5,2.9,3.5,col=S2)
rect(2.9,2.9,1.5,1.2,S2,alpha=0.65,ec='white',r=0.1)
label(3.65,3.68,"GPU FFT",fs=8.5,fw='normal'); label(3.65,3.45,"Hann · fftshift",fs=7)
label(3.65,3.2,"AvgPool→2048",fs=6.5,c=DIM,fw='normal'); label(3.65,3.0,"[B,1,2048]",fs=6,c=DIM,fw='normal')
arr(2.6,8.0,3.65,8.0,col="#37474F",lw=1.0)  # from IQ input (fork)
arr(3.65,2.9,3.65,2.9-0.01,col=S2,lw=0)    # invisible placeholder

# PSD CNN blocks
arr(4.4,3.5,4.65,3.5,col=S2)
psd_configs=[
    ("Conv1d\n1→32\nk=15 s=2","BN·ReLU\nMaxPool(2)",1.0,1.2),
    ("Conv1d\n32→64\nk=7 s=2","BN·ReLU\nMaxPool(2)",0.9,1.1),
    ("Conv1d\n64→128\nk=3","BN·ReLU\nGAP",0.8,1.0),
]
x2=4.65; psd_shapes=["[B,32,512]","[B,64,128]","[B,128,1]"]
for i,(lbl,sub,w,h) in enumerate(psd_configs):
    cnn3d(ax,x2,2.85,w,h,0.3,S2,lbl,sub)
    label(x2+w/2,2.68,psd_shapes[i],fs=6,c=DIM,fw='normal')
    if i<2:
        nx=x2+w+0.3+0.1
        arr(x2+w+0.01,3.42,nx-0.01,3.42,col=S2,lw=1.3)
    x2+=w+0.3+0.1

label(x2+0.1,3.42,"[B,128]",fs=7,c=DIM,fw='normal',ha='left')

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ATTENTION FUSION
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(10.0, 3.5, 3.6, 5.6, ATT, "SOFT ATTENTION FUSION")

# embeddings → fusion arrows
arr(9.65,7.45,10.25,7.0,col=S1); label(9.9,7.38,"e_IQ  [B,128]",fs=6.5,c=S1,ha='left',fw='normal')
arr(x2+0.55,3.42,10.25,5.8,col=S2); label(9.9,4.6,"e_PSD [B,128]",fs=6.5,c=S2,ha='left',fw='normal')

# Attention MLP nodes
xe=mlp_nodes(ax,10.3,[6,4,2],6.35,ATT,r=0.13,spacing_x=0.6)
label(10.3,9.72,"concat\n[B,256]",fs=7,c=ATT,fw='normal')
label(10.9,9.72,"→[B,128]",fs=7,c=ATT,fw='normal')
label(11.5,9.72,"Softmax\n[B,2]",fs=7,c=ATT,fw='normal')

# alpha bars
rect(11.85,7.0,0.95,0.45,S1,alpha=0.8,ec='white',r=0.06)
rect(11.85,6.35,0.95,0.45,S2,alpha=0.8,ec='white',r=0.06)
label(12.32,7.225,"α_IQ",fs=8,c='white'); label(12.32,6.575,"α_PSD",fs=8,c='white')

rect(12.95,5.8,1.3,2.5,ATT,alpha=0.35,ec=ATT,r=0.12)
label(13.6,7.35,"Weighted",fs=8,c=ATT,fw='normal'); label(13.6,7.05,"Sum",fs=8,c=ATT,fw='normal')
label(13.6,6.7,"f = α_IQ·e_IQ",fs=6.5,c=LIGHT,fw='normal',style='italic')
label(13.6,6.45,"  + α_PSD·e_PSD",fs=6.5,c=LIGHT,fw='normal',style='italic')
label(13.6,6.1,"[B, 128]",fs=7,c=DIM,fw='normal')
arr(12.82,7.0,12.95,7.0,col=ATT)

# Physical MLP
rect(10.2,1.0,3.2,1.8,PHY,alpha=0.25,ec=PHY,r=0.12)
label(11.8,2.5,"Physical MLP",fs=8,c=PHY)
xe2=mlp_nodes(ax,10.4,[3,8,4],1.9,PHY,r=0.13,spacing_x=0.7)
label(10.4,0.78,"[B,3]",fs=6.5,c=PHY,fw='normal')
label(11.1,0.78,"Linear(3→16)",fs=6.5,c=PHY,fw='normal')
label(11.8,0.78,"ReLU→[B,16]",fs=6.5,c=PHY,fw='normal')
arr(1.3,0.65+0.575,10.2,1.85,col=PHY,lw=1.2)

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(14.1, 3.5, 4.3, 5.6, CLS, "CLASSIFIER")

# concat node
circ=plt.Circle((14.3,6.3),0.22,color='#263238',ec=LIGHT,lw=1.2,zorder=8)
ax.add_patch(circ); label(14.3,6.3,"cat",fs=7.5,c=LIGHT)
label(14.3,5.9,"[B,144]",fs=6.5,c=DIM,fw='normal')
arr(14.25,5.8,14.25,5.6,col=CLS,lw=1.0)

arr(13.6+0.65,6.3,14.08,6.3,col=ATT)  # fused → cat
arr(xe2+0.13,1.9,14.3,6.08,col=PHY,lw=1.1)  # phys → cat

# Classifier MLP nodes
xc=mlp_nodes(ax,14.55,[6,4,1],6.3,CLS,r=0.14,spacing_x=0.72)
label(14.55,9.72,"[B,144]",fs=7,c=CLS,fw='normal')
label(15.27,9.72,"Linear\n144→64\nReLU·Drop",fs=6.5,c=CLS,fw='normal')
label(15.99,9.72,"Linear\n64→1\nlogit",fs=6.5,c=CLS,fw='normal')

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(18.6, 1.5, 3.1, 7.6, OUT, "OUTPUT & DECISION")

arr(xc+0.14,6.3,18.6,6.3,col=OUT)

# Sigmoid + prob bar
rect(18.7,5.7,2.8,1.4,OUT,alpha=0.3,ec=OUT,r=0.15)
label(20.1,6.7,"Sigmoid",fs=9,c=LIGHT)
label(20.1,6.35,"P(drone) ∈ (0, 1)",fs=8,c=LIGHT,fw='normal')

ax_bar=ax.inset_axes([0.869,0.485,0.09,0.06])
probs=np.array([0.03,0.12,0.82,0.45,0.91,0.07])
cols=['#EF9A9A' if p<0.75 else '#A5D6A7' for p in probs]
ax_bar.bar(range(len(probs)),probs,color=cols,ec='white',lw=0.5)
ax_bar.axhline(0.75,color='white',lw=1,ls='--')
ax_bar.set_facecolor('#1A2035'); ax_bar.tick_params(labelbottom=False,labelleft=False,length=0)
for sp in ax_bar.spines.values(): sp.set_color('#37474F')
ax_bar.set_title('P(drone) per window',fontsize=5.5,color=LIGHT,pad=1.5)

# Threshold block
rect(18.7,3.8,2.8,1.3,OUT,alpha=0.35,ec=OUT,r=0.15)
label(20.1,4.65,"Temporal Filter V2",fs=8.5,c=LIGHT)
label(20.1,4.3,"N ≥ 2 consec. windows",fs=7.5,c=LIGHT,fw='normal')
label(20.1,4.0,"Threshold τ = 0.75",fs=7.5,c=LIGHT,fw='normal')
arr(20.1,5.7,20.1,5.1,col=OUT)

# Decision
rect(18.7,2.1,2.8,1.25,OUT,alpha=0.60,ec='white',r=0.15)
label(20.1,2.85,"✈  DRONE",fs=11,c='#A5D6A7')
label(20.1,2.45,"No Drone",fs=8.5,c='#EF9A9A',fw='normal')
arr(20.1,3.8,20.1,3.35,col=OUT)

# Metrics badge
rect(18.75,0.55,2.7,1.3,CLS,alpha=0.4,ec=CLS,r=0.12)
label(20.1,1.6,"GOLDEN SET RESULTS",fs=7,c=LIGHT)
for i,(k,v) in enumerate([("F1","0.9005"),("Recall","84.6%"),("Precision","96.3%"),("Spec.","99.2%")]):
    label(18.95+i*0.68,1.2,k,fs=6,c=DIM,fw='normal')
    label(18.95+i*0.68,0.88,v,fs=7,c='white')

# ── Linea IQ→FFT fork ─────────────────────────────────────────────────────────
ax.plot([2.6,2.6],[3.5,8.0],color='#37474F',lw=1.2,ls='--',zorder=3)
ax.plot([2.6,2.9],[3.5,3.5],color=S2,lw=1.6,zorder=3)
ax.annotate("",xy=(2.9,3.5),xytext=(2.6,3.5),
            arrowprops=dict(arrowstyle="-|>",color=S2,lw=1.4,mutation_scale=10),zorder=10)

# ── Leyenda ───────────────────────────────────────────────────────────────────
items=[
    mpatches.Patch(fc=S1,   label="Stream 1 — Temporal (IQ)"),
    mpatches.Patch(fc=S2,   label="Stream 2 — Frequency (PSD)"),
    mpatches.Patch(fc=ATT,  label="Soft Attention Fusion"),
    mpatches.Patch(fc=PHY,  label="Physical Features MLP"),
    mpatches.Patch(fc=CLS,  label="Final Classifier"),
    mpatches.Patch(fc=OUT,  label="Output & Decision"),
]
ax.legend(handles=items,loc='lower center',ncol=6,fontsize=7.5,
          framealpha=0.15,edgecolor='#37474F',labelcolor='white',
          bbox_to_anchor=(0.5,-0.01),
          title="Component Legend",title_fontsize=8)

OUT_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"architecture_dualstream_v2.png")
plt.savefig(OUT_PATH,dpi=180,bbox_inches='tight',facecolor=BG)
print(f"[OK] {OUT_PATH}")
