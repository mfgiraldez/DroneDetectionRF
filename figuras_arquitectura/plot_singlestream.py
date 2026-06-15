"""
plot_singlestream.py  -  SingleStream-CVCNN Architecture Figure v3
Fondo blanco, estilo académico, sin solapamiento, layout corregido.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np, os

BG="#FAFBFC"; WHT="#FFFFFF"
IQe="#1558A8"; IQf="#EBF2FB"
CVe="#196B3A"; CVf="#ECF5EE"
S1e="#1D4ED8"; S1f="#DBEAFE"
S2e="#065F46"; S2f="#D1FAE5"
S3e="#92400E"; S3f="#FEF3C7"
PRe="#0C5A52"; PRf="#E0F5F2"
PHe="#5E1589"; PHf="#F5EBF9"
CAe="#B44800"; CAf="#FEF0E6"
MLe="#8B1A1A"; MLf="#FDF0F0"
OPe="#155724"; OPf="#D4EDDA"
ONe="#721C24"; ONf="#F8D7DA"
TXT="#1C2833"; DIM="#5D6D7E"; ARR="#2C3E50"

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"text.color":TXT})

FW,FH=30,11.5
fig,ax=plt.subplots(figsize=(FW,FH))
ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.axis("off")
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

def rb(x,y,w,h,fc=WHT,ec=ARR,lw=1.6,r=0.18,z=3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={r}",lw=lw,edgecolor=ec,facecolor=fc,zorder=z))

def tx(x,y,s,fc=TXT,fs=9,bold=False,ha="center",va="center",z=6,it=False):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,color=fc,
        fontweight="bold" if bold else "normal",style="italic" if it else "normal",zorder=z)

def hl(x0,x1,y,col="#CBD5E1",lw=0.7):
    ax.plot([x0,x1],[y,y],color=col,lw=lw,zorder=4)

def dtag(x,y,s,fs=7.0):
    ax.text(x,y,s,ha="center",va="center",fontsize=fs,color=DIM,style="italic",zorder=7,
        bbox=dict(boxstyle="round,pad=0.12",fc=WHT,ec="#CBD5E1",alpha=0.95,lw=0.5))

def ar(x0,y0,x1,y1,col=ARR,lw=1.4,arc=0.0,ms=11):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
        arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,mutation_scale=ms,
            shrinkA=4,shrinkB=4,connectionstyle=f"arc3,rad={arc:.2f}"),zorder=5)

def sbg(x,y,w,h,ec,title,fs=7.8,yoff=0.08):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle="round,pad=0,rounding_size=0.35",lw=0.9,edgecolor=ec,
        facecolor=ec,linestyle="--",alpha=0.07,zorder=1))
    ax.text(x+w/2,y+h+yoff,title,ha="center",va="bottom",
        fontsize=fs,color=ec,fontweight="bold",alpha=0.85,zorder=2)

# ── Título ──────────────────────────────────────────────────────────────────────
fig.text(0.5,0.990,"Arquitectura del Modelo de Referencia SingleStream-CVCNN",
    ha="center",va="top",fontsize=13,fontweight="bold",color=TXT)
fig.text(0.5,0.968,
    "Clasificador binario de ráfagas I/Q mediante red convolucional de valores complejos (CV-CNN) con fusión de características físicas",
    ha="center",va="top",fontsize=8.5,color=DIM)

# ════════════════════════════════════════════════════════════════════════════════
# GEOMETRÍA GLOBAL
# ════════════════════════════════════════════════════════════════════════════════
# Stream IQ:     bloques conv centrados en y ≈ 8.0
# Stream físico: fila de features centrada en y ≈ 2.5
# Concat abarca: y = 1.6 a 6.9  (toca ambas ramas)

CX0=3.10        # x inicio bloques conv
CW=2.35         # ancho bloque conv
CH=3.80         # alto bloque conv
CGAP=0.32       # gap entre bloques
CY=5.95         # y inferior bloques conv   → tope en 5.95+3.80=9.75

# ════════════════════════════════════════════════════════════════════════════════
# ENTRADAS
# ════════════════════════════════════════════════════════════════════════════════
IQX=0.28; IQY=6.85; IQW=2.35; IQH=2.65
rb(IQX,IQY,IQW,IQH,fc=IQf,ec=IQe,lw=2.0)
t_np=np.linspace(0,1,100)
ax.plot(IQX+0.12+t_np*2.10,IQY+1.9+0.24*np.sin(2*np.pi*5*t_np),color=IQe,lw=0.9,alpha=0.55,zorder=4)
ax.plot(IQX+0.12+t_np*2.10,IQY+1.35+0.24*np.cos(2*np.pi*5*t_np),color="#5B9BD5",lw=0.9,alpha=0.55,zorder=4)
tx(IQX+0.22,IQY+1.92,"I",fc=IQe,fs=7,ha="left"); tx(IQX+0.22,IQY+1.38,"Q",fc="#1F78C1",fs=7,ha="left")
tx(IQX+IQW/2,IQY+IQH-0.30,"Ráfaga I/Q",fc=IQe,fs=10,bold=True)
tx(IQX+IQW/2,IQY+0.55,"fs = 14 MHz  ·  75 ms",fc=DIM,fs=7.2)
tx(IQX+IQW/2,IQY+0.22,"[B, 2, N]",fc=DIM,fs=8,it=True)

PHX=0.28; PHY=1.15; PHW=2.35; PHH=2.65
rb(PHX,PHY,PHW,PHH,fc=PHf,ec=PHe,lw=2.0)
tx(PHX+PHW/2,PHY+PHH-0.30,"Características",fc=PHe,fs=10,bold=True)
tx(PHX+PHW/2,PHY+PHH-0.66,"Físicas",fc=PHe,fs=10,bold=True)
tx(PHX+PHW/2,PHY+1.30,"Extraídas por el",fc=DIM,fs=7.2)
tx(PHX+PHW/2,PHY+0.95,"Detector Entropía+CFAR",fc=DIM,fs=7.2)
tx(PHX+PHW/2,PHY+0.28,"[B, 8]",fc=DIM,fs=8,it=True)

tx(IQX+IQW/2,11.12,"ENTRADAS",fc=IQe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# CV-CNN BACKBONE — 4 bloques
# ════════════════════════════════════════════════════════════════════════════════
sbg(CX0-0.12,CY-0.18,4*(CW+CGAP)-CGAP+0.24,CH+0.28,CVe,
    "Backbone  ·  Red Convolucional de Valores Complejos (CV-CNN)",fs=7.8,yoff=0.06)

bspecs=[
    ("Complex Conv Block 1","1 \u2192 32 ch  ·  k=11, s=2"),
    ("Complex Conv Block 2","32 \u2192 64 ch  ·  k=11, s=2"),
    ("Complex Conv Block 3","64 \u2192 128 ch  ·  k=11, s=2"),
    ("Complex Conv Block 4","128 \u2192 128 ch  ·  k=11, s=1"),
]
subops=[
    ("ComplexConv1D",S1f,S1e,"Re(W)\u00b7I \u2212 Im(W)\u00b7Q\nIm(W)\u00b7I + Re(W)\u00b7Q"),
    ("ComplexBN",    S2f,S2e,"BN ind. Re / Im"),
    ("CReLU",        S3f,S3e,"ReLU(Re) + j\u00b7ReLU(Im)"),
]
SH=0.76; SGAP=0.10
cc=[]
for i,(ttl,params) in enumerate(bspecs):
    bx=CX0+i*(CW+CGAP); cx=bx+CW/2; cy=CY+CH/2
    rb(bx,CY,CW,CH,fc=CVf,ec=CVe,lw=1.8)
    tx(cx,CY+CH-0.28,ttl,fc=CVe,fs=8.5,bold=True)
    tx(cx,CY+CH-0.60,params,fc=DIM,fs=7.2)
    hl(bx+0.12,bx+CW-0.12,CY+CH-0.82,col="#A7C7A9")
    sub_top=CY+CH-0.90
    for j,(sn,sf,se,sd) in enumerate(subops):
        sy=sub_top-(j+1)*SH-j*SGAP
        rb(bx+0.12,sy,CW-0.24,SH,fc=sf,ec=se,lw=1.0,r=0.10,z=4)
        tx(cx,sy+SH*0.68,sn,fc=se,fs=7.8,bold=True)
        tx(cx,sy+SH*0.28,sd,fc=DIM,fs=6.5)
    cc.append((bx+CW,cy))
    if i>0:
        px=CX0+(i-1)*(CW+CGAP)+CW
        ar(px,cy,bx,cy,col=CVe,lw=1.5)

# flecha I/Q → Bloque 1
iq_exit_y=IQY+IQH/2; conv1_iny=CY+CH/2
ar(IQX+IQW,iq_exit_y,CX0,conv1_iny,col=IQe,lw=2.0)
dtag((IQX+IQW+CX0)/2,iq_exit_y+0.30,"[B, 2, N]")

tx(CX0+(4*(CW+CGAP)-CGAP)/2,11.12,"BACKBONE CV-CNN",fc=CVe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# PROYECCIÓN AL DOMINIO REAL (3 ops apiladas verticalmente)
# ════════════════════════════════════════════════════════════════════════════════
PX=CX0+4*(CW+CGAP)-CGAP+CW+0.28; PW=2.70; POH=0.88; PGAP=0.28
PY_TOP=CY+CH        # alineado con tope de bloques

pops=[
    ("|z| = \u221a(Re\u00b2 + Im\u00b2)","Complejo \u2192 Dominio Real",PRe,"#E0F5F2"),
    ("AdaptiveAvgPool1D","pool_size = 32","#0A5E4A","#D7F2EC"),
    ("Flatten  +  Linear","4096 \u2192 256  ·  Dropout(0.30)","#0E4D72","#E0EEF8"),
]
pys=[PY_TOP-(k+1)*POH-k*PGAP for k in range(len(pops))]

sbg(PX-0.12,pys[-1]-0.18,PW+0.24,PY_TOP-pys[-1]+0.26,PRe,
    "Proyección al Dominio Real",yoff=0.06)

pdims=["[B, 128, N']","[B, 128, 32]","[B, 256]"]
for k,(pn,ps,pe2,pf2) in enumerate(pops):
    rb(PX,pys[k],PW,POH,fc=pf2,ec=pe2,lw=1.6,r=0.14)
    cx_p=PX+PW/2
    tx(cx_p,pys[k]+POH*0.67,pn,fc=pe2,fs=8.2,bold=True)
    tx(cx_p,pys[k]+POH*0.28,ps,fc=DIM,fs=7.2)
    dtag(PX+PW+0.70,pys[k]+POH/2,pdims[k])
    if k>0:
        ar(PX+PW/2,pys[k-1],PX+PW/2,pys[k]+POH,col=PRe,lw=1.4)

# flecha Bloque 4 → Modulus
b4_right=CX0+3*(CW+CGAP)+CW; b4_cy=CY+CH/2
ar(b4_right,b4_cy,PX,pys[0]+POH/2,col=CVe,lw=1.8)
dtag((b4_right+PX)/2,b4_cy+0.38,"[B, 256, N']\n(complejo)")

tx(PX+PW/2,11.12,"PROYECCIÓN REAL",fc=PRe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# RAMA FÍSICA — BatchNorm + 8 feature chips
# ════════════════════════════════════════════════════════════════════════════════
PH_Y0=1.05; PH_H=3.60
sbg(CX0-0.12,PH_Y0-0.10,4*(CW+CGAP)-CGAP+0.24,PH_H,PHe,
    "Rama de Características Físicas",fs=7.8,yoff=0.06)

# BatchNorm
BX=CX0; BW2=2.35; BH2=1.40; BY=PH_Y0+0.25
rb(BX,BY,BW2,BH2,fc=PHf,ec=PHe,lw=1.6)
tx(BX+BW2/2,BY+BH2*0.74,"BatchNorm1D",fc=PHe,fs=8.5,bold=True)
tx(BX+BW2/2,BY+BH2*0.44,"Normalización de",fc=DIM,fs=7.5)
tx(BX+BW2/2,BY+BH2*0.22,"8 variables físicas",fc=DIM,fs=7.5)

# flecha Física entrada → BN
ar(PHX+PHW,PHY+PHH/2,BX,BY+BH2/2,col=PHe,lw=2.0)
dtag((PHX+PHW+BX)/2,PHY+PHH/2+0.28,"[B, 8]")

# 8 chips: 2 filas × 4 columnas
FW2=1.48; FH2=0.60; FGAPX=0.23; FGAPY=0.22
FX0=BX+BW2+0.32
fnames=[["dur_ms","z_peak","drop_b","n_act"],
        ["global_nf","global_ns","H\u0305_mean","p75_act"]]
fdesc=[["Dur. ráfaga","Pico Z-score","Caída BG","Bins activos"],
       ["Ruido global","Saturación","Entropía media","Perc. 75"]]
row_ys=[PH_Y0+0.28+(FH2+FGAPY+0.25), PH_Y0+0.28]
for row in range(2):
    fy=row_ys[row]
    for col in range(4):
        fx=FX0+col*(FW2+FGAPX)
        rb(fx,fy,FW2,FH2,fc=PHf,ec=PHe,lw=1.0,r=0.10)
        tx(fx+FW2/2,fy+FH2*0.68,fnames[row][col],fc=PHe,fs=7.5,bold=True)
        tx(fx+FW2/2,fy+FH2*0.25,fdesc[row][col],fc=DIM,fs=6.5)

FRIGHT=FX0+4*(FW2+FGAPX)-FGAPX+FW2
BY_MID=BY+BH2/2
ar(BX+BW2,BY_MID,FX0,BY_MID,col=PHe,lw=1.4)
# label medio: centro de las dos filas
PHY_MID_Y=(row_ys[0]+FH2/2+row_ys[1]+FH2/2)/2

tx(CX0+(4*(CW+CGAP)-CGAP)/2,4.78,"PROCESAMIENTO FÍSICO",fc=PHe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# CONCATENACIÓN
# ════════════════════════════════════════════════════════════════════════════════
CTX=PX+PW+1.00; CTW=1.70; CTH=5.30
# posicionar para abarcar salida linear (pys[-1]+POH/2) y física (PHY_MID_Y)
lin_cy=pys[-1]+POH/2   # y de la salida del Linear
CTY=PHY_MID_Y-0.35     # arranca un poco por debajo de la entrada física
CTH=lin_cy-CTY+0.5     # alto hasta cubrir la entrada linear con margen
if CTH<4.0: CTH=4.0

rb(CTX,CTY,CTW,CTH,fc=CAf,ec=CAe,lw=2.0)
cx_ct=CTX+CTW/2; cy_ct=CTY+CTH/2
tx(cx_ct,cy_ct+0.60,"\u2295",fc=CAe,fs=20,bold=True)
tx(cx_ct,cy_ct-0.02,"Concat",fc=CAe,fs=9.5,bold=True)
tx(cx_ct,cy_ct-0.46,"264 = 256 + 8",fc=DIM,fs=7.8)

# Entradas al Concat
iq_entry_y=CTY+CTH-0.55   # entrada superior (IQ / Linear)
ph_entry_y=CTY+0.45        # entrada inferior (físicas)

# flechas
ar(PX+PW,lin_cy,CTX,iq_entry_y,col="#0E4D72",lw=1.8)
dtag((PX+PW+CTX)/2,lin_cy+0.30,"[B, 256]")

# Ruta física→Concat con tramo en L: derecha hasta x=CTX, luego diagonal
# Usamos dos segmentos de línea + flecha final
phys_corner_x=CTX-0.05
ax.annotate("",xy=(CTX,ph_entry_y),xytext=(FRIGHT,PHY_MID_Y),
    arrowprops=dict(arrowstyle="-|>",color=PHe,lw=1.8,mutation_scale=11,
        shrinkA=4,shrinkB=4,
        connectionstyle="angle,angleA=0,angleB=-90,rad=0.3"),zorder=5)
dtag((FRIGHT+CTX)/2,PHY_MID_Y+0.32,"[B, 8]")

# marcas de entrada
tx(CTX-0.06,iq_entry_y,"I/Q",fc=DIM,fs=6.5,ha="right")
tx(CTX-0.06,ph_entry_y,"Fís.",fc=DIM,fs=6.5,ha="right")

tx(cx_ct,11.12,"FUSIÓN",fc=CAe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# MLP HEAD
# ════════════════════════════════════════════════════════════════════════════════
MX=CTX+CTW+0.55; MW=3.05; MOH=1.12; MGAP=0.28
MY_TOP=CTY+CTH-0.10   # alinear con tope de concat
mlops=[
    ("Linear  264 \u2192 256","BatchNorm  +  ReLU  +  Dropout(0.40)",MLf,MLe),
    ("Linear  256 \u2192 128","BatchNorm  +  ReLU  +  Dropout(0.40)","#FFF5F5","#A52020"),
    ("Linear  128 \u2192 1","Logit de salida","#FFF8F8","#8B1A1A"),
]
mys=[MY_TOP-(k+1)*MOH-k*MGAP for k in range(len(mlops))]

sbg(MX-0.12,mys[-1]-0.18,MW+0.24,MY_TOP-mys[-1]+0.26,MLe,
    "Cabeza Clasificadora  ·  Perceptrón Multicapa (MLP)",fs=7.8,yoff=0.06)

mdims=["[B, 256]","[B, 128]","[B, 1]"]
for k,(mn,ms2,mf2,me2) in enumerate(mlops):
    rb(MX,mys[k],MW,MOH,fc=mf2,ec=me2,lw=1.6,r=0.15)
    cx_m=MX+MW/2
    tx(cx_m,mys[k]+MOH*0.68,mn,fc=me2,fs=8.5,bold=True)
    tx(cx_m,mys[k]+MOH*0.30,ms2,fc=DIM,fs=7.2)
    dtag(MX+MW+0.58,mys[k]+MOH/2,mdims[k])
    if k>0:
        ar(MX+MW/2,mys[k-1],MX+MW/2,mys[k]+MOH,col=me2,lw=1.5)

# flecha Concat → MLP layer 1
ar(CTX+CTW,cy_ct,MX,mys[0]+MOH/2,col=CAe,lw=2.0,arc=-0.25)
dtag(CTX+CTW+0.35,cy_ct+0.42,"[B, 264]")

tx(MX+MW/2,11.12,"CLASIFICADOR MLP",fc=MLe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# SIGMOIDE + SALIDA
# ════════════════════════════════════════════════════════════════════════════════
SGX=MX+MW+0.65; SGW=1.45; SGH=1.0
SGY=mys[-1]+(MOH-SGH)/2
rb(SGX,SGY,SGW,SGH,fc="#EBF0FA",ec="#2C4A7C",lw=1.6,r=0.15)
tx(SGX+SGW/2,SGY+SGH*0.68,"\u03c3(x)",fc="#2C4A7C",fs=10,bold=True)
tx(SGX+SGW/2,SGY+SGH*0.28,"Sigmoide",fc=DIM,fs=7.8)
ar(MX+MW,mys[-1]+MOH/2,SGX,SGY+SGH/2,col=MLe,lw=1.6)

OX=SGX+SGW+0.52; OW=1.60; OH=0.88
cy_sg=SGY+SGH/2
DRON_Y=cy_sg+0.56; NODRON_Y=cy_sg-0.56-OH

rb(OX,DRON_Y,OW,OH,fc=OPf,ec=OPe,lw=1.8,r=0.14)
tx(OX+OW/2,DRON_Y+OH*0.65,"DRON",fc=OPe,fs=9.5,bold=True)
tx(OX+OW/2,DRON_Y+OH*0.27,"p \u2265 umbral",fc=DIM,fs=7.5)

rb(OX,NODRON_Y,OW,OH,fc=ONf,ec=ONe,lw=1.8,r=0.14)
tx(OX+OW/2,NODRON_Y+OH*0.65,"NO DRON",fc=ONe,fs=9.5,bold=True)
tx(OX+OW/2,NODRON_Y+OH*0.27,"p < umbral",fc=DIM,fs=7.5)

ar(SGX+SGW,cy_sg,OX,DRON_Y+OH/2,col=OPe,lw=1.5,arc=-0.35)
ar(SGX+SGW,cy_sg,OX,NODRON_Y+OH/2,col=ONe,lw=1.5,arc=0.35)

tx(OX+OW/2,11.12,"SALIDA",fc=OPe,fs=8,bold=True)

# ════════════════════════════════════════════════════════════════════════════════
# LEYENDA
# ════════════════════════════════════════════════════════════════════════════════
lg=[(IQe,IQf,"Señal I/Q"),(CVe,CVf,"Conv. Compleja"),
    (S1e,S1f,"ComplexConv1D"),(S2e,S2f,"ComplexBN"),(S3e,S3f,"CReLU"),
    (PHe,PHf,"Caract. Físicas"),(CAe,CAf,"Concatenación"),
    (MLe,MLf,"Perceptrón MLP"),(OPe,OPf,"Dron"),(ONe,ONf,"No Dron")]
LY=0.30; dx2=(FW-1.0)/len(lg)
for i,(ec2,fc2,lb) in enumerate(lg):
    lx=0.5+i*dx2
    ax.add_patch(FancyBboxPatch((lx,LY-0.13),0.36,0.26,
        boxstyle="round,pad=0,rounding_size=0.04",lw=1.0,edgecolor=ec2,facecolor=fc2,zorder=6))
    tx(lx+0.52,LY+0.01,lb,fc=DIM,fs=7.5,ha="left",va="center")

OUT=r"c:\repos\DroneDetectionRF\figuras_arquitectura"
fig.savefig(os.path.join(OUT,"singlestream_architecture.pdf"),dpi=300,bbox_inches="tight",facecolor=BG)
fig.savefig(os.path.join(OUT,"singlestream_architecture.png"),dpi=300,bbox_inches="tight",facecolor=BG)
print("OK"); plt.close(fig)
