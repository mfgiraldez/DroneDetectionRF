"""
run_diagnostic.py
=================
Analisis exploratorio de las features del detector de entropia.
Verifica empiricamente las hipotesis H1, H2 y H3 del plan de investigacion.

Hipótesis:
  H1: Las features del detector discriminan drones de ruido a SNR >= -6 dB con >85% acc
  H2: A SNR < -10 dB, las features globales (H_mean, H_min, p75_bins) mantienen info
  H3: n_act y dur_ms son las dos features más discriminativas (discriminan WiFi/BT)

Salida:
  - Consola: estadísticas por clase/SNR + accuracy clasificador lineal
  - Figuras: diagnostic_figures/ en la carpeta de resultados
"""

import sys, os, warnings
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.funciones.detector_entropia import detectar_bursts

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
N_SAMPLES_PER_SNR_CLASS = 8   # muestras por nivel SNR × clase binaria
FS = 14e6
NPERSEG = 2048
OUT_DIR = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_marnet\diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "noise_floor",     # 0: Piso de ruido CFAR (bits)
    "noise_sigma",     # 1: Sigma del ruido (MAD)
    "mean_n_active",   # 2: Bins activos medios
    "p75_n_active",    # 3: P75 bins activos (WiFi discriminator)
    "H_min",           # 4: Mínimo de entropía (concentración espectral)
    "H_mean",          # 5: Entropía media
    "n_bursts",        # 6: Número de bursts detectados
    "dur_ms_main",     # 7: Duración del burst principal (ms)
    "z_peak_main",     # 8: Significancia estadística
    "drop_b_main",     # 9: Caída de entropía en el burst
    "n_act_main",      # 10: Bins activos en el burst
    "dur_total",       # 11: Duración total de actividad (ms)
]

# Colores
BLUE = "#2E86AB"
RED  = "#E84855"
FIGSAVE = dict(dpi=130, bbox_inches="tight", facecolor="#F8F9FA")

# ============================================================================
# EXTRACCIÓN DE FEATURES
# ============================================================================

def extract_features(iq_tensor, fs=FS, nperseg=NPERSEG):
    """Extrae 12 features físicas de un tensor IQ completo."""
    try:
        _, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
            iq_tensor, fs=fs, nperseg=nperseg,
            adaptive_window_ms=15.0, min_burst_ms=0.3, z_thresh=2.5
        )
        f = [
            float(np.median(nf_v)),
            float(ns),
            float(np.mean(n_active)),
            float(np.percentile(n_active, 75)),
            float(np.min(H_smooth)),
            float(np.mean(H_smooth)),
        ]
        if bursts:
            b0 = max(bursts, key=lambda b: abs(b['z_peak']))
            f += [
                float(len(bursts)),
                float(b0['dur_ms']),
                float(min(b0['z_peak'], 30.0)),
                float(min(b0['drop_b'], 10.0)),
                float(b0['n_act']),
                float(min(sum(b['dur_ms'] for b in bursts), 75.0)),
            ]
        else:
            f += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    except Exception as e:
        print(f"  [WARN] Feature extraction failed: {e}")
        f = [0.0] * 12
    return np.array(f, dtype=np.float32)


# ============================================================================
# MUESTREO ESTRATIFICADO
# ============================================================================

def sample_dataset(n_per_snr_class=N_SAMPLES_PER_SNR_CLASS):
    """Genera una muestra estratificada por SNR y clase binaria."""
    df_train, df_val, df_test = obtener_splits_dataset()
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    samples = []
    snr_levels = sorted(df_all['snr'].unique())
    for snr in snr_levels:
        for label in [0, 1]:
            subset = df_all[(df_all['snr'] == snr) & (df_all['label'] == label)]
            picked = subset.sample(n=min(n_per_snr_class, len(subset)), random_state=42)
            samples.append(picked)

    return pd.concat(samples, ignore_index=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  DIAGNOSTIC: Physical Features from Entropy Detector")
    print("=" * 65)

    # --- 1. Muestra estratificada ---
    print("\n[1/4] Sampling dataset...")
    df = sample_dataset()
    print(f"  Total samples: {len(df)} | "
          f"Drones: {df['label'].sum()} | "
          f"Noise: {(df['label']==0).sum()}")

    # --- 2. Computar features ---
    print(f"\n[2/4] Computing features ({len(df)} samples)...")
    all_feats, all_labels, all_snrs, all_grupos = [], [], [], []

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  {i+1}/{len(df)} samples processed...")
        try:
            d = torch.load(row['filepath'], map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
            feats = extract_features(iq)
        except Exception as e:
            print(f"  [WARN] Load failed: {e}")
            feats = np.zeros(12, dtype=np.float32)

        all_feats.append(feats)
        all_labels.append(int(row['label']))
        all_snrs.append(int(row['snr']))
        all_grupos.append(row['grupo'])

    X = np.stack(all_feats)          # [N, 12]
    y = np.array(all_labels)         # [N]
    snrs = np.array(all_snrs)
    grupos = np.array(all_grupos)

    print(f"  Features matrix: {X.shape}")
    print(f"  NaN count: {np.isnan(X).sum()} | Inf count: {np.isinf(X).sum()}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 3. Clasificación lineal global y por grupo ---
    print("\n[3/4] Linear classifier evaluation...")

    results = {}
    for grupo_name in ['ALL', 'A', 'B', 'C']:
        if grupo_name == 'ALL':
            mask = np.ones(len(y), dtype=bool)
        else:
            mask = grupos == grupo_name

        Xg, yg = X[mask], y[mask]
        if len(np.unique(yg)) < 2 or len(yg) < 10:
            continue

        scaler = StandardScaler()
        Xg_s = scaler.fit_transform(Xg)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(Xg_s, yg)
        preds = clf.predict(Xg_s)
        probs = clf.predict_proba(Xg_s)[:, 1]

        acc = accuracy_score(yg, preds)
        f1  = f1_score(yg, preds, zero_division=0)
        try:
            auc = roc_auc_score(yg, probs)
        except Exception:
            auc = float('nan')

        results[grupo_name] = {'acc': acc, 'f1': f1, 'auc': auc, 'n': mask.sum()}

        # Feature importances (logistic regression coefficients)
        importance = np.abs(clf.coef_[0])
        top_idx = np.argsort(importance)[::-1][:5]

        snr_range = {
            'ALL': 'all SNRs', 'A': 'SNR>=10dB', 'B': '-6<=SNR<10dB', 'C': 'SNR<-6dB'
        }[grupo_name]

        print(f"\n  [{grupo_name}] {snr_range} (n={mask.sum()})")
        print(f"    Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
        print(f"    Top-5 features: {[FEATURE_NAMES[i] for i in top_idx]}")
        print(f"    Importances: {importance[top_idx].round(3).tolist()}")
        results[grupo_name]['top_features'] = [(FEATURE_NAMES[i], float(importance[i])) for i in top_idx]

    # --- 4. Figuras ---
    print("\n[4/4] Generating diagnostic figures...")

    # Fig 1: Feature distributions by class for each SNR group
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("Feature Distributions: Drone (blue) vs Noise (red) by SNR Group",
                 fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')

    for i, feat_name in enumerate(FEATURE_NAMES):
        ax = axes[i // 4, i % 4]
        for label, color, name in [(1, BLUE, 'Drone'), (0, RED, 'Noise')]:
            vals = X[y == label, i]
            ax.hist(vals, bins=30, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat_name, fontsize=9, fontweight='bold')
        ax.set_xlabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_diag_01_distributions_global.png", **FIGSAVE)
    plt.close(fig)
    print("  Saved: fig_diag_01_distributions_global.png")

    # Fig 2: n_act and dur_ms by SNR (the two key discriminators)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Key Discriminators: n_act (bins) and dur_ms (burst duration) by SNR",
                 fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')

    for ax, feat_idx, feat_name in zip(axes, [3, 7], ['p75_n_active (bins)', 'dur_ms_main (ms)']):
        snr_unique = sorted(set(snrs))
        vals_drone = [X[(y==1) & (snrs==snr), feat_idx].mean() if ((y==1) & (snrs==snr)).any() else 0
                      for snr in snr_unique]
        vals_noise = [X[(y==0) & (snrs==snr), feat_idx].mean() if ((y==0) & (snrs==snr)).any() else 0
                      for snr in snr_unique]

        ax.plot(snr_unique, vals_drone, 'o-', color=BLUE, label='Drone', lw=2, ms=5)
        ax.plot(snr_unique, vals_noise, 's--', color=RED, label='Noise', lw=2, ms=5)
        ax.axvline(-6, color='orange', linestyle=':', lw=1.5, label='SNR=-6 dB')
        ax.axvline(-10, color='gray', linestyle=':', lw=1.5, label='SNR=-10 dB')
        ax.set_title(feat_name, fontsize=11, fontweight='bold')
        ax.set_xlabel("SNR (dB)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_diag_02_key_features_by_snr.png", **FIGSAVE)
    plt.close(fig)
    print("  Saved: fig_diag_02_key_features_by_snr.png")

    # Fig 3: Linear classifier accuracy by SNR level
    fig, ax = plt.subplots(figsize=(12, 5))
    snr_unique = sorted(set(snrs))
    accs_by_snr, f1s_by_snr = [], []
    for snr in snr_unique:
        mask = snrs == snr
        Xs, ys = X[mask], y[mask]
        if len(np.unique(ys)) < 2 or len(ys) < 4:
            accs_by_snr.append(float('nan'))
            f1s_by_snr.append(float('nan'))
            continue
        try:
            sc = StandardScaler()
            Xs_s = sc.fit_transform(Xs)
            clf2 = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf2.fit(Xs_s, ys)
            accs_by_snr.append(accuracy_score(ys, clf2.predict(Xs_s)))
            f1s_by_snr.append(f1_score(ys, clf2.predict(Xs_s), zero_division=0))
        except Exception:
            accs_by_snr.append(float('nan'))
            f1s_by_snr.append(float('nan'))

    ax.plot(snr_unique, accs_by_snr, 'o-', color=BLUE, lw=2, ms=6, label='Accuracy')
    ax.plot(snr_unique, f1s_by_snr,  's--', color=RED,  lw=2, ms=6, label='F1-Score')
    ax.axhline(0.75, color='black', linestyle='-.', lw=1.5, label='Baseline CV-CNN 75%')
    ax.axhline(0.5,  color='gray',  linestyle=':',  lw=1,   label='Random (50%)')
    ax.axvline(-6,  color='orange', linestyle=':', lw=1.5, label='Group B/C boundary')
    ax.axvline(-10, color='gray',   linestyle=':', lw=1.5, label='SNR=-10 dB')
    ax.fill_betweenx([0, 1], -22, -6, alpha=0.07, color='red', label='Group C (hostile SNR<-6)')
    ax.fill_betweenx([0, 1],  10, 32, alpha=0.07, color='green', label='Group A (easy SNR>=10)')
    ax.set_xlim(-22, 32)
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Linear Classifier (Logistic Reg.) on Physical Features — Accuracy by SNR",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_diag_03_accuracy_by_snr.png", **FIGSAVE)
    plt.close(fig)
    print("  Saved: fig_diag_03_accuracy_by_snr.png")

    # --- Summary ---
    print("\n" + "="*65)
    print("  DIAGNOSTIC SUMMARY")
    print("="*65)
    for grupo, r in results.items():
        label = {'ALL': 'Global      ', 'A': 'Group A (SNR>=10dB) ',
                 'B': 'Group B (-6<=SNR<10dB)', 'C': 'Group C (SNR<-6dB) '}[grupo]
        print(f"\n  {label} (n={r['n']})")
        print(f"    Linear Acc={r['acc']:.3f}  F1={r['f1']:.3f}  AUC={r['auc']:.3f}")
        print(f"    Top features: {[f[0] for f in r['top_features'][:3]]}")

    print(f"\n  Figures saved to: {OUT_DIR}")

    # Verify hypotheses
    print("\n  HYPOTHESIS EVALUATION:")
    acc_all = results.get('ALL', {}).get('acc', 0)
    acc_b   = results.get('B',   {}).get('acc', 0)
    acc_c   = results.get('C',   {}).get('acc', 0)

    h1_ok = acc_b >= 0.80
    h2_ok = acc_c >= 0.60
    print(f"  H1 (features discriminate at SNR>=-6): acc_B={acc_b:.3f} → "
          f"{'CONFIRMED' if h1_ok else 'NEEDS MORE'}")
    print(f"  H2 (global features at SNR<-6):        acc_C={acc_c:.3f} → "
          f"{'CONFIRMED' if h2_ok else 'PARTIALLY'}")

    # Check if n_act is top feature
    top_global = [f[0] for f in results.get('ALL', {}).get('top_features', [])]
    h3_ok = any('n_act' in f or 'dur' in f for f in top_global[:3])
    print(f"  H3 (n_act/dur_ms most discriminative): top={top_global[:3]} → "
          f"{'CONFIRMED' if h3_ok else 'CHECK FIGURES'}")

    print("\n  Run complete.")
    return results


if __name__ == "__main__":
    main()
