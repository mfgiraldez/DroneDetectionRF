"""
calibrate_drone_durations.py
============================
Para cada target de dron (SNR=20dB), carga un fichero aleatorio, aplica el
detector de entropía y el modelo Teacher para identificar los bursts reales,
y extrae estadísticas de duración.

Salida: curriculum_alumn_v1/drone_duration_ref.json
{
  "5": {"target": 5, "snr": 20, "n_bursts": 4,
        "dur_min": 0.52, "dur_p10": 0.54, "dur_median": 0.58,
        "dur_mean": 0.57, "dur_max": 0.62, "dur_std": 0.04,
        "fichero": "IQdata_sample123_target5_snr20.pt"},
  ...
}
"""
import sys, os, glob, random, json, itertools
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.detector_entropia import detectar_bursts
from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as TeacherCVCNN
from NoisyUAV.funciones import TARGET_NOISE

# ── CONFIGURACION ─────────────────────────────────────────────────────────────
DATA_DIR     = Path(r"C:\TFM_data\NoisyUAV\drone_RF_data")
TEACHER_CKPT = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_teacher_v1\checkpoints\teacher_model_best.pt")
OUT_DIR      = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1")
OUT_JSON     = OUT_DIR / "drone_duration_ref.json"

SNR_CALIB    = 20       # SNR alta para calibración (señal limpia, sin ruido)
MIN_PROB_IA  = 0.50     # Umbral mínimo para considerar un burst como dron
N_FILES      = 5        # Ficheros aleatorios por target (promedia la calibración)

# Parámetros del detector a alta SNR (más estrictos → menos falsas alarmas)
# Con 20dB el dron domina, podemos ser más exigentes
FS=14e6; NPERSEG=2048; Z_THRESH=3.5; MIN_BURST_MS=0.5; MERGE_GAP_MS=1.5
MIN_Z_ABS=4.0; BG_MULT=4; MAX_BINS_FRAC=0.25; SMOOTH_MS=0.2; ADAPTIVE_WINDOW_MS=15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

def get_suelo_ms(h_seg, u_seg, dt):
    drop_max = np.max(u_seg - h_seg) + 1e-8
    mask = h_seg < (u_seg - 0.7 * drop_max)
    if not np.any(mask): return 0.0
    return max(len(list(g)) for k, g in itertools.groupby(mask) if k) * dt

def infer_bursts(iq, bursts, model, phys_mean, phys_std, t_ms, H_smooth, umbral_v, n_active, nf_v, ns):
    """Pasa todos los bursts por el Teacher y devuelve prob_ia por burst."""
    global_nf     = float(np.median(nf_v))
    global_ns     = float(np.clip(ns, 0, 5))
    global_H_mean = float(np.mean(H_smooth))
    global_p75_act= float(np.percentile(n_active, 75))
    dt = float(t_ms[1] - t_ms[0])

    model.eval()
    with torch.no_grad():
        for b in bursts:
            h_seg = H_smooth[b['i0']:b['i1']+1]
            u_seg = umbral_v[b['i0']:b['i1']+1]
            b['suelo_ms'] = get_suelo_ms(h_seg, u_seg, dt)

            idx_i = int(b['t0']*1e-3*FS); idx_f = int(b['t1']*1e-3*FS)
            p_raw = iq[:, idx_i:idx_f]
            p_norm = p_raw / p_raw.pow(2).mean().clamp(min=1e-12).sqrt()
            T_LEN = int(9.4e-3*FS)
            p_pad = (torch.cat([p_norm, torch.zeros(2, T_LEN-p_norm.shape[1])], dim=1)
                     if p_norm.shape[1] < T_LEN else p_norm[:, :T_LEN])

            feat = torch.from_numpy(np.array([
                np.clip(b['dur_ms'],0,75), np.clip(abs(b['z_peak']),0,30),
                np.clip(b['drop_b'],0,10), np.clip(b['n_act'],0,2048),
                global_nf, global_ns, global_H_mean, global_p75_act
            ], dtype=np.float32)).to(DEVICE)
            feat_norm = torch.clamp((feat-phys_mean)/(phys_std+1e-8),-5.,5.).unsqueeze(0)

            b['prob_ia'] = torch.sigmoid(
                model(p_pad.unsqueeze(0).to(DEVICE), feat_norm)
            ).item()
    return bursts


def calibrar_target(target, teacher_model, phys_mean, phys_std):
    """Procesa N_FILES aleatorios del target en SNR_CALIB y devuelve durations."""
    pattern = str(DATA_DIR / f"IQdata_*_target{target}_snr{SNR_CALIB}.pt")
    archivos = glob.glob(pattern)
    if not archivos:
        print(f"  ⚠️  Target {target}: sin ficheros a SNR={SNR_CALIB}dB")
        return None

    random.shuffle(archivos)
    archivos = archivos[:N_FILES]

    todas_durs = []
    ficheros_usados = []

    for fpath in archivos:
        try:
            d  = torch.load(fpath, map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
        except Exception as e:
            print(f"    ❌ Error cargando {Path(fpath).name}: {e}")
            continue

        t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
            iq, fs=FS, nperseg=NPERSEG, z_thresh=Z_THRESH,
            min_burst_ms=MIN_BURST_MS, merge_gap_ms=MERGE_GAP_MS,
            bg_mult=BG_MULT, max_bins_frac=MAX_BINS_FRAC,
            smooth_ms=SMOOTH_MS, adaptive_window_ms=ADAPTIVE_WINDOW_MS)

        if len(bursts) == 0:
            continue

        bursts = infer_bursts(iq, bursts, teacher_model, phys_mean, phys_std,
                              t_ms, H_smooth, umbral_v, n_active, nf_v, ns)

        # Filtramos: solo bursts que el Teacher confirma como dron
        drone_bursts = [b for b in bursts if b['prob_ia'] >= MIN_PROB_IA]
        if drone_bursts:
            todas_durs.extend([b['dur_ms'] for b in drone_bursts])
            ficheros_usados.append(Path(fpath).name)

    if not todas_durs:
        print(f"  ⚠️  Target {target}: Teacher no confirmó ningún burst como dron")
        return None

    durs = np.array(todas_durs)
    # Referencia = MEDIA de las duraciones confirmadas.
    # No usamos el mínimo (peligroso: una detección espuria corrompe la referencia).
    return {
        "target":       target,
        "snr_calib":    SNR_CALIB,
        "n_bursts":     len(durs),
        "n_ficheros":   len(ficheros_usados),
        "ficheros":     ficheros_usados,
        "dur_ref":      float(np.mean(durs)),      # <-- referencia principal
        "dur_median":   float(np.median(durs)),
        "dur_std":      float(np.std(durs)),
        "dur_min":      float(np.min(durs)),        # solo informativo
        "dur_max":      float(np.max(durs)),        # solo informativo
    }


def main():
    if not TEACHER_CKPT.exists():
        print(f"❌ No se encuentra el checkpoint del Teacher en:\n   {TEACHER_CKPT}")
        return

    print("=" * 60)
    print("  CALIBRACIÓN DE DURACIONES POR TARGET (SNR=20dB)")
    print("=" * 60)
    print(f"  Teacher: {TEACHER_CKPT.name}")
    print(f"  Ficheros por target: {N_FILES}")
    print(f"  Umbral Teacher: prob_ia >= {MIN_PROB_IA}")
    print()

    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    teacher_model = TeacherCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    teacher_model.load_state_dict(ckpt['model_state'])
    teacher_model.eval()

    phys_mean = torch.from_numpy(ckpt['phys_mean']).to(DEVICE)
    phys_std  = torch.from_numpy(ckpt['phys_std']).to(DEVICE)

    # Descubrir todos los targets de dron disponibles a SNR_CALIB
    patron = str(DATA_DIR / f"IQdata_*_snr{SNR_CALIB}.pt")
    archivos_all = glob.glob(patron)
    targets_disponibles = set()
    for f in archivos_all:
        m = Path(f).stem  # IQdata_sampleXXX_targetY_snrZ
        partes = m.split('_')
        for p in partes:
            if p.startswith('target'):
                t = int(p.replace('target', ''))
                if t != TARGET_NOISE:
                    targets_disponibles.add(t)

    targets_disponibles = sorted(targets_disponibles)
    print(f"Targets de dron encontrados a SNR={SNR_CALIB}dB: {targets_disponibles}\n")

    resultados = {}
    for target in tqdm(targets_disponibles, desc="Calibrando targets"):
        print(f"\n→ Target {target}:")
        stats = calibrar_target(target, teacher_model, phys_mean, phys_std)
        if stats:
            resultados[str(target)] = stats
            print(f"  ✅ n_bursts={stats['n_bursts']} | "
                  f"dur_ref(mean)={stats['dur_ref']:.2f}ms | "
                  f"median={stats['dur_median']:.2f}ms | "
                  f"std={stats['dur_std']:.2f}ms | "
                  f"[{stats['dur_min']:.2f} - {stats['dur_max']:.2f}]ms")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(resultados, f, indent=2)

    print(f"\n✅ JSON guardado en: {OUT_JSON}")
    print(f"   Targets calibrados: {list(resultados.keys())}")


if __name__ == '__main__':
    main()
