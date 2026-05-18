"""
Script de Evaluación Masiva (Stage 1) - Dataset NoisyUAV

Este script recorre de forma paralela los 17,744 archivos del dataset,
aplica el Detector de Entropía (Stage 1) a cada uno y exporta un
log estructurado en formato Parquet (`metadata_stage1.parquet`).

El Parquet resultante se utilizará en el siguiente paso para:
1. Extraer las estadísticas oficiales de falsos positivos (Pfa vs SNR).
2. Filtrar el Label Noise (Disonancia BT) usando Isolation Forest.
3. Extraer el dataset limpio final para la CNN (Stage 2).
"""

import os
import glob
import time
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from funciones.cargador import DATA_DIR, _PATRON_NOMBRE, cargar_muestra
from funciones.detector_entropia import detectar_bursts

# Parámetros del detector (los mismos que usamos visualmente en el notebook)
FS = 14e6
NPERSEG = 2048
Z_THRESH = 3.0
MIN_BURST_MS = 0.5
MERGE_GAP_MS = 1.0
MIN_Z_ABS = 4.0
BG_MULT = 4
MAX_BINS_FRAC = 0.25
ADAPTIVE_WINDOW_MS = 10

def procesar_archivo(file_path: str) -> list:
    """
    Worker task: Carga UN archivo utilizando cargador.py, ejecuta el detector y
    devuelve una lista de diccionarios (uno por cada burst detectado).
    """
    try:
        # Reutilizamos exactamente la misma función validada que usa el Notebook
        iq_tensor, sample_id, target, snr = cargar_muestra(file_path)
        nombre = os.path.basename(file_path)
        
        # Obtener los bursts usando la misma lógica que matplotlib/plotly
        t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
            iq_tensor, 
            fs=FS, 
            nperseg=NPERSEG, 
            z_thresh=Z_THRESH, 
            min_burst_ms=MIN_BURST_MS, 
            merge_gap_ms=MERGE_GAP_MS, 
            min_z_abs=MIN_Z_ABS, 
            bg_mult=BG_MULT, 
            max_bins_frac=MAX_BINS_FRAC, 
            adaptive_window_ms=ADAPTIVE_WINDOW_MS
        )

        resultados = []
        
        # Caso 0 bursts (Rechazo correcto o Falso Negativo)
        if not bursts:
            resultados.append({
                "archivo": nombre,
                "sample_id": sample_id,
                "target": target,
                "snr": snr,
                "num_bursts_total": 0,
                "burst_idx": -1,
                "t0_ms": -1.0,
                "t1_ms": -1.0,
                "frame_i0": -1,
                "frame_i1": -1,
                "dur_ms": 0.0,
                "drop_b": 0.0,
                "z_peak": 0.0,
                "n_active_mean": 0.0
            })
        else:
            # Registrar cada métrica física para el posterior Isolation Forest
            for idx, b in enumerate(bursts):
                resultados.append({
                    "archivo": nombre,
                    "sample_id": sample_id,
                    "target": target,
                    "snr": snr,
                    "num_bursts_total": len(bursts),
                    "burst_idx": idx,
                    "t0_ms": float(b["t0"]),
                    "t1_ms": float(b["t1"]),
                    "frame_i0": int(b["i0"]),
                    "frame_i1": int(b["i1"]),
                    "dur_ms": float(b["dur_ms"]),
                    "drop_b": float(b["drop_b"]),
                    "z_peak": float(b["z_peak"]),
                    "n_active_mean": float(b["n_act"])
                })
                
        return resultados

    except Exception as e:
        print(f"Error procesando {nombre}: {e}")
        return []

def mass_evaluate(data_dir: str = DATA_DIR, output_parquet: str = "metadata_stage1.parquet"):
    patron = os.path.join(data_dir, "*.pt")
    archivos = glob.glob(patron)
    total_archivos = len(archivos)
    
    if total_archivos == 0:
        print(f"Error: No se encontraron archivos .pt en {data_dir}")
        return

    print(f"🚀 Iniciando Evaluación Masiva Stage 1 sobre {total_archivos} archivos...")
    print(f"Parámetros: Z_THRESH={Z_THRESH}, ADAPTIVE_WIN={ADAPTIVE_WINDOW_MS}ms, MAX_BINS={MAX_BINS_FRAC*100}%")
    
    t_start = time.time()
    
    todas_las_filas = []
    
    # Procesamiento Paralelo Multihilo CPU-bound
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Encolar tareas
        futuros = {executor.submit(procesar_archivo, ruta): ruta for ruta in archivos}
        
        # Barra de progreso
        with tqdm(total=total_archivos, desc="Procesando Dataset", unit="archivos") as pbar:
            for futuro in as_completed(futuros):
                resultado_archivo = futuro.result()
                if resultado_archivo:
                    todas_las_filas.extend(resultado_archivo)
                pbar.update(1)

    t_end = time.time()
    print(f"✅ Procesamiento completado en {(t_end - t_start)/60:.2f} minutos.")
    
    if todas_las_filas:
        df = pd.DataFrame(todas_las_filas)
        df.to_parquet(output_parquet, index=False)
        print(f"💾 Guardado log maestro ({len(df)} detecciones) en: {output_parquet}")
        
        # Pequeño reporte crudo
        archivos_con_bursts = df[df["num_bursts_total"] > 0]["archivo"].nunique()
        archivos_noise = df[df["target"] == 4]["archivo"].nunique()
        ruido_con_bursts = df[(df["target"] == 4) & (df["num_bursts_total"] > 0)]["archivo"].nunique()
        
        print("\n📈 [RESUMEN RÁPIDO]")
        print(f"  Archivos con Detecciones: {archivos_con_bursts} / {total_archivos} ({(archivos_con_bursts/total_archivos)*100:.1f}%)")
        if archivos_noise > 0:
            print(f"  Falsas Alarmas (Noise Cls 4): {ruido_con_bursts} archivos / {archivos_noise} ({(ruido_con_bursts/archivos_noise)*100:.1f}% leakage)")
    else:
        print("⚠️ No se extrajo ningún dato.")

if __name__ == "__main__":
    base_dir = os.path.dirname(DATA_DIR)
    out_dir = os.path.join(base_dir, "stage1")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "metadata_stage1.parquet")
    print(f"Destino planificado: {out_file}")
    mass_evaluate(DATA_DIR, out_file)
