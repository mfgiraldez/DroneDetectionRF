
# extractor_dataset_final.py

# Este script lee el parquet de la Stage 1, carga los tensores RF crudos originales,
# aplica los recortes (crops) exactos definidos por los frames (start, end) y los 
# guarda individualmente en disco listos para entrenar la CV-CNN en Stage 2.

# Esquema de salida:
# C:\TFM_data\NoisyUAV\stage2\
#     class_0_no_drone\
#         burst_...pt
#     class_1_drone\
#         burst_...pt
# """

import os
import torch
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

PARQUET_PATH = r"C:\TFM_data\NoisyUAV\stage1\metadata_stage1.parquet"
ORIGINAL_DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
STAGE2_OUT_DIR = r"C:\TFM_data\NoisyUAV\stage2"
CLASS_0_DIR = os.path.join(STAGE2_OUT_DIR, "class_0_no_drone")
CLASS_1_DIR = os.path.join(STAGE2_OUT_DIR, "class_1_drone")

# Parámetros del STFT que usamos en Stage 1 para mapear de frames a muestras
NPERSEG = 2048
HOP_SIZE = NPERSEG // 2  # Usamos noverlap = nperseg // 2

def setup_directories():
    os.makedirs(CLASS_0_DIR, exist_ok=True)
    os.makedirs(CLASS_1_DIR, exist_ok=True)

def process_single_file(file_name: str, group_df: pd.DataFrame) -> list:
    """
    Toma todas las detecciones (ráfagas) que pertenecen a UN mismo archivo original,
    carga ese archivo original solo 1 vez de disco, recorta sus ráfagas y guarda
    los tensores resultantes. Retorna metadata del crop para el CSV maestro.
    """
    file_path = os.path.join(ORIGINAL_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        return []

    try:
        # Cargar todo el tensor original 1 sola vez por archivo (~1M muestras)
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        iq_tensor = data["x_iq"]  # [2, N] (usualmente N=1048576) o numpy array
        
        # Compatibilidad si venía guardado como numpy (lo pasamos a torch)
        if hasattr(iq_tensor, 'numpy') == False: # if it's already a numpy array
            iq_tensor = torch.as_tensor(iq_tensor)

        total_samples = iq_tensor.shape[1]

        crop_metadata_list = []

        # Procesar cada ráfaga (burst) detectada en este archivo
        for _, row in group_df.iterrows():
            idx = int(row["burst_idx"])
            target = int(row["target"])
            if target == 4:
                label = 0 # NO_DRONE
                out_folder = CLASS_0_DIR
            else:
                label = 1 # DRONE
                out_folder = CLASS_1_DIR

            # Mapeo conservador: el STFT se montó con tamaño de ventana NPERSEG y steps de HOP_SIZE.
            # Para capturar el burst sin perder energía:
            frame_i0 = int(row["frame_i0"])
            frame_i1 = int(row["frame_i1"])
            
            start_sample = max(0, frame_i0 * HOP_SIZE)
            end_sample = min(total_samples, frame_i1 * HOP_SIZE + NPERSEG)

            # Extraemos el crop raw I/Q y HACEMOS CLONE() para romper el puntero
            # al Storage original. PyTorch por defecto guarda el buffer subyacente
            # entero (~8MB) aunque solo guardes una sub-vista, si no clonamos.
            iq_crop = iq_tensor[:, start_sample:end_sample].clone()

            # Si el crop es muy pequeño o inválido, lo saltamos
            if iq_crop.shape[1] < HOP_SIZE:
                continue

            # Generamos nombre de archivo
            crop_filename = f"{file_name.replace('.pt', '')}_burst{idx}.pt"
            crop_path = os.path.join(out_folder, crop_filename)

            out_dict = {
                "x_iq": iq_crop,
                "label": label,
                "target": target,
                "snr": float(row["snr"])
            }
            # Guardamos el tensor pequenito comprimido
            torch.save(out_dict, crop_path)

            crop_metadata_list.append({
                "crop_id": crop_filename,
                "file_path": crop_path,
                "label": label,
                "target": target,
                "snr": float(row["snr"]),
                "dur_ms": float(row["dur_ms"]),
                "z_peak": float(row["z_peak"]),
                "samples_len": iq_crop.shape[1]
            })

        return crop_metadata_list

    except Exception as e:
        print(f"Error parseando {file_name}: {e}")
        return []

def main():
    print(f"Leyendo metadata de Stage 1 desde: {PARQUET_PATH}")
    if not os.path.exists(PARQUET_PATH):
        print("No se encuentra el parquet. Asegúrate de haber ejecutado stage 1 primero.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    # Ignorar archivos enteros que no tuvieron detecciones (burst_idx == -1)
    df_bursts = df[df["burst_idx"] != -1].copy()

    print(f"Preparando directorios en {STAGE2_OUT_DIR} ...")
    setup_directories()

    # Agrupar por nombre de archivo. Así cargamos cada archivo I/Q de disco 
    # exactamente 1 vez, y sacamos todas sus ráfagas, siendo ultra eficientes (I/O bound).
    grouped = list(df_bursts.groupby("archivo"))
    print(f"Hay un total de {len(df_bursts)} ráfagas distribuidas en {len(grouped)} archivos.")

    all_crops_metadata = []

    # Paralelismo fuerte
    print("🚀 Extrayendo crops I/Q brutos (esto generará muchos ficheros pequeños) ...")
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        futures = {executor.submit(process_single_file, fname, grp): fname for fname, grp in grouped}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Crops"):
            try:
                res = future.result()
                all_crops_metadata.extend(res)
            except Exception as e:
                print(f"Unhandled exception in parallel execution: {e}")

    # Guarda el indexado general
    meta_df = pd.DataFrame(all_crops_metadata)
    meta_csv_path = os.path.join(STAGE2_OUT_DIR, "metadata_stage2.csv")
    meta_df.to_csv(meta_csv_path, index=False)

    print("\n✅ Extracción del Dataset Stage 2 terminada.")
    print(f"Clase 0 (NO_DRONE) construida: {len(meta_df[meta_df['label'] == 0])} crops.")
    print(f"Clase 1 (DRONE) construida:    {len(meta_df[meta_df['label'] == 1])} crops.")
    print(f"Log maestro Stage 2 guardado en: {meta_csv_path}")

if __name__ == "__main__":
    main()
