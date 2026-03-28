"""
Módulo de carga de datos para el dataset NoisyUAV v2.
Maneja la lectura de tensores .pt, el parsing de metadatos desde los nombres
de archivo, y la lectura de los CSVs estadísticos del dataset.

Dataset: Glüge et al. (2024) - "Robust Low-Cost Drone Detection and 
Classification Using CNNs in Low SNR Environments"
"""

import os
import re
import glob
import torch
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict

# ============================================================================
# CONSTANTES DEL DATASET
# ============================================================================

DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"

FREQ_MUESTREO = 14e6  # 14 MHz (downsampled desde 56 MHz originales)

LONGITUD_MUESTRA = 1_048_576  # 2^20 muestras I/Q por archivo

DURACION_MUESTRA = LONGITUD_MUESTRA / FREQ_MUESTREO  # ~74.9 ms

NOMBRES_CLASES = {
    0: "DJI",
    1: "FutabaT14",
    2: "FutabaT7",
    3: "Graupner",
    4: "Noise",       # ⚠️ Noise = 4 (orden alfabético del CSV, NO como en el paper)
    5: "Taranis",
    6: "Turnigy",
}

# Para detección binaria: drones (0-3, 5-6) → 1, ruido (4) → 0
NOMBRES_BINARIOS = {0: "No-Dron", 1: "Dron"}
TARGET_NOISE = 4  # Índice de la clase de ruido

# Regex para parsear el nombre del archivo
_PATRON_NOMBRE = re.compile(
    r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt"
)


# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def cargar_muestra(filepath: str) -> Tuple[torch.Tensor, int, int, int]:
    """
    Carga una muestra individual del dataset NoisyUAV.

    Cada archivo .pt contiene un diccionario con las claves:
        - 'x_iq': Tensor [2, 1048576] con canales I (Re) y Q (Im)
        - 'y': Tensor escalar con el índice de clase (0-6)
        - 'snr': Tensor escalar con el SNR en dB

    Args:
        filepath: Ruta absoluta o relativa al archivo .pt

    Returns:
        Tupla (iq_tensor, sample_id, target, snr):
            - iq_tensor: Tensor [2, 1048576] con canales I (Re) y Q (Im)
            - sample_id: Índice numérico de la muestra (del nombre del archivo)
            - target: Clase (0-6)
            - snr: Relación señal-ruido en dB
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el nombre no sigue el patrón esperado.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    # Parsear metadatos del nombre del archivo
    nombre = os.path.basename(filepath)
    match = _PATRON_NOMBRE.match(nombre)
    if not match:
        raise ValueError(
            f"El nombre '{nombre}' no sigue el patrón esperado: "
            "IQdata_sampleX_targetY_snrZ.pt"
        )

    sample_id = int(match.group(1))

    # Cargar el diccionario del archivo .pt
    data = torch.load(filepath, map_location="cpu", weights_only=False)

    # Extraer datos del diccionario
    iq_tensor = data["x_iq"]           # [2, 1048576]
    target = int(data["y"].item())     # escalar → int
    snr = int(data["snr"].item())      # escalar → int

    return iq_tensor, sample_id, target, snr


def cargar_metadatos_dataset(
    data_dir: str = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee los archivos CSV de estadísticas del dataset.

    Args:
        data_dir: Directorio raíz del dataset.

    Returns:
        Tupla (class_stats, snr_stats):
            - class_stats: DataFrame con columnas ['class', 'count']
            - snr_stats: DataFrame con columnas ['SNR', 'count']
    """
    class_path = os.path.join(data_dir, "class_stats.csv")
    snr_path = os.path.join(data_dir, "SNR_stats.csv")

    if not os.path.exists(class_path):
        raise FileNotFoundError(f"No se encontró class_stats.csv en {data_dir}")
    if not os.path.exists(snr_path):
        raise FileNotFoundError(f"No se encontró SNR_stats.csv en {data_dir}")

    class_stats = pd.read_csv(class_path, index_col=0)
    snr_stats = pd.read_csv(snr_path, index_col=0)

    return class_stats, snr_stats


def obtener_muestras_por_clase(
    data_dir: str = DATA_DIR,
    target: int = 0,
    snr: Optional[int] = None,
    n: int = 5,
) -> List[str]:
    """
    Obtiene rutas de muestras filtradas por clase y, opcionalmente, por SNR.

    Args:
        data_dir: Directorio raíz del dataset.
        target: Índice de clase (0-6).
        snr: Si se especifica, filtra por este nivel de SNR exacto (en dB).
        n: Número máximo de muestras a devolver.

    Returns:
        Lista de rutas absolutas a archivos .pt que cumplen los filtros.
    """
    if snr is not None:
        patron = os.path.join(data_dir, f"IQdata_sample*_target{target}_snr{snr}.pt")
    else:
        patron = os.path.join(data_dir, f"IQdata_sample*_target{target}_snr*.pt")

    archivos = sorted(glob.glob(patron))

    if not archivos:
        nombre_clase = NOMBRES_CLASES.get(target, f"target={target}")
        snr_info = f" con SNR={snr} dB" if snr is not None else ""
        print(f"⚠️ No se encontraron muestras para '{nombre_clase}'{snr_info}")

    return archivos[:n]


def obtener_una_muestra(
    data_dir: str = DATA_DIR,
    target: int = 0,
    snr: Optional[int] = None,
    index: int = 0,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Conveniencia: carga la primera muestra que coincida con los filtros.

    Args:
        data_dir: Directorio raíz del dataset.
        target: Índice de clase (0-6).
        snr: Nivel de SNR en dB (opcional).

    Returns:
        Tupla (iq_tensor, sample_id, target, snr)
    """
    rutas = obtener_muestras_por_clase(data_dir, target, snr, n=index+1)
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron muestras para target={target}"
            + (f", snr={snr}" if snr is not None else "")
        )
    return cargar_muestra(rutas[index])
