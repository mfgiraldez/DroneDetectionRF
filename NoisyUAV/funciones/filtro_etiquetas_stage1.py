import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import os

def run_analysis():
    # 1. Cargar datos
    parquet_path = r"C:\TFM_data\NoisyUAV\stage1\metadata_stage1.parquet"
    if not os.path.exists(parquet_path):
         print(f"No se encontró el archivo: {parquet_path}")
         return

    df = pd.read_parquet(parquet_path)
    
    # 2. Eliminar los archivos que no tuvieron bursts (burst_idx == -1)
    df_bursts = df[df["burst_idx"] != -1].copy()
    
    print("=== ESTADÍSTICAS GLOBALES ===")
    print(f"Detecciones totales: {len(df_bursts)}")
    print(f"Detecciones en archivos Ruido (Clase 4): {len(df_bursts[df_bursts['target'] == 4])}")
    print(f"Detecciones en archivos Drones (Clase != 4): {len(df_bursts[df_bursts['target'] != 4])}")

    # 3. Features físicos para el modelo
    # IMPORTANTE: NO usamos z_peak ni SNR. Queremos aislar por la pura morfología 
    # de onda del protocolo (duración y ancho de banda ocupado en bins).
    features = ["dur_ms", "n_active_mean"]
    
    # 4. Entrenar el modelo de Ruido (Isolation Forest)
    # Entrenamos EXCLUSIVAMENTE con Target 4 (Puro Background Noise + BT + WiFi leaks)
    df_noise = df_bursts[df_bursts['target'] == 4]
    X_train = df_noise[features].to_numpy()
    
    # contamination=0.01 asume que el 1% del ruido de Target 4 podría ser ruido anómalo 
    # (ajuste fino del envolvente)
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    
    # 5. Predecir en TODO EL DATASET (Noise y Drones)
    X_all = df_bursts[features].to_numpy()
    
    # El Isolation Forest devuelve: 1 (Normal / Background) y -1 (Anomalía / Drone)
    preds = iso_forest.predict(X_all)
    
    df_bursts["is_anomaly"] = (preds == -1)
    
    # Asignar Etiquetas Limpias:
    # CLASE 0 (No-Dron): Si la señal es Background (1), sin importar el archivo de donde vino.
    # CLASE 1 (Dron): Si la señal es Anormal (-1) Y el archivo de origen es un Dron (target != 4).
    # Descartamos las Anomalías en Target 4 (no las metemos a Clase 1 porque podrían ser glitches)
    
    def asignar_clase_stage2(row):
        if not row["is_anomaly"]: 
            return 0  # Es el fondo (Bluetooth/WiFi)
        else:
            if row["target"] == 4:
                return -1 # Glitch anómalo en el ruido (descartar para estar seguros)
            else:
                return 1  # Dron verdadero
                
    df_bursts["label_stage2"] = df_bursts.apply(asignar_clase_stage2, axis=1)

    print("\n=== RESULTADO DE LA LIMPIEZA ===")
    print(f"Muestras asignadas a Clase 0 (Background Noise validado): {len(df_bursts[df_bursts['label_stage2'] == 0])}")
    print(f"Muestras asignadas a Clase 1 (Ráfagas Drone validadas): {len(df_bursts[df_bursts['label_stage2'] == 1])}")
    print(f"Muestras descartadas (inconsistencias): {len(df_bursts[df_bursts['label_stage2'] == -1])}")

    # 6. Visualización Interactiva HTML
    # Tomamos una muestra para no saturar plotly
    df_plot = df_bursts.sample(n=min(10000, len(df_bursts)), random_state=42)
    
    fig = px.scatter(
        df_plot, 
        x="dur_ms", 
        y="n_active_mean", 
        color="label_stage2",
        hover_data=["target", "snr", "z_peak"],
        title="Label Dissonance Resolver (Isolation Forest)",
        color_continuous_scale="Viridis",
        labels={"label_stage2": "Stage 2 Label", "dur_ms": "Duración de Ráfaga (ms)", "n_active_mean": "Bins Activos Promedio"}
    )
    
    out_html = r"C:\TFM_data\NoisyUAV\stage1\analisis_etiquetas.html"
    fig.write_html(out_html)
    print(f"\nGráfico interactivo guardado en: {out_html}")
    
    # 7. Guardar el dataframe limpio final
    out_parquet = r"C:\TFM_data\NoisyUAV\stage1\metadata_stage1_filtered.parquet"
    df_final = df_bursts[df_bursts["label_stage2"] != -1].copy()
    df_final.to_parquet(out_parquet, index=False)
    print(f"Metadata purgada para Stage 2 guardada en: {out_parquet}")

if __name__ == "__main__":
    run_analysis()
