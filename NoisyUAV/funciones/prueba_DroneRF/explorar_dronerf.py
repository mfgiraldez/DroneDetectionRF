import pandas as pd
import numpy as np

def test_load():
    csv_path = r"C:\TFM_data\DroneRF\DroneRF\AR drone\RF Data_10100_H\RF Data_10100_H\10100H_0.csv"
    print(f"Cargando {csv_path}...")
    
    # Read the first chunk to see what structural format it has
    try:
        df = pd.read_csv(csv_path, header=None, nrows=100)
        print("Dimensión de los primeros 100 registros:")
        print(df.shape)
        print("Primeras filas (valores y NaNs si hay):")
        print(df.head())
    except Exception as e:
        print(f"Error cargando CSV: {e}")

if __name__ == '__main__':
    test_load()
