"""
Módulo para la descarga automatizada de subconjuntos del dataset RFUAV.
"""

import os
import logging
from huggingface_hub import hf_hub_download

# Configuración del logger para salida profesional en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RFUAVDownloader:
    def __init__(self, target_dir: str = "../data/raw"):
        """
        Inicializa el gestor de descargas.
        
        Args:
            target_dir (str): Directorio de destino para los archivos crudos.
        """
        self.repo_id = "kitofrank/RFUAV"
        self.repo_type = "dataset"
        self.target_dir = os.path.abspath(target_dir)
        
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            logging.info(f"Directorio creado: {self.target_dir}")

    def download_drone_data(self, filename: str) -> str:
        """
        Descarga un archivo específico desde el repositorio de Hugging Face.
        
        Args:
            filename (str): Nombre exacto del archivo en el repositorio (ej. 'DJI MINI3.rar').
            
        Returns:
            str: Ruta local absoluta del archivo descargado.
        """
        logging.info(f"Iniciando descarga de {filename} desde {self.repo_id}...")
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                filename=filename,
                local_dir=self.target_dir,
                local_dir_use_symlinks=False
            )
            logging.info(f"Descarga completada con éxito. Archivo en: {local_path}")
            return local_path
        except Exception as e:
            logging.error(f"Error durante la descarga de {filename}: {str(e)}")
            raise

if __name__ == "__main__":
    # Ejecución de prueba para la reunión
    downloader = RFUAVDownloader()
    
    # Seleccionamos ~7.7 GB de datos para la demostración
    archivos_objetivo = [
        "YUNZHUO H16.rar",
        "DJI MINI3.rar"
    ]
    
    for archivo in archivos_objetivo:
        downloader.download_drone_data(archivo)