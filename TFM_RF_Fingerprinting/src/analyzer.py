"""
Módulo para el procesamiento y análisis de señales I/Q en banda base.
Adaptado a la estructura fragmentada (1 segundo) del dataset RFUAV.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IQSignalAnalyzer:
    def __init__(self, iq_file_path: str, default_sample_rate: float = 100e6):
        """
        Inicializa el analizador a partir de un archivo .iq específico y busca su XML asociado
        implementando un mecanismo de tolerancia a fallos en la nomenclatura.
        """
        import glob
        
        self.iq_path = os.path.abspath(iq_file_path)
        self.folder_path = os.path.dirname(self.iq_path)
        
        if not os.path.exists(self.iq_path):
            raise FileNotFoundError(f"No se encontró el archivo IQ: {self.iq_path}")

        self.metadatos = {
            "Drone": "Desconocido",
            "CenterFrequency": 2400e6,
            "SampleRate": default_sample_rate,
            "SNR": "Desconocido"
        }
        
        # 1. Intento principal: Búsqueda estricta por prefijo (ej. pack2_1-2s.iq -> pack2.xml)
        filename = os.path.basename(self.iq_path)
        pack_prefix = filename.split('_')[0]
        exact_xml_path = os.path.join(self.folder_path, f"{pack_prefix}.xml")
        
        if os.path.exists(exact_xml_path):
            self.xml_path = exact_xml_path
        else:
            # 2. Plan de Respaldo (Fallback): Buscar CUALQUIER archivo .xml en la carpeta
            xml_files = glob.glob(os.path.join(self.folder_path, "*.xml"))
            if xml_files:
                self.xml_path = xml_files[0]  # Tomamos el primer XML disponible
                logging.info(f"Nomenclatura inconsistente detectada. Se usará {os.path.basename(self.xml_path)} para el archivo {filename}.")
            else:
                self.xml_path = None
                
        # Parseo de metadatos si se encontró algún XML
        if self.xml_path:
            self._parse_xml_metadata(self.xml_path)
        else:
            logging.warning(f"Ausencia total de archivos XML en la carpeta {self.folder_path}. Usando valores por defecto.")
            
        self.fs = self.metadatos["SampleRate"]

    def _parse_xml_metadata(self, xml_path: str) -> None:
        """Extrae los parámetros físicos del SDR desde el archivo XML."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            node_drone = root.find('Drone')
            if node_drone is not None: self.metadatos["Drone"] = node_drone.text
            
            node_cf = root.find('CenterFrequency')
            if node_cf is not None: self.metadatos["CenterFrequency"] = float(node_cf.text)
            
            node_sr = root.find('SampleRate')
            if node_sr is not None: self.metadatos["SampleRate"] = float(node_sr.text)
            
            node_snr = root.find('ReferenceSNRLevel')
            if node_snr is not None: self.metadatos["SNR"] = node_snr.text
            
            logging.info(f"Metadatos cargados del XML: UAV={self.metadatos['Drone']} | SNR={self.metadatos['SNR']} | Fs={self.metadatos['SampleRate']/1e6} MSps")
        except Exception as e:
            logging.error(f"Error parseando {xml_path}: {str(e)}.")

    def read_binary_chunk(self, duration: float = 0.1, offset_sec: float = 0.0) -> np.ndarray:
        """
        Lee un fragmento del archivo binario.
        IMPORTANTE: Como los archivos ya están troceados en 1 segundo, (offset_sec + duration) no debe exceder 1.0.
        """
        if offset_sec + duration > 1.0:
            logging.warning("¡Cuidado! Estás intentando leer más allá de 1 segundo y estos archivos están limitados a 1s de grabación.")

        samples_to_read = int(self.fs * duration)
        offset_bytes = int(self.fs * offset_sec * 8) 

        logging.info(f"Extrayendo ráfaga: {duration}s desde el offset {offset_sec}s del archivo {os.path.basename(self.iq_path)}...")
        
        with open(self.iq_path, 'rb') as f:
            f.seek(offset_bytes)
            raw_data = np.fromfile(f, dtype=np.float32, count=samples_to_read * 2)

        complex_signal = raw_data[0::2] + 1j * raw_data[1::2]
        return complex_signal

    def generate_analysis_panel(self, iq_data: np.ndarray) -> None:
            """Genera el panel de análisis en 3 dominios: Tiempo, Tiempo-Frecuencia y Frecuencia."""
            import matplotlib.ticker as ticker
            import matplotlib.gridspec as gridspec
            
            titulo_dinamico = f"Firma RF UAV: {self.metadatos['Drone']} (SNR: {self.metadatos['SNR']} | Fs: {self.fs/1e6} MSps)"
            time_sec = np.arange(len(iq_data)) / self.fs

            # Configuramos el Grid: 3 filas y 2 columnas
            # Le damos más altura al espectrograma (Fila 1) porque es la gráfica principal
            fig = plt.figure(figsize=(14, 11))
            gs = gridspec.GridSpec(3, 2, width_ratios=[50, 1], height_ratios=[1, 1.5, 1], wspace=0.02, hspace=0.3)
            fig.suptitle(titulo_dinamico, fontsize=15, fontweight='bold')

            # ==========================================
            # 1. DOMINIO DEL TIEMPO (Fila 0)
            # ==========================================
            ax_time = fig.add_subplot(gs[0, 0])
            magnitude = np.abs(iq_data)
            max_idx, min_idx = np.argmax(magnitude), np.argmin(magnitude)
            
            ax_time.plot(time_sec, magnitude, color='#1f77b4', linewidth=0.5, label='Envolvente I/Q')
            ax_time.scatter(time_sec[max_idx], magnitude[max_idx], color='red', zorder=5, 
                            label=f'Pico Máx a {time_sec[max_idx]*1000:.2f} ms')
            ax_time.scatter(time_sec[min_idx], magnitude[min_idx], color='green', zorder=5, 
                            label=f'Pico Mín a {time_sec[min_idx]*1000:.2f} ms')
            
            ax_time.set_ylabel("Amplitud Lineal")
            ax_time.set_title("1. Envolvente Temporal (Búsqueda de Transitorios)")
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(loc='upper right', fontsize=9)
            ax_time.set_xlim([0, time_sec[-1]])
            
            # Formateamos a ms pero ocultamos las etiquetas porque las pondrá el espectrograma
            ax_time.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*1000:g}"))
            ax_time.tick_params(labelbottom=False)

            # ==========================================
            # 2. DOMINIO TIEMPO-FRECUENCIA (STFT / Fila 1)
            # ==========================================
            # Comparte eje X con ax_time
            ax_freq = fig.add_subplot(gs[1, 0], sharex=ax_time) 
            NFFT = 1024
            Pxx, freqs, bins, im = ax_freq.specgram(
                iq_data, NFFT=NFFT, Fs=self.fs, noverlap=NFFT // 2, cmap='viridis', scale='dB'
            )
            
            ax_freq.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*1000:g}"))
            ax_freq.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y/1e6:g}"))
            
            ax_freq.set_xlabel("Tiempo (ms)")
            ax_freq.set_ylabel("Frecuencia (MHz)")
            ax_freq.set_title("2. Espectrograma STFT (Evolución de saltos de frecuencia)")

            # --- BARRA DE COLOR ---
            cax = fig.add_subplot(gs[1, 1])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label('Densidad Espectral (dB/Hz)', fontsize=9)

            # ==========================================
            # 3. DOMINIO DE LA FRECUENCIA (PSD / Fila 2)
            # ==========================================
            # Esta gráfica tiene su propio eje X (Frecuencia)
            ax_psd = fig.add_subplot(gs[2, 0])
            ax_psd.psd(iq_data, NFFT=1024, Fs=self.fs, color='#9467bd', linewidth=1.5)
            
            ax_psd.set_ylabel("Potencia (dB/Hz)")
            ax_psd.set_xlabel("Frecuencia (MHz)")
            ax_psd.set_title("3. Densidad Espectral de Potencia (Método de Welch)")
            ax_psd.grid(True, alpha=0.3)
            ax_psd.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:g}"))

            plt.tight_layout()
            plt.show()