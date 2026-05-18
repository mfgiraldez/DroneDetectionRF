import subprocess
import time
import sys
import os

# Rutas
PYTHON_EXE = r"C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe"
PIPELINE_SCRIPT = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\pipeline.py"
LOG_FILE = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\pipeline_output.log"
SUPERVISOR_LOG = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v5\supervisor.log"

def log_msg(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [SUPERVISOR] {msg}"
    print(formatted_msg)
    with open(SUPERVISOR_LOG, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

def main():
    log_msg("Iniciando supervisión del pipeline V5...")
    
    attempt = 1
    while True:
        log_msg(f"Lanzando Intento #{attempt}...")
        
        # Abrimos el log en modo append para el pipeline
        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                [PYTHON_EXE, PIPELINE_SCRIPT],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(PIPELINE_SCRIPT)
            )
            
            # Esperamos a que termine
            exit_code = process.wait()
            
        if exit_code == 0:
            log_msg("¡Pipeline completado con éxito! Finalizando supervisión.")
            break
        else:
            log_msg(f"El pipeline falló con código {exit_code}. Reiniciando en 10 segundos...")
            attempt += 1
            time.sleep(10)

if __name__ == "__main__":
    main()
