import time
import os
import psutil
import subprocess
from pathlib import Path

import urllib.request
import json

# ================================================================
# CONFIGURACIÓN DE NOTIFICACIONES
# ================================================================
# 1. Notificaciones Push instantáneas (SIN configuración)
# Abre esta URL en el móvil para ver los avisos en vivo:
NTFY_TOPIC = "mfgiraldez_tfm_alerts"  # https://ntfy.sh/mfgiraldez_tfm_alerts

# 2. Correo electrónico (Opcional, requiere Contraseña de Aplicación de Gmail)
EMAIL_SENDER = "mfgiraldez@gmail.com"
EMAIL_PASSWORD = "eput htat vuxu mjsq"  # No es la contraseña normal
EMAIL_RECEIVER = "mfgiraldez@gmail.com"

def send_alert(title, message):
    print(f"\n[ALERTA] {title}: {message}")
    
    # 1. Enviar vía ntfy.sh (Llega instantáneo al móvil si abres la web)
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", 
                                     data=message.encode('utf-8'), 
                                     headers={"Title": title.encode('utf-8')})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print("  (Aviso: No se pudo enviar notificación ntfy)")
        
    # 2. Enviar vía Email (Solo si se han puesto las credenciales)
    if "TU_CORREO" not in EMAIL_SENDER:
        import smtplib
        from email.mime.text import MIMEText
        try:
            msg = MIMEText(message)
            msg['Subject'] = title
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECEIVER
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
                smtp.send_message(msg)
        except Exception as e:
            print(f"  (Aviso: No se pudo enviar email: {e})")

# ================================================================

# Configuración de Archivos
FIG_PATH = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_burst_v2\figures\training_curves_ieee.png")
CHECKPOINT_PATH = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_burst_v2\checkpoints\best_model.pt")

print("================================================================")
print("VIGILANTE AUTOMÁTICO INICIADO")
print("================================================================")
print("Esperando a que la época actual (Epoch 5) termine de validar...")

# Obtenemos el timestamp actual de la gráfica.
# La gráfica se guarda exactamente al final de cada época.
try:
    initial_mtime = os.path.getmtime(FIG_PATH)
except FileNotFoundError:
    print(f"AVISO: No se ha encontrado la gráfica todavía. Esperando a que se cree...")
    initial_mtime = 0

# 1. Esperar a que la época termine (detectado por el guardado de la nueva gráfica)
while True:
    try:
        current_mtime = os.path.getmtime(FIG_PATH)
        if current_mtime > initial_mtime and initial_mtime != 0:
            print(f"\n[+] ¡Época completada! La gráfica se ha actualizado.")
            time.sleep(5)
            # Leer resultados si existe history.json
            res_msg = "Resultados desconocidos."
            hist_path = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_burst_v2\history.json")
            if hist_path.exists():
                with open(hist_path, "r") as f:
                    history = json.load(f)
                    if len(history.get('val_loss', [])) > 0:
                        res_msg = f"Val Loss: {history['val_loss'][-1]:.4f} | Val F1: {history['val_f1'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}"
            
            send_alert("Hito 1: Validación Epoch 5 Terminada", f"El proceso original ha guardado la época 5.\n{res_msg}")
            break
        elif initial_mtime == 0 and current_mtime > 0:
            # Se acaba de crear la primera gráfica
            print(f"\n[+] ¡Época completada! La gráfica se ha creado por primera vez.")
            time.sleep(5)
            send_alert("Hito 1: Validación Epoch 5 Terminada", "El proceso original ha guardado la primera época con éxito.")
            break
    except FileNotFoundError:
        pass
        
    time.sleep(10) # Comprobamos cada 10 segundos para no saturar CPU

print(f"[i] Último modelo guardado (best_model.pt) actualizado a las: {time.ctime(os.path.getmtime(CHECKPOINT_PATH)) if CHECKPOINT_PATH.exists() else 'N/A'}")

# 2. Matar el proceso original (run_burst_experiment.py)
print("\n[*] Cerrando run_burst_experiment.py de forma segura...")
killed = False
for proc in psutil.process_iter(['pid', 'cmdline']):
    try:
        if proc.cmdline() and 'run_burst_experiment.py' in ' '.join(proc.cmdline()):
            print(f"    Matando proceso {proc.pid}...")
            proc.kill()
            killed = True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not killed:
    print("    AVISO: No se encontró el proceso run_burst_experiment.py en ejecución.")

# 3. Iniciar la Pipeline del Teacher
print("\n================================================================")
print("INICIANDO PIPELINE TEACHER (FASE 1)")
print("================================================================")

send_alert("Hito 2: Iniciando Creación Dataset", "Comienza la extracción de bursts con umbral dinámico para el Teacher.")
subprocess.run(["python", r"curriculum_teacher_v1\build_teacher_dataset.py"], cwd=r"c:\repos\DroneDetectionRF\NoisyUAV")
send_alert("Hito 3: Dataset Creado", "El proceso de creación del dataset del Teacher ha finalizado.")

print("\n[+] Paso 2/2: Iniciando Entrenamiento del Teacher...")
send_alert("Hito 4: Iniciando Entrenamiento", "Arranca el entrenamiento de la red Teacher (run_teacher_experiment.py).")
subprocess.run(["python", "-u", r"curriculum_teacher_v1\run_teacher_experiment.py"], cwd=r"c:\repos\DroneDetectionRF\NoisyUAV")
send_alert("Hito 5: Entrenamiento Finalizado", "La pipeline del Teacher ha terminado por completo. ¡Todo guardado con éxito!")

print("\n================================================================")
print("TODAS LAS TAREAS AUTOMATIZADAS HAN FINALIZADO CON ÉXITO")
print("================================================================")
