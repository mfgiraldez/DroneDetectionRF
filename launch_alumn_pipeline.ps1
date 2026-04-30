Write-Host "⏳ Esperando a que termine el entrenamiento del Teacher (run_teacher_experiment.py)..." -ForegroundColor Cyan

while ($true) {
    # Buscamos procesos de python que estén ejecutando el script del teacher
    $proc = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*run_teacher_experiment.py*" }
    if (-not $proc) {
        Write-Host "✅ El Teacher ha terminado su proceso o no se encontró ejecutándose." -ForegroundColor Green
        break
    }
    # Esperamos 60 segundos antes de volver a comprobar
    Start-Sleep -Seconds 60
}

Write-Host "--------------------------------------------------------" -ForegroundColor White
Write-Host "🚀 PASO 1: Generando dataset Alumno (Pseudo-Labeling V9)" -ForegroundColor Yellow
python NoisyUAV\curriculum_alumn_v1\build_alumn_dataset.py

Write-Host "--------------------------------------------------------" -ForegroundColor White
Write-Host "🚀 PASO 2: Entrenando Alumno (Fine-Tuning)" -ForegroundColor Yellow
python NoisyUAV\curriculum_alumn_v1\run_alumn_experiment.py

Write-Host "--------------------------------------------------------" -ForegroundColor White
Write-Host "🏁 Pipeline Alumno completado con éxito." -ForegroundColor Green
