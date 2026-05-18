import os
import re

directory = r'c:\repos\DroneDetectionRF\NoisyUAV'
moves = {
    'cargador': 'dataset.cargador',
    'dataset': 'dataset.dataset',
    'dataset_stage2': 'dataset.dataset_stage2',
    'build_burst_dataset': 'dataset.build_burst_dataset',
    'extractor_dataset_final': 'dataset.extractor_dataset_final',
    'preparar_splits': 'dataset.preparar_splits',
    'filtro_etiquetas_stage1': 'dataset.filtro_etiquetas_stage1',
    'detector_entropia': 'dsp_rf.detector_entropia',
    'detector_masivo_stage1': 'dsp_rf.detector_masivo_stage1',
    'evaluar_entropia': 'dsp_rf.evaluar_entropia',
    'physical_features': 'dsp_rf.physical_features',
    'prueba_ShannonEntropy': 'dsp_rf.prueba_ShannonEntropy',
    'entrenar_stage2': 'entrenamiento.entrenar_stage2',
    'entrenar_stage2_norm': 'entrenamiento.entrenar_stage2_norm',
    'evaluacion_analisis_modelo': 'evaluacion.evaluacion_analisis_modelo',
    'plot_styles': 'visualizacion.plot_styles',
    'visualizacion': 'visualizacion.visualizacion',
}

files_updated = 0
for root, dirs, files in os.walk(directory):
    if 'old_code' in root or '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.ipynb'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue
            
            new_content = content
            for old, new in moves.items():
                # Reemplazamos 'funciones.old' por 'funciones.new' asegurandonos de que es palabra completa
                pattern = r'funciones\.' + old + r'\b'
                replacement = r'funciones.' + new
                new_content = re.sub(pattern, replacement, new_content)
            
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated: {path}")
                files_updated += 1

print(f"Total files updated: {files_updated}")
