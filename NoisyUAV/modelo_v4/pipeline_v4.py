import os, subprocess, sys, argparse

def run_command(command, description):
    print(f"\n>>> {description}...")
    print(f"Comando: {command}")
    res = subprocess.run(command, shell=True)
    if res.returncode != 0:
        print(f"[ERROR] La fase falló con código {res.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Neural Scanner (V4) Dual-Stream")
    parser.add_argument("--skip-dataset", action="store_true", help="Salta la fase de creación del dataset.")
    parser.add_argument("--skip-train", action="store_true", help="Salta la fase de entrenamiento.")
    parser.add_argument("--skip-eval", action="store_true", help="Salta la fase de evaluación.")
    args = parser.parse_args()

    print("==================================================")
    print("       PIPELINE NEURAL SCANNER (V4) DUAL-STREAM   ")
    print("==================================================")
    
    # Ruta base
    base_dir = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4"
    
    # 1. Crear Dataset
    if not args.skip_dataset:
        run_command(f"python {os.path.join(base_dir, 'build_dataset_v4.py')}", 
                    "FASE 1: Extracción Quirúrgica de Ráfagas y Silencios")
    else:
        print("\n>>> Saltando FASE 1: Extracción Quirúrgica (Dataset)...")
    
    # 2. Entrenar Modelo
    # train_v4.py ya incluye lógica para reanudar desde el último checkpoint si existe
    if not args.skip_train:
        run_command(f"python {os.path.join(base_dir, 'train_v4.py')}", 
                    "FASE 2: Entrenamiento con Aumentación AWGN Dinámica")
    else:
        print("\n>>> Saltando FASE 2: Entrenamiento...")
    
    # 3. Evaluación Golden Set
    if not args.skip_eval:
        best_ckpt = os.path.join(base_dir, "checkpoints", "best_model.pth")
        run_command(f"python {os.path.join(base_dir, 'evaluate_v4_cfar.py')} --ckpt {best_ckpt}", 
                    "FASE 3: Evaluación CFAR-Guided (Golden Set)")
    else:
        print("\n>>> Saltando FASE 3: Evaluación...")

    print("\n" + "="*50)
    print("  PIPELINE COMPLETADO CON ÉXITO")
    print("  Figuras en: NoisyUAV/modelo_v4/figures_golden_cfar")
    print("="*50)

if __name__ == "__main__":
    main()