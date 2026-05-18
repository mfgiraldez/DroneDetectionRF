"""
launch_alumn.py
===============
Orquestador Python del pipeline completo del Alumno.
Ejecuta en secuencia:
  1. calibrate_drone_durations.py  (si no existe el JSON)
  2. build_alumn_dataset.py
  3. run_alumn_experiment.py

Uso:
  cd c:\repos\DroneDetectionRF
  python NoisyUAV\curriculum_alumn_v1\launch_alumn.py
"""
import sys, os, subprocess, time
from pathlib import Path

ROOT = Path(r"c:\repos\DroneDetectionRF")
ALUMN_DIR = ROOT / "NoisyUAV" / "curriculum_alumn_v1"
DURATION_JSON = ALUMN_DIR / "drone_duration_ref.json"

STEPS = [
    {
        "name":   "Calibración de duraciones de dron",
        "script": ALUMN_DIR / "calibrate_drone_durations.py",
        "skip_if_exists": DURATION_JSON,
        "skip_msg": "drone_duration_ref.json ya existe — saltando calibración.",
    },
    {
        "name":   "Generación del dataset Alumno (Pseudo-Labeling V9)",
        "script": ALUMN_DIR / "build_alumn_dataset.py",
        "skip_if_exists": None,
    },
    {
        "name":   "Entrenamiento del Alumno (Fine-Tuning)",
        "script": ALUMN_DIR / "run_alumn_experiment.py",
        "skip_if_exists": None,
    },
]

SEP = "─" * 60

def run_step(step):
    name   = step["name"]
    script = step["script"]
    skip   = step.get("skip_if_exists")

    print(f"\n{SEP}")
    print(f"🚀  {name}")
    print(SEP)

    if skip and skip.exists():
        print(f"   ⏭️  {step['skip_msg']}")
        return True

    if not script.exists():
        print(f"   ❌ Script no encontrado: {script}")
        return False

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n   ❌ FALLO en '{name}' (código {result.returncode})")
        print(f"   Pipeline interrumpido.")
        return False

    print(f"\n   ✅ '{name}' completado en {elapsed/60:.1f} min.")
    return True


def main():
    print("=" * 60)
    print("  PIPELINE ALUMNO — Calibración + Dataset + Entrenamiento")
    print("=" * 60)

    for step in STEPS:
        ok = run_step(step)
        if not ok:
            sys.exit(1)

    print(f"\n{SEP}")
    print("🏁  Pipeline Alumno completado con éxito.")
    print(SEP)


if __name__ == '__main__':
    main()
