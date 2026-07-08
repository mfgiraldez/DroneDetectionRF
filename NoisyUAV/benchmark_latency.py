import os
import sys
import torch
import time
import json
import numpy as np

# Asegurar importaciones
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar modelos
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN
from NoisyUAV.modelo_v2_1_dual.model import DualStreamCVCNN

def benchmark_model(name, model, input_shape, is_dual=False, num_warmup=50, num_iters=1000, device='cpu', sliding_window_steps=1):
    model = model.to(device)
    model.eval()

    # Generar datos dummy
    if is_dual:
        x_iq = torch.randn(input_shape).to(device)
        phys = torch.randn((input_shape[0], 3)).to(device)
        inputs = (x_iq, phys)
    else:
        x_iq = torch.randn(input_shape).to(device)
        phys = torch.randn((input_shape[0], 8)).to(device)
        inputs = (x_iq, phys)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            for _ in range(sliding_window_steps):
                model(*inputs)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                for _ in range(sliding_window_steps):
                    model(*inputs)
                    
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                for _ in range(sliding_window_steps):
                    model(*inputs)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    
    print(f"[{device.type.upper()}] {name}: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
    return {"mean_ms": mean_ms, "std_ms": std_ms}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando latencia en: {device}")
    
    results = {}
    
    # 1. SingleStream-CVCNN
    print("\nEvaluando SingleStream-CVCNN...")
    model_single = BurstCVCNN()
    res1 = benchmark_model("SingleStream-CVCNN", model_single, (1, 2, 131072), is_dual=False, device=device)
    results["SingleStream"] = res1
    
    # 2. DualStream-Attention (V2.1)
    print("\nEvaluando DualStream-Attention (V2.1)...")
    model_dual = DualStreamCVCNN()
    res2 = benchmark_model("DualStream-Attention", model_dual, (1, 2, 131072), is_dual=True, device=device)
    results["DualStream_V2_1"] = res2

    # 3. DualStream-SlidingWindow (V2.2)
    # Procesa 16 ventanas secuencialmente
    print("\nEvaluando DualStream-SlidingWindow (V2.2)...")
    res3 = benchmark_model("DualStream-SlidingWindow", model_dual, (1, 2, 131072), is_dual=True, device=device, sliding_window_steps=16)
    results["DualStream_V2_2"] = res3

    # Guardar a disco
    out_path = os.path.join(project_root, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResultados guardados en: {out_path}")

if __name__ == "__main__":
    main()
