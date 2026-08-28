'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
IEEE benchmarking utilities for the segmentation architectures:
    - Parameter counts.
    - FLOPs / computational complexity (torch.profiler, thop fallback).
    - Inference speed (latency ms/image and throughput FPS).
    - Peak GPU memory usage (falls back to process CPU RAM when no GPU).
    - Full benchmark report combining all of the above (+ training time).
'''

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    '''
    Counts the total and trainable parameters of a model.

    Args:
        model: the PyTorch model to inspect.

    Returns:
        Dict with "total_params" and "trainable_params".
    '''
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def measure_flops(model: nn.Module,
                  input_size: Tuple[int, int, int, int],
                  device: str = "cpu") -> Optional[float]:
    '''
    Estimates the forward-pass FLOPs of a model for one input tensor.

    Tries torch.profiler first (ships with PyTorch, counts conv/matmul
    FLOPs); falls back to `thop` if installed. Returns None if neither
    approach works.

    Args:
        model: the PyTorch model to profile.
        input_size: (B, C, H, W) of the probe tensor (use B=1 for per-image FLOPs).
        device: "cuda" or "cpu".

    Returns:
        Estimated FLOPs (float) for a single forward pass, or None.
    '''
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    # --- Attempt 1: torch.profiler with_flops (no extra dependency) ---
    try:
        from torch.profiler import profile, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with torch.inference_mode():
            with profile(activities=activities, with_flops=True) as prof:
                model(x)
        flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
        if flops > 0:
            return float(flops)
    except Exception:
        pass

    # --- Attempt 2: thop (counts MACs; FLOPs ~= 2 * MACs) ---
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model, inputs=(x,), verbose=False)
        return float(2 * macs)
    except Exception:
        return None


def measure_inference_speed(model: nn.Module,
                            input_size: Tuple[int, int, int, int],
                            device: str = "cpu",
                            warmup: int = 10,
                            iterations: int = 50) -> Dict[str, float]:
    '''
    Measures single-image inference latency and throughput.

    Uses torch.cuda.synchronize() around each timing when on GPU so the
    asynchronous kernel launches are fully accounted for.

    Args:
        model: the PyTorch model to time.
        input_size: (B, C, H, W) of the probe tensor (use B=1 for latency).
        device: "cuda" or "cpu".
        warmup: untimed forward passes to stabilize clocks/caches.
        iterations: timed forward passes to average over.

    Returns:
        Dict with "latency_ms_mean", "latency_ms_std", and "fps".
    '''
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    timings_ms = []
    with torch.inference_mode():
        # Warmup passes (not timed)
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed passes
        for _ in range(iterations):
            start = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    timings = torch.tensor(timings_ms)
    mean_ms = float(timings.mean())
    std_ms = float(timings.std())
    batch = input_size[0]
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "fps": 1000.0 * batch / mean_ms,
    }


def measure_memory(model: nn.Module,
                   input_size: Tuple[int, int, int, int],
                   device: str = "cpu") -> Dict[str, float]:
    '''
    Measures peak memory of one forward pass.

    On GPU: peak allocated CUDA memory (torch.cuda.max_memory_allocated).
    On CPU: process RSS delta measured with psutil (approximate, reported
    with "memory_type": "cpu_rss").

    Args:
        model: the PyTorch model to measure.
        input_size: (B, C, H, W) of the probe tensor.
        device: "cuda" or "cpu".

    Returns:
        Dict with "peak_memory_mb" and "memory_type" ("gpu" or "cpu_rss").
    '''
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return {"peak_memory_mb": float(peak_mb), "memory_type": "gpu"}

    # CPU fallback: resident-set-size delta around the forward pass
    try:
        import psutil
        process = psutil.Process()
        before_mb = process.memory_info().rss / (1024 ** 2)
        with torch.inference_mode():
            model(x)
        after_mb = process.memory_info().rss / (1024 ** 2)
        return {"peak_memory_mb": float(max(0.0, after_mb - before_mb)),
                "memory_type": "cpu_rss"}
    except ImportError:
        return {"peak_memory_mb": float("nan"), "memory_type": "unavailable"}


def benchmark_model(model: nn.Module,
                    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
                    device: str = "cpu",
                    warmup: int = 10,
                    iterations: int = 50,
                    arch_name: str = "",
                    training_time_s: Optional[float] = None,
                    epochs: Optional[int] = None) -> Dict:
    '''
    Runs the full efficiency benchmark on a model and returns a report dict:
    parameters, FLOPs (GFLOPs), inference latency/FPS, peak memory, and
    (optionally) total / per-epoch training time.

    Args:
        model: the PyTorch model to benchmark.
        input_size: (B, C, H, W) probe size (B=1 for per-image numbers).
        device: "cuda" or "cpu".
        warmup: untimed warmup iterations for the speed test.
        iterations: timed iterations for the speed test.
        arch_name: architecture label stored in the report.
        training_time_s: total training wall-clock seconds (optional).
        epochs: number of epochs trained, to derive seconds/epoch (optional).

    Returns:
        Dict with every benchmark metric (JSON-serializable).
    '''
    report: Dict = {
        "arch": arch_name or model.__class__.__name__,
        "device": device,
        "input_size": list(input_size),
    }

    report.update(count_parameters(model))
    report["params_million"] = report["total_params"] / 1e6

    # Memory is measured first so earlier probes don't mask the CPU RSS delta
    report.update(measure_memory(model, input_size, device))

    flops = measure_flops(model, input_size, device)
    report["flops"] = flops
    report["gflops"] = (flops / 1e9) if flops is not None else None

    report.update(measure_inference_speed(model, input_size, device,
                                          warmup=warmup, iterations=iterations))

    if training_time_s is not None:
        report["training_time_s"] = float(training_time_s)
        if epochs:
            report["training_time_per_epoch_s"] = float(training_time_s) / epochs

    return report


def print_benchmark(report: Dict) -> None:
    '''
    Pretty-prints a benchmark report produced by benchmark_model().

    Args:
        report: the dict returned by benchmark_model().
    '''
    print("=" * 62)
    print(f" Benchmark report: {report['arch']}  (device: {report['device']})")
    print("=" * 62)
    print(f" Input size            : {report['input_size']}")
    print(f" Total parameters      : {report['total_params']:,} "
          f"({report['params_million']:.2f}M)")
    if report.get("gflops") is not None:
        print(f" FLOPs (fwd pass)      : {report['gflops']:.2f} GFLOPs")
    else:
        print(" FLOPs (fwd pass)      : unavailable")
    print(f" Latency (mean +/- std): {report['latency_ms_mean']:.2f} "
          f"+/- {report['latency_ms_std']:.2f} ms/image")
    print(f" Throughput            : {report['fps']:.2f} FPS")
    print(f" Peak memory           : {report['peak_memory_mb']:.1f} MB "
          f"({report['memory_type']})")
    if "training_time_s" in report:
        print(f" Training time (total) : {report['training_time_s']:.1f} s")
        if "training_time_per_epoch_s" in report:
            print(f" Training time / epoch : {report['training_time_per_epoch_s']:.1f} s")
    print("=" * 62)


def save_benchmark(report: Dict, target_dir: str, file_name: str) -> Path:
    '''
    Saves a benchmark report dict as JSON.

    Args:
        report: the dict returned by benchmark_model().
        target_dir: directory to save into (created if missing).
        file_name: JSON file name (e.g. "unet_resnet18_binary.json").

    Returns:
        Path of the saved JSON file.
    '''
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    out_file = target_path / file_name
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Benchmark report saved to: {out_file}")
    return out_file
