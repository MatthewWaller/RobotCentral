#!/usr/bin/env python3

"""
Simple benchmark script to compare GR00T inference performance
between the original server-client approach and the optimized direct approach.
"""

import time
import numpy as np
from pathlib import Path
import torch
import sys
import os

# Add the isaac-gr00t directory to the path
sys.path.append(str(Path(__file__).parent / "isaac-gr00t"))

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.service import ExternalRobotInferenceClient
import subprocess
import threading
import signal


def create_dummy_observation():
    """Create a dummy observation similar to what a robot would send"""
    return {
        "video.webcam": np.zeros((1, 720, 1280, 3), dtype=np.uint8),  # Use expected resolution
        "state.single_arm": np.zeros((1, 5), dtype=np.float64),
        "state.gripper": np.zeros((1, 1), dtype=np.float64),
        "annotation.human.task_description": ["Place the block in the bowl"],
    }


def benchmark_direct_inference(model_path, denoising_steps=2, num_iterations=10):
    """Benchmark direct inference approach"""
    print(f"🚀 Benchmarking DIRECT inference ({denoising_steps} denoising steps)")
    
    # Load data config
    data_config = DATA_CONFIG_MAP["so100"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Initialize policy directly
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="new_embodiment",
        denoising_steps=denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Warm up
    dummy_obs = create_dummy_observation()
    print("Warming up model...")
    _ = policy.get_action(dummy_obs)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        _ = policy.get_action(dummy_obs)
        inference_time = time.time() - start_time
        times.append(inference_time)
        print(f"  Iteration {i+1}: {inference_time:.3f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"📊 Direct inference - Average: {avg_time:.3f}s (±{std_time:.3f}s)")
    return avg_time, std_time


def benchmark_server_client(model_path, denoising_steps=4, num_iterations=10):
    """Benchmark server-client approach"""
    print(f"🐌 Benchmarking SERVER-CLIENT inference ({denoising_steps} denoising steps)")
    
    # Start server in background
    server_cmd = [
        "python", "isaac-gr00t/scripts/inference_service.py", 
        "--server",
        "--model_path", model_path,
        "--embodiment-tag", "new_embodiment",
        "--data-config", "so100",
        "--denoising-steps", str(denoising_steps),
        "--port", "5556"  # Use different port to avoid conflicts
    ]
    
    print("Starting server...")
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(10)
    
    try:
        # Initialize client
        client = ExternalRobotInferenceClient(host="localhost", port=5556)
        
        # Test connection
        if not client.ping():
            print("❌ Server not responding, skipping server-client benchmark")
            return None, None
        
        # Prepare observation
        dummy_obs = create_dummy_observation()
        
        # Warm up
        print("Warming up server...")
        _ = client.get_action(dummy_obs)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            _ = client.get_action(dummy_obs)
            inference_time = time.time() - start_time
            times.append(inference_time)
            print(f"  Iteration {i+1}: {inference_time:.3f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"📊 Server-client - Average: {avg_time:.3f}s (±{std_time:.3f}s)")
        
        return avg_time, std_time
        
    finally:
        # Clean up server
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()


def main():
    """Run the benchmark comparison"""
    print("=== GR00T Inference Performance Benchmark ===\n")
    
    # Configuration
    model_path = "./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned"
    num_iterations = 5
    
    # Check if model path exists
    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        print("Please update the model_path variable in this script")
        return
    
    print(f"Model path: {model_path}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Benchmark direct inference (optimized)
    direct_avg, direct_std = benchmark_direct_inference(
        model_path, denoising_steps=2, num_iterations=num_iterations
    )
    
    print("\n" + "=" * 60)
    
    # Benchmark server-client inference (original)
    server_avg, server_std = benchmark_server_client(
        model_path, denoising_steps=4, num_iterations=num_iterations
    )
    
    print("\n" + "=" * 60)
    
    # Compare results
    print("🏆 PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Direct inference (2 steps):     {direct_avg:.3f}s (±{direct_std:.3f}s)")
    
    if server_avg is not None:
        print(f"Server-client (4 steps):        {server_avg:.3f}s (±{server_std:.3f}s)")
        speedup = server_avg / direct_avg
        print(f"Speedup:                        {speedup:.1f}x faster")
        
        if speedup > 2:
            print("🎉 Significant performance improvement achieved!")
        elif speedup > 1.5:
            print("✅ Good performance improvement")
        else:
            print("⚠️  Modest performance improvement")
    else:
        print("Server-client:                  Failed to benchmark")
    
    print("\n🔧 OPTIMIZATION TIPS:")
    print("- Reduce denoising steps further (try 1 step)")
    print("- Use smaller image resolution")
    print("- Ensure CUDA is available and working")
    print("- Consider TensorRT for additional speedup")


if __name__ == "__main__":
    main() 