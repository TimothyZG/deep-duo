import argparse
import os
import time
import torch
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mf.model_factory import create_model

def benchmark_model(model, input_tensor, device, warmup=10, reps=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None

    start_time = time.time()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None
    total_time = time.time() - start_time

    avg_latency = total_time / reps
    throughput = input_tensor.size(0) / avg_latency
    return avg_latency, throughput

def benchmark_combined(model1, model2, input_tensor, device, warmup=10, reps=50):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model1(input_tensor)
            _ = model2(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None

    start_time = time.time()
    with torch.no_grad():
        for _ in range(reps):
            _ = model1(input_tensor)
            _ = model2(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None
    total_time = time.time() - start_time

    avg_latency = total_time / reps
    throughput = input_tensor.size(0) / avg_latency  # 2x because both models infer same batch
    return avg_latency, throughput

def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference throughput")
    parser.add_argument('--model1', type=str, required=True, help='First model name')
    parser.add_argument('--model2', type=str, default=None, help='Second model name (optional)')
    parser.add_argument('--m_head', type=int, default=1, help='Number of ensemble heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(args.batch_size, 3, 224, 224).to(device)

    results = []

    # Model 1
    model1 = create_model(args.model1, num_classes=1000, m_head=args.m_head).to(device)
    latency1, throughput1 = benchmark_model(model1, dummy_input, device)
    print(f"{args.model1} | Latency: {latency1:.4f}s | Throughput: {throughput1:.2f} samples/sec")
    results.append({
        "model_name": args.model1,
        "m_head": args.m_head,
        "batch_size": args.batch_size,
        "latency_sec_per_batch": latency1,
        "throughput_samples_per_sec": throughput1
    })

    # Model 2 (optional)
    if args.model2:
        model2 = create_model(args.model2, num_classes=1000, m_head=args.m_head).to(device)
        latency2, throughput2 = benchmark_model(model2, dummy_input, device)
        print(f"{args.model2} | Latency: {latency2:.4f}s | Throughput: {throughput2:.2f} samples/sec")
        results.append({
            "model_name": args.model2,
            "m_head": args.m_head,
            "batch_size": args.batch_size,
            "latency_sec_per_batch": latency2,
            "throughput_samples_per_sec": throughput2
        })

        # Combined inference
        latency_comb, throughput_comb = benchmark_combined(model1, model2, dummy_input, device)
        print(f"Combined ({args.model1} + {args.model2}) | Latency: {latency_comb:.4f}s | Throughput: {throughput_comb:.2f} samples/sec")
        results.append({
            "model_name": f"{args.model1}+{args.model2}",
            "m_head": args.m_head,
            "batch_size": args.batch_size * 2,  # total inferences per batch
            "latency_sec_per_batch": latency_comb,
            "throughput_samples_per_sec": throughput_comb
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = "throughput.csv"
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved throughput results to {csv_path}")

if __name__ == '__main__':
    main()
