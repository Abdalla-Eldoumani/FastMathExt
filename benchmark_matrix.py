import subprocess
import sys
import os
import numpy as np
import re
import time
import argparse
import gc

def run_benchmark(iterations, sizes=[1000, 1500, 1750]):
    results = {}
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {size}x{size} MATRICES")
        print(f"{'='*60}")
        
        cpp_times = []
        numpy_times = []
        
        print(f"Running benchmark for {iterations} iterations...")
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r")
            
            # Run test_matrix.py with specific size argument
            result = subprocess.run([sys.executable, "test_matrix.py", str(size)], 
                                   capture_output=True, text=True)
            
            output = result.stdout
            
            # Extract timing information
            match_cpp = re.search(f"Matrix size {size}x{size}:.*?\nC\+\+ Implementation Time: ([\d\.]+) seconds", 
                                 output, re.DOTALL)
            match_numpy = re.search(f"Matrix size {size}x{size}:.*?\nNumPy Implementation Time: ([\d\.]+) seconds", 
                                   output, re.DOTALL)
            
            if match_cpp and match_numpy:
                cpp_time = float(match_cpp.group(1))
                numpy_time = float(match_numpy.group(1))
                
                cpp_times.append(cpp_time)
                numpy_times.append(numpy_time)
            
            # Force garbage collection to clean up memory
            gc.collect()
        
        if cpp_times and numpy_times:
            results[size] = {
                "cpp": {
                    "mean": np.mean(cpp_times),
                    "std": np.std(cpp_times),
                    "min": np.min(cpp_times),
                    "max": np.max(cpp_times)
                },
                "numpy": {
                    "mean": np.mean(numpy_times),
                    "std": np.std(numpy_times),
                    "min": np.min(numpy_times),
                    "max": np.max(numpy_times)
                }
            }
    
    # Print all results
    print("\n\n" + "="*80)
    print(f"BENCHMARK RESULTS ({iterations} iterations per matrix size)")
    print("="*80)
    
    for size in sizes:
        if size in results:
            r = results[size]
            
            print(f"\n### Performance Statistics ({size}x{size} matrices)\n")
            
            print("| Implementation | Mean Time  | Std Dev   | Min Time  | Max Time  |")
            print("| -------------- | ---------- | --------- | --------- | --------- |")
            print(f"| **C++**        | {r['cpp']['mean']:.4f} s   | {r['cpp']['std']:.4f} s  | {r['cpp']['min']:.4f} s  | {r['cpp']['max']:.4f} s  |")
            print(f"| **NumPy**      | {r['numpy']['mean']:.4f} s   | {r['numpy']['std']:.4f} s  | {r['numpy']['min']:.4f} s  | {r['numpy']['max']:.4f} s  |")
            
            # Calculate performance improvement
            improvement = (r['numpy']['mean'] - r['cpp']['mean']) / r['numpy']['mean'] * 100
            
            print(f"\n- C++ is ~{improvement:.2f}% faster on average.")
            print(f"- {'More' if r['cpp']['std'] < r['numpy']['std'] else 'Less'} consistent performance (lower standard deviation).")
            print(f"- {'Faster' if r['cpp']['max'] < r['numpy']['max'] else 'Slower'} worst-case performance (lower max time).")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matrix multiplication benchmark.")
    parser.add_argument("iterations", type=int, nargs="?", default=5,
                        help="Number of iterations to run per matrix size (default: 5)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 1500, 1750],
                        help="Matrix sizes to benchmark (default: 1000 1500 1750)")
    args = parser.parse_args()
    
    run_benchmark(args.iterations, args.sizes)