# FastMathExt

FastMathExt is a high-performance C++ extension for Python that provides efficient implementations of mathematical functions. It currently includes:

1. **Exact Factorial Calculation** (with arbitrary precision)  
2. **Highly Optimized Matrix Multiplication** leveraging advanced CPU features and multi-threading.

## Key Improvements in v1.2

### 1. Improved Memory Layout
- **Flat memory layout** (`Matrix` class) ensures **contiguous memory access** for more efficient caching and prefetching.
- Better **cache locality** for large matrices.

### 2. Enhanced Cache Blocking Strategy
- **Multi-level blocking**:
  - **L3 blocking** (512×512)
  - **L2 blocking** (128×128)
  - **L1 blocking** (32×32)
- **Explicit memory prefetching** and **thread-local block accumulation** to reduce latency and improve parallel performance.

### 3. Algorithmic Enhancement: Strassen's Algorithm
- **Strassen's algorithm** for large square matrices (n ≥ 512).
- Reduces theoretical complexity from O(n³) to ~O(n^2.807).
- **Adaptive threshold** to switch back to standard algorithm for smaller sub-matrices.
- **OpenMP task-based parallelism** for Strassen's sub-multiplications.

### 4. Advanced OpenMP Parallelization
- **Task-based parallelism** for Strassen's seven sub-multiplications.
- Fine-grained **dynamic scheduling** with reduced thread sync points.
- Improves **scalability** and ensures **efficient load balancing** across available threads.

### 5. Advanced SIMD and Memory Optimizations
- **AVX2** and **FMA** (Fused Multiply-Add) for efficient vectorization.
- **Memory alignment** and **explicit prefetching** of blocks.
- **Reduced branching** within the critical compute loops.

## Features

### Factorial Calculation
- **Exact factorial** calculation using Python's arbitrary-precision arithmetic under the hood.
- **No precision loss**, even for factorials of very large numbers.
- Returns results as **strings** to handle extremely large outputs.
- **Error handling** for invalid inputs (e.g., negative numbers).

### Matrix Multiplication
- **High-performance** implementation designed to rival or exceed NumPy speeds.
- **Advanced optimizations**:
  - AVX2 SIMD, FMA instructions, OpenMP parallelization.
  - Cache-aware blocking strategy (L3 → L2 → L1).
  - **Strassen's Algorithm** for very large, square matrices (≥ 512×512).
- Supports both **square** and **non-square** matrices.
- **Double-precision accuracy**, matching NumPy's floating-point output.

## Requirements

- Python 3.6 or higher
- C++ compiler (Microsoft Visual C++ 14.0 or greater for Windows)
- pybind11
- NumPy (for comparison testing)
- psutil (for benchmarking)

### Windows-specific Requirements

If you're on Windows, you'll need:
- Microsoft Visual C++ Build Tools
  - Download from: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - During installation, select "Desktop development with C++"

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdalla-Eldoumani/FastMathExt.git
cd FastMathExt
```

2. Install the required Python packages:
```bash
pip install pybind11 setuptools numpy psutil
python setup.py build_ext --inplace
```

## Usage

### Factorial Calculation
```python
import MathExt

# Calculate small factorials
result = MathExt.factorial(5)
print(result)  # Output: 120

# Calculate large factorials
result = MathExt.factorial(100)
print(result)  # Output: exact 100! (158 digits)
```

- You can also run the included test script:
```bash
python test_mathext.py
```

### Matrix Multiplication
```python
import MathExt
import numpy as np
import time

# Create two matrices
A = np.random.rand(1000, 1000).tolist()
B = np.random.rand(1000, 1000).tolist()

# Time FastMathExt implementation
start = time.time()
result = MathExt.matrix_multiply(A, B)
mathext_time = time.time() - start

# Time NumPy implementation
start = time.time()
np_result = np.dot(A, B)
numpy_time = time.time() - start

print(f"FastMathExt Time: {mathext_time:.4f} seconds")
print(f"NumPy Time: {numpy_time:.4f} seconds")
print("Results match:", np.allclose(result, np_result))
```

### Testing and Benchmarking
- The package includes comprehensive testing and benchmarking scripts.

- Basic Testing
```bash
python test_matrix.py           # Run default tests
python test_matrix.py 1000 1500 # test with specified matrix sizes
```

- Advanced Benchmarking
```bash
python benchmark_matrix.py                      # Run default benchmark
python benchmark_matrix.py 100 --sizes 500 1000 # Run 1000 iterations with specified sizes
```

# Comprehensive Performance Results

## Detailed Benchmark (10,000 iterations per size)

| Matrix Size | Implementation | Mean Time  | Median Time | Std Dev    | Min Time   | Max Time   |
|-------------|----------------|------------|-------------|------------|------------|------------|
| 250×250     | **C++**        | 0.0103 s   | 0.0100 s    | 0.0019 s   | 0.0069 s   | 0.0165 s   |
| 250×250     | NumPy          | 0.0174 s   | 0.0170 s    | 0.0034 s   | 0.0120 s   | 0.0290 s   |
| 250×250     | Improvement    | **40.8%**  | **41.2%**   | Better     | Better     | Better     |
| 500×500     | **C++**        | 0.0409 s   | 0.0410 s    | 0.0076 s   | 0.0280 s   | 0.0640 s   |
| 500×500     | NumPy          | 0.0671 s   | 0.0660 s    | 0.0131 s   | 0.0470 s   | 0.1100 s   |
| 500×500     | Improvement    | **39.1%**  | **37.9%**   | Better     | Better     | Better     |
| 750×750     | **C++**        | 0.1011 s   | 0.0990 s    | 0.0189 s   | 0.0570 s   | 0.1596 s   |
| 750×750     | NumPy          | 0.1352 s   | 0.1300 s    | 0.0306 s   | 0.0850 s   | 0.2320 s   |
| 750×750     | Improvement    | **25.2%**  | **23.9%**   | Better     | Better     | Better     |
| 1000×1000   | **C++**        | 0.2133 s   | 0.2030 s    | 0.0459 s   | 0.1360 s   | 0.3530 s   |
| 1000×1000   | NumPy          | 0.2754 s   | 0.2560 s    | 0.0724 s   | 0.1550 s   | 0.4918 s   |
| 1000×1000   | Improvement    | **22.6%**  | **20.7%**   | Better     | Better     | Better     |
| 1250×1250   | **C++**        | 0.3573 s   | 0.3370 s    | 0.0802 s   | 0.2210 s   | 0.6066 s   |
| 1250×1250   | NumPy          | 0.4130 s   | 0.3870 s    | 0.1052 s   | 0.2180 s   | 0.7237 s   |
| 1250×1250   | Improvement    | **13.5%**  | **12.9%**   | Better     | Comparable | Better     |
| 1500×1500   | **C++**        | 0.5250 s   | 0.4980 s    | 0.1125 s   | 0.3080 s   | 0.8614 s   |
| 1500×1500   | NumPy          | 0.5854 s   | 0.5570 s    | 0.1415 s   | 0.3050 s   | 0.9991 s   |
| 1500×1500   | Improvement    | **10.3%**  | **10.6%**   | Better     | Comparable | Better     |
| 1750×1750   | **C++**        | 0.7471 s   | 0.7220 s    | 0.1922 s   | 0.4350 s   | 1.2938 s   |
| 1750×1750   | NumPy          | 0.7279 s   | 0.7215 s    | 0.2113 s   | 0.3790 s   | 1.2841 s   |
| 1750×1750   | Improvement    | **–2.6%**  | **–0.1%**   | Better     | Slower     | Comparable |

---

## Key Performance Insights

- **Size-dependent performance advantage**  
  - Small-to-medium matrices (250×250 to 750×750): C++ is 25–41 % faster than NumPy  
  - Medium-to-large matrices (1000×1000 to 1500×1500): C++ maintains a 10–23 % lead  
  - Very large matrices (1750×1750): performance converges, with NumPy slightly edging out C++

- **Consistency advantages**  
  - Lower standard deviation across all sizes  
  - More predictable runs with fewer extreme outliers  
  - Better minimum times on small-to-medium workloads

- **Scalability characteristics**  
  - Near-linear runtime growth with matrix size  
  - Excellent multi-core efficiency  
  - Superior cache utilization on sizes fitting in each cache level

---

## Performance Distribution

The benchmark harness captured 10,000 measurements per size, automatically removed statistical outliers, and then computed metrics. This provides a robust view of typical, best-case, and worst-case timings.

---

## Implementation Details (Matrix Multiplication)

### Matrix Multiplication Optimizations

#### SIMD Instructions
- **AVX2** for 256-bit vector operations on doubles  
- **FMA** to fuse multiply-add, reducing rounding errors and boosting throughput

#### Hierarchical Cache Blocking
- **Three-level blocking**:  
  - L3 block = 512×512  
  - L2 block = 128×128  
  - L1 block = 32×32  
- **Explicit prefetching** to minimize cache misses  
- **Thread-local accumulators** to avoid false sharing

#### Parallelization
- **OpenMP** for parallel loops and sections  
- **Task-based Strassen** for large sub-multiplications  
- **Dynamic scheduling** for balanced workload across cores

#### Strassen’s Algorithm
- Applied only when size ≥ 512×512  
- Seven recursive multiplications, each parallelized  
- Falls back to standard multiply below an adaptive threshold

#### Memory Alignment and Prefetching
- **Flat, contiguous layout** for fewer cache misses  
- Aligned allocations and prefetch hints in critical loops  

---

## Performance Characteristics

### Reduced Cache Misses
- Localized access via hierarchical blocking  
- Contiguous memory arrays minimize eviction

### High Parallel Efficiency
- Fewer synchronization barriers  
- Balanced OpenMP scheduling keeps cores busy

### Scalability
- Near-linear speed-up up to 1500×1500 matrices  
- Optimized data paths to reduce cache-to-cache transfers