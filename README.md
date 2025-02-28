# FastMathExt

FastMathExt is a high-performance C++ extension for Python that provides efficient implementations of mathematical functions. It currently includes:

1. **Exact Factorial Calculation** (with arbitrary precision)  
2. **Highly Optimized Matrix Multiplication** leveraging advanced CPU features and multi-threading.

## Key Improvements in v1.0.0

### 1. Improved Memory Layout
- **Flat memory layout** (`Matrix` class) ensures **contiguous memory access** for more efficient caching and prefetching.
- Better **cache locality** for large matrices.

### 2. Enhanced Cache Blocking Strategy
- **Multi-level blocking**:
  - **L3 blocking** (512×512)
  - **L2 blocking** (128×128)
  - **L1 blocking** (32×32)
- **Explicit memory prefetching** and **thread-local block accumulation** to reduce latency and improve parallel performance.

### 3. Algorithmic Enhancement: Strassen’s Algorithm
- **Strassen’s algorithm** for large square matrices (n ≥ 1024).
- Reduces theoretical complexity from O(n³) to ~O(n^2.807).
- **Adaptive threshold** to switch back to standard algorithm for smaller sub-matrices.
- **OpenMP task-based parallelism** for Strassen’s sub-multiplications.

### 4. Advanced OpenMP Parallelization
- **Task-based parallelism** for Strassen’s seven sub-multiplications.
- Fine-grained **dynamic scheduling** with reduced thread sync points.
- Improves **scalability** and ensures **efficient load balancing** across available threads.

### 5. Advanced SIMD and Memory Optimizations
- **AVX2** and **FMA** (Fused Multiply-Add) for efficient vectorization.
- **Memory alignment** and **explicit prefetching** of blocks.
- **Reduced branching** within the critical compute loops.

## Features

### Factorial Calculation
- **Exact factorial** calculation using Python’s arbitrary-precision arithmetic under the hood.
- **No precision loss**, even for factorials of very large numbers.
- Returns results as **strings** to handle extremely large outputs.
- **Error handling** for invalid inputs (e.g., negative numbers).

### Matrix Multiplication
- **High-performance** implementation designed to rival or exceed NumPy speeds.
- **Advanced optimizations**:
  - AVX2 SIMD, FMA instructions, OpenMP parallelization.
  - Cache-aware blocking strategy (L3 → L2 → L1).
  - **Strassen’s Algorithm** for very large, square matrices (≥ 1024×1024).
- Supports both **square** and **non-square** matrices.
- **Double-precision accuracy**, matching NumPy’s floating-point output.

## Requirements

- Python 3.6 or higher
- C++ compiler (Microsoft Visual C++ 14.0 or greater for Windows)
- pybind11

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
pip install pybind11 setuptools
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

- You can also run the included test script:
```bash
python test_matrix.py
```

#### Performance Statistics (1000x1000 matrices)

**Latest Benchmark (10,000 measurements)**  
Tested on the same Intel i7-10750H system:

| Implementation | Mean Time  | Std Dev   | Min Time  | Max Time  |
| -------------- | ---------- | --------- | --------- | --------- |
| **C++**        | 0.2574 s   | 0.0704 s  | 0.1290 s  | 0.8141 s  |
| **NumPy**      | 0.2799 s   | 0.0907 s  | 0.1280 s  | 1.1814 s  |

- C++ is ~8.05% faster on average.  
- More consistent performance (lower standard deviation).  
- Faster worst-case performance (lower max time).

**Prior 1500-Measurement Benchmark**

| Implementation | Mean Time  | Std Dev   | Min Time  | Max Time  |
| -------------- | ---------- | --------- | --------- | --------- |
| **C++**        | 0.2451 s   | 0.0832 s  | 0.1460 s  | 0.6280 s  |
| **NumPy**      | 0.2647 s   | 0.1128 s  | 0.1340 s  | 0.8991 s  |

- C++ is ~7.39% faster on average.  

- Both benchmarks demonstrate consistent gains over NumPy, with lower variance and better worst-case performance.

## Implementation Details (Matrix Multiplication)

### Matrix Multiplication Optimizations
#### **SIMD Instructions**
  - AVX2 for 256-bit wide vector operations on double-precision floats.
  - FMA used to fuse multiply and add, reducing rounding errors and improving throughput.

#### **Hierarchical Cache Blocking**
  - Three-level blocking: L3 (512×512) → L2 (128×128) → L1 (32×32).
  - Explicit memory prefetching to reduce cache misses.
  - Thread-local accumulation to avoid false sharing.

#### **Parallelization**
  - OpenMP used for parallel loops and sections.
  - Task-based approach in Strassen’s algorithm to leverage multi-core systems.
  - Dynamic scheduling for load balancing.

#### **Strassen’s Algorithm**
  - Used only for large, square matrices (≥ 1024×1024).
  - Seven recursive sub-multiplications, each parallelized.
  - Switches back to standard multiplication below an adaptive threshold.

#### **Memory Alignment and Prefetching**
  - Flat memory layout ensures contiguous storage and fewer cache misses.
  - Alignment and prefetching help maintain throughput in critical loops.

### Performance Characteristics
#### **Reduced Cache Misses**
  - Localized access patterns with hierarchical blocking and contiguous memory arrays.

#### **High Parallel Efficiency**
  - Reduced synchronization points.
  - Balanced task distribution across cores with OpenMP’s dynamic scheduling.

#### **Scalability**
  - Near-linear scaling with the number of cores for large matrix sizes.
  - Minimizes overhead for data transfers between caches.