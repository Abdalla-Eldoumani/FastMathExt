# FastMathExt

FastMathExt is a high-performance C++ extension for Python that provides efficient implementation of mathematical functions. Currently, it implements an exact factorial calculator and a highly optimized matrix multiplication that outperforms NumPy's performance, leveraging advanced C++ features, SIMD instructions, and multi-threading.

## Features

### Factorial Calculation
- Exact factorial calculation without precision loss
- Handles arbitrarily large numbers (1000+ factorial)
- Returns precise results as strings to maintain accuracy
- Efficient implementation using Python's arbitrary-precision arithmetic
- Proper error handling for invalid inputs

### Matrix Multiplication
- High-performance implementation rivaling NumPy's speed
- Advanced optimizations:
  - AVX2 SIMD instructions for vectorized operations
  - OpenMP parallelization for multi-threading
  - Cache-friendly blocking strategy
  - FMA (Fused Multiply-Add) instructions
  - Aligned memory access patterns
- Support for both square and non-square matrices
- Matches NumPy's accuracy with double precision
- Performance comparable to or better than NumPy in many cases

#### Performance Statistics (1000x1000 matrices)
Based on 1500 measurements with an Intel i7-10750H CPU:

**C++ Implementation:**
- Mean time: 0.2451 seconds
- Standard deviation: 0.0832 seconds
- Min time: 0.1460 seconds
- Max time: 0.6280 seconds

**NumPy Implementation:**
- Mean time: 0.2647 seconds
- Standard deviation: 0.1128 seconds
- Min time: 0.1340 seconds
- Max time: 0.8991 seconds

**Performance Advantage:**
- FastMathExt is 7.39% faster than NumPy on average
- More consistent performance (lower standard deviation)
- Better worst-case performance (lower maximum time)

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

## Implementation Details (Matrix Multiplication)

### Matrix Multiplication Optimizations
#### **SIMD Instructions**
  - Uses AVX2 instructions for 256-bit vector operations
  - Processes 4 doubles simultaneously
  - Employs FMA instructions for better throughput

#### **Cache Optimization**
  - Block size tuned to L1 cache (32KB)
  - Two-level blocking strategy
  - Aligned memory access patterns
  - Local accumulation arrays for better cache utilization

#### **Parallelization**
  - OpenMP parallel processing
  - Dynamic scheduling for better load balancing
  - Optimized for modern multi-core processors
  - Thread count tuned to CPU's logical core count

#### **Precision**
  - Matches NumPy's accuracy with double precision
  - FMA instructions for better precision
  - Proper handling of edge cases

### Performance Characteristics
#### **Consistency**
  - Lower standard deviation than NumPy (0.0832s vs 0.1128s)
  - More predictable execution times
  - Better worst-case performance (0.6280s vs 0.8991s)

#### **Scalability**
  - Near-linear scaling with core count
  - Efficient use of all available CPU threads
  - Optimal performance on modern Intel processors
  - Balanced resource utilization