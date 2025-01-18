# FastMathExt

FastMathExt is a high-performance C++ extension for Python that provides efficient implementation of mathematical functions. It includes fast factorial calculation and optimized matrix multiplication, leveraging C++ and OpenMP for optimal performance.

## Features

### Factorial Calculation
- Fast factorial calculation using iteration
- Modulo arithmetic (10^9 + 7) to handle large numbers
- Dynamic programming approach for optimization

### Matrix Multiplication
- Cache-optimized matrix multiplication
- OpenMP parallelization for multi-threaded performance
- Support for both square and non-square matrices
- Comparable performance to NumPy for large matrices
- Block matrix multiplication strategy for better cache utilization

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

# Calculate the factorial of 5
result = MathExt.factorial(5)
print(result) # Output: 120
```

- You can also run the included test script:
```bash
python test_mathext.py
```

### Matrix Multiplication
```python
import MathExt
import numpy as np

# Create two matrices
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
B = [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]]

# Multiply matrices using FastMathExt
result = MathExt.matrix_multiply(A, B)

# Compare with NumPy
np_result = np.dot(A, B)
print("Results match:", np.allclose(result, np_result))
```

- You can also run the included test script:
```bash
python test_matrix.py
```