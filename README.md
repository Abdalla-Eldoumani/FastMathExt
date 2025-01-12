# FastMathExt

FastMathExt is a high-performance C++ extension for Python that provides efficient implementation of mathematical functions. Currently, it implements a fast factorial calculation using iterative approach with modulo arithmetic to prevent integer overflow.

## Features

- Fast factorial calculation using iteration
- Modulo arithmetic (10^9 + 7) to handle large numbers
- Written in C++ for optimal performance
- Python bindings using pybind11
- Proper error handling for invalid inputs

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
pip install mathext
```

## Usage

```python
import mathext

# Calculate the factorial of 5
result = mathext.factorial(5)
print(result) # Output: 120
```

- You can also run the included test script:
```bash
python test_mathext.py
```
