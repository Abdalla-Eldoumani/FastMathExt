#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace py = pybind11;

std::string factorial(long long n) {
    if (n < 0) {
        throw std::invalid_argument("Input must be non-negative");
    }

    if (n <= 1) {
        return "1";
    }

    py::object result = py::cast(1);
    
    for (long long i = 2; i <= n; i++) {
        result = result * py::cast(i);
    }

    return py::str(result);
}

void validate_matrices(const std::vector<std::vector<double>>& A, 
                      const std::vector<std::vector<double>>& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        throw std::invalid_argument("Matrices cannot be empty");
    }
    if (A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    }
}

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    
    validate_matrices(A, B);
    
    const int M = static_cast<int>(A.size());
    const int K = static_cast<int>(A[0].size());
    const int N = static_cast<int>(B[0].size());
    
    // Initialize result matrix with zeros
    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));
    
    // Cache blocking parameters
    const int BLOCK_SIZE = 32;
    
    // Set number of threads
    #pragma omp parallel
    {
        // Block multiplication with cache optimization
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < M; i += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                for (int k = 0; k < K; k += BLOCK_SIZE) {
                    // Process blocks
                    for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ++ii) {
                        for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); ++jj) {
                            double sum = C[ii][jj];
                            for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); ++kk) {
                                sum += A[ii][kk] * B[kk][jj];
                            }
                            C[ii][jj] = sum;
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

PYBIND11_MODULE(MathExt, m) {
    m.doc() = "Extension module for efficient calculations";
    m.def("matrix_multiply", &matrix_multiply,
          "Multiply two matrices using optimized blocking",
          py::arg("A"), py::arg("B"),
          py::return_value_policy::move);
    
    m.doc() = "Extension module for exact factorial calculations using arbitrary precision";
    m.def("factorial", &factorial, 
          "Calculate exact factorial of n\n"
          "Returns the precise result as a string to handle arbitrary-precision numbers",
          py::arg("n"));
}