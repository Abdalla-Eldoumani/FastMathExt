#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <immintrin.h>
#include <memory>

namespace py = pybind11;

// CPU-specific optimizations for Intel i7-10750H (Comet Lake)
#ifdef CPU_INTEL_COMET_LAKE
    constexpr int L1_BLOCK_SIZE = 32;
    constexpr int L2_BLOCK_SIZE = 128;
    constexpr int VECTOR_SIZE = 4;
    constexpr int NUM_THREADS = 12;
#else
    constexpr int L1_BLOCK_SIZE = 32;
    constexpr int L2_BLOCK_SIZE = 128;
    constexpr int VECTOR_SIZE = 4;
    constexpr int NUM_THREADS = 8;
#endif

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
        throw std::invalid_argument("Matrix dimensions mismatch");
    }
}

// Optimized matrix multiplication using blocking and AVX2
std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    
    validate_matrices(A, B);
    
    const int M = static_cast<int>(A.size());
    const int K = static_cast<int>(A[0].size());
    const int N = static_cast<int>(B[0].size());
    
    // Initialize result matrix
    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Main computation with blocking
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < M; i += L1_BLOCK_SIZE) {
        for (int j = 0; j < N; j += L1_BLOCK_SIZE) {
            // Local accumulation array for each thread
            alignas(32) double local_sum[L1_BLOCK_SIZE][L1_BLOCK_SIZE] = {{0.0}};
            
            for (int k = 0; k < K; k += L1_BLOCK_SIZE) {
                // Block multiplication
                for (int ii = i; ii < std::min(i + L1_BLOCK_SIZE, M); ++ii) {
                    for (int kk = k; kk < std::min(k + L1_BLOCK_SIZE, K); ++kk) {
                        const double a_val = A[ii][kk];
                        
                        // Use AVX2 for inner loop when possible
                        int jj = j;
                        for (; jj + VECTOR_SIZE <= std::min(j + L1_BLOCK_SIZE, N); jj += VECTOR_SIZE) {
                            __m256d b_vec = _mm256_setr_pd(
                                B[kk][jj], B[kk][jj + 1],
                                B[kk][jj + 2], B[kk][jj + 3]
                            );
                            __m256d sum_vec = _mm256_loadu_pd(&local_sum[ii - i][jj - j]);
                            __m256d a_vec = _mm256_set1_pd(a_val);
                            sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                            _mm256_storeu_pd(&local_sum[ii - i][jj - j], sum_vec);
                        }
                        
                        // Handle remaining elements
                        for (; jj < std::min(j + L1_BLOCK_SIZE, N); ++jj) {
                            local_sum[ii - i][jj - j] += a_val * B[kk][jj];
                        }
                    }
                }
            }
            
            // Copy block results to output matrix
            for (int ii = i; ii < std::min(i + L1_BLOCK_SIZE, M); ++ii) {
                for (int jj = j; jj < std::min(j + L1_BLOCK_SIZE, N); ++jj) {
                    C[ii][jj] = local_sum[ii - i][jj - j];
                }
            }
        }
    }
    
    return C;
}

PYBIND11_MODULE(MathExt, m) {
    m.doc() = "Extension module for efficient matrix multiplication";
    m.def("matrix_multiply", &matrix_multiply,
          "Multiply two matrices using optimized SIMD and blocking",
          py::arg("A"), py::arg("B"),
          py::return_value_policy::move);
    
    m.doc() = "Extension module for exact factorial calculations using arbitrary precision";
    m.def("factorial", &factorial, 
          "Calculate exact factorial of n\n"
          "Returns the precise result as a string to handle arbitrary-precision numbers",
          py::arg("n"));
}