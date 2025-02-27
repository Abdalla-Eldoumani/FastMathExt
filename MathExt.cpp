#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <immintrin.h>
#include <memory>

namespace py = pybind11;

constexpr int L1_BLOCK_SIZE = 32;
constexpr int L2_BLOCK_SIZE = 128;
constexpr int L3_BLOCK_SIZE = 512;
constexpr int VECTOR_SIZE = 4;
constexpr int NUM_THREADS = 12;

class Matrix {
private:
    std::vector<double> data;
    int rows;
    int cols;

public:
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    Matrix(const std::vector<std::vector<double>>& mat) {
        rows = mat.size();
        cols = mat[0].size();
        data.resize(rows * cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = mat[i][j];
            }
        }
    }
    
    double& at(int i, int j) {
        return data[i * cols + j];
    }
    
    double at(int i, int j) const {
        return data[i * cols + j];
    }
    
    std::vector<std::vector<double>> to_vector() const {
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i * cols + j];
            }
        }
        return result;
    }
    
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    
    const double* get_row_ptr(int row) const {
        return &data[row * cols];
    }
    
    double* get_row_ptr(int row) {
        return &data[row * cols];
    }
};

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

bool should_use_strassen(int M, int N, int K) {
    // Only use Strassen for large, square matrices
    return (M >= 1024 && N >= 1024 && K >= 1024 && 
            M == N && N == K);
}

Matrix strassen_multiply(const Matrix& A, const Matrix& B, int threshold = 128) {
    int n = A.get_rows();
    
    if (n <= threshold) {
        Matrix C(n, n);
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                __m256d a_val = _mm256_set1_pd(A.at(i, k));
                for (int j = 0; j < n; j += VECTOR_SIZE) {
                    if (j + VECTOR_SIZE <= n) {
                        __m256d b_vec = _mm256_loadu_pd(&B.at(k, j));
                        __m256d c_vec = _mm256_loadu_pd(&C.at(i, j));
                        c_vec = _mm256_fmadd_pd(a_val, b_vec, c_vec);
                        _mm256_storeu_pd(&C.at(i, j), c_vec);
                    }
                    else {
                        for (int jr = j; jr < n; jr++) {
                            C.at(i, jr) += A.at(i, k) * B.at(k, jr);
                        }
                    }
                }
            }
        }
        return C;
    }
    
    int m = n / 2;
    
    Matrix A11(m, m), A12(m, m), A21(m, m), A22(m, m);
    Matrix B11(m, m), B12(m, m), B21(m, m), B22(m, m);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A11.at(i, j) = A.at(i, j);
                    B11.at(i, j) = B.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A12.at(i, j) = A.at(i, j + m);
                    B12.at(i, j) = B.at(i, j + m);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A21.at(i, j) = A.at(i + m, j);
                    B21.at(i, j) = B.at(i + m, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A22.at(i, j) = A.at(i + m, j + m);
                    B22.at(i, j) = B.at(i + m, j + m);
                }
            }
        }
    }
    
    Matrix P1, P2, P3, P4, P5, P6, P7;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A11.at(i, j) + A22.at(i, j);
                }
            }
            
            Matrix temp2(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp2.at(i, j) = B11.at(i, j) + B22.at(i, j);
                }
            }
            
            P1 = strassen_multiply(temp1, temp2, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A21.at(i, j) + A22.at(i, j);
                }
            }
            
            P2 = strassen_multiply(temp1, B11, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = B12.at(i, j) - B22.at(i, j);
                }
            }
            
            P3 = strassen_multiply(A11, temp1, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = B21.at(i, j) - B11.at(i, j);
                }
            }
            
            P4 = strassen_multiply(A22, temp1, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A11.at(i, j) + A12.at(i, j);
                }
            }
            
            P5 = strassen_multiply(temp1, B22, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A21.at(i, j) - A11.at(i, j);
                }
            }
            
            Matrix temp2(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp2.at(i, j) = B11.at(i, j) + B12.at(i, j);
                }
            }
            
            P6 = strassen_multiply(temp1, temp2, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A12.at(i, j) - A22.at(i, j);
                }
            }
            
            Matrix temp2(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp2.at(i, j) = B21.at(i, j) + B22.at(i, j);
                }
            }
            
            P7 = strassen_multiply(temp1, temp2, threshold);
        }
    }
    
    Matrix C(n, n);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i, j) = P1.at(i, j) + P4.at(i, j) - P5.at(i, j) + P7.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i, j + m) = P3.at(i, j) + P5.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i + m, j) = P2.at(i, j) + P4.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i + m, j + m) = P1.at(i, j) - P2.at(i, j) + P3.at(i, j) + P6.at(i, j);
                }
            }
        }
    }
    
    return C;
}

inline void prefetch_matrix_block(const Matrix& mat, int i, int j, int block_size) {
    for (int ii = i; ii < i + block_size && ii < mat.get_rows(); ii += 4) {
        _mm_prefetch(mat.get_row_ptr(ii), _MM_HINT_T0);
    }
}

Matrix standard_multiply(const Matrix& A, const Matrix& B) {
    int M = A.get_rows();
    int K = A.get_cols();
    int N = B.get_cols();
    
    Matrix C(M, N);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for (int i2 = 0; i2 < M; i2 += L2_BLOCK_SIZE) {
            for (int j2 = 0; j2 < N; j2 += L2_BLOCK_SIZE) {
                for (int k2 = 0; k2 < K; k2 += L2_BLOCK_SIZE) {
                    if (k2 + L2_BLOCK_SIZE < K) {
                        prefetch_matrix_block(A, i2, k2 + L2_BLOCK_SIZE, std::min(L2_BLOCK_SIZE, M - i2));
                        prefetch_matrix_block(B, k2 + L2_BLOCK_SIZE, j2, std::min(L2_BLOCK_SIZE, N - j2));
                    }
                    
                    for (int i1 = i2; i1 < std::min(i2 + L2_BLOCK_SIZE, M); i1 += L1_BLOCK_SIZE) {
                        for (int j1 = j2; j1 < std::min(j2 + L2_BLOCK_SIZE, N); j1 += L1_BLOCK_SIZE) {
                            alignas(32) double local_sum[L1_BLOCK_SIZE][L1_BLOCK_SIZE] = {{0.0}};
                            
                            for (int k1 = k2; k1 < std::min(k2 + L2_BLOCK_SIZE, K); k1 += L1_BLOCK_SIZE) {
                                for (int i = i1; i < std::min(i1 + L1_BLOCK_SIZE, M); ++i) {
                                    for (int k = k1; k < std::min(k1 + L1_BLOCK_SIZE, K); ++k) {
                                        double a_val = A.at(i, k);
                                        __m256d a_vec = _mm256_set1_pd(a_val);
                                        
                                        int j = j1;
                                        for (; j + VECTOR_SIZE <= std::min(j1 + L1_BLOCK_SIZE, N); j += VECTOR_SIZE) {
                                            __m256d b_vec = _mm256_loadu_pd(&B.at(k, j));
                                            __m256d sum_vec = _mm256_loadu_pd(&local_sum[i - i1][j - j1]);
                                            sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                                            _mm256_storeu_pd(&local_sum[i - i1][j - j1], sum_vec);
                                        }
                                        
                                        for (; j < std::min(j1 + L1_BLOCK_SIZE, N); ++j) {
                                            local_sum[i - i1][j - j1] += a_val * B.at(k, j);
                                        }
                                    }
                                }
                            }
                            
                            for (int i = i1; i < std::min(i1 + L1_BLOCK_SIZE, M); ++i) {
                                for (int j = j1; j < std::min(j1 + L1_BLOCK_SIZE, N); ++j) {
                                    C.at(i, j) += local_sum[i - i1][j - j1];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

Matrix optimized_matrix_multiply(const Matrix& A, const Matrix& B) {
    int M = A.get_rows();
    int K = A.get_cols();
    int N = B.get_cols();
    
    if (should_use_strassen(M, N, K)) {
        return strassen_multiply(A, B);
    }
    
    return standard_multiply(A, B);
}

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    
    validate_matrices(A, B);
    
    Matrix A_flat(A);
    Matrix B_flat(B);
    
    Matrix C_flat = optimized_matrix_multiply(A_flat, B_flat);
    
    return C_flat.to_vector();
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