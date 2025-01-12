#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

const long long MOD = 1000000007; // 10^9 + 7

long long factorial(long long n) {
    if (n < 0) {
        throw std::invalid_argument("Input must be non-negative");
    }
    
    long long result = 1;
    for (long long i = 2; i <= n; i++) {
        result = (result * i) % MOD;
    }
    return result;
}

PYBIND11_MODULE(mathext, m) {
    m.doc() = "Extension module for efficient factorial and fibonacci calculations";
    
    m.def("factorial", &factorial, "Calculate factorial of n using iteration",
          py::arg("n"), py::return_value_policy::copy);
}