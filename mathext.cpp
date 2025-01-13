#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

const long long MOD = 1000000007; // 10^9 + 7
std::vector<long long> fact_dp(1, 1);

long long factorial(long long n) {
    if (n < 0) {
        throw std::invalid_argument("Input must be non-negative");
    }

    if (n < fact_dp.size()) {
        return fact_dp[n];
    }

    for (long long i = fact_dp.size(); i <= n; i++) {
        fact_dp.push_back((fact_dp[i-1] * i) % MOD);
    }

    return fact_dp[n];
}

PYBIND11_MODULE(mathext, m) {
    m.doc() = "Extension module for efficient factorial and fibonacci calculations";
    
    m.def("factorial", &factorial, "Calculate factorial of n using dynamic programming",
          py::arg("n"), py::return_value_policy::copy);
}