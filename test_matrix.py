import numpy as np
import MathExt
import time

def test_matrix_multiplication():
    # Test case 1: Small square matrices
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    result = MathExt.matrix_multiply(A, B)
    np_result = np.dot(A, B)
    
    print("Small matrix test:")
    print("C++ Result:", result)
    print("NumPy Result:", np_result)
    print("Match:", np.allclose(result, np_result))
    
    # Test case 2: Large matrices performance comparison
    size = 1000
    A = np.random.rand(size, size).tolist()
    B = np.random.rand(size, size).tolist()
    
    # Time C++ implementation
    start_time = time.time()
    cpp_result = MathExt.matrix_multiply(A, B)
    cpp_time = time.time() - start_time
    
    # Time NumPy implementation
    start_time = time.time()
    np_result = np.dot(A, B)
    np_time = time.time() - start_time
    
    print(f"\nLarge matrix ({size}x{size}) performance test:")
    print(f"C++ Implementation Time: {cpp_time:.4f} seconds")
    print(f"NumPy Implementation Time: {np_time:.4f} seconds")
    print("Results Match:", np.allclose(cpp_result, np_result))
    
    # Test case 3: Non-square matrices
    A = np.random.rand(100, 200).tolist()
    B = np.random.rand(200, 50).tolist()
    
    result = MathExt.matrix_multiply(A, B)
    np_result = np.dot(A, B)
    
    print("\nNon-square matrix test:")
    print("Results Match:", np.allclose(result, np_result))

def warmup():
    size = 500
    A = np.random.rand(size, size).tolist()
    B = np.random.rand(size, size).tolist()
    _ = MathExt.matrix_multiply(A, B)
    _ = np.dot(np.array(A), np.array(B))

if __name__ == "__main__":
    warmup()
    test_matrix_multiplication()