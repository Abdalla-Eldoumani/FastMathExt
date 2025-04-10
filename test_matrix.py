import gc
import sys
import time
import MathExt
import numpy as np

def test_matrix_multiplication():
    try:
        print("Starting matrix multiplication tests...")
        
        # Test case 1: Small square matrices
        print("\nSmall matrix test:")
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        B = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
        
        result = MathExt.matrix_multiply(A, B)
        np_result = np.dot(A, B)
        
        print("C++ Result:", result)
        print("NumPy Result:", np_result.tolist())
        print("Match:", np.allclose(result, np_result))

        # Test different matrix sizes
        sizes = [100, 250, 500, 1000, 1500, 2000]
        
        print("\nLarge matrix performance tests:\n")
        for size in sizes:
            print(f"\nMatrix size {size}x{size}:")
            # Generate random matrices
            A = np.random.rand(size, size).tolist()
            B = np.random.rand(size, size).tolist()
            
            # Warm up the CPU
            _ = MathExt.matrix_multiply([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
            _ = np.dot(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]))
            
            # Time C++ implementation
            start_time = time.time()
            result = MathExt.matrix_multiply(A, B)
            cpp_time = time.time() - start_time
            
            # Time NumPy implementation
            start_time = time.time()
            np_result = np.dot(A, B)
            numpy_time = time.time() - start_time
            
            print(f"C++ Implementation Time: {cpp_time:.4f} seconds")
            print(f"NumPy Implementation Time: {numpy_time:.4f} seconds")
            print(f"Results Match: {np.allclose(result, np_result)}")
            print(f"Speed Ratio (NumPy/C++): {numpy_time/cpp_time:.2f}")
            
            gc.collect()

        # Test case 3: Non-square matrices
        print("\nNon-square matrix test:")
        A = np.random.rand(100, 200).tolist()
        B = np.random.rand(200, 50).tolist()
        
        result = MathExt.matrix_multiply(A, B)
        np_result = np.dot(A, B)
        print(f"Results Match: {np.allclose(result, np_result)}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Current directory: {sys.path[0]}")
    print("\nStarting tests...")
    test_matrix_multiplication()