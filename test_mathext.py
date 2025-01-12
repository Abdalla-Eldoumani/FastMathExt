import mathext

def test_factorial():
    print("Testing factorial function:")
    test_cases = [0, 1, 5, 10, 25, 50, 100, 1000]
    
    for n in test_cases:
        try:
            result = mathext.factorial(n)
            print(f"factorial({n}) = {result}")
        except Exception as e:
            print(f"Error calculating factorial({n}): {str(e)}")
    
    try:
        mathext.factorial(-1)
    except ValueError as e:
        print("Correctly caught negative input:", str(e))

if __name__ == "__main__":
    test_factorial()