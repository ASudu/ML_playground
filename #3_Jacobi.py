import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Solve the linear system Ax = b using the Jacobi iterative method.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        n (int): Number of iterations.

    Raises:
        ValueError: If A has a zero or near-zero diagonal entry.
    """
    # Jacobi method iteratively solves each equation for x[i] using
    # the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i))
    m = A.shape[0]
    x = np.zeros(m)
    b = b.flatten()
    diag = A.diagonal()
    tol = 1e-6
    
    if np.any(np.abs(diag) < 1e-12):
        raise ValueError("Zero or near-zero diagonal entry in A.")
    
    D_inv = 1.0 / diag
    R = A - np.diagflat(diag)
    
    for k in range(n):
        x_new = D_inv * (b - R @ x)
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x_new):
            return np.round(x_new, 4)
        x = np.round(x_new, 4)
    
    return x

def main():
    # Test Case 1 - sol: [0.146, 0.2032, -0.5175]
    print(solve_jacobi(np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]),2))

    # Test Case 2 - sol: [-0.0806, 0.9324, 2.4422]
    print(solve_jacobi(np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]), np.array([4, 6, 7]),5))

    # Test Case 3 - sol: [1.7083, -1.9583, -0.7812]
    print(solve_jacobi(np.array([[4,2,-2],[1,-3,-1],[3,-1,4]]), np.array([0,7,5]),3))

if __name__ == "__main__":
    main()