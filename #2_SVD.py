import numpy as np
import math

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """Compute the SVD of a 2x2 matrix.

    Args:
        A (np.ndarray): Input 2x2 matrix.

    Returns:
        tuple: (U, s, V^T) where U and V are the orthogonal matrices and s is the vector of singular values.
    """
    # Concept: Diagonalise A^T*A to get singular values - then use U = A @ V @ S^(-1)
    B = A.T @ A # A^T*A

    # Diagonalize B using eigen-decomposition via atan2
    theta = 0.5 * math.atan2(2 * B[0,1], B[0,0] - B[1,1])
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

    D = R.T @ B @ R
    s = np.sqrt([D[0,0], D[1,1]])
    if s[0] < s[1]:
        s = s[::-1]
        R = R[[1,0], :]

    V = R
    sig_inv = np.diag(1/s)
    U = A @ V @ sig_inv
    SVD = [U, s, V.T]
    return SVD

def main():
    # Test 1 - sol: (array([[ 0.70710678, -0.70710678], [ 0.70710678, 0.70710678]]), array([3., 1.]), array([[ 0.70710678, 0.70710678], [-0.70710678, 0.70710678]]))
    print(svd_2x2_singular_values(np.array([[2, 1], [1, 2]])))

    # Test 2 - sol: array([[ 0.40455358, 0.9145143 ], [ 0.9145143 , -0.40455358]]), array([5.4649857 , 0.36596619]), array([[ 0.57604844, 0.81741556], [-0.81741556, 0.57604844]])
    print(svd_2x2_singular_values(np.array([[1, 2], [3, 4]])))

if __name__ == "__main__":
    main()