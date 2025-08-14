import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Calculate the covariance matrix of a set of vectors.

    Args:
        vectors (list[list[float]]): A list of vectors (each vector is a list of floats).
    """

    X = np.array(vectors)
    n_feat, n_samp = X.shape
    cov_mat = np.zeros((n_feat, n_feat))

    for i in range(n_feat):
        for j in range(n_feat):
            xi = X[i]
            xj = X[j]
            cov_mat[i][j] = np.sum((xi - np.mean(xi))*(xj - np.mean(xj)))/(n_samp - 1)

    return cov_mat

def main():
    # Test 1 - sol: [[1.0, 1.0], [1.0, 1.0]]
    print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))

    # Test 2 - sol: [[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]
    print(calculate_covariance_matrix([[1, 5, 6], [2, 3, 4], [7, 8, 9]]))

if __name__ == "__main__":
    main()