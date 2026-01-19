import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """Compute the coefficients of a linear regression model using the normal equation.

    Args:
        X (list[list[float]]): The input features, where each inner list represents a training example.
        y (list[float]): The target values for each training example.

    Returns:
        list[float]: The coefficients of the linear regression model.
    """
    A = np.array(X)
    b = np.array(y)

    # Our problem is Ax = b
    # Normal equation is A^T Ax = A^T b
    c = np.linalg.inv(A.T @ A) @ A.T @ b
    theta = np.array([round(x,4) for x in c.flatten()])

    return theta

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """Compute the coefficients of a linear regression model using gradient descent.

    Args:
        X (np.ndarray): The input features.
        y (np.ndarray): The target values.
        alpha (float): The learning rate.
        iterations (int): The number of iterations.

    Returns:
        np.ndarray: The coefficients of the linear regression model.
    """
    m, n = X.shape
    Y = y.reshape((m,-1))
    theta = np.zeros((n, 1))

    # theta_j = theta_j - (alpha/m) * sum((X @ theta - Y) * X[:, j])
    for i in range(iterations):
        grad = (X.T @ ((X @ theta) - Y))/m
        theta -= alpha * grad

    theta = np.round(theta, decimals=4).flatten() # round off only at the end

    return theta

def main():
    # Test 1 - sol: [4.0, -1.0, -0.0]
    print(f"Normal eqn: {linear_regression_normal_equation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1])}")

    # Test 2 - sol: [0.1107, 0.9513]
    print(f"Gradient descent: {linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000)}")

if __name__ == "__main__":
    main()
