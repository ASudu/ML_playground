import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform feature scaling on the input data.

    Args:
        data (np.ndarray): The input data to be scaled.

    Returns:
        tuple[np.ndarray, np.ndarray]: The standardized and normalized versions of the input data.
    """
    # Standardization
    Mean, Std = np.mean(data, axis=0), np.std(data, axis=0)
    standardized_data = (data - Mean)/(Std)

    # Min-max normalization
    Min, Max = np.min(data, axis=0), np.max(data, axis=0)
    normalized_data = (data - Min)/(Max - Min)

    return standardized_data, normalized_data

def main():
    # Test - sol: ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))

if __name__ == "__main__":
    main()