import numpy as np
from itertools import chain

np.random.seed(42)

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Generate train-test index splits for k-fold cross-validation.
    
    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector (unused here, just for shape validation).
        k (int): Number of folds.
        shuffle (bool): Whether to shuffle indices before splitting.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list of (train_indices, test_indices) tuples.
    """
    folds = []
    n_samples = X.shape[0]
    indices = [i for i in range(n_samples)]

    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = [n_samples // k + (1 if i < n_samples % k else 0) for i in range(k)]
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size

    results = []
    for i in range(k):
        test_indices = folds[i]
        train_indices = list(chain.from_iterable(folds[j] for j in range(k) if j != i))
        results.append((train_indices, test_indices))

    
    return results

def main():
    # Test 1 - sol: [([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
    print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False))

    # Test 2 - sol: [([2, 9, 4, 3, 6], [8, 1, 5, 0, 7]), ([8, 1, 5, 0, 7], [2, 9, 4, 3, 6])]
    print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=2, shuffle=True))

    # Test 3 - sol: [([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4]), ([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], [5, 6, 7, 8, 9]), ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14])]
    print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]), np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]), k=3, shuffle=False))

    # Test 4 - sol: [([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]), ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])]
    print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=2, shuffle=False))

if __name__ == "__main__":
    main()
