import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]],
                       distance_metric: str="euclidean", max_iterations: int=100) -> list[tuple[float, float]]:
    """Perform K-means clustering given k initial centroids and max number of iterations.

    Args:
        points (list[tuple[float, float]]): List of points to cluster.
        k (int): Number of clusters.
        initial_centroids (list[tuple[float, float]]): Initial centroids for the clusters.
        distance_metric (str): Distance metric to use ("euclidean", "manhattan", "chebyshev").
        max_iterations (int): Maximum number of iterations.

    Returns:
        list[tuple[float, float]]: Final centroids of the clusters.
    """
	
    def dist(p1, p2, name="euclidean"):
        """Calculate distance between two points."""
        d = 0

        if name == "euclidean": # 2-norm
            return np.linalg.norm(np.array(p1) - np.array(p2))
        
        elif name == "manhattan": # 1-norm
            return np.sum(np.abs(np.array(p1) - np.array(p2)))
        
        elif name == "chebyshev": # infinity-norm
            return np.max(np.abs(np.array(p1) - np.array(p2)))
        else:
            raise ValueError("Unknown distance metric")

    centroids = initial_centroids
    clus_map = {tuple(p): 0 for p in points}
    clus_flag = False # Flag to check if cluster assignments change

    for i in range(max_iterations):
        # For each point calculate distance to centroids
        for p, _ in clus_map.items():
            distances = [dist(p, c) for c in centroids]
            clus_map[p] = np.argmin(distances)
            if clus_map[p] != np.argmin(distances):
                clus_flag = True
        
        # Update centroids
        clus_to_point = {i: [] for i in range(k)}

        for point, cluster in clus_map.items():
            clus_to_point[cluster].append(list(point))
        
        centroids = [tuple(np.mean(x, axis=0)) for x in clus_to_point.values()]

        if not clus_flag:
            break

    return centroids, clus_to_point

def main():
    # Test 1 - sol: [(1.0, 2.0), (10.0, 2.0)]
    print(k_means_clustering([(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], 2, [(1, 1), (10, 1)], 10))

    # Test 2 - sol: [(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)]
    print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10))

    # Test 3 - sol: [(2.5, 2.5)]
    print(k_means_clustering([(1, 1), (2, 2), (3, 3), (4, 4)], 1, [(0,0)], 10))

    # Test 4 - sol: [(0.5, 0.5), (0.5, 5.5), (5.5, 0.5), (5.5, 5.5)]
    print(k_means_clustering([(0, 0), (1, 0), (0, 1), (1, 1), (5, 5), (6, 5), (5, 6), (6, 6),(0, 5), (1, 5), (0, 6), (1, 6), (5, 0), (6, 0), (5, 1), (6, 1)], 4, [(0, 0), (0, 5), (5, 0), (5, 5)], 10))


if __name__ == "__main__":
    main()