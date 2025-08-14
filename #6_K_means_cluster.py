import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    """Perform K-means clustering given k initial centroids and max number of iterations.

    Args:
        points (list[tuple[float, float]]): List of points to cluster.
        k (int): Number of clusters.
        initial_centroids (list[tuple[float, float]]): Initial centroids for the clusters.
        max_iterations (int): Maximum number of iterations.

    Returns:
        list[tuple[float, float]]: Final centroids of the clusters.
    """
	
    def dist(p1, p2):
        d = 0

        for i in range(len(p1)):
            d += (p1[i] - p2[i])**2
        return np.sqrt(d)

    centroids = initial_centroids
    clus_map = {p: 0 for p in points}

    for i in range(max_iterations):
        # For each point calculate distance to centroids
        for p in clus_map:
            distances = [dist(p, c) for c in centroids]
            clus_map[p] = np.argmin(distances)
        
        # Update centroids
        clus_to_point = {i: [] for i in range(k)}

        for point, cluster in clus_map.items():
            clus_to_point[cluster].append(list(point))
        
        centroids = [tuple(np.mean(x, axis=0)) for x in clus_to_point.values()]

    return centroids

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