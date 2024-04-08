import numpy as np
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iteration=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iteration):
            distances = np.array([Kmeans.euclidean_distance(data_point, self.centroids) for data_point in X])
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([np.mean(X[labels == i], axis=0) for i in range(self.k)])

            if np.allclose(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids

        return labels, self.centroids

random_points = np.random.randint(0, 100, (100, 2))

kmeans = Kmeans(k=3)
labels, centroids = kmeans.fit(random_points)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c=range(len(centroids)), marker="*", s=200)
plt.show()




        