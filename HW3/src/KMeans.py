import numpy as np
from tqdm import tqdm

class K_Means:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.centroids = None
        self.max_iter = max_iter
    def fit(self, data):
        tmp = np.random.choice(len(data), self.k)
        self.centroids = data[tmp]

        for step in tqdm(range(self.max_iter)):
            clusters = np.zeros((len(data), self.k))
            for i in range(len(data)):
                distances = [np.linalg.norm(data[i] - self.centroids[j]) for j in range(self.k)]
                clusters[i, np.argmin(distances)] = 1

            prev_centroids = np.copy(self.centroids)
            for i in range(self.k):
                self.centroids[i] = np.average(data[np.where(clusters[:, i] == 1)], axis=0)
            if np.linalg.norm(self.centroids - prev_centroids) < 0.001:
                break
        return self
    def predict(self, data):
        clusters = np.zeros((len(data), self.k))
        total_distances = np.zeros((len(data), self.k))
        for i in range(len(data)):
            distances = [np.linalg.norm(data[i] - self.centroids[j]) for j in range(self.k)]
            total_distances[i] = distances
            clusters[i ,np.argmin(distances)] = 1
        return clusters, total_distances

class Agglomerative_Clustering:
    def __init__(self, k):
        self.k = k
    def fit(self, data):
        self.clusters = np.zeros((len(data), len(data[0])))
        for i in range(len(data)):
            self.clusters[i] = data[i]
        while len(self.clusters) > self.k:
            running_min = np.inf
            running_min_index = None
            #print(len(self.clusters))
            for i in range(len(self.clusters)):
                for j in range(i,len(self.clusters)):
                    if np.linalg.norm(self.clusters[i] - self.clusters[j]) < running_min:
                        running_min = np.linalg.norm(self.clusters[i] - self.clusters[j])
                        running_min_index = (i, j)
                if running_min < 0.000001:
                    break
            avg = (self.clusters[running_min_index[0]] + self.clusters[running_min_index[1]]) / 2
            self.clusters = np.delete(self.clusters, running_min_index[1], axis=0)
            self.clusters = np.delete(self.clusters, running_min_index[0], axis=0)
            self.clusters = np.vstack((self.clusters, avg))
        self.centroids = self.clusters
        return self
    def predict(self, data):
        clusters = np.zeros((len(data), self.k))
        total_distances = np.zeros((len(data), self.k))
        for i in range(len(data)):
            distances = [np.linalg.norm(data[i] - self.clusters[j]) for j in range(self.k)]
            total_distances[i] = distances
            clusters[i ,np.argmin(distances)] = 1
        return clusters, total_distances


