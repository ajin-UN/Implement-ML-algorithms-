import numpy as np
from random import randint

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:
        https://en.wikipedia.org/wiki/K-means_clustering
        The KMeans algorithm has two steps:
        1. Update assignments
        2. Update the means
        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.
        Use only numpy to implement this algorithm.
        Args:
            n_clusters (int): Number of clusters to cluster the given data into.
        """
        self.n_clusters = n_clusters
        self.means = []


    def fit(self, features):
        size = len(features)
        labels=np.zeros(size)
        self.classifications=np.zeros(size)+1

        for i in range(self.n_clusters):
            self.means.append(features[randint(0,size)])

        while not np.allclose(labels,self.classifications):
            labels = self.classifications.copy()
            self.classifications = self.update_assignments(features)            
            self.update_means(self.classifications,features)


    def update_means(self, classifications,features):
        m = len(self.means)
        n = len(self.means[0])
        c = len(classifications)
        for i in range(m):
            for j in range(n):
                self.means[i][j] = np.mean([features[x][j] for x in range(c) if i==classifications[x]])


    
    def update_assignments(self, features):
        size = len(features)
        for feature in range(size):
            distances = [np.linalg.norm(features[feature]-mean) for mean in self.means]
            temp = np.argmin(distances)
            self.classifications[feature]=temp
        return self.classifications



    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.
        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """

        predictions=[]

        for feature in features:
            distances=[np.linalg.norm(feature-mean) for mean in self.means]
            classification = distances.index(min(distances))
            predictions.append(classification)
        return np.array(predictions)