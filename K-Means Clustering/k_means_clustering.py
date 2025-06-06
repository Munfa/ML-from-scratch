import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def eucleadian_distance(x1, x2):
    d = np.sqrt(np.sum((x1 - x2)**2, axis=1)) # x1 is the datapoints and x2 is the centroids
    return d

def initialize_centroids(x, k):
    # choosing random centroids within X data and avoiding to choose the same centroids multiple times
    indices = np.random.choice(x.shape[0], k, replace=False) 
    return x[indices]

def update_centroids(X, y, k):
    centroids = np.zeros((k, X.shape[1])) # shape = k-rows and X-features
    for i in range(k):
        cluster = X[y == i]  # selecting the data belongs to each cluster
        if(len(cluster) > 0):
            centroids[i] = np.mean(cluster, axis=0) # getting the mean of a cluster as the new centroid
    
    return centroids

# defining maximum iteration and tolerance for checking convergence
def train_data(X, k, max_iter=200, tol=0.00001):
    np.random.seed(123) # to get a consistent initial centroids and result
    centroids = initialize_centroids(X, k) # getting random initial centroids

    for _ in range(max_iter):
        clusters = [] # an empty list for clusters

        for datapoint in X:
            distances = eucleadian_distance(datapoint, centroids)
            cluster_num = np.argmin(distances)      # the datapoint belongs to this cluster
            clusters.append(cluster_num)   
        
        clusters = np.array(clusters) # turning into numpy array to find the datapoints in each cluster
        new_centroids = update_centroids(X, clusters, k) # updating centroids
        
        centroids_change = eucleadian_distance(centroids, new_centroids) # calculating to check if centroids changed much
        if (centroids_change < tol).all():
            break   # we consider the centroids cannot be updated anymore

        centroids = new_centroids

    return centroids, clusters # returning centroids and clusters
        

# creating a toy dataset 
X, _ = datasets.make_blobs(n_samples=500, n_features=2, centers=3, random_state=42)

centroids, clusters = train_data(X, k=3)

plt.scatter(X[:,0], X[:,1], c=clusters, cmap='spring')
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='+')
plt.show()