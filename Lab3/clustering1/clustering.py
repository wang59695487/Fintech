import numpy as np
import scipy.io as sio
from clustering1.plot import plot
from clustering1.todo import kmeans
from clustering1.todo import kmeans2
from clustering1.todo import spectral
from clustering1.todo import knn_graph



cluster_data = sio.loadmat('clustering1/cluster_data.mat')
X = cluster_data['X']



#----------------k-means--------------#
idx = kmeans(X, 2)
plot(X, idx, "Clustering-kmeans")

#----------------k-means++------------#

idx = kmeans2(X, 2)
plot(X, idx, "Clustering-kmeans")


#----------------谱聚类----------------#
W = knn_graph(X, 5, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 15, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 30, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 50, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 100, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 0.3)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 0.5)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 1.0)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 2.0)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 5.0)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

W = knn_graph(X, 20, 10.0)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")

