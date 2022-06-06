import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import silhouette_score, davies_bouldin_score

N_CLUSTERS_MAX = 10
EPS = 0.05
MINPTS = 4

def kmeans(X, n):
    return KMeans(n).fit_predict(X)

def dbscan(X, eps, minpts=MINPTS):
    return DBSCAN(eps=eps, min_samples=minpts).fit_predict(X)

def euclid(a, b):
    s = 0
    for i in range(len(a)):
        s += pow(a[i] - b[i], 2)
    return sqrt(s)

def elbow(points, labels):
    clusters = np.array([points[labels == i] for i in set(labels)], dtype=object)
    centroids = np.array([cluster.mean(axis=0) for cluster in clusters])
    res = 0
    for i in range(len(clusters)):
        s = 0
        for point in clusters[i]:
            s += euclid(point, centroids[i]) ** 2
        res += s
    return res / len(clusters)

def neighbours_distances(points, n_neighbours):
    res = []
    for i in range(len(points)):
        dists = list()
        for j in range(len(points)):
            if i != j:
                dists.append(euclid(points[i], points[j]))
        dists.sort()
        dists = dists[:n_neighbours]
        res.append(np.mean(dists))
    res.sort()
    return res





data = np.genfromtxt('../data.csv', delimiter=',')
data = data[1:]
#print(data, len(data))

DBI_res = []
elbow_res = []
silhouette_res = []

x = range(2, N_CLUSTERS_MAX + 1)
for k in x:
    res = kmeans(data, k)
    DBI_res.append(davies_bouldin_score(data, res))
    elbow_res.append(elbow(data, res))
    silhouette_res.append(silhouette_score(data, res))


plt.figure(1)
plt.title("DBI Kmeans")
plt.plot(x, DBI_res)
plt.xticks(x)
plt.figure(2)
plt.title("Elbow Kmeans")
plt.plot(x, elbow_res)
plt.xticks(x)
plt.figure(3)
plt.title("Silhouette Kmeans")
plt.plot(x, silhouette_res)
plt.xticks(x)

dists = neighbours_distances(data, MINPTS)
plt.figure(4)
plt.title("Eps")
plt.plot(dists)
#plt.show()

DBI_res.clear()
silhouette_res.clear()
elbow_res.clear()
eps = [i for i in range(500, 10000, 500)]

for epsi in eps:
    res = dbscan(data, epsi)
    print(epsi, np.unique(res))
    DBI_res.append(davies_bouldin_score(data, res))
    elbow_res.append(elbow(data, res))
    silhouette_res.append(silhouette_score(data, res))

plt.figure(5)
plt.title("DBI DBSCAN")
plt.plot(eps, DBI_res)
plt.xticks(eps)
plt.figure(6)
plt.title("Elbow DBSCAN")
plt.plot(eps, elbow_res)
plt.xticks(eps)
plt.figure(7)
plt.title("Silhouette DBSCAN")
plt.plot(eps, silhouette_res)
plt.xticks(eps)

plt.show()
