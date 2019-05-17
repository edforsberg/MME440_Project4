import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

#def k maeans
def Kmeans(X,k=2):
    # Clusters data into k clusters, where k can be a vector of different integers, 
    # in which case clustering is made for each k.
    if isinstance(k, int):
        kmeans = KMeans(n_clusters=k).fit(X)
        return kmeans.labels_, kmeans.inertia_
    
    labels = np.zeros([X.shape[0],len(k)])
    inertia = np.zeros(len(k))
    for i in range(len(k)):
        kmeans = KMeans(n_clusters=k[i]).fit(X)
        labels[:,i] = kmeans.labels_
        inertia[i] = kmeans.inertia_
    
    return labels, inertia


def PlotElbow(inertia,k):
    plt.plot(k,inertia,'.-')