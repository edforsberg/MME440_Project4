import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

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
    plt.show()
    
def PlotAllMethods(X,k,reducedFeatures):
    pca = PCA(reducedFeatures).fit(X)
    isomap = Isomap(n_components=reducedFeatures).fit(X)
    
    X_PCA = pca.transform(X)
    X_ISO = isomap.transform(X)
    
    labels_orig, inertia_orig = Kmeans(X,k)
    labels_PCA, inertia_PCA = Kmeans(X_PCA,k)
    labels_ISO, inertia_ISO = Kmeans(X_ISO,k)
    
    PlotElbow(inertia_orig,k)
    PlotElbow(inertia_PCA,k)
    PlotElbow(inertia_ISO,k)
    
    