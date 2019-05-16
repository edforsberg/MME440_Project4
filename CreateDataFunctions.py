import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import math
import sklearn.datasets as dts
#from sklearn.utils import shuffle

def normal_data(nrObservations, nrFeatures, seed):
    """
    Generates a standard normally distibuted dataset with "nrFeatures" dimensions and "nrObservations" observations
    
    :param int nrFeatures: Number of dimensions in the dataset
    :param int nrDataPoints: Number of observations in the dataset
    :return ndarray data: numpy array with data  (one observation per row)
    """
    np.random.seed(seed)
    data = np.random.normal(loc=0.0, scale=1.0, size=[nrObservations, nrFeatures])
    
    return data

def normal_data_varying_range(nrObservations, nrFeatures, seed, mean_range = [0,10], variance_range = [0,5]):
    """
    Generates a standard normally distibuted dataset with "nrFeatures" dimensions and "nrObservations" observations. In this dataset each dimension has a randomized mean in the interval [0,10] (unless specified), and a rendomized variance in the interval [0,5] (unless specified)
    
    :param int nrFeatures: Number of dimensions in the dataset
    :param int nrDataPoints: Number of observations in the dataset
    :param list mean_range: The range for which values of the mean can take (default [0,10])
    :param list spread_range: The range around the mean that values of the observations can take (default [0,5])
    :return ndarray data: numpy array with data  (one observation per row)
    """
    np.random.seed(seed)
    data = np.empty([nrObservations, nrFeatures])
    for i in range(0,nrFeatures):
        data[:, i] = np.random.normal(loc=np.random.uniform(low=mean_range[0], high=mean_range[1]), scale=np.random.uniform(low=variance_range[0], high=variance_range[1]), size=[nrObservations, 1])[:,0]
    
    return data


def uniform_data(nrObservations, nrFeatures, seed):
    """
    Generates a uniformly distibuted dataset with "nrFeatures" dimensions and "nrObservations" observations, all features will lie in the range [0,1]
    
    :param int nrFeatures: Number of dimensions in the dataset
    :param int nrDataPoints: Number of observations in the dataset
    :return ndarray data: numpy array with data  (one observation per row)
    """
    np.random.seed(seed)
    data = np.random.uniform(size = (nrObservations, nrFeatures))    
    return data

def normal_data_with_cov(nrObservations, nrFeatures, seed):
    
    np.random.seed(seed)
    cov_mat = dts.make_spd_matrix(nrFeatures)
    mean_array = np.zeros(nrFeatures)
    data = np.random.multivariate_normal(mean = mean_array, cov = cov_mat, size=nrObservations)
    
    return data
