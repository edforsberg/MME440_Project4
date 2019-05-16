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
