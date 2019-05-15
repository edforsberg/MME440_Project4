ximport matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import math
#from sklearn.utils import shuffle

def normal_data(nrObservations, nrFeatures):
    """
    Generates a standard normally distibuted dataset with "nrFeatures" dimensions and "nrObservations" observations
    
    :param int nrFeatures: Number of dimensions in the dataset
    :param int nrDataPoints: Number of observations in the dataset
    :return data: numpy array with data  (one observation per row)
    """
    data = np.random.normal(loc=0.0, scale=1.0, size=[nrObservations, nrFeatures])
    
    return data





def CreateData1(nrDataPoints = 1000, nrFeatures = 30):
    
    return data
