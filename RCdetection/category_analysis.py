import numpy as np 
import mymds

import csv
import lof
import copy

def category_mean_relative(data,feature_mean):
    data_array = np.array(data)
    data_mean = np.mean(data_array, axis=0)
    data_mean_relative = data_mean - feature_mean
    data_mean_relative = data_mean_relative.tolist()
    return data_mean_relative

