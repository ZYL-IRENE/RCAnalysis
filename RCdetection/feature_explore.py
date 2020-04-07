import numpy as np 
import mymds

import csv
import lof
import copy

def feature_analysis(data):
    data_array = np.array(data)
    #feature  correlations
    feature_cor = np.corrcoef(data_array.T) 
    feature_distance = 1-feature_cor
    #MDS feature position 
    clf1 = mymds.MyMDS(2)
    feature_position = clf1.fit(feature_distance)
    feature_position = feature_position.tolist()
    # min and max of feature position
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf") 
    max_y = float("-inf") 
    for position in feature_position:
        min_x = min(min_x, position[0])
        min_y = min(min_y, position[1])
        max_x = max(max_x, position[0])
        max_y = max(max_y, position[1])
    feature_position_min = [min_x, min_y]
    feature_position_max = [max_x, max_y]
    #feature variance 
    k_list = []
    k_list = lof.initialize_k_list(data)
    l = lof.LOF(data, k_list)
    normalized_data_array = np.array(l.instances)
    feature_variance = []
    for feature in normalized_data_array.T:
        feature_variance.append(np.var(feature))

    return feature_position,feature_position_min,feature_position_max,feature_variance