import csv
import copy
import numpy as np 
import json

import mymds
import lof
import feature_explore


# open data file
data = []
isFirstLine = 1
with open('./static/shuttle20.csv', newline='') as readfile:
    reader = csv.reader(readfile)
    for row in reader:
        # do not read the first row!
        if isFirstLine == 1:
            tmp1 = map(eval,row)
            tmp2 = list(tmp1)
            feature_name = tmp2
            isFirstLine = 0
        else:
        	tmp1 = map(eval,row)
        	tmp2 = list(tmp1)
        	data.append(tmp2)
        	# empty data in csv will cause "unexpected EOF while parsing"
#print(data)
print("data loaded!")

# calculate feature distribution
(feature_position,feature_position_min,feature_position_max,feature_variance) = feature_explore.feature_analysis(data)


#calculate rare category
k_list = []
k_list = lof.initialize_k_list(data)
benchmark = 1 # the gap between k and RC's size

l = lof.LOF(data, k_list)

normalized_data_array = np.array(l.instances)
(data_position, data_position_min, data_position_max) = mymds.get_position(normalized_data_array,2)
print(data_position)



'''
rare_centers = lof.outliers(data, k_list)
print(rare_centers)

while True:
    rare_centers = []
    if len(data) <=3:
        break
    if not k_list:
        break
    rare_centers = lof.outliers(data, k_list)
    print(rare_centers)
    if rare_centers:
        # center point is the outlier with the max confidence
        center = rare_centers[0] 
        center_index = center["index"]
        center_kinf = center["k_inf"]
        # center point and its k-neighbours
        neighbours = lof.get_neighbours(center_kinf, data[center_index], data) 
        # put togather as RC
        category = copy.deepcopy(neighbours)
        category.append(data[center_index])

        # visualize neighbour
        # expand neighbour
        # mark this RC

        # remove this category from data
        data.remove(data[center_index])
        for neighbour in neighbours:
            data.remove(neighbour)
        
        k_list = lof.refine_k_list(k_list, data, category, benchmark)
    else:
        break
'''
print("finish")
