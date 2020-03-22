import csv
import lof
import copy

# open data file
data = []
isFirstLine = 1
with open('./static/mydata10.csv', newline='') as readfile:
    reader = csv.reader(readfile)
    for row in reader:
        # do not read the first row!
        if isFirstLine == 1:
            isFirstLine = 0
        else:
        	tmp1 = map(eval,row)
        	tmp2 = list(tmp1)
        	data.append(tmp2)
        	# empty data in csv will cause "unexpected EOF while parsing"
#print(data)
print("data loaded!")



k_list = []
k_list = lof.initialize_k_list(data)
benchmark = 1 # the gap between k and RC's size
while True:
    rare_centers = []
    print(data)
    print(k_list)
    if len(data) <=2:
        break
    if not k_list:
        break
    rare_centers = lof.outliers(data, k_list)
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
print("finish")
