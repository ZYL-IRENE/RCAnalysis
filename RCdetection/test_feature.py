import numpy as np 
import mymds

import csv
import lof
import copy

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

data = np.array(data)
clf1 = mymds.MyMDS(2)
result = clf1.fit(data)
result = result.tolist()
print(result)