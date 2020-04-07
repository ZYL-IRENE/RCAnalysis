import numpy as np 

a = [[1,2,3],[2,4,6],[3,6,9],[3,5,8]]
a = np.array(a)

b = np.corrcoef(a.T) 
print(b)