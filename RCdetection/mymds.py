import numpy as np
import matplotlib.pyplot as plt
 
class MyMDS:
    def __init__(self,n_components):
        self.n_components=n_components
    
    def fit(self,data):
        m,n=data.shape
        dist=np.zeros((m,m))
        disti=np.zeros(m)
        distj=np.zeros(m)
        B=np.zeros((m,m))
        for i in range(m):
            dist[i]=np.sum(np.square(data[i]-data),axis=1).reshape(1,m)
        for i in range(m):
            disti[i]=np.mean(dist[i,:])
            distj[i]=np.mean(dist[:,i])
        distij=np.mean(dist)
        for i in range(m):
            for j in range(m):
                B[i,j] = -0.5*(dist[i,j] - disti[i] - distj[j] + distij)
        lamda,V=np.linalg.eigh(B)
        index=np.argsort(-lamda)[:self.n_components]
        diag_lamda=np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
        V_selected=V[:,index]
        Z=V_selected.dot(diag_lamda)
        return Z

def get_position(data,d):
    data = np.array(data)
    clf1 = MyMDS(d)
    result = clf1.fit(data)
    result = result.tolist()
    # min and max of feature position
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf") 
    max_y = float("-inf") 
    for i, position in enumerate(result):
        min_x = min(min_x, position[0])
        min_y = min(min_y, position[1])
        max_x = max(max_x, position[0])
        max_y = max(max_y, position[1])
        position.append(i)

    data_position_min = [min_x, min_y]
    data_position_max = [max_x, max_y]

    return result, data_position_min, data_position_max