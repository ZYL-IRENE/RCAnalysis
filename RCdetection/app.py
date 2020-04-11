from flask import Flask, render_template, request, jsonify
import csv
import copy
import numpy as np
import mymds
import lof
import feature_explore
import category_analysis as ca

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # open data file
    data = []
    isFirstLine = 1
    with open('./static/shuttle20.csv', newline='') as readfile:
        reader = csv.reader(readfile)
        for row in reader:
            # do not read the first row!
            if isFirstLine == 1:
                feature_name = row
                isFirstLine = 0
            else:
                tmp1 = map(eval, row)
                tmp2 = list(tmp1)
                data.append(tmp2)
            # empty data in csv will cause "unexpected EOF while parsing"
    # print(data)
    print("data loaded!")

    # calculate feature distribution
    (feature_position, feature_position_min, feature_position_max, feature_variance, feature_mean) = feature_explore.feature_analysis(
        data)

    # initialize LOF class
    k_list = []
    k_list = lof.initialize_k_list(data)
    benchmark = 1  # the gap between k and RC's size

    l = lof.LOF(data, k_list)
    # calculate data position
    normalized_data_array = np.array(l.instances)
    (data_position, data_position_min, data_position_max) = mymds.get_position(normalized_data_array, 2)

    '''print("start to compute RC")

    rare_centers = lof.outliers(data, k_list)

    center = rare_centers[0] 
    center_index = center["index"]
    center_kinf = center["k_inf"]
    # center point and its k-neighbours
    neighbours = lof.get_neighbours(center_kinf, data[center_index], data) 
    # put togather as RC
    category = copy.deepcopy(neighbours)
    category.append(data[center_index])
    #print(category)

    category_mean = ca.category_mean_relative(category,feature_mean)'''
    category_mean = [-3.066666666666663, -0.8555555555555556, -0.8944444444444457, -0.1222222222222222, 3.811111111111117, 5.094444444444445, 2.1611111111111114, -4.916666666666664, -7.033333333333333]
    print(category_mean)

    return_data = {"raw_data": data, "feature_position": feature_position, "feature_position_min": feature_position_min,
                   "feature_position_max": feature_position_max, "feature_variance": feature_variance,
                   "feature_name": feature_name, "data_position": data_position, "data_position_min": data_position_min,
                   "data_position_max": data_position_max, "category_mean":category_mean}



        ######### POST request #########
    '''if request.method == "POST":
        print("This is post")
        list_add = []
        getdata = json.loads(request.get_data())["data"]
        print(getdata)'''

    print("This is post")
    recv_data = request.get_data()
    print(recv_data)


    return render_template('index.html', data=return_data)

'''
@app.route('/new',methods=['POST'])  
def new():
    recv_data = request.get_data() 
    print(recv_data)
   
    return recv_data                 
'''
if __name__ == '__main__':
    app.run()
