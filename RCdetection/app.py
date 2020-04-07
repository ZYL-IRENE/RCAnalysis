from flask import Flask, render_template
import csv
import copy
import numpy as np
import mymds
import lof
import feature_explore

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
    (feature_position, feature_position_min, feature_position_max, feature_variance) = feature_explore.feature_analysis(
        data)

    # initialize LOF class
    k_list = []
    k_list = lof.initialize_k_list(data)
    benchmark = 1  # the gap between k and RC's size

    l = lof.LOF(data, k_list)
    # calculate data position
    normalized_data_array = np.array(l.instances)
    (data_position, data_position_min, data_position_max) = mymds.get_position(normalized_data_array, 2)

    return_data = {"raw_data": data, "feature_position": feature_position, "feature_position_min": feature_position_min,
                   "feature_position_max": feature_position_max, "feature_variance": feature_variance,
                   "feature_name": feature_name, "data_position": data_position, "data_position_min": data_position_min,
                   "data_position_max": data_position_max}
    return render_template('index.html', data=return_data)


if __name__ == '__main__':
    app.run()
