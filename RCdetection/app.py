from flask import Flask, render_template
import csv


app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    data = []
    isFirstLine = 1
    with open('./static/mydata10.csv', newline='') as readfile:
        reader = csv.reader(readfile)
        for row in reader:
            # do not read the first row!
            if isFirstLine == 1:
                isFirstLine = 0
            else:
                tmp1 = map(eval, row)
                tmp2 = list(tmp1)
                data.append(tmp2)
            # empty data in csv will cause "unexpected EOF while parsing"
    # print(data)
    print("data loaded!")
    data = {"this": data}
    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run()
