import numpy as np
import json
from flask import Flask, request, make_response
from flask_cors import CORS
from model import predictData, train_data as td, addXAndy, else_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    print('ok')
    return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict_data():
    print(request.json)
    X = np.asarray(request.json.get('datas').split(',')).astype('float64').reshape(1, -1)
    print(X, X.shape)
    y_predict = predictData(X)
    print(y_predict)
    return json.dumps(y_predict)


@app.route('/train', methods=['GET'])
def train_data():
    # print(int(request.args.get('train')))
    al_acc_list = td(int(request.args.get('train')))
    data = else_model()
    data['al'] = al_acc_list
    return json.dumps(data)


@app.route('/add', methods=['POST'])
def add_new_data():
    print(request.json)
    xd = np.asarray(request.json.get('x').split(',')).astype('float64')
    yd = np.asarray([int(request.json.get('y'))]).astype('int')
    print(xd, yd)
    addXAndy(xd, yd)
    return 'ok'


if __name__ == '__main__':
    app.run()
