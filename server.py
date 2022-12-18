from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import threading

from data import dataloader
from utils import get_filenames
from main import main

app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'application/json'

@app.route("/create_dataset", methods=['POST', 'OPTIONS'])
def create_dataset():
    if (request.method == 'OPTIONS'):
        return "ok", 200
    data = request.get_json()
    tickers = data.get('tickers', '')
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')

    dataloader.get_data_yfinance(tickers, start_date, end_date, '1d')
    filename = f"dataset_{start_date}_{end_date}_{tickers.replace(' ', '_')}.csv"
    return jsonify({'status': 'ok','name': filename})

@app.route("/get_datasets", methods=['POST'])
def get_datasets():
    if (request.method == 'OPTIONS'):
        return "ok", 200
    names = get_filenames('data/downloads')
    return jsonify({ 'status':'ok', 'datas':names})


@app.route("/get_models", methods=['GET'])
def get_models():
    if (request.method == 'OPTIONS'):
        return "ok", 200
    names = get_filenames('models/')
    names = [n.replace('.py','') for n in names]
    names = [n for n in names if n!='__init__']
    
    return jsonify({ 'status':'ok', 'datas':names})


@app.route("/train", methods=['POST', 'OPTIONS'])
def train():
    if (request.method == 'OPTIONS'):
        return "ok", 200
    # get all parameters
    data = request.get_json()
    dataset_file_name = data.get('dataset', '')
    name = data.get('name', '')
    timestep = data.get('timestep', '')
    batch_size = data.get('batch_size', '')

    dataset_file_name = dataset_file_name.split('.')[0]
    dataset_file_name = dataset_file_name.split('_')
    startDate = dataset_file_name[1]
    endDate = dataset_file_name[2]
    tickers = ' '.join(dataset_file_name[3:])
    # async call to train function
    thread = threading.Thread(target=lambda: main(name, tickers, startDate, endDate, '1d', timestep, 2,batch_size))
    thread.daemon = True
    thread.start() # swap with thread.run()
    

    return jsonify({'status': 'ok'})

@app.route('/get_names', methods=['POST'])
def get_names_by_dataset():
    # Get dataset name
    data = request.get_json()
    dataset_name = data.get('dataset_name', '')
    dataset_name = dataset_name.replace('.csv', '')
    # Get config.json in the right folder
    try:
        f = open("save/" +dataset_name+ '/info.json', "r")
        return f.read()
    except:
        return []
    # Return names

@app.route('/get_results_chart', methods=['POST', 'OPTIONS'])
def get_results_chart():
    if (request.method == 'OPTIONS'):
        return "ok", 200
    data = request.get_json()
    dataset_name = data.get('dataset_name', '')
    model_name = data.get('model_name')
    print(dataset_name)
    print(model_name)

    # open the json file
    f = open(f"save/{dataset_name}/results_{model_name}/chart.json", "r")
    return f.read()

@app.route('/get_results_metrics', methods=['POST'])
def get_results_metrics():
    if (request.method == 'OPTIONS'):
        return "ok", 200
    data = request.get_json()
    dataset_name = data.get('dataset_name', '')
    model_name = data.get('model_name')

    # open the json file
    f = open(f"save/{dataset_name}/results_{model_name}/metrics_results.json", "r")
    return f.read()