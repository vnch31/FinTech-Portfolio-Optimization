from os import listdir
from os.path import isfile, join
import json
import os


def get_filenames(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    print(onlyfiles)
    return onlyfiles


def load_models_config(nb_tickers, config_path='models_config.json'):
    config = None
    with open(config_path, 'r') as fd:
        config = json.load(fd)

    if not config:
        print("Impossible to load models_config.json")
        os._exit(1)

    for k, v in config.items():
        v['params']['input_dim'] = nb_tickers
        v['params']['output_dim'] = nb_tickers

    return config
