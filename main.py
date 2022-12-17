# imports
import pandas as pd
import numpy as np
import argparse
import logging
import sys
import json
import os
import torch

# local imports
from data import dataloader, autotickers
from train.trainer import Trainer

# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def main(name, tickers, start_date, end_date, interval, timestep, train, batch_size, modelsconfig, device='cpu'):
    # logs parameters
    logging.debug(
        "Creating, training and testing models with the following parameters:")
    logging.debug(f"""{tickers}
from {start_date} to {end_date} with {interval} interval
retrain every {train} year with {timestep} days
""")

    # download or load data
    logging.debug("Getting data")
    filename = f"dataset_{start_date}_{end_date}_{tickers.replace(' ', '_')}.csv"
    df = dataloader.get_data_yfinance(tickers=tickers, start_date=start_date, end_date=end_date, interval=interval)

    # setting training device 
    device = 'cpu'

    if torch.cuda.is_available(): # gpu
        device = 'cuda'
        
    logging.debug(f"Training and testing with the device: {device}")

    trainer = Trainer(name=name, dataset_name=filename, data=df, train_step=train, timestep=timestep, batch_size=batch_size, modelsconfig=modelsconfig)

    trainer.run(models=['lstm', 'tcn', 'rnn', 'gru', 'transformer'])


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        description="Dataloader options, if no argument given config.json will be used"
    )
    parser.add_argument('-c', '--config', help="Config File (default: config.json)")
    parser.add_argument('-m', '--modelsconfig', help="Models Config File (default: models_config.json)")
    parser.add_argument('-n', '--name', help="Name of the model")
    parser.add_argument('-a', '--autotickers', help="Using tickers from Sentiment Analysis (Max 20), required: start & end", default='5', type=int)
    parser.add_argument('-t', '--tickers', nargs='+',
                        help="Tickers to use, required: start & end")
    parser.add_argument('-s', '--start', help="Start date : YYYY-MM-DD")
    parser.add_argument('-e', '--end', help="End date : YYYY-MM-DD")
    parser.add_argument('-i', '--interval',
                        help="Fetch data by interval", default='1d')
    parser.add_argument(
        '-r', '--retrain', help='Retrain interval', default=1, type=int)
    parser.add_argument(
        '--timestep', help="Timestep for the weight prediction", default=50, type=int)
    parser.add_argument('-d', '--device', help='Device to use', default='cpu')
    args = parser.parse_args()

    # variables
    config = {}

    # check args
    if args.autotickers or args.tickers:
        if args.start == None or args.end == None:
            logging.error(f"Tickers / AutoTickers requires start & end date ")
            exit(1)

    # use file configuration
    file_config = 'config.json'
    if args.config:
        file_config = args.config
    logging.debug(f"use config: " + file_config)
    try:
        with open(file_config, 'r') as fd:
            config = json.load(fd)
    except Exception as e:
        logging.error(f"An error occured while reading the file")
        logging.error(str(e))
        exit(1)

    # file configuration can be override by manual parameters
    if (args.start and args.end):
        # manually set tickers
        if args.tickers:
            config['tickers'] = ' '.join(args.tickers)

        # check use autotickers to get uncorrelated tickers
        elif args.autotickers:
            tickers = autotickers.get_auto_tickers(args.start, args.end, args.autotickers)        
            config['tickers'] = ' '.join(tickers)

    if (args.start and args.end):
        config['start_date'] = args.start
        config['end_date'] = args.end
        config['interval'] = args.interval
        config['timestep'] = args.timestep
        config['train'] = args.retrain
        config['device'] = args.device


    models_config = 'models_config.json'
    if args.modelsconfig:
        models_config = args.modelsconfig

    config['modelsconfig'] = models_config
    logging.debug(f"use models config: " + models_config)

    main(**config)
