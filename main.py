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


def main(name, tickers, start_date, end_date, interval, timestep, train, batch_size, device='cpu'):
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

    trainer = Trainer(name=name, dataset_name=filename, data=df, train_step=train, timestep=timestep, batch_size=batch_size)

    # trainer.run(models=['lstm', 'tcn', 'rnn', 'gru', 'transformer'])
    trainer.run(models=['lstm', 'transformer'])


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        description="Dataloader options, if no argument given config.json will be used"
    )
    parser.add_argument('-c', '--config', help="Config File")
    parser.add_argument('-m', '--modelconfig', help="Model Config File")
    parser.add_argument('-n', '--name', help="Name of the model")
    parser.add_argument('-a', '--autotickers', help="Using tickers from Sentiment Analysis (Max 20)")
    parser.add_argument('-z', '--numtickers', help="Number of tickers selected from Sentiment Analysis (Max 20), Default: 5", default='5', type=int)
    parser.add_argument('-t', '--tickers', nargs='+',
                        help="Tickers to retrieve (will override auto-tickers)")
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

    # check for args
    if (args.tickers and args.start and args.end):
        config['tickers'] = ' '.join(args.tickers)

    if (args.start and args.end):
        config['start_date'] = args.start
        config['end_date'] = args.end
        config['interval'] = args.interval
        config['timestep'] = args.timestep
        config['train'] = args.retrain
        config['device'] = args.device

    # use file configuration
    else:
        file_config = 'config.json'
        if args.config:
            file_config = args.config
        logging.debug(f"use config:" + file_config)
        try:
            with open(file_config, 'r') as fd:
                config = json.load(fd)
        except Exception as e:
            logging.error(f"An error occured while reading the file")
            logging.error(str(e))
            os.exit(1)

    # TODO: check args

    main(**config)
