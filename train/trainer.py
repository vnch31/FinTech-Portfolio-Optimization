import logging
import os
import json
import sys

import pandas as pd
import numpy as np
# pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader

from train import losses
from train.optimizer import Optimizer
from train.backtest import Backtest
from models.lstm import LSTM
from models.gru import GRU
from models.rnn import RNN

# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class Trainer():
    def __init__(self, data, train_step, timestep, batch_size=64, device='cpu'):
        # backtesting data
        self.data = data
        self.tickers = data.Ticker.unique()
        # defined later
        self.years = None

        # training, testing parameters paramaters
        self.train_step = train_step
        self.timestep = timestep
        self.batch_size = batch_size
        self.device = device

        # training data
        self.train_dataset = []

        # preprocess data
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        # process data
        # sort by tickers and date
        self.data = self.data.sort_values(by=['Date', 'Ticker'])

        # add return and log return values
        for ticker in self.data.Ticker.unique():
            # return and log return
            self.data.loc[self.data['Ticker'] == ticker,
                          'Return'] = self.data.loc[self.data['Ticker'] == ticker, 'Close'].pct_change()
            self.data.loc[self.data['Ticker'] == ticker, 'Log_return'] = np.log1p(
                self.data.loc[self.data['Ticker'] == ticker, 'Return'])

        # remove NaN values
        self.data = self.data.dropna()

        # convert date to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        # define all years
        self.years = self.data.Date.dt.year.unique()

        # split dataset by year
        self.train_dataset = self._split_years(self.data.copy())

        # create numpy dataset
        self.train_dataset = [self._create_dataset(
            df_years, 'Log_return') for df_years in self.train_dataset]

    def _create_dataset(self, df: pd.DataFrame, feature_name):
        # TODO: add more features, only 1 feature now
        # get all dates
        dates = df['Date'].unique()

        # create numpy array
        dataset = np.zeros([len(dates), len(self.tickers), 1])

        # use apply or other method (faster than iterrows)
        for _, row in df.iterrows():
            # get index corresponding to date
            index_date = np.where(dates == row.Date)[0][0]

            # get index corresponding to ticker
            index_ticker = np.where(self.tickers == row.Ticker)[0][0]

            # get features
            features = row[feature_name]

            # add to dataset
            dataset[index_date, index_ticker] = features

        # return dataset
        return dataset

    def _split_years(self, dt):
        dt['year'] = dt['Date'].dt.year
        return [dt[dt['year'] == y] for y in dt['year'].unique()]

    def _split_train_valid(self, dataset, train=0.8):
        # train split
        train_valid_idx = int(np.round(len(dataset)*train))

        # split train-valid
        dataset_train = dataset[:train_valid_idx]
        dataset_valid = dataset[train_valid_idx:]

        return dataset_train, dataset_valid

    def _create_dataset_lookback(self, dataset: np.array, timestep=50):
        # working vars
        X_data = np.array([dataset[0: 0+timestep].reshape(timestep, -1)])
        y_data = np.array([dataset[1: 0+timestep+1].reshape(timestep, -1)])

        # create sequences with length timestep (30)
        for index in range(1, len(dataset) - (timestep + 1)):

            # create X with timestep length
            X_data = np.vstack(
                (X_data, np.array([dataset[index: index+timestep].reshape(timestep, -1)])))

            # create y with timestep+1 data
            y_data = np.vstack(
                (y_data, np.array([dataset[index+1:index+timestep+1].reshape(timestep, -1)])))

        return X_data, y_data

    def _make_dataloader(self, X, y):
        # to tensors
        X = torch.from_numpy(X).type(torch.Tensor).to(self.device)
        y = torch.from_numpy(y).type(torch.Tensor).to(self.device)

        # make pytorch dataset
        pt_dataset = TensorDataset(X, y)

        # make dataloader
        dataloader = DataLoader(pt_dataset)

        return dataloader

    def _train_test(
        self,
        model_name: str,
        optimizer_name: str,
        hyper_params: dict,
        params: dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model=None
    ):
        if model is None:
            # model
            model = None
            if model_name.lower() == 'lstm':
                logging.debug(f"[+] Creating LSTM model")
                model = LSTM(**params).to(self.device)
            elif model_name.lower() == 'rnn':
                logging.debug(f"[+] Creating RNN model")
                model = RNN(**params).to(self.device)
            elif model_name.lower() == 'gru':
                logging.debug(f"[+] Creating GRU model")
                model = GRU(**params).to(self.device)
            # elif model_name.lower() == 'encoder':
            #     logging.debug("[+] Creating Transformer Encoder model")
            #     model = TransformerEncoder(**params).to(self.device)
            # elif model_name.lower() == 'tcn':
            #     logging.debug("[+] Creating TCN model")
            #     model = TCN(**params).to(device)

            if not model:
                logging.error("[-] An error occured, model is None")
                os.exit(1)

        # optimizer
        logging.debug(f"[+] Setting {optimizer_name} optimizer")
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=hyper_params['learning_rate'])
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=hyper_params['learning_rate'], momentum=0.9)
        elif optimizer_name.lower() == 'adadelta':
            optimizer = torch.optim.Adadelta(
                model.parameters(), lr=hyper_params['learning_rate'])

        # scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)

        # trainer
        opti = Optimizer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=losses.sharpe_ratio_loss_github,
            train_loader=train_loader,
            valid_loader=valid_loader
        )

        # train model
        logging.debug(f"[+] Training model...")
        opti.train(n_epochs=hyper_params['epochs'])

        # plot losses
        # opti.plot_losses()

        return model, opti

    def run(self, models=['lstm']):
        # create dictionnary of models
        all_models = {}

        for model in models:
            # list of models through years for backtesting
            all_models[model] = []

        # load model configuration
        with open('models_config.json', 'r') as fd:
            models_config = json.load(fd)

        # ----- Training -----
        # iter through every year
        for i in range(0, len(self.train_dataset)-self.train_step, self.train_step):
            # year training
            logging.debug(
                f"Training from {self.years[i]}, {self.years[i+self.train_step-1]}")
            # train step = 2
            if i == 0:
                # split every train_step years (2)
                train_valid_set = np.concatenate(
                    (self.train_dataset[0], self.train_dataset[0+self.train_step-1]))
            else:
                # split every_step + 50 last days of the previous years to build dataset
                # i -> i+step
                train_valid_set = np.concatenate(
                    (self.train_dataset[i-1][-self.timestep:], self.train_dataset[i], self.train_dataset[i+self.train_step-1]))

            # train - valid split
            train_set, valid_set = self._split_train_valid(
                train_valid_set, train=0.8)

            # create lookback window
            X_train, y_train = self._create_dataset_lookback(
                train_set, timestep=self.timestep)
            X_valid, y_valid = self._create_dataset_lookback(
                valid_set, timestep=self.timestep)

            # make dataloader
            train_dataloader = self._make_dataloader(X_train, y_train)
            valid_dataloader = self._make_dataloader(X_valid, y_valid)

            # iter through models
            for model_name in models:
                # first iteration, create the model
                if len(all_models[model_name]) == 0:
                    model, _ = self._train_test(model_name=model_name, optimizer_name='sgd', train_loader=train_dataloader,
                                                valid_loader=valid_dataloader, params=models_config[model_name]['params'], hyper_params=models_config[model_name]['hyper_params'])
                    all_models[model_name].append(model)
                else:
                    model, _ = self._train_test(model=all_models[model_name][-1], model_name=model_name, optimizer_name='sgd', train_loader=train_dataloader,
                                                valid_loader=valid_dataloader, params=models_config[model_name]['params'], hyper_params=models_config[model_name]['hyper_params'])
                    all_models[model_name].append(model)

        # ---- Test -----
        backtest = Backtest(self.data.copy(), all_models, self.tickers, self.train_step)
        backtest.run(self.timestep)

        # ----- Save -----













