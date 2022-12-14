import torch
import pandas as pd
import numpy as np


class Strategy():
    def __init__(self, name: str, tickers: list):
        # name of the strategy
        self.name = name

        # tickers
        self.tickers = tickers

        # cumulative returns each day
        self.last_return = 0
        self.cum_returns = {}
        # returns of each day
        self.daily_returns = {}
        # allocation of each day
        self.allocations = {}

    def _compute_returns(self, weights, data):
        daily_returns = []
        for i, w in enumerate(weights):
            ret = w * data[i]["Return"].iloc[-1]
            daily_returns.append(ret)
        # log returns
        #print(f"{self.name}: {daily_returns}")
        log_return = np.log(1+np.sum(daily_returns))
        #print(f"{self.name}: {log_return}")
        return log_return

    def _compute_weights(self, data):
        """
        Compute the weights for the next day
        store in allocation
        """
        # Warning only use data[:-1] otherwise dataleakage
        pass

    def forward(self, data):
        # compute weights for day
        daily_allocations = self._compute_weights(data)
        daily_return = self._compute_returns(daily_allocations, data)

        # save returns of the day
        self.allocations[data[0]["Date"].iloc[-1]] = daily_allocations
        self.daily_returns[data[0]["Date"].iloc[-1]] = daily_return

        # cumulative returns
        self.last_return += daily_return
        self.cum_returns[data[0]["Date"].iloc[-1]] = self.last_return


class DeepLearningStrategy(Strategy):
    def __init__(self, name: str, tickers: list, models: list, train_step: int):
        super(self.__class__, self).__init__(name, tickers)

        # store models
        self.models = models

        # re-train step
        self.train_step = train_step
        # manage model change
        self.current_year = None
        self.cpt = 0
        # define model we are using (depends on year)
        self.current_model = 0

    def _compute_weights(self, data):
        # actual year
        year = data[0]["Date"].iloc[-1].year

        # first pass
        if self.current_year is None:
            print("First year: ", year)
            self.current_year = year
        elif self.current_year != year:
            print("Changing year: ", year)
            self.current_year = year
            self.cpt += 1
            if self.cpt % self.train_step == 0:
                if len(self.models) == self.current_model+1:
                    print("Last model")
                else:
                    print("Changing model: ", self.current_model)
                    self.cpt = 0
                    self.current_model += 1

                # remove last data
        dl_data = []
        for d in data:
            # keep last 50 days and only log returns
            dl_data.append(d["Log_return"].iloc[:-2])

        # reshape data to fit model input
        dl_data = np.expand_dims(np.array(dl_data, dtype=np.float64), axis=0)
        dl_data = np.transpose(dl_data, (0, 2, 1))

        # data to tensor
        tensor = torch.from_numpy(dl_data).type(torch.Tensor).to('cpu')

        # compute weights
        out = self.models[self.current_model](tensor)
        out = out.detach().numpy()[0]

        return out
