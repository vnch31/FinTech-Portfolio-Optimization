import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from train.strategy import Strategy, DeepLearningStrategy
from train.strategies import RandomStrategy, Allocation1Strategy

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

# --- Utils function
def get_mdd(daily):
    np_array_daily = np.array(list(daily.values()))
    peak_lower = np.argmax(np.maximum.accumulate(np_array_daily) - np_array_daily)
    peak_upper = np.argmax(np_array_daily[:peak_lower])
    return (np_array_daily[peak_lower] - np_array_daily[peak_upper]) / np_array_daily[peak_upper]


class Backtest():
    def __init__(self, dataset: pd.DataFrame, models: dict, tickers: list, train_step: int):
        self.dataset = self._reshape_dataset(dataset.copy(), train_step)
        self.models = models
        self.tickers = tickers
        self.train_step = train_step
        self.strategies = []
        
        self._create_strategies()

    def _reshape_dataset(self, data: pd.DataFrame, train_step: int):
        years = data.Date.dt.year.unique()
        first_train_year = years[train_step-1]
        data['year'] = pd.to_datetime(data["Date"], "%Y")
        return data.loc[data["year"].dt.year > first_train_year]

    def _create_strategies(self):
        # hard code strategy for now
        # basic strategies
        self.strategies.append(RandomStrategy('random', self.tickers))
        self.strategies.append(Allocation1Strategy(
            'equal strategy', self.tickers))

        # add deep learning strategies
        for m_name, m_list in self.models.items():
            dl_strat = DeepLearningStrategy(
                name=m_name, tickers=self.tickers, models=m_list, train_step=self.train_step)
            self.strategies.append(dl_strat)

    def _to_df(self, name, input_dict):
        res = pd.DataFrame(list(input_dict.items()), columns=["Date", name])
        res["Date"] = pd.to_datetime(res["Date"])
        res.index = pd.DatetimeIndex(res["Date"])
        res = res.drop("Date", axis=1)
        return res

    def _plot_curve(self, results):
        results.plot()
        plt.show()

    def run(self, timestep=50, plot=True):
        """
        Loop through all the data and give it to the strategy
        """
        logging.debug("Backtesting strategies")
        # loop through dataset with timestep
        for i in range(timestep+1, len(self.dataset[self.dataset["Ticker"] == self.tickers[0]])-1):
            # data of the day: array of data for all tickers
            data_day_i = []
            for ticker in self.tickers:
                data_ticker = self.dataset[self.dataset["Ticker"]
                                           == ticker].iloc[i-timestep-1:i+1]
                data_day_i.append(data_ticker)

            # pass to all strategy
            for strategy in self.strategies:
                strategy.forward(data_day_i)

        # show results
        logging.debug("Backtesting results")
        results = []
        for strategy in self.strategies:
            logging.debug("---------------------------------")
            logging.debug(f"Results of strategy: {strategy.name}")
            expected_return = np.array(list(strategy.daily_returns.values())).mean()*252
            volatility = np.array(list(strategy.daily_returns.values())).std()*np.sqrt(252)
            logging.debug(f"Expected returns: {expected_return*100}%")
            logging.debug(f"Volatilty: {volatility*100}%")
            logging.debug(f"Sharpe Ratio: {expected_return/volatility}")
            logging.debug(f"MDD: {get_mdd(strategy.daily_returns)}")
            results.append((strategy.name, strategy.cum_returns))

        results = [self._to_df(res[0], res[1]) for res in results]
        df_results = pd.concat(results, axis=1)
        self._plot_curve(df_results)
    
        return df_results
