import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from train.strategy import Strategy, DeepLearningStrategy
from train.strategies import RandomStrategy, EqualWeightPortfolio

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
        years = data.index.year.unique()
        first_train_year = years[train_step-1]
        data['year'] = data.index.year
        return data.loc[data["year"] > first_train_year]

    def _create_strategies(self):
        # hard code strategy for now
        # basic strategies
        # self.strategies.append(RandomStrategy('random', self.tickers))
        self.strategies.append(EqualWeightPortfolio(
            'Equal Weight Portfolio', self.tickers))

        # add deep learning strategies
        for m_name, m_list in self.models.items():
            dl_strat = DeepLearningStrategy(
                name=m_name, tickers=self.tickers, models=m_list, train_step=self.train_step)
            self.strategies.append(dl_strat)

    def _preprocess_comparison(self, df):
        # compute log returns 
        df['Return'] = df['Close'].pct_change()
        df['Log_return'] = np.log1p(df['Return'])

        # remove first training years
        df.index = pd.to_datetime(df.index, utc=True)
        df = self._reshape_dataset(df, self.train_step)

        # compute cumulative returns
        df['cum_return'] = df['Log_return'].cumsum()

        return df

    def _to_df(self, name, input_dict):
        res = pd.DataFrame(list(input_dict.items()), columns=["Date", name])
        res["Date"] = res['Date'].dt.date
        res.index = res["Date"]
        res.drop(["Date"], axis=1,inplace=True)
        return res

    def _plot_curve(self, results):
        results.plot()
        plt.show()

    def run(self, comparison_index: pd.DataFrame, timestep=50, plot=False):
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
        results = {}
        return_results = []

        # comparison index
        comparison_results = self._preprocess_comparison(comparison_index)
        index_name = 'S&P500'
        logging.debug("---------------------------------")
        logging.debug(f"Results of : {index_name}")
        expected_return = np.array(comparison_results["Log_return"]).mean()*252
        volatility = np.array(comparison_results["Log_return"]).std()*np.sqrt(252)
        logging.debug(f"Expected returns: {expected_return*100}%")
        logging.debug(f"Volatilty: {volatility*100}%")
        logging.debug(f"Sharpe Ratio: {expected_return/volatility}")
        results[index_name] = {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': (expected_return/volatility)
        }
        # compute cum sum
        # save return results
        return_results.append((index_name, dict(zip(comparison_results.index, comparison_results.cum_return))))

        # strategies results
        for strategy in self.strategies:
            logging.debug("---------------------------------")
            logging.debug(f"Results of strategy: {strategy.name}")
            expected_return = np.array(list(strategy.daily_returns.values())).mean()*252
            volatility = np.array(list(strategy.daily_returns.values())).std()*np.sqrt(252)
            logging.debug(f"Expected returns: {expected_return*100}%")
            logging.debug(f"Volatilty: {volatility*100}%")
            logging.debug(f"Sharpe Ratio: {expected_return/volatility}")
            logging.debug(f"MDD: {get_mdd(strategy.daily_returns)}")
            # save return results
            return_results.append((strategy.name, strategy.cum_returns))
            # save metrics
            results[strategy.name] = {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': (expected_return/volatility)
            }

        return_results = [self._to_df(res[0], res[1]) for res in return_results]
        df_results = pd.concat(return_results, axis=1)
        df_results.dropna(inplace=True)
        
        if plot:
            self._plot_curve(df_results)
    
        return df_results, results
