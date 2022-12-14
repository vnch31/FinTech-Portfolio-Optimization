import numpy as np

from train.strategy import Strategy

class RandomStrategy(Strategy):
    def _compute_weights(self, data):
        return np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]

class Allocation1Strategy(Strategy):
    def _compute_weights(self, data):
        return np.full(len(self.tickers), (1/len(self.tickers)))