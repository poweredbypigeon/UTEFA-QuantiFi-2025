"""
UTEFA QuantiFi - Combined Algorithm and Backtesting Script

This script combines the functionality of `algorithm.py` and `backtest.py` into a single file.
It allows for hyperparameter tuning by running the simulation repeatedly with different parameters.
"""

import csv
from typing import Dict, List
from statsmodels.tsa.arima.model import ARIMA
from contextlib import contextmanager
from sklearn.covariance import LedoitWolf
import threading
import numpy as np 

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.2


# Define Market, Portfolio, Context, and update_portfolio as in algorithm.py
class Market:
    transaction_fee = 0.005

    def __init__(self) -> None:
        self.stocks = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }

class Portfolio:
    def __init__(self) -> None:
        self.shares = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }
        self.cash = 100000.0

    def evaluate(self, curMarket: Market) -> float:
        total_value = self.cash
        for stock_name, num_shares in self.shares.items():
            total_value += num_shares * curMarket.stocks[stock_name]
        return total_value

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]
        if cost > self.cash:
            raise ValueError("Insufficient cash to buy shares")
        self.shares[stock_name] += shares_to_buy
        self.cash -= cost

    def sell(self, stock_name: str, shares_to_sell: float, curMarket: Market) -> None:
        if shares_to_sell > self.shares[stock_name]:
            raise ValueError("Insufficient shares to sell")
        proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        self.shares[stock_name] -= shares_to_sell
        self.cash += proceeds

class Context:
    def __init__(self) -> None:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.pct_change = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day = 0

def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context, hyperparameters: Dict):
    # data update. 
    for stock in curMarket.stocks:
        context.price_history[stock].append(curMarket.stocks[stock])
        context.pct_change[stock].append(
            (curMarket.stocks[stock] - context.price_history[stock][-2]) / context.price_history[stock][-2]
            if len(context.price_history[stock]) > 1 else 0)
    # strategy
    model = ARIMA(context.pct_change, order=(hyperparameters["d"], hyperparameters["p"], hyperparameters["q"]))
    fitted = model.fit()
    mu_next = model.forecast(steps=hyperparameters["steps"]).iloc[0]
    resid = model.resid 
""" 
    recent_returns = context.pct_change[stock].iloc[-lookback:]

    # Apply exponential weighting if specified
    if ewma_alpha is not None:
        weights = np.exp(-ewma_alpha * np.arange(len(recent_returns))[::-1])
        weights /= weights.sum()  # Normalize weights
        weighted_returns = recent_returns.mul(weights, axis=0)
    else:
        weighted_returns = recent_returns

    # Fit Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    shrunk_cov = lw.fit(weighted_returns).covariance_

    if context.day == 0:
        for stock in curMarket.stocks:
            max_shares = curPortfolio.cash / (curMarket.stocks[stock] * (1 + Market.transaction_fee))
            curPortfolio.buy(stock, max_shares / 5, curMarket)
    context.day += 1 """

# Backtesting functions
def load_price_data(csv_file: str) -> Dict[str, List[float]]:
    price_data = {"Stock_A": [], "Stock_B": [], "Stock_C": [], "Stock_D": [], "Stock_E": []}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for stock in price_data.keys():
                price_data[stock].append(float(row[stock]))
    return price_data

def update_market(market: Market, day: int, price_data: Dict[str, List[float]]) -> None:
    for stock_name in market.stocks:
        market.stocks[stock_name] = price_data[stock_name][day]

def run_simulation(price_data: Dict[str, List[float]], hyperparameters: Dict):
    # right now the hyperparameters are not used, but they can be integrated into the strategy. 
    market = Market()
    portfolio = Portfolio()
    context = Context()

    update_market(market, 0, price_data)
    for day in range(252):
        update_portfolio(market, portfolio, context, hyperparameters)
        if day < 251:
            update_market(market, day + 1, price_data)

    return portfolio.evaluate(market)

# Main function for hyperparameter tuning
if __name__ == "__main__":
    price_data = load_price_data("data.csv")

    best_value = 0
    best_params = None

    # Example hyperparameter tuning loop
    for param1 in range(1, 5):
        for param2 in range(1, 5):
            hyperparameters = {"param1": param1, "param2": param2}
            final_value = run_simulation(price_data, hyperparameters)
            print(f"Params: {hyperparameters}, Final Value: ${final_value:,.2f}")

            if final_value > best_value:
                best_value = final_value
                best_params = hyperparameters

    print(f"Best Parameters: {best_params}, Best Portfolio Value: ${best_value:,.2f}")