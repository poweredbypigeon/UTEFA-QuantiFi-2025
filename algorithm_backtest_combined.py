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
from scipy.optimize import minimize

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.2




STOCK_NAMES = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]



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
        self.total_transaction_costs = 0.0  # Track total transaction costs
        self.transaction_occurred = False  # Flag to track transactions
    def evaluate(self, curMarket: Market) -> float:
        total_value = self.cash
        for stock_name, num_shares in self.shares.items():
            total_value += num_shares * curMarket.stocks[stock_name]
        return total_value

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]
        transaction_cost = Market.transaction_fee * shares_to_buy * curMarket.stocks[stock_name]
        if cost > self.cash:
            raise ValueError("Insufficient cash to buy shares")
        self.shares[stock_name] += shares_to_buy
        self.cash -= cost
        self.total_transaction_costs += transaction_cost  # Add transaction cost
        self.transaction_occurred = True  # Set flag when a transaction occurs

        # DEBUG ONLY
        return f"BUY: {shares_to_buy:.2f} shares of {stock_name} at ${curMarket.stocks[stock_name]:.2f}, Cost: ${cost:.2f}, Transaction Cost: ${transaction_cost:.2f}, Cash Remaining: ${self.cash:.2f}"

    def sell(self, stock_name: str, shares_to_sell: float, curMarket: Market) -> None:
        if shares_to_sell > self.shares[stock_name]:
            raise ValueError("Insufficient shares to sell")
        proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        transaction_cost = Market.transaction_fee * shares_to_sell * curMarket.stocks[stock_name]
        self.shares[stock_name] -= shares_to_sell
        self.cash += proceeds
        self.total_transaction_costs += transaction_cost  # Add transaction cost
        self.transaction_occurred = True  # Set flag when a transaction occurs

        # DEBUG ONLY
        return f"SELL: {shares_to_sell:.2f} shares of {stock_name} at ${curMarket.stocks[stock_name]:.2f}, Proceeds: ${proceeds:.2f}, Transaction Cost: ${transaction_cost:.2f}, Cash After Sale: ${self.cash:.2f}"

class Context:
    def __init__(self) -> None:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.pct_change = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day = 0
        self.last_rebalance_day = -1000
        # Current target weights (start all in cash)
        self.target_weights = {stock: 0.0 for stock in STOCK_NAMES}

def _compute_returns_matrix(price_history: dict, window: int) -> np.ndarray | None:
    """
    Build a T x N matrix of returns for the last window days.
    Rows = time 
    Columns = stocks
    """
    all_returns = []
    for stock in STOCK_NAMES:
        prices = np.asarray(price_history[stock], dtype=float)
        if prices.size < 2:
            return None
        # keep only the most recent window+1 prices
        if prices.size > window + 1:
            prices = prices[-(window + 1):]
        rets = np.diff(prices) / prices[:-1]
        all_returns.append(rets)

    min_len = min(len(r) for r in all_returns)
    if min_len < 2:
        return None

    trimmed = np.vstack([r[-min_len:] for r in all_returns]).T  # shape (T, N)
    return trimmed


def _forecast_arima_expected_returns(price_history: dict,
                                     order: tuple[int, int, int] = (1, 1, 1)
                                     ) -> np.ndarray:
    """
    Fit a simple ARIMA model per stock on prices and forecast 1-step-ahead price.
    Convert forecast to expected return.
    """
    exp_rets = []
    for stock in STOCK_NAMES:
        prices = np.asarray(price_history[stock], dtype=float)

        # Require some history for a stable fit
        if prices.size < (order[0] + order[1] + 10):
            exp_rets.append(0.0)
            continue

        try:
            model = ARIMA(prices, order=order)
            fit = model.fit()
            forecast_price = float(fit.forecast(steps=1)[0])
            last_price = float(prices[-1])
            if last_price > 0:
                exp_rets.append(forecast_price / last_price - 1.0)
            else:
                exp_rets.append(0.0)
        except Exception:
            # If ARIMA fails, fall back to 0 expected return
            exp_rets.append(0.0)

    return np.asarray(exp_rets, dtype=float)



def _markowitz_max_sharpe(mu: np.ndarray,
                          Sigma: np.ndarray,
                          max_weight: float = 1) -> np.ndarray:
    """
    Maximize Sharpe ratio: (w^T mu) / sqrt(w^T Σ w)
    Constraints:
      - sum(w) = 1
      - 0 <= w_i <= max_weight
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]

    def objective(w):
        ret = float(w.dot(mu))
        vol = float(np.sqrt(w.T.dot(Sigma).dot(w)))
        if vol == 0.0:
            return 1e6
        return -ret / vol  # minimize negative Sharpe

    # Constraint: weights sum to 1
    constraints = (
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
    )

    # Bounds: long-only, capped per asset
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, bounds=bounds, constraints=constraints)

    if not result.success:
        # If optimization fails, fall back to equal weight
        return np.ones(n) / n

    return result.x
 

def _rebalance_to_target_weights(curMarket, curPortfolio, target_weights: dict):
    """
    Trade the portfolio to move towards `target_weights` given current prices.
    """
    # Reset the transaction flag at the start of rebalancing
    changed = False
    logs = [] 

    # Compute total portfolio value
    total_value = curPortfolio.evaluate(curMarket)

    # Target values and shares
    target_values = {}
    target_shares = {}
    for stock in STOCK_NAMES:
        price = float(curMarket.stocks[stock])
        w = float(target_weights.get(stock, 0.0))
        target_values[stock] = total_value * w
        target_shares[stock] = target_values[stock] / price if price > 0 else 0.0

    # 1) Sell down positions that are above target (raise cash first)
    for stock in STOCK_NAMES:
        current_shares = float(curPortfolio.shares[stock])
        desired_shares = float(target_shares[stock])
        diff = desired_shares - current_shares
        if diff < -1e-6:
            shares_to_sell = -diff
            try:
                log = curPortfolio.sell(stock, shares_to_sell, curMarket)
                logs.append(log)
                changed = True
            except ValueError:
                # In case of rounding issues, sell what we can
                available = max(0.0, curPortfolio.shares[stock])
                if available > 0:
                    try:
                        log = curPortfolio.sell(stock, available, curMarket)
                        logs.append(log)
                        changed = True
                    except ValueError:
                        pass
    
    

    # 2) Buy up positions that are below target
    for stock in STOCK_NAMES:
        current_shares = float(curPortfolio.shares[stock])
        desired_shares = float(target_shares[stock])
        diff = desired_shares - current_shares
        if diff > 1e-6:
            price = float(curMarket.stocks[stock])
            # Max shares we can afford for this stock right now
            max_affordable = curPortfolio.cash / ((1.0 + Market.transaction_fee) * price)
            shares_to_buy = min(diff, max_affordable)
            if shares_to_buy > 0:
                try:
                    log = curPortfolio.buy(stock, shares_to_buy, curMarket)
                    logs.append(log)
                except ValueError:
                    # If we can't buy (tiny cash etc.), just skip
                    pass

    return (changed, logs)
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

def generate_file_name(hyperparameters: Dict) -> str:
    return (f"backtest_LB{hyperparameters['LOOKBACK_DAYS']}_"
            f"RF{hyperparameters['REBALANCE_FREQUENCY']}_"
            f"MW{hyperparameters['MAX_WEIGHT_PER_STOCK']}_"
            f"MH{hyperparameters['MIN_HISTORY_FOR_MODEL']}.txt")

def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context, hyperparameters: Dict):
    log_lines = []  # Store log lines for writing to a file
    LOOKBACK_DAYS = hyperparameters.get("LOOKBACK_DAYS", 60)           # days of returns for covariance
    REBALANCE_FREQUENCY = hyperparameters.get("REBALANCE_FREQUENCY", 10)      # rebalance every 5 days
    MAX_WEIGHT_PER_STOCK = hyperparameters.get("MAX_WEIGHT_PER_STOCK", 0.4)   # cap per stock (40%)
    MIN_HISTORY_FOR_MODEL = hyperparameters.get("MIN_HISTORY_FOR_MODEL", 20)   # minimum days of price history before using ARIMA/Markowitz
    changed = False 

    # 1) Update price history with current day's prices
    for stock in STOCK_NAMES:
        price = float(curMarket.stocks[stock])
        context.price_history[stock].append(price)

    # 2) If not enough history yet, do nothing (stay in cash / whatever we currently hold)
    history_len = len(context.price_history[STOCK_NAMES[0]])
    if history_len < MIN_HISTORY_FOR_MODEL:
        context.day += 1
        return (False, "")

    # 3) Only rebalance every REBALANCE_FREQUENCY days
    if (context.day - context.last_rebalance_day) < REBALANCE_FREQUENCY:
        context.day += 1
        return (False, "")

    # 4) Build returns matrix and covariance
    returns_matrix = _compute_returns_matrix(context.price_history, LOOKBACK_DAYS)
    if returns_matrix is None:
        context.day += 1
        return (False, "")

    Sigma = np.cov(returns_matrix, rowvar=False)
    Sigma = Sigma + 1e-6 * np.eye(Sigma.shape[0])

    # 5) ARIMA expected returns (directly used as mu)
    mu_arima = _forecast_arima_expected_returns(context.price_history)

    # 6) Markowitz optimization to get target weights
    weights = _markowitz_max_sharpe(mu_arima, Sigma, max_weight=MAX_WEIGHT_PER_STOCK)
    new_target_weights = {stock: float(w) for stock, w in zip(STOCK_NAMES, weights)}

    changed, logs = _rebalance_to_target_weights(curMarket, curPortfolio, new_target_weights)

    # 7) Update bookkeeping
    context.last_rebalance_day = context.day
    context.day += 1

    return (changed, logs)


def run_simulation(price_data: Dict[str, List[float]], hyperparameters: Dict):
    market = Market()
    portfolio = Portfolio()
    context = Context()
    log_lines = []  # Store log lines for writing to a file
    update_market(market, 0, price_data)
    for day in range(252):
        changed, logs = update_portfolio(market, portfolio, context, hyperparameters)
        if day < 251:
            update_market(market, day + 1, price_data)
        # Check if the portfolio weights have changed significantly
        if changed:
            log_lines.extend(logs)
            # Log portfolio changes only when rebalanced
            log_lines.append(f"Day {context.day + 1}: Portfolio Value: ${portfolio.evaluate(market):,.2f}")
            log_lines.append(f"  Cash: ${portfolio.cash:,.2f}")
            for stock, shares in portfolio.shares.items():
                stock_value = shares * market.stocks[stock]
                log_lines.append(f"  {stock}: {shares:.2f} shares, Value: ${stock_value:,.2f}")
            log_lines.append("")  # Blank line for readability

    with open("logs/" + generate_file_name(hyperparameters), "w") as log_file:
        log_file.write("\n\nPORTFOLIO LOG:\n")
        log_file.write("\n".join(log_lines))

        log_file.write(f"\n Total Transaction Costs: ${portfolio.total_transaction_costs:,.2f}\n")
    print(f"Total Transaction Costs: ${portfolio.total_transaction_costs:,.2f}")  # Output total transaction costs
    return portfolio.evaluate(market)

# Main function for hyperparameter tuning
if __name__ == "__main__":
    price_data = load_price_data("data.csv")

    best_value = 0
    best_params = None

    # Example hyperparameter tuning loop

    LOOKBACK_DAYS_OPTIONS = [30]
    REBALANCE_FREQUENCY_OPTIONS = [20]
    MAX_WEIGHT_PER_STOCK_OPTIONS = [0.4]
    MIN_HISTORY_FOR_MODEL_OPTIONS = [10] # good enough!

    for lookback in LOOKBACK_DAYS_OPTIONS:
        for rebalance_freq in REBALANCE_FREQUENCY_OPTIONS:
            for max_weight in MAX_WEIGHT_PER_STOCK_OPTIONS:
                for min_history in MIN_HISTORY_FOR_MODEL_OPTIONS:
                    hyperparameters = {
                        "LOOKBACK_DAYS": lookback,
                        "REBALANCE_FREQUENCY": rebalance_freq,
                        "MAX_WEIGHT_PER_STOCK": max_weight,
                        "MIN_HISTORY_FOR_MODEL": min_history
                    }

                    final_value = run_simulation(price_data, hyperparameters)
                    print(f"Params: {hyperparameters}, Final Value: ${final_value:,.2f}")

                    if final_value > best_value:
                        best_value = final_value
                        best_params = hyperparameters

    print(f"Best Parameters: {best_params}, Best Portfolio Value: ${best_value:,.2f}")

'''
REBALANCE_FREQUENCY: ???
MAX_WEIGHT_PER_STOCK: 0.4. 
MIN_HISTORY_FOR_MODEL: 10 > 20 > 30. 
'''