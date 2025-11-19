"""
UTEFA QuantiFi - Contestant Template

This template provides the structure for implementing your trading strategy.
Your goal is to maximize portfolio value over 252 (range 0 to 251) trading days.

IMPORTANT:
- Implement your strategy in the update_portfolio() function
- You can store any data you need in the Context class
- Transaction fees apply to both buying and selling (0.5%)
- Do not modify the Market or Portfolio class structures

"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize




class Market:
    """
    Represents the stock market with current prices for all available stocks.
    
    Attributes:
        transaction_fee: Float representing the transaction fee (0.5% = 0.005)
        stocks: Dictionary mapping stock names to their current prices
    """
    transaction_fee = 0.005
    
    def __init__(self) -> None:
        # Initialize with 5 stocks
        # Prices will be set by the backtesting script from the CSV data
        self.stocks = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }

    def updateMarket(self):
        """
        Updates stock prices to reflect market changes.
        This function will be implemented during grading.
        DO NOT MODIFY THIS METHOD.
        """
        pass


class Portfolio:
    """
    Represents your investment portfolio containing shares and cash.
    
    Attributes:
        shares: Dictionary mapping stock names to number of shares owned
        cash: Float representing available cash balance
    """
    
    def __init__(self) -> None:
        # Start with no shares and $100,000 cash
        self.shares = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }
        self.cash = 100000.0

    def evaluate(self, curMarket: Market) -> float:
        """
        Calculate the total value of the portfolio (shares + cash).
        
        Args:
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing total portfolio value
        """
        total_value = self.cash
        
        for stock_name, num_shares in self.shares.items():
            total_value += num_shares * curMarket.stocks[stock_name]
        
        return total_value

    def sell(self, stock_name: str, shares_to_sell: float, curMarket: Market) -> None:
        """
        Sell shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to sell (must match keys in self.shares)
            shares_to_sell: Number of shares to sell (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_sell is invalid or exceeds owned shares
        """
        if shares_to_sell <= 0:
            raise ValueError("Number of shares must be positive")

        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")

        if shares_to_sell > self.shares[stock_name]:
            raise ValueError(f"Attempted to sell {shares_to_sell} shares of {stock_name}, but only {self.shares[stock_name]} available")

        # Update portfolio
        self.shares[stock_name] -= shares_to_sell
        sale_proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        self.cash += sale_proceeds

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        """
        Buy shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to buy (must match keys in self.shares)
            shares_to_buy: Number of shares to buy (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_buy is invalid or exceeds available cash
        """
        if shares_to_buy <= 0:
            raise ValueError("Number of shares must be positive")
        
        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]
        
        if cost > self.cash + 0.01:
            raise ValueError(f"Attempted to spend ${cost:.2f}, but only ${self.cash:.2f} available")

        # Update portfolio
        self.shares[stock_name] += shares_to_buy
        self.cash -= cost

    def get_position_value(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to get the current value of a specific position.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing the total value of owned shares for this stock
        """
        return self.shares[stock_name] * curMarket.stocks[stock_name]

    def get_max_buyable_shares(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to calculate the maximum number of shares that can be bought.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing maximum shares that can be purchased with available cash
        """
        price_per_share = curMarket.stocks[stock_name] * (1 + Market.transaction_fee)
        return self.cash / price_per_share if price_per_share > 0 else 0

STOCK_NAMES = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

#PARAMETERS
LOOKBACK_DAYS = 60           # days of returns for covariance
REBALANCE_FREQUENCY = 10      # rebalance every 5 days
MAX_WEIGHT_PER_STOCK = 0.4   # cap per stock (40%)
MIN_HISTORY_FOR_MODEL = 20   # minimum days of price history before using ARIMA/Markowitz

class Context:
    """
    Store any data you need for your trading strategy.
    
    This class is completely customizable. Use it to track:
    - Historical prices
    - Calculated indicators (moving averages, momentum, etc.)
    - Trading signals
    - Strategy state
    
    Example usage:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day_counter = 0
    """
    
    def __init__(self) -> None:
        # PUT WHATEVER YOU WANT HERE
        # Example: Track price history for technical analysis

       # Track price history for each stock
        self.price_history = {stock: [] for stock in STOCK_NAMES}
        # Day counter
        self.day = 0
        # Last rebalance day
        self.last_rebalance_day = -REBALANCE_FREQUENCY
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
                                     order: tuple[int, int, int] = (1, 1, 0)
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
                          max_weight: float = MAX_WEIGHT_PER_STOCK) -> np.ndarray:
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
 

def _rebalance_to_target_weights(curMarket, curPortfolio, target_weights: dict) -> None:
    """
    Trade the portfolio to move towards `target_weights` given current prices.
    """
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
                curPortfolio.sell(stock, shares_to_sell, curMarket)
            except ValueError:
                # In case of rounding issues, sell what we can
                available = max(0.0, curPortfolio.shares[stock])
                if available > 0:
                    try:
                        curPortfolio.sell(stock, available, curMarket)
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
                    curPortfolio.buy(stock, shares_to_buy, curMarket)
                except ValueError:
                    # If we can't buy (tiny cash etc.), just skip
                    pass


def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    Implement your trading strategy here.
    
    This function is called once per trading day, before the market updates.
    
    Args:
        curMarket: Current Market object with stock prices
        curPortfolio: Current Portfolio object with your holdings
        context: Context object for storing strategy data
    
    Example strategy (DO NOT USE THIS - IT'S JUST A PLACEHOLDER):
        # Track prices
        for stock in curMarket.stocks:
            context.price_history[stock].append(curMarket.stocks[stock])
        
        
        # Simple buy-and-hold: invest all cash on day 0
        if context.day == 0:
            for stock in curMarket.stocks:
                max_shares = curPortfolio.get_max_buyable_shares(stock, curMarket)
                if max_shares > 0:
                    curPortfolio.buy(stock, max_shares / 5, curMarket)  # Split equally
        
        context.day += 1
    """
    # YOUR TRADING STRATEGY GOES HERE
    # 1) Update price history with current day's prices
    # 1) Update price history with current day's prices
    for stock in STOCK_NAMES:
        price = float(curMarket.stocks[stock])
        context.price_history[stock].append(price)

    # 2) If not enough history yet, do nothing (stay in cash / whatever we currently hold)
    history_len = len(context.price_history[STOCK_NAMES[0]])
    if history_len < MIN_HISTORY_FOR_MODEL:
        context.day += 1
        return

    # 3) Only rebalance every REBALANCE_FREQUENCY days
    if (context.day - context.last_rebalance_day) < REBALANCE_FREQUENCY:
        context.day += 1
        return

    # 4) Build returns matrix and covariance
    returns_matrix = _compute_returns_matrix(context.price_history, LOOKBACK_DAYS)
    if returns_matrix is None:
        # Safety: in case returns matrix fails for some reason, skip rebalancing
        context.day += 1
        return

    Sigma = np.cov(returns_matrix, rowvar=False)
    # Add a small ridge term for numerical stability
    Sigma = Sigma + 1e-6 * np.eye(Sigma.shape[0])

    # 5) ARIMA expected returns (directly used as mu)
    mu_arima = _forecast_arima_expected_returns(context.price_history)

    # 6) Markowitz optimization to get target weights
    weights = _markowitz_max_sharpe(mu_arima, Sigma, max_weight=MAX_WEIGHT_PER_STOCK)
    context.target_weights = {stock: float(w) for stock, w in zip(STOCK_NAMES, weights)}

    # 7) Rebalance portfolio towards target weights
    _rebalance_to_target_weights(curMarket, curPortfolio, context.target_weights)

    # 8) Update bookkeeping
    context.last_rebalance_day = context.day
    context.day += 1


###SIMULATION###
if __name__ == "__main__":
    market = Market()
    portfolio = Portfolio()
    context = Context()

    # Simulate 252 trading days (one trading year)
    for day in range(252):
        update_portfolio(market, portfolio, context)
        market.updateMarket()

    # Print final portfolio value
    final_value = portfolio.evaluate(market)
    print(f"Final Portfolio Value: ${final_value:,.2f}")
