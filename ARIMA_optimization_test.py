# testing the best (p, d, q) parameters for ARIMA model
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from itertools import product

STOCK_NAMES = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]


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

def load_price_data(csv_file: str) -> dict:
    """
    Load price data from a CSV file.
    """
    price_data = {stock: [] for stock in STOCK_NAMES}
    df = pd.read_csv(csv_file)
    for stock in STOCK_NAMES:
        price_data[stock] = df[stock].tolist()
    return price_data

def test_arima_accuracy(price_history: dict, order: tuple[int, int, int]) -> dict:
    """
    Test the accuracy of ARIMA forecasts for each stock.
    """
    errors = {}
    for stock in STOCK_NAMES:
        prices = np.asarray(price_history[stock], dtype=float)
        if len(prices) < 20:  # Require at least 20 data points
            errors[stock] = None
            continue

        # Split data into training and testing sets
        train_size = int(len(prices) * 0.8)
        train, test = prices[:train_size], prices[train_size:]

        # Fit ARIMA model on training data
        try:
            model = ARIMA(train, order=order)
            fit = model.fit()

            # Forecast for the test set
            forecast = fit.forecast(steps=len(test))

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test, forecast))
            errors[stock] = rmse
        except Exception as e:
            errors[stock] = str(e)  # Record the error message if ARIMA fails

    return errors

if __name__ == "__main__":
    # Load price data
    price_data = load_price_data("data.csv")

    # Test ARIMA accuracy for different (p, d, q) values
    best_error = float('inf')
    best_pdq = None
    
    p_values = [1, 2, 3, 4, 5]
    d_values = [0, 1, 2]
    q_values = [1, 2, 3, 4, 5]

    pdq_combinations = list(product(p_values, d_values, q_values))

    
    for pdq in pdq_combinations:
        
        print(f"Testing ARIMA order: {pdq}")
        errors = test_arima_accuracy(price_data, pdq)
        for stock, error in errors.items():
            total_error = sum(e for e in errors.values() if isinstance(e, float))
            if total_error < best_error:
                best_error = total_error
                best_pdq = pdq
            if error is not None:
                pass 
                # print(f"  {stock}: RMSE = {error:.4f}")
            else:
                print(f"  {stock}: Not enough data or model failed.")
    print(f"Best ARIMA order: {best_pdq} with total RMSE: {best_error:.4f}")