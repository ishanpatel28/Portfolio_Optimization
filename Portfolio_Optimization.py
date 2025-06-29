import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

def download_data(tickers, start_date, end_date):
    # Download stock data (auto_adjust=True by default)
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    # If columns are multi-index, select only 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    returns = data.pct_change().dropna()
    return returns

def optimize_portfolio(returns):
    print("Returns shape:", returns.shape)

    n = returns.shape[1] # Number of Assets
    mu = returns.mean().values # Expected returns (average daily returns)
    Sigma = returns.cov().values # Covariance matrix (risk between assets)

    w = cp.Variable(n) # Portfolio weights to be determined by optimizer
    gamma = cp.Parameter(nonneg=True) # Risk aversion parameter (non-negative)
    gamma.value = 1.0 # Set risk aversion level (adjust to be more/less risk averse)
    ret = mu @ w # Expected portfolio return (dot product of returns and weights)
    risk = cp.quad_form(w, Sigma) # Portfolio variance (quadratic form using covariance)
    # Define optimization problem: maximize return - risk penalty
    prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                  [cp.sum(w) == 1, # Weights must sum to 1 (full investment)
                   w >= 0])        # No short-selling (weights >= 0)
    prob.solve() # solve the optimization problem

    print("Weights shape:", w.value.shape)
    return w.value # Return the optimal weights as a numpy array

def plot_portfolio(weights, tickers):
    plt.figure(figsize=(8,6)) # Set figure size
    plt.bar(tickers, weights) # Create a bar chart of weights by ticker
    plt.title('Optimized Portfolio Weights') # Chart title
    plt.xlabel('Stocks') # X-axis label
    plt.ylabel('Weight') # Y-axis label
    plt.show() # Display the plot

if __name__ == "__main__":
    # List of stock stickers to include in portfolio
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    # Download stock return data for one year
    returns = download_data(tickers, '2022-01-01', '2023-01-01')
    # Optimize portfolio weights based on returns
    weights = optimize_portfolio(returns)

    print("Optimized Portfolio Weights:")
    # Print weights for each ticker with 4 decimal places
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight:.4f}")

        # Plot the portfolio allocation
        plot_portfolio(weights, tickers)
