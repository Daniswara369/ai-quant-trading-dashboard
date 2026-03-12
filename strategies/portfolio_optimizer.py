"""
Portfolio optimization module — Mean-Variance optimization.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_portfolio_stats(returns_df: pd.DataFrame, weights: np.ndarray):
    """Calculate expected portfolio return and volatility."""
    mean_returns = returns_df.mean() * 252  # Annualize
    cov_matrix = returns_df.cov() * 252
    
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_volatility if port_volatility > 0 else 0
    
    return port_return, port_volatility, sharpe


def optimize_portfolio(returns_df: pd.DataFrame, num_portfolios: int = 5000) -> dict:
    """
    Monte Carlo portfolio optimization.
    
    Args:
        returns_df: DataFrame of daily returns for each asset.
        num_portfolios: Number of random portfolios to simulate.
    
    Returns:
        Dictionary with optimal weights and portfolio metrics.
    """
    n_assets = len(returns_df.columns)
    results = {
        "returns": [],
        "volatility": [],
        "sharpe": [],
        "weights": [],
    }
    
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = calculate_portfolio_stats(returns_df, weights)
        results["returns"].append(ret)
        results["volatility"].append(vol)
        results["sharpe"].append(sharpe)
        results["weights"].append(weights)
    
    # Find optimal portfolios
    max_sharpe_idx = np.argmax(results["sharpe"])
    min_vol_idx = np.argmin(results["volatility"])
    
    optimal = {
        "max_sharpe": {
            "weights": dict(zip(returns_df.columns, results["weights"][max_sharpe_idx])),
            "return": results["returns"][max_sharpe_idx],
            "volatility": results["volatility"][max_sharpe_idx],
            "sharpe": results["sharpe"][max_sharpe_idx],
        },
        "min_volatility": {
            "weights": dict(zip(returns_df.columns, results["weights"][min_vol_idx])),
            "return": results["returns"][min_vol_idx],
            "volatility": results["volatility"][min_vol_idx],
            "sharpe": results["sharpe"][min_vol_idx],
        },
        "efficient_frontier": {
            "returns": results["returns"],
            "volatility": results["volatility"],
            "sharpe": results["sharpe"],
        },
    }
    
    return optimal
