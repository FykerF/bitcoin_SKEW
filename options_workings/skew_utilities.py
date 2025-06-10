#!/usr/bin/env python3
"""
SKEW Utilities
==============
Additional utility functions for SKEW analysis including
probability calculations and trading signal generation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Optional


def skew_to_probabilities(skew_series: Union[pd.Series, np.ndarray, List[float]], 
                         volatility_series: Union[pd.Series, np.ndarray, List[float], float] = 0.20,
                         standard_deviations: List[float] = [1.5, 2.0, 2.5, 3.0],
                         dates: Optional[Union[pd.DatetimeIndex, List]] = None) -> pd.DataFrame:
    """
    Convert SKEW index values to tail probabilities.
    
    Parameters:
    -----------
    skew_series : pd.Series, np.array, or list
        Series of SKEW index values (typically 100-150)
    volatility_series : pd.Series, np.array, list, or float, default=0.20
        Series of return volatility values (annualized) or single value
    standard_deviations : list, default=[1.5, 2.0, 2.5, 3.0]
        Standard deviation levels for probability calculation
    dates : DatetimeIndex or list, optional
        Dates corresponding to the observations
    
    Returns:
    --------
    pd.DataFrame with SKEW values and corresponding tail probabilities
    """
    
    # Convert inputs to numpy arrays
    skew_array = np.asarray(skew_series)
    
    # Handle volatility input
    if np.isscalar(volatility_series):
        vol_array = np.full_like(skew_array, volatility_series)
    else:
        vol_array = np.asarray(volatility_series)
    
    # Validate input lengths
    if len(skew_array) != len(vol_array):
        raise ValueError("SKEW and volatility series must have same length")
    
    # Convert SKEW to skewness parameter S
    S_array = (100 - skew_array) / 10
    
    # Create base DataFrame
    data = {
        'skew': skew_array,
        'volatility': vol_array,
        'skewness_S': S_array
    }
    
    # Calculate tail probabilities for each standard deviation level
    for std_dev in standard_deviations:
        # Left tail probabilities (downside risk)
        left_probs = calculate_tail_probabilities_vectorized(S_array, std_dev)
        data[f'prob_{std_dev}std_below'] = left_probs
        
        # Tail risk ratios vs normal distribution
        normal_prob = stats.norm.cdf(-std_dev)
        data[f'tail_ratio_{std_dev}std'] = left_probs / normal_prob
    
    # Create DataFrame
    if dates is not None:
        df = pd.DataFrame(data, index=dates)
    else:
        df = pd.DataFrame(data)
    
    return df


def calculate_tail_probabilities_vectorized(skewness_array: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Vectorized calculation of left tail probabilities using Gram-Charlier expansion.
    
    Formula: P(X ≤ -σ) = Φ(-σ) - (S/6) × (σ² - 1) × φ(-σ)
    """
    
    x = std_dev
    
    # Normal distribution components
    normal_cdf = stats.norm.cdf(-x)
    normal_pdf = stats.norm.pdf(-x)
    
    # Gram-Charlier adjustment: -(S/6) × (x² - 1) × φ(-x)
    gc_adjustment = -(skewness_array / 6) * (x**2 - 1) * normal_pdf
    
    # Final adjusted probability
    adjusted_prob = normal_cdf + gc_adjustment
    
    # Ensure probabilities are between 0 and 1
    return np.clip(adjusted_prob, 0, 1)


def generate_exit_signals(probability_series: pd.Series, 
                         threshold: float = 0.5,
                         cooldown_periods: int = 15) -> pd.Series:
    """
    Generate trading exit signals based on tail probabilities.
    
    Parameters:
    -----------
    probability_series : pd.Series
        Series of tail probabilities
    threshold : float
        Probability threshold for signal generation (default 0.5)
    cooldown_periods : int
        Number of periods to wait after a signal (default 15)
    
    Returns:
    --------
    pd.Series: Binary series where 1 = stay in market, 0 = exit market
    """
    
    # Generate raw signals (1 when probability exceeds threshold)
    raw_signals = (probability_series > threshold).astype(int)
    
    # Apply cooldown period
    result = []
    counter = 0
    
    for i in range(len(raw_signals)):
        if raw_signals.iloc[i] == 1 and counter == 0:
            # New signal - start cooldown
            counter = cooldown_periods
            result.append(0)  # Exit signal
        elif counter > 0:
            # In cooldown period
            counter -= 1
            result.append(0)  # Stay out
        else:
            # No signal, no cooldown
            result.append(1)  # Stay in
    
    return pd.Series(result, index=probability_series.index)


def calculate_signal_performance(returns: pd.Series, signals: pd.Series) -> dict:
    """
    Calculate performance metrics for a trading signal.
    
    Parameters:
    -----------
    returns : pd.Series
        Asset returns
    signals : pd.Series
        Binary trading signals (1 = in market, 0 = out)
    
    Returns:
    --------
    dict: Performance metrics including returns, Sharpe ratio, etc.
    """
    
    # Align series
    aligned = pd.DataFrame({'returns': returns, 'signals': signals}).dropna()
    
    # Calculate signal returns
    signal_returns = aligned['returns'] * aligned['signals']
    
    # Calculate metrics
    metrics = {
        'total_return': (1 + signal_returns).prod() - 1,
        'buy_hold_return': (1 + aligned['returns']).prod() - 1,
        'signal_sharpe': signal_returns.mean() / signal_returns.std() * np.sqrt(252),
        'buy_hold_sharpe': aligned['returns'].mean() / aligned['returns'].std() * np.sqrt(252),
        'days_in_market': aligned['signals'].sum(),
        'days_out_of_market': len(aligned) - aligned['signals'].sum(),
        'pct_time_in_market': aligned['signals'].mean() * 100
    }
    
    return metrics


def check_future_falls(returns: pd.Series, n: int, d: int, w: int, cumulative: bool = True) -> pd.Series:
    """
    Check if a fall of n*std occurs within the next d days after each day t.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns (daily returns)
    n : int
        Number of standard deviations for the fall threshold
    d : int
        Number of days to look forward
    w : int
        Window size for calculating rolling standard deviation
    cumulative : bool
        If True, check cumulative returns over any period up to d days.
        If False, check only single-day returns.
    
    Returns:
    --------
    pd.Series: Binary series where 1 indicates a fall of n*std occurred within next d days, 0 otherwise
    """
    
    # Calculate rolling standard deviation with window w
    rolling_std = returns.rolling(window=w, min_periods=1).std()
    
    # Initialize result series with zeros
    result = pd.Series(0, index=returns.index)
    
    # For each day, check if there's a fall of n*std in the next d days
    for i in range(len(returns) - d):
        # Current day's standard deviation
        current_std = rolling_std.iloc[i]
        
        # Skip if std is NaN or zero
        if pd.isna(current_std) or current_std == 0:
            continue
        
        # Calculate the threshold for a fall (negative return of n*std)
        threshold = -n * current_std
        
        # Check returns in the next d days
        future_returns = returns.iloc[i+1:i+d+1]
        
        if cumulative:
            # Check cumulative returns for all possible periods
            for j in range(1, len(future_returns) + 1):
                # Calculate cumulative return from day i to day i+j
                cum_return = (1 + future_returns.iloc[:j]).prod() - 1
                
                # Check if cumulative return is below threshold
                if cum_return <= threshold:
                    result.iloc[i] = 1
                    break
        else:
            # Check if any single future return is below the threshold
            if (future_returns <= threshold).any():
                result.iloc[i] = 1
    
    return result


def plot_skew_analysis(master_df: pd.DataFrame, figsize=(12, 10)):
    """
    Create comprehensive visualization of SKEW analysis results.
    
    Parameters:
    -----------
    master_df : pd.DataFrame
        Master DataFrame with all metrics
    figsize : tuple
        Figure size (default (12, 10))
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Bitcoin Price
    ax1 = axes[0]
    ax1.plot(master_df.index, master_df['btc_price'], 'b-', linewidth=1.5)
    ax1.set_ylabel('BTC Price ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Bitcoin SKEW Analysis Dashboard', fontsize=14, fontweight='bold')
    
    # Plot 2: SKEW Index
    ax2 = axes[1]
    ax2.plot(master_df.index, master_df['skew_index'], 'g-', linewidth=1.5)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Neutral (100)')
    ax2.set_ylabel('SKEW Index', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Plot 3: Rolling Volatility
    ax3 = axes[2]
    ax3.plot(master_df.index, master_df['rolling_vol'] * 100, 'orange', linewidth=1.5)
    ax3.set_ylabel('Volatility (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling Skewness
    ax4 = axes[3]
    ax4.plot(master_df.index, master_df['rolling_skew'], 'purple', linewidth=1.5)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero skew')
    ax4.set_ylabel('Realized Skewness', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Example: Convert SKEW to probabilities
    skew_values = [110, 115, 120, 125, 130, 135, 140]
    volatility = 0.25  # 25% annualized volatility
    
    prob_df = skew_to_probabilities(
        skew_series=skew_values,
        volatility_series=volatility,
        standard_deviations=[1.0, 1.5, 2.0, 3.0]
    )
    
    print("SKEW to Probability Conversion:")
    print(prob_df)
    
    # Example: Generate exit signals
    exit_signals = generate_exit_signals(
        prob_df['prob_2.0std_below'],
        threshold=0.3,
        cooldown_periods=10
    )
    
    print("\nExit Signals:")
    print(exit_signals)