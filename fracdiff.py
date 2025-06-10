import numpy as np
import pandas as pd
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_weights_ffd(d, thres=1e-5):
    """
    Compute weights for fractional differencing with threshold cutoff.
    
    Parameters:
    d : float
        Fractional differencing parameter (0 < d < 1)
    thres : float
        Threshold below which weights are considered too small
        
    Returns:
    numpy.array : Array of weights
    """
    w = [1.0]
    k = 1
    
    while True:
        # Compute next weight using recursive formula
        w_k = -w[-1] * (d - k + 1) / k
        
        # Stop if weight falls below threshold
        if abs(w_k) < thres:
            break
            
        w.append(w_k)
        k += 1
    
    return np.array(w[::-1])  # Reverse for correct order

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Compute fractionally differenced series using fixed-width window.
    
    Parameters:
    series : array-like
        Input time series
    d : float
        Fractional differencing parameter
    thres : float
        Threshold for weight cutoff
        
    Returns:
    pandas.Series : Fractionally differenced series
    """
    # Get weights
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    # Convert to pandas Series if not already
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Initialize output series
    df = {}
    
    # Apply fractional differencing with rolling window
    for name in series.index[width:]:
        # Get the window of values
        window_vals = series.loc[series.index <= name].iloc[-len(w):].values
        
        # Compute fractionally differenced value
        df[name] = np.dot(w, window_vals)
    
    return pd.Series(df)

def frac_diff_rolling_threshold(series, d_range=(0.0, 1.0), step=0.01, 
                               min_threshold=1e-5, max_threshold=1e-3,
                               target_stationarity=0.95):
    """
    Find optimal fractional differencing parameter and threshold using rolling window.
    
    Parameters:
    series : array-like
        Input time series
    d_range : tuple
        Range of d values to test
    step : float
        Step size for d parameter search
    min_threshold : float
        Minimum threshold to test
    max_threshold : float
        Maximum threshold to test
    target_stationarity : float
        Target p-value for stationarity test (ADF)
        
    Returns:
    dict : Results including optimal parameters and differenced series
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Convert to pandas Series
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    results = []
    d_values = np.arange(d_range[0] + step, d_range[1], step)
    thresholds = np.logspace(np.log10(min_threshold), np.log10(max_threshold), 10)
    
    best_result = None
    best_score = float('inf')
    
    for d in tqdm(d_values):
        for thres in thresholds:
            try:
                # Compute fractionally differenced series
                frac_diff_series = frac_diff_ffd(series, d, thres)
                
                # Skip if too few observations
                if len(frac_diff_series) < 50:
                    continue
                
                # Test for stationarity (ADF test)
                adf_stat, p_value, _, _, _, _ = adfuller(frac_diff_series.dropna())
                
                # Calculate correlation with original series (memory preservation)
                correlation = frac_diff_series.corr(series.loc[frac_diff_series.index])
                
                # Get weight count (window size)
                w = get_weights_ffd(d, thres)
                weight_count = len(w)
                
                # Score: balance stationarity, memory preservation, and efficiency
                # Lower score is better
                stationarity_score = max(0, p_value - target_stationarity)
                memory_loss = max(0, 0.5 - abs(correlation))  # Penalize too low correlation
                efficiency_penalty = weight_count / 100  # Prefer smaller windows
                
                score = stationarity_score + memory_loss + efficiency_penalty
                
                result = {
                    'd': d,
                    'threshold': thres,
                    'adf_pvalue': p_value,
                    'correlation': correlation,
                    'weight_count': weight_count,
                    'score': score,
                    'is_stationary': p_value < 0.05
                }
                
                results.append(result)
                
                # Update best result
                if score < best_score and p_value < 0.05:
                    best_score = score
                    best_result = result.copy()
                    best_result['series'] = frac_diff_series
                    
            except Exception as e:
                continue
    
    return best_result, results

def plot_frac_diff_analysis(original_series, frac_diff_series, d, threshold):
    """
    Plot analysis of fractional differencing results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original vs Fractionally Differenced
    axes[0, 0].plot(original_series.index, original_series.values, 
                    label='Original', alpha=0.7)
    axes[0, 0].plot(frac_diff_series.index, frac_diff_series.values, 
                    label=f'Frac Diff (d={d:.3f})', alpha=0.8)
    axes[0, 0].set_title('Original vs Fractionally Differenced Series')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Weights visualization
    w = get_weights_ffd(d, threshold)
    axes[0, 1].plot(range(len(w)), w, 'o-')
    axes[0, 1].set_title(f'Fractional Differencing Weights (threshold={threshold:.1e})')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ACF comparison
    from statsmodels.tsa.stattools import acf
    
    lags = min(40, len(original_series) // 4)
    acf_orig = acf(original_series.dropna(), nlags=lags)
    acf_diff = acf(frac_diff_series.dropna(), nlags=lags)
    
    axes[1, 0].plot(range(len(acf_orig)), acf_orig, 'o-', label='Original', alpha=0.7)
    axes[1, 0].plot(range(len(acf_diff)), acf_diff, 'o-', label='Frac Diff', alpha=0.7)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Autocorrelation Function Comparison')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[1, 1].hist(original_series.dropna(), bins=30, alpha=0.5, 
                    label='Original', density=True)
    axes[1, 1].hist(frac_diff_series.dropna(), bins=30, alpha=0.5, 
                    label='Frac Diff', density=True)
    axes[1, 1].set_title('Distribution Comparison')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample time series (random walk with trend)
    np.random.seed(42)
    n = 1000
    trend = np.linspace(0, 5, n)
    noise = np.random.normal(0, 0.5, n)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    
    # Create non-stationary series
    ts = trend + 0.3 * random_walk + noise
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    series = pd.Series(ts, index=dates)
    
    print("Finding optimal fractional differencing parameters...")
    best_result, all_results = frac_diff_rolling_threshold(series)
    
    if best_result:
        print(f"\nOptimal Parameters:")
        print(f"d = {best_result['d']:.4f}")
        print(f"Threshold = {best_result['threshold']:.2e}")
        print(f"ADF p-value = {best_result['adf_pvalue']:.4f}")
        print(f"Correlation with original = {best_result['correlation']:.4f}")
        print(f"Window size = {best_result['weight_count']} periods")
        
        # Plot analysis
        plot_frac_diff_analysis(series, best_result['series'], 
                               best_result['d'], best_result['threshold'])
        
        # Show weight decay
        weights = get_weights_ffd(best_result['d'], best_result['threshold'])
        print(f"\nWeight decay pattern (first 10 weights):")
        for i, w in enumerate(weights[:10]):
            print(f"w[{i}] = {w:.6f}")
            
    else:
        print("No suitable parameters found. Try adjusting the search range or thresholds.")
        
    # Show top 5 results
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values('score')
        print(f"\nTop 5 parameter combinations:")
        print(df_results[['d', 'threshold', 'adf_pvalue', 'correlation', 'weight_count']].head())