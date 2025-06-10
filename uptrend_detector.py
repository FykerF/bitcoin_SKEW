import pandas as pd
import numpy as np

def detect_uptrend(price_series, method='combined', lookback=14, ma_short=7, ma_long=21, 
                   rsi_period=14, adx_period=14, min_trend_strength=25):
    """
    Detect uptrend from price series and return binary indicator.
    
    Parameters:
    -----------
    price_series : pd.Series
        Price series (close prices)
    method : str
        Detection method: 'ma', 'momentum', 'rsi_adx', 'combined'
    lookback : int
        Lookback period for momentum calculations
    ma_short : int
        Short moving average period
    ma_long : int  
        Long moving average period
    rsi_period : int
        RSI calculation period
    adx_period : int
        ADX calculation period
    min_trend_strength : float
        Minimum ADX value to consider trend valid
        
    Returns:
    --------
    pd.Series : Binary series (1=uptrend, 0=downtrend/sideways)
    """
    
    if method == 'ma':
        return _ma_uptrend(price_series, ma_short, ma_long)
    elif method == 'momentum':
        return _momentum_uptrend(price_series, lookback)
    elif method == 'rsi_adx':
        return _rsi_adx_uptrend(price_series, rsi_period, adx_period, min_trend_strength)
    elif method == 'combined':
        return _combined_uptrend(price_series, ma_short, ma_long, lookback, 
                               rsi_period, adx_period, min_trend_strength)
    else:
        raise ValueError("Method must be 'ma', 'momentum', 'rsi_adx', or 'combined'")

def _ma_uptrend(price_series, ma_short=7, ma_long=21):
    """Moving Average crossover method"""
    sma_short = price_series.rolling(ma_short).mean()
    sma_long = price_series.rolling(ma_long).mean()
    
    # Uptrend when short MA > long MA AND price > short MA
    uptrend = (sma_short > sma_long) & (price_series > sma_short)
    
    return uptrend.astype(int)

def _momentum_uptrend(price_series, lookback=14):
    """Momentum-based method"""
    # Price momentum
    price_momentum = (price_series / price_series.shift(lookback) - 1)
    
    # Rate of change
    roc = price_series.pct_change(lookback)
    
    # Recent performance vs longer term
    recent_performance = price_series.pct_change(lookback//2).rolling(3).mean()
    
    # Uptrend conditions
    uptrend = (
        (price_momentum > 0) &  # Positive momentum
        (roc > 0) &             # Positive rate of change
        (recent_performance > 0) # Recent strength
    )
    
    return uptrend.astype(int)

def _calculate_rsi(price_series, period=14):
    """Calculate RSI"""
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_adx(price_series, period=14):
    """Simplified ADX calculation using only close prices"""
    # Use price movements as proxy for true range
    price_change = price_series.diff().abs()
    
    # Smooth the movements
    smoothed_movement = price_change.rolling(period).mean()
    
    # Directional movement
    positive_dm = np.where(price_series.diff() > 0, price_series.diff(), 0)
    negative_dm = np.where(price_series.diff() < 0, -price_series.diff(), 0)
    
    # Smooth directional movements
    positive_dm_smooth = pd.Series(positive_dm, index=price_series.index).rolling(period).mean()
    negative_dm_smooth = pd.Series(negative_dm, index=price_series.index).rolling(period).mean()
    
    # Directional indicators
    di_positive = 100 * (positive_dm_smooth / smoothed_movement)
    di_negative = 100 * (negative_dm_smooth / smoothed_movement)
    
    # DX
    dx = 100 * abs(di_positive - di_negative) / (di_positive + di_negative)
    
    # ADX
    adx = dx.rolling(period).mean()
    
    return adx, di_positive, di_negative

def _rsi_adx_uptrend(price_series, rsi_period=14, adx_period=14, min_trend_strength=25):
    """RSI + ADX method"""
    rsi = _calculate_rsi(price_series, rsi_period)
    adx, di_pos, di_neg = _calculate_adx(price_series, adx_period)
    
    # Uptrend conditions:
    # 1. RSI > 50 (bullish momentum)
    # 2. ADX > min_trend_strength (strong trend)
    # 3. +DI > -DI (positive direction)
    uptrend = (
        (rsi > 50) &
        (adx > min_trend_strength) &
        (di_pos > di_neg)
    )
    
    return uptrend.astype(int)

def _combined_uptrend(price_series, ma_short=7, ma_long=21, lookback=14, 
                     rsi_period=14, adx_period=14, min_trend_strength=25):
    """Combined method using multiple indicators"""
    
    # Get individual signals
    ma_signal = _ma_uptrend(price_series, ma_short, ma_long)
    momentum_signal = _momentum_uptrend(price_series, lookback)
    rsi_adx_signal = _rsi_adx_uptrend(price_series, rsi_period, adx_period, min_trend_strength)
    
    # Voting system: need at least 2 out of 3 signals
    total_signals = ma_signal + momentum_signal + rsi_adx_signal
    uptrend = (total_signals >= 2).astype(int)
    
    return uptrend

def detect_uptrend_simple(price_series, ma_periods=[7, 21], momentum_days=10):
    """
    Ultra-simple version: Price above MAs + positive momentum
    
    Parameters:
    -----------
    price_series : pd.Series
        Price series
    ma_periods : list
        Moving average periods [short, long]
    momentum_days : int
        Days for momentum calculation
        
    Returns:
    --------
    pd.Series : Binary uptrend indicator
    """
    
    ma_short = price_series.rolling(ma_periods[0]).mean()
    ma_long = price_series.rolling(ma_periods[1]).mean()
    
    # Simple momentum
    momentum = price_series.pct_change(momentum_days)
    
    # Uptrend conditions:
    # 1. Price > short MA > long MA
    # 2. Positive momentum over momentum_days
    uptrend = (
        (price_series > ma_short) &
        (ma_short > ma_long) &
        (momentum > 0)
    )
    
    return uptrend.astype(int)

def detect_uptrend_strength(price_series, return_strength=False, **kwargs):
    """
    Enhanced version that can return trend strength instead of just binary
    
    Parameters:
    -----------
    price_series : pd.Series
        Price series
    return_strength : bool
        If True, return trend strength (0-1), if False return binary
    **kwargs : 
        Parameters passed to detect_uptrend()
        
    Returns:
    --------
    pd.Series : Binary (0/1) or strength (0.0-1.0) indicator
    """
    
    # Calculate multiple signals
    ma_signal = _ma_uptrend(price_series, kwargs.get('ma_short', 7), kwargs.get('ma_long', 21))
    momentum_signal = _momentum_uptrend(price_series, kwargs.get('lookback', 14))
    
    try:
        rsi_adx_signal = _rsi_adx_uptrend(
            price_series, 
            kwargs.get('rsi_period', 14),
            kwargs.get('adx_period', 14),
            kwargs.get('min_trend_strength', 25)
        )
    except:
        # Fallback if ADX calculation fails
        rsi = _calculate_rsi(price_series, kwargs.get('rsi_period', 14))
        rsi_adx_signal = (rsi > 50).astype(int)
    
    if return_strength:
        # Return average of all signals (0.0 to 1.0)
        strength = (ma_signal + momentum_signal + rsi_adx_signal) / 3.0
        return strength
    else:
        # Return binary (need majority vote)
        total_signals = ma_signal + momentum_signal + rsi_adx_signal
        return (total_signals >= 2).astype(int)

# Convenience functions for different use cases
def quick_uptrend(price_series):
    """Quick and simple uptrend detection"""
    return detect_uptrend_simple(price_series)

def robust_uptrend(price_series):
    """More robust uptrend detection with multiple confirmations"""
    return detect_uptrend(price_series, method='combined')

def momentum_uptrend(price_series, days=10):
    """Pure momentum-based uptrend"""
    return detect_uptrend(price_series, method='momentum', lookback=days)

# Example usage and testing
def test_uptrend_detector():
    """Test the uptrend detector with sample data"""
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Simulate trending price data
    trend = np.linspace(100, 150, 200)  # Upward trend
    noise = np.random.normal(0, 2, 200)  # Add noise
    prices = trend + noise
    
    # Add some volatility periods
    prices[50:70] *= 0.95  # Small dip
    prices[120:140] *= 1.1  # Rally
    
    price_series = pd.Series(prices, index=dates, name='price')
    
    # Test different methods
    methods = ['ma', 'momentum', 'rsi_adx', 'combined']
    results = {}
    
    print("Testing Uptrend Detection Methods")
    print("=" * 40)
    
    for method in methods:
        try:
            uptrend = detect_uptrend(price_series, method=method)
            uptrend_pct = uptrend.mean() * 100
            results[method] = uptrend
            print(f"{method:10s}: {uptrend_pct:5.1f}% uptrend periods")
        except Exception as e:
            print(f"{method:10s}: Error - {str(e)}")
            results[method] = pd.Series(0, index=price_series.index)
    
    # Test simple version
    simple_uptrend = detect_uptrend_simple(price_series)
    simple_pct = simple_uptrend.mean() * 100
    print(f"{'simple':10s}: {simple_pct:5.1f}% uptrend periods")
    results['simple'] = simple_uptrend
    
    # Test strength version
    strength = detect_uptrend_strength(price_series, return_strength=True)
    avg_strength = strength.mean()
    print(f"{'strength':10s}: {avg_strength:5.2f} average strength")
    
    return results, price_series

if __name__ == "__main__":
    # Run test
    results, sample_prices = test_uptrend_detector()
    
    print("\nExample usage:")
    print("uptrend = detect_uptrend(btc_prices, method='combined')")
    print("uptrend = quick_uptrend(btc_prices)  # Simple version")
    print("strength = detect_uptrend_strength(btc_prices, return_strength=True)")