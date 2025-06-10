#!/usr/bin/env python3
"""
Bitcoin SKEW Index Analysis Pipeline
=====================================
This script loads Bitcoin options data, calculates SKEW index values,
downloads Bitcoin price data, and creates a comprehensive dataset with
price, returns, volatility, realized skewness, and SKEW index values.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import SKEW calculation functions
from skew_calculations import calculate_weighted_skew


def load_bitcoin_options_data():
    """
    Load all Bitcoin options data from text files.
    
    Returns:
    --------
    pd.DataFrame: Combined options data from all available files
    """
    print("Loading Bitcoin options data...")
    
    # Initialize with first file
    data = pd.read_csv('btc_eod_202106.txt', sep=", ", engine='python')
    already_read = ['btc_eod_202106.txt']
    
    # Read all available files
    for year in [2021, 2022]:
        for month in range(1, 13):
            try:
                if month < 10:
                    path = f"btc_eod_{year}0{month}.txt"
                else:
                    path = f"btc_eod_{year}{month}.txt"
                
                if path not in already_read:
                    temp_data = pd.read_csv(path, sep=", ", engine='python')
                    data = pd.concat([data, temp_data], axis=0)
                    already_read.append(path)
                    print(f"  ✓ Loaded {path}")
            except:
                pass
    
    # Clean column names (remove brackets)
    data.columns = [col.strip('[]') for col in data.columns]
    
    print(f"Total records loaded: {len(data):,}")
    return data


def filter_options_for_skew(data):
    """
    Filter options data for SKEW calculation.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw options data
        
    Returns:
    --------
    pd.DataFrame: Filtered data suitable for SKEW calculation
    """
    # Filter for options with volume > 0 and DTE between 25-60 days
    filtered = data[(data["VOLUME"] > 0) & (data["DTE"] > 25) & (data["DTE"] < 60)]
    print(f"Filtered to {len(filtered):,} options with volume > 0 and 25 < DTE < 60")
    return filtered


def calculate_daily_skew(filtered_data, yields_data):
    """
    Calculate SKEW index for each trading day.
    
    Parameters:
    -----------
    filtered_data : pd.DataFrame
        Filtered options data
    yields_data : pd.DataFrame
        Risk-free rate data
        
    Returns:
    --------
    pd.Series: SKEW values indexed by date
    """
    print("\nCalculating SKEW index values...")
    
    # Get unique dates
    dates = filtered_data['QUOTE_DATE'].drop_duplicates()
    
    # Reindex yields to match option dates and forward-fill
    yields_aligned = yields_data.reindex(dates).ffill()
    
    # Calculate SKEW for each date
    skews = {}
    for date in tqdm(dates, desc="Processing dates"):
        day_data = filtered_data[filtered_data["QUOTE_DATE"] == date]
        try:
            # Use 7-day target DTE for consistency
            skew = calculate_weighted_skew(day_data, yields_aligned.loc[date].values[0], target_dte=7)
            skews[date] = skew
        except Exception as e:
            # Skip dates where calculation fails
            pass
    
    return pd.Series(skews)


def clean_skew_outliers(skew_series, tail_fraction=0.07):
    """
    Clean extreme outliers from SKEW values using tail smoothing.
    
    Parameters:
    -----------
    skew_series : pd.Series
        Raw SKEW values
    tail_fraction : float
        Fraction of data in each tail to smooth (default 0.07)
        
    Returns:
    --------
    pd.Series: Cleaned SKEW values
    """
    def clean_tails(data, tail_frac=0.05):
        """Clean extreme values from data tails while preserving distribution shape."""
        if isinstance(data, pd.Series):
            raw = data.values
        else:
            raw = data.copy()
        
        n = len(raw)
        work = np.sort(raw)
        cover = 1.0 - 2.0 * tail_frac
        
        # Find interval with desired coverage having minimum data span
        istart = 0
        istop = int(cover * (n + 1)) - 1
        if istop >= n:
            istop = n - 1
        
        best = float('inf')
        best_start = best_stop = 0
        
        while istop < n:
            range_val = work[istop] - work[istart]
            if range_val < best:
                best = range_val
                best_start = istart
                best_stop = istop
            istart += 1
            istop += 1
        
        minval = work[best_start]
        maxval = work[best_stop]
        
        # Handle pathological situation
        if maxval <= minval:
            maxval *= 1.0 + 1e-10
            minval *= 1.0 - 1e-10
        
        # Clean the tails
        limit = (maxval - minval) * (1.0 - cover)
        scale = -1.0 / (maxval - minval)
        
        cleaned = raw.copy()
        for i in range(n):
            if cleaned[i] < minval:  # Left tail
                cleaned[i] = minval - limit * (1.0 - np.exp(scale * (minval - cleaned[i])))
            elif cleaned[i] > maxval:  # Right tail
                cleaned[i] = maxval + limit * (1.0 - np.exp(scale * (cleaned[i] - maxval)))
        
        return cleaned
    
    print(f"\nCleaning SKEW outliers (tail fraction: {tail_fraction})...")
    cleaned = pd.Series(clean_tails(skew_series, tail_fraction), index=skew_series.index)
    
    # Forward fill any remaining NaN values
    cleaned = cleaned.ffill()
    
    print(f"  Original range: [{skew_series.min():.1f}, {skew_series.max():.1f}]")
    print(f"  Cleaned range: [{cleaned.min():.1f}, {cleaned.max():.1f}]")
    
    return cleaned


def download_bitcoin_price_data(start_date, end_date):
    """
    Download Bitcoin price data from Yahoo Finance.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pd.DataFrame: Bitcoin price data
    """
    print(f"\nDownloading Bitcoin price data from {start_date} to {end_date}...")
    
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    
    # Flatten multi-index columns if present
    if isinstance(btc_data.columns, pd.MultiIndex):
        btc_data.columns = [col[0] for col in btc_data.columns]
    
    print(f"  ✓ Downloaded {len(btc_data)} days of price data")
    return btc_data


def calculate_price_metrics(btc_data, volatility_window=7, skewness_window=30):
    """
    Calculate returns, rolling volatility, and rolling skewness.
    
    Parameters:
    -----------
    btc_data : pd.DataFrame
        Bitcoin price data
    volatility_window : int
        Rolling window for volatility calculation (default 7 days)
    skewness_window : int
        Rolling window for skewness calculation (default 30 days)
        
    Returns:
    --------
    pd.DataFrame: DataFrame with calculated metrics
    """
    print("\nCalculating price metrics...")
    
    # Calculate returns
    btc_data['btc_return'] = btc_data['Close'].pct_change()
    
    # Calculate rolling volatility (annualized)
    btc_data['rolling_vol'] = btc_data['btc_return'].rolling(volatility_window).std() * np.sqrt(252)
    
    # Calculate rolling realized skewness
    def calculate_skewness(x):
        if len(x) < 3:
            return np.nan
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)
        if std_x == 0:
            return 0
        skew = np.mean(((x - mean_x) / std_x) ** 3)
        return skew
    
    btc_data['rolling_skew'] = btc_data['btc_return'].rolling(
        window=skewness_window, min_periods=3
    ).apply(calculate_skewness, raw=True)
    
    print(f"  ✓ Calculated returns")
    print(f"  ✓ Calculated {volatility_window}-day rolling volatility")
    print(f"  ✓ Calculated {skewness_window}-day rolling skewness")
    
    return btc_data


def create_master_dataframe(btc_data, skew_series):
    """
    Combine all data into a single aligned DataFrame.
    
    Parameters:
    -----------
    btc_data : pd.DataFrame
        Bitcoin price data with calculated metrics
    skew_series : pd.Series
        SKEW index values
        
    Returns:
    --------
    pd.DataFrame: Master DataFrame with all metrics aligned
    """
    print("\nCreating master DataFrame...")
    
    # Create base DataFrame with price data
    master_df = pd.DataFrame({
        'btc_price': btc_data['Close'],
        'btc_return': btc_data['btc_return'],
        'rolling_vol': btc_data['rolling_vol'],
        'rolling_skew': btc_data['rolling_skew']
    })
    
    # Add SKEW index
    master_df['skew_index'] = skew_series
    
    # Drop rows with any NaN values to ensure complete data
    initial_len = len(master_df)
    master_df = master_df.dropna()
    
    print(f"  ✓ Combined all metrics")
    print(f"  ✓ Dropped {initial_len - len(master_df)} incomplete rows")
    print(f"  ✓ Final dataset: {len(master_df)} days")
    
    return master_df


def save_results(master_df, output_file='bitcoin_skew_analysis.csv'):
    """
    Save the master DataFrame to CSV file.
    
    Parameters:
    -----------
    master_df : pd.DataFrame
        Master DataFrame with all metrics
    output_file : str
        Output filename (default 'bitcoin_skew_analysis.csv')
    """
    print(f"\nSaving results to {output_file}...")
    master_df.to_csv(output_file)
    print(f"  ✓ Saved {len(master_df)} rows × {len(master_df.columns)} columns")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for col in master_df.columns:
        print(f"{col:15s}: mean={master_df[col].mean():8.4f}, "
              f"std={master_df[col].std():8.4f}, "
              f"min={master_df[col].min():8.4f}, "
              f"max={master_df[col].max():8.4f}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Bitcoin SKEW Index Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Load options data
    options_data = load_bitcoin_options_data()
    
    # Step 2: Filter options for SKEW calculation
    filtered_options = filter_options_for_skew(options_data)
    
    # Step 3: Load yields data (assuming it exists)
    print("\nLoading yields data...")
    try:
        yields = pd.read_excel('rate.xlsx')
        yields = yields.set_index('Date')
        yields = yields / 100  # Convert percentage to decimal
        print("  ✓ Loaded yields data")
    except:
        print("  ! Warning: Could not load yields data, using default rate")
        # Create default yields data
        dates = filtered_options['QUOTE_DATE'].drop_duplicates()
        yields = pd.DataFrame({'Rate': 0.001}, index=pd.to_datetime(dates))
    
    # Step 4: Calculate SKEW values
    skew_raw = calculate_daily_skew(filtered_options, yields)
    
    # Step 5: Clean SKEW outliers
    skew_clean = clean_skew_outliers(skew_raw)
    
    # Step 6: Download Bitcoin price data
    btc_data = download_bitcoin_price_data('2021-06-01', '2023-01-01')
    
    # Step 7: Align Bitcoin data with SKEW dates
    btc_data = btc_data.reindex(skew_clean.index)
    
    # Step 8: Calculate price metrics
    btc_data = calculate_price_metrics(btc_data)
    
    # Step 9: Create master DataFrame
    master_df = create_master_dataframe(btc_data, skew_clean)
    
    # Step 10: Save results
    save_results(master_df)
    
    print("\n✓ Analysis complete!")
    print("=" * 60)
    
    return master_df


if __name__ == "__main__":
    # Run the main analysis
    master_df = main()
    
    # Optional: Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(master_df.head())